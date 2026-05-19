"""Sprint S9 — garde-fou anti-régression pour la commande ``serve``.

Bug observé en prod (HuggingFace Space, 2026-05-10) :
``picarones serve`` plantait avec ``ModuleNotFoundError: No module
named 'picarones.web'`` parce que ``_serve.py:70`` passait
``"picarones.web.app:app"`` à ``uvicorn.run()``.  Le paquet
``picarones.web/`` a été supprimé au sprint H.4 (mai 2026) lors
du retrait complet du legacy ; la string passée à uvicorn n'a
pas été migrée car aucun test ne l'exerçait.

Ce fichier verrouille deux contrats :

1. La string passée à ``uvicorn.run`` dans ``serve_cmd`` doit
   référencer un module **réellement importable** au format
   ``module.path:attribute``.
2. Le module et l'attribut ``app`` (instance FastAPI) doivent
   tous deux exister.

Le test n'exécute pas uvicorn — il extrait la string par
introspection AST du fichier source pour rester rapide et ne
pas dépendre du serveur réel.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path


def _extract_uvicorn_target() -> str:
    """Parse ``_serve.py`` pour extraire le 1er argument de l'appel
    à ``uvicorn.run(...)``.

    Implémentation par AST plutôt que par exécution : on ne veut
    pas démarrer un serveur ni dépendre de uvicorn pour tester
    la validité de la string.
    """
    serve_path = (
        Path(__file__).resolve().parents[3]
        / "picarones" / "interfaces" / "cli" / "_serve.py"
    )
    tree = ast.parse(serve_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "run"
                and isinstance(func.value, ast.Name)
                and func.value.id == "uvicorn"
            ):
                # 1er arg positionnel = target.
                if node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Constant):
                        return str(first_arg.value)
    raise AssertionError(
        "Pas d'appel ``uvicorn.run(<string>, ...)`` trouvé dans "
        "_serve.py — la commande serve a-t-elle été refactorée ?"
    )


class TestUvicornTargetIsImportable:
    """Le bug ``ModuleNotFoundError: picarones.web`` observé en
    prod ne doit jamais réapparaître : la string ``module:attr``
    passée à uvicorn doit être valide à tout instant."""

    def test_target_format_is_module_colon_attr(self) -> None:
        target = _extract_uvicorn_target()
        assert ":" in target, (
            f"Target uvicorn invalide : {target!r} — "
            "format attendu ``module.path:attr``."
        )
        module_path, _, attr = target.partition(":")
        assert module_path, "Module path vide avant ':'"
        assert attr, "Attribut vide après ':'"

    def test_target_module_is_actually_importable(self) -> None:
        """Le module nommé doit pouvoir être importé — c'est
        exactement ce qui plantait en prod (``picarones.web``
        n'existait plus depuis le sprint H.4)."""
        target = _extract_uvicorn_target()
        module_path, _, _ = target.partition(":")
        try:
            importlib.import_module(module_path)
        except ImportError as exc:
            raise AssertionError(
                f"``uvicorn.run({target!r}, ...)`` référence un "
                f"module non-importable : {exc}.  "
                "Probable régression : le module a été renommé "
                "sans mise à jour de _serve.py."
            ) from exc

    def test_target_attribute_exists_on_module(self) -> None:
        target = _extract_uvicorn_target()
        module_path, _, attr = target.partition(":")
        module = importlib.import_module(module_path)
        assert hasattr(module, attr), (
            f"Module {module_path!r} n'a pas l'attribut {attr!r} — "
            f"uvicorn planterait à l'instanciation."
        )

    def test_target_attribute_is_a_fastapi_app(self) -> None:
        """Garantit que l'attribut est bien une instance FastAPI
        (pas un module, pas une fonction) — sinon uvicorn
        rapporterait une erreur cryptique."""
        from fastapi import FastAPI

        target = _extract_uvicorn_target()
        module_path, _, attr = target.partition(":")
        module = importlib.import_module(module_path)
        app = getattr(module, attr)
        assert isinstance(app, FastAPI), (
            f"{target!r} → {type(app).__name__} ; FastAPI attendu."
        )

    def test_target_uses_canonical_v2_path(self) -> None:
        """Garde-fou explicite : depuis v2.0, la web app vit dans
        ``picarones.interfaces.web.app``, pas dans
        ``picarones.web.app`` (paquet supprimé au sprint H.4)."""
        target = _extract_uvicorn_target()
        assert "picarones.interfaces.web" in target, (
            f"Target {target!r} n'utilise pas le chemin canonique "
            "v2.0 ``picarones.interfaces.web.app:app``."
        )
        assert "picarones.web." not in target, (
            f"Target {target!r} référence le paquet legacy "
            "``picarones.web.*`` supprimé au sprint H.4."
        )
