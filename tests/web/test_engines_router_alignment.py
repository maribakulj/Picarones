"""Phase 7.3 audit code-quality (2026-05) — le router
``/api/engines`` doit lister uniquement des engines connus de la
factory ``picarones.adapters.ocr.factory``.

L'audit avait identifié un risque de drift :

- ``picarones.adapters.ocr.factory._SUPPORTED`` (source de vérité)
- ``picarones.interfaces.web.routers.engines._collect_engines_sync``
  énumère manuellement ``tesseract``, ``pero_ocr``, ``kraken``,
  ``calamari``, puis ``cloud_ocr_specs = (mistral_ocr, google_vision,
  azure_doc_intel)``.

Si quelqu'un ajoute un nouvel adapter OCR à la factory sans
mettre à jour le router, ``/api/engines`` mentirait au frontend.
Si l'inverse, le frontend afficherait un engine non instanciable.

Plutôt qu'une refonte (le router a besoin de métadonnées
supplémentaires — label humain, nom de variable d'env, type
``local``/``cloud`` — que la factory ne fournit pas), ce test
verrouille l'invariant minimal : tout engine ID listé par le
router doit exister dans ``_SUPPORTED`` de la factory.
"""

from __future__ import annotations


def test_router_engines_subset_of_factory_supported() -> None:
    """Chaque engine listé par le router est dans la factory."""
    from picarones.adapters.ocr.factory import _SUPPORTED

    # Inspection AST du router pour extraire les engine_ids cités.
    import ast
    from pathlib import Path

    router_path = (
        Path(__file__).resolve().parents[2]
        / "picarones" / "interfaces" / "web" / "routers" / "engines.py"
    )
    tree = ast.parse(router_path.read_text(encoding="utf-8"))

    # Stratégie : extraire toutes les chaînes littérales qui
    # ressemblent à un engine_id (snake_case court).  Les chaînes
    # d'engine candidates sont les premiers args de
    # ``check_engine(name, ...)`` et le 1er tuple de
    # ``cloud_ocr_specs = ((engine_id, ...), ...)``.
    cited: set[str] = set()
    for node in ast.walk(tree):
        # ``check_engine("name", ...)``
        if isinstance(node, ast.Call):
            func = node.func
            is_check = (
                (isinstance(func, ast.Name) and func.id == "check_engine")
                or (
                    isinstance(func, ast.Attribute)
                    and func.attr == "check_engine"
                )
            )
            if is_check and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    cited.add(first.value)
        # ``cloud_ocr_specs = (("id", "label", "ENV"), ...)``
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "cloud_ocr_specs"
            and isinstance(node.value, ast.Tuple)
        ):
            for tup in node.value.elts:
                if isinstance(tup, ast.Tuple) and tup.elts:
                    first = tup.elts[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        cited.add(first.value)

    supported = set(_SUPPORTED)
    unknown = cited - supported
    assert not unknown, (
        f"Le router ``/api/engines`` mentionne des engines qui "
        f"n'existent pas dans la factory : {sorted(unknown)}.\n"
        f"Factory supportée : {sorted(supported)}.\n"
        f"Soit ajouter ces engines à la factory, soit les retirer du router."
    )


def test_factory_supported_at_least_canonical() -> None:
    """``_SUPPORTED`` contient les engines historiquement annoncés."""
    from picarones.adapters.ocr.factory import _SUPPORTED

    minimal = {
        "tesseract",
        "pero_ocr",
        "kraken",
        "calamari",
        "mistral_ocr",
        "google_vision",
        "azure_doc_intel",
        "precomputed",
    }
    actual = set(_SUPPORTED)
    missing = minimal - actual
    assert not missing, (
        f"Engines canoniques absents de _SUPPORTED : {sorted(missing)}. "
        f"Régression de la factory."
    )
