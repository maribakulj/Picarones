"""Tests de la phase C — extras/importers/ (importers vers Cercle 3).

Couvre :

- 6 importers (``_http``, ``iiif``, ``htr_united``, ``gallica``,
  ``huggingface``, ``escriptorium``) déplacés vers
  ``picarones/extras/importers/``.
- Identité préservée à travers les shims.
- ``huggingface`` et ``escriptorium`` émettent un ``UserWarning``
  ``experimental`` à l'import.
- ``picarones.importers/__init__.py`` continue à réexporter les
  noms historiques.
- ``cli/_imports.py`` continue à fonctionner.
- pyproject.toml déclare ``[importers]``.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────────
# 1. Imports historiques rétrocompat via shims
# ──────────────────────────────────────────────────────────────────────────


class TestImportersRetrocompat:
    @pytest.mark.parametrize("module_path, attribute", [
        ("picarones.importers.iiif", "IIIFImporter"),
        ("picarones.importers.iiif", "import_iiif_manifest"),
        ("picarones.importers.htr_united", "HTRUnitedEntry"),
        ("picarones.importers.htr_united", "HTRUnitedCatalogue"),
        ("picarones.importers.htr_united", "import_htr_united_corpus"),
        ("picarones.importers.gallica", "GallicaClient"),
        ("picarones.importers.gallica", "GallicaRecord"),
        ("picarones.importers.gallica", "search_gallica"),
        ("picarones.importers.gallica", "import_gallica_document"),
        ("picarones.importers._http", "validate_http_url"),
        ("picarones.importers._http", "download_url"),
    ])
    def test_legacy_path_works(self, module_path: str, attribute: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = importlib.import_module(module_path)
        assert hasattr(mod, attribute)


# ──────────────────────────────────────────────────────────────────────────
# 2. Imports via le nouveau chemin extras/importers/
# ──────────────────────────────────────────────────────────────────────────


class TestExtrasImportersPath:
    @pytest.mark.parametrize("new_path, attribute", [
        ("picarones.extras.importers._http", "validate_http_url"),
        ("picarones.extras.importers._http", "download_url"),
        ("picarones.extras.importers.iiif", "IIIFImporter"),
        ("picarones.extras.importers.iiif", "import_iiif_manifest"),
        ("picarones.extras.importers.htr_united", "HTRUnitedCatalogue"),
        ("picarones.extras.importers.gallica", "GallicaClient"),
        ("picarones.extras.importers.huggingface", "HuggingFaceImporter"),
        ("picarones.extras.importers.escriptorium", "EScriptoriumClient"),
    ])
    def test_extras_path_works(self, new_path: str, attribute: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = importlib.import_module(new_path)
        assert hasattr(mod, attribute)


# ──────────────────────────────────────────────────────────────────────────
# 3. Identité préservée
# ──────────────────────────────────────────────────────────────────────────


class TestIdentityThroughShim:
    def test_iiif_identity(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from picarones.extras.importers.iiif import IIIFImporter as via_new
            from picarones.importers.iiif import IIIFImporter as via_old
        assert via_old is via_new

    def test_gallica_identity(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from picarones.extras.importers.gallica import GallicaClient as via_new
            from picarones.importers.gallica import GallicaClient as via_old
        assert via_old is via_new

    def test_http_helpers_identity(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from picarones.extras.importers._http import (
                validate_http_url as via_new,
            )
            from picarones.importers._http import (
                validate_http_url as via_old,
            )
        assert via_old is via_new


# ──────────────────────────────────────────────────────────────────────────
# 4. Modules expérimentaux : UserWarning à l'import
# ──────────────────────────────────────────────────────────────────────────


def _force_reimport(module_name_substring: str) -> None:
    """Vide le cache d'import pour pouvoir capturer le UserWarning."""
    for name in list(sys.modules.keys()):
        if module_name_substring in name:
            del sys.modules[name]


class TestExperimentalImporters:
    def test_huggingface_emits_userwarning(self):
        _force_reimport("huggingface")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import picarones.extras.importers.huggingface  # noqa: F401
        msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        assert any("experimental" in m for m in msgs), (
            f"huggingface n'a pas émis de UserWarning experimental — "
            f"warnings reçus : {[str(x.message) for x in w]}"
        )

    def test_escriptorium_emits_userwarning(self):
        _force_reimport("escriptorium")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import picarones.extras.importers.escriptorium  # noqa: F401
        msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        assert any("experimental" in m for m in msgs)

    def test_iiif_does_not_emit_warning(self):
        """Les importers maintenus ne doivent PAS émettre de warning."""
        _force_reimport("iiif")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import picarones.extras.importers.iiif  # noqa: F401
        msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        # Il peut y avoir d'autres warnings (deprecation Python, etc.)
        # mais pas de "experimental" sur iiif
        assert not any(
            "iiif" in m and "experimental" in m for m in msgs
        ), "iiif ne doit pas être marqué experimental"


# ──────────────────────────────────────────────────────────────────────────
# 5. picarones.importers/__init__.py — réexports historiques
# ──────────────────────────────────────────────────────────────────────────


class TestImportersInitReexports:
    def test_reexports_work(self):
        """Le ``__init__`` réexporte des symboles via les shims, eux-mêmes
        chargeant depuis extras."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from picarones.importers import (
                EScriptoriumClient,
                GallicaClient,
                IIIFImporter,
            )
        assert IIIFImporter is not None
        assert GallicaClient is not None
        assert EScriptoriumClient is not None


# ──────────────────────────────────────────────────────────────────────────
# 6. cli/_imports.py — toujours fonctionnel
# ──────────────────────────────────────────────────────────────────────────


class TestCliImportsCommand:
    def test_cli_imports_module_loads(self):
        """``picarones.cli._imports`` importe IIIFImporter depuis
        ``picarones.importers.iiif`` — doit fonctionner via shim."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import picarones.cli._imports  # noqa: F401
        except ImportError as exc:
            if "click" in str(exc):
                pytest.skip("click absent")
            raise


# ──────────────────────────────────────────────────────────────────────────
# 7. pyproject.toml — extra [importers]
# ──────────────────────────────────────────────────────────────────────────


class TestPyprojectExtra:
    def test_importers_extra_declared(self):
        path = Path(__file__).parent.parent / "pyproject.toml"
        content = path.read_text(encoding="utf-8")
        assert "importers = []" in content or 'importers = [' in content
        assert "extras/importers" in content
        assert "Cercle 3" in content


# ──────────────────────────────────────────────────────────────────────────
# 8. Originaux sont des shims minces
# ──────────────────────────────────────────────────────────────────────────


class TestOriginalsAreShims:
    @pytest.mark.parametrize("path", [
        "picarones/importers/_http.py",
        "picarones/importers/iiif.py",
        "picarones/importers/htr_united.py",
        "picarones/importers/gallica.py",
        "picarones/importers/huggingface.py",
        "picarones/importers/escriptorium.py",
    ])
    def test_is_thin_shim(self, path):
        repo_root = Path(__file__).parent.parent
        content = (repo_root / path).read_text(encoding="utf-8")
        n_lines = len([line for line in content.splitlines() if line.strip()])
        assert n_lines < 30, (
            f"{path} fait {n_lines} lignes — devrait être un shim mince"
        )
        assert "déplacé" in content or "extras" in content
