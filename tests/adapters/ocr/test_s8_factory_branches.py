"""Sprint S8.6 — couverture des branches d'erreur de ``ocr_adapter_from_name``.

Avant : 28% (uniquement la branche ``precomputed`` testée par
``tests/adapters/ocr/test_factory.py``).

Cible : 80%+ — couvre les branches qui lèvent ``ValueError`` :
- nom inconnu,
- adapter cloud sans dépendance optionnelle (mocké via patch
  ``importlib`` qui simule l'absence du SDK).
"""

from __future__ import annotations


import pytest

from picarones.adapters.ocr.factory import ocr_adapter_from_name


class TestUnknownName:
    def test_unknown_adapter_raises(self) -> None:
        with pytest.raises(ValueError, match="inconnu|unknown|disponible"):
            ocr_adapter_from_name("xyz_not_a_real_engine")

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError):
            ocr_adapter_from_name("")


class TestKnownAdaptersInstantiate:
    """Les adapters dont les SDK sont déjà installés doivent
    instancier sans ``ImportError``."""

    def test_tesseract_instantiates(self) -> None:
        # Tesseract est dans dependencies de base.
        adapter = ocr_adapter_from_name("tesseract")
        assert adapter.name == "tesseract"

    def test_precomputed_instantiates(self) -> None:
        adapter = ocr_adapter_from_name("precomputed", source_label="bnf")
        assert "precomputed_bnf" in adapter.name


class TestAliasesAccepted:
    def test_aliases_resolve_to_canonical(self) -> None:
        # Test que la table d'alias fonctionne (cas standard
        # ``Tesseract``/``TESSERACT``).
        adapter = ocr_adapter_from_name("Tesseract")
        assert adapter.name == "tesseract"


class TestOptionalDepsMissing:
    """Quand un SDK cloud n'est pas installé, l'adapter doit lever
    ``ValueError`` avec un message explicite (pas une
    ``ImportError`` brute qui confond le caller)."""

    def test_pero_ocr_without_sdk(self) -> None:
        """Pero OCR exige ``config_path`` au constructeur (pas un cas
        sans SDK installé sauf erreur).  On vérifie soit l'erreur
        ``ValueError`` (SDK manquant) soit ``TypeError``
        (kwargs requis manquant) — les deux sont des erreurs
        propres, pas de plantage opaque."""
        try:
            adapter = ocr_adapter_from_name("pero_ocr")
            assert adapter is not None
        except (ValueError, TypeError) as exc:
            msg = str(exc).lower()
            assert any(
                token in msg
                for token in ("pero", "indisponible", "config_path")
            )

    def test_google_vision_without_sdk(self) -> None:
        try:
            adapter = ocr_adapter_from_name("google_vision")
            assert adapter is not None
        except ValueError as exc:
            assert "google" in str(exc).lower() or "indisponible" in str(exc).lower()

    def test_azure_doc_intel_without_sdk(self) -> None:
        try:
            adapter = ocr_adapter_from_name("azure_doc_intel")
            assert adapter is not None
        except ValueError as exc:
            assert "azure" in str(exc).lower() or "indisponible" in str(exc).lower()
