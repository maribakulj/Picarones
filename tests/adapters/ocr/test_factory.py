"""Sprint H.2.b — factory canonique ``ocr_adapter_from_name``.

Vérifie l'équivalent canonique de
``picarones.adapters.legacy_engines.factory.engine_from_name`` :

- Résolution des alias (``tess`` → ``tesseract``, etc.) ;
- Construction effective des 6 adapters supportés (1 sans deps,
  4 cloud avec deps optionnelles, 1 precomputed) ;
- ``ValueError`` propre sur nom inconnu / dépendance absente,
  avec message d'erreur listant les moteurs supportés ;
- Insensibilité à la casse du nom.
"""

from __future__ import annotations

import pytest

from picarones.adapters.ocr import ocr_adapter_from_name
from picarones.adapters.ocr.base import BaseOCRAdapter
from picarones.adapters.ocr.tesseract import TesseractAdapter


class TestTesseract:
    def test_canonical_name(self) -> None:
        adapter = ocr_adapter_from_name("tesseract")
        assert isinstance(adapter, TesseractAdapter)
        assert adapter.name == "tesseract"

    def test_alias_tess(self) -> None:
        adapter = ocr_adapter_from_name("tess")
        assert isinstance(adapter, TesseractAdapter)
        # L'alias normalise vers le nom canonique "tesseract".
        assert adapter.name == "tesseract"

    def test_uppercase_name_normalized(self) -> None:
        adapter = ocr_adapter_from_name("Tesseract")
        assert isinstance(adapter, TesseractAdapter)

    def test_kwargs_propagate(self) -> None:
        adapter = ocr_adapter_from_name(
            "tesseract", lang="eng", psm=3,
        )
        assert adapter.lang == "eng"

    def test_invalid_kwarg_raises_typeerror(self) -> None:
        # Pas de masquage des fautes de frappe.
        with pytest.raises(TypeError):
            ocr_adapter_from_name("tesseract", langg="fra")

    def test_invalid_psm_raises_ocr_adapter_error(self) -> None:
        from picarones.adapters.ocr.base import OCRAdapterError
        with pytest.raises(OCRAdapterError, match="psm"):
            ocr_adapter_from_name("tesseract", psm=99)


class TestPrecomputed:
    """``precomputed`` n'a pas de dépendance optionnelle — doit
    toujours être instanciable."""

    def test_canonical_name(self) -> None:
        adapter = ocr_adapter_from_name(
            "precomputed", source_label="bnf",
        )
        assert isinstance(adapter, BaseOCRAdapter)
        assert "bnf" in adapter.name


class TestCloudAdapters:
    """Les adapters cloud sont importables sans la dépendance
    système (pas de credentials nécessaires à l'instanciation —
    la lib client est résolue paresseusement à execute())."""

    def test_mistral_ocr_via_alias(self) -> None:
        adapter = ocr_adapter_from_name(
            "mistral", model="mistral-ocr-latest", api_key="fake",
        )
        assert isinstance(adapter, BaseOCRAdapter)
        assert adapter.name == "mistral_ocr"

    def test_google_vision_via_alias(self) -> None:
        adapter = ocr_adapter_from_name(
            "google", api_key="fake",
        )
        assert isinstance(adapter, BaseOCRAdapter)
        assert adapter.name == "google_vision"

    def test_azure_doc_intel_via_alias(self) -> None:
        adapter = ocr_adapter_from_name(
            "azure", endpoint="https://x.com", api_key="fake",
        )
        assert isinstance(adapter, BaseOCRAdapter)
        assert adapter.name == "azure_doc_intel"


class TestPeroOCR:
    """Pero OCR a une dépendance optionnelle ``pero-ocr`` — peut
    être absent dans l'environnement de test."""

    def test_canonical_or_helpful_error(self, tmp_path) -> None:
        cfg = tmp_path / "fake_pero.ini"
        cfg.write_text("# fake config", encoding="utf-8")
        try:
            adapter = ocr_adapter_from_name(
                "pero_ocr", config_path=str(cfg),
            )
            assert isinstance(adapter, BaseOCRAdapter)
        except ValueError as exc:
            # Si ``pero-ocr`` n'est pas installé, on attend un
            # message d'erreur qui explique comment l'installer.
            assert "pero-ocr" in str(exc).lower()


class TestUnknownName:
    def test_unknown_raises_with_supported_list(self) -> None:
        with pytest.raises(ValueError) as ctx:
            ocr_adapter_from_name("not_a_real_engine")
        msg = str(ctx.value)
        # Le message liste les moteurs supportés et les alias —
        # utile pour le diagnostic.
        assert "tesseract" in msg
        assert "alias" in msg.lower()

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError):
            ocr_adapter_from_name("")
