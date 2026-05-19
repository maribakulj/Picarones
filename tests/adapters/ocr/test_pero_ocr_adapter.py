"""Sprint A14-S31 — ``PeroOCRAdapter`` natif au contrat S26.

Tests de l'adapter Pero OCR migré nativement (pas de shim sur le
legacy ``picarones.engines.pero_ocr``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from picarones.adapters.ocr import (
    BaseOCRAdapter,
    OCRAdapterError,
    PeroOCRAdapter,
)
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.pipeline.types import RunContext


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_image_artifact(uri: str) -> Artifact:
    return Artifact(
        id="d1:initial:image",
        document_id="d1",
        type=ArtifactType.IMAGE,
        uri=uri,
    )


def _make_context() -> RunContext:
    return RunContext(
        document_id="d1",
        code_version="1.0.0",
        pipeline_name="test",
    )


def _make_dummy_image(tmp_path: Path) -> Path:
    """Crée un fichier image réel pour que PIL puisse l'ouvrir.

    On utilise PIL pour générer une image PNG 10x10 valide, parce que
    pero_ocr ne mock pas PIL.Image.open complètement.
    """
    try:
        from PIL import Image
        import numpy as np
        image_path = tmp_path / "page.png"
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        Image.fromarray(arr).save(image_path)
        return image_path
    except ImportError:
        pytest.skip("PIL/numpy not available")


def _make_dummy_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "pero.ini"
    config_path.write_text("[PARSER]\nname = stub\n")
    return config_path


# ──────────────────────────────────────────────────────────────────────
# Constructeur
# ──────────────────────────────────────────────────────────────────────


class TestPeroOCRAdapterConstructor:
    def test_with_required_config_path(self, tmp_path: Path) -> None:
        cfg = _make_dummy_config(tmp_path)
        adapter = PeroOCRAdapter(config_path=cfg)
        assert adapter.name == "pero_ocr"
        assert adapter.config_path == cfg

    def test_custom_name(self, tmp_path: Path) -> None:
        cfg = _make_dummy_config(tmp_path)
        adapter = PeroOCRAdapter(config_path=cfg, name="my_pero")
        assert adapter.name == "my_pero"

    def test_rejects_empty_name(self, tmp_path: Path) -> None:
        cfg = _make_dummy_config(tmp_path)
        with pytest.raises(OCRAdapterError, match="vide"):
            PeroOCRAdapter(config_path=cfg, name="")

    def test_rejects_invalid_chars_in_name(self, tmp_path: Path) -> None:
        cfg = _make_dummy_config(tmp_path)
        with pytest.raises(OCRAdapterError, match="invalide"):
            PeroOCRAdapter(config_path=cfg, name="bad name")

    def test_rejects_empty_config_path(self) -> None:
        with pytest.raises(OCRAdapterError, match="config_path"):
            PeroOCRAdapter(config_path="")

    def test_accepts_string_config_path(self, tmp_path: Path) -> None:
        cfg = _make_dummy_config(tmp_path)
        adapter = PeroOCRAdapter(config_path=str(cfg))
        assert adapter.config_path == Path(str(cfg))


# ──────────────────────────────────────────────────────────────────────
# Contrat BaseOCRAdapter
# ──────────────────────────────────────────────────────────────────────


class TestPeroOCRAdapterContract:
    def test_inherits_base_adapter(self, tmp_path: Path) -> None:
        cfg = _make_dummy_config(tmp_path)
        adapter = PeroOCRAdapter(config_path=cfg)
        assert isinstance(adapter, BaseOCRAdapter)

    def test_input_types(self) -> None:
        assert PeroOCRAdapter.input_types == frozenset({ArtifactType.IMAGE})

    def test_output_types(self) -> None:
        assert PeroOCRAdapter.output_types == frozenset({ArtifactType.RAW_TEXT})

    def test_execution_mode_is_cpu(self) -> None:
        assert PeroOCRAdapter.execution_mode == "cpu"


# ──────────────────────────────────────────────────────────────────────
# execute() — validation des inputs
# ──────────────────────────────────────────────────────────────────────


class TestPeroOCRAdapterInputValidation:
    def test_missing_image_input_raises(self, tmp_path: Path) -> None:
        adapter = PeroOCRAdapter(config_path=_make_dummy_config(tmp_path))
        with pytest.raises(OCRAdapterError, match="IMAGE manquant"):
            adapter.execute(inputs={}, params={}, context=_make_context())

    def test_image_artifact_without_uri_raises(self, tmp_path: Path) -> None:
        adapter = PeroOCRAdapter(config_path=_make_dummy_config(tmp_path))
        artifact = Artifact(
            id="d1:img",
            document_id="d1",
            type=ArtifactType.IMAGE,
            uri=None,
        )
        with pytest.raises(OCRAdapterError, match="sans URI"):
            adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )

    def test_image_path_does_not_exist_raises(self, tmp_path: Path) -> None:
        adapter = PeroOCRAdapter(config_path=_make_dummy_config(tmp_path))
        artifact = _make_image_artifact("/nonexistent/img.png")
        with pytest.raises(OCRAdapterError, match="introuvable"):
            adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )

    def test_config_path_missing_raises_at_first_run(
        self, tmp_path: Path,
    ) -> None:
        """Si le config_path n'existe pas sur disque, l'erreur est levée
        au premier execute() (lazy parser init)."""
        nonexistent_cfg = tmp_path / "missing.ini"
        adapter = PeroOCRAdapter(config_path=nonexistent_cfg)
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))
        # On mock pero_ocr pour passer l'import et tester le check de config_path.
        fake_pero = MagicMock()
        with patch.dict(sys.modules, {
            "pero_ocr": fake_pero,
            "pero_ocr.document_ocr": MagicMock(),
            "pero_ocr.document_ocr.page_parser": MagicMock(),
            "pero_ocr.document_ocr.layout": MagicMock(),
        }):
            with pytest.raises(OCRAdapterError, match="config_path"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )


# ──────────────────────────────────────────────────────────────────────
# execute() — chemin nominal
# ──────────────────────────────────────────────────────────────────────


class TestPeroOCRAdapterExecute:
    def _patch_pero_modules(self, page_layout_factory):
        """Helper : retourne un context manager qui mock pero_ocr."""
        fake_page_parser_module = MagicMock()
        fake_page_parser_module.PageParser = MagicMock()
        fake_layout_module = MagicMock()
        fake_layout_module.PageLayout = page_layout_factory

        return patch.dict(sys.modules, {
            "pero_ocr": MagicMock(),
            "pero_ocr.document_ocr": MagicMock(),
            "pero_ocr.document_ocr.page_parser": fake_page_parser_module,
            "pero_ocr.document_ocr.layout": fake_layout_module,
        })

    def test_nominal_extracts_text_in_line_order(self, tmp_path: Path) -> None:
        # PageLayout simulé avec 2 régions × 2 lignes
        line1 = MagicMock()
        line1.transcription = "Bonjour le monde"
        line1.transcription_confidence = 0.9
        line2 = MagicMock()
        line2.transcription = "Tout va bien"
        line2.transcription_confidence = 0.8
        line3 = MagicMock()
        line3.transcription = "Deuxième région"
        line3.transcription_confidence = 0.7

        region1 = MagicMock()
        region1.lines = [line1, line2]
        region2 = MagicMock()
        region2.lines = [line3]

        page_layout_instance = MagicMock()
        page_layout_instance.regions = [region1, region2]

        # PageLayout(id, page_size=...) returns notre instance.
        page_layout_factory = MagicMock(return_value=page_layout_instance)

        cfg = _make_dummy_config(tmp_path)
        adapter = PeroOCRAdapter(config_path=cfg)
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with self._patch_pero_modules(page_layout_factory):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )

        produced = result[ArtifactType.RAW_TEXT]
        assert produced.type == ArtifactType.RAW_TEXT
        out_text = Path(produced.uri).read_text(encoding="utf-8")
        assert out_text == "Bonjour le monde\nTout va bien\nDeuxième région"

    def test_skips_lines_without_transcription(self, tmp_path: Path) -> None:
        line_with = MagicMock()
        line_with.transcription = "Présent"
        line_without = MagicMock()
        line_without.transcription = None

        region = MagicMock()
        region.lines = [line_with, line_without]
        page_layout_instance = MagicMock()
        page_layout_instance.regions = [region]
        page_layout_factory = MagicMock(return_value=page_layout_instance)

        cfg = _make_dummy_config(tmp_path)
        adapter = PeroOCRAdapter(config_path=cfg)
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with self._patch_pero_modules(page_layout_factory):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        out_text = Path(result[ArtifactType.RAW_TEXT].uri).read_text(
            encoding="utf-8",
        )
        assert out_text == "Présent"

    def test_writes_to_stem_name_txt_pattern(self, tmp_path: Path) -> None:
        page_layout_instance = MagicMock()
        page_layout_instance.regions = []
        page_layout_factory = MagicMock(return_value=page_layout_instance)
        cfg = _make_dummy_config(tmp_path)
        adapter = PeroOCRAdapter(config_path=cfg, name="my_pero")
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with self._patch_pero_modules(page_layout_factory):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        out_path = Path(result[ArtifactType.RAW_TEXT].uri)
        from picarones.adapters.output_paths import _pipeline_path_segment
        seg = _pipeline_path_segment(_make_context())
        assert out_path.name == f"page.{seg}.my_pero.txt"
        assert out_path.parent == tmp_path

    def test_pero_not_installed_raises_clean_error(
        self, tmp_path: Path,
    ) -> None:
        cfg = _make_dummy_config(tmp_path)
        adapter = PeroOCRAdapter(config_path=cfg)
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        # Force absence du module pero_ocr.
        with patch.dict(sys.modules, {
            "pero_ocr": None,
            "pero_ocr.document_ocr.page_parser": None,
            "pero_ocr.document_ocr.layout": None,
        }):
            with pytest.raises(OCRAdapterError):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )

    def test_artifact_id_uses_adapter_name(self, tmp_path: Path) -> None:
        page_layout_instance = MagicMock()
        page_layout_instance.regions = []
        page_layout_factory = MagicMock(return_value=page_layout_instance)
        cfg = _make_dummy_config(tmp_path)
        adapter = PeroOCRAdapter(config_path=cfg, name="custom")
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with self._patch_pero_modules(page_layout_factory):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        produced = result[ArtifactType.RAW_TEXT]
        assert produced.id == "d1:custom:raw_text"
        assert produced.document_id == "d1"
        assert produced.produced_by_step == "ocr"

    def test_pero_internal_error_wrapped(self, tmp_path: Path) -> None:
        page_layout_factory = MagicMock(
            side_effect=RuntimeError("Pero crashed"),
        )
        cfg = _make_dummy_config(tmp_path)
        adapter = PeroOCRAdapter(config_path=cfg)
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with self._patch_pero_modules(page_layout_factory):
            with pytest.raises(OCRAdapterError, match="RuntimeError.*Pero crashed"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )

    def test_parser_lazy_init_and_reused(self, tmp_path: Path) -> None:
        """Le parser est instancié au premier execute() et réutilisé."""
        page_layout_instance = MagicMock()
        page_layout_instance.regions = []
        page_layout_factory = MagicMock(return_value=page_layout_instance)

        cfg = _make_dummy_config(tmp_path)
        adapter = PeroOCRAdapter(config_path=cfg)
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))
        assert adapter._parser is None

        with self._patch_pero_modules(page_layout_factory):
            adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
            first_parser = adapter._parser
            assert first_parser is not None
            adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
            # Le parser doit être le même au deuxième appel.
            assert adapter._parser is first_parser
