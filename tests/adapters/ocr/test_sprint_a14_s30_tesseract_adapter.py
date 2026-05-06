"""Sprint A14-S30 — ``TesseractAdapter`` natif au contrat S26.

Tests de l'adapter Tesseract migré nativement (pas de shim sur le
legacy ``picarones.engines.tesseract``).

Couvre :

1. Constructeur :
   - rejet des paramètres invalides (name, psm, oem) ;
   - valeurs par défaut ;
   - propriétés en lecture.

2. ``execute`` :
   - cas nominal (mock pytesseract) → Artifact RAW_TEXT avec URI ;
   - input IMAGE absent → OCRAdapterError ;
   - artefact image sans URI → OCRAdapterError ;
   - image inexistante → OCRAdapterError ;
   - pytesseract non installé → OCRAdapterError ;
   - Tesseract lève → OCRAdapterError ;
   - écriture du fichier de sortie au bon emplacement ;
   - tesseract_cmd appliqué.

3. Contrat ``BaseOCRAdapter`` :
   - input_types / output_types / execution_mode ;
   - hérite bien de BaseOCRAdapter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from picarones.adapters.ocr import (
    BaseOCRAdapter,
    OCRAdapterError,
    TesseractAdapter,
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


# ──────────────────────────────────────────────────────────────────────
# Constructeur
# ──────────────────────────────────────────────────────────────────────


class TestTesseractAdapterConstructor:
    def test_defaults(self) -> None:
        adapter = TesseractAdapter()
        assert adapter.name == "tesseract"
        assert adapter.lang == "fra"
        assert adapter.psm == 6
        assert adapter.oem == 3

    def test_custom_name(self) -> None:
        adapter = TesseractAdapter(name="my_tesseract_lat")
        assert adapter.name == "my_tesseract_lat"

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(OCRAdapterError, match="vide"):
            TesseractAdapter(name="")

    def test_rejects_whitespace_name(self) -> None:
        with pytest.raises(OCRAdapterError, match="vide"):
            TesseractAdapter(name="   ")

    def test_rejects_invalid_chars_in_name(self) -> None:
        with pytest.raises(OCRAdapterError, match="invalide"):
            TesseractAdapter(name="bad name with space")

    def test_rejects_psm_out_of_range(self) -> None:
        with pytest.raises(OCRAdapterError, match=r"psm.*\[0, 13\]"):
            TesseractAdapter(psm=14)
        with pytest.raises(OCRAdapterError, match=r"psm.*\[0, 13\]"):
            TesseractAdapter(psm=-1)

    def test_rejects_oem_out_of_range(self) -> None:
        with pytest.raises(OCRAdapterError, match=r"oem.*\[0, 3\]"):
            TesseractAdapter(oem=4)
        with pytest.raises(OCRAdapterError, match=r"oem.*\[0, 3\]"):
            TesseractAdapter(oem=-1)

    def test_accepts_psm_boundary_values(self) -> None:
        TesseractAdapter(psm=0)
        TesseractAdapter(psm=13)

    def test_accepts_oem_boundary_values(self) -> None:
        TesseractAdapter(oem=0)
        TesseractAdapter(oem=3)


# ──────────────────────────────────────────────────────────────────────
# Contrat BaseOCRAdapter
# ──────────────────────────────────────────────────────────────────────


class TestTesseractAdapterContract:
    def test_inherits_base_adapter(self) -> None:
        adapter = TesseractAdapter()
        assert isinstance(adapter, BaseOCRAdapter)

    def test_input_types(self) -> None:
        assert TesseractAdapter.input_types == frozenset({ArtifactType.IMAGE})

    def test_output_types(self) -> None:
        """``output_types`` est l'ensemble maximal produit (constante de
        classe).  Si ``expose_confidences=False``, l'execute() omet
        CONFIDENCES du dict — le YAML ``PipelineSpec`` doit alors
        déclarer seulement ``[raw_text]`` pour cohérence.
        """
        assert TesseractAdapter.output_types == frozenset(
            {ArtifactType.RAW_TEXT, ArtifactType.CONFIDENCES},
        )

    def test_execution_mode_is_cpu(self) -> None:
        """Tesseract est CPU-bound — utilise un ProcessPool dans le runner."""
        assert TesseractAdapter.execution_mode == "cpu"


# ──────────────────────────────────────────────────────────────────────
# execute() — validation des inputs
# ──────────────────────────────────────────────────────────────────────


class TestTesseractAdapterInputValidation:
    def test_missing_image_input_raises(self, tmp_path: Path) -> None:
        adapter = TesseractAdapter()
        with pytest.raises(OCRAdapterError, match="IMAGE manquant"):
            adapter.execute(inputs={}, params={}, context=_make_context())

    def test_image_artifact_without_uri_raises(self) -> None:
        adapter = TesseractAdapter()
        artifact = Artifact(
            id="d1:img",
            document_id="d1",
            type=ArtifactType.IMAGE,
            uri=None,  # explicit no URI
        )
        with pytest.raises(OCRAdapterError, match="sans URI"):
            adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )

    def test_image_path_does_not_exist_raises(self) -> None:
        adapter = TesseractAdapter()
        artifact = _make_image_artifact("/nonexistent/path/img.png")
        with pytest.raises(OCRAdapterError, match="introuvable"):
            adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )


# ──────────────────────────────────────────────────────────────────────
# execute() — chemin nominal et erreurs Tesseract
# ──────────────────────────────────────────────────────────────────────


class TestTesseractAdapterExecute:
    def _create_dummy_image(self, tmp_path: Path) -> Path:
        """Crée un fichier vide qui sert d'image (les tests mocquent
        pytesseract donc le contenu n'est pas analysé)."""
        path = tmp_path / "page.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n")  # signature PNG basique
        return path

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    def test_nominal_execution(
        self, mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Cas nominal : pytesseract retourne du texte → Artifact RAW_TEXT
        avec URI vers un fichier produit."""
        mock_image_to_string.return_value = "Bonjour le monde\n"
        mock_image_open.return_value.__enter__.return_value = MagicMock()
        adapter = TesseractAdapter()
        image_path = self._create_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        result = adapter.execute(
            inputs={ArtifactType.IMAGE: artifact},
            params={},
            context=_make_context(),
        )
        assert ArtifactType.RAW_TEXT in result
        produced = result[ArtifactType.RAW_TEXT]
        assert produced.type == ArtifactType.RAW_TEXT
        assert produced.uri is not None

        # Le fichier de sortie existe et contient le texte stripé.
        out_path = Path(produced.uri)
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == "Bonjour le monde"

        # Convention : <stem>.<name>.txt à côté de l'image.
        assert out_path.name == "page.tesseract.txt"
        assert out_path.parent == tmp_path

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    def test_custom_name_changes_output_filename(
        self, mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_image_to_string.return_value = "x"
        mock_image_open.return_value.__enter__.return_value = MagicMock()
        adapter = TesseractAdapter(name="tess_lat_psm6")
        image_path = self._create_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        result = adapter.execute(
            inputs={ArtifactType.IMAGE: artifact},
            params={},
            context=_make_context(),
        )
        out_path = Path(result[ArtifactType.RAW_TEXT].uri)
        assert out_path.name == "page.tess_lat_psm6.txt"

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    def test_lang_psm_oem_passed_to_pytesseract(
        self, mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_image_to_string.return_value = "x"
        mock_image_open.return_value.__enter__.return_value = MagicMock()
        adapter = TesseractAdapter(lang="lat", psm=4, oem=1)
        image_path = self._create_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        adapter.execute(
            inputs={ArtifactType.IMAGE: artifact},
            params={},
            context=_make_context(),
        )

        # On vérifie l'appel à pytesseract.image_to_string avec les bons args.
        assert mock_image_to_string.called
        kwargs = mock_image_to_string.call_args.kwargs
        assert kwargs["lang"] == "lat"
        assert "--psm 4" in kwargs["config"]
        assert "--oem 1" in kwargs["config"]

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    def test_tesseract_cmd_applied_when_set(
        self, mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_image_to_string.return_value = "x"
        mock_image_open.return_value.__enter__.return_value = MagicMock()
        # Ré-import temporaire pour récupérer le module.
        import pytesseract  # type: ignore[import-untyped]
        adapter = TesseractAdapter(tesseract_cmd="/custom/bin/tesseract")
        image_path = self._create_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        adapter.execute(
            inputs={ArtifactType.IMAGE: artifact},
            params={},
            context=_make_context(),
        )
        assert pytesseract.pytesseract.tesseract_cmd == "/custom/bin/tesseract"

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    def test_tesseract_exception_wrapped_in_ocr_error(
        self, mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_image_open.return_value.__enter__.return_value = MagicMock()
        mock_image_to_string.side_effect = RuntimeError("Tesseract crashed")
        adapter = TesseractAdapter()
        image_path = self._create_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with pytest.raises(OCRAdapterError, match="RuntimeError.*Tesseract crashed"):
            adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )

    def test_pytesseract_not_installed_raises_clean_error(
        self, tmp_path: Path,
    ) -> None:
        """Si pytesseract est introuvable, l'erreur est claire et
        propose une commande pip."""
        adapter = TesseractAdapter()
        image_path = self._create_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        # Simule que pytesseract est absent.
        with patch.dict(sys.modules, {"pytesseract": None}):
            with pytest.raises(
                OCRAdapterError, match="pytesseract.*pip install",
            ):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    def test_artifact_id_uses_adapter_name(
        self, mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_image_to_string.return_value = "x"
        mock_image_open.return_value.__enter__.return_value = MagicMock()
        adapter = TesseractAdapter(name="custom_name")
        image_path = self._create_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        result = adapter.execute(
            inputs={ArtifactType.IMAGE: artifact},
            params={},
            context=_make_context(),
        )
        produced = result[ArtifactType.RAW_TEXT]
        assert produced.id == "d1:custom_name:raw_text"
        assert produced.document_id == "d1"
        assert produced.produced_by_step == "ocr"

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    def test_text_is_stripped(
        self, mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Le texte est strippé des whitespaces extérieurs comme dans
        le legacy."""
        mock_image_to_string.return_value = "  \n\nHello world\n\n  "
        mock_image_open.return_value.__enter__.return_value = MagicMock()
        adapter = TesseractAdapter()
        image_path = self._create_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        result = adapter.execute(
            inputs={ArtifactType.IMAGE: artifact},
            params={},
            context=_make_context(),
        )
        out_text = Path(result[ArtifactType.RAW_TEXT].uri).read_text(
            encoding="utf-8",
        )
        assert out_text == "Hello world"
