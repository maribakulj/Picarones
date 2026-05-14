"""Phase B5 — production native ALTO XML par ``TesseractAdapter``.

Tesseract sait nativement produire un ALTO 4 via
``pytesseract.image_to_alto_xml``.  Ce test vérifie que :

1. Le flag ``expose_alto`` (off par défaut, compat ascendante) ajoute
   un ``Artifact ALTO_XML`` à la sortie d'``execute()``.
2. La sortie est validée structurellement (XML bien formé) avant
   d'être promue en artefact.
3. Les défaillances (Tesseract qui plante, sortie vide, XML mal
   formé) sont absorbées en warning sans casser l'OCR ``RAW_TEXT``.
4. Un test ``@pytest.mark.live`` invoque le vrai binaire
   ``tesseract`` et vérifie que l'ALTO produit est valide.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from picarones.adapters.ocr import TesseractAdapter
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.pipeline.types import RunContext


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


_PNG_HEADER = b"\x89PNG\r\n\x1a\n"


_ALTO_VALID = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Layout>
    <Page ID="page_1" PHYSICAL_IMG_NR="1" WIDTH="1000" HEIGHT="1500">
      <PrintSpace ID="ps_1">
        <TextBlock ID="block_1">
          <TextLine ID="line_1">
            <String ID="word_1" CONTENT="Bonjour"
                    HPOS="100" VPOS="100" WIDTH="80" HEIGHT="20"/>
            <String ID="word_2" CONTENT="monde"
                    HPOS="200" VPOS="100" WIDTH="60" HEIGHT="20"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
"""


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


def _create_dummy_image(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    path.write_bytes(_PNG_HEADER)
    return path


# ──────────────────────────────────────────────────────────────────────
# Constructeur
# ──────────────────────────────────────────────────────────────────────


class TestExposeAltoFlag:
    def test_default_off(self) -> None:
        """Compat ascendante : ``expose_alto`` est désactivé par défaut.

        Les pipelines existants qui consomment ``RAW_TEXT`` /
        ``CONFIDENCES`` ne reçoivent aucun nouvel artefact non
        sollicité.
        """
        adapter = TesseractAdapter()
        assert adapter.expose_alto is False

    def test_can_be_enabled(self) -> None:
        adapter = TesseractAdapter(expose_alto=True)
        assert adapter.expose_alto is True

    def test_alto_xml_in_class_output_types(self) -> None:
        """Phase B5 — ``ALTO_XML`` est dans le set maximal de
        l'adapter (le YAML ``output_types`` du step décide quels
        types l'aval consomme).
        """
        assert ArtifactType.ALTO_XML in TesseractAdapter.output_types

    def test_default_output_still_includes_raw_text(self) -> None:
        """Pas de régression : ``RAW_TEXT`` et ``CONFIDENCES`` restent
        dans le set maximal."""
        assert ArtifactType.RAW_TEXT in TesseractAdapter.output_types
        assert ArtifactType.CONFIDENCES in TesseractAdapter.output_types


# ──────────────────────────────────────────────────────────────────────
# execute() — pas de production ALTO si expose_alto=False
# ──────────────────────────────────────────────────────────────────────


class TestExecuteNoAlto:
    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    @patch("pytesseract.image_to_alto_xml")
    def test_alto_function_not_called_by_default(
        self,
        mock_image_to_alto: MagicMock,
        mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Sans ``expose_alto``, ``pytesseract.image_to_alto_xml``
        n'est jamais invoqué — pas de coût Tesseract additionnel."""
        mock_image_to_string.return_value = "Bonjour le monde"
        mock_image_open.return_value.__enter__.return_value = MagicMock()
        adapter = TesseractAdapter(
            expose_alto=False, expose_confidences=False,
        )
        image_path = _create_dummy_image(tmp_path)

        result = adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
            params={}, context=_make_context(),
        )

        # ALTO absent du résultat.
        assert ArtifactType.ALTO_XML not in result
        # ``image_to_alto_xml`` jamais invoqué.
        mock_image_to_alto.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# execute() — production ALTO quand expose_alto=True
# ──────────────────────────────────────────────────────────────────────


class TestExecuteAltoEnabled:
    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    @patch("pytesseract.image_to_alto_xml")
    def test_alto_artifact_produced(
        self,
        mock_image_to_alto: MagicMock,
        mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Avec ``expose_alto=True``, un ``Artifact ALTO_XML`` est
        produit en plus du ``RAW_TEXT``."""
        mock_image_to_string.return_value = "Bonjour monde"
        mock_image_to_alto.return_value = _ALTO_VALID.encode("utf-8")
        mock_image_open.return_value.__enter__.return_value = MagicMock()

        adapter = TesseractAdapter(
            expose_alto=True, expose_confidences=False,
        )
        image_path = _create_dummy_image(tmp_path)

        result = adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
            params={}, context=_make_context(),
        )

        assert ArtifactType.ALTO_XML in result
        alto_artifact = result[ArtifactType.ALTO_XML]
        assert alto_artifact.type == ArtifactType.ALTO_XML
        assert alto_artifact.uri is not None
        # Le fichier ALTO existe et contient l'XML retourné par Tesseract.
        alto_path = Path(alto_artifact.uri)
        assert alto_path.exists()
        assert alto_path.suffix == ".xml"
        assert "alto" in alto_path.name.lower()
        assert "Bonjour" in alto_path.read_text(encoding="utf-8")

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    @patch("pytesseract.image_to_alto_xml")
    def test_alto_called_with_correct_lang_and_config(
        self,
        mock_image_to_alto: MagicMock,
        mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        """``image_to_alto_xml`` reçoit les mêmes ``lang``/``config``
        que ``image_to_string`` — cohérence des paramètres OCR."""
        mock_image_to_string.return_value = "x"
        mock_image_to_alto.return_value = _ALTO_VALID.encode("utf-8")
        mock_image_open.return_value.__enter__.return_value = MagicMock()

        adapter = TesseractAdapter(
            lang="lat", psm=4, oem=1,
            expose_alto=True, expose_confidences=False,
        )
        image_path = _create_dummy_image(tmp_path)
        adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
            params={}, context=_make_context(),
        )

        # Vérification que image_to_alto_xml a été invoqué avec
        # la bonne langue et la bonne config.
        assert mock_image_to_alto.call_count == 1
        kwargs = mock_image_to_alto.call_args.kwargs
        assert kwargs["lang"] == "lat"
        assert kwargs["config"] == "--oem 1 --psm 4"

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    @patch("pytesseract.image_to_alto_xml")
    def test_alto_failure_does_not_break_raw_text(
        self,
        mock_image_to_alto: MagicMock,
        mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Si ``image_to_alto_xml`` lève une exception, l'OCR
        ``RAW_TEXT`` reste valide — l'ALTO est juste sauté avec
        un warning loggé.
        """
        mock_image_to_string.return_value = "Bonjour"
        mock_image_to_alto.side_effect = RuntimeError("Tesseract ALTO crash")
        mock_image_open.return_value.__enter__.return_value = MagicMock()

        adapter = TesseractAdapter(
            expose_alto=True, expose_confidences=False,
        )
        image_path = _create_dummy_image(tmp_path)
        result = adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
            params={}, context=_make_context(),
        )

        # RAW_TEXT toujours présent.
        assert ArtifactType.RAW_TEXT in result
        # ALTO absent (best-effort skip).
        assert ArtifactType.ALTO_XML not in result

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    @patch("pytesseract.image_to_alto_xml")
    def test_alto_empty_output_skipped(
        self,
        mock_image_to_alto: MagicMock,
        mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Un ALTO vide ou que des espaces n'est pas promu en artefact."""
        mock_image_to_string.return_value = "x"
        mock_image_to_alto.return_value = b""
        mock_image_open.return_value.__enter__.return_value = MagicMock()

        adapter = TesseractAdapter(
            expose_alto=True, expose_confidences=False,
        )
        image_path = _create_dummy_image(tmp_path)
        result = adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
            params={}, context=_make_context(),
        )

        assert ArtifactType.ALTO_XML not in result

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    @patch("pytesseract.image_to_alto_xml")
    def test_alto_malformed_xml_skipped(
        self,
        mock_image_to_alto: MagicMock,
        mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Un ALTO mal formé (balise non fermée, etc.) n'est pas promu
        en artefact — la validation ``safe_parse_xml`` rejette."""
        mock_image_to_string.return_value = "x"
        # XML invalide : pas de balise root fermante.
        mock_image_to_alto.return_value = b"<alto><Page></alto>"
        mock_image_open.return_value.__enter__.return_value = MagicMock()

        adapter = TesseractAdapter(
            expose_alto=True, expose_confidences=False,
        )
        image_path = _create_dummy_image(tmp_path)
        result = adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
            params={}, context=_make_context(),
        )

        assert ArtifactType.ALTO_XML not in result

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    @patch("pytesseract.image_to_alto_xml")
    def test_alto_string_output_normalized(
        self,
        mock_image_to_alto: MagicMock,
        mock_image_to_string: MagicMock,
        mock_image_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        """``pytesseract.image_to_alto_xml`` peut retourner un ``str``
        au lieu de ``bytes`` selon la version — l'adapter doit gérer
        les deux types."""
        mock_image_to_string.return_value = "x"
        mock_image_to_alto.return_value = _ALTO_VALID  # str, pas bytes
        mock_image_open.return_value.__enter__.return_value = MagicMock()

        adapter = TesseractAdapter(
            expose_alto=True, expose_confidences=False,
        )
        image_path = _create_dummy_image(tmp_path)
        result = adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(image_path))},
            params={}, context=_make_context(),
        )

        assert ArtifactType.ALTO_XML in result


# ──────────────────────────────────────────────────────────────────────
# Test live — vraie exécution Tesseract
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.live
class TestExecuteAltoLive:
    """Tests qui invoquent le vrai binaire ``tesseract``.

    Activés uniquement avec ``pytest -m live``.  Skipped sans le
    binaire (vérifié au fixture).
    """

    @pytest.fixture
    def real_image(self, tmp_path: Path) -> Path:
        """Crée une image PNG avec du texte rendu via Pillow.

        Tesseract devrait être capable de transcrire ce texte.
        """
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (300, 80), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((10, 30), "Bonjour", fill=(0, 0, 0))
        path = tmp_path / "live_page.png"
        img.save(path)
        return path

    def test_real_tesseract_produces_valid_alto(
        self, real_image: Path, tmp_path: Path,
    ) -> None:
        """Vrai Tesseract → ALTO XML structurellement valide."""
        from picarones.formats.alto.parser import parse_alto

        adapter = TesseractAdapter(
            lang="eng", psm=7,
            expose_alto=True, expose_confidences=False,
        )

        result = adapter.execute(
            inputs={ArtifactType.IMAGE: _make_image_artifact(str(real_image))},
            params={}, context=_make_context(),
        )

        assert ArtifactType.ALTO_XML in result, (
            "Tesseract n'a pas produit d'ALTO — vérifier l'installation "
            "tesseract + pytesseract."
        )
        alto_path = Path(result[ArtifactType.ALTO_XML].uri)
        assert alto_path.exists()
        # Le parser ALTO de Picarones doit accepter la sortie Tesseract.
        parsed = parse_alto(alto_path.read_text(encoding="utf-8"))
        assert parsed is not None
