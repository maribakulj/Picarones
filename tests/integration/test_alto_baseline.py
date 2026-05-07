"""Tests du chantier 1 — reconstructeur ALTO baseline + métriques (ALTO, ALTO).

Couvre :

- :class:`picarones.modules.TextToAltoMonoRegion` : produit un ALTO 4.2
  conforme, déterministe, qui tolère absence d'image / image
  introuvable / dimensions invalides.
- :func:`picarones.measurements.alto_metrics.extract_text_from_alto` : parsing
  tolérant (avec/sans namespace, ALTO partiel, GT ``AltoGT`` ou ``str``).
- Métriques ``alto_text_cer`` / ``alto_text_wer`` enregistrées sur
  ``(ALTO, ALTO)`` et découvrables via ``compute_at_junction``.
- Bout-en-bout : ``PipelineRunner`` exécute une pipeline
  ``MockOCR → TextToAltoMonoRegion`` et calcule automatiquement la
  jonction ``(ALTO, ALTO)`` contre une ``AltoGT`` du document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pytest

from picarones.measurements.alto_metrics import (
    alto_text_cer,
    extract_text_from_alto,
)
from picarones.evaluation.corpus import AltoGT, Document, GTLevel, TextGT
from picarones.evaluation.metric_registry import compute_at_junction, select_metrics
from picarones.domain.artifacts import ArtifactType
from picarones.domain.module_protocol import BaseModule
from picarones.evaluation.pipeline import (
    PipelineRunner,
    PipelineSpec,
    PipelineStep,
)
from picarones.adapters.legacy_modules import TextToAltoMonoRegion
from picarones.adapters.legacy_modules.alto_text_to_mono_region import _build_alto_xml


# ──────────────────────────────────────────────────────────────────────────
# 1. _build_alto_xml — fonction pure
# ──────────────────────────────────────────────────────────────────────────


class TestBuildAltoXml:
    def test_minimal_text_produces_valid_xml(self):
        xml = _build_alto_xml("hello world", width=1000, height=2000)
        # Doit être parsable
        root = ET.fromstring(xml)
        assert "alto" in root.tag.lower()

    def test_namespace_is_alto_v4(self):
        xml = _build_alto_xml("a", width=100, height=200)
        assert "loc.gov/standards/alto/ns-v4" in xml

    def test_two_lines_produce_two_textlines(self):
        xml = _build_alto_xml("line one\nline two", width=1000, height=1000)
        root = ET.fromstring(xml)
        textlines = [el for el in root.iter() if el.tag.endswith("TextLine")]
        assert len(textlines) == 2

    def test_each_word_produces_one_string(self):
        xml = _build_alto_xml("alpha beta gamma", width=1000, height=1000)
        root = ET.fromstring(xml)
        strings = [el for el in root.iter() if el.tag.endswith("String")]
        assert len(strings) == 3
        contents = [s.attrib["CONTENT"] for s in strings]
        assert contents == ["alpha", "beta", "gamma"]

    def test_xml_chars_are_escaped(self):
        xml = _build_alto_xml('<foo> & "bar"', width=100, height=100)
        # Ne doit pas casser le parser malgré les méta-caractères XML
        root = ET.fromstring(xml)
        strings = [el for el in root.iter() if el.tag.endswith("String")]
        contents = "".join(s.attrib["CONTENT"] for s in strings)
        # Les caractères doivent être préservés sémantiquement
        assert "<foo>" in contents
        assert "&" in contents

    def test_invalid_dimensions_fall_back(self, caplog):
        with caplog.at_level("WARNING"):
            xml = _build_alto_xml("test", width=0, height=-5)
        root = ET.fromstring(xml)
        page = next(el for el in root.iter() if el.tag.endswith("Page"))
        # Dimensions de repli documentées
        assert int(page.attrib["WIDTH"]) > 0
        assert int(page.attrib["HEIGHT"]) > 0

    def test_empty_text_still_valid_xml(self):
        xml = _build_alto_xml("", width=100, height=100)
        root = ET.fromstring(xml)
        textlines = [el for el in root.iter() if el.tag.endswith("TextLine")]
        # Une TextLine vide est émise (placeholder) — ALTO valide
        assert len(textlines) == 1

    def test_deterministic(self):
        xml1 = _build_alto_xml("hello world", width=500, height=800)
        xml2 = _build_alto_xml("hello world", width=500, height=800)
        assert xml1 == xml2

    def test_image_filename_in_description(self):
        xml = _build_alto_xml(
            "x", width=10, height=10, image_filename="page_42.png",
        )
        assert "page_42.png" in xml


# ──────────────────────────────────────────────────────────────────────────
# 2. TextToAltoMonoRegion — module
# ──────────────────────────────────────────────────────────────────────────


class TestTextToAltoMonoRegion:
    def test_module_declares_correct_types(self):
        m = TextToAltoMonoRegion()
        assert m.input_types == (ArtifactType.IMAGE, ArtifactType.TEXT)
        assert m.output_types == (ArtifactType.ALTO,)
        assert m.execution_mode == "cpu"

    def test_module_name_default(self):
        assert TextToAltoMonoRegion().name == "alto_text_to_mono_region"

    def test_module_name_overridable(self):
        m = TextToAltoMonoRegion(config={"name": "my_baseline"})
        assert m.name == "my_baseline"

    def test_validate_inputs_missing_raises(self):
        m = TextToAltoMonoRegion()
        with pytest.raises(ValueError, match="entrées manquantes"):
            m.process({ArtifactType.TEXT: "hello"})

    def test_process_with_dimensions_tuple(self):
        m = TextToAltoMonoRegion()
        outputs = m.process({
            ArtifactType.IMAGE: (1024, 768),
            ArtifactType.TEXT: "hello world",
        })
        assert ArtifactType.ALTO in outputs
        xml = outputs[ArtifactType.ALTO]
        assert 'WIDTH="1024"' in xml
        assert 'HEIGHT="768"' in xml

    def test_process_with_missing_image_falls_back(self, caplog):
        m = TextToAltoMonoRegion()
        with caplog.at_level("WARNING"):
            outputs = m.process({
                ArtifactType.IMAGE: "/nonexistent/path/image.png",
                ArtifactType.TEXT: "hello",
            })
        xml = outputs[ArtifactType.ALTO]
        # Tombe sur les valeurs par défaut documentées
        assert 'WIDTH="2000"' in xml
        assert 'HEIGHT="3000"' in xml

    def test_process_accepts_textgt_payload(self):
        m = TextToAltoMonoRegion()
        outputs = m.process({
            ArtifactType.IMAGE: (100, 100),
            ArtifactType.TEXT: TextGT(text="aze rty"),
        })
        xml = outputs[ArtifactType.ALTO]
        assert 'CONTENT="aze"' in xml
        assert 'CONTENT="rty"' in xml

    def test_metadata_is_traceable(self):
        meta = TextToAltoMonoRegion().metadata()
        assert meta["module_kind"] == "alto_reconstructor"
        assert meta["variant"] == "mono_region_baseline"
        assert meta["deterministic"] is True

    def test_repr_contains_input_output_types(self):
        m = TextToAltoMonoRegion()
        r = repr(m)
        assert "image" in r
        assert "alto" in r
        assert "text" in r


# ──────────────────────────────────────────────────────────────────────────
# 3. extract_text_from_alto — parser tolérant
# ──────────────────────────────────────────────────────────────────────────


class TestExtractTextFromAlto:
    def test_round_trip_through_baseline(self):
        # Le reconstructeur produit un ALTO ; le parser doit
        # retrouver le texte (modulo whitespace).
        m = TextToAltoMonoRegion()
        out = m.process({
            ArtifactType.IMAGE: (200, 200),
            ArtifactType.TEXT: "hello world\nsecond line",
        })
        text = extract_text_from_alto(out[ArtifactType.ALTO])
        assert text == "hello world\nsecond line"

    def test_empty_string_returns_empty(self):
        assert extract_text_from_alto("") == ""

    def test_none_returns_empty(self):
        assert extract_text_from_alto(None) == ""

    def test_invalid_xml_returns_empty(self, caplog):
        with caplog.at_level("WARNING"):
            assert extract_text_from_alto("<not xml") == ""

    def test_alto_v2_without_namespace(self):
        xml = (
            '<?xml version="1.0"?><alto>'
            '<Layout><Page><PrintSpace><TextBlock>'
            '<TextLine><String CONTENT="foo"/><String CONTENT="bar"/>'
            '</TextLine>'
            '</TextBlock></PrintSpace></Page></Layout></alto>'
        )
        assert extract_text_from_alto(xml) == "foo bar"

    def test_accepts_altogt_object(self):
        xml = '<alto><TextLine><String CONTENT="x"/></TextLine></alto>'
        gt = AltoGT(xml_content=xml)
        assert extract_text_from_alto(gt) == "x"

    def test_multiple_textlines_joined_with_newline(self):
        xml = (
            '<alto>'
            '<TextLine><String CONTENT="a"/></TextLine>'
            '<TextLine><String CONTENT="b"/></TextLine>'
            '</alto>'
        )
        assert extract_text_from_alto(xml) == "a\nb"


# ──────────────────────────────────────────────────────────────────────────
# 4. Registre typé — métriques (ALTO, ALTO)
# ──────────────────────────────────────────────────────────────────────────


class TestAltoMetricsRegistration:
    def test_alto_metrics_are_registered(self):
        # L'import du module doit avoir peuplé le registre.
        import picarones.measurements.alto_metrics  # noqa: F401

        applicable = select_metrics(
            (ArtifactType.ALTO, ArtifactType.ALTO),
        )
        names = {spec.name for spec in applicable}
        assert "alto_text_cer" in names
        assert "alto_text_wer" in names
        assert "alto_text_mer" in names
        assert "alto_text_wil" in names

    def test_compute_at_junction_runs_alto_metrics(self):
        import picarones.measurements.alto_metrics  # noqa: F401
        ref = '<alto><TextLine><String CONTENT="hello"/></TextLine></alto>'
        hyp = '<alto><TextLine><String CONTENT="hello"/></TextLine></alto>'
        results = compute_at_junction(
            ref, hyp,
            (ArtifactType.ALTO, ArtifactType.ALTO),
        )
        assert results["alto_text_cer"] == pytest.approx(0.0)

    def test_alto_text_cer_value_is_correct(self):
        # jiwer est une dépendance dure de Picarones (cf. pyproject.toml) ;
        # un environnement sans jiwer ne peut pas faire tourner le bench
        # de toute façon.
        try:
            import jiwer  # noqa: F401
        except ImportError:
            pytest.skip("jiwer absent du runtime")
        ref = '<alto><TextLine><String CONTENT="abcd"/></TextLine></alto>'
        hyp = '<alto><TextLine><String CONTENT="abXd"/></TextLine></alto>'
        # 1 substitution sur 4 caractères → CER ≈ 0.25
        cer = alto_text_cer(ref, hyp)
        assert cer == pytest.approx(0.25, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────
# 5. End-to-end — pipeline OCR → ALTO + jonction (ALTO, ALTO)
# ──────────────────────────────────────────────────────────────────────────


class _MockOCRModule(BaseModule):
    """Simule un OCR qui retourne un texte fixe."""

    input_types = (ArtifactType.IMAGE,)
    output_types = (ArtifactType.TEXT,)
    execution_mode = "cpu"

    def __init__(self, text: str):
        self._text = text

    @property
    def name(self) -> str:
        return "mock_ocr"

    def process(self, inputs: dict[ArtifactType, Any]) -> dict[ArtifactType, Any]:
        self.validate_inputs(inputs)
        return {ArtifactType.TEXT: self._text}


class TestPipelineOCRToAltoEndToEnd:
    def test_pipeline_ocr_then_alto_runs_and_evaluates(self, tmp_path: Path):
        # Document avec une image factice (jamais lue par le mock OCR
        # mais nécessaire pour les dimensions ALTO du reconstructeur).
        img = tmp_path / "page_001.png"
        img.write_bytes(b"\x89PNG\r\n")  # En-tête PNG (Pillow lèvera, repli sur défaut)
        gt_alto = (
            '<alto><TextLine><String CONTENT="bonjour"/>'
            '<String CONTENT="monde"/></TextLine></alto>'
        )
        doc = Document(
            image_path=img,
            ground_truth="bonjour monde",
            ground_truths={
                GTLevel.TEXT: TextGT(text="bonjour monde"),
                GTLevel.ALTO: AltoGT(xml_content=gt_alto),
            },
        )

        pipeline = PipelineSpec(
            name="ocr_to_alto",
            steps=[
                PipelineStep("ocr", _MockOCRModule(text="bonjour monde")),
                PipelineStep("alto", TextToAltoMonoRegion()),
            ],
        )
        result = PipelineRunner.run(
            pipeline, doc, {ArtifactType.IMAGE: str(img)},
        )

        assert result.error is None, result.error
        assert result.succeeded, [s.error for s in result.steps]
        # L'étape OCR a évalué (TEXT, TEXT) → CER = 0
        ocr_step = result.steps[0]
        assert "text" in ocr_step.junction_metrics
        assert ocr_step.junction_metrics["text"]["cer"] == pytest.approx(0.0)
        # L'étape ALTO a évalué (ALTO, ALTO) → CER ≈ 0 sur le texte extrait
        alto_step = result.steps[1]
        assert "alto" in alto_step.junction_metrics
        assert alto_step.junction_metrics["alto"]["alto_text_cer"] == pytest.approx(
            0.0, abs=1e-6,
        )

    def test_pipeline_with_imperfect_ocr_shows_propagation(
        self, tmp_path: Path,
    ):
        """Quand l'OCR introduit une erreur, elle doit se voir aussi
        à la jonction ALTO — preuve que la mesure suit l'erreur le
        long du DAG."""
        img = tmp_path / "p.png"
        img.write_bytes(b"\x89PNG\r\n")
        gt_text = "abcd efgh"
        # Le reconstructeur baseline reproduira le texte tel quel ;
        # le CER sur le texte extrait de l'ALTO doit être identique
        # au CER sur le texte de l'OCR.
        gt_alto = (
            '<alto><TextLine><String CONTENT="abcd"/>'
            '<String CONTENT="efgh"/></TextLine></alto>'
        )
        doc = Document(
            image_path=img,
            ground_truth=gt_text,
            ground_truths={
                GTLevel.TEXT: TextGT(text=gt_text),
                GTLevel.ALTO: AltoGT(xml_content=gt_alto),
            },
        )
        pipeline = PipelineSpec(
            name="ocr_to_alto_imperfect",
            steps=[
                PipelineStep("ocr", _MockOCRModule(text="abXd efgh")),
                PipelineStep("alto", TextToAltoMonoRegion()),
            ],
        )
        result = PipelineRunner.run(
            pipeline, doc, {ArtifactType.IMAGE: str(img)},
        )

        ocr_cer = result.steps[0].junction_metrics["text"]["cer"]
        alto_cer = result.steps[1].junction_metrics["alto"]["alto_text_cer"]
        # Le baseline ne corrige pas, ne dégrade pas — les deux CER
        # sont identiques (preuve que le canal information est intact
        # à travers le reconstructeur ALTO).
        assert ocr_cer > 0
        assert alto_cer == pytest.approx(ocr_cer, abs=1e-6)
