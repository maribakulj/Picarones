"""Test E2E — pipeline OCR → reconstructeur ALTO de référence.

Chantier 1 du plan d'évolution post-Sprint 97. Ce test valide
**bout-en-bout** que :

1. ``BaseOCREngine`` factorisé honore le contrat ``BaseModule``
   (``process()`` propage le texte sous ``ArtifactType.TEXT``).
2. Le reconstructeur ALTO de référence
   (:class:`picarones.modules.TextToAltoMonoRegion`) accepte
   ``IMAGE + TEXT`` et produit un ``ArtifactType.ALTO`` valide.
3. Le ``PipelineRunner`` (Sprint 63) chaîne les deux étapes, propage
   ``IMAGE`` initial à la 2ᵉ étape (Sprint 66 — bag versionné), et
   évalue automatiquement aux jonctions ``(TEXT, TEXT)`` et
   ``(ALTO, ALTO)``.
4. Les métriques ``alto_text_cer/wer/mer/wil`` sont enregistrées et
   sélectionnées par ``compute_at_junction``.
5. Le YAML de référence ``examples/pipelines/ocr_to_alto.yaml`` est
   chargeable par :func:`pipeline_spec_loader.load_pipeline_spec_from_yaml`
   (validation de format).

L'objectif n'est pas de tester Tesseract (qui peut ne pas être
installé en CI) mais de valider la chaîne ``BaseModule + pipeline
runner + reconstructeur ALTO + métriques`` avec un mock OCR
déterministe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from picarones.core.corpus import AltoGT, Document, GTLevel, TextGT
from picarones.core.metric_registry import select_metrics
from picarones.core.modules import ArtifactType, BaseModule
from picarones.core.pipeline_runner import (
    PipelineRunner,
    PipelineSpec,
    PipelineStep,
)
from picarones.modules import TextToAltoMonoRegion


class _DeterministicOCR(BaseModule):
    """OCR factice : retourne un texte fixe quelles que soient les entrées."""

    input_types = (ArtifactType.IMAGE,)
    output_types = (ArtifactType.TEXT,)
    execution_mode = "cpu"

    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def name(self) -> str:
        return "deterministic_ocr"

    def process(self, inputs: dict[ArtifactType, Any]) -> dict[ArtifactType, Any]:
        self.validate_inputs(inputs)
        return {ArtifactType.TEXT: self._text}


@pytest.fixture
def doc_with_text_and_alto_gt(tmp_path: Path) -> Document:
    """Document factice avec GT TEXT + GT ALTO synchronisées."""
    img = tmp_path / "page_001.png"
    img.write_bytes(b"\x89PNG\r\n")  # En-tête PNG (Pillow lèvera, fallback dimensions par défaut)
    gt_text = "bonjour le monde"
    gt_alto = (
        '<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">'
        '<TextLine><String CONTENT="bonjour"/>'
        '<String CONTENT="le"/>'
        '<String CONTENT="monde"/></TextLine>'
        '</alto>'
    )
    return Document(
        image_path=img,
        ground_truth=gt_text,
        ground_truths={
            GTLevel.TEXT: TextGT(text=gt_text),
            GTLevel.ALTO: AltoGT(xml_content=gt_alto),
        },
    )


class TestPipelineOCRToAltoBaseline:
    """Valide la chaîne complète au comportement déterministe."""

    def test_alto_metrics_registered_on_alto_alto_junction(self):
        """Les métriques ALTO doivent être découvrables par le runner."""
        applicable = select_metrics(
            (ArtifactType.ALTO, ArtifactType.ALTO),
        )
        names = {m.name for m in applicable}
        assert {"alto_text_cer", "alto_text_wer"}.issubset(names)

    def test_perfect_ocr_produces_zero_cer_at_both_junctions(
        self, doc_with_text_and_alto_gt: Document,
    ):
        """OCR sans erreur → CER = 0 au TEXT et à l'ALTO."""
        try:
            import jiwer  # noqa: F401
        except ImportError:
            pytest.skip("jiwer absent — calcul CER impossible")

        spec = PipelineSpec(
            name="perfect_ocr",
            steps=[
                PipelineStep("ocr", _DeterministicOCR("bonjour le monde")),
                PipelineStep("alto", TextToAltoMonoRegion()),
            ],
        )
        result = PipelineRunner.run(
            spec, doc_with_text_and_alto_gt,
            {ArtifactType.IMAGE: str(doc_with_text_and_alto_gt.image_path)},
        )

        assert result.error is None, result.error
        assert result.succeeded, [s.error for s in result.steps]

        ocr_metrics = result.steps[0].junction_metrics["text"]
        alto_metrics = result.steps[1].junction_metrics["alto"]
        assert ocr_metrics["cer"] == pytest.approx(0.0)
        assert alto_metrics["alto_text_cer"] == pytest.approx(0.0, abs=1e-6)

    def test_imperfect_ocr_propagates_to_alto_junction(
        self, doc_with_text_and_alto_gt: Document,
    ):
        """Le baseline ne corrige pas → CER ALTO ≈ CER TEXT.

        Cette propriété est essentielle : elle prouve que le canal
        d'information est intact à travers le reconstructeur (le
        baseline n'absorbe pas et n'introduit pas d'erreurs).
        """
        try:
            import jiwer  # noqa: F401
        except ImportError:
            pytest.skip("jiwer absent — calcul CER impossible")

        spec = PipelineSpec(
            name="imperfect_ocr",
            steps=[
                # 1 substitution (j → u) + 1 espace en moins
                PipelineStep("ocr", _DeterministicOCR("bonjuor le monde")),
                PipelineStep("alto", TextToAltoMonoRegion()),
            ],
        )
        result = PipelineRunner.run(
            spec, doc_with_text_and_alto_gt,
            {ArtifactType.IMAGE: str(doc_with_text_and_alto_gt.image_path)},
        )

        text_cer = result.steps[0].junction_metrics["text"]["cer"]
        alto_cer = result.steps[1].junction_metrics["alto"]["alto_text_cer"]
        assert text_cer > 0
        assert alto_cer == pytest.approx(text_cer, abs=1e-6)

    def test_pipeline_succeeds_without_alto_gt(self, tmp_path: Path):
        """Quand le doc n'a pas de GT ALTO, la jonction (ALTO, ALTO) est
        sautée silencieusement (compute_at_junction n'est pas appelé)."""
        img = tmp_path / "p.png"
        img.write_bytes(b"\x89PNG\r\n")
        doc = Document(
            image_path=img,
            ground_truth="hello",
            # Pas de GT ALTO
        )
        spec = PipelineSpec(
            name="no_alto_gt",
            steps=[
                PipelineStep("ocr", _DeterministicOCR("hello")),
                PipelineStep("alto", TextToAltoMonoRegion()),
            ],
        )
        result = PipelineRunner.run(
            spec, doc, {ArtifactType.IMAGE: str(img)},
        )
        assert result.succeeded
        # L'étape ALTO produit bien un ALTO
        alto_step = result.steps[1]
        assert ArtifactType.ALTO in alto_step.output_types
        # Mais pas de jonction (ALTO, ALTO) puisque la GT manque
        assert "alto" not in alto_step.junction_metrics


class TestYamlSpec:
    """Valide que le YAML de référence livré dans examples/ est
    chargeable proprement."""

    def test_reference_yaml_is_loadable(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML absent")
        from picarones.core.pipeline_spec_loader import load_pipeline_spec_from_yaml

        repo_root = Path(__file__).resolve().parents[2]
        yaml_path = repo_root / "examples" / "pipelines" / "ocr_to_alto.yaml"
        assert yaml_path.exists(), f"YAML de référence absent : {yaml_path}"

        try:
            spec = load_pipeline_spec_from_yaml(yaml_path)
        except Exception as exc:
            # On tolère les erreurs d'import des classes référencées
            # quand pytesseract ou pero-ocr ne sont pas installés
            # (CI minimaliste).  Le test vérifie que la structure
            # YAML est syntaxiquement valide.
            from picarones.core.pipeline_spec_loader import PipelineSpecLoadError
            if isinstance(exc, PipelineSpecLoadError) and (
                "tesseract" in str(exc).lower() or "pytesseract" in str(exc).lower()
            ):
                pytest.skip(f"environnement sans Tesseract : {exc}")
            raise
        # Si on arrive ici, la structure est valide ET les classes
        # sont importables — le YAML est exécutable bout-en-bout.
        assert spec.name == "ocr_to_alto_baseline"
        assert len(spec.steps) == 2
        assert spec.steps[0].name == "ocr"
        assert spec.steps[1].name == "alto"
