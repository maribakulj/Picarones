"""Tests Phase 7.B.2 — couverture des helpers internes du runner legacy.

Le runner legacy consume désormais ``PipelineExecutor`` via des
helpers privés (``_translate_canonical_error``,
``_gt_payload_to_value``, ``_artifact_type_to_gt_level``,
``_compute_junction_metrics_for_step``).  Les ~440 tests Sprint 63
existants couvrent la boucle nominale, mais certaines branches
défensives ne sont pas atteintes — d'où ces tests unitaires ciblés.

Ces tests seront supprimés en sub-phase 7.D avec les helpers
eux-mêmes.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from picarones.domain.artifacts import ArtifactType
from picarones.evaluation.corpus import (
    AltoGT,
    Document,
    EntitiesGT,
    PageGT,
    ReadingOrderGT,
    TextGT,
)
from picarones.pipeline._legacy_module_adapter import _PayloadRegistry
from picarones.pipeline._legacy_translator import (
    artifact_type_to_gt_level as _artifact_type_to_gt_level,
    build_legacy_step_result as _build_legacy_step_result,
    compute_junction_metrics_for_step as _compute_junction_metrics_for_step,
    gt_payload_to_value as _gt_payload_to_value,
    translate_canonical_error as _translate_canonical_error,
)
from picarones.pipeline.types import StepResult as _CanonicalStepResult


# ──────────────────────────────────────────────────────────────────────────
# 1. _translate_canonical_error — 6 branches + passthrough
# ──────────────────────────────────────────────────────────────────────────


class TestTranslateCanonicalError:
    def test_none_input_returns_none(self) -> None:
        assert _translate_canonical_error(None) is None

    def test_adapter_raised_strips_prefix(self) -> None:
        msg = _translate_canonical_error("adapter_raised: TypeError: bla")
        assert msg == "TypeError: bla"

    def test_missing_input(self) -> None:
        msg = _translate_canonical_error("missing_input: text@step1")
        assert msg == "entrée manquante : text@step1"

    def test_missing_output_parses_list_repr(self) -> None:
        msg = _translate_canonical_error(
            "missing_output: ['raw_text', 'alto_xml']",
        )
        assert msg == "sortie manquante : raw_text,alto_xml"

    def test_missing_output_single_value(self) -> None:
        msg = _translate_canonical_error("missing_output: ['raw_text']")
        assert msg == "sortie manquante : raw_text"

    def test_adapter_not_found(self) -> None:
        # Branche défensive : l'API legacy ne peut pas produire ce
        # cas (le resolver est un dict.__getitem__ construit par
        # nous), mais le canonique pourrait l'émettre — le runner
        # doit savoir le traduire au cas où.
        msg = _translate_canonical_error("adapter_not_found: tesseract")
        assert msg == "adapter introuvable : tesseract"

    def test_adapter_resolver_failed(self) -> None:
        msg = _translate_canonical_error(
            "adapter_resolver_failed: Connection refused",
        )
        assert msg == "résolution adapter échouée : Connection refused"

    def test_unknown_format_passthrough(self) -> None:
        # Format inconnu : le runner laisse le message tel quel.
        assert _translate_canonical_error("foo: bar") == "foo: bar"

    def test_empty_string_passthrough(self) -> None:
        # Edge case : chaîne vide sans préfixe matchant.
        assert _translate_canonical_error("") == ""


# ──────────────────────────────────────────────────────────────────────────
# 2. _gt_payload_to_value — 5 types de GT + passthrough
# ──────────────────────────────────────────────────────────────────────────


class TestGtPayloadToValue:
    def test_text_gt_returns_text(self) -> None:
        gt = TextGT(text="hello world")
        assert _gt_payload_to_value(gt) == "hello world"

    def test_entities_gt_returns_list(self) -> None:
        gt = EntitiesGT(
            entities=[{"text": "Paris", "label": "LOC", "start": 0, "end": 5}],
        )
        result = _gt_payload_to_value(gt)
        assert result == [
            {"text": "Paris", "label": "LOC", "start": 0, "end": 5},
        ]

    def test_reading_order_returns_region_order(self) -> None:
        gt = ReadingOrderGT(region_order=["r1", "r2", "r3"])
        assert _gt_payload_to_value(gt) == ["r1", "r2", "r3"]

    def test_alto_gt_passthrough(self) -> None:
        # ALTO/PAGE retournés tels quels — la métrique sait quoi
        # en faire selon sa signature.
        gt = AltoGT(xml_content="<alto/>")
        assert _gt_payload_to_value(gt) is gt

    def test_page_gt_passthrough(self) -> None:
        gt = PageGT(xml_content="<page/>")
        assert _gt_payload_to_value(gt) is gt

    def test_unknown_payload_passthrough(self) -> None:
        # Cas défensif : un caller passe un truc qui n'est pas un
        # GTPayload typé (par exemple un dict brut).  Le runner
        # retourne tel quel.
        assert _gt_payload_to_value({"foo": "bar"}) == {"foo": "bar"}
        assert _gt_payload_to_value("plain string") == "plain string"


# ──────────────────────────────────────────────────────────────────────────
# 3. _artifact_type_to_gt_level — types sans GT mapping
# ──────────────────────────────────────────────────────────────────────────


class TestArtifactTypeToGtLevel:
    def test_image_returns_none(self) -> None:
        # IMAGE n'est jamais évalué (entrée typique d'un OCR).
        assert _artifact_type_to_gt_level(ArtifactType.IMAGE) is None

    def test_confidences_returns_none(self) -> None:
        assert _artifact_type_to_gt_level(ArtifactType.CONFIDENCES) is None

    def test_alignment_returns_none(self) -> None:
        assert _artifact_type_to_gt_level(ArtifactType.ALIGNMENT) is None

    def test_canonical_document_returns_none(self) -> None:
        assert _artifact_type_to_gt_level(
            ArtifactType.CANONICAL_DOCUMENT,
        ) is None


# ──────────────────────────────────────────────────────────────────────────
# 4. _compute_junction_metrics_for_step — branches sans payload
# ──────────────────────────────────────────────────────────────────────────


def _make_doc(text: str | None = None) -> Document:
    """Construit un Document minimal avec ou sans GT texte."""
    gts: dict = {}
    if text is not None:
        from picarones.evaluation.corpus import GTLevel
        gts[GTLevel.TEXT] = TextGT(text=text)
    return Document(doc_id="doc1", image_path=None, ground_truths=gts)


def _make_canonical_step_result(
    produced: dict[str, str], step_id: str = "ocr",
) -> _CanonicalStepResult:
    return _CanonicalStepResult(
        step_id=step_id,
        succeeded=True,
        duration_seconds=0.001,
        produced_artifacts=produced,
    )


class TestComputeJunctionMetricsForStep:
    def test_no_gt_for_artifact_type_skipped(self) -> None:
        # IMAGE n'a pas de GT mapping → la jonction est skip.
        doc = _make_doc(text="hello")
        registry = _PayloadRegistry()
        canonical_sr = _make_canonical_step_result({"image": "doc1:ocr:image"})
        registry.store("doc1:ocr:image", "/path/img.png")

        metrics = _compute_junction_metrics_for_step(
            [ArtifactType.IMAGE], canonical_sr, registry, doc,
        )
        # Pas d'évaluation parce que IMAGE n'a pas de GTLevel.
        assert metrics == {}

    def test_no_gt_payload_on_document_skipped(self) -> None:
        # Document SANS TextGT — la jonction RAW_TEXT est skip.
        doc = _make_doc(text=None)
        registry = _PayloadRegistry()
        canonical_sr = _make_canonical_step_result(
            {"raw_text": "doc1:ocr:raw_text"},
        )
        registry.store("doc1:ocr:raw_text", "produced text")

        metrics = _compute_junction_metrics_for_step(
            [ArtifactType.RAW_TEXT], canonical_sr, registry, doc,
        )
        assert metrics == {}

    def test_artifact_id_missing_from_registry_skipped(self) -> None:
        # Le canonique a déclaré avoir produit un artifact mais le
        # registre ne contient pas son payload (ne devrait pas
        # arriver, mais branche défensive).
        doc = _make_doc(text="hello")
        registry = _PayloadRegistry()
        canonical_sr = _make_canonical_step_result(
            {"raw_text": "doc1:ocr:raw_text_unknown"},
        )
        # On ne stocke RIEN dans le registry.

        metrics = _compute_junction_metrics_for_step(
            [ArtifactType.RAW_TEXT], canonical_sr, registry, doc,
        )
        assert metrics == {}

    def test_missing_artifact_id_in_produced_artifacts(self) -> None:
        # produced_artifacts ne contient pas le type demandé.
        doc = _make_doc(text="hello")
        registry = _PayloadRegistry()
        canonical_sr = _make_canonical_step_result({})

        metrics = _compute_junction_metrics_for_step(
            [ArtifactType.RAW_TEXT], canonical_sr, registry, doc,
        )
        assert metrics == {}

    def test_compute_at_junction_exception_skipped_with_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Si compute_at_junction lève, on logue et on continue.
        doc = _make_doc(text="hello")
        registry = _PayloadRegistry()
        canonical_sr = _make_canonical_step_result(
            {"raw_text": "doc1:ocr:raw_text"},
        )
        registry.store("doc1:ocr:raw_text", "produced text")

        with patch(
            "picarones.pipeline._legacy_translator.compute_at_junction",
            side_effect=RuntimeError("boom"),
        ):
            with caplog.at_level("WARNING"):
                metrics = _compute_junction_metrics_for_step(
                    [ArtifactType.RAW_TEXT], canonical_sr, registry, doc,
                )

        assert metrics == {}
        assert any("boom" in rec.message for rec in caplog.records)

    def test_nominal_path_computes_metrics(self) -> None:
        # Cas heureux : tout est en place, des métriques sont
        # calculées.
        doc = _make_doc(text="hello world")
        registry = _PayloadRegistry()
        canonical_sr = _make_canonical_step_result(
            {"raw_text": "doc1:ocr:raw_text"},
        )
        registry.store("doc1:ocr:raw_text", "hello world")

        metrics = _compute_junction_metrics_for_step(
            [ArtifactType.RAW_TEXT], canonical_sr, registry, doc,
        )
        # Au minimum CER est calculé (= 0 puisque hyp == GT).
        assert "raw_text" in metrics
        assert metrics["raw_text"]["cer"] == 0.0


# ──────────────────────────────────────────────────────────────────────────
# 5. PipelineStep / PipelineSpec / PipelineResult — branches __repr__
# ──────────────────────────────────────────────────────────────────────────


from picarones.domain.module_protocol import BaseModule
from picarones.pipeline.legacy_runner import (
    PipelineResult,
    PipelineSpec,
    PipelineStep,
    StepResult,
)


class _MockModule(BaseModule):
    """Module factice minimal pour les tests __repr__."""

    input_types = (ArtifactType.IMAGE,)
    output_types = (ArtifactType.RAW_TEXT,)
    execution_mode = "io"

    @property
    def name(self) -> str:
        return "mock-module"

    def process(self, inputs):  # noqa: D401
        return {ArtifactType.RAW_TEXT: ""}


class TestPipelineStepRepr:
    def test_repr_without_inputs_from(self) -> None:
        step = PipelineStep(name="ocr", module=_MockModule())
        r = repr(step)
        assert "PipelineStep(ocr:" in r
        assert "image" in r
        assert "raw_text" in r

    def test_repr_with_inputs_from(self) -> None:
        step = PipelineStep(
            name="ocr",
            module=_MockModule(),
            inputs_from={ArtifactType.IMAGE: "__initial__"},
        )
        r = repr(step)
        assert "image@__initial__" in r


class TestPipelineSpecRepr:
    def test_repr_lists_steps(self) -> None:
        spec = PipelineSpec(
            name="my-pipe",
            steps=[
                PipelineStep(name="s1", module=_MockModule()),
                PipelineStep(name="s2", module=_MockModule()),
            ],
        )
        r = repr(spec)
        assert "my-pipe" in r
        assert "s1" in r
        assert "s2" in r
        assert "→" in r


class TestPipelineResultJunctionMetricsFor:
    def test_returns_none_when_no_step_succeeded(self) -> None:
        # Tous les steps ont une erreur → junction_metrics_for retourne None.
        result = PipelineResult(pipeline_name="p", doc_id="d")
        result.steps.append(
            StepResult(
                step_name="s1",
                duration_seconds=0.0,
                output_types=(),
                error="forced failure",
            ),
        )
        assert result.junction_metrics_for(ArtifactType.RAW_TEXT) is None


# ──────────────────────────────────────────────────────────────────────────
# 6. _build_legacy_step_result — branche défensive parsing type invalide
# ──────────────────────────────────────────────────────────────────────────


# _build_legacy_step_result importé en haut depuis _legacy_translator


class TestBuildLegacyStepResultDefensive:
    def test_skips_invalid_artifact_type_value(self) -> None:
        # Cas défensif : produced_artifacts contient une clé qui
        # n'est pas un ArtifactType.value valide.  Ne devrait pas
        # arriver dans le flow normal du PipelineExecutor (qui ne
        # produit que des ArtifactType valides), mais le runner
        # défend en profondeur.
        legacy_step = PipelineStep(name="s", module=_MockModule())
        canonical_sr = _make_canonical_step_result(
            {
                "raw_text": "doc1:s:raw_text",
                "completely_invalid_type": "doc1:s:invalid",
            },
        )
        registry = _PayloadRegistry()
        registry.store("doc1:s:raw_text", "hello")
        doc = _make_doc(text="hello")

        legacy_sr = _build_legacy_step_result(
            legacy_step, canonical_sr, registry, doc,
        )
        # L'ArtifactType invalide est ignoré ; seul RAW_TEXT figure
        # dans output_types.
        assert ArtifactType.RAW_TEXT in legacy_sr.output_types
        assert len(legacy_sr.output_types) == 1
