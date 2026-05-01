"""Tests Sprint 64 — orchestration corpus-wide d'une pipeline composée.

Couvre :

1. ``default_initial_inputs`` : ``IMAGE`` extrait de
   ``Document.image_path`` ; absence d'image → dict vide.
2. ``run_pipeline_benchmark`` :
   - corpus vide → résultat avec ``n_docs == 0``
   - 1 doc, pipeline OK → succeeded == 1, agrégat cohérent
   - N docs, mix succès/échecs → comptes corrects, failing_doc_ids
     listés
   - Métriques agrégées (mean/median CER) sur N docs
   - Factory personnalisée respectée
   - Factory qui lève sur un doc → erreur capturée, autres docs
     continuent
   - Spec invalide → tous les docs échouent en amont
3. ``StepAggregate`` :
   - success_rate
   - error_breakdown (missing_input, raised_exception,
     missing_output, other)
   - junction_metrics agrégés correctement par type d'artefact
4. ``PipelineBenchmarkResult.aggregate_for_step`` retourne le bon
   agrégat ou ``None``.
5. Philosophie : tous les modules utilisés sont des **mocks**
   définis dans le test — Picarones n'expose aucun module métier.
"""

from __future__ import annotations

from typing import Any

from picarones.core.corpus import Corpus, Document, GTLevel, TextGT
from picarones.core.modules import ArtifactType, BaseModule
from picarones.measurements.pipeline_benchmark import (
    PipelineBenchmarkResult,
    StepAggregate,
    default_initial_inputs,
    run_pipeline_benchmark,
)
from picarones.core.pipeline import PipelineSpec, PipelineStep


# ──────────────────────────────────────────────────────────────────────────
# Mocks (inchangés vs Sprint 63 — uniquement à but de test)
# ──────────────────────────────────────────────────────────────────────────


class MockOCR(BaseModule):
    input_types = (ArtifactType.IMAGE,)
    output_types = (ArtifactType.TEXT,)
    execution_mode: Any = "io"

    def __init__(self, fn) -> None:
        self._fn = fn

    @property
    def name(self) -> str:
        return "mock-ocr"

    def process(self, inputs):
        return {ArtifactType.TEXT: self._fn(inputs[ArtifactType.IMAGE])}


class MockCrasherSometimes(BaseModule):
    """OCR qui lève sur les documents dont le path contient un
    marqueur, et qui produit du texte sinon."""

    input_types = (ArtifactType.IMAGE,)
    output_types = (ArtifactType.TEXT,)
    execution_mode: Any = "io"

    def __init__(self, crash_on: str) -> None:
        self._crash_on = crash_on

    @property
    def name(self) -> str:
        return "mock-flaky"

    def process(self, inputs):
        path = inputs[ArtifactType.IMAGE]
        if self._crash_on in path:
            raise RuntimeError(f"refusé : {path}")
        return {ArtifactType.TEXT: "ok"}


class MockTextRewriter(BaseModule):
    input_types = (ArtifactType.TEXT,)
    output_types = (ArtifactType.TEXT,)
    execution_mode: Any = "cpu"

    @property
    def name(self) -> str:
        return "mock-rewriter"

    def process(self, inputs):
        return {ArtifactType.TEXT: inputs[ArtifactType.TEXT].upper()}


def _make_corpus(n: int = 3, name: str = "demo") -> Corpus:
    docs = []
    for i in range(n):
        gt = f"texte {i}"
        docs.append(Document(
            image_path=f"/tmp/d{i}.png",
            ground_truth=gt,
            doc_id=f"d{i}",
            ground_truths={GTLevel.TEXT: TextGT(text=gt)},
        ))
    return Corpus(name=name, documents=docs)


# ──────────────────────────────────────────────────────────────────────────
# 1. default_initial_inputs
# ──────────────────────────────────────────────────────────────────────────


class TestDefaultInitialInputs:
    def test_returns_image_path(self) -> None:
        doc = Document(
            image_path="/tmp/x.png", ground_truth="t", doc_id="d",
        )
        assert default_initial_inputs(doc) == {
            ArtifactType.IMAGE: "/tmp/x.png",
        }

    def test_empty_when_no_image_path(self) -> None:
        doc = Document(image_path="", ground_truth="t", doc_id="d")
        assert default_initial_inputs(doc) == {}


# ──────────────────────────────────────────────────────────────────────────
# 2. run_pipeline_benchmark — chemins nominaux
# ──────────────────────────────────────────────────────────────────────────


class TestRunBenchmarkBasic:
    def test_empty_corpus(self) -> None:
        corpus = Corpus(name="empty", documents=[])
        spec = PipelineSpec(
            name="ocr",
            steps=[PipelineStep("ocr", MockOCR(lambda p: "x"))],
        )
        result = run_pipeline_benchmark(spec, corpus)
        assert result.n_docs == 0
        assert result.per_doc_results == []
        # L'agrégation existe toujours (1 entrée par étape)
        assert len(result.per_step_aggregates) == 1
        assert result.per_step_aggregates[0].n_docs == 0
        assert result.per_step_aggregates[0].n_succeeded == 0

    def test_single_doc_success(self) -> None:
        corpus = _make_corpus(1)
        spec = PipelineSpec(
            name="ocr",
            steps=[PipelineStep("ocr", MockOCR(lambda p: "texte 0"))],
        )
        result = run_pipeline_benchmark(spec, corpus)
        assert result.n_docs == 1
        assert result.n_pipelines_succeeded == 1
        assert result.n_pipelines_failed == 0
        agg = result.aggregate_for_step("ocr")
        assert agg.n_succeeded == 1
        assert agg.success_rate == 1.0

    def test_metrics_aggregation(self) -> None:
        corpus = _make_corpus(3)
        # OCR parfait : produit le ground_truth depuis le path
        # (path == /tmp/d0.png → "texte 0", etc.)
        def ocr_perfect(path: str) -> str:
            idx = path.replace("/tmp/d", "").replace(".png", "")
            return f"texte {idx}"
        spec = PipelineSpec(
            name="ocr",
            steps=[PipelineStep("ocr", MockOCR(ocr_perfect))],
        )
        result = run_pipeline_benchmark(spec, corpus)
        agg = result.aggregate_for_step("ocr")
        cer_stats = agg.junction_metrics["text"]["cer"]
        assert cer_stats["mean"] == 0.0
        assert cer_stats["median"] == 0.0
        assert cer_stats["n"] == 3


# ──────────────────────────────────────────────────────────────────────────
# 3. Mix succès / échecs
# ──────────────────────────────────────────────────────────────────────────


class TestMixedResults:
    def test_some_docs_fail(self) -> None:
        corpus = _make_corpus(4)
        # Crash sur les docs dont le path contient "d2"
        spec = PipelineSpec(
            name="flaky",
            steps=[PipelineStep("ocr", MockCrasherSometimes("d2"))],
        )
        result = run_pipeline_benchmark(spec, corpus)
        assert result.n_pipelines_succeeded == 3
        assert result.n_pipelines_failed == 1
        agg = result.aggregate_for_step("ocr")
        assert agg.n_failed == 1
        assert "d2" in agg.failing_doc_ids
        # Type d'erreur catégorisé
        assert agg.error_breakdown.get("raised_exception", 0) == 1

    def test_two_steps_second_fails_for_subset(self) -> None:
        """Étape 1 réussit toujours, étape 2 plante pour 1 doc.

        Le rewriter est OK ici, on simule un échec en faisant lever
        l'OCR sur d1 puis on laisse rewriter produire à partir des
        OCR qui ont survécu.
        """
        corpus = _make_corpus(3)
        spec = PipelineSpec(
            name="ocr_then_rewrite",
            steps=[
                PipelineStep("ocr", MockCrasherSometimes("d1")),
                PipelineStep("rewrite", MockTextRewriter()),
            ],
        )
        result = run_pipeline_benchmark(spec, corpus)
        # OCR : 2 OK, 1 KO
        ocr_agg = result.aggregate_for_step("ocr")
        assert ocr_agg.n_succeeded == 2
        assert ocr_agg.n_failed == 1
        # Rewriter : 2 OK (ceux dont l'OCR a réussi), 1 « entrée
        # manquante » (pour d1 dont l'OCR a planté)
        rw_agg = result.aggregate_for_step("rewrite")
        assert rw_agg.n_succeeded == 2
        assert rw_agg.n_failed == 1
        assert rw_agg.error_breakdown.get("missing_input", 0) == 1


# ──────────────────────────────────────────────────────────────────────────
# 4. Spec invalide → tous les docs échouent en amont
# ──────────────────────────────────────────────────────────────────────────


class TestInvalidSpec:
    def test_invalid_spec_propagates_to_all_docs(self) -> None:
        corpus = _make_corpus(2)
        # Pipeline qui demande TEXT mais on ne fournit que IMAGE
        # et aucun OCR ne précède
        spec = PipelineSpec(
            name="bad",
            steps=[PipelineStep("rewrite", MockTextRewriter())],
        )
        result = run_pipeline_benchmark(spec, corpus)
        assert all(pr.error is not None for pr in result.per_doc_results)
        # Aucune étape n'a été exécutée → tous les docs ont 0 step
        for pr in result.per_doc_results:
            assert pr.steps == []
        # L'agrégat de l'étape "rewrite" est entièrement en
        # pipeline_aborted
        agg = result.aggregate_for_step("rewrite")
        assert agg.n_docs == 2
        assert agg.n_failed == 2
        assert agg.error_breakdown.get("pipeline_aborted", 0) == 2


# ──────────────────────────────────────────────────────────────────────────
# 5. Factory personnalisée
# ──────────────────────────────────────────────────────────────────────────


class TestCustomFactory:
    def test_custom_factory_used(self) -> None:
        corpus = _make_corpus(2)
        # Factory qui injecte directement du TEXT — pas besoin d'OCR
        def factory(doc):
            return {ArtifactType.TEXT: doc.ground_truth}

        spec = PipelineSpec(
            name="rewrite_only",
            steps=[PipelineStep("rewrite", MockTextRewriter())],
        )
        result = run_pipeline_benchmark(spec, corpus, factory)
        assert result.n_pipelines_succeeded == 2
        agg = result.aggregate_for_step("rewrite")
        assert agg.n_succeeded == 2

    def test_factory_raises_for_one_doc(self) -> None:
        corpus = _make_corpus(3)
        def flaky_factory(doc):
            if doc.doc_id == "d1":
                raise ValueError("factory cassée pour d1")
            return {ArtifactType.IMAGE: doc.image_path}

        spec = PipelineSpec(
            name="ocr",
            steps=[PipelineStep("ocr", MockOCR(lambda p: "ok"))],
        )
        result = run_pipeline_benchmark(spec, corpus, flaky_factory)
        # 2 OK, 1 KO en amont
        assert result.n_pipelines_succeeded == 2
        assert result.n_pipelines_failed == 1
        # Le doc d1 a un PipelineResult avec erreur factory et
        # aucune étape exécutée
        d1 = next(pr for pr in result.per_doc_results if pr.doc_id == "d1")
        assert "ValueError" in d1.error
        assert d1.steps == []


# ──────────────────────────────────────────────────────────────────────────
# 6. Dataclasses
# ──────────────────────────────────────────────────────────────────────────


class TestDataclasses:
    def test_step_aggregate_default(self) -> None:
        agg = StepAggregate(step_name="x")
        assert agg.success_rate == 0.0
        assert agg.junction_metrics == {}
        assert agg.error_breakdown == {}

    def test_step_aggregate_success_rate(self) -> None:
        agg = StepAggregate(
            step_name="x", n_docs=10, n_succeeded=7, n_failed=3,
        )
        assert agg.success_rate == 0.7

    def test_pipeline_benchmark_result_aggregate_for_step(self) -> None:
        result = PipelineBenchmarkResult(
            pipeline_name="p", corpus_name="c",
            per_step_aggregates=[
                StepAggregate(step_name="a", n_docs=1),
                StepAggregate(step_name="b", n_docs=2),
            ],
        )
        assert result.aggregate_for_step("a").n_docs == 1
        assert result.aggregate_for_step("b").n_docs == 2
        assert result.aggregate_for_step("c") is None
