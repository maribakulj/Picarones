"""Tests Phase 7.B.3 — couverture des nouvelles branches du benchmark.

Le ``run_pipeline_benchmark`` consomme désormais directement
``PipelineExecutor.run_plan`` (au lieu de ``PipelineRunner.run``).
Ce changement introduit deux nouvelles branches non triviales :

1. ``_initial_input_types_for_corpus`` : la factory peut lever
   document par document → on inspecte tant qu'on en trouve un
   qui marche.
2. ``run_pipeline_benchmark`` peut lever ``planning_error`` si le
   ``PipelinePlanner`` rejette la spec canonique (cohérence avec
   les messages d'erreur legacy).

Ces tests seront supprimés en sub-phase 7.D avec le module lui-même.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from picarones.domain.artifacts import ArtifactType
from picarones.domain.module_protocol import BaseModule
from picarones.evaluation.corpus import Corpus, Document, GTLevel, TextGT
from picarones.pipeline.legacy_pipeline_benchmark import (
    _initial_input_types_for_corpus,
    default_initial_inputs,
    run_pipeline_benchmark,
)
from picarones.pipeline.legacy_runner import PipelineSpec, PipelineStep


# ──────────────────────────────────────────────────────────────────────────
# Mocks
# ──────────────────────────────────────────────────────────────────────────


class _MockOCR(BaseModule):
    input_types = (ArtifactType.IMAGE,)
    output_types = (ArtifactType.RAW_TEXT,)
    execution_mode = "io"

    @property
    def name(self) -> str:
        return "mock-ocr"

    def process(self, inputs):
        return {ArtifactType.RAW_TEXT: "hello"}


def _make_doc(doc_id: str = "d") -> Document:
    return Document(
        doc_id=doc_id,
        image_path="/tmp/img.png",
        ground_truths={GTLevel.TEXT: TextGT(text="hello")},
    )


def _make_corpus(n: int = 2) -> Corpus:
    return Corpus(
        name="demo",
        documents=[_make_doc(doc_id=f"d{i}") for i in range(n)],
    )


# ──────────────────────────────────────────────────────────────────────────
# 1. _initial_input_types_for_corpus : factory qui lève sur tous les docs
# ──────────────────────────────────────────────────────────────────────────


class TestInitialInputTypesForCorpus:
    def test_returns_first_successful_factory_keys(self) -> None:
        # Cas heureux : la factory retourne des inputs depuis le
        # premier doc.
        docs = [_make_doc(doc_id=f"d{i}") for i in range(3)]
        result = _initial_input_types_for_corpus(docs, default_initial_inputs)
        assert result == (ArtifactType.IMAGE,)

    def test_skips_failing_factories_returns_first_ok(self) -> None:
        # La factory lève sur les 2 premiers docs mais marche sur le
        # 3ème → on retourne les types du 3ème.
        call_count = [0]

        def flaky_factory(doc):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("nope")
            return {ArtifactType.RAW_TEXT: "from-flaky"}

        docs = [_make_doc(doc_id=f"d{i}") for i in range(3)]
        result = _initial_input_types_for_corpus(docs, flaky_factory)
        assert result == (ArtifactType.RAW_TEXT,)
        assert call_count[0] == 3

    def test_returns_empty_tuple_when_all_factories_fail(self) -> None:
        # Toutes les factories lèvent → tuple vide ; la validation
        # amont legacy remontera "entrée manquante" pour chaque doc.
        def always_failing(doc):
            raise RuntimeError("permanent failure")

        docs = [_make_doc(doc_id=f"d{i}") for i in range(2)]
        result = _initial_input_types_for_corpus(docs, always_failing)
        assert result == ()

    def test_returns_empty_for_empty_corpus(self) -> None:
        result = _initial_input_types_for_corpus([], default_initial_inputs)
        assert result == ()


# ──────────────────────────────────────────────────────────────────────────
# 2. run_pipeline_benchmark : planner.plan lève → planning_error
# ──────────────────────────────────────────────────────────────────────────


class TestRunPipelineBenchmarkPlanningError:
    def test_planning_error_marks_all_docs_with_same_message(self) -> None:
        # On force PipelinePlanner.plan à lever — tous les documents
        # doivent recevoir le même PipelineResult avec le préfixe
        # "planning_error: …" et aucune étape ne doit être exécutée.
        spec = PipelineSpec(
            name="planning-fails",
            steps=[PipelineStep(name="ocr", module=_MockOCR())],
        )
        corpus = _make_corpus(n=2)

        with patch(
            "picarones.pipeline.legacy_pipeline_benchmark."
            "PipelinePlanner.plan",
            side_effect=RuntimeError("forced planning failure"),
        ):
            result = run_pipeline_benchmark(spec, corpus)

        assert result.n_docs == 2
        assert len(result.per_doc_results) == 2
        for pr in result.per_doc_results:
            assert pr.error is not None
            assert pr.error.startswith("planning_error: RuntimeError:")
            assert "forced planning failure" in pr.error
            assert pr.steps == []
        # Aucun aggregate puisqu'on est sorti avant la boucle.
        assert result.per_step_aggregates == []

    def test_planning_error_logs_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        spec = PipelineSpec(
            name="planning-fails-log",
            steps=[PipelineStep(name="ocr", module=_MockOCR())],
        )
        corpus = _make_corpus(n=1)

        with patch(
            "picarones.pipeline.legacy_pipeline_benchmark."
            "PipelinePlanner.plan",
            side_effect=ValueError("schema mismatch"),
        ):
            with caplog.at_level("WARNING"):
                run_pipeline_benchmark(spec, corpus)

        # Le warning identifie la spec en cause.
        assert any(
            "planning a levé sur planning-fails-log" in rec.message
            for rec in caplog.records
        )
