"""Sprint D.2.c-f — features additionnelles dans
``run_benchmark_via_service``.

Couvre les paramètres legacy auparavant ignorés :

- D.2.c (``output_json``) : déjà actif depuis D.1.d, couvert par
  ``les tests bout-en-bout du benchmark_runner``.
- D.2.d (``over_normalization``) : pour les pipelines OCR+LLM avec
  étape OCR amont, ``DocumentResult.pipeline_metadata`` porte
  désormais une clé ``over_normalization``.
- D.2.e (``entity_extractor``) : pour les documents avec une GT
  ``ENTITIES``, les métriques NER sont calculées + attachées.
- D.2.f (``profile``) : un profil inconnu lève ``PicaronesError``
  au démarrage du bench.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picarones.adapters.llm.base import BaseLLMAdapter
from picarones.adapters.ocr.base import BaseOCRAdapter
from picarones.app.services._benchmark_ner import (
    aggregate_ner_metrics as _aggregate_ner_metrics,
)
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.evaluation.corpus import (
    Corpus,
    Document,
    EntitiesGT,
)
from tests._migration_helpers import run_via_orchestrator


# ──────────────────────────────────────────────────────────────────────
# Mocks (canoniques)
# ──────────────────────────────────────────────────────────────────────


class _MockOCR(BaseOCRAdapter):
    def __init__(self, name: str = "mock_ocr", text: str = "ocr") -> None:
        self._name = name
        self._text = text

    @property
    def name(self) -> str:
        return self._name

    def execute(self, inputs, params, context):
        from pathlib import Path

        out_dir = Path(context.workspace_uri)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{context.document_id}_mock.txt"
        out_path.write_text(self._text, encoding="utf-8")
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:{self._name}:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
                uri=str(out_path),
            ),
        }


class _MockLLM(BaseLLMAdapter):
    def __init__(self, model: str = "mock-1", text: str = "corrected") -> None:
        super().__init__(model=model, config={})
        self._text = text

    @property
    def name(self) -> str:
        return "mock_llm"

    @property
    def default_model(self) -> str:
        return "mock-1"

    def _call(self, prompt, image_b64=None):
        return self._text


def _make_simple_corpus(tmp_path: Path, n: int = 1) -> Corpus:
    docs = []
    for i in range(n):
        img = tmp_path / f"doc{i}.png"
        img.write_bytes(b"x")
        docs.append(Document(
            image_path=img,
            ground_truth=f"texte {i}",
            doc_id=f"doc{i}",
        ))
    return Corpus(name="cdef_test", documents=docs)


# ──────────────────────────────────────────────────────────────────────
# D.2.f — profile validation
# ──────────────────────────────────────────────────────────────────────


class TestProfileValidation:
    """Sprint D.2.f — ``profile`` est validé au démarrage."""

    def test_unknown_profile_raises(self, tmp_path: Path) -> None:
        corpus = _make_simple_corpus(tmp_path)
        ocr = _MockOCR()

        with pytest.raises(ValueError, match="profil"):
            run_via_orchestrator(
                corpus, [ocr], profile="not_a_real_profile",
            )

    def test_standard_profile_accepted(self, tmp_path: Path) -> None:
        corpus = _make_simple_corpus(tmp_path)
        ocr = _MockOCR()
        bm = run_via_orchestrator(corpus, [ocr], profile="standard")
        assert bm.engine_reports

    def test_default_profile_is_standard(self, tmp_path: Path) -> None:
        """Pas de kwarg = utilise ``standard``, qui passe la validation."""
        corpus = _make_simple_corpus(tmp_path)
        ocr = _MockOCR()
        bm = run_via_orchestrator(corpus, [ocr])
        assert bm.engine_reports

    def test_validation_happens_before_bench(self, tmp_path: Path) -> None:
        """Le profil invalide lève AVANT toute exécution OCR (sinon on
        gâche du temps de calcul pour un nom mal orthographié)."""
        corpus = _make_simple_corpus(tmp_path)

        call_counter = {"n": 0}

        class _CountingOCR(_MockOCR):
            def _run_ocr(self, image_path):
                call_counter["n"] += 1
                return "ocr"

        ocr = _CountingOCR()
        with pytest.raises(ValueError):
            run_via_orchestrator(
                corpus, [ocr], profile="oops",
            )
        # OCR jamais appelé.
        assert call_counter["n"] == 0


# ──────────────────────────────────────────────────────────────────────
# D.2.d — over_normalization
# ──────────────────────────────────────────────────────────────────────


class TestOverNormalization:
    """Sprint D.2.d — les pipelines OCR+LLM avec OCR amont ont
    une clé ``over_normalization`` dans ``pipeline_metadata``."""

    def test_ocr_only_has_no_over_normalization(self, tmp_path: Path) -> None:
        """Un moteur OCR seul (pas de pipeline) n'a pas
        d'``over_normalization`` puisqu'il n'y a pas de LLM."""
        corpus = _make_simple_corpus(tmp_path)
        ocr = _MockOCR(text="texte 0")
        bm = run_via_orchestrator(corpus, [ocr])

        dr = bm.engine_reports[0].document_results[0]
        assert "over_normalization" not in dr.pipeline_metadata

    def test_pipeline_text_only_computes_over_normalization(
        self, tmp_path: Path,
    ) -> None:
        """Pipeline OCR+LLM en mode ``text_only`` : le LLM reçoit le
        texte OCR et le corrige.  ``over_normalization`` doit
        apparaître dans pipeline_metadata."""
        from picarones.pipeline.llm_pipeline_config import (
            OCRLLMPipelineConfig,
        )

        corpus = _make_simple_corpus(tmp_path)
        ocr = _MockOCR(name="upstream_ocr", text="texto 0")  # 1 erreur
        llm = _MockLLM(model="m1", text="texte 0")  # corrige bien
        pipeline = OCRLLMPipelineConfig(
            ocr_adapter=ocr,
            llm_adapter=llm,
            mode="text_only",
        )

        bm = run_via_orchestrator(corpus, [pipeline])

        dr = bm.engine_reports[0].document_results[0]
        assert dr.pipeline_metadata.get("is_pipeline") is True
        assert "over_normalization" in dr.pipeline_metadata
        # Le payload est un dict via OverNormalizationResult.as_dict().
        ov = dr.pipeline_metadata["over_normalization"]
        assert isinstance(ov, dict)

    def test_pipeline_zero_shot_has_no_over_normalization(
        self, tmp_path: Path,
    ) -> None:
        """Pipeline zero-shot : le VLM reçoit l'image directement, pas
        d'OCR amont, donc pas d'``ocr_intermediate`` et pas
        d'``over_normalization``."""
        from picarones.pipeline.llm_pipeline_config import (
            OCRLLMPipelineConfig,
        )

        corpus = _make_simple_corpus(tmp_path)
        llm = _MockLLM(model="vlm-1", text="texte 0")
        pipeline = OCRLLMPipelineConfig(
            llm_adapter=llm,
            mode="zero_shot",
        )

        bm = run_via_orchestrator(corpus, [pipeline])
        dr = bm.engine_reports[0].document_results[0]
        # Pipeline mais pas d'OCR amont → pas d'over_normalization.
        assert "over_normalization" not in dr.pipeline_metadata


# ──────────────────────────────────────────────────────────────────────
# D.2.e — NER attach via entity_extractor
# ──────────────────────────────────────────────────────────────────────


class TestNERAttach:
    """Sprint D.2.e — quand ``entity_extractor`` est fourni, les
    documents avec une GT ``ENTITIES`` reçoivent un ``ner_metrics``
    et l'engine_report a un ``aggregated_ner``."""

    def _make_corpus_with_entities(
        self, tmp_path: Path, n: int = 2,
    ) -> Corpus:
        from picarones.domain.artifacts import ArtifactType

        docs = []
        for i in range(n):
            img = tmp_path / f"d{i}.png"
            img.write_bytes(b"x")
            doc = Document(
                image_path=img,
                ground_truth=f"Jean {i} habite Paris",
                doc_id=f"d{i}",
            )
            doc.ground_truths[ArtifactType.ENTITIES] = EntitiesGT(
                entities=[
                    {"label": "PER", "start": 0, "end": 6 + len(str(i)),
                     "text": f"Jean {i}"},
                    {"label": "LOC", "start": 14 + len(str(i)),
                     "end": 19 + len(str(i)), "text": "Paris"},
                ],
            )
            docs.append(doc)
        return Corpus(name="ner_test", documents=docs)

    def test_no_extractor_no_ner_metrics(self, tmp_path: Path) -> None:
        corpus = self._make_corpus_with_entities(tmp_path)
        ocr = _MockOCR(text="Jean 0 habite Paris")

        bm = run_via_orchestrator(corpus, [ocr])
        report = bm.engine_reports[0]
        for dr in report.document_results:
            assert dr.ner_metrics is None
        assert report.aggregated_ner is None

    def test_extractor_attaches_metrics_to_doc(self, tmp_path: Path) -> None:
        """Quand l'extracteur retourne des entités sur l'hypothèse,
        ``ner_metrics`` apparaît sur le DocumentResult."""
        corpus = self._make_corpus_with_entities(tmp_path)
        ocr = _MockOCR(text="Jean 0 habite Paris")  # match parfait

        def extractor(text: str) -> list[dict]:
            # Reproduit les entités GT sur l'hypothèse.
            ents = []
            if "Jean 0" in text:
                ents.append({"label": "PER", "start": 0, "end": 6,
                             "text": "Jean 0"})
            if "Paris" in text:
                idx = text.find("Paris")
                ents.append({"label": "LOC", "start": idx,
                             "end": idx + 5, "text": "Paris"})
            return ents

        bm = run_via_orchestrator(
            corpus, [ocr], entity_extractor=extractor,
        )

        report = bm.engine_reports[0]
        d0 = next(d for d in report.document_results if d.doc_id == "d0")
        assert d0.ner_metrics is not None
        # Les entités matchent → tp > 0.
        assert d0.ner_metrics["true_positives"] > 0

    def test_aggregated_ner_present_when_any_doc_evaluated(
        self, tmp_path: Path,
    ) -> None:
        corpus = self._make_corpus_with_entities(tmp_path)
        ocr = _MockOCR(text="Jean 0 habite Paris")

        def extractor(text: str) -> list[dict]:
            return [{"label": "PER", "start": 0, "end": 6, "text": "Jean 0"}]

        bm = run_via_orchestrator(
            corpus, [ocr], entity_extractor=extractor,
        )

        report = bm.engine_reports[0]
        assert report.aggregated_ner is not None
        assert "global" in report.aggregated_ner
        assert "precision" in report.aggregated_ner["global"]

    def test_doc_without_entities_gt_skipped(self, tmp_path: Path) -> None:
        """Un document sans GT ``ENTITIES`` n'est pas évalué NER —
        ``ner_metrics`` reste ``None`` même si l'extracteur est
        fourni."""
        # Corpus mixte : 1 doc avec ENTITIES, 1 sans.
        from picarones.domain.artifacts import ArtifactType

        img1 = tmp_path / "d1.png"
        img1.write_bytes(b"x")
        doc_with = Document(
            image_path=img1, ground_truth="Jean", doc_id="with_ent",
        )
        doc_with.ground_truths[ArtifactType.ENTITIES] = EntitiesGT(
            entities=[{"label": "PER", "start": 0, "end": 4, "text": "Jean"}],
        )

        img2 = tmp_path / "d2.png"
        img2.write_bytes(b"x")
        doc_without = Document(
            image_path=img2, ground_truth="rien", doc_id="without_ent",
        )

        corpus = Corpus(
            name="mixed", documents=[doc_with, doc_without],
        )
        ocr = _MockOCR(text="Jean")

        def extractor(text: str) -> list[dict]:
            return [{"label": "PER", "start": 0, "end": 4, "text": "Jean"}]

        bm = run_via_orchestrator(
            corpus, [ocr], entity_extractor=extractor,
        )

        report = bm.engine_reports[0]
        d_with = next(
            d for d in report.document_results if d.doc_id == "with_ent"
        )
        d_without = next(
            d for d in report.document_results if d.doc_id == "without_ent"
        )

        assert d_with.ner_metrics is not None
        assert d_without.ner_metrics is None

    def test_extractor_exception_does_not_crash_bench(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        corpus = self._make_corpus_with_entities(tmp_path, n=1)
        ocr = _MockOCR(text="Jean 0 habite Paris")

        def buggy_extractor(text: str) -> list[dict]:
            raise RuntimeError("NER backend down")

        with caplog.at_level("WARNING"):
            bm = run_via_orchestrator(
                corpus, [ocr], entity_extractor=buggy_extractor,
            )

        report = bm.engine_reports[0]
        # Le bench a abouti — pas d'exception propagée.
        assert len(report.document_results) == 1
        # ner_metrics non attaché à cause du crash.
        assert report.document_results[0].ner_metrics is None


# ──────────────────────────────────────────────────────────────────────
# D.2.e — agrégation NER (helper interne testé directement)
# ──────────────────────────────────────────────────────────────────────


class TestAggregateNERMetrics:
    """Tests unitaires de ``_aggregate_ner_metrics`` — équivalent
    fonctionnel de l'ex-``measurements.runner.ner_attach._aggregate_ner``."""

    def test_empty_returns_none(self) -> None:
        from picarones.evaluation.benchmark_result import (
            DocumentResult,
        )
        from picarones.evaluation.metric_result import MetricsResult

        # Aucun ner_metrics sur les docs.
        drs = [
            DocumentResult(
                doc_id="d", image_path="", ground_truth="",
                hypothesis="", metrics=MetricsResult(), duration_seconds=0,
            ),
        ]
        assert _aggregate_ner_metrics(drs) is None

    def test_aggregates_global_prf(self) -> None:
        from picarones.evaluation.benchmark_result import (
            DocumentResult,
        )
        from picarones.evaluation.metric_result import MetricsResult

        dr1 = DocumentResult(
            doc_id="d1", image_path="", ground_truth="",
            hypothesis="", metrics=MetricsResult(), duration_seconds=0,
        )
        dr1.ner_metrics = {
            "true_positives": 5,
            "false_positives": 1,
            "false_negatives": 2,
            "per_category": {},
            "hallucinated_entities": [],
            "missed_entities": [],
        }
        dr2 = DocumentResult(
            doc_id="d2", image_path="", ground_truth="",
            hypothesis="", metrics=MetricsResult(), duration_seconds=0,
        )
        dr2.ner_metrics = {
            "true_positives": 3,
            "false_positives": 0,
            "false_negatives": 1,
            "per_category": {},
            "hallucinated_entities": [],
            "missed_entities": [],
        }

        agg = _aggregate_ner_metrics([dr1, dr2])

        assert agg is not None
        # tp=8, fp=1, fn=3 → P=8/9, R=8/11, F1=2*P*R/(P+R)
        assert agg["global"]["precision"] == pytest.approx(8 / 9, abs=1e-4)
        assert agg["global"]["recall"] == pytest.approx(8 / 11, abs=1e-4)
        assert agg["n_documents"] == 2

    def test_per_category_aggregation(self) -> None:
        from picarones.evaluation.benchmark_result import (
            DocumentResult,
        )
        from picarones.evaluation.metric_result import MetricsResult

        dr = DocumentResult(
            doc_id="d", image_path="", ground_truth="",
            hypothesis="", metrics=MetricsResult(), duration_seconds=0,
        )
        dr.ner_metrics = {
            "true_positives": 4,
            "false_positives": 1,
            "false_negatives": 1,
            "per_category": {
                "PER": {
                    "support": 3, "recall": 1.0, "precision": 1.0,
                    "f1": 1.0,
                },
                "LOC": {
                    "support": 2, "recall": 0.5, "precision": 0.5,
                    "f1": 0.5,
                },
            },
            "hallucinated_entities": [],
            "missed_entities": [],
        }

        agg = _aggregate_ner_metrics([dr])

        assert "PER" in agg["per_category"]
        assert "LOC" in agg["per_category"]
        # PER : 3/3 → P=R=F1=1.0
        assert agg["per_category"]["PER"]["recall"] == pytest.approx(1.0)
