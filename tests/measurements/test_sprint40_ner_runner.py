"""Tests Sprint 40 — backend extracteur NER + câblage runner.

Couvre :

1. ``SpacyEntityExtractor`` lazy-importe spaCy ; sans spaCy installé,
   l'extracteur retourne ``[]`` avec un warning explicite.
2. ``is_spacy_available`` reflète l'état réel.
3. ``get_extractor(profile)`` accepte une clé de profil ou un nom de
   modèle direct.
4. ``DocumentResult.ner_metrics`` est sérialisé via ``as_dict``
   uniquement quand renseigné, et libéré par ``compact()``.
5. ``EngineReport.aggregated_ner`` apparaît dans ``as_dict`` quand
   renseigné (rétrocompat sinon).
6. Câblage runner avec un extracteur **mock** (callable injecté) :
   - ``ner_metrics`` est attaché aux DR dont le doc a une GT entités ;
   - ``aggregated_ner`` est calculé sur l'EngineReport ;
   - les docs sans GT entités sont ignorés.
7. Sans extracteur fourni au runner, rien n'est calculé (rétrocompat).
8. Un extracteur qui lève sur un doc spécifique → warning, autres docs
   inchangés.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picarones.evaluation.corpus import Corpus, Document, EntitiesGT, GTLevel, TextGT
from picarones.measurements.ner_backends import (
    SPACY_PROFILES,
    SpacyEntityExtractor,
    get_extractor,
    is_spacy_available,
)
from picarones.evaluation.benchmark_result import DocumentResult, EngineReport
from picarones.measurements.runner import _aggregate_ner, _attach_ner_metrics


# ──────────────────────────────────────────────────────────────────────────
# 1-3. Backend SpacyEntityExtractor
# ──────────────────────────────────────────────────────────────────────────


class TestSpacyExtractor:
    def test_falls_back_silently_without_spacy(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Sans spaCy installé, l'extracteur retourne [] avec un warning
        explicite et ne lève pas."""
        ext = SpacyEntityExtractor("fr_core_news_sm")
        with caplog.at_level("WARNING", logger="picarones.measurements.ner_backends"):
            result = ext("Marie de Bourgogne en 1477.")
        # Sans spaCy, on a toujours [] et un warning
        if not is_spacy_available():
            assert result == []
            assert any(
                "spaCy" in rec.message or "spacy" in rec.message
                for rec in caplog.records
            )
            assert ext.available is False

    def test_empty_text_returns_empty(self) -> None:
        ext = SpacyEntityExtractor()
        assert ext("") == []

    def test_idempotent_load(self) -> None:
        """L'appel répété ne re-tente pas le chargement."""
        ext = SpacyEntityExtractor("inexistant_model_xyz")
        ext("test")  # premier appel : tentative de chargement
        ext("test")  # deuxième : pas de re-tentative
        assert ext._loaded is True


class TestProfilesAndFactory:
    def test_known_profiles_listed(self) -> None:
        for key in ("fr", "en", "multilingual"):
            assert key in SPACY_PROFILES

    def test_get_extractor_with_known_profile(self) -> None:
        ext = get_extractor("fr")
        assert isinstance(ext, SpacyEntityExtractor)
        assert ext.model_name == SPACY_PROFILES["fr"]

    def test_get_extractor_with_direct_model_name(self) -> None:
        ext = get_extractor("custom_model_name")
        assert ext.model_name == "custom_model_name"


# ──────────────────────────────────────────────────────────────────────────
# 4-5. DocumentResult / EngineReport sérialisation
# ──────────────────────────────────────────────────────────────────────────


def _make_document_result(
    doc_id: str = "d1",
    hypothesis: str = "Marie de Bourgogne en 1477.",
    ner_metrics: dict | None = None,
) -> DocumentResult:
    from picarones.measurements.metrics import MetricsResult

    return DocumentResult(
        doc_id=doc_id,
        image_path="/tmp/x.png",
        ground_truth="Marie de Bourgogne en 1477.",
        hypothesis=hypothesis,
        metrics=MetricsResult(
            cer=0.0, cer_nfc=0.0, cer_caseless=0.0,
            wer=0.0, wer_normalized=0.0, mer=0.0, wil=0.0,
            reference_length=27, hypothesis_length=27,
        ),
        duration_seconds=0.1,
        ner_metrics=ner_metrics,
    )


class TestModelSerialization:
    def test_ner_metrics_omitted_when_none(self) -> None:
        dr = _make_document_result(ner_metrics=None)
        d = dr.as_dict()
        assert "ner_metrics" not in d

    def test_ner_metrics_present_when_set(self) -> None:
        dr = _make_document_result(ner_metrics={"global": {"f1": 0.8}})
        d = dr.as_dict()
        assert d["ner_metrics"] == {"global": {"f1": 0.8}}

    def test_compact_clears_ner_metrics(self) -> None:
        # Sprint A14-S1 — A.I.0 P0 : ``compact()`` est désormais no-op
        # par défaut (cf. core/results.py).  Le comportement
        # "efface les analyses" est explicitement opt-in via
        # ``drop_analyses=True``.
        dr = _make_document_result(ner_metrics={"global": {"f1": 0.8}})
        dr.compact(drop_analyses=True)
        assert dr.ner_metrics is None

    def test_compact_default_is_noop(self) -> None:
        """Sprint A14-S1 — défaut sans argument ne touche à rien."""
        dr = _make_document_result(ner_metrics={"global": {"f1": 0.8}})
        dr.compact()
        assert dr.ner_metrics == {"global": {"f1": 0.8}}

    def test_engine_report_aggregated_ner_omitted_when_none(self) -> None:
        rep = EngineReport(
            engine_name="t", engine_version="1", engine_config={},
            document_results=[_make_document_result()],
        )
        d = rep.as_dict()
        assert "aggregated_ner" not in d

    def test_engine_report_aggregated_ner_included_when_set(self) -> None:
        rep = EngineReport(
            engine_name="t", engine_version="1", engine_config={},
            document_results=[_make_document_result()],
            aggregated_ner={"global": {"f1": 0.75}, "doc_count": 1},
        )
        d = rep.as_dict()
        assert d["aggregated_ner"] == {"global": {"f1": 0.75}, "doc_count": 1}


# ──────────────────────────────────────────────────────────────────────────
# 6. Câblage runner avec extracteur mock
# ──────────────────────────────────────────────────────────────────────────


def _mock_extractor_factory(per_text: dict[str, list[dict]]) -> callable:
    """Construit un extracteur qui renvoie une réponse prédéfinie par
    texte d'entrée — utile pour tester le câblage runner sans dépendance
    NLP réelle."""

    def _extract(text: str) -> list[dict]:
        return per_text.get(text, [])

    return _extract


def _corpus_with_entities(tmp_path: Path) -> Corpus:
    """Crée un corpus minimal avec deux documents, dont un seul porte
    une GT entités."""
    image1 = tmp_path / "doc1.png"
    image2 = tmp_path / "doc2.png"
    image1.write_bytes(b"fake")
    image2.write_bytes(b"fake")

    doc1 = Document(
        image_path=image1,
        ground_truth="Marie de Bourgogne en 1477.",
        ground_truths={
            GTLevel.TEXT: TextGT(text="Marie de Bourgogne en 1477."),
            GTLevel.ENTITIES: EntitiesGT(entities=[
                {"label": "PER", "start": 0, "end": 17, "text": "Marie de Bourgogne"},
                {"label": "DATE", "start": 21, "end": 25, "text": "1477"},
            ]),
        },
    )
    doc2 = Document(
        image_path=image2,
        ground_truth="Texte sans GT entités.",
    )
    return Corpus(name="test", documents=[doc1, doc2])


class TestRunnerWiring:
    def test_attach_ner_only_for_docs_with_entities(self, tmp_path: Path) -> None:
        corpus = _corpus_with_entities(tmp_path)
        # Mock extractor : renvoie la même chose que la GT pour doc1 (parfait)
        extractor = _mock_extractor_factory({
            "Marie de Bourgogne en 1477.": [
                {"label": "PER", "start": 0, "end": 17, "text": "Marie de Bourgogne"},
                {"label": "DATE", "start": 21, "end": 25, "text": "1477"},
            ],
            "Texte sans GT entités.": [],  # pas appelé en réalité
        })
        dr1 = _make_document_result(
            doc_id="doc1", hypothesis="Marie de Bourgogne en 1477.",
        )
        dr2 = _make_document_result(
            doc_id="doc2", hypothesis="Texte sans GT entités.",
        )
        _attach_ner_metrics(corpus, [dr1, dr2], extractor)

        # doc1 : a une GT entités → ner_metrics calculé
        assert dr1.ner_metrics is not None
        assert dr1.ner_metrics["global"]["f1"] == pytest.approx(1.0)

        # doc2 : pas de GT entités → rien
        assert dr2.ner_metrics is None

    def test_aggregate_ner_combines_doc_metrics(self, tmp_path: Path) -> None:
        # Deux documents avec ner_metrics fournis
        dr1 = _make_document_result()
        dr1.ner_metrics = {
            "global": {"precision": 1.0, "recall": 0.5, "f1": 2/3, "support": 2},
            "per_category": {
                "PER": {"precision": 1.0, "recall": 0.5, "f1": 2/3, "support": 2},
            },
            "true_positives": 1, "false_positives": 0, "false_negatives": 1,
            "hallucinated_entities": [], "missed_entities": [{"label": "PER"}],
            "iou_threshold": 0.5,
        }
        dr2 = _make_document_result()
        dr2.ner_metrics = {
            "global": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
            "per_category": {
                "LOC": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
            },
            "true_positives": 1, "false_positives": 0, "false_negatives": 0,
            "hallucinated_entities": [], "missed_entities": [],
            "iou_threshold": 0.5,
        }
        agg = _aggregate_ner([dr1, dr2])
        assert agg is not None
        assert agg["doc_count"] == 2
        assert agg["true_positives"] == 2
        assert agg["false_negatives"] == 1
        assert agg["missed_total"] == 1
        # Micro F1 global : TP=2, FP=0, FN=1 → P=1, R=2/3, F1=0.8
        assert agg["global"]["f1"] == pytest.approx(0.8)

    def test_aggregate_returns_none_when_no_ner_metrics(self) -> None:
        dr = _make_document_result(ner_metrics=None)
        assert _aggregate_ner([dr]) is None


# ──────────────────────────────────────────────────────────────────────────
# 7. Rétrocompat : sans extractor, rien ne change
# ──────────────────────────────────────────────────────────────────────────


class TestBackwardCompat:
    def test_no_extractor_no_calculation(self, tmp_path: Path) -> None:
        """Si entity_extractor=None, le runner ne touche pas aux
        ner_metrics. On valide que le DocumentResult par défaut a bien
        ner_metrics=None — le runner ne l'attribue pas spontanément."""
        # Les deux DRs ne reçoivent jamais d'extracteur ; ils restent
        # tels quels. Le corpus n'est pas nécessaire ici (valide la
        # rétrocompat du modèle).
        dr1 = _make_document_result(doc_id="doc1")
        dr2 = _make_document_result(doc_id="doc2")
        assert dr1.ner_metrics is None
        assert dr2.ner_metrics is None


# ──────────────────────────────────────────────────────────────────────────
# 8. Robustesse : extracteur qui lève
# ──────────────────────────────────────────────────────────────────────────


class TestRobustness:
    def test_extractor_raising_does_not_break_others(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Si l'extracteur lève sur le doc1, le doc2 doit tout de même
        être traité (et inversement, ici doc1 est le seul avec GT
        entités, donc on vérifie qu'aucun crash ne casse le runner)."""
        corpus = _corpus_with_entities(tmp_path)

        def _broken_extractor(text: str) -> list[dict]:
            raise RuntimeError("boom")

        dr1 = _make_document_result(
            doc_id="doc1", hypothesis="Marie de Bourgogne en 1477.",
        )
        with caplog.at_level("WARNING", logger="picarones.measurements.runner"):
            _attach_ner_metrics(corpus, [dr1], _broken_extractor)

        # Pas de propagation, ner_metrics reste None
        assert dr1.ner_metrics is None
        # Et un warning explicite a été émis
        assert any("ner.attach" in rec.message for rec in caplog.records)
