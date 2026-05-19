"""Sprint A14-S16 — SearchView + métriques de recherchabilité."""

from __future__ import annotations

import pytest

from picarones.domain import Artifact, ArtifactType, MetricSpec
from picarones.evaluation.metrics.search import (
    levenshtein_distance,
    numerical_sequence_preservation,
    searchability_recall,
)
from picarones.evaluation.projectors import (
    AltoToText,
    CanonicalToText,
    PageToText,
    ProjectorRegistry,
)
from picarones.evaluation.registry import MetricRegistry
from picarones.evaluation.views import (
    DEFAULT_SEARCH_METRICS,
    DefaultEvaluationViewExecutor,
    build_search_view,
)


# ──────────────────────────────────────────────────────────────────
# Métriques individuelles
# ──────────────────────────────────────────────────────────────────


class TestLevenshtein:
    def test_identical(self) -> None:
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty(self) -> None:
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_single_substitution(self) -> None:
        assert levenshtein_distance("hello", "hallo") == 1

    def test_kitten_sitting(self) -> None:
        # Cas canonique : kitten → sitting (k→s, e→i, +g) = 3 ops
        assert levenshtein_distance("kitten", "sitting") == 3


class TestSearchabilityRecall:
    def test_perfect_match(self) -> None:
        recall = searchability_recall("hello world", "hello world")
        assert recall == 1.0

    def test_fuzzy_match_within_threshold(self) -> None:
        # "monde" vs "monds" → 1 substitution, ≤ 2 → match
        recall = searchability_recall("le monde", "le monds")
        assert recall == 1.0

    def test_fuzzy_match_beyond_threshold(self) -> None:
        # "monde" vs "rabbit" → distance > 2 → pas de match
        recall = searchability_recall("le monde", "le rabbit")
        # "le" matche, "monde" non → 1/2 = 0.5
        assert recall == 0.5

    def test_empty_gt_returns_zero(self) -> None:
        assert searchability_recall("", "hello") == 0.0

    def test_multiplicity_respected(self) -> None:
        # GT a "le" deux fois, hyp une seule fois → 1/2
        recall = searchability_recall("le le monde", "le monde")
        assert abs(recall - 2 / 3) < 1e-9  # "le", "monde" matchent (1 "le" non)

    def test_case_insensitive_by_default(self) -> None:
        assert searchability_recall("Bonjour", "bonjour") == 1.0

    def test_negative_max_distance_raises(self) -> None:
        with pytest.raises(ValueError, match="max_distance"):
            searchability_recall("a", "b", max_distance=-1)


class TestNumericalSequencePreservation:
    def test_perfect_year_preservation(self) -> None:
        score = numerical_sequence_preservation(
            "fait à Paris en 1789",
            "fait à Paris en 1789",
        )
        assert score == 1.0

    def test_year_corrupted(self) -> None:
        # GT contient "1789", hyp contient "1798" (pas dans hyp_years)
        # Mais "1798" est aussi une année 4 chiffres valide qui matche
        # le regex.  Vérifions la sémantique : on cherche les années
        # GT dans les années hyp.
        score = numerical_sequence_preservation(
            "année 1789",
            "année 1798",
        )
        # 1789 (GT) n'est PAS dans hyp_years = [1798] → 0/1 = 0.0
        assert score == 0.0

    def test_partial_preservation(self) -> None:
        score = numerical_sequence_preservation(
            "1789, 1799, 1815",
            "1789 et 1815",  # 1799 perdu
        )
        # 2/3 préservés
        assert abs(score - 2 / 3) < 1e-9

    def test_no_years_in_gt(self) -> None:
        score = numerical_sequence_preservation(
            "pas de date ici",
            "pas de date là",
        )
        assert score == 0.0  # convention : pas d'années GT → 0.0

    def test_year_regex_bounds(self) -> None:
        # Année 999 → trop court (3 chiffres)
        # Année 1000 → OK
        # Année 2099 → hors plage (regex 2[0-2][0-9])
        score = numerical_sequence_preservation("an 999 et 1000", "an 999 et 1000")
        # Seul "1000" est détecté en GT → comparé à hyp où "1000" présent aussi
        assert score == 1.0


# ──────────────────────────────────────────────────────────────────
# SearchView shape
# ──────────────────────────────────────────────────────────────────


class TestSearchViewShape:
    def test_default_view_accepts_5_types(self) -> None:
        view = build_search_view()
        for t in (
            ArtifactType.RAW_TEXT,
            ArtifactType.CORRECTED_TEXT,
            ArtifactType.ALTO_XML,
            ArtifactType.PAGE_XML,
            ArtifactType.CANONICAL_DOCUMENT,
        ):
            assert view.accepts(t)

    def test_default_metrics(self) -> None:
        view = build_search_view()
        assert view.metric_names == DEFAULT_SEARCH_METRICS

    def test_projection_for_alto_routes_correctly(self) -> None:
        view = build_search_view()
        spec = view.projection_for(ArtifactType.ALTO_XML)
        assert spec is not None
        assert spec.projector_name == "alto_to_text"

    def test_warnings_signal_higher_is_better_inversion(self) -> None:
        view = build_search_view()
        text = " ".join(view.warnings)
        assert "higher_is_better" in text or "OPPOSÉ" in text


# ──────────────────────────────────────────────────────────────────
# SearchView avec executor
# ──────────────────────────────────────────────────────────────────


def _build_search_executor(payloads: dict[str, str]) -> DefaultEvaluationViewExecutor:
    metrics = MetricRegistry()
    metrics.register(
        MetricSpec(
            name="searchability_recall",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
            higher_is_better=True,
        ),
        searchability_recall,
    )
    metrics.register(
        MetricSpec(
            name="numerical_sequence_preservation",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
            higher_is_better=True,
        ),
        numerical_sequence_preservation,
    )
    projectors = ProjectorRegistry()
    projectors.register(AltoToText())
    projectors.register(PageToText())
    projectors.register(CanonicalToText())

    def loader(art: Artifact) -> str:
        if art.id not in payloads:
            raise KeyError(art.id)
        return payloads[art.id]

    return DefaultEvaluationViewExecutor.from_registries(metrics, projectors, loader)


class TestSearchViewWithExecutor:
    def test_perfect_text_yields_recall_1(self) -> None:
        payloads = {
            "cand": "le petit chat noir 1789",
            "gt": "le petit chat noir 1789",
        }
        executor = _build_search_executor(payloads)
        view = build_search_view()
        cand = Artifact(id="cand", document_id="d", type=ArtifactType.RAW_TEXT)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        result = executor.evaluate(view, cand, gt, pipeline_name="test")
        assert result.metric_values["searchability_recall"] == 1.0
        assert result.metric_values["numerical_sequence_preservation"] == 1.0

    def test_partial_text_quality_with_year_loss(self) -> None:
        payloads = {
            "cand": "le pelit chat noir 1798",  # erreur typo + année corrompue
            "gt": "le petit chat noir 1789",
        }
        executor = _build_search_executor(payloads)
        view = build_search_view()
        cand = Artifact(id="cand", document_id="d", type=ArtifactType.RAW_TEXT)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        result = executor.evaluate(view, cand, gt, pipeline_name="test")
        # "petit"→"pelit" = 1 sub, OK ; "1789"→"1798" = 2 subs, OK pour
        # searchability fuzzy.  Donc searchability_recall ≈ 1.0.
        assert result.metric_values["searchability_recall"] >= 0.8
        # Mais l'année 1789 N'EST PAS dans hyp → preservation = 0.
        assert result.metric_values["numerical_sequence_preservation"] == 0.0
