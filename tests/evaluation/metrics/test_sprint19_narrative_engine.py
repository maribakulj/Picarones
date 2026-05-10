"""Tests Sprint 19 — Moteur narratif complet (détecteurs + arbitre + rendu).

Sprint 4 du plan rapport. Couvre :
  1. Les 9 détecteurs implémentés (scénarios canoniques + cas vides).
  2. L'arbitre : tri par importance, non-redondance, contradiction Nemenyi/Wilcoxon.
  3. Le renderer : chargement des templates YAML, déterminisme.
  4. Le garde-fou anti-hallucination : tout nombre rendu existe dans le JSON.
  5. L'intégration au rapport HTML (section synthèse, reproductibilité).
"""

from __future__ import annotations

import hashlib
import re

import pytest

from picarones.reports.narrative import (
    Fact,
    FactImportance,
    FactType,
    build_synthesis,
    extract_numbers,
    render_fact,
    render_synthesis,
    select_facts,
)
from picarones.reports.narrative.detectors import (
    detect_confidence_warning,
    detect_error_profile_outlier,
    detect_global_leader_cer,
    detect_llm_hallucination_flag,
    detect_robustness_fragile,
    detect_significant_gap,
    detect_speed_winner,
    detect_statistical_tie,
    detect_stratum_collapse,
    detect_stratum_winner,
)


# ---------------------------------------------------------------------------
# Fixtures — données de benchmark minimales et contrôlées
# ---------------------------------------------------------------------------

def _minimal_data(**overrides) -> dict:
    base = {
        "meta": {"document_count": 10},
        "ranking": [
            {"engine": "A", "mean_cer": 0.05, "mean_wer": 0.15, "documents": 10, "failed": 0},
            {"engine": "B", "mean_cer": 0.12, "mean_wer": 0.25, "documents": 10, "failed": 0},
            {"engine": "C", "mean_cer": 0.30, "mean_wer": 0.50, "documents": 10, "failed": 0},
        ],
        "engines": [
            {"name": "A", "cer": 0.05, "wer": 0.15, "is_pipeline": False, "is_vlm": False},
            {"name": "B", "cer": 0.12, "wer": 0.25, "is_pipeline": False, "is_vlm": False},
            {"name": "C", "cer": 0.30, "wer": 0.50, "is_pipeline": False, "is_vlm": False},
        ],
        "documents": [],
        "statistics": {
            "pairwise_wilcoxon": [],
            "bootstrap_cis": [],
            "friedman": {},
            "nemenyi": {"tied_groups": [], "mean_ranks": {}, "critical_distance": 0.0},
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Détecteurs individuels
# ---------------------------------------------------------------------------

class TestGlobalLeaderCer:
    def test_emits_fact_with_cer_pct_and_n_docs(self):
        facts = detect_global_leader_cer(_minimal_data())
        assert len(facts) == 1
        f = facts[0]
        assert f.type == FactType.GLOBAL_LEADER_CER
        assert f.importance == FactImportance.CRITICAL
        assert f.payload["engine"] == "A"
        assert f.payload["cer_pct"] == 5.0
        assert f.payload["n_docs"] == 10
        assert f.payload["runner_up"] == "B"

    def test_empty_when_no_ranking(self):
        assert detect_global_leader_cer(_minimal_data(ranking=[])) == []


class TestSignificantGap:
    def test_emits_when_leader_vs_runnerup_is_significant(self):
        data = _minimal_data(statistics={
            "pairwise_wilcoxon": [
                {"engine_a": "A", "engine_b": "B", "p_value": 0.002,
                 "significant": True, "n_pairs": 10},
            ],
            "bootstrap_cis": [], "friedman": {},
            "nemenyi": {"tied_groups": [], "mean_ranks": {}},
        })
        facts = detect_significant_gap(data)
        assert len(facts) == 1
        assert facts[0].payload["leader"] == "A"
        assert facts[0].payload["runner_up"] == "B"
        assert facts[0].payload["p_value"] == pytest.approx(0.002)

    def test_empty_when_not_significant(self):
        data = _minimal_data(statistics={
            "pairwise_wilcoxon": [
                {"engine_a": "A", "engine_b": "B", "p_value": 0.4,
                 "significant": False, "n_pairs": 10},
            ],
            "bootstrap_cis": [], "friedman": {},
            "nemenyi": {"tied_groups": [], "mean_ranks": {}},
        })
        assert detect_significant_gap(data) == []


class TestStatisticalTie:
    def test_emits_for_each_tied_group(self):
        data = _minimal_data(statistics={
            "pairwise_wilcoxon": [],
            "bootstrap_cis": [],
            "friedman": {},
            "nemenyi": {
                "tied_groups": [["A", "B"], ["C"]],
                "mean_ranks": {"A": 1.2, "B": 1.5, "C": 3.0},
                "critical_distance": 0.8,
                "alpha": 0.05,
                "n_blocks": 10,
            },
        })
        facts = detect_statistical_tie(data)
        assert len(facts) == 1
        assert set(facts[0].engines_involved) == {"A", "B"}
        assert facts[0].payload["includes_leader"] is True


class TestErrorProfileOutlier:
    def test_flags_engine_with_atypical_profile(self):
        engines = [
            {"name": "A", "aggregated_taxonomy": {"distribution": {"visual_confusion": 0.50, "abbreviation_error": 0.10}}},
            {"name": "B", "aggregated_taxonomy": {"distribution": {"visual_confusion": 0.20, "abbreviation_error": 0.10}}},
            {"name": "C", "aggregated_taxonomy": {"distribution": {"visual_confusion": 0.15, "abbreviation_error": 0.10}}},
        ]
        data = _minimal_data(engines=engines)
        facts = detect_error_profile_outlier(data)
        flagged = [f for f in facts if f.payload["engine"] == "A"]
        assert flagged
        assert flagged[0].payload["error_class"] == "visual_confusion"

    def test_empty_when_no_taxonomy(self):
        assert detect_error_profile_outlier(_minimal_data()) == []


class TestLlmHallucinationFlag:
    def test_flags_pipeline_with_high_rate(self):
        engines = [
            {"name": "tesseract", "aggregated_hallucination": {"hallucinating_doc_rate": 0.05},
             "is_pipeline": False, "is_vlm": False},
            {"name": "gpt-4o", "aggregated_hallucination": {
                "hallucinating_doc_rate": 0.45, "anchor_score_mean": 0.55, "length_ratio_mean": 1.4},
             "is_pipeline": True, "is_vlm": True},
        ]
        data = _minimal_data(engines=engines)
        facts = detect_llm_hallucination_flag(data)
        assert len(facts) == 1
        assert facts[0].payload["engine"] == "gpt-4o"
        assert facts[0].payload["hallucinating_rate_pct"] == 45.0

    def test_ignores_non_llm_engines(self):
        engines = [
            {"name": "tesseract", "aggregated_hallucination": {"hallucinating_doc_rate": 0.9},
             "is_pipeline": False, "is_vlm": False},
        ]
        data = _minimal_data(engines=engines)
        assert detect_llm_hallucination_flag(data) == []


class TestStratumDetectors:
    def _docs_with_strata(self):
        # 6 docs — 3 en "gothique", 3 en "humaniste"
        # Engine A est super bon en humaniste, moyen en gothique
        # Engine B est moyen partout
        docs = []
        for i in range(3):
            docs.append({
                "doc_id": f"goth{i}",
                "script_type": "gothique",
                "engine_results": [
                    {"engine": "A", "cer": 0.12, "error": None},
                    {"engine": "B", "cer": 0.15, "error": None},
                ],
            })
        for i in range(3):
            docs.append({
                "doc_id": f"hum{i}",
                "script_type": "humaniste",
                "engine_results": [
                    {"engine": "A", "cer": 0.02, "error": None},
                    {"engine": "B", "cer": 0.10, "error": None},
                ],
            })
        return docs

    def test_stratum_winner_detected(self):
        docs = self._docs_with_strata()
        engines = [{"name": "A", "cer": 0.07}, {"name": "B", "cer": 0.12}]
        data = _minimal_data(documents=docs, engines=engines)
        facts = detect_stratum_winner(data)
        humanist = [f for f in facts if f.stratum == "humaniste"]
        assert humanist
        assert humanist[0].payload["engine"] == "A"

    def test_stratum_collapse_detected(self):
        # Engine A globalement bon (0.05) mais s'effondre sur "cursive" (0.30)
        docs = []
        for i in range(5):
            docs.append({
                "doc_id": f"good{i}",
                "script_type": "textualis",
                "engine_results": [{"engine": "A", "cer": 0.04, "error": None}],
            })
        for i in range(3):
            docs.append({
                "doc_id": f"bad{i}",
                "script_type": "cursive",
                "engine_results": [{"engine": "A", "cer": 0.30, "error": None}],
            })
        engines = [{"name": "A", "cer": 0.10}]
        data = _minimal_data(documents=docs, engines=engines)
        facts = detect_stratum_collapse(data)
        assert any(f.stratum == "cursive" for f in facts)


class TestSpeedWinner:
    def test_detects_fast_engine_with_comparable_quality(self):
        # "fast" est 50× plus rapide ET n'est qu'à 6 % de CER en plus du leader
        # (dans la marge de tolérance de qualité du détecteur).
        docs = []
        for i in range(5):
            docs.append({
                "doc_id": f"d{i}",
                "engine_results": [
                    {"engine": "fast", "cer": 0.053, "error": None, "duration": 0.1},
                    {"engine": "slow", "cer": 0.050, "error": None, "duration": 5.0},
                ],
            })
        engines = [{"name": "fast", "cer": 0.053}, {"name": "slow", "cer": 0.050}]
        ranking = [
            {"engine": "slow", "mean_cer": 0.050, "documents": 5, "failed": 0},
            {"engine": "fast", "mean_cer": 0.053, "documents": 5, "failed": 0},
        ]
        data = _minimal_data(documents=docs, engines=engines, ranking=ranking)
        facts = detect_speed_winner(data)
        assert facts, "speed_winner devrait détecter un moteur 50× plus rapide"
        assert facts[0].payload["engine"] == "fast"
        assert facts[0].payload["speedup"] >= 3.0

    def test_ignores_fast_engine_with_bad_quality(self):
        # "fast" est rapide mais a un CER 3× celui du leader — pas un speed winner
        docs = [{
            "doc_id": f"d{i}",
            "engine_results": [
                {"engine": "fast", "cer": 0.15, "error": None, "duration": 0.1},
                {"engine": "slow", "cer": 0.05, "error": None, "duration": 5.0},
            ],
        } for i in range(5)]
        engines = [{"name": "fast", "cer": 0.15}, {"name": "slow", "cer": 0.05}]
        ranking = [
            {"engine": "slow", "mean_cer": 0.05, "documents": 5, "failed": 0},
            {"engine": "fast", "mean_cer": 0.15, "documents": 5, "failed": 0},
        ]
        data = _minimal_data(documents=docs, engines=engines, ranking=ranking)
        assert detect_speed_winner(data) == []


class TestConfidenceWarning:
    def test_wide_ci_triggers_warning(self):
        cis = [
            {"engine": "A", "mean": 0.05, "ci_lower": 0.01, "ci_upper": 0.25},
            {"engine": "B", "mean": 0.12, "ci_lower": 0.08, "ci_upper": 0.16},
        ]
        data = _minimal_data(statistics={
            "pairwise_wilcoxon": [], "bootstrap_cis": cis,
            "friedman": {}, "nemenyi": {"tied_groups": [], "mean_ranks": {}},
        })
        facts = detect_confidence_warning(data)
        assert len(facts) == 1
        assert facts[0].payload["engine"] == "A"


class TestRobustnessFragile:
    def test_detects_collapse_under_high_degradation(self):
        data = _minimal_data(robustness={
            "curves": [
                {"engine": "X", "degradation_type": "noise", "points": [
                    {"level": 0, "cer": 0.05},
                    {"level": 80, "cer": 0.40},
                ]},
                {"engine": "Y", "degradation_type": "noise", "points": [
                    {"level": 0, "cer": 0.05},
                    {"level": 80, "cer": 0.08},
                ]},
            ],
        })
        facts = detect_robustness_fragile(data)
        names = {f.payload["engine"] for f in facts}
        assert "X" in names
        assert "Y" not in names


# ---------------------------------------------------------------------------
# Arbitre
# ---------------------------------------------------------------------------

class TestArbiter:
    def _fact(self, t, imp=FactImportance.HIGH, engines=("A",), stratum=None, payload=None):
        return Fact(type=t, importance=imp, payload=payload or {},
                    engines_involved=tuple(engines), stratum=stratum)

    def test_sort_by_importance_descending(self):
        f1 = self._fact(FactType.SPEED_WINNER, imp=FactImportance.MEDIUM)
        f2 = self._fact(FactType.GLOBAL_LEADER_CER, imp=FactImportance.CRITICAL, engines=("B",))
        selected = select_facts([f1, f2])
        assert selected[0].type == FactType.GLOBAL_LEADER_CER

    def test_max_facts_limit(self):
        facts = [self._fact(FactType.ERROR_PROFILE_OUTLIER, engines=(f"E{i}",)) for i in range(10)]
        selected = select_facts(facts, max_facts=3)
        assert len(selected) == 3

    def test_deduplicates_same_engine_same_type(self):
        f1 = self._fact(FactType.ERROR_PROFILE_OUTLIER, engines=("A",), payload={"x": 1})
        f2 = self._fact(FactType.ERROR_PROFILE_OUTLIER, engines=("A",), payload={"x": 2})
        selected = select_facts([f1, f2])
        assert len(selected) == 1

    def test_keeps_complementary_facts_for_same_engine(self):
        leader = self._fact(FactType.GLOBAL_LEADER_CER, imp=FactImportance.CRITICAL, engines=("A",))
        gap = self._fact(FactType.SIGNIFICANT_GAP, imp=FactImportance.CRITICAL, engines=("A", "B"))
        selected = select_facts([leader, gap])
        # Les deux doivent survivre (paire complémentaire)
        types = {f.type for f in selected}
        assert FactType.GLOBAL_LEADER_CER in types
        assert FactType.SIGNIFICANT_GAP in types

    def test_low_importance_filtered(self):
        low = Fact(type=FactType.SPEED_WINNER, importance=FactImportance.LOW,
                   payload={}, engines_involved=("A",))
        high = self._fact(FactType.GLOBAL_LEADER_CER, imp=FactImportance.CRITICAL, engines=("A",))
        selected = select_facts([low, high])
        assert all(f.importance >= FactImportance.MEDIUM for f in selected)

    def test_nemenyi_tie_suppresses_contradicting_wilcoxon_gap(self):
        # Si A et B sont dans le même groupe Nemenyi, on ne doit pas afficher
        # un SIGNIFICANT_GAP entre A et B en plus.
        tie = self._fact(FactType.STATISTICAL_TIE, imp=FactImportance.CRITICAL,
                         engines=("A", "B", "C"))
        gap = self._fact(FactType.SIGNIFICANT_GAP, imp=FactImportance.CRITICAL,
                         engines=("A", "B"))
        selected = select_facts([tie, gap])
        types = {f.type for f in selected}
        assert FactType.STATISTICAL_TIE in types
        assert FactType.SIGNIFICANT_GAP not in types


# ---------------------------------------------------------------------------
# Rendu et déterminisme
# ---------------------------------------------------------------------------

class TestRenderer:
    def test_render_fact_with_known_template(self):
        f = Fact(
            type=FactType.GLOBAL_LEADER_CER,
            importance=FactImportance.CRITICAL,
            payload={"engine": "testseract", "cer_pct": 4.2, "n_docs": 50,
                     "cer": 0.042, "n_engines": 3},
            engines_involved=("testseract",),
        )
        text = render_fact(f, "fr")
        assert "testseract" in text
        assert "4.2" in text
        assert "50" in text

    def test_render_respects_language(self):
        f = Fact(
            type=FactType.GLOBAL_LEADER_CER,
            importance=FactImportance.CRITICAL,
            payload={"engine": "X", "cer_pct": 1.0, "n_docs": 10,
                     "cer": 0.01, "n_engines": 2},
        )
        fr = render_fact(f, "fr")
        en = render_fact(f, "en")
        assert fr != en
        assert "Sur ce corpus" in fr
        assert "On this corpus" in en

    def test_render_missing_key_does_not_crash(self):
        # Payload incomplet volontairement
        f = Fact(
            type=FactType.GLOBAL_LEADER_CER,
            importance=FactImportance.CRITICAL,
            payload={"engine": "only_name"},
        )
        text = render_fact(f)
        # Doit renvoyer une phrase non vide, même si certains placeholders sont manquants
        assert "only_name" in text

    def test_render_synthesis_deterministic(self):
        facts = [
            Fact(type=FactType.GLOBAL_LEADER_CER, importance=FactImportance.CRITICAL,
                 payload={"engine": "A", "cer_pct": 3.1, "n_docs": 20,
                          "cer": 0.031, "n_engines": 2},
                 engines_involved=("A",)),
        ]
        s1 = render_synthesis(facts, "fr")
        s2 = render_synthesis(facts, "fr")
        assert s1 == s2


class TestBuildSynthesisE2E:
    def test_full_pipeline_produces_sentences(self):
        data = _minimal_data(statistics={
            "pairwise_wilcoxon": [
                {"engine_a": "A", "engine_b": "B", "p_value": 0.01,
                 "significant": True, "n_pairs": 10},
            ],
            "bootstrap_cis": [
                {"engine": "A", "mean": 0.05, "ci_lower": 0.04, "ci_upper": 0.06},
                {"engine": "B", "mean": 0.12, "ci_lower": 0.11, "ci_upper": 0.13},
            ],
            "friedman": {},
            "nemenyi": {"tied_groups": [["A"], ["B"], ["C"]],
                        "mean_ranks": {"A": 1.0, "B": 2.0, "C": 3.0},
                        "critical_distance": 0.5},
        })
        result = build_synthesis(data, "fr")
        assert "sentences" in result
        assert "facts" in result
        assert len(result["sentences"]) >= 1
        # Au moins la mention du leader
        assert any("A" in s for s in result["sentences"])

    def test_pipeline_deterministic_across_calls(self):
        data = _minimal_data()
        s1 = build_synthesis(data, "fr")
        s2 = build_synthesis(data, "fr")
        assert s1 == s2


# ---------------------------------------------------------------------------
# Garde-fou anti-hallucination : traçabilité des nombres
# ---------------------------------------------------------------------------

# ``_numbers_in_payload`` vit dans ``tests/measurements/_helpers.py`` ;
# on le ré-expose sous son ancien nom privé pour compatibilité avec les
# tests qui l'importent depuis ce module (ex. test_sprint23).
from tests.evaluation.metrics._helpers import numbers_in_payload as _numbers_in_payload  # noqa: E402


# Sprint 23 : whitelist vidée. Tout nombre rendu dans la synthèse doit
# venir du payload d'un Fact. Le seuil de confiance (95) est désormais
# propagé via ``confidence_level`` dans le payload de
# ``FactType.CONFIDENCE_WARNING`` et l'unité du coût (1000 pages) via
# ``cost_unit_pages`` dans ``PARETO_ALTERNATIVE`` / ``COST_OUTLIER``.
# Aucun littéral hors-payload n'est plus autorisé.
_TEMPLATE_CONSTANTS: frozenset[str] = frozenset()


class TestAntiHallucinationTraceability:
    """Chaque nombre dans la synthèse doit venir du payload d'un Fact
    (lui-même traçable au JSON d'entrée par construction des détecteurs)
    ou appartenir à la liste limitative des constantes de template.
    """

    def test_every_number_in_synthesis_is_traceable(self):
        data = _minimal_data(statistics={
            "pairwise_wilcoxon": [
                {"engine_a": "A", "engine_b": "B", "p_value": 0.0123,
                 "significant": True, "n_pairs": 10},
            ],
            "bootstrap_cis": [
                {"engine": "A", "mean": 0.05, "ci_lower": 0.01, "ci_upper": 0.25},
                {"engine": "B", "mean": 0.12, "ci_lower": 0.11, "ci_upper": 0.13},
            ],
            "friedman": {"statistic": 5.2, "p_value": 0.07, "significant": False},
            "nemenyi": {
                "tied_groups": [["A", "B"]],
                "mean_ranks": {"A": 1.3, "B": 1.7, "C": 3.0},
                "critical_distance": 0.856,
                "alpha": 0.05,
                "n_blocks": 10,
            },
        })
        result = build_synthesis(data, "fr")
        # Concaténer tous les payloads des Facts retenus
        allowed = set(_TEMPLATE_CONSTANTS)
        for f in result["facts"]:
            allowed |= _numbers_in_payload(f.get("payload", {}))

        unknown = []
        for sentence in result["sentences"]:
            for num in extract_numbers(sentence):
                num_norm = num.replace(",", ".")
                if num_norm not in allowed:
                    unknown.append((num, sentence))
        assert not unknown, f"Nombres non traçables : {unknown}"


# ---------------------------------------------------------------------------
# Intégration au rapport HTML
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def benchmark_result():
    from picarones.evaluation import synthetic as fixtures
    return fixtures.generate_sample_benchmark(n_docs=8)


class TestReportIntegration:
    def test_report_contains_synthesis_section(self, benchmark_result, tmp_path):
        from picarones.reports.html.generator import ReportGenerator
        out = tmp_path / "report.html"
        ReportGenerator(benchmark_result).generate(out)
        html = out.read_text(encoding="utf-8")
        assert 'class="synth-card"' in html
        assert 'id="synth-title"' in html
        # Au moins une phrase rendue
        assert re.search(r'<ul class="synth-list">\s*<li>', html)

    def test_report_synthesis_is_deterministic(self, benchmark_result, tmp_path):
        from picarones.reports.html.generator import ReportGenerator
        out1 = tmp_path / "r1.html"
        out2 = tmp_path / "r2.html"
        ReportGenerator(benchmark_result).generate(out1)
        ReportGenerator(benchmark_result).generate(out2)
        # Extraire la section synth et comparer
        h1 = out1.read_text(encoding="utf-8")
        h2 = out2.read_text(encoding="utf-8")
        s1 = re.search(r'<section class="synth-card".*?</section>', h1, re.DOTALL)
        s2 = re.search(r'<section class="synth-card".*?</section>', h2, re.DOTALL)
        assert s1 and s2
        assert hashlib.sha256(s1.group().encode()).hexdigest() == \
               hashlib.sha256(s2.group().encode()).hexdigest()

    def test_default_registry_has_all_types_registered(self):
        from picarones.reports.narrative import _DEFAULT_REGISTRY
        from picarones.domain.facts import FactType

        registered = set(_DEFAULT_REGISTRY.registered_types())
        # Tous les types de FactType doivent avoir un détecteur enregistré.
        # Sprint 36 : ENSEMBLE_OPPORTUNITY a porté le total à 13.
        assert registered == set(FactType)

    def test_english_locale_produces_english_sentences(self, benchmark_result, tmp_path):
        from picarones.reports.html.generator import ReportGenerator
        out = tmp_path / "report_en.html"
        ReportGenerator(benchmark_result, lang="en").generate(out)
        html = out.read_text(encoding="utf-8")
        m = re.search(r'<ul class="synth-list">(.*?)</ul>', html, re.DOTALL)
        assert m
        ul_content = m.group(1)
        # Soit "On this corpus" (leader) soit "Engines" (tie) soit "The gap"
        assert any(marker in ul_content for marker in
                   ("On this corpus", "Engines ", "The gap", "statistically"))
