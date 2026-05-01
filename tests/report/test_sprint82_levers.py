"""Tests Sprint 82 — A.I.9 : section « Leviers d'amélioration ».

Couvre :

1. Modèle ``Lever`` + registre.
2. Les 5 détecteurs : ``dominant_recoverable_class``,
   ``pareto_concentration``, ``complementarity_observation``,
   ``lexical_modernization_observation``,
   ``robustness_projection_observation``.
3. Pipeline ``detect_levers`` (ordre, robustesse aux exceptions).
4. Rendu HTML : cards, anti-injection, masquage adaptatif.
5. Anti-hallucination : chaque chiffre rendu est dans le payload.
6. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from picarones.measurements.levers import (
    Lever,
    LeverImportance,
    LeverType,
    detect_complementarity_observation,
    detect_dominant_recoverable_class,
    detect_levers,
    detect_lexical_modernization_observation,
    detect_pareto_concentration,
    detect_robustness_projection_observation,
    iter_lever_detectors,
)
from picarones.report.levers_render import build_levers_section_html


# ──────────────────────────────────────────────────────────────────────────
# 1. Modèle + registre
# ──────────────────────────────────────────────────────────────────────────


class TestModel:
    def test_lever_as_dict(self) -> None:
        lv = Lever(
            type=LeverType.DOMINANT_RECOVERABLE_CLASS,
            importance=LeverImportance.HIGH,
            payload={"engine": "t", "share_recoverable_pct": 65.0},
            engines_involved=("t",),
        )
        d = lv.as_dict()
        assert d["type"] == "dominant_recoverable_class"
        assert d["importance"] == 70
        assert d["engines_involved"] == ["t"]

    def test_registry_contains_five_detectors(self) -> None:
        types = {e.lever_type for e in iter_lever_detectors()}
        assert LeverType.DOMINANT_RECOVERABLE_CLASS in types
        assert LeverType.PARETO_CONCENTRATION in types
        assert LeverType.COMPLEMENTARITY_OBSERVATION in types
        assert LeverType.LEXICAL_MODERNIZATION_OBSERVATION in types
        assert LeverType.ROBUSTNESS_PROJECTION_OBSERVATION in types

    def test_registry_priority_sorted(self) -> None:
        priorities = [e.priority for e in iter_lever_detectors()]
        assert priorities == sorted(priorities)


# ──────────────────────────────────────────────────────────────────────────
# 2. Détecteur dominant_recoverable_class
# ──────────────────────────────────────────────────────────────────────────


class TestDominantRecoverable:
    def test_emits_when_share_above_threshold(self) -> None:
        data = {"engines": [{
            "name": "t",
            "aggregated_taxonomy": {
                "case_error": 30,
                "ligature_error": 10,
                "abbreviation_error": 25,  # 65 récupérables
                "lacuna": 20,
                "diacritic_error": 15,
            },
        }]}
        levers = detect_dominant_recoverable_class(data)
        assert len(levers) == 1
        lv = levers[0]
        assert lv.payload["engine"] == "t"
        assert lv.payload["n_recoverable"] == 65
        assert lv.payload["n_total_errors"] == 100
        assert lv.payload["share_recoverable_pct"] == 65.0
        assert lv.importance == LeverImportance.HIGH

    def test_silent_when_below_threshold(self) -> None:
        data = {"engines": [{
            "name": "t",
            "aggregated_taxonomy": {"lacuna": 80, "case_error": 20},
        }]}
        assert detect_dominant_recoverable_class(data) == []

    def test_silent_when_no_taxonomy(self) -> None:
        data = {"engines": [{"name": "t"}]}
        assert detect_dominant_recoverable_class(data) == []

    def test_top_classes_sorted_descending(self) -> None:
        data = {"engines": [{
            "name": "t",
            "aggregated_taxonomy": {
                "case_error": 50,
                "ligature_error": 5,
                "abbreviation_error": 30,
            },
        }]}
        lv = detect_dominant_recoverable_class(data)[0]
        names = [c["class"] for c in lv.payload["top_classes"]]
        assert names == ["case_error", "abbreviation_error", "ligature_error"]

    def test_accepts_counts_subdict(self) -> None:
        data = {"engines": [{
            "name": "t",
            "aggregated_taxonomy": {"counts": {"case_error": 60, "lacuna": 40}},
        }]}
        levers = detect_dominant_recoverable_class(data)
        assert len(levers) == 1
        assert levers[0].payload["n_recoverable"] == 60

    def test_medium_when_share_in_30_50(self) -> None:
        data = {"engines": [{
            "name": "t",
            "aggregated_taxonomy": {"case_error": 35, "lacuna": 65},
        }]}
        lv = detect_dominant_recoverable_class(data)[0]
        assert lv.importance == LeverImportance.MEDIUM


# ──────────────────────────────────────────────────────────────────────────
# 3. Détecteur pareto_concentration
# ──────────────────────────────────────────────────────────────────────────


class TestParetoConcentration:
    def test_concentrated_corpus(self) -> None:
        # 10 docs : 2 catastrophiques (CER 0.8), 8 OK (CER 0.05) → 80 %
        # du CER total est concentré sur 20 % des docs.
        data = {
            "ranking": [{"engine": "t", "mean_cer": 0.20}],
            "per_doc_cer": {"t": [0.8, 0.8] + [0.05] * 8},
        }
        levers = detect_pareto_concentration(data)
        assert len(levers) == 1
        p = levers[0].payload
        assert p["n_docs"] == 10
        assert p["n_docs_top"] == 2
        assert p["cer_share_pct"] >= 70

    def test_uniform_corpus_silent(self) -> None:
        data = {
            "ranking": [{"engine": "t", "mean_cer": 0.10}],
            "per_doc_cer": {"t": [0.10] * 10},
        }
        assert detect_pareto_concentration(data) == []

    def test_reads_engine_per_doc(self) -> None:
        data = {
            "ranking": [{"engine": "t", "mean_cer": 0.20}],
            "engines": [{
                "name": "t",
                "per_doc": [
                    {"cer": 0.9}, {"cer": 0.9},
                    {"cer": 0.05}, {"cer": 0.05}, {"cer": 0.05},
                    {"cer": 0.05}, {"cer": 0.05}, {"cer": 0.05},
                    {"cer": 0.05}, {"cer": 0.05},
                ],
            }],
        }
        levers = detect_pareto_concentration(data)
        assert len(levers) == 1

    def test_no_ranking_silent(self) -> None:
        assert detect_pareto_concentration({}) == []

    def test_no_per_doc_silent(self) -> None:
        data = {"ranking": [{"engine": "t", "mean_cer": 0.10}]}
        assert detect_pareto_concentration(data) == []


# ──────────────────────────────────────────────────────────────────────────
# 4. Détecteur complementarity_observation
# ──────────────────────────────────────────────────────────────────────────


class TestComplementarity:
    def test_emits_when_relative_gap_above_threshold(self) -> None:
        data = {"inter_engine_analysis": {
            "complementarity_gap": {
                "absolute_gap": 0.10,
                "relative_gap": 0.30,
                "best_engine": "t",
                "best_recall": 0.70,
                "oracle_recall": 0.80,
            },
        }}
        levers = detect_complementarity_observation(data)
        assert len(levers) == 1
        p = levers[0].payload
        assert p["best_engine"] == "t"
        assert p["absolute_gap_pct"] == 10.0
        assert p["relative_gap_pct"] == 30.0

    def test_silent_when_below_threshold(self) -> None:
        data = {"inter_engine_analysis": {
            "complementarity_gap": {"absolute_gap": 0.02, "relative_gap": 0.05},
        }}
        assert detect_complementarity_observation(data) == []

    def test_silent_when_no_data(self) -> None:
        assert detect_complementarity_observation({}) == []

    def test_high_when_relative_gap_above_50(self) -> None:
        data = {"inter_engine_analysis": {
            "complementarity_gap": {"absolute_gap": 0.30, "relative_gap": 0.60},
        }}
        lv = detect_complementarity_observation(data)[0]
        assert lv.importance == LeverImportance.HIGH


# ──────────────────────────────────────────────────────────────────────────
# 5. Détecteur lexical_modernization_observation
# ──────────────────────────────────────────────────────────────────────────


class TestLexicalModernization:
    def test_emits_top_three(self) -> None:
        data = {"engines": [{
            "name": "gpt4o",
            "lexical_modernization": {
                "n_gt_tokens": 50,
                "tokens": {
                    "maistre": {"n_total": 10, "n_modernized": 10,
                                "rate_modernized": 1.0,
                                "variants": {"maître": 10}},
                    "veoir": {"n_total": 5, "n_modernized": 5,
                              "rate_modernized": 1.0,
                              "variants": {"voir": 5}},
                    "nostre": {"n_total": 8, "n_modernized": 6,
                               "rate_modernized": 0.75,
                               "variants": {"notre": 6}},
                    "ami": {"n_total": 3, "n_modernized": 0,
                            "rate_modernized": 0.0, "variants": {}},
                },
            },
        }]}
        levers = detect_lexical_modernization_observation(data)
        assert len(levers) == 1
        top = levers[0].payload["top_tokens"]
        gt_tokens = [t["gt_token"] for t in top]
        # Tri par rate desc, puis n_total desc → maistre, veoir, nostre
        assert gt_tokens == ["maistre", "veoir", "nostre"]
        assert levers[0].importance == LeverImportance.HIGH

    def test_silent_when_no_tokens_above_min_rate(self) -> None:
        data = {"engines": [{
            "name": "t",
            "lexical_modernization": {
                "tokens": {"a": {"n_total": 10, "n_modernized": 1,
                                 "rate_modernized": 0.10, "variants": {}}},
            },
        }]}
        assert detect_lexical_modernization_observation(data) == []

    def test_silent_when_n_total_below_min(self) -> None:
        data = {"engines": [{
            "name": "t",
            "lexical_modernization": {
                "tokens": {"a": {"n_total": 1, "n_modernized": 1,
                                 "rate_modernized": 1.0, "variants": {}}},
            },
        }]}
        assert detect_lexical_modernization_observation(data) == []

    def test_silent_when_no_lexical_field(self) -> None:
        data = {"engines": [{"name": "t"}]}
        assert detect_lexical_modernization_observation(data) == []


# ──────────────────────────────────────────────────────────────────────────
# 6. Détecteur robustness_projection_observation
# ──────────────────────────────────────────────────────────────────────────


class TestRobustnessProjection:
    def test_emits_when_deficit_above_threshold(self) -> None:
        data = {"robustness_projection_aggregated": {
            "tess": {
                "total_expected_deficit": 0.06,
                "n_degradation_types": 2,
                "worst_degradation_type": "noise",
                "worst_degradation_deficit": 0.04,
            },
        }}
        levers = detect_robustness_projection_observation(data)
        assert len(levers) == 1
        p = levers[0].payload
        assert p["engine"] == "tess"
        assert p["total_expected_deficit_pct"] == 6.0
        assert p["worst_degradation_type"] == "noise"
        assert levers[0].importance == LeverImportance.HIGH

    def test_silent_when_deficit_too_low(self) -> None:
        data = {"robustness_projection_aggregated": {
            "tess": {"total_expected_deficit": 0.005},
        }}
        assert detect_robustness_projection_observation(data) == []

    def test_silent_when_no_data(self) -> None:
        assert detect_robustness_projection_observation({}) == []

    def test_sorted_by_deficit_descending(self) -> None:
        data = {"robustness_projection_aggregated": {
            "a": {"total_expected_deficit": 0.03,
                  "n_degradation_types": 1},
            "b": {"total_expected_deficit": 0.08,
                  "n_degradation_types": 2},
        }}
        levers = detect_robustness_projection_observation(data)
        assert [lv.payload["engine"] for lv in levers] == ["b", "a"]


# ──────────────────────────────────────────────────────────────────────────
# 7. Pipeline detect_levers
# ──────────────────────────────────────────────────────────────────────────


class TestDetectLevers:
    def test_aggregates_multiple_types(self) -> None:
        data = {
            "engines": [{
                "name": "t",
                "aggregated_taxonomy": {"case_error": 60, "lacuna": 40},
            }],
            "robustness_projection_aggregated": {
                "t": {"total_expected_deficit": 0.07,
                      "n_degradation_types": 2},
            },
        }
        levers = detect_levers(data)
        types = [lv.type for lv in levers]
        assert LeverType.DOMINANT_RECOVERABLE_CLASS in types
        assert LeverType.ROBUSTNESS_PROJECTION_OBSERVATION in types

    def test_sorted_by_importance_desc(self) -> None:
        # HIGH (robustness 7%) avant MEDIUM (recoverable 35%)
        data = {
            "engines": [{
                "name": "t",
                "aggregated_taxonomy": {"case_error": 35, "lacuna": 65},
            }],
            "robustness_projection_aggregated": {
                "t": {"total_expected_deficit": 0.07,
                      "n_degradation_types": 2},
            },
        }
        levers = detect_levers(data)
        importances = [int(lv.importance) for lv in levers]
        assert importances == sorted(importances, reverse=True)

    def test_empty_input_returns_empty(self) -> None:
        assert detect_levers({}) == []


# ──────────────────────────────────────────────────────────────────────────
# 8. Rendu HTML
# ──────────────────────────────────────────────────────────────────────────


def _load_labels(lang: str) -> dict:
    p = (
        Path(__file__).parent.parent.parent
        / "picarones" / "report" / "i18n" / f"{lang}.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


class TestRender:
    def test_empty_returns_empty(self) -> None:
        assert build_levers_section_html([]) == ""

    def test_card_per_lever(self) -> None:
        levers = [
            Lever(
                type=LeverType.DOMINANT_RECOVERABLE_CLASS,
                importance=LeverImportance.HIGH,
                payload={"engine": "t", "share_recoverable_pct": 65.0,
                         "n_recoverable": 65, "n_total_errors": 100,
                         "top_classes": [{"class": "case_error", "count": 50}]},
            ),
        ]
        labels = _load_labels("fr")
        html = build_levers_section_html(levers, labels)
        assert "lever-card" in html
        assert "65" in html
        assert "case_error" in html
        assert "Important" in html

    def test_anti_injection(self) -> None:
        levers = [
            Lever(
                type=LeverType.DOMINANT_RECOVERABLE_CLASS,
                importance=LeverImportance.HIGH,
                payload={"engine": "<script>alert(1)</script>",
                         "share_recoverable_pct": 60.0,
                         "n_recoverable": 60, "n_total_errors": 100,
                         "top_classes": []},
            ),
        ]
        html = build_levers_section_html(levers, _load_labels("fr"))
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_unknown_type_skipped(self) -> None:
        # Lever-like dict avec type inconnu → ignoré
        bad = {"type": "unknown_type", "importance": 70, "payload": {}}
        html = build_levers_section_html([bad], _load_labels("fr"))
        assert html == ""

    def test_accepts_dict_input(self) -> None:
        d = {
            "type": "complementarity_observation",
            "importance": 40,
            "payload": {"absolute_gap_pct": 12.0, "relative_gap_pct": 25.0,
                        "absolute_gap": 0.12, "relative_gap": 0.25},
        }
        html = build_levers_section_html([d], _load_labels("fr"))
        assert "12" in html and "25" in html

    def test_renders_in_english(self) -> None:
        levers = [
            Lever(
                type=LeverType.PARETO_CONCENTRATION,
                importance=LeverImportance.HIGH,
                payload={"engine": "t", "n_docs": 10, "n_docs_top": 2,
                         "top_share_pct": 20.0,
                         "cer_share_of_total": 0.78,
                         "cer_share_pct": 78.0},
            ),
        ]
        html = build_levers_section_html(levers, _load_labels("en"))
        assert "Improvement leverages" in html
        assert "78" in html


# ──────────────────────────────────────────────────────────────────────────
# 9. Anti-hallucination : chaque chiffre rendu provient du payload
# ──────────────────────────────────────────────────────────────────────────


def _numbers_in(s: str) -> set[str]:
    """Extrait les nombres du HTML rendu visible.

    On retire :
    - les styles inline ;
    - les entités HTML (``&#x27;`` ne contient pas le chiffre 27) ;
    - les balises elles-mêmes (``<h3>`` ne contient pas le chiffre 3).
    """
    s_clean = re.sub(r'style="[^"]*"', "", s)
    s_clean = re.sub(r"&#x?[0-9a-fA-F]+;", "", s_clean)
    s_clean = re.sub(r"<[^>]+>", " ", s_clean)
    return set(re.findall(r"\d+(?:\.\d+)?", s_clean))


def _payload_numbers(payload: dict) -> set[str]:
    out: set[str] = set()
    def _walk(v):
        if isinstance(v, (int, float)):
            out.add(str(v))
            # Aussi forme entière "65" si 65.0
            if isinstance(v, float) and v.is_integer():
                out.add(str(int(v)))
        elif isinstance(v, dict):
            for vv in v.values():
                _walk(vv)
        elif isinstance(v, list):
            for vv in v:
                _walk(vv)
    _walk(payload)
    return out


class TestAntiHallucination:
    def test_dominant_numbers_traceable_fr(self) -> None:
        lv = Lever(
            type=LeverType.DOMINANT_RECOVERABLE_CLASS,
            importance=LeverImportance.HIGH,
            payload={"engine": "tess", "share_recoverable_pct": 65.0,
                     "n_recoverable": 65, "n_total_errors": 100,
                     "top_classes": [{"class": "case_error", "count": 50}]},
        )
        html = build_levers_section_html([lv], _load_labels("fr"))
        rendered = _numbers_in(html)
        allowed = _payload_numbers(lv.payload)
        # Tout chiffre du HTML doit être dans le payload
        assert rendered.issubset(allowed), (
            f"non traçable : {rendered - allowed}"
        )

    def test_pareto_numbers_traceable_en(self) -> None:
        lv = Lever(
            type=LeverType.PARETO_CONCENTRATION,
            importance=LeverImportance.HIGH,
            payload={"engine": "tess", "n_docs": 47, "n_docs_top": 9,
                     "top_share_pct": 19.1,
                     "cer_share_of_total": 0.81,
                     "cer_share_pct": 80.7},
        )
        html = build_levers_section_html([lv], _load_labels("en"))
        rendered = _numbers_in(html)
        allowed = _payload_numbers(lv.payload)
        assert rendered.issubset(allowed), (
            f"non traçable : {rendered - allowed}"
        )

    def test_robustness_numbers_traceable_fr(self) -> None:
        lv = Lever(
            type=LeverType.ROBUSTNESS_PROJECTION_OBSERVATION,
            importance=LeverImportance.HIGH,
            payload={"engine": "tess", "total_expected_deficit": 0.058,
                     "total_expected_deficit_pct": 5.8,
                     "n_degradation_types": 3,
                     "worst_degradation_type": "noise",
                     "worst_degradation_deficit": 0.041,
                     "worst_degradation_deficit_pct": 4.1},
        )
        html = build_levers_section_html([lv], _load_labels("fr"))
        rendered = _numbers_in(html)
        allowed = _payload_numbers(lv.payload)
        assert rendered.issubset(allowed), (
            f"non traçable : {rendered - allowed}"
        )


# ──────────────────────────────────────────────────────────────────────────
# 10. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


_LEVERS_KEYS = {
    "levers_title", "levers_note",
    "levers_top_classes",
    "levers_importance_high", "levers_importance_medium",
    "levers_importance_low",
    "levers_label_dominant_recoverable_class",
    "levers_label_pareto_concentration",
    "levers_label_complementarity_observation",
    "levers_label_lexical_modernization_observation",
    "levers_label_robustness_projection_observation",
    "levers_dominant_recoverable_phrase",
    "levers_pareto_phrase",
    "levers_complementarity_phrase",
    "levers_complementarity_phrase_with_engine",
    "levers_lexical_phrase",
    "levers_robustness_phrase",
    "levers_robustness_phrase_with_worst",
}


class TestI18nCompleteness:
    def test_fr_has_all_keys(self) -> None:
        d = _load_labels("fr")
        missing = _LEVERS_KEYS - d.keys()
        assert not missing, f"manque FR : {missing}"

    def test_en_has_all_keys(self) -> None:
        d = _load_labels("en")
        missing = _LEVERS_KEYS - d.keys()
        assert not missing, f"manque EN : {missing}"
