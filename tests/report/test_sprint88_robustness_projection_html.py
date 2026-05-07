"""Tests Sprint 88 — A.I.8 vue HTML : déficit projeté de robustesse.

Couvre :

1. ``build_robustness_projection_html`` :
   - vide / None → ``""``
   - rendu complet (résumé + détail)
   - calcul automatique de ``aggregated`` si non fourni
   - tri par déficit décroissant
   - colonne « pire dégradation » formatée
   - cellules colorées selon l'amplitude du déficit
2. Anti-injection sur nom de moteur + type de dégradation.
3. Bout-en-bout : intégration avec
   ``project_robustness_on_corpus`` + ``aggregate_projection_per_engine``.
4. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.evaluation.metrics.robustness_projection import (
    aggregate_projection_per_engine,
    project_robustness_on_corpus,
)
from picarones.reports_v2.html.renderers.robustness_projection import (
    build_robustness_projection_html,
)


def _load_labels(lang: str) -> dict:
    p = (
        Path(__file__).parent.parent.parent
        / "picarones" / "reports_v2" / "i18n" / f"{lang}.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


def _curve(engine: str, deg: str) -> dict:
    return {
        "engine_name": engine,
        "degradation_type": deg,
        "levels": [0, 5, 10, 20],
        "cer_values": [0.05, 0.10, 0.20, 0.50],
        "critical_threshold_level": 10,
        "cer_threshold": 0.20,
    }


# ──────────────────────────────────────────────────────────────────────────
# 1. build_robustness_projection_html
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def test_none_returns_empty(self) -> None:
        assert build_robustness_projection_html(None) == ""

    def test_empty_returns_empty(self) -> None:
        assert build_robustness_projection_html({}) == ""

    def test_renders_summary_and_detail(self) -> None:
        projection = {
            "tess": {
                "noise": {
                    "n_docs": 50, "n_docs_with_data": 48,
                    "expected_cer_mean": 0.18, "baseline_cer": 0.05,
                    "deficit_vs_baseline": 0.13,
                    "n_docs_above_critical": 12,
                    "critical_threshold_cer": 0.20,
                },
            },
        }
        labels = _load_labels("fr")
        html = build_robustness_projection_html(projection, labels=labels)
        assert "<table" in html
        assert "tess" in html
        assert "noise" in html
        # Déficit total = 0.13 → 13.00 pts
        assert "+13.00" in html
        # Le summary contient le worst type
        assert "Pire dégradation" in html
        assert "Détail" in html

    def test_auto_computes_aggregate(self) -> None:
        # Ne fournit que projection → aggregated calculé depuis
        projection = {
            "tess": {
                "noise": {
                    "n_docs": 10, "n_docs_with_data": 10,
                    "deficit_vs_baseline": 0.05,
                    "n_docs_above_critical": 0,
                },
            },
        }
        html = build_robustness_projection_html(
            projection, labels=_load_labels("fr"),
        )
        # Total = 0.05 = 5.00 points
        assert "+5.00" in html

    def test_sorted_by_deficit_descending(self) -> None:
        projection = {
            "low": {
                "noise": {
                    "n_docs": 1, "n_docs_with_data": 1,
                    "deficit_vs_baseline": 0.01,
                    "n_docs_above_critical": 0,
                },
            },
            "high": {
                "noise": {
                    "n_docs": 1, "n_docs_with_data": 1,
                    "deficit_vs_baseline": 0.10,
                    "n_docs_above_critical": 1,
                },
            },
        }
        html = build_robustness_projection_html(
            projection, labels=_load_labels("fr"),
        )
        # « high » apparaît avant « low » dans le résumé
        assert html.index("high") < html.index("low")

    def test_anti_injection_engine(self) -> None:
        projection = {
            "<script>alert(1)</script>": {
                "noise": {
                    "n_docs": 1, "n_docs_with_data": 1,
                    "deficit_vs_baseline": 0.05,
                    "n_docs_above_critical": 0,
                },
            },
        }
        html = build_robustness_projection_html(
            projection, labels=_load_labels("fr"),
        )
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_anti_injection_deg_type(self) -> None:
        projection = {
            "tess": {
                "<img/>": {
                    "n_docs": 1, "n_docs_with_data": 1,
                    "deficit_vs_baseline": 0.05,
                    "n_docs_above_critical": 0,
                },
            },
        }
        html = build_robustness_projection_html(
            projection, labels=_load_labels("fr"),
        )
        assert "<img/>" not in html
        assert "&lt;img" in html

    def test_handles_missing_deficit(self) -> None:
        projection = {
            "tess": {
                "noise": {
                    "n_docs": 5, "n_docs_with_data": 5,
                    "deficit_vs_baseline": None,
                    "n_docs_above_critical": 0,
                },
            },
        }
        html = build_robustness_projection_html(
            projection, labels=_load_labels("fr"),
        )
        assert "—" in html  # Cellule déficit vide

    def test_renders_in_english(self) -> None:
        projection = {
            "tess": {
                "noise": {
                    "n_docs": 1, "n_docs_with_data": 1,
                    "deficit_vs_baseline": 0.05,
                    "n_docs_above_critical": 0,
                },
            },
        }
        html = build_robustness_projection_html(
            projection, labels=_load_labels("en"),
        )
        assert "Projected robustness deficit" in html


# ──────────────────────────────────────────────────────────────────────────
# 2. Bout-en-bout (Sprint 81 + Sprint 88)
# ──────────────────────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_full_pipeline_renders(self) -> None:
        curves = [_curve("tess", "noise"), _curve("pero", "noise")]
        qualities = [
            {"noise_level": 7.5}, {"noise_level": 5}, {"noise_level": 15},
        ]
        projection = project_robustness_on_corpus(curves, qualities)
        aggregated = aggregate_projection_per_engine(projection)
        html = build_robustness_projection_html(
            projection, aggregated, _load_labels("fr"),
        )
        assert "<table" in html
        # Les deux moteurs apparaissent
        assert "tess" in html
        assert "pero" in html


# ──────────────────────────────────────────────────────────────────────────
# 3. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


_KEYS = {
    "robproj_title", "robproj_note", "robproj_summary", "robproj_detail",
    "robproj_engine", "robproj_total", "robproj_n_types", "robproj_worst",
    "robproj_deg_type", "robproj_n_docs", "robproj_n_with_data",
    "robproj_deficit", "robproj_above",
}


class TestI18n:
    def test_fr(self) -> None:
        d = _load_labels("fr")
        assert not _KEYS - d.keys()

    def test_en(self) -> None:
        d = _load_labels("en")
        assert not _KEYS - d.keys()
