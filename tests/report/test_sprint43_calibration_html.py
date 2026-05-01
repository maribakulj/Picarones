"""Tests Sprint 43 — section calibration dans le rapport HTML.

Couvre :

1. ``build_calibration_summary_html`` rend le tableau résumé (ECE/MCE
   par moteur).
2. ``build_reliability_diagram_svg`` rend un SVG avec les barres
   d'accuracy par bin, les points (avg_conf, accuracy) et la diagonale.
3. ``build_reliability_diagrams_grid_html`` génère un SVG par moteur
   ayant ``aggregated_calibration``, dans une grille.
4. **Masquage adaptatif** : les fonctions retournent ``""`` si aucun
   moteur n'a de ``aggregated_calibration`` (cas par défaut tant que
   les engines n'exposent pas leurs confidences natives).
5. **Anti-injection** : un nom de moteur avec balises HTML est
   échappé.
6. **Intégration ReportGenerator** : la section apparaît quand au
   moins un moteur a ``aggregated_calibration``, est omise sinon.
7. **i18n FR/EN** : les clés sont présentes et utilisées.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picarones.fixtures import generate_sample_benchmark
from picarones.report.calibration_render import (
    build_calibration_summary_html,
    build_reliability_diagram_svg,
    build_reliability_diagrams_grid_html,
)
from picarones.report.generator import ReportGenerator


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _make_calibration(
    ece: float = 0.05, mce: float = 0.12,
    n_predictions: int = 1000, n_bins: int = 10,
) -> dict:
    """Calibration agrégée réaliste avec quelques bins peuplés."""
    bins = []
    for k in range(n_bins):
        if k >= 5:
            bins.append({
                "bin_low": k / n_bins, "bin_high": (k + 1) / n_bins,
                "avg_confidence": (k + 0.5) / n_bins,
                "accuracy": max(0, (k - 1) / n_bins),
                "count": n_predictions // 5,
                "gap": 0.1,
            })
        else:
            bins.append({
                "bin_low": k / n_bins, "bin_high": (k + 1) / n_bins,
                "avg_confidence": None, "accuracy": None,
                "count": 0, "gap": None,
            })
    return {
        "ece": ece, "mce": mce, "n_bins": n_bins,
        "n_predictions": n_predictions,
        "overall_accuracy": 0.78, "overall_confidence": 0.82,
        "doc_count": 50, "bins": bins,
    }


def _engine_with_calibration(name: str = "tess", **kwargs) -> dict:
    return {"name": name, "aggregated_calibration": _make_calibration(**kwargs)}


def _engine_without_calibration(name: str = "no_cal") -> dict:
    return {"name": name, "aggregated_calibration": None}


# ──────────────────────────────────────────────────────────────────────────
# 1. Résumé
# ──────────────────────────────────────────────────────────────────────────


class TestSummaryTable:
    def test_renders_row_per_engine(self) -> None:
        engines = [_engine_with_calibration("a"), _engine_with_calibration("b")]
        html = build_calibration_summary_html(engines)
        assert "calibration-summary" in html
        assert "a" in html
        assert "b" in html
        # ECE et MCE rendus en pourcentage
        assert "5.00 %" in html  # ECE 0.05
        assert "12.00 %" in html  # MCE 0.12

    def test_includes_overall_accuracy_and_confidence(self) -> None:
        html = build_calibration_summary_html([_engine_with_calibration("x")])
        assert "78.0 %" in html  # accuracy
        assert "82.0 %" in html  # confidence

    def test_n_predictions_formatted_with_thousand_sep(self) -> None:
        html = build_calibration_summary_html(
            [_engine_with_calibration("x", n_predictions=12345)],
        )
        # 12 345 (espace insécable selon la convention française)
        assert "12 345" in html or "12345" in html

    def test_engine_without_calibration_omitted(self) -> None:
        engines = [_engine_with_calibration("a"), _engine_without_calibration("b")]
        html = build_calibration_summary_html(engines)
        assert "a" in html
        # Le moteur sans calibration ne doit pas avoir de ligne
        # (vérification approximative : son nom n'apparaît pas en gras)
        assert "<td" in html and ">b</td>" not in html


# ──────────────────────────────────────────────────────────────────────────
# 2. SVG reliability diagram
# ──────────────────────────────────────────────────────────────────────────


class TestReliabilityDiagramSvg:
    def test_returns_svg_with_bars_and_diagonal(self) -> None:
        svg = build_reliability_diagram_svg(_make_calibration())
        assert "<svg" in svg
        # Au moins une barre par bin non vide (5 bins peuplés)
        assert svg.count("<rect") >= 5
        # La diagonale en pointillé
        assert "stroke-dasharray" in svg
        # Au moins un point par bin non vide
        assert svg.count("<circle") >= 5

    def test_returns_empty_when_no_data(self) -> None:
        assert build_reliability_diagram_svg(None) == ""
        assert build_reliability_diagram_svg({}) == ""
        # bins tous à count = 0
        assert build_reliability_diagram_svg({
            "bins": [
                {"bin_low": 0, "bin_high": 1,
                 "avg_confidence": None, "accuracy": None,
                 "count": 0, "gap": None},
            ],
        }) == ""

    def test_axis_labels_via_i18n(self) -> None:
        labels = {
            "reliability_x_axis": "CUSTOM_X",
            "reliability_y_axis": "CUSTOM_Y",
        }
        svg = build_reliability_diagram_svg(_make_calibration(), labels=labels)
        assert "CUSTOM_X" in svg
        assert "CUSTOM_Y" in svg


# ──────────────────────────────────────────────────────────────────────────
# 3. Grille de reliability diagrams
# ──────────────────────────────────────────────────────────────────────────


class TestGrid:
    def test_one_svg_per_engine(self) -> None:
        engines = [
            _engine_with_calibration("a"), _engine_with_calibration("b"),
            _engine_with_calibration("c"),
        ]
        html = build_reliability_diagrams_grid_html(engines)
        assert html.count("<svg") == 3

    def test_engine_without_calibration_omitted_from_grid(self) -> None:
        engines = [_engine_with_calibration("a"), _engine_without_calibration("b")]
        html = build_reliability_diagrams_grid_html(engines)
        # Un seul SVG (pour "a")
        assert html.count("<svg") == 1


# ──────────────────────────────────────────────────────────────────────────
# 4. Masquage adaptatif
# ──────────────────────────────────────────────────────────────────────────


class TestAdaptiveMasking:
    def test_summary_empty_when_no_engine_has_calibration(self) -> None:
        assert build_calibration_summary_html([]) == ""
        assert build_calibration_summary_html(
            [_engine_without_calibration("a")],
        ) == ""

    def test_grid_empty_when_no_engine_has_calibration(self) -> None:
        assert build_reliability_diagrams_grid_html([]) == ""
        assert build_reliability_diagrams_grid_html(
            [_engine_without_calibration("a")],
        ) == ""


# ──────────────────────────────────────────────────────────────────────────
# 5. Anti-injection
# ──────────────────────────────────────────────────────────────────────────


class TestAntiInjection:
    def test_engine_name_escaped_in_summary(self) -> None:
        engine = _engine_with_calibration("<script>alert(1)</script>")
        html = build_calibration_summary_html([engine])
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_engine_name_escaped_in_grid(self) -> None:
        engine = _engine_with_calibration("<img src=x>")
        html = build_reliability_diagrams_grid_html([engine])
        assert "<img src=x>" not in html
        assert "&lt;img" in html


# ──────────────────────────────────────────────────────────────────────────
# 6. Intégration ReportGenerator
# ──────────────────────────────────────────────────────────────────────────


class TestReportIntegration:
    def test_section_absent_when_no_calibration(self, tmp_path: Path) -> None:
        bench = generate_sample_benchmark()
        for r in bench.engine_reports:
            assert r.aggregated_calibration is None

        out = tmp_path / "report.html"
        ReportGenerator(bench).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "calibration-summary" not in html
        assert "reliability-diagrams-grid" not in html

    def test_section_present_when_at_least_one_engine_has_calibration(
        self, tmp_path: Path,
    ) -> None:
        bench = generate_sample_benchmark()
        bench.engine_reports[0].aggregated_calibration = _make_calibration()

        out = tmp_path / "report.html"
        ReportGenerator(bench).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "calibration-summary" in html
        assert "reliability-diagrams-grid" in html
        assert "5.00 %" in html  # ECE

    def test_french_locale_uses_french_labels(self, tmp_path: Path) -> None:
        bench = generate_sample_benchmark()
        bench.engine_reports[0].aggregated_calibration = _make_calibration()

        out = tmp_path / "report_fr.html"
        ReportGenerator(bench, lang="fr").generate(out)
        html = out.read_text(encoding="utf-8")
        assert "Diagramme de fiabilité" in html
        assert "Précision moyenne" in html

    def test_english_locale_uses_english_labels(self, tmp_path: Path) -> None:
        bench = generate_sample_benchmark()
        bench.engine_reports[0].aggregated_calibration = _make_calibration()

        out = tmp_path / "report_en.html"
        ReportGenerator(bench, lang="en").generate(out)
        html = out.read_text(encoding="utf-8")
        assert "Reliability diagram" in html
        assert "Mean accuracy" in html


# ──────────────────────────────────────────────────────────────────────────
# 7. i18n FR/EN
# ──────────────────────────────────────────────────────────────────────────


REQUIRED_KEYS = (
    "h_calibration",
    "calibration_note",
    "calibration_summary_caption",
    "calibration_engine_label",
    "calibration_ece_label",
    "calibration_mce_label",
    "calibration_n_label",
    "calibration_acc_label",
    "calibration_conf_label",
    "calibration_docs_label",
    "reliability_diagram_title",
    "reliability_x_axis",
    "reliability_y_axis",
)


class TestI18NCompleteness:
    @pytest.mark.parametrize("lang", ["fr", "en"])
    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_key_present(self, lang: str, key: str) -> None:
        path = (
            Path(__file__).parent.parent.parent
            / "picarones" / "report" / "i18n" / f"{lang}.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert key in data, f"Clé {key!r} manquante dans {lang}.json"
        assert data[key].strip(), f"Clé {key!r} vide dans {lang}.json"
