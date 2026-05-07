"""Tests Sprint 77 — A.I.4 chantier 3 : taxonomie comparative.

Couvre :

1. ``compare_taxonomies`` :
   - Proportions correctement normalisées (somme = 1)
   - Deltas signés (b - a)
   - Catégorisation par récupérabilité
   - Cas dégénéré : deux comptes vides → None
   - Classes apparaissant chez un seul moteur
   - Totaux par récupérabilité
2. Rendu HTML :
   - Diagramme miroir SVG bien formé
   - Tableau récupérabilité présent
   - "" si data None
   - "" si classes vides
3. Anti-injection : noms moteurs avec ``<script>``.
4. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.measurements.taxonomy_comparison import (
    RECOVERABILITY,
    compare_taxonomies,
)
from picarones.reports_v2.html.renderers.taxonomy_comparison import (
    build_taxonomy_comparison_html,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. compare_taxonomies
# ──────────────────────────────────────────────────────────────────────────


class TestCompare:
    def test_proportions_sum_to_one(self) -> None:
        result = compare_taxonomies(
            "A", {"case_error": 8, "lacuna": 2},
            "B", {"case_error": 1, "lacuna": 9},
        )
        assert result is not None
        assert sum(result["proportions_a"].values()) == 1.0
        assert sum(result["proportions_b"].values()) == 1.0

    def test_deltas_signed(self) -> None:
        result = compare_taxonomies(
            "A", {"case_error": 8, "lacuna": 2},
            "B", {"case_error": 2, "lacuna": 8},
        )
        # B a plus de lacuna, moins de case_error
        assert result["deltas"]["lacuna"] > 0
        assert result["deltas"]["case_error"] < 0

    def test_recoverability_categorization(self) -> None:
        result = compare_taxonomies(
            "A", {"case_error": 10},   # 100% recoverable
            "B", {"lacuna": 10},       # 100% irrecoverable
        )
        totals = result["totals_by_recoverability"]
        assert totals["recoverable"]["a"] == 1.0
        assert totals["irrecoverable"]["b"] == 1.0
        assert totals["recoverable"]["b"] == 0.0
        assert totals["irrecoverable"]["a"] == 0.0

    def test_returns_none_when_both_empty(self) -> None:
        assert compare_taxonomies("A", {}, "B", {}) is None
        assert compare_taxonomies("A", {"case_error": 0}, "B", {}) is None

    def test_class_in_only_one_engine(self) -> None:
        result = compare_taxonomies(
            "A", {"case_error": 5},
            "B", {"lacuna": 5, "case_error": 5},
        )
        # case_error présent chez les deux
        assert result["proportions_a"]["case_error"] == 1.0
        assert result["proportions_a"]["lacuna"] == 0.0
        assert result["proportions_b"]["lacuna"] == 0.5

    def test_totals_a_and_b_correct(self) -> None:
        result = compare_taxonomies(
            "A", {"case_error": 7, "lacuna": 3},
            "B", {"case_error": 2, "lacuna": 8},
        )
        assert result["total_a"] == 10
        assert result["total_b"] == 10

    def test_recoverability_constant_complete(self) -> None:
        # Sanité : RECOVERABILITY couvre toutes les classes du module
        from picarones.measurements.taxonomy import ERROR_CLASSES
        for cls in ERROR_CLASSES:
            assert cls in RECOVERABILITY


# ──────────────────────────────────────────────────────────────────────────
# 2. Rendu HTML
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def test_returns_empty_when_none(self) -> None:
        assert build_taxonomy_comparison_html(None) == ""

    def test_renders_svg(self) -> None:
        data = compare_taxonomies(
            "Tesseract", {"case_error": 8, "lacuna": 2},
            "Pero", {"case_error": 2, "lacuna": 8},
        )
        html = build_taxonomy_comparison_html(data)
        assert "<svg" in html
        assert "</svg>" in html

    def test_engine_names_displayed(self) -> None:
        data = compare_taxonomies(
            "Tesseract", {"case_error": 5},
            "Pero", {"lacuna": 5},
        )
        html = build_taxonomy_comparison_html(data)
        assert "Tesseract" in html
        assert "Pero" in html

    def test_class_labels_present(self) -> None:
        data = compare_taxonomies(
            "A", {"case_error": 5},
            "B", {"lacuna": 5},
        )
        html = build_taxonomy_comparison_html(data)
        assert "case_error" in html
        assert "lacuna" in html

    def test_recoverability_summary_present(self) -> None:
        data = compare_taxonomies(
            "A", {"case_error": 5},
            "B", {"lacuna": 5},
        )
        html = build_taxonomy_comparison_html(data)
        assert "Récupérable" in html
        assert "Irrécupérable" in html

    def test_proportions_displayed(self) -> None:
        data = compare_taxonomies(
            "A", {"case_error": 8, "lacuna": 2},
            "B", {"case_error": 2, "lacuna": 8},
        )
        html = build_taxonomy_comparison_html(data)
        # 80.0% présent dans le SVG (proportion case_error de A)
        assert "80.0%" in html

    def test_color_codes_present(self) -> None:
        data = compare_taxonomies(
            "A", {"case_error": 5},  # recoverable → vert
            "B", {"lacuna": 5},      # irrecoverable → rouge
        )
        html = build_taxonomy_comparison_html(data)
        assert "#5fa860" in html  # vert
        assert "#d8553b" in html  # rouge


# ──────────────────────────────────────────────────────────────────────────
# 3. Anti-injection
# ──────────────────────────────────────────────────────────────────────────


class TestAntiInjection:
    def test_engine_name_escaped(self) -> None:
        data = compare_taxonomies(
            "<script>alert(1)</script>", {"case_error": 5},
            "Pero", {"lacuna": 5},
        )
        html = build_taxonomy_comparison_html(data)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_label_via_i18n_escaped(self) -> None:
        data = compare_taxonomies(
            "A", {"case_error": 5}, "B", {"lacuna": 5},
        )
        labels = {"taxocomp_recoverable": "<b>Hack</b>"}
        html = build_taxonomy_comparison_html(data, labels=labels)
        assert "<b>Hack</b>" not in html
        assert "&lt;b&gt;Hack&lt;/b&gt;" in html


# ──────────────────────────────────────────────────────────────────────────
# 4. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


class TestI18nCompleteness:
    def _load(self, lang: str) -> dict:
        path = (
            Path(__file__).parent.parent.parent
            / "picarones" / "reports_v2" / "i18n" / f"{lang}.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_all_keys_fr(self) -> None:
        d = self._load("fr")
        for key in (
            "taxocomp_title", "taxocomp_note", "taxocomp_level_label",
            "taxocomp_recoverable", "taxocomp_difficult",
            "taxocomp_irrecoverable",
        ):
            assert key in d, f"manque clé FR : {key}"

    def test_all_keys_en(self) -> None:
        d_fr = self._load("fr")
        d_en = self._load("en")
        for key in d_fr:
            if key.startswith("taxocomp_"):
                assert key in d_en, f"manque clé EN : {key}"
