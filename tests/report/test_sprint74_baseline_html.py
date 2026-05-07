"""Tests Sprint 74 — A.I.3 chantier 1 : encart « Ce corpus est-il habituel ? ».

Couvre :

1. ``build_corpus_difficulty_baseline_html`` :
   - Phrase factuelle rendue (harder / easier / usual)
   - Chaîne vide si ``percentile_data is None``
   - SVG omis si ``historical_values`` vide / None
   - SVG rendu si valeurs fournies
2. SVG :
   - Bien formé (``<svg ...>...</svg>``)
   - Point courant placé au bon endroit (couleur selon position)
   - Boîte Q1-Q3, médiane, moustaches min-max
3. Anti-injection : labels i18n contenant ``<script>`` échappés.
4. Complétude i18n : nouvelles clés ``baseline_corpus_*`` présentes
   en FR et EN.
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.reports_v2.html.renderers.baseline import (
    _build_difficulty_boxplot_svg,
    _quantiles,
    build_corpus_difficulty_baseline_html,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. _quantiles
# ──────────────────────────────────────────────────────────────────────────


class TestQuantiles:
    def test_simple(self) -> None:
        v = [1.0, 2.0, 3.0, 4.0, 5.0]
        mn, q1, med, q3, mx = _quantiles(v)
        assert mn == 1.0
        assert mx == 5.0
        assert med == 3.0

    def test_empty(self) -> None:
        assert _quantiles([]) == (0.0, 0.0, 0.0, 0.0, 0.0)

    def test_single(self) -> None:
        assert _quantiles([0.5]) == (0.5, 0.5, 0.5, 0.5, 0.5)


# ──────────────────────────────────────────────────────────────────────────
# 2. SVG boxplot
# ──────────────────────────────────────────────────────────────────────────


class TestSvg:
    def test_well_formed(self) -> None:
        svg = _build_difficulty_boxplot_svg(
            [0.1, 0.2, 0.3, 0.4, 0.5], current=0.35,
        )
        assert svg.startswith('<svg')
        assert svg.endswith('</svg>')
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg

    def test_empty_returns_empty(self) -> None:
        assert _build_difficulty_boxplot_svg([], current=0.5) == ""

    def test_point_color_harder(self) -> None:
        # current > Q3 → rouge
        svg = _build_difficulty_boxplot_svg(
            [0.1, 0.2, 0.3, 0.4, 0.5], current=0.95,
        )
        assert "#d8553b" in svg

    def test_point_color_easier(self) -> None:
        # current < Q1 → bleu
        svg = _build_difficulty_boxplot_svg(
            [0.3, 0.4, 0.5, 0.6, 0.7], current=0.1,
        )
        assert "#3b87d8" in svg

    def test_point_color_usual(self) -> None:
        # current entre Q1 et Q3 → vert
        svg = _build_difficulty_boxplot_svg(
            [0.1, 0.2, 0.3, 0.4, 0.5], current=0.3,
        )
        assert "#5fa860" in svg

    def test_contains_box_and_whiskers(self) -> None:
        svg = _build_difficulty_boxplot_svg(
            [0.1, 0.2, 0.3, 0.4, 0.5], current=0.3,
        )
        # Au moins un rect (boîte) et plusieurs lignes (moustaches)
        assert "<rect" in svg
        assert "<line" in svg
        # Cercle pour le point courant
        assert "<circle" in svg

    def test_degenerate_all_same(self) -> None:
        # Toutes les valeurs identiques : ne doit pas crasher
        svg = _build_difficulty_boxplot_svg(
            [0.5, 0.5, 0.5], current=0.5,
        )
        assert svg.startswith('<svg')

    def test_current_outside_historical_range(self) -> None:
        # Le point courant peut dépasser les valeurs historiques
        svg = _build_difficulty_boxplot_svg(
            [0.1, 0.2, 0.3], current=0.99,
        )
        assert svg.startswith('<svg')


# ──────────────────────────────────────────────────────────────────────────
# 3. build_corpus_difficulty_baseline_html
# ──────────────────────────────────────────────────────────────────────────


class TestBuildHtml:
    def test_returns_empty_when_no_data(self) -> None:
        assert build_corpus_difficulty_baseline_html(None) == ""

    def test_renders_phrase_harder(self) -> None:
        data = {
            "current_difficulty": 0.62,
            "percentile": 88.0,
            "n_runs": 47,
            "median_historical": 0.40,
            "harder_than_usual": True,
            "easier_than_usual": False,
        }
        html = build_corpus_difficulty_baseline_html(data)
        assert "0.62" in html
        assert "88" in html
        assert "47" in html
        assert "plus difficile" in html

    def test_renders_phrase_easier(self) -> None:
        data = {
            "current_difficulty": 0.10,
            "percentile": 12.0,
            "n_runs": 30,
            "median_historical": 0.40,
            "harder_than_usual": False,
            "easier_than_usual": True,
        }
        html = build_corpus_difficulty_baseline_html(data)
        assert "plus facile" in html

    def test_renders_phrase_usual(self) -> None:
        data = {
            "current_difficulty": 0.40,
            "percentile": 50.0,
            "n_runs": 20,
            "median_historical": 0.40,
            "harder_than_usual": False,
            "easier_than_usual": False,
        }
        html = build_corpus_difficulty_baseline_html(data)
        assert "dans la moyenne" in html

    def test_svg_omitted_when_no_history_values(self) -> None:
        data = {
            "current_difficulty": 0.40,
            "percentile": 50.0,
            "n_runs": 20,
            "median_historical": 0.40,
            "harder_than_usual": False,
            "easier_than_usual": False,
        }
        html = build_corpus_difficulty_baseline_html(data)
        assert "<svg" not in html

    def test_svg_present_when_history_provided(self) -> None:
        data = {
            "current_difficulty": 0.62,
            "percentile": 88.0,
            "n_runs": 5,
            "median_historical": 0.30,
            "harder_than_usual": True,
            "easier_than_usual": False,
        }
        html = build_corpus_difficulty_baseline_html(
            data, historical_values=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        assert "<svg" in html


# ──────────────────────────────────────────────────────────────────────────
# 4. Anti-injection
# ──────────────────────────────────────────────────────────────────────────


class TestAntiInjection:
    def test_label_via_i18n_escaped(self) -> None:
        data = {
            "current_difficulty": 0.40, "percentile": 50.0,
            "n_runs": 20, "median_historical": 0.40,
            "harder_than_usual": False, "easier_than_usual": False,
        }
        labels = {"baseline_corpus_title": "<b>Hack</b>"}
        html = build_corpus_difficulty_baseline_html(data, labels=labels)
        assert "<b>Hack</b>" not in html
        assert "&lt;b&gt;Hack&lt;/b&gt;" in html


# ──────────────────────────────────────────────────────────────────────────
# 5. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


class TestI18nCompleteness:
    def _load(self, lang: str) -> dict:
        path = (
            Path(__file__).parent.parent.parent
            / "picarones" / "reports_v2" / "i18n" / f"{lang}.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_all_keys_present_fr(self) -> None:
        d = self._load("fr")
        for key in (
            "baseline_corpus_title",
            "baseline_corpus_harder",
            "baseline_corpus_easier",
            "baseline_corpus_usual",
        ):
            assert key in d, f"manque clé FR : {key}"

    def test_all_keys_present_en(self) -> None:
        d_fr = self._load("fr")
        d_en = self._load("en")
        for key in d_fr:
            if key.startswith("baseline_corpus_"):
                assert key in d_en, f"manque clé EN : {key}"
