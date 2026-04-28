"""Tests Sprint 76 — A.I.4 chantier 2 : évolution intra-document.

Couvre :

1. ``compute_taxonomy_position_heatmap`` :
   - GT/hyp identiques → total_errors = 0, per_class entièrement zéro
   - GT vide → ``None``
   - Erreur en début de doc → bin[0] non nul, autres bins nuls
   - Erreur en fin de doc → bin[n_bins-1] non nul
   - Erreurs uniformément distribuées → tous bins ≈ 1
   - Cas dégénéré ``n_bins=0`` → ``ValueError``
   - Doc avec moins de mots que n_bins → distribution sparse correcte
2. Rendu HTML :
   - Bien formé (SVG)
   - ``""`` si data None
   - ``""`` si total_errors=0
   - ``""`` si toutes les classes ont 0 erreur
3. Anti-injection : labels i18n contenant ``<script>``.
4. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picarones.core.taxonomy_intra_doc import (
    compute_taxonomy_position_heatmap,
)
from picarones.report.taxonomy_intra_doc_render import (
    build_taxonomy_intra_doc_html,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. Couche de calcul
# ──────────────────────────────────────────────────────────────────────────


class TestCompute:
    def test_identical_no_errors(self) -> None:
        result = compute_taxonomy_position_heatmap(
            "alpha beta gamma delta epsilon",
            "alpha beta gamma delta epsilon",
        )
        assert result is not None
        assert result["total_errors"] == 0
        for cls, counts in result["per_class"].items():
            assert all(c == 0 for c in counts)

    def test_empty_gt_returns_none(self) -> None:
        assert compute_taxonomy_position_heatmap("", "anything") is None
        assert compute_taxonomy_position_heatmap(None, None) is None

    def test_error_at_start(self) -> None:
        # 10 mots GT, erreur sur le premier seulement
        gt = "alphA beta gamma delta epsilon zeta eta theta iota kappa"
        # Avec 10 bins et 10 mots → 1 mot par bin
        # Erreur de casse en position 0 → bin 0
        hyp = "Alpha beta gamma delta epsilon zeta eta theta iota kappa"
        result = compute_taxonomy_position_heatmap(gt, hyp, n_bins=10)
        assert result is not None
        assert result["total_errors"] == 1
        assert result["totals_per_bin"][0] == 1
        for i in range(1, 10):
            assert result["totals_per_bin"][i] == 0

    def test_error_at_end(self) -> None:
        gt = "alpha beta gamma delta epsilon zeta eta theta iota kappA"
        hyp = "alpha beta gamma delta epsilon zeta eta theta iota Kappa"
        result = compute_taxonomy_position_heatmap(gt, hyp, n_bins=10)
        assert result is not None
        assert result["total_errors"] == 1
        # Position 9 sur 10 → bin 9
        assert result["totals_per_bin"][9] == 1

    def test_uniform_distribution(self) -> None:
        # 10 mots, 1 erreur de casse sur chacun → 1 erreur par bin
        gt = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa"
        hyp = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        result = compute_taxonomy_position_heatmap(gt, hyp, n_bins=10)
        assert result is not None
        assert result["total_errors"] == 10
        # Tous les bins ≈ 1
        assert all(c == 1 for c in result["totals_per_bin"])

    def test_invalid_n_bins(self) -> None:
        with pytest.raises(ValueError):
            compute_taxonomy_position_heatmap("a b", "a b", n_bins=0)
        with pytest.raises(ValueError):
            compute_taxonomy_position_heatmap("a b", "a b", n_bins=-1)

    def test_per_class_breakdown(self) -> None:
        # 1 erreur de casse + 1 lacune
        gt = "Alpha beta gamma"
        hyp = "alpha beta"  # alpha→Alpha (case) ; gamma manque (lacuna)
        result = compute_taxonomy_position_heatmap(gt, hyp, n_bins=3)
        assert result is not None
        assert sum(result["per_class"]["case_error"]) == 1
        assert sum(result["per_class"]["lacuna"]) == 1

    def test_more_bins_than_words(self) -> None:
        # 3 mots et 10 bins → certains bins resteront vides
        result = compute_taxonomy_position_heatmap(
            "Alpha Beta Gamma", "alpha beta gamma", n_bins=10,
        )
        assert result is not None
        assert sum(result["totals_per_bin"]) == 3


# ──────────────────────────────────────────────────────────────────────────
# 2. Rendu HTML
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def test_returns_empty_when_none(self) -> None:
        assert build_taxonomy_intra_doc_html(None) == ""

    def test_returns_empty_when_no_errors(self) -> None:
        data = compute_taxonomy_position_heatmap("a b c", "a b c")
        # total_errors=0 → ""
        assert build_taxonomy_intra_doc_html(data) == ""

    def test_renders_svg(self) -> None:
        data = compute_taxonomy_position_heatmap(
            "Alpha beta gamma delta",
            "alpha Beta gamma DELTA",
            n_bins=4,
        )
        html = build_taxonomy_intra_doc_html(data)
        assert "<svg" in html
        assert "</svg>" in html

    def test_class_labels_present(self) -> None:
        data = compute_taxonomy_position_heatmap(
            "Alpha", "alpha", n_bins=5,
        )
        html = build_taxonomy_intra_doc_html(data)
        assert "case_error" in html

    def test_n_words_displayed(self) -> None:
        data = compute_taxonomy_position_heatmap(
            "Alpha beta", "alpha BETA", n_bins=5,
        )
        html = build_taxonomy_intra_doc_html(data)
        assert "2" in html  # n_words_gt = 2


# ──────────────────────────────────────────────────────────────────────────
# 3. Anti-injection
# ──────────────────────────────────────────────────────────────────────────


class TestAntiInjection:
    def test_label_via_i18n_escaped(self) -> None:
        data = compute_taxonomy_position_heatmap(
            "Alpha", "alpha", n_bins=5,
        )
        labels = {"intradoc_title": "<b>Hack</b>"}
        html = build_taxonomy_intra_doc_html(data, labels=labels)
        assert "<b>Hack</b>" not in html
        assert "&lt;b&gt;Hack&lt;/b&gt;" in html


# ──────────────────────────────────────────────────────────────────────────
# 4. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


class TestI18nCompleteness:
    def _load(self, lang: str) -> dict:
        path = (
            Path(__file__).parent.parent
            / "picarones" / "report" / "i18n" / f"{lang}.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_all_keys_fr(self) -> None:
        d = self._load("fr")
        for key in ("intradoc_title", "intradoc_note", "intradoc_n_words"):
            assert key in d, f"manque clé FR : {key}"

    def test_all_keys_en(self) -> None:
        d_fr = self._load("fr")
        d_en = self._load("en")
        for key in d_fr:
            if key.startswith("intradoc_"):
                assert key in d_en, f"manque clé EN : {key}"
