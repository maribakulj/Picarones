"""Tests Sprint 94 — B.3 : métrique d'absorption d'erreur.

Couvre :

1. ``compute_error_absorption`` :
   - identité before == after == GT
   - perfect correction
   - perfect introduction
   - mix correction + introduction
   - GT vide → None
   - case insensitive
   - multiplicité
2. ``aggregate_error_absorption`` :
   - somme corpus-wide + recalcul micro
   - empty
3. Cas réaliste pipeline OCR → LLM modernisant.
4. Vue HTML : adaptive, anti-injection, FR + EN.
5. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picarones.measurements.error_absorption import (
    aggregate_error_absorption,
    compute_error_absorption,
)
from picarones.report.error_absorption_render import (
    build_error_absorption_html,
)


def _load_labels(lang: str) -> dict:
    p = (
        Path(__file__).parent.parent
        / "picarones" / "report" / "i18n" / f"{lang}.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────
# 1. compute_error_absorption
# ──────────────────────────────────────────────────────────────────────────


class TestCompute:
    def test_identity_no_errors(self) -> None:
        gt = "le roi signa la charte"
        r = compute_error_absorption(gt, gt, gt)
        assert r is not None
        assert r["n_errors_before"] == 0
        assert r["n_errors_after"] == 0
        assert r["n_corrected"] == 0
        assert r["n_introduced"] == 0
        assert r["correction_rate"] is None
        assert r["introduction_rate"] is None

    def test_perfect_correction(self) -> None:
        gt = "le roi signa la charte"
        before = "lc roi signz la charte"  # 2 erreurs
        after = "le roi signa la charte"   # 0 erreur
        r = compute_error_absorption(gt, before, after)
        assert r["n_errors_before"] == 2
        assert r["n_errors_after"] == 0
        assert r["n_corrected"] == 2
        assert r["n_introduced"] == 0
        assert r["correction_rate"] == 1.0
        assert r["introduction_rate"] is None
        assert r["net_improvement"] == 2

    def test_pure_introduction(self) -> None:
        # before = GT, after dégrade
        gt = "le roi signa"
        r = compute_error_absorption(gt, gt, "le souverain signa")
        assert r["n_errors_before"] == 0
        assert r["n_errors_after"] == 1
        assert r["n_corrected"] == 0
        assert r["n_introduced"] == 1
        assert r["correction_rate"] is None
        assert r["introduction_rate"] == 1.0
        assert r["net_improvement"] == -1
        assert "roi" in r["introduced_tokens"]

    def test_mix_correction_and_introduction(self) -> None:
        gt = "maistre Pierre du Bois"
        before = "maistre Pier du Bois"  # 1 erreur (Pierre absent)
        after = "maître Pierre du Bois"  # 1 erreur (maistre → maître)
        r = compute_error_absorption(gt, before, after)
        assert r["n_corrected"] == 1
        assert "Pierre" in r["corrected_tokens"]
        assert r["n_introduced"] == 1
        assert "maistre" in r["introduced_tokens"]
        assert r["correction_rate"] == 1.0
        assert r["introduction_rate"] == 1.0
        assert r["net_improvement"] == 0

    def test_empty_gt_returns_none(self) -> None:
        assert compute_error_absorption("", "x", "y") is None

    def test_case_insensitive(self) -> None:
        gt = "Le Roi"
        before = "le roi"  # casse différente, considéré OK
        after = "le roi"
        r = compute_error_absorption(gt, before, after)
        assert r["n_errors_before"] == 0
        assert r["n_errors_after"] == 0

    def test_case_sensitive_opt_in(self) -> None:
        gt = "Le Roi"
        before = "le roi"
        r = compute_error_absorption(
            gt, before, "Le Roi", case_sensitive=True,
        )
        # before n'a pas la bonne casse → 2 erreurs
        assert r["n_errors_before"] == 2
        assert r["n_errors_after"] == 0
        assert r["n_corrected"] == 2

    def test_multiplicity(self) -> None:
        gt = "le le roi"
        before = "le roi"     # 1 occurrence "le" manque
        after = "le le roi"
        r = compute_error_absorption(gt, before, after)
        assert r["n_errors_before"] == 1
        assert r["n_corrected"] == 1


# ──────────────────────────────────────────────────────────────────────────
# 2. aggregate_error_absorption
# ──────────────────────────────────────────────────────────────────────────


class TestAggregate:
    def test_micro_rate(self) -> None:
        d1 = compute_error_absorption(
            "abc def ghi", "abc xxx ghi", "abc def ghi",
        )
        d2 = compute_error_absorption(
            "alpha beta", "alpha xxx", "alpha beta",
        )
        agg = aggregate_error_absorption([d1, d2])
        assert agg is not None
        assert agg["n_corrected"] == 2
        assert agg["n_introduced"] == 0
        assert agg["correction_rate"] == pytest.approx(1.0)

    def test_skips_none(self) -> None:
        agg = aggregate_error_absorption([None, None])
        assert agg is None

    def test_empty(self) -> None:
        assert aggregate_error_absorption([]) is None

    def test_sample_capped(self) -> None:
        # Beaucoup d'erreurs introduites
        per_doc = []
        for i in range(10):
            r = compute_error_absorption(
                "x", "x", f"y{i}",  # 1 introduction
            )
            per_doc.append(r)
        agg = aggregate_error_absorption(per_doc, sample_tokens=5)
        # On ne capture que 5 dans l'échantillon malgré 10
        assert len(agg["introduced_tokens_sample"]) <= 5


# ──────────────────────────────────────────────────────────────────────────
# 3. Cas réaliste
# ──────────────────────────────────────────────────────────────────────────


class TestRealistic:
    def test_llm_modernizing_signature(self) -> None:
        # OCR fait 2 erreurs ; LLM corrige 1 mais modernise 1
        gt = "maistre Pierre signe la charte"
        before = "maistre Pier signe la charte"  # Pierre cassé
        after = "maître Pierre signe la charte"  # corrigé Pierre, modernisé maistre
        r = compute_error_absorption(gt, before, after)
        # 1 corrigée, 1 introduite → net = 0 mais signal d'absorption
        assert r["n_corrected"] >= 1
        assert r["n_introduced"] >= 1
        assert "Pierre" in r["corrected_tokens"]
        assert "maistre" in r["introduced_tokens"]


# ──────────────────────────────────────────────────────────────────────────
# 4. Vue HTML
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def test_empty_returns_empty(self) -> None:
        assert build_error_absorption_html(None) == ""
        assert build_error_absorption_html([]) == ""

    def test_renders_table(self) -> None:
        per_doc = [compute_error_absorption(
            "le roi signa", "lc roi signa", "le roi signa",
        )]
        agg = aggregate_error_absorption(per_doc)
        agg["junction_name"] = "ocr_to_llm"
        html = build_error_absorption_html([agg], _load_labels("fr"))
        assert "<table" in html
        assert "ocr_to_llm" in html
        # Colonne corrected / introduced
        assert "Corrigées" in html
        assert "Introduites" in html

    def test_anti_injection(self) -> None:
        per_doc = [compute_error_absorption(
            "le roi", "lc roi", "le roi",
        )]
        agg = aggregate_error_absorption(per_doc)
        agg["junction_name"] = "<script>alert(1)</script>"
        html = build_error_absorption_html([agg], _load_labels("fr"))
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_renders_sample_introduced(self) -> None:
        per_doc = [compute_error_absorption(
            "le roi signa", "le roi signa", "le souverain signa",
        )]
        agg = aggregate_error_absorption(per_doc)
        agg["junction_name"] = "llm_postcorr"
        html = build_error_absorption_html([agg], _load_labels("fr"))
        # « roi » apparaît dans la colonne échantillon (introduit)
        assert "roi" in html

    def test_renders_in_english(self) -> None:
        per_doc = [compute_error_absorption(
            "le roi", "lc roi", "le roi",
        )]
        agg = aggregate_error_absorption(per_doc)
        agg["junction_name"] = "ocr_to_llm"
        html = build_error_absorption_html([agg], _load_labels("en"))
        assert "Error absorption per junction" in html


# ──────────────────────────────────────────────────────────────────────────
# 5. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


_KEYS = {
    "absorption_title", "absorption_note",
    "absorption_junction", "absorption_errors_before",
    "absorption_errors_after", "absorption_corrected",
    "absorption_introduced", "absorption_corr_rate",
    "absorption_intro_rate", "absorption_net",
    "absorption_sample",
}


class TestI18n:
    def test_fr(self) -> None:
        d = _load_labels("fr")
        assert not _KEYS - d.keys()

    def test_en(self) -> None:
        d = _load_labels("en")
        assert not _KEYS - d.keys()
