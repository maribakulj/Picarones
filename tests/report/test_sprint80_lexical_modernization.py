"""Tests Sprint 80 — A.I.7 : sur-normalisation lexicale.

Couvre :

1. ``compute_lexical_modernization`` :
   - Token GT modernisé systématiquement → 100 %
   - Token GT préservé → 0 %
   - Plusieurs variantes hyp pour un même gt
   - Stop-list filtre les tokens
   - Casse insensible par défaut
   - Token GT supprimé (lacuna) → modernisé vers ∅
   - GT vide → tokens vide
2. ``aggregate_lexical_modernization`` :
   - Somme correcte sur N docs
3. ``top_modernized_tokens`` :
   - Tri décroissant par rate
   - ``min_total`` filtre les anecdotiques
   - Tokens à 0 % exclus
4. Rendu HTML :
   - Tableau, ``""`` si data None ou aucun modernisé
   - Anti-injection
5. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.measurements.lexical_modernization import (
    aggregate_lexical_modernization,
    compute_lexical_modernization,
    top_modernized_tokens,
)
from picarones.reports_v2.html.renderers.lexical_modernization import (
    build_lexical_modernization_html,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. compute_lexical_modernization
# ──────────────────────────────────────────────────────────────────────────


class TestCompute:
    def test_systematic_modernization(self) -> None:
        gt = "maistre maistre maistre"
        hyp = "maître maître maître"
        result = compute_lexical_modernization(gt, hyp)
        slot = result["tokens"]["maistre"]
        assert slot["n_total"] == 3
        assert slot["n_modernized"] == 3
        assert slot["rate_modernized"] == 1.0
        assert slot["variants"] == {"maître": 3}

    def test_preserved_token(self) -> None:
        gt = "nostre nostre"
        hyp = "nostre nostre"
        result = compute_lexical_modernization(gt, hyp)
        slot = result["tokens"]["nostre"]
        assert slot["n_total"] == 2
        assert slot["n_modernized"] == 0
        assert slot["rate_modernized"] == 0.0

    def test_partial_modernization(self) -> None:
        gt = "maistre maistre maistre maistre"
        hyp = "maître maistre maître maître"
        result = compute_lexical_modernization(gt, hyp)
        slot = result["tokens"]["maistre"]
        assert slot["n_total"] == 4
        assert slot["n_modernized"] == 3
        assert slot["rate_modernized"] == 0.75

    def test_multiple_variants(self) -> None:
        gt = "veoir veoir veoir"
        hyp = "voir voyr voir"
        result = compute_lexical_modernization(gt, hyp)
        slot = result["tokens"]["veoir"]
        assert slot["n_total"] == 3
        assert slot["n_modernized"] == 3
        assert slot["variants"] == {"voir": 2, "voyr": 1}

    def test_stop_list_filter(self) -> None:
        gt = "maistre le veoir"
        hyp = "maître la voir"
        result = compute_lexical_modernization(
            gt, hyp, stop_list=["le"],
        )
        # « le » filtré, mais maistre et veoir présents
        assert "le" not in result["tokens"]
        assert "maistre" in result["tokens"]
        assert "veoir" in result["tokens"]

    def test_case_insensitive_default(self) -> None:
        gt = "Maistre maistre"
        hyp = "Maître maître"
        result = compute_lexical_modernization(gt, hyp)
        # Les deux formes sont distinctes en sortie display mais
        # appariées correctement en match
        assert result["tokens"]["Maistre"]["n_modernized"] == 1
        assert result["tokens"]["maistre"]["n_modernized"] == 1

    def test_deletion_counted_as_modernized(self) -> None:
        gt = "maistre veoir"
        hyp = "maître"  # veoir manque
        result = compute_lexical_modernization(gt, hyp)
        # veoir → ∅ compté comme modernisé
        slot = result["tokens"]["veoir"]
        assert slot["n_modernized"] == 1
        assert "∅" in slot["variants"]

    def test_empty_gt(self) -> None:
        result = compute_lexical_modernization("", "anything")
        assert result["tokens"] == {}
        assert result["n_gt_tokens"] == 0

    def test_none_inputs(self) -> None:
        result = compute_lexical_modernization(None, None)
        assert result["tokens"] == {}


# ──────────────────────────────────────────────────────────────────────────
# 2. aggregate
# ──────────────────────────────────────────────────────────────────────────


class TestAggregate:
    def test_sum_across_docs(self) -> None:
        d1 = compute_lexical_modernization(
            "maistre maistre", "maître maître",
        )
        d2 = compute_lexical_modernization(
            "maistre", "maître",
        )
        agg = aggregate_lexical_modernization([d1, d2])
        assert agg["tokens"]["maistre"]["n_total"] == 3
        assert agg["tokens"]["maistre"]["n_modernized"] == 3
        assert agg["tokens"]["maistre"]["rate_modernized"] == 1.0

    def test_empty_iterable(self) -> None:
        agg = aggregate_lexical_modernization([])
        assert agg["tokens"] == {}
        assert agg["n_gt_tokens"] == 0


# ──────────────────────────────────────────────────────────────────────────
# 3. top_modernized_tokens
# ──────────────────────────────────────────────────────────────────────────


class TestTop:
    def test_sorted_by_rate_desc(self) -> None:
        gt = "a a b b c c d d"
        hyp = "x x y b z c d d"
        # a: 100% (2/2 modernisé), b: 50%, c: 50%, d: 0%
        result = compute_lexical_modernization(gt, hyp)
        top = top_modernized_tokens(result, n=10)
        # a en premier
        assert top[0][0] == "a"
        # d exclu (0%)
        names = [t[0] for t in top]
        assert "d" not in names

    def test_min_total_filter(self) -> None:
        gt = "rare maistre maistre maistre"
        hyp = "moderne maître maître maître"
        result = compute_lexical_modernization(gt, hyp)
        # Avec min_total=2 : rare (1) exclu, maistre (3) conservé
        top = top_modernized_tokens(result, min_total=2)
        names = [t[0] for t in top]
        assert "rare" not in names
        assert "maistre" in names


# ──────────────────────────────────────────────────────────────────────────
# 4. Rendu HTML
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def test_returns_empty_when_none(self) -> None:
        assert build_lexical_modernization_html(None) == ""

    def test_returns_empty_when_no_modernizations(self) -> None:
        result = compute_lexical_modernization("a b c", "a b c")
        # Aucun modernisé
        assert build_lexical_modernization_html(result) == ""

    def test_renders_table(self) -> None:
        result = compute_lexical_modernization(
            "maistre veoir", "maître voir",
        )
        html = build_lexical_modernization_html(result)
        assert "<table" in html
        assert "maistre" in html
        assert "maître" in html

    def test_rate_displayed_as_percent(self) -> None:
        result = compute_lexical_modernization(
            "maistre maistre maistre maistre",
            "maître maistre maître maître",
        )
        html = build_lexical_modernization_html(result)
        # 75% présent
        assert "75%" in html

    def test_anti_injection_token(self) -> None:
        gt = "<script>alert(1)</script> normal"
        hyp = "MODERNIZED normal"
        result = compute_lexical_modernization(gt, hyp)
        html = build_lexical_modernization_html(result)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html


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

    def test_all_keys_fr(self) -> None:
        d = self._load("fr")
        for key in (
            "lexmod_title", "lexmod_note", "lexmod_gt_label",
            "lexmod_hyp_label", "lexmod_n_label", "lexmod_rate_label",
        ):
            assert key in d, f"manque clé FR : {key}"

    def test_all_keys_en(self) -> None:
        d_fr = self._load("fr")
        d_en = self._load("en")
        for key in d_fr:
            if key.startswith("lexmod_"):
                assert key in d_en, f"manque clé EN : {key}"
