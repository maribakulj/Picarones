"""Régression — audit scientifique (mai 2026).

Chaque test verrouille une correction de l'audit de fiabilité
scientifique afin qu'aucune régression ne ré-introduise un calcul
faux ou une donnée trompeuse.  Les identifiants Fxx renvoient au
rapport d'audit.

Ces tests s'exécutent sur le chemin **sans scipy** (installation par
défaut ``[dev,web]``), qui est le chemin de production le plus courant
et celui où les défauts F2/F9 étaient atteignables.
"""

from __future__ import annotations

import math

import pytest

from picarones.evaluation._diff_utils import compute_char_diff, diff_stats
from picarones.evaluation.metric_result import MetricsResult, aggregate_metrics
from picarones.evaluation.metrics.confusion import build_confusion_matrix
from picarones.evaluation.metrics.text_metrics import compute_metrics
from picarones.evaluation.statistics.wilcoxon import (
    _exact_signed_rank_two_sided_p,
    wilcoxon_test,
)


# ──────────────────────────────────────────────────────────────────────────
# F1 — CER/WER micro-moyenné (pondéré par la longueur)
# ──────────────────────────────────────────────────────────────────────────


class TestF1MicroAverage:
    def test_compute_metrics_stores_exact_edit_counts(self) -> None:
        """Les comptes bruts permettent de recomposer le CER exact."""
        m = compute_metrics("abcde fghij", "abXde fg")
        assert m.cer_errors is not None and m.cer_ref_chars is not None
        # CER = distance_édition / caractères_référence (def. exacte).
        assert m.cer == pytest.approx(m.cer_errors / m.cer_ref_chars)
        assert m.wer == pytest.approx(m.wer_errors / m.wer_ref_words)

    def test_micro_average_is_length_weighted(self) -> None:
        """Le micro-CER pondère par la longueur ; la macro-moyenne non.

        Doc court : 'ab' → 'aX'  (1 erreur / 2 car  = 0.50)
        Doc long  : 100·'a' → 90·'a'+10·'b' (10 err / 100 car = 0.10)
        macro mean = (0.50 + 0.10)/2 = 0.30
        micro      = (1 + 10) / (2 + 100) = 11/102 ≈ 0.1078
        """
        docs = [
            compute_metrics("ab", "aX"),
            compute_metrics("a" * 100, "a" * 90 + "b" * 10),
        ]
        agg = aggregate_metrics(docs)
        assert agg["cer"]["mean"] == pytest.approx(0.30, abs=1e-6)
        assert agg["cer_micro"]["value"] == pytest.approx(11 / 102, abs=1e-6)
        assert agg["cer_micro"]["total_errors"] == 11
        assert agg["cer_micro"]["total_reference_units"] == 102

    def test_micro_absent_when_no_raw_counts(self) -> None:
        """Fixture legacy sans comptes → pas de clé micro (repli médiane)."""
        legacy = [
            MetricsResult(cer=0.1, wer=0.1, reference_length=10),
            MetricsResult(cer=0.2, wer=0.2, reference_length=10),
        ]
        agg = aggregate_metrics(legacy)
        assert "cer_micro" not in agg
        assert agg["cer"]["mean"] == pytest.approx(0.15)

    def test_round_trip_preserves_counts(self) -> None:
        m = compute_metrics("le roy de France", "le roi de Frace")
        restored = MetricsResult.from_dict(m.as_dict())
        assert restored.cer_errors == m.cer_errors
        assert restored.cer_ref_chars == m.cer_ref_chars
        assert restored.wer_errors == m.wer_errors
        assert restored.wer_ref_words == m.wer_ref_words


# ──────────────────────────────────────────────────────────────────────────
# F2 — Wilcoxon : plus aucune p-value fabriquée pour petit n
# ──────────────────────────────────────────────────────────────────────────


class TestF2WilcoxonExactSmallN:
    def test_no_false_positive_for_n_le_5(self) -> None:
        """Pour n ≤ 5, la significativité bilatérale à 5 % est
        mathématiquement impossible (p_min = 2/2ⁿ ≥ 0.0625).

        L'ancienne table renvoyait p=0.04 « significatif » quand un
        moteur dominait l'autre sur les 5 documents — un faux positif.
        """
        # Différences toutes positives, magnitudes distinctes → pas
        # d'ex-aequo → chemin exact, W = 0.
        worse = [0.20, 0.31, 0.42, 0.53, 0.64]
        better = [0.10, 0.20, 0.30, 0.40, 0.50]
        res = wilcoxon_test(better, worse)
        assert res["method"] == "exact"
        assert res["p_value"] == pytest.approx(0.0625)
        assert res["significant"] is False

    @pytest.mark.parametrize(
        "n,w,expected",
        [
            (6, 0, 2 / 64),          # plus petit n significatif à 5 %
            (7, 2, 0.046875),
            (8, 3, 0.0390625),
            (8, 4, 0.0546875),       # juste au-dessus du seuil
            (10, 8, 0.0488281),
        ],
    )
    def test_exact_pvalues_match_statistical_tables(
        self, n: int, w: int, expected: float,
    ) -> None:
        total = n * (n + 1) // 2
        p = _exact_signed_rank_two_sided_p(n, w, total - w)
        assert p == pytest.approx(expected, abs=1e-6)

    def test_n5_pvalue_distribution_is_well_formed(self) -> None:
        """La p-value exacte est un vrai quantile ∈ ]0, 1], jamais une
        constante fabriquée comme 0.04 ou 0.20."""
        seen = set()
        total = 5 * 6 // 2
        for w in range(total + 1):
            p = _exact_signed_rank_two_sided_p(5, w, total - w)
            assert 0.0 < p <= 1.0
            seen.add(round(p, 6))
        assert 0.04 not in seen and 0.20 not in seen
        assert min(seen) == pytest.approx(0.0625)  # = 2/32

    def test_ties_use_corrected_normal_approx(self) -> None:
        a = [1, 2, 2, 3, 5, 5, 7, 9, 9, 11, 2, 4]
        b = [1, 1, 2, 3, 4, 5, 6, 9, 8, 10, 2, 3]
        res = wilcoxon_test(a, b)
        assert res["has_ties"] is True
        assert res["method"] == "normal_approx"
        assert 0.0 < res["p_value"] <= 1.0


# ──────────────────────────────────────────────────────────────────────────
# F9 — correction de continuité standard, bornée à 0
# ──────────────────────────────────────────────────────────────────────────


class TestF3SyntheticDataIntegrity:
    """Une donnée fabriquée ne peut pas être prise pour un résultat réel."""

    def test_synthetic_benchmark_is_flagged(self) -> None:
        from picarones.evaluation.synthetic import generate_sample_benchmark

        bm = generate_sample_benchmark(n_docs=3)
        assert bm.is_demo is True
        d = bm.as_dict()
        assert d["is_demo"] is True
        assert d["corpus"]["is_demo"] is True

    def test_real_benchmark_defaults_to_not_demo(self) -> None:
        from picarones.evaluation.benchmark_result import BenchmarkResult

        bm = BenchmarkResult(
            corpus_name="Vrai corpus", corpus_source=None,
            document_count=0, engine_reports=[],
        )
        assert bm.is_demo is False
        assert bm.as_dict()["is_demo"] is False

    def test_is_demo_round_trips_through_json(self) -> None:
        import json

        from picarones.evaluation.benchmark_result import BenchmarkResult
        from picarones.evaluation.synthetic import generate_sample_benchmark

        bm = generate_sample_benchmark(n_docs=3)
        rt = BenchmarkResult.from_dict(json.loads(json.dumps(bm.as_dict())))
        assert rt.is_demo is True

    def test_html_report_carries_unremovable_banner_for_demo(
        self, tmp_path,
    ) -> None:
        from picarones.evaluation.synthetic import generate_sample_benchmark
        from picarones.reports.html.generator import ReportGenerator

        bm = generate_sample_benchmark(n_docs=3)
        out = tmp_path / "demo.html"
        ReportGenerator(bm).generate(str(out))
        html = out.read_text(encoding="utf-8")
        assert 'role="alert"' in html
        assert "DÉMONSTRATION" in html or "DEMONSTRATION" in html
        # Pas de bouton de fermeture sur le bandeau d'intégrité.
        assert "onclick=\"this.remove()\"" not in html

    def test_real_html_report_has_no_demo_banner(self, tmp_path) -> None:
        from picarones.evaluation.benchmark_result import BenchmarkResult
        from picarones.reports.html.generator import ReportGenerator

        bm = BenchmarkResult(
            corpus_name="Vrai corpus", corpus_source=None,
            document_count=0, engine_reports=[],
        )
        out = tmp_path / "real.html"
        ReportGenerator(bm).generate(str(out))
        html = out.read_text(encoding="utf-8")
        assert "DÉMONSTRATION" not in html
        assert "DEMONSTRATION DATA" not in html


class TestF4MinimalAlignment:
    """Confusion matrix / diff alignés sur Levenshtein (≡ CER)."""

    @pytest.mark.parametrize(
        "gt,hyp",
        [
            ("maistre Jehan Froissart", "maiſtre Iehan Froiflart"),
            ("le roy de France", "le roi de la France"),
            ("abcdefghij", "aXcdefghijKL"),
            ("ſuſpicion", "fufpicion"),
            ("", "inséré"),
            ("supprimé", ""),
        ],
    )
    def test_confusion_total_equals_levenshtein_distance(
        self, gt: str, hyp: str,
    ) -> None:
        """S+D+I de la matrice = distance d'édition de Levenshtein,
        donc cohérent avec le numérateur du CER (jiwer).

        Sous Ratcliff–Obershelp (difflib, ancien code) cette égalité
        était fausse dès qu'une insertion/suppression décalait la suite.
        """
        from rapidfuzz.distance import Levenshtein

        cm = build_confusion_matrix(
            gt, hyp, ignore_whitespace=False, ignore_correct=True,
        )
        total = (
            cm.total_substitutions
            + cm.total_insertions
            + cm.total_deletions
        )
        assert total == Levenshtein.distance(gt, hyp)

    def test_char_diff_is_minimal_edit(self) -> None:
        """Le diff caractère ne sur-segmente pas : le nombre d'opérations
        non-equal égale la distance de Levenshtein (1 op = 1 édition)."""
        from rapidfuzz.distance import Levenshtein

        gt, hyp = "abcdef", "aXcdefY"
        ops = compute_char_diff(gt, hyp)
        st = diff_stats(ops)
        edits = st["replace"] + st["insert"] + st["delete"]
        assert edits == Levenshtein.distance(gt, hyp) == 2


class TestF5ImanDavenport:
    """Friedman expose la correction F recommandée par Demšar (2006)."""

    def test_f_statistic_fields_present_and_consistent(self) -> None:
        from picarones.evaluation.statistics.friedman_nemenyi import (
            friedman_test,
        )

        # 3 moteurs, 6 documents, séparation nette.
        m = {
            "A": [0.05, 0.04, 0.06, 0.05, 0.05, 0.04],
            "B": [0.15, 0.14, 0.16, 0.15, 0.15, 0.14],
            "C": [0.30, 0.31, 0.29, 0.30, 0.30, 0.31],
        }
        r = friedman_test(m)
        assert r["f_df1"] == 2  # k-1
        assert r["f_df2"] == 2 * (6 - 1)  # (k-1)(n-1)
        assert r["decision_basis"] == "iman_davenport_F"
        assert r["f_p_value"] is not None
        # F recommandé moins conservateur que χ² : p_F ≤ p_χ².
        assert r["f_p_value"] <= r["p_value"] + 1e-9
        assert r["significant"] is True

    def test_f_sf_matches_distribution_tables(self) -> None:
        from picarones.evaluation.statistics.friedman_nemenyi import _f_sf

        # Valeurs critiques F connues à α=0.05.
        assert _f_sf(4.103, 2, 10) == pytest.approx(0.05, abs=1e-3)
        assert _f_sf(3.490, 3, 12) == pytest.approx(0.05, abs=1e-3)
        assert _f_sf(2.711, 5, 20) == pytest.approx(0.05, abs=1e-3)
        assert _f_sf(float("inf"), 3, 9) == 0.0
        assert _f_sf(0.0, 3, 9) == 1.0

    def test_perfect_concordance_gives_infinite_F(self) -> None:
        from picarones.evaluation.statistics.friedman_nemenyi import (
            friedman_test,
        )

        # Ordre des moteurs identique sur tous les documents.
        m = {
            "A": [0.1, 0.1, 0.1, 0.1],
            "B": [0.2, 0.2, 0.2, 0.2],
            "C": [0.3, 0.3, 0.3, 0.3],
        }
        r = friedman_test(m)
        assert r["f_statistic"] == float("inf")
        assert r["f_p_value"] == 0.0
        assert r["significant"] is True


class TestF6NemenyiOutOfTable:
    """q_α extrapolé vers le haut (conservateur), plus de clamp."""

    def test_q_alpha_is_monotone_increasing_beyond_table(self) -> None:
        from picarones.evaluation.statistics.friedman_nemenyi import (
            _nemenyi_critical_value,
        )

        # Réutiliser q(50) pour tout k>50 (ancien code) sous-estimait
        # q ⇒ anti-conservateur.  Doit désormais croître.
        assert _nemenyi_critical_value(100, 0.05) > _nemenyi_critical_value(
            50, 0.05,
        )
        assert _nemenyi_critical_value(60, 0.01) > _nemenyi_critical_value(
            50, 0.01,
        )

    def test_nemenyi_posthoc_flags_extrapolation(self) -> None:
        from picarones.evaluation.statistics.friedman_nemenyi import (
            nemenyi_posthoc,
        )

        small = {f"e{i}": [0.1 * (i + 1), 0.2 * (i + 1)] for i in range(3)}
        r = nemenyi_posthoc(small)
        assert r["q_alpha_extrapolated"] is False


class TestF9ContinuityCorrection:
    def test_no_signal_gives_non_significant(self) -> None:
        """W ≈ μ (aucun effet) ⇒ z borné à 0 ⇒ p = 1.0, jamais < 1
        par sur-correction (ancienne forme |（W+½)−μ|)."""
        # Beaucoup d'ex-aequo et différences symétriques → approx normale.
        a = [0.10, 0.20, 0.10, 0.20, 0.10, 0.20, 0.10, 0.20,
             0.10, 0.20, 0.10, 0.20]
        b = [0.20, 0.10, 0.20, 0.10, 0.20, 0.10, 0.20, 0.10,
             0.20, 0.10, 0.20, 0.10]
        res = wilcoxon_test(a, b)
        assert res["p_value"] == pytest.approx(1.0)
        assert res["significant"] is False
