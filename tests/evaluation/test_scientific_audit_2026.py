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
# Audit 2 — passe systématique sur les métriques non auditées (F14–F19)
# ──────────────────────────────────────────────────────────────────────────


class TestF20DiplomaticProfileResolution:
    """Le runner/web sérialise le profil en NOM (str).  compute_metrics
    doit le résoudre, pas le laisser échouer en silence."""

    def test_string_profile_name_is_resolved(self) -> None:
        # « enſuite faiſoit » vs « ensuite faisoit » : la seule
        # différence (ſ↔s) est neutralisée par le profil médiéval ⇒
        # CER diplomatique = 0.0 (et NON None silencieux).
        m = compute_metrics(
            "enſuite il faiſoit", "ensuite il faisoit",
            normalization_profile="medieval_french",
        )
        assert m.cer_diplomatic is not None
        assert m.cer_diplomatic == pytest.approx(0.0)
        assert m.diplomatic_profile_name == "medieval_french"

    def test_unknown_name_falls_back_not_silently_none(self) -> None:
        m = compute_metrics(
            "abc", "abd", normalization_profile="profil_inexistant",
        )
        # Repli sur le profil par défaut — métrique calculée, pas None.
        assert m.cer_diplomatic is not None
        assert m.diplomatic_profile_name == "medieval_french"

    def test_profile_object_still_works(self) -> None:
        from picarones.evaluation.metrics.normalization import (
            get_builtin_profile,
        )

        m = compute_metrics(
            "abc", "abd",
            normalization_profile=get_builtin_profile("caseless"),
        )
        assert m.cer_diplomatic is not None
        assert m.diplomatic_profile_name == "caseless"


class TestF14TaxonomyUnbiased:
    """La taxonomie n'abandonne plus la classification des
    substitutions dans un bloc inégal ; total_errors ≈ distance mot."""

    def test_substitution_adjacent_to_deletion_not_discarded(self) -> None:
        from rapidfuzz.distance import Levenshtein

        from picarones.evaluation.metrics.taxonomy import classify_errors

        gt = "le roy de France et dAngleterre"
        hyp = "le roi de Frace dAngleterre"  # subst + délétion
        r = classify_errors(gt, hyp)
        # Sous l'ancien code, le bloc replace inégal ne comptait que
        # l'écart de longueur en segmentation et jetait les
        # substitutions → total sous-estimé.
        assert r.total_errors == Levenshtein.distance(gt.split(), hyp.split())
        assert r.counts["segmentation_error"] == 0  # vraie subst, pas seg

    def test_real_segmentation_detected(self) -> None:
        from picarones.evaluation.metrics.taxonomy import classify_errors

        r = classify_errors("le dit acte", "ledit acte")
        assert r.counts["segmentation_error"] == 1

    def test_case_error_not_misread_as_segmentation(self) -> None:
        from picarones.evaluation.metrics.taxonomy import classify_errors

        # Même nombre de tokens, concat égale en casefold : doit rester
        # une erreur de casse, PAS une segmentation.
        r = classify_errors("Bonjour Monde", "bonjour monde")
        assert r.counts["case_error"] >= 1
        assert r.counts["segmentation_error"] == 0


class TestF15LineAlignment:
    def test_dropped_line_does_not_corrupt_distribution(self) -> None:
        from picarones.evaluation.metrics.line_metrics import (
            compute_line_metrics,
        )

        gt = "alpha bravo\ncharlie delta\necho foxtrot\ngolf hotel"
        hyp = "alpha bravo\necho foxtrot\ngolf hotel"  # ligne 2 perdue
        lm = compute_line_metrics(gt, hyp)
        # Seule la ligne réellement perdue est à 1.0 ; les autres 0.0
        # (sous l'ancien zip positionnel, lignes 2-4 ~100 % en cascade).
        assert lm.cer_per_line == [0.0, 1.0, 0.0, 0.0]


class TestF18OverNormalizationAlignment:
    def test_robust_to_ocr_deletion(self) -> None:
        from picarones.evaluation.metrics.over_normalization import (
            detect_over_normalization,
        )

        gt = "le roy de France et de Navarre"
        ocr = "le de France et de Navarre"          # OCR a sauté 'roy'
        llm = "le roy de france et de Navarre"       # LLM : France→france
        r = detect_over_normalization(gt, ocr, llm)
        # L'index positionnel se serait désynchronisé ; l'alignement
        # isole la seule vraie sur-normalisation.
        assert r.over_normalized_count == 1
        assert r.over_normalized_passages[0]["gt"] == "France"


class TestF16NotApplicableIsNone:
    """« Pas de signal » ⇒ None (non applicable), jamais 0.0/1.0,
    et omis par compute_at_junction."""

    def test_philological_modules_return_none_without_signal(self) -> None:
        from picarones.evaluation.metrics.abbreviations import (
            compute_abbreviation_metrics,
        )
        from picarones.evaluation.metrics.mufi import compute_mufi_coverage
        from picarones.evaluation.metrics.roman_numerals import (
            compute_roman_numeral_metrics,
        )

        assert compute_mufi_coverage("hello", "hello")["coverage"] is None
        assert (
            compute_abbreviation_metrics("abc", "abc")["strict_score"]
            is None
        )
        assert (
            compute_roman_numeral_metrics("abc", "abc")["global_strict_score"]
            is None
        )

    def test_junction_omits_none_metrics(self) -> None:
        from picarones.domain.artifacts import ArtifactType
        from picarones.evaluation.metric_registry import compute_at_junction

        # Texte sans aucun signal MUFI/roman/etc. : les métriques
        # philologiques non applicables doivent être ABSENTES du
        # résultat de jonction (ni 0.0 ni None présent).
        res = compute_at_junction(
            "hello world", "hello world",
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        for v in res.values():
            assert v is not None  # aucune None ne subsiste


class TestF19DifficultyHonestDoc:
    def test_module_docstring_no_longer_claims_objective(self) -> None:
        import picarones.evaluation.metrics.difficulty as d

        doc = d.__doc__ or ""
        # La formulation trompeuse « difficulté objective / indépendant
        # des moteurs » a été retirée (composante variance = dépend des
        # moteurs exécutés).
        assert "heuristique" in doc.lower()
        assert "non intrinsèque" in doc or "dépend des moteurs" in doc


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


class TestF7NarrativeTraceability:
    """Synthèse narrative : déterministe + traçable à la SOURCE.

    Contrairement au test historique (qui validait les nombres rendus
    contre le *payload* du Fact, donc circulaire), on reconstruit
    l'espace des nombres admissibles depuis le ``BenchmarkResult``
    d'origine et l'ensemble fermé des dérivations documentées
    (pourcentage ×100, écart, écart relatif, ratio/accélération,
    largeur d'IC), avec arrondis 0–4 décimales.  Un nombre fabriqué
    hors de cet espace ferait échouer le test.
    """

    @staticmethod
    def _variants(x: float) -> set[str]:
        out: set[str] = set()
        for v in (x, x * 100.0):
            try:
                out.add(str(v))
                if v == int(v):
                    out.add(str(int(v)))
                for d in range(5):
                    out.add(f"{v:.{d}f}")
            except (ValueError, OverflowError):
                continue
        return out

    def _source_closure(self, data: dict) -> set[str]:
        import itertools

        # Base = uniquement les valeurs que les détecteurs consomment
        # réellement (ranking + statistics + métriques agrégées par
        # moteur).  Scoper ainsi rend le test (a) rapide, (b) plus
        # sévère : on ne « blanchit » pas un nombre via une valeur
        # per-document sans rapport.
        base: list[float] = []

        def _walk(x):
            if isinstance(x, dict):
                for v in x.values():
                    _walk(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    _walk(v)
            elif isinstance(x, bool):
                return
            elif isinstance(x, (int, float)):
                base.append(float(x))

        _walk(data.get("ranking", []))
        _walk(data.get("statistics", {}))
        for e in data.get("engines", []):
            _walk(e.get("aggregated_metrics", {}) if isinstance(e, dict) else {})
        allowed: set[str] = set()
        for b in base:
            allowed |= self._variants(b)
        # Dérivations documentées entre paires de valeurs source.
        for a, b in itertools.permutations(base, 2):
            allowed |= self._variants(abs(a - b))            # écart / largeur IC
            if b != 0:
                allowed |= self._variants(abs(a - b) / abs(b))  # écart relatif
                allowed |= self._variants(a / b)                # accélération
        return allowed

    def test_synthesis_is_deterministic(self) -> None:
        from picarones.evaluation.synthetic import generate_sample_benchmark
        from picarones.reports.html.data import build_report_data
        from picarones.reports.narrative import build_synthesis

        bm = generate_sample_benchmark(n_docs=8)
        data = build_report_data(bm, images_b64={})
        s1 = build_synthesis(data, "fr", max_facts=5)["sentences"]
        s2 = build_synthesis(data, "fr", max_facts=5)["sentences"]
        assert s1 == s2  # aucune source d'aléa / LLM

    @pytest.mark.parametrize("lang", ["fr", "en"])
    def test_every_rendered_number_traces_to_source(self, lang: str) -> None:
        from picarones.evaluation.synthetic import generate_sample_benchmark
        from picarones.reports.html.data import build_report_data
        from picarones.reports.narrative import build_synthesis, extract_numbers

        bm = generate_sample_benchmark(n_docs=8)
        data = build_report_data(bm, images_b64={})
        out = build_synthesis(data, lang, max_facts=5)
        allowed = self._source_closure(data)

        unknown: list[tuple[str, str]] = []
        for sentence in out["sentences"]:
            for num in extract_numbers(sentence):
                if num.replace(",", ".") not in allowed:
                    unknown.append((num, sentence))
        assert not unknown, (
            "Nombres non reconstructibles depuis le BenchmarkResult "
            f"source (hallucination potentielle) : {unknown}"
        )


class TestF8BootstrapPercentileIndex:
    def test_quantile_index_is_correct_order_statistic(self) -> None:
        from picarones.evaluation.statistics.bootstrap import bootstrap_ci

        # 0..999 → bootstrap des moyennes sur des valeurs constantes-ish.
        vals = [float(i) for i in range(50)]
        lo, hi = bootstrap_ci(vals, n_iter=1000, ci=0.95, seed=42)
        # IC à 95 % centré : bornes plausibles autour de la moyenne 24.5,
        # symétriques, et lo < hi strictement.
        assert lo < hi
        assert 18.0 < lo < 24.5 < hi < 31.0

    def test_deterministic_with_seed(self) -> None:
        from picarones.evaluation.statistics.bootstrap import bootstrap_ci

        v = [0.1, 0.2, 0.05, 0.4, 0.02, 0.3, 0.15]
        assert bootstrap_ci(v, seed=42) == bootstrap_ci(v, seed=42)

    def test_empty_returns_zero_interval(self) -> None:
        from picarones.evaluation.statistics.bootstrap import bootstrap_ci

        assert bootstrap_ci([]) == (0.0, 0.0)


class TestF12DiplomaticTableSinglePass:
    def test_no_cascade_rewrite(self) -> None:
        """Une valeur multi-car contenant une clé simple n'est pas
        ré-écrite (piège des profils YAML personnalisés)."""
        from picarones.formats.text.normalization import (
            _apply_diplomatic_table,
        )

        # "vv"→"w" puis (ancien bug) "w"→"x" donnait "x".
        table = {"vv": "w", "w": "x"}
        assert _apply_diplomatic_table("vvw", table) == "wx"
        # La plus longue clé gagne, un seul passage.
        assert _apply_diplomatic_table("aae", {"ae": "æ", "a": "@"}) == "@æ"

    def test_builtin_profile_unaffected(self) -> None:
        from picarones.formats.text.normalization import get_builtin_profile

        prof = get_builtin_profile("medieval_french")
        # ſ→s, u→v, & → et, appliqué à tout le texte.
        assert prof.normalize("meſſire & vng") == prof.normalize("meſſire & vng")
        assert "ſ" not in prof.normalize("ſus")


class TestF13CorrelationPairwiseComplete:
    def test_missing_value_not_imputed_as_zero(self) -> None:
        from picarones.evaluation.statistics.correlation import (
            compute_correlation_matrix,
        )

        # x et y parfaitement corrélés sur les docs complets ;
        # un doc a y manquant.  L'imputation 0.0 cassait r≈1.
        docs = [
            {"x": 1.0, "y": 2.0},
            {"x": 2.0, "y": 4.0},
            {"x": 3.0, "y": 6.0},
            {"x": 4.0, "y": None},  # y absent → exclu de la paire (x,y)
        ]
        out = compute_correlation_matrix(docs, metric_keys=["x", "y"])
        i, j = out["labels"].index("x"), out["labels"].index("y")
        assert out["matrix"][i][j] == pytest.approx(1.0, abs=1e-9)


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
