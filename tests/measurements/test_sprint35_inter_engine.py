"""Tests Sprint 35 — métriques inter-moteurs (Étape 2 du plan).

Couvre les deux familles de mesures du module ``picarones.measurements.inter_engine`` :

1. **Divergence taxonomique** : KL et JS-divergence sur les
   distributions de classes d'erreur, plus la matrice triangulaire
   inter-moteurs.  Tests : invariants mathématiques (positivité, JS
   symétrique et bornée, KL(p,p)=0), comportement sur clés disjointes.

2. **Complémentarité** : oracle token recall, gap absolu/relatif vs
   meilleur moteur seul, taux de désaccord par paire.  Tests : cas
   parfait (oracle = best = 1), cas où un ensemble apporte un vrai gain,
   cas d'égalité parfaite (gap = 0), garde-fous (référence vide,
   hypothèses vides).

Les fonctions sont pures ; pas besoin de fixtures d'I/O ni de moteurs
réels.
"""

from __future__ import annotations

import math

import pytest

from picarones.measurements.inter_engine import (
    complementarity_gap,
    jensen_shannon_divergence,
    kl_divergence,
    oracle_token_recall,
    pairwise_disagreement_rate,
    taxonomy_divergence_matrix,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. KL-divergence
# ──────────────────────────────────────────────────────────────────────────


class TestKLDivergence:
    def test_self_divergence_is_zero(self) -> None:
        p = {"a": 0.4, "b": 0.3, "c": 0.3}
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_kl_is_non_negative(self) -> None:
        p = {"a": 0.7, "b": 0.2, "c": 0.1}
        q = {"a": 0.1, "b": 0.4, "c": 0.5}
        assert kl_divergence(p, q) > 0
        assert kl_divergence(q, p) > 0

    def test_kl_is_asymmetric_in_general(self) -> None:
        # Choix asymétrique non symétrique par permutation
        p = {"a": 0.9, "b": 0.05, "c": 0.05}
        q = {"a": 0.4, "b": 0.4, "c": 0.2}
        assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p), abs=1e-3)

    def test_disjoint_keys_handled(self) -> None:
        # Pas de clé en commun : doit retourner une valeur finie grâce
        # au lissage epsilon.
        p = {"a": 1.0}
        q = {"b": 1.0}
        kl = kl_divergence(p, q)
        assert math.isfinite(kl)
        assert kl > 0

    def test_empty_distributions_return_zero(self) -> None:
        assert kl_divergence({}, {}) == 0.0


# ──────────────────────────────────────────────────────────────────────────
# 2. Jensen-Shannon divergence
# ──────────────────────────────────────────────────────────────────────────


class TestJensenShannonDivergence:
    def test_self_divergence_is_zero(self) -> None:
        p = {"a": 0.4, "b": 0.3, "c": 0.3}
        assert jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_symmetric(self) -> None:
        p = {"a": 0.7, "b": 0.2, "c": 0.1}
        q = {"a": 0.1, "b": 0.4, "c": 0.5}
        assert jensen_shannon_divergence(p, q) == pytest.approx(
            jensen_shannon_divergence(q, p), abs=1e-9
        )

    def test_bounded_in_unit_interval(self) -> None:
        # JS en bits ∈ [0, 1].  Distributions extrêmes : disjointes.
        p = {"a": 1.0}
        q = {"b": 1.0}
        js = jensen_shannon_divergence(p, q)
        assert 0.0 <= js <= 1.0
        # Les distributions disjointes donnent une JS proche de 1 (la
        # borne est atteinte asymptotiquement).
        assert js > 0.5

    def test_close_distributions_have_small_js(self) -> None:
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 0.51, "b": 0.49}
        assert jensen_shannon_divergence(p, q) < 0.01


# ──────────────────────────────────────────────────────────────────────────
# 3. Matrice de divergence inter-moteurs
# ──────────────────────────────────────────────────────────────────────────


class TestDivergenceMatrix:
    @pytest.fixture
    def engines(self) -> dict[str, dict[str, float]]:
        return {
            "tesseract": {"visual": 0.5, "casse": 0.3, "abbrev": 0.2},
            "pero": {"visual": 0.2, "casse": 0.3, "abbrev": 0.5},
            "mistral": {"visual": 0.4, "casse": 0.4, "abbrev": 0.2},
        }

    def test_diagonal_is_zero(
        self, engines: dict[str, dict[str, float]]
    ) -> None:
        mat = taxonomy_divergence_matrix(engines)
        for name in engines:
            assert mat[name][name] == pytest.approx(0.0, abs=1e-9)

    def test_js_matrix_is_symmetric(
        self, engines: dict[str, dict[str, float]]
    ) -> None:
        mat = taxonomy_divergence_matrix(engines, metric="js")
        for a in engines:
            for b in engines:
                assert mat[a][b] == pytest.approx(mat[b][a], abs=1e-9)

    def test_kl_matrix_is_asymmetric(
        self, engines: dict[str, dict[str, float]]
    ) -> None:
        mat = taxonomy_divergence_matrix(engines, metric="kl")
        # Au moins une paire doit être asymétrique
        asymmetric_found = any(
            abs(mat[a][b] - mat[b][a]) > 1e-6
            for a in engines for b in engines if a != b
        )
        assert asymmetric_found

    def test_unknown_metric_raises(
        self, engines: dict[str, dict[str, float]]
    ) -> None:
        with pytest.raises(ValueError, match="metric"):
            taxonomy_divergence_matrix(engines, metric="hellinger")

    def test_distinguishes_specialized_engines(self) -> None:
        """Deux moteurs avec profils opposés doivent ressortir comme
        candidats à un ensemble (JS élevée)."""
        engines = {
            "visual_specialist": {"visual": 0.9, "casse": 0.05, "abbrev": 0.05},
            "abbrev_specialist": {"visual": 0.05, "casse": 0.05, "abbrev": 0.9},
            "balanced": {"visual": 0.33, "casse": 0.33, "abbrev": 0.34},
        }
        mat = taxonomy_divergence_matrix(engines, metric="js")
        # Les deux spécialistes doivent diverger plus l'un de l'autre que
        # n'importe lequel d'eux du moteur balanced.
        assert mat["visual_specialist"]["abbrev_specialist"] > mat["visual_specialist"]["balanced"]
        assert mat["visual_specialist"]["abbrev_specialist"] > mat["abbrev_specialist"]["balanced"]


# ──────────────────────────────────────────────────────────────────────────
# 4. Oracle token recall
# ──────────────────────────────────────────────────────────────────────────


class TestOracleTokenRecall:
    def test_perfect_engine_oracle_is_one(self) -> None:
        ref = "le manuscrit est ancien"
        hyps = {"perfect": ref}
        assert oracle_token_recall(ref, hyps) == pytest.approx(1.0)

    def test_no_engine_recovers_anything(self) -> None:
        ref = "alpha beta gamma"
        hyps = {"a": "x y z", "b": "x y z"}
        assert oracle_token_recall(ref, hyps) == pytest.approx(0.0)

    def test_complementarity_pays_off(self) -> None:
        """A et B se complètent : aucun ne fait tout, ensemble ils font tout."""
        ref = "alpha beta gamma delta"
        hyps = {
            "a": "alpha beta x y",       # alpha + beta seulement
            "b": "x y gamma delta",      # gamma + delta seulement
        }
        assert oracle_token_recall(ref, hyps) == pytest.approx(1.0)
        # Et chacun seul ne fait que la moitié
        from picarones.measurements.inter_engine import complementarity_gap
        gap = complementarity_gap(ref, hyps)
        assert gap["best_single_recall"] == pytest.approx(0.5)
        assert gap["oracle_recall"] == pytest.approx(1.0)
        assert gap["absolute_gap"] == pytest.approx(0.5)
        # Tout l'écart restant est récupérable → relative_gap = 1
        assert gap["relative_gap"] == pytest.approx(1.0)

    def test_multiplicity_is_respected(self) -> None:
        """Si la GT a deux 'le' et le moteur n'en produit qu'un, recall = 0.5
        sur ce token."""
        ref = "le chat le chien"  # 2× 'le', 1× 'chat', 1× 'chien'
        hyps = {"a": "le chat le chien"}  # parfait
        assert oracle_token_recall(ref, hyps) == pytest.approx(1.0)
        hyps2 = {"a": "le chat chien"}  # un seul 'le'
        assert oracle_token_recall(ref, hyps2) == pytest.approx(3 / 4)

    def test_empty_reference_returns_one(self) -> None:
        assert oracle_token_recall("", {"a": "anything"}) == pytest.approx(1.0)

    def test_no_hypotheses_returns_zero(self) -> None:
        assert oracle_token_recall("alpha", {}) == pytest.approx(0.0)

    def test_oracle_is_at_least_best_single(self) -> None:
        """Invariant : l'oracle est toujours ≥ au meilleur moteur seul."""
        ref = "alpha beta gamma delta epsilon"
        hyps = {
            "a": "alpha beta gamma x y",
            "b": "alpha x gamma delta z",
            "c": "x y z delta epsilon",
        }
        gap = complementarity_gap(ref, hyps)
        assert gap["oracle_recall"] >= gap["best_single_recall"]


# ──────────────────────────────────────────────────────────────────────────
# 5. Gap et désaccord par paire
# ──────────────────────────────────────────────────────────────────────────


class TestComplementarityGap:
    def test_no_gap_when_engines_are_redundant(self) -> None:
        ref = "alpha beta gamma"
        hyps = {"a": "alpha beta x", "b": "alpha beta x"}  # redondants
        gap = complementarity_gap(ref, hyps)
        # Les deux ratent le même token → oracle = best_single
        assert gap["absolute_gap"] == pytest.approx(0.0)
        assert gap["relative_gap"] == pytest.approx(0.0)

    def test_best_engine_named(self) -> None:
        ref = "alpha beta gamma"
        hyps = {
            "tesseract": "alpha x x",  # 1/3
            "pero": "alpha beta x",    # 2/3
        }
        gap = complementarity_gap(ref, hyps)
        assert gap["best_engine"] == "pero"

    def test_empty_reference(self) -> None:
        gap = complementarity_gap("", {"a": "anything"})
        assert gap["oracle_recall"] == 1.0
        assert gap["best_single_recall"] == 1.0
        assert gap["absolute_gap"] == 0.0


class TestPairwiseDisagreement:
    def test_identical_hypotheses_zero_disagreement(self) -> None:
        ref = "alpha beta gamma"
        h = "alpha beta x"
        assert pairwise_disagreement_rate(ref, h, h) == pytest.approx(0.0)

    def test_complete_disagreement_when_complementary(self) -> None:
        ref = "alpha beta"
        # A préserve alpha, B préserve beta — désaccord sur les deux
        rate = pairwise_disagreement_rate(ref, "alpha x", "x beta")
        assert rate == pytest.approx(1.0)

    def test_empty_reference_returns_zero(self) -> None:
        assert pairwise_disagreement_rate("", "x", "y") == 0.0
