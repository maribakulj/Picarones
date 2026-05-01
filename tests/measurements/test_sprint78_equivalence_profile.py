"""Tests Sprint 78 — A.I.5 : équivalences diplomatiques en curseur fin.

Couvre :

1. Catalogue ``BUILTIN_EQUIVALENCES`` :
   - Au moins une règle par profil
   - Règles canoniques nommées (longs_s, u_eq_v, etc.)
   - Pas de noms doublons
2. ``list_equivalences_by_profile`` :
   - Sans filtre → toutes les règles
   - Avec filtre → règles du profil seulement
3. ``apply_selected_equivalences`` :
   - Application sélective : seul ``longs_s`` actif → ſ→s mais
     pas u→v
   - Liste vide → pas de changement
   - Texte vide / None → ``""``
   - Règle inconnue silencieusement ignorée
4. ``compute_cer_with_equivalences`` :
   - Sans équivalences : CER élevé
   - Avec les bonnes équivalences : CER baisse
   - Application bilatérale (GT et hyp)
"""

from __future__ import annotations

from picarones.measurements.equivalence_profile import (
    BUILTIN_EQUIVALENCES,
    EquivalenceRule,
    apply_selected_equivalences,
    compute_cer_with_equivalences,
    list_equivalences_by_profile,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. Catalogue
# ──────────────────────────────────────────────────────────────────────────


class TestCatalog:
    def test_canonical_rules_present(self) -> None:
        for name in (
            "longs_s", "u_eq_v", "i_eq_j", "y_eq_i", "vv_eq_w",
            "ae_ligature", "oe_ligature", "thorn_th", "eth_th",
            "yogh_y",
        ):
            assert name in BUILTIN_EQUIVALENCES, (
                f"règle canonique manquante : {name}"
            )

    def test_rule_structure(self) -> None:
        for rule in BUILTIN_EQUIVALENCES.values():
            assert isinstance(rule, EquivalenceRule)
            assert rule.name
            assert rule.source
            assert rule.target
            assert rule.description
            assert rule.profile_tag

    def test_unique_names(self) -> None:
        names = list(BUILTIN_EQUIVALENCES.keys())
        assert len(names) == len(set(names))

    def test_longs_s_correct(self) -> None:
        rule = BUILTIN_EQUIVALENCES["longs_s"]
        assert rule.source == "ſ"
        assert rule.target == "s"


# ──────────────────────────────────────────────────────────────────────────
# 2. list_equivalences_by_profile
# ──────────────────────────────────────────────────────────────────────────


class TestListByProfile:
    def test_no_filter_returns_all(self) -> None:
        all_rules = list_equivalences_by_profile()
        assert len(all_rules) == len(BUILTIN_EQUIVALENCES)

    def test_filter_by_medieval_french(self) -> None:
        rules = list_equivalences_by_profile("medieval_french")
        assert all(r.profile_tag == "medieval_french" for r in rules)
        assert len(rules) > 0

    def test_unknown_profile_returns_empty(self) -> None:
        rules = list_equivalences_by_profile("nonexistent")
        assert rules == []


# ──────────────────────────────────────────────────────────────────────────
# 3. apply_selected_equivalences
# ──────────────────────────────────────────────────────────────────────────


class TestApply:
    def test_selective_longs_s_only(self) -> None:
        result = apply_selected_equivalences("ſeparare", ["longs_s"])
        assert result == "separare"

    def test_selective_excludes_unselected(self) -> None:
        # u_eq_v non sélectionné → "u" doit rester
        result = apply_selected_equivalences("ſupra", ["longs_s"])
        assert result == "supra"

    def test_multiple_selected(self) -> None:
        # Avec plusieurs règles, toutes appliquées
        result = apply_selected_equivalences(
            "ſupra", ["longs_s", "u_eq_v"],
        )
        # ſ→s puis u→v → "svpra"
        assert "ſ" not in result

    def test_empty_selection_unchanged(self) -> None:
        assert apply_selected_equivalences("ſeparare", []) == "ſeparare"

    def test_empty_text(self) -> None:
        assert apply_selected_equivalences("", ["longs_s"]) == ""
        assert apply_selected_equivalences(None, ["longs_s"]) == ""

    def test_unknown_rule_ignored(self, caplog) -> None:
        result = apply_selected_equivalences(
            "ſeparare", ["longs_s", "nonexistent_rule"],
        )
        # longs_s appliqué, règle inconnue ignorée
        assert result == "separare"


# ──────────────────────────────────────────────────────────────────────────
# 4. compute_cer_with_equivalences
# ──────────────────────────────────────────────────────────────────────────


class TestComputeCer:
    def test_cer_drops_with_equivalences(self) -> None:
        gt = "ſeparare"
        hyp = "separare"
        cer_no_eq = compute_cer_with_equivalences(gt, hyp, [])
        cer_with_eq = compute_cer_with_equivalences(gt, hyp, ["longs_s"])
        assert cer_no_eq > 0
        assert cer_with_eq == 0.0

    def test_bilateral_application(self) -> None:
        # Les deux côtés sont normalisés : si gt et hyp se
        # neutralisent par la règle, CER = 0
        gt = "ſupra"   # avec ſ
        hyp = "ſupra"  # avec ſ aussi
        cer = compute_cer_with_equivalences(gt, hyp, ["longs_s"])
        assert cer == 0.0

    def test_unrelated_diff_remains(self) -> None:
        # Différence indépendante des équivalences sélectionnées
        gt = "ſalpha"
        hyp = "ſbeta"
        cer = compute_cer_with_equivalences(gt, hyp, ["longs_s"])
        # ſ → s appliqué aux deux : "salpha" vs "sbeta" → CER > 0
        assert cer > 0

    def test_empty_inputs(self) -> None:
        assert compute_cer_with_equivalences("", "", ["longs_s"]) == 0.0
