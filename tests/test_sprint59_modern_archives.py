"""Tests Sprint 59 — abréviations et marqueurs des archives
modernes XIXᵉ-XXᵉ.

Couvre :

1. ``get_category`` : marqueurs des 9 catégories ; marqueurs
   inconnus → ``None``.
2. ``detect_modern_markers`` :
   - reconnaissance par catégorie
   - greedy « plus long gagne » (S.A.R. avant S.A.)
   - frontière de mot pour les abréviations courtes
   - ordre préservé
   - texte vide / None
3. ``compute_modern_archives_metrics`` :
   - **Diplomatique** : tous marqueurs préservés → strict = expansion = 1
   - **Modernisant** : abrégés remplacés par formes développées →
     strict = 0, expansion = 1
   - **Erreur** : signaux faibles partout
   - **Mixte** : breakdown per_category cohérent
   - cas dégénérés (GT sans marqueur, vide, None)
4. ``missed_markers`` : entrées avec index, marker, category,
   expansion_preserved.
5. **Cas réalistes** par catégorie :
   - Notice biblio : « vol. II p. 45 » → modernisant le développe.
   - État civil : « ép. Martin, vve Durand » → discriminant.
   - Adresse : « bd Voltaire, arr. XIᵉ ».
   - Politesse : « S.A.R. le duc » vs « Son Altesse Royale ».
   - Monnaie : « 100 ₶ 5 s. 6 d. ».
6. Comptage exhaustif : ``n_strict_preserved + len(missed_markers
   non-expansion) + cas mixtes`` cohérent.
7. Intégration registre typé.
"""

from __future__ import annotations

import pytest

from picarones.core.metric_registry import compute_at_junction, select_metrics
from picarones.measurements.modern_archives import (
    ADDRESS,
    ADMINISTRATIVE,
    BIBLIOGRAPHIC,
    CIVIL_STATUS,
    CIVILITY_TITLES,
    CURRENCY,
    LATIN_ABBR_MODERN,
    ORDINALS,
    TYPOGRAPHIC_PUNCTUATION,
    compute_modern_archives_metrics,
    detect_modern_markers,
    get_category,
    get_expansions,
    modern_archives_expansion_score,
    modern_archives_strict_score,
)
from picarones.core.modules import ArtifactType


# ──────────────────────────────────────────────────────────────────────────
# 1. get_category
# ──────────────────────────────────────────────────────────────────────────


class TestGetCategory:
    @pytest.mark.parametrize(
        "marker,expected",
        [
            # civility_titles
            ("Mme", "civility_titles"),
            ("Mlle", "civility_titles"),
            ("Mgr", "civility_titles"),
            ("Dr", "civility_titles"),
            ("M.", "civility_titles"),
            ("S.A.R.", "civility_titles"),
            # ordinals
            ("1ᵉʳ", "ordinals"),
            ("1ʳᵉ", "ordinals"),
            ("XIXᵉ", "ordinals"),
            # currency
            ("₶", "currency"),
            ("£", "currency"),
            ("d.", "currency"),
            # administrative
            ("arr.", "administrative"),
            ("dép.", "administrative"),
            # civil_status
            ("°", "civil_status"),
            ("†", "civil_status"),
            ("ép.", "civil_status"),
            ("vve", "civil_status"),
            # typographic_punctuation
            ("«", "typographic_punctuation"),
            ("—", "typographic_punctuation"),
            ("…", "typographic_punctuation"),
            # latin_abbr_modern
            ("e.g.", "latin_abbr_modern"),
            ("etc.", "latin_abbr_modern"),
            ("op. cit.", "latin_abbr_modern"),
            # bibliographic
            ("vol.", "bibliographic"),
            ("p.", "bibliographic"),
            ("n°", "bibliographic"),
            ("r°", "bibliographic"),
            # address
            ("bd", "address"),
            ("av.", "address"),
            ("fbg", "address"),
            # inconnu
            ("xyz", None),
            ("", None),
        ],
    )
    def test_categorize(self, marker: str, expected: str | None) -> None:
        assert get_category(marker) == expected


# ──────────────────────────────────────────────────────────────────────────
# 2. get_expansions
# ──────────────────────────────────────────────────────────────────────────


class TestGetExpansions:
    def test_known_marker(self) -> None:
        assert "Madame" in get_expansions("Mme")
        assert "boulevard" in get_expansions("bd")
        assert "page" in get_expansions("p.")

    def test_unknown_marker(self) -> None:
        assert get_expansions("xyz") == ()
        assert get_expansions("") == ()


# ──────────────────────────────────────────────────────────────────────────
# 3. detect_modern_markers
# ──────────────────────────────────────────────────────────────────────────


class TestDetectMarkers:
    def test_detects_civility(self) -> None:
        markers = detect_modern_markers("Mme Dupont et Mgr Martin")
        cats = sorted({cat for _i, _m, cat in markers})
        assert cats == ["civility_titles"]
        names = sorted({m for _i, m, _c in markers})
        assert names == ["Mgr", "Mme"]

    def test_detects_ordinals(self) -> None:
        markers = detect_modern_markers("le 1ᵉʳ et le XIXᵉ siècle, 3ᵉ étage")
        cats = [cat for _i, _m, cat in markers]
        assert all(c == "ordinals" for c in cats)
        assert len(cats) == 3

    def test_detects_currency(self) -> None:
        markers = detect_modern_markers("100 ₶ 5 s. 6 d. et 50 £")
        cats = sorted({cat for _i, _m, cat in markers})
        assert cats == ["currency"]
        assert len(markers) == 4

    def test_detects_civil_status(self) -> None:
        markers = detect_modern_markers("° 1850 † 1920 ép. Durand vve")
        cats = sorted({cat for _i, _m, cat in markers})
        assert cats == ["civil_status"]

    def test_detects_typographic_punctuation(self) -> None:
        markers = detect_modern_markers("« voici » — pas mal…")
        cats = sorted({cat for _i, _m, cat in markers})
        assert cats == ["typographic_punctuation"]

    def test_detects_latin_abbr(self) -> None:
        markers = detect_modern_markers("cf. p. 12, etc. ; ibid., op. cit.")
        cats = {cat for _i, _m, cat in markers}
        assert "latin_abbr_modern" in cats
        # « cf. », « etc. », « ibid. », « op. cit. » → 4 latins
        latin = [m for _i, m, c in markers if c == "latin_abbr_modern"]
        assert sorted(latin) == ["cf.", "etc.", "ibid.", "op. cit."]

    def test_detects_bibliographic(self) -> None:
        markers = detect_modern_markers("vol. II t. 3 p. 12 pp. 12 fasc. 4 n° 7")
        cats = {cat for _i, _m, cat in markers}
        assert "bibliographic" in cats

    def test_detects_address(self) -> None:
        markers = detect_modern_markers("bd Voltaire, av. de l'Opéra, r. de Rivoli")
        names = sorted({m for _i, m, c in markers if c == "address"})
        assert names == ["av.", "bd", "r."]

    def test_greedy_longest_wins(self) -> None:
        # « S.A.R. » doit gagner sur « S.M. » — ce sont deux marqueurs
        # distincts, mais la stratégie greedy garantit qu'on ne
        # détecte pas « S. " ou « A.R. " séparément.
        markers = detect_modern_markers("S.A.R. le duc")
        names = [m for _i, m, _c in markers]
        assert names == ["S.A.R."]

    def test_word_boundary_for_short_abbr(self) -> None:
        # « M. » dans « M.A.M. » ne doit PAS être détecté (pas de
        # frontière espace/fin/ponctuation après le point final
        # avant un autre caractère mot).
        # Cas positif : « M. Dupont » → 1 détection
        markers_pos = detect_modern_markers("M. Dupont")
        m_titles = [m for _i, m, c in markers_pos if c == "civility_titles"]
        assert "M." in m_titles
        # Cas litigieux : « M.A.M. » ne doit pas matcher 3 fois
        markers_neg = detect_modern_markers("M.A.M.")
        m_negs = [m for _i, m, c in markers_neg if c == "civility_titles"]
        # Au plus le dernier « M. » avec point final accepté
        assert m_negs.count("M.") <= 1

    def test_word_boundary_blocks_false_positive(self) -> None:
        # « bd » dans « abdomen » ne doit pas matcher (pas en
        # frontière de mot).
        markers = detect_modern_markers("son abdomen est gonflé")
        assert all(m != "bd" for _i, m, _c in markers)

    def test_preserves_order(self) -> None:
        markers = detect_modern_markers("Mme au 3ᵉ étage du bd Voltaire")
        names = [m for _i, m, _c in markers]
        assert names == ["Mme", "3ᵉ", "bd"]

    def test_empty_input(self) -> None:
        assert detect_modern_markers("") == []
        assert detect_modern_markers(None) == []

    def test_text_without_markers(self) -> None:
        assert detect_modern_markers("hello world without abbreviations") == []


# ──────────────────────────────────────────────────────────────────────────
# 4. compute_modern_archives_metrics — scénarios standards
# ──────────────────────────────────────────────────────────────────────────


class TestComputeMetrics:
    @pytest.fixture
    def gt(self) -> str:
        return "Mme Dupont S.A.R. ép. au bd Voltaire vol. II p. 45"

    def test_diplomatic_full_preservation(self, gt: str) -> None:
        m = compute_modern_archives_metrics(gt, gt)
        assert m["global_strict_score"] == pytest.approx(1.0)
        assert m["global_expansion_score"] == pytest.approx(1.0)
        assert m["missed_markers"] == []

    def test_modernizing_loses_strict_keeps_expansion(self, gt: str) -> None:
        # Toutes les abréviations sont développées
        hyp = (
            "Madame Dupont Son Altesse Royale épouse au boulevard "
            "Voltaire volume II page 45"
        )
        m = compute_modern_archives_metrics(gt, hyp)
        assert m["global_strict_score"] == pytest.approx(0.0)
        assert m["global_expansion_score"] == pytest.approx(1.0)

    def test_erroneous_loses_both(self, gt: str) -> None:
        # On ne préserve ni l'abrégé ni le développé
        hyp = "Femme Dupont l'altesse au quai Voltaire."
        m = compute_modern_archives_metrics(gt, hyp)
        assert m["global_strict_score"] == pytest.approx(0.0)
        assert m["global_expansion_score"] < 0.5

    def test_mixed_per_category(self) -> None:
        gt = "Mme Dupont au bd Voltaire vol. II"
        # Préserve civility + bibliographic, perd address
        hyp = "Mme Dupont au boulevard Voltaire vol. II"
        m = compute_modern_archives_metrics(gt, hyp)
        assert m["per_category"]["civility_titles"]["strict_score"] == 1.0
        assert m["per_category"]["bibliographic"]["strict_score"] == 1.0
        assert m["per_category"]["address"]["strict_score"] == 0.0
        # Address : bd → boulevard, donc expansion satisfaite
        assert m["per_category"]["address"]["expansion_score"] == 1.0


# ──────────────────────────────────────────────────────────────────────────
# 5. Cas dégénérés
# ──────────────────────────────────────────────────────────────────────────


class TestDegenerateCases:
    def test_gt_without_markers(self) -> None:
        m = compute_modern_archives_metrics("hello world", "hello world")
        assert m["n_markers_reference"] == 0
        assert m["global_strict_score"] == 0.0
        assert m["global_expansion_score"] == 0.0
        assert m["per_category"] == {}

    def test_empty_gt(self) -> None:
        m = compute_modern_archives_metrics("", "anything")
        assert m["n_markers_reference"] == 0

    def test_none_inputs(self) -> None:
        m = compute_modern_archives_metrics(None, None)
        assert m["n_markers_reference"] == 0

    def test_empty_hyp_with_markers_in_gt(self) -> None:
        m = compute_modern_archives_metrics("Mme bd vol.", "")
        assert m["n_strict_preserved"] == 0
        assert m["global_strict_score"] == 0.0
        assert len(m["missed_markers"]) == 3
        for entry in m["missed_markers"]:
            assert entry["expansion_preserved"] is False


# ──────────────────────────────────────────────────────────────────────────
# 6. missed_markers
# ──────────────────────────────────────────────────────────────────────────


class TestMissedMarkers:
    def test_missed_markers_have_required_fields(self) -> None:
        gt = "Mme bd vol."
        hyp = "Madame boulevard volume"
        m = compute_modern_archives_metrics(gt, hyp)
        # Modernisant : tous strict ratés mais expansion préservée
        assert len(m["missed_markers"]) == 3
        for entry in m["missed_markers"]:
            assert "index" in entry
            assert "marker" in entry
            assert "category" in entry
            assert "expansion_preserved" in entry
            assert entry["expansion_preserved"] is True

    def test_missed_marker_distinguishes_pure_loss(self) -> None:
        gt = "Mme Dupont au bd Voltaire"
        # Préserve « bd Voltaire » mais perd « Mme » sans le développer
        hyp = "Femme Dupont au bd Voltaire"
        m = compute_modern_archives_metrics(gt, hyp)
        # Mme : ni abrégé ni développé → expansion_preserved = False
        mme_missed = [e for e in m["missed_markers"] if e["marker"] == "Mme"]
        assert len(mme_missed) == 1
        assert mme_missed[0]["expansion_preserved"] is False


# ──────────────────────────────────────────────────────────────────────────
# 7. Cas réalistes par catégorie
# ──────────────────────────────────────────────────────────────────────────


class TestRealisticBibliographicCitation:
    def test_diplomatic_vs_modernizing(self) -> None:
        gt = "Voir vol. II t. 3 p. 45 pp. 50-60 fasc. 4 n° 7"
        hyp_modern = (
            "Voir volume II tome 3 page 45 pages 50-60 fascicule 4 numéro 7"
        )
        m_diplo = compute_modern_archives_metrics(gt, gt)
        m_mod = compute_modern_archives_metrics(gt, hyp_modern)
        assert m_diplo["per_category"]["bibliographic"]["strict_score"] == 1.0
        assert m_mod["per_category"]["bibliographic"]["strict_score"] == 0.0
        assert m_mod["per_category"]["bibliographic"]["expansion_score"] == 1.0


class TestRealisticVitalRecord:
    def test_vital_record_discriminates(self) -> None:
        gt = "Marie Dupont, ép. Martin, vve Durand, † 1920"
        hyp_modern = (
            "Marie Dupont, épouse Martin, veuve Durand, décédée 1920"
        )
        m_diplo = compute_modern_archives_metrics(gt, gt)
        m_mod = compute_modern_archives_metrics(gt, hyp_modern)
        assert m_diplo["per_category"]["civil_status"]["strict_score"] == 1.0
        assert m_mod["per_category"]["civil_status"]["strict_score"] == 0.0
        assert m_mod["per_category"]["civil_status"]["expansion_score"] == 1.0


class TestRealisticAddress:
    def test_address_typical_modernization(self) -> None:
        gt = "demeurant 14 bd Voltaire, arr. XIᵉ"
        hyp_modern = "demeurant 14 boulevard Voltaire, arrondissement XIe"
        m = compute_modern_archives_metrics(gt, hyp_modern)
        # bd → boulevard (expansion ok), arr. → arrondissement (ok)
        assert m["per_category"]["address"]["expansion_score"] == 1.0
        assert m["per_category"]["administrative"]["expansion_score"] == 1.0
        # XIᵉ → XIe (forme plate, expansion ok)
        assert m["per_category"]["ordinals"]["expansion_score"] == 1.0


class TestRealisticHonorific:
    def test_royal_protocol_full_preservation(self) -> None:
        gt = "S.A.R. le duc et S.M. la reine"
        m = compute_modern_archives_metrics(gt, gt)
        assert m["per_category"]["civility_titles"]["strict_score"] == 1.0

    def test_royal_protocol_modernized(self) -> None:
        gt = "S.A.R. le duc et S.M. la reine"
        hyp = "Son Altesse Royale le duc et Sa Majesté la reine"
        m = compute_modern_archives_metrics(gt, hyp)
        assert m["per_category"]["civility_titles"]["strict_score"] == 0.0
        assert m["per_category"]["civility_titles"]["expansion_score"] == 1.0


class TestRealisticCurrency:
    def test_ancien_regime_currency(self) -> None:
        gt = "100 ₶ 5 s. 6 d."
        m = compute_modern_archives_metrics(gt, gt)
        assert m["per_category"]["currency"]["strict_score"] == 1.0
        # Modernisant : développement complet
        hyp = "100 livres tournois 5 sous 6 deniers"
        m_mod = compute_modern_archives_metrics(gt, hyp)
        assert m_mod["per_category"]["currency"]["strict_score"] == 0.0
        assert m_mod["per_category"]["currency"]["expansion_score"] == 1.0


class TestRealisticTypographicPunctuation:
    def test_quotation_typographic_vs_ascii(self) -> None:
        gt = "il dit « bonjour » et — bien sûr — sortit…"
        hyp_ascii = 'il dit "bonjour" et - bien sûr - sortit...'
        m = compute_modern_archives_metrics(gt, hyp_ascii)
        # Strict : aucune ponctuation typographique préservée
        ptyp = m["per_category"]["typographic_punctuation"]
        assert ptyp["strict_score"] == 0.0
        # Expansion : ASCII équivalents acceptés
        assert ptyp["expansion_score"] == 1.0


# ──────────────────────────────────────────────────────────────────────────
# 8. Comptage exhaustif
# ──────────────────────────────────────────────────────────────────────────


class TestExhaustiveAccounting:
    def test_strict_plus_strict_missed_equals_total(self) -> None:
        gt = "Mme au bd Voltaire vol. II ép. Martin"
        hyp = "Mme au boulevard Voltaire vol. II épouse Martin"
        m = compute_modern_archives_metrics(gt, hyp)
        assert (
            m["n_strict_preserved"] + len(m["missed_markers"])
            == m["n_markers_reference"]
        )

    def test_per_category_counts_consistent(self) -> None:
        gt = "Mme bd vol. II ép. Martin"
        hyp = "Mme boulevard volume II ép. Martin"
        m = compute_modern_archives_metrics(gt, hyp)
        for _cat, scores in m["per_category"].items():
            assert scores["n_strict_preserved"] <= scores["n_total"]
            assert scores["n_expansion_preserved"] <= scores["n_total"]
            # Strict ⊆ expansion (un strict est aussi une expansion)
            assert scores["n_strict_preserved"] <= scores["n_expansion_preserved"]


# ──────────────────────────────────────────────────────────────────────────
# 9. Tables exposées
# ──────────────────────────────────────────────────────────────────────────


class TestExposedTables:
    def test_all_categories_non_empty(self) -> None:
        for table in (
            CIVILITY_TITLES, ORDINALS, CURRENCY, ADMINISTRATIVE,
            CIVIL_STATUS, TYPOGRAPHIC_PUNCTUATION, LATIN_ABBR_MODERN,
            BIBLIOGRAPHIC, ADDRESS,
        ):
            assert len(table) >= 1

    def test_table_entries_well_formed(self) -> None:
        # Chaque entrée : (marker_str, expansions_tuple)
        for table in (
            CIVILITY_TITLES, ORDINALS, CURRENCY, ADMINISTRATIVE,
            CIVIL_STATUS, TYPOGRAPHIC_PUNCTUATION, LATIN_ABBR_MODERN,
            BIBLIOGRAPHIC, ADDRESS,
        ):
            for entry in table:
                assert len(entry) == 2
                marker, expansions = entry
                assert isinstance(marker, str) and marker
                assert isinstance(expansions, tuple)
                # Chaque expansion non vide
                for exp in expansions:
                    assert isinstance(exp, str) and exp


# ──────────────────────────────────────────────────────────────────────────
# 10. Raccourcis
# ──────────────────────────────────────────────────────────────────────────


class TestShortcuts:
    def test_strict_shortcut_matches_full_call(self) -> None:
        gt = "Mme au bd Voltaire"
        hyp = "Madame au boulevard Voltaire"
        full = compute_modern_archives_metrics(gt, hyp)
        assert modern_archives_strict_score(gt, hyp) == pytest.approx(
            full["global_strict_score"],
        )

    def test_expansion_shortcut_matches_full_call(self) -> None:
        gt = "Mme au bd Voltaire"
        hyp = "Madame au boulevard Voltaire"
        full = compute_modern_archives_metrics(gt, hyp)
        assert modern_archives_expansion_score(gt, hyp) == pytest.approx(
            full["global_expansion_score"],
        )


# ──────────────────────────────────────────────────────────────────────────
# 11. Intégration registre typé
# ──────────────────────────────────────────────────────────────────────────


class TestRegistryIntegration:
    def test_strict_metric_registered(self) -> None:
        import picarones.measurements.modern_archives  # noqa: F401

        selected = select_metrics(
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        names = {spec.name for spec in selected}
        assert "modern_archives_strict_score" in names
        assert "modern_archives_expansion_score" in names

    def test_compute_at_junction_strict(self) -> None:
        out = compute_at_junction(
            "Mme au bd", "Mme au bd",
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        assert out["modern_archives_strict_score"] == pytest.approx(1.0)

    def test_compute_at_junction_expansion(self) -> None:
        out = compute_at_junction(
            "Mme au bd", "Madame au boulevard",
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        assert out["modern_archives_strict_score"] == pytest.approx(0.0)
        assert out["modern_archives_expansion_score"] == pytest.approx(1.0)
