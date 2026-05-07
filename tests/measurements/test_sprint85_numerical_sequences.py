"""Tests Sprint 85 — A.II.5b : précision sur séquences numériques.

Couvre :

1. Détection par catégorie (year, roman, foliation, currency, regnal).
2. ``compute_numerical_sequence_metrics`` :
   - identité → 1.0 sur strict et value
   - perte totale → 0.0
   - GT vide → scores 0.0 (pas None — convention float)
   - value préservée mais pas strict (XIV → 14)
   - foliotation recto/verso non interchangeables
   - multiplicité respectée
3. Cas réalistes : charte XVIII, registre paroissial.
4. Enregistrement registre typé : strict + value.
"""

from __future__ import annotations

from picarones.evaluation.metrics.numerical_sequences import (
    CATEGORIES,
    _detect_currencies,
    _detect_foliations,
    _detect_regnal,
    _detect_romans_with_values,
    _detect_years,
    compute_numerical_sequence_metrics,
    numerical_sequence_strict_score,
    numerical_sequence_value_score,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. Détection par catégorie
# ──────────────────────────────────────────────────────────────────────────


class TestDetectYears:
    def test_classic_year(self) -> None:
        years = _detect_years("né en 1789 puis mort en 1856")
        assert years == [("1789", 1789), ("1856", 1856)]

    def test_year_with_context(self) -> None:
        years = _detect_years("1ᵉʳ janvier 1789")
        assert years == [("1789", 1789)]

    def test_outside_range_ignored(self) -> None:
        # 999 (3 chiffres) et 2123 (au-delà 2099) : non détectés
        assert _detect_years("999 et 2123") == []

    def test_empty(self) -> None:
        assert _detect_years("") == []


class TestDetectRomans:
    def test_classic(self) -> None:
        out = _detect_romans_with_values("Tome IV, MDCLXVIII")
        forms = [f for f, _ in out]
        assert "IV" in forms
        assert "MDCLXVIII" in forms

    def test_min_length_filters_single_letter(self) -> None:
        # I, V, X seuls → ignorés (min_length=2)
        out = _detect_romans_with_values("I prononce le V")
        forms = [f for f, _ in out]
        assert "I" not in forms


class TestDetectFoliations:
    def test_recto_verso_preserved(self) -> None:
        out = _detect_foliations("voir f. 12r et f. 12v")
        keys = [k for _, k in out]
        assert "12r" in keys
        assert "12v" in keys

    def test_page_range(self) -> None:
        out = _detect_foliations("pp. 12-15")
        assert ("pp. 12-15", "12-15") in out

    def test_n_degree(self) -> None:
        out = _detect_foliations("voir n° 42")
        assert any(k == "42" for _, k in out)


class TestDetectCurrencies:
    def test_ancien_regime(self) -> None:
        out = _detect_currencies("12 livres 5 sols 8 deniers")
        units = [v[1] for _, v in out]
        assert "livre" in units
        assert "sol" in units
        assert "denier" in units

    def test_modern_units(self) -> None:
        out = _detect_currencies("100 £ et 50 €")
        units = [v[1] for _, v in out]
        assert "£" in units
        assert "€" in units


class TestDetectRegnal:
    def test_simple_regnal(self) -> None:
        out = _detect_regnal("l'an III de la République")
        # « l'an III » + « an III de la République » fusionnés en une
        # seule occurrence par le regex
        assert any(v == 3 for _, v in out)

    def test_an_de_grace(self) -> None:
        out = _detect_regnal("écrit en l'an de grâce 1450")
        assert any(v == 1450 for _, v in out)


# ──────────────────────────────────────────────────────────────────────────
# 2. compute_numerical_sequence_metrics
# ──────────────────────────────────────────────────────────────────────────


class TestComputeMetrics:
    def test_identity(self) -> None:
        gt = "Tome IV, an de grâce 1789, f. 12r, 5 livres"
        r = compute_numerical_sequence_metrics(gt, gt)
        assert r["global_strict_score"] == 1.0
        assert r["global_value_score"] == 1.0

    def test_total_loss(self) -> None:
        gt = "1789 IV f. 12r 5 livres"
        hyp = "alpha beta gamma delta"
        r = compute_numerical_sequence_metrics(gt, hyp)
        assert r["global_strict_score"] == 0.0
        assert r["global_value_score"] == 0.0
        assert r["n_total"] >= 1

    def test_empty_gt_returns_zero(self) -> None:
        r = compute_numerical_sequence_metrics("", "anything")
        # Pas de séquence en GT → scores 0 (pas de division par 0)
        assert r["global_strict_score"] == 0.0
        assert r["global_value_score"] == 0.0
        assert r["n_total"] == 0

    def test_value_preserved_form_lost(self) -> None:
        # « XIV » en GT ; hypothèse contient « 14 » en année
        # (impossible ici car 14 < 1000 et hors plage years).
        # Cas plus robuste : « MMXX » (2020) → hyp « 2020 ».
        # Mais value_extractor de roman_numerals attend un int
        # romain — si hypothesis n'a pas « MMXX » mais bien
        # « 2020 », le détecteur roman ne trouve rien, donc
        # le roman GT est lost en valeur aussi (cohérent : on
        # ne fait pas de cross-category match).
        # On teste donc le mode strict vs value sur foliotation :
        gt = "voir f. 12r"
        hyp = "voir fol. 12r"   # forme différente, valeur identique (12r)
        r = compute_numerical_sequence_metrics(gt, hyp)
        # « f. 12r » et « fol. 12r » ont la même clé de valeur
        # (« 12r »), donc value=1, strict=0
        assert r["per_category"]["foliation"]["value"] == 1
        assert r["per_category"]["foliation"]["strict"] == 0

    def test_recto_verso_not_interchangeable(self) -> None:
        # f. 12r (GT) et f. 12v (hyp) : recto/verso différents,
        # donc lost en value et en strict
        r = compute_numerical_sequence_metrics("f. 12r", "f. 12v")
        assert r["per_category"]["foliation"]["strict"] == 0
        assert r["per_category"]["foliation"]["value"] == 0

    def test_multiplicity(self) -> None:
        # 2 occurrences en GT, 1 en hyp → 1 préservée
        gt = "1789 et 1789"
        hyp = "1789"
        r = compute_numerical_sequence_metrics(gt, hyp)
        assert r["per_category"]["year"]["n_total"] == 2
        assert r["per_category"]["year"]["strict"] == 1
        assert "1789" in r["per_category"]["year"]["lost_items"]

    def test_categories_constant(self) -> None:
        # Sanity : les 5 catégories sont déclarées
        assert set(CATEGORIES) == {
            "year", "roman", "foliation", "currency", "regnal",
        }

    def test_per_category_breakdown_keys(self) -> None:
        r = compute_numerical_sequence_metrics("1789", "1789")
        for cat in CATEGORIES:
            assert cat in r["per_category"]
            for k in (
                "n_total", "strict", "value",
                "strict_score", "value_score", "lost_items",
            ):
                assert k in r["per_category"][cat]


# ──────────────────────────────────────────────────────────────────────────
# 3. Cas réalistes
# ──────────────────────────────────────────────────────────────────────────


class TestRealistic:
    def test_charte_18e_strict_preserved(self) -> None:
        gt = (
            "Donné à Paris l'an de grâce 1789, "
            "f. 12r, contre 25 livres 4 sols et 6 deniers."
        )
        hyp = (
            "Donné à Paris l'an de grâce 1789, "
            "f. 12r, contre 25 livres 4 sols et 6 deniers."
        )
        r = compute_numerical_sequence_metrics(gt, hyp)
        assert r["global_strict_score"] == 1.0

    def test_baptismal_register_modernized(self) -> None:
        # OCR modernisant : XVIII → 18 (forme romaine perdue)
        gt = "Au siècle XVIII, en l'an 1750, f. 3r"
        hyp = "Au siècle 18, en l'an 1750, f. 3r"
        r = compute_numerical_sequence_metrics(gt, hyp)
        # XVIII forme perdue (le hyp n'a pas un romain reconnaissable)
        assert "XVIII" in r["per_category"]["roman"]["lost_items"]
        # Année et foliation préservées
        assert r["per_category"]["year"]["strict"] == 1
        assert r["per_category"]["foliation"]["strict"] == 1


# ──────────────────────────────────────────────────────────────────────────
# 4. Registre typé
# ──────────────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_strict_and_value_metrics_registered(self) -> None:
        from picarones.core.metric_registry import select_metrics
        from picarones.core.modules import ArtifactType

        metrics = select_metrics((ArtifactType.TEXT, ArtifactType.TEXT))
        names = [m.name for m in metrics]
        assert "numerical_sequence_strict_score" in names
        assert "numerical_sequence_value_score" in names

    def test_strict_score_callable(self) -> None:
        v = numerical_sequence_strict_score("1789", "1789")
        assert v == 1.0

    def test_value_score_with_form_drift(self) -> None:
        # « f. 12r » vs « fol. 12r » : value préservée, strict perdu
        strict = numerical_sequence_strict_score("f. 12r", "fol. 12r")
        value = numerical_sequence_value_score("f. 12r", "fol. 12r")
        assert strict == 0.0
        assert value == 1.0

    def test_metric_via_compute_at_junction(self) -> None:
        from picarones.core.metric_registry import compute_at_junction
        from picarones.core.modules import ArtifactType

        results = compute_at_junction(
            "1789, IV", "1789, IV",
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        assert results.get("numerical_sequence_strict_score") == 1.0
        assert results.get("numerical_sequence_value_score") == 1.0
