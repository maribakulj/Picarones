"""Précision sur séquences numériques — Sprint 85 (A.II.5b).

Sprint 85 — A.II.5b du plan d'évolution 2026.

Pourquoi ce module
------------------
Pour un économiste-historien, un éditeur de chartes ou un
archiviste, la **fidélité aux séquences numériques** est un
proxy direct de la qualité éditoriale.  Un OCR qui rate
*« 1789 »* dans une charte révolutionnaire ou *« f. 12v »*
dans une cote d'archives produit un corpus inutilisable pour la
recherche fine, même si le CER global est respectable.

Catégories couvertes
--------------------
1. **Dates arabes** : ``1789``, ``1450``, ``1ᵉʳ janvier 1789``
   (le module détecte les **années** sur 4 chiffres dans la
   plage [1000-2099]).
2. **Numéraux romains** : ``MDCLXVIII``, ``XIV``, ``Tome IV``.
   Réutilise ``picarones.measurements.roman_numerals`` (Sprint 60).
3. **Foliotation** : ``f. 12``, ``f. 12r``, ``fol. 24v``,
   ``p. 5``, ``pp. 12-15``, ``n° 42``.
4. **Montants** : ``12 livres``, ``5 sols``, ``8 deniers``,
   ``100 £``, ``50 ₣``, ``20 €``, formes Ancien Régime
   (``l.``, ``s.``, ``d.``).
5. **Années régnales** : ``an III``, ``l'an V``, ``an de
   grâce 1450``, ``an de la République``.

Méthode
-------
Pour chaque catégorie, on extrait les occurrences (regex
spécialisée) en GT et en hypothèse.  On classe ensuite chaque
GT en **3 statuts** :

- ``strict_preserved`` : forme exacte présente dans
  l'hypothèse (sensible à la casse seulement pour la
  foliotation, sinon la convention est documentée par
  catégorie) ;
- ``value_preserved`` : la **valeur** apparaît même si la
  forme diffère (ex. ``XIV`` GT et ``14`` hypothèse —
  considéré comme valeur préservée mais forme non) ;
- ``lost`` : aucune trace exploitable.

Sortie
------
``compute_numerical_sequence_metrics(reference, hypothesis)``
retourne :

```
{
    "global_strict_score": float,        # ∈ [0, 1]
    "global_value_score": float,         # ∈ [0, 1]
    "n_total": int,
    "per_category": {
        "year": {"n_total": int, "strict": int, "value": int,
                 "strict_score": float, "value_score": float,
                 "lost_items": list[str]},
        "roman": {...},
        "foliation": {...},
        "currency": {...},
        "regnal": {...},
    },
}
```

Limites
-------
- Les regex sont **conservatrices** : on rate quelques
  formes rares plutôt que de produire des faux positifs (par
  exemple, ``mil cinq cens`` en français médiéval n'est pas
  détecté comme année — la couche calcul s'en tient aux
  formes les plus reconnaissables).  Pour un corpus
  spécifique, l'utilisateur peut composer ses propres
  détecteurs et les passer via ``custom_detectors``.
- ``value_preserved`` exige une équivalence de **valeur
  numérique** : ``XIV`` ↔ ``14`` est OK pour les romains ;
  ``f. 12v`` ↔ ``f. 12r`` n'est **pas** OK pour la
  foliotation (recto/verso est une information distincte).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from picarones.core.metric_registry import register_metric
from picarones.domain.artifacts import ArtifactType
from picarones.measurements.roman_numerals import (
    detect_roman_numerals,
    roman_to_int,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Constantes / catégories
# ──────────────────────────────────────────────────────────────────────────


CATEGORIES = ("year", "roman", "foliation", "currency", "regnal")


# Dates arabes — 4 chiffres dans la plage [1000-2099].
# On exige une frontière de mot pour ne pas attraper
# « 12345 » (volume) ou « 0001 » (numéro de page).
_RE_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


# Foliotation : f. 12, f. 12r, fol. 24v, p. 5, pp. 12-15, n° 42
# La capture conserve la forme intégrale (avec ponctuation et
# r/v) parce que recto/verso est une information distincte.
_RE_FOLIATION = re.compile(
    r"\b(?:fol\.?|f\.|pp\.|p\.|n\.°|n°)\s*"  # préfixe : fol., f., pp., p., n°
    r"(\d+(?:\s*-\s*\d+)?)"                  # nombre ou plage (12 / 12-15)
    r"\s*([rvRV])?",                         # suffixe optionnel r/v
    re.UNICODE,
)


# Montants : nombre suivi d'une unité monétaire.
# On accepte espaces multiples mais pas de saut de ligne.
_RE_CURRENCY = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*"                # montant (entier ou décimal)
    r"(livres?|sols?|deniers?|écus?|florins?|francs?|"
    r"l\.|s\.|d\.|£|€|₣)"                    # unité
    r"(?=\b|[\s,;.!?:]|$)",                  # frontière souple post-symbole
    re.UNICODE | re.IGNORECASE,
)


# Années régnales : « an III », « an de grâce 1450 »,
# « l'an V de la République ».
# Capture le numéral (romain ou arabe).
_RE_REGNAL = re.compile(
    r"\b(?:l['’]\s*)?an\s+(?:de\s+(?:grâce|la\s+R[eé]publique)\s+)?"
    r"([IVXLCDMivxlcdm]+|\d{1,4})\b",
    re.UNICODE,
)


# ──────────────────────────────────────────────────────────────────────────
# Détection par catégorie
# ──────────────────────────────────────────────────────────────────────────


def _detect_years(text: str) -> list[tuple[str, int]]:
    """Retourne [(forme, valeur)] pour chaque année 4 chiffres."""
    if not text:
        return []
    return [(m.group(0), int(m.group(0))) for m in _RE_YEAR.finditer(text)]


def _detect_romans_with_values(text: str) -> list[tuple[str, int]]:
    """Numéraux romains accompagnés de leur valeur entière.
    Délègue à ``roman_numerals.detect_roman_numerals`` (Sprint 60),
    qui retourne ``(start, form, value)``.
    """
    if not text:
        return []
    out: list[tuple[str, int]] = []
    for _start, form, value in detect_roman_numerals(text, min_length=2):
        if value is not None:
            out.append((form, value))
    return out


def _detect_foliations(text: str) -> list[tuple[str, str]]:
    """Foliotation. Retourne [(forme_complète, clé_normalisée)] où la
    clé inclut le suffixe r/v normalisé (recto/verso).
    """
    if not text:
        return []
    out: list[tuple[str, str]] = []
    for m in _RE_FOLIATION.finditer(text):
        full = m.group(0).strip()
        nums = re.sub(r"\s+", "", m.group(1))  # ex : "12-15"
        suffix = (m.group(2) or "").lower()
        key = f"{nums}{suffix}"
        out.append((full, key))
    return out


def _detect_currencies(text: str) -> list[tuple[str, tuple[str, str]]]:
    """Montants. Clé = (montant_normalisé, unité_canonique).

    L'unité canonique compresse les variantes (« livres » et
    « livre » → « livre » ; « £ » reste « £ »).
    """
    if not text:
        return []
    canon = {
        "livre": "livre", "livres": "livre", "l.": "livre",
        "sol": "sol", "sols": "sol", "s.": "sol",
        "denier": "denier", "deniers": "denier", "d.": "denier",
        "écu": "écu", "écus": "écu",
        "florin": "florin", "florins": "florin",
        "franc": "franc", "francs": "franc",
        "£": "£", "€": "€", "₣": "₣",
    }
    out: list[tuple[str, tuple[str, str]]] = []
    for m in _RE_CURRENCY.finditer(text):
        amount = m.group(1).replace(",", ".")
        unit_raw = m.group(2).lower()
        unit = canon.get(unit_raw, unit_raw)
        out.append((m.group(0), (amount, unit)))
    return out


def _detect_regnal(text: str) -> list[tuple[str, int]]:
    """Années régnales. Retourne [(forme, valeur_int)] avec la
    valeur extraite (romain → int ou arabe → int).
    """
    if not text:
        return []
    out: list[tuple[str, int]] = []
    for m in _RE_REGNAL.finditer(text):
        numeral = m.group(1)
        value: Optional[int]
        if numeral.isdigit():
            value = int(numeral)
        else:
            value = roman_to_int(numeral)
        if value is not None:
            out.append((m.group(0), value))
    return out


_DETECTORS = {
    "year": _detect_years,
    "roman": _detect_romans_with_values,
    "foliation": _detect_foliations,
    "currency": _detect_currencies,
    "regnal": _detect_regnal,
}


# ──────────────────────────────────────────────────────────────────────────
# Calcul principal
# ──────────────────────────────────────────────────────────────────────────


def _classify_per_category(
    gt_items: list,
    hyp_items: list,
    *,
    form_extractor,
    value_extractor,
) -> dict:
    """Pour chaque item GT, le classe en strict_preserved /
    value_preserved / lost.

    Multiplicité respectée : un item hypothèse ne peut servir
    qu'à un seul match (forme prioritaire sur valeur).
    """
    hyp_used = [False] * len(hyp_items)
    n_strict = 0
    n_value = 0
    lost: list[str] = []
    # Première passe : matchs stricts (forme exacte)
    matched: list[bool] = [False] * len(gt_items)
    for gi, gt_item in enumerate(gt_items):
        gt_form = form_extractor(gt_item)
        for hi, hyp_item in enumerate(hyp_items):
            if hyp_used[hi]:
                continue
            if form_extractor(hyp_item) == gt_form:
                hyp_used[hi] = True
                matched[gi] = True
                n_strict += 1
                break
    # Deuxième passe : matchs sur valeur (forme différente)
    for gi, gt_item in enumerate(gt_items):
        if matched[gi]:
            n_value += 1  # strict implique value
            continue
        gt_val = value_extractor(gt_item)
        for hi, hyp_item in enumerate(hyp_items):
            if hyp_used[hi]:
                continue
            if value_extractor(hyp_item) == gt_val:
                hyp_used[hi] = True
                matched[gi] = True
                n_value += 1
                break
        if not matched[gi]:
            lost.append(form_extractor(gt_item))
    n_total = len(gt_items)
    return {
        "n_total": n_total,
        "strict": n_strict,
        "value": n_value,
        "strict_score": n_strict / n_total if n_total else 0.0,
        "value_score": n_value / n_total if n_total else 0.0,
        "lost_items": lost,
    }


def compute_numerical_sequence_metrics(
    reference: Optional[str],
    hypothesis: Optional[str],
) -> dict:
    """Calcule la précision sur séquences numériques.

    Returns
    -------
    dict
        Voir docstring du module.  Si ``reference`` est vide
        ou ne contient aucune séquence détectée, retourne
        ``{n_total: 0, ...}`` avec scores à 0 (pas None).
    """
    ref = reference or ""
    hyp = hypothesis or ""

    # Spécifications par catégorie : (gt_items, hyp_items,
    # extractor de forme, extractor de valeur).
    specs: dict[str, dict] = {}
    # year : (form="1789", value=1789)
    specs["year"] = {
        "gt": _detect_years(ref),
        "hyp": _detect_years(hyp),
        "form": lambda it: it[0],
        "value": lambda it: it[1],
    }
    # roman : (form="MDCLXVIII", value=1668)
    specs["roman"] = {
        "gt": _detect_romans_with_values(ref),
        "hyp": _detect_romans_with_values(hyp),
        "form": lambda it: it[0],
        "value": lambda it: it[1],
    }
    # foliation : (form="f. 12r", value="12r")
    specs["foliation"] = {
        "gt": _detect_foliations(ref),
        "hyp": _detect_foliations(hyp),
        "form": lambda it: it[0],
        "value": lambda it: it[1],
    }
    # currency : (form="12 livres", value=("12", "livre"))
    specs["currency"] = {
        "gt": _detect_currencies(ref),
        "hyp": _detect_currencies(hyp),
        "form": lambda it: it[0],
        "value": lambda it: it[1],
    }
    # regnal : (form="an III", value=3)
    specs["regnal"] = {
        "gt": _detect_regnal(ref),
        "hyp": _detect_regnal(hyp),
        "form": lambda it: it[0],
        "value": lambda it: it[1],
    }

    per_category: dict[str, dict] = {}
    total = 0
    total_strict = 0
    total_value = 0
    for cat, spec in specs.items():
        breakdown = _classify_per_category(
            spec["gt"], spec["hyp"],
            form_extractor=spec["form"],
            value_extractor=spec["value"],
        )
        per_category[cat] = breakdown
        total += breakdown["n_total"]
        total_strict += breakdown["strict"]
        total_value += breakdown["value"]

    return {
        "n_total": total,
        "global_strict_score": (
            total_strict / total if total else 0.0
        ),
        "global_value_score": (
            total_value / total if total else 0.0
        ),
        "per_category": per_category,
    }


# ──────────────────────────────────────────────────────────────────────────
# Enregistrement registre typé
# ──────────────────────────────────────────────────────────────────────────


@register_metric(
    name="numerical_sequence_strict_score",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description=(
        "Précision sur séquences numériques en mode strict (forme "
        "préservée). Couvre années arabes, numéraux romains, "
        "foliotation, montants Ancien Régime, années régnales."
    ),
)
def numerical_sequence_strict_score(reference: str, hypothesis: str) -> float:
    return compute_numerical_sequence_metrics(
        reference, hypothesis,
    )["global_strict_score"]


@register_metric(
    name="numerical_sequence_value_score",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description=(
        "Précision sur séquences numériques en mode valeur "
        "(la valeur est préservée même si la forme diffère, "
        "ex. XIV → 14)."
    ),
)
def numerical_sequence_value_score(reference: str, hypothesis: str) -> float:
    return compute_numerical_sequence_metrics(
        reference, hypothesis,
    )["global_value_score"]


__all__ = [
    "CATEGORIES",
    "compute_numerical_sequence_metrics",
    "numerical_sequence_strict_score",
    "numerical_sequence_value_score",
]
