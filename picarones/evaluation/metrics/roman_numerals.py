"""Numéraux romains — Sprint 60.

Phase 5.C.batch7 — module relocalisé depuis
``picarones.measurements.roman_numerals`` vers
``picarones.evaluation.metrics.roman_numerals``.  Le chemin legacy
reste disponible via un shim avec ``DeprecationWarning`` ;
suppression prévue en 2.0.

Sprint 60 — Étape 3 / extension philologique transversale du plan
d'évolution 2026.

Pourquoi ce module
------------------
Les numéraux romains traversent **toutes les périodes patrimoniales**
servies par Picarones :

- **Médiéval** : minuscules avec ``j`` final pour le dernier ``i``
  (``ij`` = 2, ``iij`` = 3, ``viij`` = 8, ``mcclxxxij`` = 1282).
  Convention scribale standard dans les chartes et registres.
- **Imprimé ancien** : majuscules (``Tome IV``, ``Chap. VII``).
- **Moderne** : majuscules pour les souverains (``Louis XIV``) et
  les siècles (``XIXᵉ siècle`` — la partie exposant ᵉ est gérée
  par le Sprint 59 ``ordinals``, ce module ne traite que la partie
  numérale ``XIX``).

Quatre traitements possibles d'un numéral par l'OCR
----------------------------------------------------
Pour chaque numéral romain présent dans la GT, l'OCR peut :

1. **Préserver strictement** : forme exacte gardée
   (``mcclxxxij`` → ``mcclxxxij``).  Édition diplomatique idéale.
2. **Préserver en changeant la casse** : la valeur est intacte mais
   la convention typographique est modifiée
   (``xiv`` → ``XIV``).  Modernisation typographique courante.
3. **Préserver en supprimant le ``j`` final** :
   (``mcclxxxij`` → ``mcclxxxii``).  Modernisation orthographique
   médiévale → standard académique moderne.
4. **Convertir en chiffres arabes** : la valeur est préservée mais
   le système de numération est modernisé
   (``XIV`` → ``14``).  Modernisation profonde, perte de
   l'information typographique.
5. **Perdre** : aucune trace de la valeur dans l'hypothèse.

Ce module retourne un breakdown par statut pour que le chercheur
juge lui-même la convention adoptée par chaque moteur, **sans
classification automatique imposée**.

Stratégie de découpage
----------------------
Cohérente avec NER (38), Flesch (52), Reading order F1 (53),
Layout F1 (54), Bloc Unicode (55), Abréviations (56), MUFI (57),
Imprimé ancien (58), Archives modernes (59) : couche de calcul
pure d'abord ; câblage runner et HTML dans des sprints dédiés.

Limites documentées
-------------------
- Détection greedy par regex ``\\b[IVXLCDMivxlcdmj]+\\b`` puis
  validation par parsing.  Les faux positifs restent possibles sur
  des mots courts (``I`` pronom anglais, ``MM`` initiales, ``LL``).
  Le paramètre ``min_length`` permet de filtrer les single-letter.
- Pas de gestion des notations rares avec barre suscript pour
  multiplier par 1000 (V̄ = 5000, X̄ = 10000) — usage très rare en
  corpus patrimonial européen courant.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from picarones.evaluation.metric_registry import register_metric
from picarones.domain.artifacts import ArtifactType

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Table de conversion + parsing
# ──────────────────────────────────────────────────────────────────────────

ROMAN_VALUES: dict[str, int] = {
    "I": 1,    "V": 5,    "X": 10,
    "L": 50,   "C": 100,  "D": 500,  "M": 1000,
}

# Caractères acceptés en entrée (incluant minuscules + j médiéval).
_ROMAN_CHARS = "IVXLCDMivxlcdmj"
_ROMAN_RE = re.compile(rf"\b[{_ROMAN_CHARS}]+\b")


def _normalize_roman(s: str) -> str:
    """Normalise un numéral romain : majuscule + ``j`` final → ``i``.

    Les manuscrits médiévaux notent traditionnellement le dernier
    ``i`` d'une suite par ``j`` (« ij », « iij », « viij »…).  On
    convertit pour pouvoir parser comme un numéral standard.
    """
    if not s:
        return ""
    upper = s.upper()
    if upper.endswith("J"):
        upper = upper[:-1] + "I"
    return upper


def _parse_normalized_roman(s: str) -> Optional[int]:
    """Parse un numéral romain **après normalisation** (majuscule,
    sans ``j`` médiéval).  Retourne ``None`` si la chaîne n'est pas
    un numéral romain valide.

    Validation : on parse en additionnant/soustrayant selon la règle
    classique, puis on **regénère la forme standard** et on compare
    pour rejeter les formes non canoniques (« IIII » au lieu de
    « IV », « VV » au lieu de « X »).  Cette stricte validation
    garantit qu'on ne compte pas des séquences absurdes comme
    « XXXX » comme un numéral.

    Note : les manuscrits médiévaux utilisent fréquemment « IIII »
    pour 4 (notation soustractive plus tardive).  On accepte donc
    aussi cette forme via une règle relâchée : tant que les valeurs
    sont décroissantes ou suivent la règle soustractive standard,
    on accepte.
    """
    if not s or not all(c in "IVXLCDM" for c in s):
        return None
    # Calcul par soustraction.
    total = 0
    prev_value = 0
    for ch in reversed(s):
        v = ROMAN_VALUES[ch]
        if v < prev_value:
            total -= v
        else:
            total += v
        prev_value = v
    if total <= 0:
        return None
    # Validation relâchée : on accepte les formes médiévales (IIII,
    # VIIII) mais on rejette les vraiment absurdes (IIIII, VVVV).
    if not _is_plausible_roman(s):
        return None
    return total


def _is_plausible_roman(s: str) -> bool:
    """Validation relâchée d'un numéral romain (majuscule).

    On rejette :

    - 5 caractères identiques d'affilée ou plus (« IIIII », « XXXXX »).
    - Les répétitions de V, L, D (jamais répétés en notation
      classique : « VV », « LL », « DD »).
    - Les paires soustractives non standard.  En romain canonique,
      seules sont valides : IV, IX, XL, XC, CD, CM.  Toute autre
      combinaison « petit avant grand » est rejetée.  Cela élimine
      les faux positifs sur des mots français comme « ici » (qui
      formerait sinon « I + C » = 99) ou « IL » qui formerait 49.
    """
    if not s:
        return False
    # Pas de répétitions invalides
    for forbidden in ("VV", "LL", "DD", "IIIII", "XXXXX", "CCCCC", "MMMMMM"):
        if forbidden in s:
            return False
    # Paires soustractives autorisées (toutes les autres sont rejetées)
    legal_subtractive = {"IV", "IX", "XL", "XC", "CD", "CM"}
    for i in range(len(s) - 1):
        a, b = s[i], s[i + 1]
        if ROMAN_VALUES[a] < ROMAN_VALUES[b]:
            if (a + b) not in legal_subtractive:
                return False
    return True


def roman_to_int(s: Optional[str]) -> Optional[int]:
    """Convertit une chaîne en numéral romain entier.  Tolère casse
    et ``j`` médiéval final.  Retourne ``None`` si invalide.
    """
    if not s:
        return None
    return _parse_normalized_roman(_normalize_roman(s))


def int_to_roman(n: int) -> str:
    """Convertit un entier en numéral romain majuscule standard.

    Utilise la notation classique (IV, IX, XL, XC, CD, CM) — pas la
    forme médiévale relâchée.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    pairs = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"),  (90, "XC"),  (50, "L"),  (40, "XL"),
        (10, "X"),   (9, "IX"),   (5, "V"),   (4, "IV"),
        (1, "I"),
    ]
    out: list[str] = []
    for value, symbol in pairs:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)


# ──────────────────────────────────────────────────────────────────────────
# Détection dans le texte
# ──────────────────────────────────────────────────────────────────────────


def detect_roman_numerals(
    text: Optional[str],
    *,
    min_length: int = 1,
) -> list[tuple[int, str, int]]:
    """Retourne les numéraux romains valides dans ``text``.

    Forme : ``[(start_index, numeral_string, integer_value), ...]``
    triée par index croissant.

    Parameters
    ----------
    text:
        Texte à analyser.
    min_length:
        Longueur minimale d'un numéral retenu.  Par défaut ``1``.
        Mettre à ``2`` pour filtrer les single-letter ambigus (``I``
        pronom, ``M`` initiale).

    Faux positifs connus
    --------------------
    - ``I`` (pronom anglais), ``M`` ou ``D`` en initiale d'une
      personne ne peuvent pas être distingués sans NER.  Le chercheur
      qui s'inquiète de ces faux positifs peut passer
      ``min_length=2``.
    """
    if not text:
        return []
    found: list[tuple[int, str, int]] = []
    for match in _ROMAN_RE.finditer(text):
        s = match.group(0)
        if len(s) < min_length:
            continue
        value = roman_to_int(s)
        if value is None:
            continue
        found.append((match.start(), s, value))
    return found


# ──────────────────────────────────────────────────────────────────────────
# Classification de la restitution dans l'hypothèse
# ──────────────────────────────────────────────────────────────────────────

# Statuts possibles, dans l'ordre de priorité (un numéral est
# classé selon le premier statut qui s'applique).

STATUS_STRICT_PRESERVED   = "strict_preserved"
STATUS_CASE_CHANGED       = "case_changed"
STATUS_J_DROPPED          = "j_dropped"
STATUS_CONVERTED_TO_ARABIC = "converted_to_arabic"
STATUS_LOST               = "lost"

ALL_STATUSES = (
    STATUS_STRICT_PRESERVED,
    STATUS_CASE_CHANGED,
    STATUS_J_DROPPED,
    STATUS_CONVERTED_TO_ARABIC,
    STATUS_LOST,
)

# Statuts qui indiquent une préservation de la valeur (par opposition
# à la perte).
VALUE_PRESERVING_STATUSES = frozenset({
    STATUS_STRICT_PRESERVED,
    STATUS_CASE_CHANGED,
    STATUS_J_DROPPED,
    STATUS_CONVERTED_TO_ARABIC,
})


def _classify_restitution(numeral: str, value: int, hyp: str) -> str:
    """Classifie comment ``numeral`` (de valeur ``value``) est
    restitué dans ``hyp`` selon les 5 statuts définis."""
    # 1. Forme stricte présente
    if re.search(r"(?<![A-Za-z])" + re.escape(numeral) + r"(?![A-Za-z])", hyp):
        return STATUS_STRICT_PRESERVED
    # 2. Variante de casse seule
    swapped = numeral.swapcase()
    if swapped != numeral and re.search(
        r"(?<![A-Za-z])" + re.escape(swapped) + r"(?![A-Za-z])", hyp,
    ):
        return STATUS_CASE_CHANGED
    # 3. ``j`` final remplacé par ``i`` (ou inverse)
    if numeral.lower().endswith("j"):
        no_j = numeral[:-1] + ("I" if numeral[-1] == "J" else "i")
    elif numeral.lower().endswith("i"):
        no_j = numeral[:-1] + ("J" if numeral[-1] == "I" else "j")
    else:
        no_j = numeral
    if no_j != numeral and re.search(
        r"(?<![A-Za-z])" + re.escape(no_j) + r"(?![A-Za-z])", hyp,
    ):
        return STATUS_J_DROPPED
    # Variante de casse + j-flip combinés
    no_j_swapped = no_j.swapcase()
    if no_j_swapped != numeral and re.search(
        r"(?<![A-Za-z])" + re.escape(no_j_swapped) + r"(?![A-Za-z])", hyp,
    ):
        return STATUS_J_DROPPED
    # 4. Conversion en chiffres arabes
    if re.search(r"(?<!\d)" + str(value) + r"(?!\d)", hyp):
        return STATUS_CONVERTED_TO_ARABIC
    # 5. Perdu
    return STATUS_LOST


# ──────────────────────────────────────────────────────────────────────────
# Calcul de la métrique
# ──────────────────────────────────────────────────────────────────────────


def compute_roman_numeral_metrics(
    reference: Optional[str],
    hypothesis: Optional[str],
    *,
    min_length: int = 1,
) -> dict:
    """Calcule la préservation des numéraux romains.

    Pour chaque numéral romain dans la GT, on classifie sa
    restitution dans l'hypothèse selon l'un des 5 statuts (forme
    stricte / casse modifiée / j supprimé / conversion arabe / perdu).

    Returns
    -------
    dict
        ``{
            "n_numerals_reference": int,
            "n_strict_preserved": int,
            "n_value_preserved": int,    # tous statuts sauf LOST
            "global_strict_score": float,
            "global_value_score": float,
            "per_status": {status: count for status in ALL_STATUSES},
            "per_numeral": [
                {"index", "numeral", "value", "status"}
            ],
            "lost_numerals": [
                {"index", "numeral", "value"}
            ],
        }``

    Cas dégénérés
    -------------
    - GT vide ou sans numéral → tous compteurs à 0, scores à 0.0,
      ``per_status`` initialisé à 0 sur tous les statuts.
    - GT avec numéraux + hyp vide → tous classés ``lost``,
      strict_score = value_score = 0.0.
    """
    ref = reference or ""
    hyp = hypothesis or ""

    detected = detect_roman_numerals(ref, min_length=min_length)
    n_total = len(detected)
    per_status_init = {status: 0 for status in ALL_STATUSES}

    if n_total == 0:
        return {
            "n_numerals_reference": 0,
            "n_strict_preserved": 0,
            "n_value_preserved": 0,
            "global_strict_score": 0.0,
            "global_value_score": 0.0,
            "per_status": per_status_init,
            "per_numeral": [],
            "lost_numerals": [],
        }

    per_status: dict[str, int] = dict(per_status_init)
    per_numeral: list[dict] = []
    lost: list[dict] = []
    for index, numeral, value in detected:
        status = _classify_restitution(numeral, value, hyp)
        per_status[status] = per_status.get(status, 0) + 1
        per_numeral.append({
            "index": index,
            "numeral": numeral,
            "value": value,
            "status": status,
        })
        if status == STATUS_LOST:
            lost.append({"index": index, "numeral": numeral, "value": value})

    n_strict = per_status[STATUS_STRICT_PRESERVED]
    n_value = sum(per_status[s] for s in VALUE_PRESERVING_STATUSES)

    return {
        "n_numerals_reference": n_total,
        "n_strict_preserved": n_strict,
        "n_value_preserved": n_value,
        "global_strict_score": n_strict / n_total,
        "global_value_score": n_value / n_total,
        "per_status": per_status,
        "per_numeral": per_numeral,
        "lost_numerals": lost,
    }


def roman_numeral_strict_score(
    reference: Optional[str], hypothesis: Optional[str],
) -> float:
    """Raccourci : taux global de préservation **stricte** des
    numéraux romains ∈ [0, 1]."""
    return compute_roman_numeral_metrics(
        reference, hypothesis,
    )["global_strict_score"]


def roman_numeral_value_score(
    reference: Optional[str], hypothesis: Optional[str],
) -> float:
    """Raccourci : taux global de préservation de la **valeur** des
    numéraux romains (toute forme confondue : strict, case_changed,
    j_dropped, arabe) ∈ [0, 1]."""
    return compute_roman_numeral_metrics(
        reference, hypothesis,
    )["global_value_score"]


# ──────────────────────────────────────────────────────────────────────────
# Enregistrement dans le registre typé (Sprint 34)
# ──────────────────────────────────────────────────────────────────────────


@register_metric(
    name="roman_numeral_strict_score",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description=(
        "Taux de préservation stricte des numéraux romains "
        "(forme exacte gardée : casse, j médiéval final). "
        "Métrique transversale aux périodes médiévale, imprimé "
        "ancien et moderne."
    ),
    higher_is_better=True,
    tags={"text", "roman_numerals", "philology"},
)
def _registered_strict(reference: str, hypothesis: str) -> float:
    return roman_numeral_strict_score(reference, hypothesis)


@register_metric(
    name="roman_numeral_value_score",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description=(
        "Taux de préservation de la valeur numérique des numéraux "
        "romains, indépendamment de la forme (strict, casse "
        "changée, j supprimé, conversion en chiffres arabes). "
        "Le breakdown per_status permet au chercheur de juger la "
        "convention adoptée."
    ),
    higher_is_better=True,
    tags={"text", "roman_numerals", "philology"},
)
def _registered_value(reference: str, hypothesis: str) -> float:
    return roman_numeral_value_score(reference, hypothesis)


__all__ = [
    "ROMAN_VALUES",
    "ALL_STATUSES",
    "STATUS_STRICT_PRESERVED",
    "STATUS_CASE_CHANGED",
    "STATUS_J_DROPPED",
    "STATUS_CONVERTED_TO_ARABIC",
    "STATUS_LOST",
    "VALUE_PRESERVING_STATUSES",
    "compute_roman_numeral_metrics",
    "detect_roman_numerals",
    "int_to_roman",
    "roman_numeral_strict_score",
    "roman_numeral_value_score",
    "roman_to_int",
]
