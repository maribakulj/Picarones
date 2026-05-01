"""Score d'expansion d'abréviations médiévales — Sprint 56.

Sprint 56 — A.II.3.2 du plan d'évolution 2026 (axe philologique).

Pourquoi ce module
------------------
Sur les manuscrits médiévaux (chartes, registres, copies de droit
canonique), les scribes utilisent intensivement des **signes
d'abréviation** : ``ꝑ`` (per/par), ``ꝓ`` (pro), ``ꝗ`` (qui),
``ꝙ`` (quia), ``ꝯ`` (con/-us), ``⁊`` (et), tilde combinant pour
``-en/-an``, etc.

Un OCR/HTR a deux comportements possibles face à ces signes :

1. **Préservation** : la forme abrégée est gardée telle quelle
   (``ꝑ`` → ``ꝑ``).  C'est le comportement attendu d'une
   transcription **diplomatique** (édition critique).
2. **Développement** : le signe est remplacé par sa forme
   développée (``ꝑ`` → ``per``).  C'est le comportement attendu
   d'une édition **modernisée**.

Une troisième possibilité — et c'est l'erreur qu'on cherche à
détecter : le signe est **mal restitué** (remplacé par un
caractère ASCII proche, supprimé, ou mal développé).

Ce module produit deux scores complémentaires :

- ``abbreviation_strict_score`` : taux d'abréviations GT dont la
  **forme abrégée Unicode est préservée** dans l'OCR.
- ``abbreviation_expansion_score`` : taux d'abréviations GT dont
  **soit** la forme abrégée, **soit** la forme développée
  attendue, est présente dans l'OCR.

Le **ratio** des deux dit beaucoup sur la convention adoptée :

- ``strict ≈ expansion`` proche de 1 → le moteur est diplomatique
  (préserve l'abrégé) ;
- ``strict << expansion`` → le moteur est modernisant (développe
  systématiquement) ;
- les deux faibles → le moteur perd les abréviations (signal
  d'erreur OCR).

Stratégie de découpage
----------------------
Cohérente avec NER (Sprint 38), Flesch (52), Reading order F1 (53),
Layout F1 (54), Bloc Unicode (55) : couche de calcul pure d'abord.
Le câblage runner et la vue HTML suivent dans des sprints dédiés.

Limites documentées
-------------------
- L'alignement est **bag-of-occurrences** (proxy positionnel
  simple) : on compte les occurrences GT et on vérifie leur
  présence dans l'hyp.  Pas d'alignement séquentiel rigoureux.
- La table d'abréviations couvre les signes les plus courants en
  scriptura latine européenne (Capelli).  Elle est extensible via
  ``ABBREVIATION_EXPANSIONS``.
- Pour les abréviations marquées par un **tilde combinant**
  (``p̃``, ``q̃``), on détecte la séquence ``lettre + U+0303``.
  Pas de gestion fine des polices Capelli/MUFI complètes.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

from picarones.core.metric_registry import register_metric
from picarones.core.modules import ArtifactType

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Table d'expansions
# ──────────────────────────────────────────────────────────────────────────

# Signes d'abréviation latins médiévaux les plus courants.
# Source : Capelli, "Lexicon Abbreviaturarum" (1929) + MUFI.
#
# La clé est une chaîne (1 ou 2 code-points pour le cas tilde
# combinant) ; la valeur est la liste des expansions courantes
# acceptées (les détails varient selon la convention éditoriale,
# on accepte plusieurs formes).
ABBREVIATION_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "ꝑ": ("per", "par"),                       # U+A751
    "ꝓ": ("pro",),                              # U+A753
    "ꝗ": ("qui",),                              # U+A757
    "ꝙ": ("quia",),                             # U+A759
    "ꝯ": ("us", "con"),                         # U+A76F
    "⁊": ("et",),                               # U+204A "et" tironien
    "ꝝ": ("rum",),                              # U+A75D
    "ꝫ": ("et",),                               # U+A76B
    "ꝭ": ("is",),                               # U+A76D
    # Tilde combinant après lettre (U+0303 = ̃) : pẽ, qũ, etc.
    "p̃": ("par", "per"),
    "q̃": ("que", "qui"),
    "ñ": ("an", "en"),                          # U+00F1 (Latin-1 Sup)
    # Note : ñ existe aussi comme caractère latin moderne (espagnol),
    # donc l'attribuer aux abréviations introduit du bruit ; on
    # laisse au benchmark le soin d'évaluer.  Pour les éditeurs
    # médiévistes qui veulent restreindre, ils peuvent passer par
    # une table custom (à venir).
}


# Set des "premiers code-points" reconnus comme début d'une
# abréviation (pour balayage rapide).
_ABBR_FIRST_CHARS: frozenset[str] = frozenset(
    abbr[0] for abbr in ABBREVIATION_EXPANSIONS
)


# Combining tilde (U+0303) — utilisé pour la détection p̃, q̃, etc.
_COMBINING_TILDE = "̃"


# ──────────────────────────────────────────────────────────────────────────
# Détection d'abréviations dans un texte
# ──────────────────────────────────────────────────────────────────────────


def detect_abbreviations(text: Optional[str]) -> list[str]:
    """Liste des abréviations médiévales détectées dans ``text``,
    dans l'ordre d'apparition.

    Reconnaît :

    - Les caractères Unicode dédiés présents dans
      ``ABBREVIATION_EXPANSIONS`` (``ꝑ``, ``ꝓ``, ``⁊``…).
    - Les séquences ``lettre + U+0303`` (tilde combinant) si la
      paire est dans la table (``p̃``, ``q̃``).

    Doublons conservés : si le texte contient deux ``ꝑ``, la liste
    en a deux.  Cohérent avec le calcul bag-of-occurrences en aval.
    """
    if not text:
        return []
    found: list[str] = []
    # Forme NFD pour reconnaître les ã, p̃, q̃ même quand l'utilisateur
    # passe la forme NFC (« ñ » = U+00F1 sera traité par le mapping
    # direct ; les séquences manuelles ``p`` + tilde combinant restent
    # détectables).
    text_nfd = unicodedata.normalize("NFD", text)
    i = 0
    while i < len(text_nfd):
        ch = text_nfd[i]
        # Cas 1 : lettre + tilde combinant
        if i + 1 < len(text_nfd) and text_nfd[i + 1] == _COMBINING_TILDE:
            seq = ch + _COMBINING_TILDE
            if seq in ABBREVIATION_EXPANSIONS:
                found.append(seq)
                i += 2
                continue
        # Cas 2 : caractère unicode dédié
        if ch in ABBREVIATION_EXPANSIONS:
            found.append(ch)
        i += 1
    return found


# ──────────────────────────────────────────────────────────────────────────
# Scores
# ──────────────────────────────────────────────────────────────────────────


def _hyp_contains_abbr(hypothesis: str, abbr: str) -> bool:
    """Vrai si la forme abrégée ``abbr`` apparaît telle quelle dans
    ``hypothesis``.  Sensible aux deux formes NFC / NFD pour les
    séquences à tilde combinant."""
    if abbr in hypothesis:
        return True
    # Pour les séquences ``lettre + tilde combinant``, l'hyp peut
    # avoir une forme NFC (ex. ``ñ`` au lieu de ``n + U+0303``).
    nfd = unicodedata.normalize("NFD", hypothesis)
    return abbr in nfd


def _hyp_contains_expansion(
    hypothesis: str, expansions: tuple[str, ...],
) -> bool:
    """Vrai si l'une des formes développées apparaît dans ``hypothesis``
    (recherche insensible à la casse, sur les frontières de mots
    pour limiter les faux positifs sur les sous-chaînes courtes
    type ``us`` ou ``et``)."""
    if not expansions:
        return False
    hyp_lower = hypothesis.lower()
    for exp in expansions:
        if not exp:
            continue
        # Recherche frontière de mot pour les expansions courtes.
        # Pour ``per`` ou ``pro`` : on accepte le développement à
        # n'importe quelle position d'un mot (tolère ``per`` dans
        # ``permettre``, c'est imprécis mais pragmatique).  Pour
        # les expansions très courtes (≤ 2 lettres), on impose un
        # mot complet pour limiter le bruit.
        if len(exp) <= 2:
            if re.search(rf"\b{re.escape(exp)}\b", hyp_lower):
                return True
        else:
            if exp.lower() in hyp_lower:
                return True
    return False


def compute_abbreviation_metrics(
    reference: Optional[str],
    hypothesis: Optional[str],
) -> dict:
    """Calcule les scores d'abréviation strict et d'expansion.

    Parameters
    ----------
    reference:
        Texte GT (avec abréviations médiévales originales).
    hypothesis:
        Texte produit par l'OCR.

    Returns
    -------
    dict
        ``{
            "n_abbreviations_in_reference": int,
            "n_strict_preserved":   int,    # forme abrégée préservée
            "n_expansion_preserved": int,    # abrégée OU développée
            "strict_score":   float,         # ∈ [0, 1]
            "expansion_score": float,        # ∈ [0, 1]
            "per_abbreviation": [
                {"abbr", "strict_preserved", "expansion_preserved",
                 "expansions"},
                ...
            ],
        }``

    Cas dégénérés
    -------------
    - GT vide ou sans abréviation détectée → tous les compteurs à 0
      et les scores à ``0.0`` (convention : on ne récompense pas
      l'absence d'abréviations).
    - GT non vide avec abréviations + hyp vide → tous les scores
      à ``0.0``.
    """
    ref = reference or ""
    hyp = hypothesis or ""

    abbreviations = detect_abbreviations(ref)
    n = len(abbreviations)
    if n == 0:
        return {
            "n_abbreviations_in_reference": 0,
            "n_strict_preserved": 0,
            "n_expansion_preserved": 0,
            "strict_score": 0.0,
            "expansion_score": 0.0,
            "per_abbreviation": [],
        }

    n_strict = 0
    n_expansion = 0
    per_abbr: list[dict] = []
    for abbr in abbreviations:
        expansions = ABBREVIATION_EXPANSIONS.get(abbr, ())
        strict_ok = _hyp_contains_abbr(hyp, abbr)
        # Expansion : on accepte la forme abrégée OU le développement.
        # Convention : si l'OCR a préservé la forme abrégée, c'est
        # aussi compté comme valide pour le score d'expansion (le
        # moteur n'a pas perdu l'information ; il a juste choisi
        # une convention diplomatique).
        expansion_ok = strict_ok or _hyp_contains_expansion(hyp, expansions)
        if strict_ok:
            n_strict += 1
        if expansion_ok:
            n_expansion += 1
        per_abbr.append({
            "abbr": abbr,
            "strict_preserved": strict_ok,
            "expansion_preserved": expansion_ok,
            "expansions": list(expansions),
        })

    return {
        "n_abbreviations_in_reference": n,
        "n_strict_preserved": n_strict,
        "n_expansion_preserved": n_expansion,
        "strict_score": n_strict / n,
        "expansion_score": n_expansion / n,
        "per_abbreviation": per_abbr,
    }


def abbreviation_strict_score(
    reference: Optional[str], hypothesis: Optional[str],
) -> float:
    """Raccourci : taux de préservation **stricte** des abréviations
    Unicode (forme abrégée gardée telle quelle)."""
    return compute_abbreviation_metrics(reference, hypothesis)["strict_score"]


def abbreviation_expansion_score(
    reference: Optional[str], hypothesis: Optional[str],
) -> float:
    """Raccourci : taux de préservation par expansion (forme abrégée
    OU forme développée présente dans l'hyp)."""
    return compute_abbreviation_metrics(reference, hypothesis)["expansion_score"]


# ──────────────────────────────────────────────────────────────────────────
# Enregistrement dans le registre typé (Sprint 34)
# ──────────────────────────────────────────────────────────────────────────


@register_metric(
    name="abbreviation_strict_score",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description=(
        "Taux d'abréviations médiévales (Unicode dédié + lettre + "
        "tilde combinant) dont la forme abrégée est préservée telle "
        "quelle dans l'OCR. Idéal pour les éditions diplomatiques."
    ),
    higher_is_better=True,
    tags={"text", "abbreviation", "philology", "medieval"},
)
def _registered_strict(reference: str, hypothesis: str) -> float:
    return abbreviation_strict_score(reference, hypothesis)


@register_metric(
    name="abbreviation_expansion_score",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description=(
        "Taux d'abréviations dont SOIT la forme abrégée Unicode SOIT "
        "la forme développée attendue (per, pro, et…) est présente "
        "dans l'OCR. Score plus large que strict_score."
    ),
    higher_is_better=True,
    tags={"text", "abbreviation", "philology", "medieval"},
)
def _registered_expansion(reference: str, hypothesis: str) -> float:
    return abbreviation_expansion_score(reference, hypothesis)


__all__ = [
    "ABBREVIATION_EXPANSIONS",
    "detect_abbreviations",
    "compute_abbreviation_metrics",
    "abbreviation_strict_score",
    "abbreviation_expansion_score",
]
