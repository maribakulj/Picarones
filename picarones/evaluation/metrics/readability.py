"""Métriques de lisibilité (Flesch)

A.II.2.3 du plan d'évolution 2026 : couche de calcul pure
de la métrique Flesch, indépendante de tout alignement OCR/GT.

Pourquoi ce module
------------------
Les LLM produisent du texte plus « lisse » que les manuscrits
historiques.  Cette tendance à la modernisation est mesurable par la
différence de score de lisibilité entre la GT et la sortie OCR/LLM —
**indépendamment des classes taxonomiques** et **sans alignement
caractère/mot**.  C'est l'avantage clé du score Flesch : il fonctionne
même quand l'OCR est très dégradé (cas d'un LLM qui invente du texte
moderne plausible mais déconnecté de la GT).

Stratégie de découpage
----------------------
Comme pour le NER (Sprint 38) et la calibration (Sprint 39), on
découpe :

- **Sprint 52** (ici) — couche de calcul pure : ``flesch_score`` et
  ``flesch_delta``.  Aucune dépendance externe ; les heuristiques de
  comptage de syllabes sont en pur Python, déterministes, testées.
- **Sprints suivants** — câblage runner pour calculer
  ``flesch_delta`` par document et l'agréger au moteur, puis vue HTML.

Formules
--------
- **Anglais** (Flesch original 1948) :
  ``206.835 - 1.015 × (mots/phrases) - 84.6 × (syllabes/mots)``
- **Français** (Kandel-Moles 1958) :
  ``207 - 1.015 × (mots/phrases) - 73.6 × (syllabes/mots)``

Le score est borné dans ``[0, 100]`` — 100 ↔ « très facile à lire »,
0 ↔ « très difficile ».  Une **augmentation** du score quand on passe
de la GT à l'OCR signale une simplification (typique des LLM
modernisants).  Une **chute** signale une dégradation OCR.

Limites documentées
-------------------
- Le comptage de syllabes est heuristique.  En français, des règles
  comme « -ier non final = 2 syllabes » ne sont pas appliquées
  finement.  Acceptable pour une métrique de **comparaison relative**
  (delta GT vs OCR), pas pour publier une absolue.
- Sur des textes très courts (< 20 mots), la formule perd en
  fiabilité.  Le seuil minimal est documenté.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from picarones.evaluation.metric_registry import register_metric
from picarones.domain.artifacts import ArtifactType

logger = logging.getLogger(__name__)


Language = Literal["fr", "en"]

# Coefficients de la formule Flesch selon la langue.
_FLESCH_COEFFS: dict[str, tuple[float, float, float]] = {
    "en": (206.835, 1.015, 84.6),     # Flesch 1948
    "fr": (207.0,   1.015, 73.6),     # Kandel-Moles 1958
}

# Voyelles utilisées pour l'heuristique de comptage de syllabes.
# On utilise un set qui inclut les diacritiques courantes en FR/EN.
_VOWELS = set("aeiouyàâäéèêëîïôöùûüÿæœAEIOUYÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÆŒ")

# Regex de découpage en phrases : ponctuation finale + espace ou fin.
# Tolère les multiples points (« ... ») et garde un découpage robuste.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?…]+(?:\s+|$)")

# Regex de tokenisation simple (mots) : séquences de caractères "lettres".
_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)


# ──────────────────────────────────────────────────────────────────────────
# Compteurs de base
# ──────────────────────────────────────────────────────────────────────────


def count_words(text: str) -> int:
    """Nombre de mots (tokens alphanumériques) dans ``text``."""
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def count_sentences(text: str) -> int:
    """Nombre de phrases dans ``text``.

    Découpage par ponctuation finale (``.``, ``!``, ``?``, ``…``).
    Renvoie au minimum 1 si ``text`` contient au moins un mot, pour
    éviter une division par zéro dans la formule de Flesch sur les
    textes sans ponctuation finale.
    """
    if not text:
        return 0
    parts = [p for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    n = len(parts)
    if n == 0 and count_words(text) > 0:
        return 1
    return n


def count_syllables_word(word: str) -> int:
    """Heuristique de comptage de syllabes pour un mot isolé.

    Règle : on compte les **groupes de voyelles consécutives** (en
    incluant ``y`` et les diacritiques courantes).  C'est une
    approximation grossière mais déterministe et testable.

    Cas limites :
    - mot vide → 0
    - mot sans voyelle → 1 (par convention, ex. acronymes ``BNF``)
    - mot d'une seule voyelle isolée → 1
    """
    if not word:
        return 0
    word = word.lower()
    in_vowel_group = False
    count = 0
    for ch in word:
        if ch in _VOWELS:
            if not in_vowel_group:
                count += 1
                in_vowel_group = True
        else:
            in_vowel_group = False
    return count or 1


def count_syllables(text: str) -> int:
    """Somme des syllabes de tous les mots de ``text``."""
    if not text:
        return 0
    return sum(count_syllables_word(w) for w in _WORD_RE.findall(text))


# ──────────────────────────────────────────────────────────────────────────
# Score Flesch
# ──────────────────────────────────────────────────────────────────────────


def flesch_score(text: str, lang: Language = "fr") -> float:
    """Calcule le score de lisibilité Flesch pour ``text``.

    Parameters
    ----------
    text:
        Texte à évaluer.  Peut contenir ponctuation, accents, etc.
    lang:
        ``"fr"`` (Kandel-Moles 1958, défaut) ou ``"en"`` (Flesch 1948).

    Returns
    -------
    float
        Score borné dans ``[0, 100]``.  Renvoie ``0.0`` sur un texte
        vide ou sans mot exploitable.

    Notes
    -----
    Le score chute fortement avec :
    - longues phrases (mots/phrases élevé)
    - mots polysyllabiques (syllabes/mots élevé)
    Une montée du score lors du passage GT → OCR signale qu'un LLM a
    « lissé » la langue (phrases plus courtes, mots plus communs).
    """
    if lang not in _FLESCH_COEFFS:
        raise ValueError(f"Langue non supportée : {lang!r}. Choisir 'fr' ou 'en'.")

    n_words = count_words(text)
    if n_words == 0:
        return 0.0
    n_sentences = max(1, count_sentences(text))
    n_syllables = count_syllables(text)
    if n_syllables == 0:
        return 0.0

    base, k_words, k_syll = _FLESCH_COEFFS[lang]
    raw = base - k_words * (n_words / n_sentences) - k_syll * (n_syllables / n_words)
    return max(0.0, min(100.0, raw))


def flesch_delta(
    reference: str,
    hypothesis: str,
    lang: Language = "fr",
) -> float:
    """Différence ``flesch_score(hypothesis) - flesch_score(reference)``.

    Interprétation
    --------------
    - **Positif** : l'hypothèse OCR est plus lisible que la GT —
      signal d'**over-normalisation** (typique des LLM qui modernisent
      des textes anciens).
    - **Négatif** : l'OCR est moins lisible — signal de dégradation
      (caractères mal reconnus brisent la fluidité).
    - **≈ 0** : OCR fidèle à la GT en termes de complexité linguistique.

    Borné dans ``[-100, +100]``.
    """
    return flesch_score(hypothesis, lang=lang) - flesch_score(reference, lang=lang)


# ──────────────────────────────────────────────────────────────────────────
# Enregistrement dans le registre typé
# ──────────────────────────────────────────────────────────────────────────


@register_metric(
    name="flesch_delta_fr",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description=(
        "Différence de score Flesch (Kandel-Moles, FR) entre la sortie "
        "OCR et la GT. Positif = OCR plus lisible (signal "
        "d'over-normalisation LLM). Aucun alignement requis."
    ),
    higher_is_better=False,  # un delta proche de 0 = fidélité ; positif = LLM lissant
    tags={"text", "readability", "over_normalization"},
)
def _registered_flesch_delta_fr(reference: str, hypothesis: str) -> float:
    return flesch_delta(reference, hypothesis, lang="fr")


@register_metric(
    name="flesch_delta_en",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description=(
        "Flesch reading ease delta (Flesch 1948, EN) between OCR and GT. "
        "Positive = OCR easier to read than GT (LLM smoothing signal). "
        "No alignment required."
    ),
    higher_is_better=False,
    tags={"text", "readability", "over_normalization"},
)
def _registered_flesch_delta_en(reference: str, hypothesis: str) -> float:
    return flesch_delta(reference, hypothesis, lang="en")


__all__ = [
    "flesch_score",
    "flesch_delta",
    "count_words",
    "count_sentences",
    "count_syllables",
    "count_syllables_word",
]
