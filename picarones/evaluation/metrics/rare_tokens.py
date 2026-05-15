"""Rare-token recall — Sprint 71 (A.I.1 chantier 2 du plan 2026).

Pourquoi ce module
------------------
Le CER global d'un moteur peut sembler bon (ex. 5 %) tout en
masquant des **erreurs systématiques sur les tokens rares** : noms
propres, toponymes peu fréquents, mots techniques, formules latines
récurrentes mais pas dominantes.  Pour un usage prosopographique
(indexation de noms, recherche généalogique), ce sont précisément
ces tokens-là qui comptent.

Ce module mesure le **rappel sur les tokens rares** d'un corpus —
défaut : tokens dont la fréquence corpus-wide est ≤ 2 (hapax +
dis legomena, terminologie de lexicométrie classique).

Hypothèse à valider expérimentalement
-------------------------------------
La conjecture du plan A.I.1 : *« cette métrique discrimine plus
les moteurs que le CER global »*.  Si confirmée sur un corpus
patrimonial réel, elle gagne sa place dans le tableau de
classement principal — décision laissée au chercheur après
observation.

Stratégie de découpage
----------------------
Cohérente avec NER (38), Flesch (52), philologie (55-60) : couche
de calcul pure d'abord, sans intégration runner.  La vue HTML
« worst lines / rare tokens manqués » suit dans un sprint dédié.

Pas d'enregistrement dans le registre typé Sprint 34
----------------------------------------------------
La métrique exige **trois entrées** (reference, hypothesis, set
des tokens rares) et le set des rares est calculé corpus-wide
(donc connu seulement après itération sur tout le corpus).  La
signature ne rentre pas dans ``(TEXT, TEXT)``.  L'utilisateur
appelle explicitement ``compute_rare_token_recall`` avec le set
qu'il a calculé.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Tokenisation Unicode-aware
# ──────────────────────────────────────────────────────────────────────────

# Token = séquence maximale de caractères de mot Unicode (\w en
# Python 3 utilise déjà la table Unicode), incluant l'apostrophe
# typographique '’' à l'intérieur (« l'an », « d’une ») et les
# tirets internes (« peut-être »).  La ponctuation isolée et les
# espaces sont des séparateurs.

_TOKEN_RE = re.compile(
    r"\w+(?:[’'\-]\w+)*",
    flags=re.UNICODE,
)


def tokenize(text: Optional[str]) -> list[str]:
    """Tokenisation Unicode-aware.

    Conserve les contractions (``l'an``, ``d’une``) et les mots
    composés (``peut-être``, ``c'est-à-dire``) comme un seul token.
    Casse préservée — l'utilisateur normalise lui-même via
    ``case_sensitive=False`` dans les fonctions aval s'il le veut.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text)


# ──────────────────────────────────────────────────────────────────────────
# Distribution de fréquence corpus-wide
# ──────────────────────────────────────────────────────────────────────────


def frequency_distribution(
    documents: Iterable[str],
    *,
    case_sensitive: bool = False,
) -> Counter[str]:
    """Calcule ``{token: count}`` sur l'ensemble du corpus.

    Parameters
    ----------
    documents:
        Itérable de textes (typiquement les ``ground_truth`` des
        documents du corpus).
    case_sensitive:
        Si ``False`` (défaut), tous les tokens sont mis en
        minuscule avant comptage.
    """
    counter: Counter[str] = Counter()
    for doc in documents:
        tokens = tokenize(doc)
        if not case_sensitive:
            tokens = [t.lower() for t in tokens]
        counter.update(tokens)
    return counter


def extract_rare_tokens(
    documents: Iterable[str],
    *,
    max_freq: int = 2,
    case_sensitive: bool = False,
) -> frozenset[str]:
    """Retourne l'ensemble des tokens dont la fréquence
    corpus-wide est ``≤ max_freq``.

    Convention de lexicométrie : ``max_freq=1`` retourne uniquement
    les hapax legomena (1 occurrence) ; ``max_freq=2`` retourne
    hapax + dis legomena (≤ 2 occurrences) — défaut.

    Les tokens qui n'apparaissent **jamais** dans le corpus ne sont
    évidemment pas inclus (le ``Counter`` ne les liste pas).
    """
    if max_freq < 1:
        raise ValueError("max_freq doit être ≥ 1")
    counter = frequency_distribution(
        documents, case_sensitive=case_sensitive,
    )
    return frozenset(t for t, c in counter.items() if c <= max_freq)


# ──────────────────────────────────────────────────────────────────────────
# Calcul du rappel par document
# ──────────────────────────────────────────────────────────────────────────


def compute_rare_token_recall(
    reference: Optional[str],
    hypothesis: Optional[str],
    rare_tokens: Iterable[str],
    *,
    case_sensitive: bool = False,
) -> dict:
    """Calcule le rappel sur les tokens rares présents dans la GT.

    Parameters
    ----------
    reference:
        Texte GT du document.
    hypothesis:
        Texte produit par l'OCR.
    rare_tokens:
        Itérable des tokens rares — typiquement le résultat de
        ``extract_rare_tokens`` sur le corpus complet.
    case_sensitive:
        Si ``False`` (défaut), la comparaison se fait sur les
        formes minuscules.

    Returns
    -------
    dict
        ``{
            "n_rare_tokens_in_reference": int,
                # nombre d'**occurrences** de tokens rares dans la GT
                # (multiplicité préservée — un token rare présent 2
                # fois compte 2)
            "n_rare_tokens_recalled": int,
                # nombre d'occurrences correctement présentes dans hyp
                # (alignement bag-of-tokens : min(count_ref, count_hyp))
            "recall": float,
                # ratio dans [0, 1], ou 0.0 si aucun rare en GT
            "missed_tokens": list[str],
                # liste des tokens rares **manqués** (avec multiplicité,
                # ex. "Dupont" présent 2 fois en GT et 1 fois en hyp →
                # missed_tokens contient ["Dupont"] une fois)
        }``

    Cas dégénérés
    -------------
    - GT vide ou aucun token rare présent → recall = 0.0, listes
      vides (convention : on ne récompense pas l'absence de
      tokens rares).
    - Hyp vide avec rares en GT → tous manqués, recall = 0.0.
    """
    ref = reference or ""
    hyp = hypothesis or ""

    if case_sensitive:
        rare_set = frozenset(rare_tokens)
        ref_tokens = tokenize(ref)
        hyp_tokens = tokenize(hyp)
    else:
        rare_set = frozenset(t.lower() for t in rare_tokens)
        ref_tokens = [t.lower() for t in tokenize(ref)]
        hyp_tokens = [t.lower() for t in tokenize(hyp)]

    # Multiplicité : on compte uniquement les rares présents dans la GT
    ref_rare_counts: Counter[str] = Counter(
        t for t in ref_tokens if t in rare_set
    )
    n_rare_in_ref = sum(ref_rare_counts.values())
    if n_rare_in_ref == 0:
        # Audit Classe B : aucun token rare dans la GT ⇒ rappel non
        # applicable ⇒ None (omis en agrégation, pas 0.0).
        return {
            "n_rare_tokens_in_reference": 0,
            "n_rare_tokens_recalled": 0,
            "recall": None,
            "missed_tokens": [],
        }

    # Bag-of-tokens dans hyp pour les tokens rares uniquement
    hyp_rare_counts: Counter[str] = Counter(
        t for t in hyp_tokens if t in rare_set
    )
    # Recall multiplicitaire : pour chaque token, min(ref_count, hyp_count)
    n_recalled = 0
    missed: list[str] = []
    for token, ref_count in ref_rare_counts.items():
        hyp_count = hyp_rare_counts.get(token, 0)
        recalled = min(ref_count, hyp_count)
        n_recalled += recalled
        missed_count = ref_count - recalled
        if missed_count > 0:
            missed.extend([token] * missed_count)

    return {
        "n_rare_tokens_in_reference": n_rare_in_ref,
        "n_rare_tokens_recalled": n_recalled,
        "recall": n_recalled / n_rare_in_ref,
        "missed_tokens": missed,
    }


def rare_token_recall(
    reference: Optional[str],
    hypothesis: Optional[str],
    rare_tokens: Iterable[str],
    *,
    case_sensitive: bool = False,
) -> Optional[float]:
    """Raccourci : rappel ∈ [0, 1], ou ``None`` si la GT n'a aucun
    token rare (non applicable — audit Classe B, omis en agrégation)."""
    return compute_rare_token_recall(
        reference, hypothesis, rare_tokens,
        case_sensitive=case_sensitive,
    )["recall"]


__all__ = [
    "tokenize",
    "frequency_distribution",
    "extract_rare_tokens",
    "compute_rare_token_recall",
    "rare_token_recall",
]
