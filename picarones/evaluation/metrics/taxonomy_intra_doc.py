"""Évolution intra-document des classes taxonomiques — Sprint 76 (A.I.4 chantier 2).

A.I.4 chantier 2 du plan d'évolution 2026.

Pourquoi ce module
------------------
La taxonomie d'erreurs (10 classes, ``picarones/core/taxonomy.py``)
est calculée par document mais agrégée en un seul histogramme
global.  ``line_metrics.py`` (Sprint 10) a déjà une heatmap de
**CER par tranche de position** dans le document.  Ce sprint
**étend cette heatmap à toutes les classes taxonomiques** : où
dans le document apparaît tel type d'erreur ?

Lecture concrète : si ``ligature_error`` est concentré dans la
première tranche, c'est une erreur de **marge** (haut de page) ;
si réparti uniformément, c'est une erreur de **scribe**.

Implémentation
--------------
On refait la classification mot-à-mot (cohérent avec
``classify_errors``) en gardant la position du mot GT
(``i1`` dans la diff word-level).  Chaque erreur est binnifiée
selon sa position dans le document (``bin = floor(i1 / n_gt_words *
n_bins)``).

Sortie
------
``compute_taxonomy_position_heatmap(reference, hypothesis,
n_bins=10)`` retourne un dict ``{class_name: list[float]}`` où
chaque liste a ``n_bins`` valeurs représentant le **compte**
d'erreurs de cette classe dans la tranche correspondante.

Stratégie de découpage
----------------------
Couche de calcul + rendu HTML bout-en-bout, comme Sprint 75.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Optional

from rapidfuzz.distance import Levenshtein

from picarones.evaluation.metrics.taxonomy import (
    ERROR_CLASSES,
    _is_abbreviation_error,
    _is_diacritic_error,
    _is_ligature_error,
    _is_oov_word,
    _is_visual_confusion,
)

logger = logging.getLogger(__name__)


def _classify_word_pair(gt_word: str, hyp_word: str) -> str:
    """Retourne la classe taxonomique d'une erreur mot-à-mot.

    Reproduit la logique de ``taxonomy._classify_word_error`` sans
    modifier ses compteurs internes — utile pour avoir
    ``(position, class)`` paire.
    """
    if gt_word.casefold() == hyp_word.casefold() and gt_word != hyp_word:
        return "case_error"
    gt_norm = unicodedata.normalize("NFC", gt_word)
    hyp_norm = unicodedata.normalize("NFC", hyp_word)
    if _is_ligature_error(gt_norm, hyp_norm):
        return "ligature_error"
    if _is_abbreviation_error(gt_norm, hyp_norm):
        return "abbreviation_error"
    if _is_diacritic_error(gt_norm, hyp_norm):
        return "diacritic_error"
    if _is_visual_confusion(gt_norm, hyp_norm):
        return "visual_confusion"
    if _is_oov_word(hyp_word):
        return "oov_character"
    return "hapax"


def _bin_for_position(position: int, total: int, n_bins: int) -> int:
    """Retourne l'index de bin pour une position (0-based) sur un
    total de mots GT.  Garde-fou sur les bornes : si position == total
    (peut arriver pour insert en fin de doc), on clip au dernier bin.
    """
    if total <= 0 or n_bins <= 0:
        return 0
    bin_idx = int((position / total) * n_bins)
    if bin_idx >= n_bins:
        bin_idx = n_bins - 1
    if bin_idx < 0:
        bin_idx = 0
    return bin_idx


def compute_taxonomy_position_heatmap(
    reference: Optional[str],
    hypothesis: Optional[str],
    *,
    n_bins: int = 10,
) -> Optional[dict]:
    """Calcule la heatmap class × position pour un document.

    Parameters
    ----------
    reference:
        Texte GT du document.
    hypothesis:
        Texte produit par l'OCR.
    n_bins:
        Nombre de tranches de position (défaut 10, cohérent avec
        ``line_metrics.heatmap``).

    Returns
    -------
    Optional[dict]
        ``{
            "n_bins": int,
            "n_words_gt": int,           # nb mots GT
            "total_errors": int,         # somme sur toutes classes
            "per_class": {
                class_name: list[int],  # n_bins valeurs (compte par bin)
            },
            "totals_per_bin": list[int], # nb total d'erreurs par bin
        }``
        Ou ``None`` si la GT est vide.
    """
    if n_bins <= 0:
        raise ValueError("n_bins doit être > 0")
    ref = reference or ""
    hyp = hypothesis or ""
    gt_words = ref.split()
    hyp_words = hyp.split()
    n_gt = len(gt_words)
    if n_gt == 0:
        return None

    per_class: dict[str, list[int]] = {
        cls: [0] * n_bins for cls in ERROR_CLASSES
    }
    totals_per_bin: list[int] = [0] * n_bins
    total_errors = 0

    # Alignement minimal de Levenshtein (audit F14 : cohérent avec le
    # WER ; sous ce modèle ``replace`` est toujours de longueur égale,
    # donc plus de bloc inégal dont on abandonnait la classification
    # des substitutions au profit du seul écart de longueur).
    for op in Levenshtein.opcodes(gt_words, hyp_words):
        tag = op.tag
        i1, i2, j1, j2 = (
            op.src_start, op.src_end, op.dest_start, op.dest_end,
        )
        if tag == "equal":
            continue
        if tag == "delete":
            for offset in range(i2 - i1):
                position = i1 + offset
                bin_idx = _bin_for_position(position, n_gt, n_bins)
                per_class["lacuna"][bin_idx] += 1
                totals_per_bin[bin_idx] += 1
                total_errors += 1
        elif tag == "insert":
            # Tout mot inséré est une erreur (OOV → classe 8, sinon
            # hapax → classe 6) : auparavant les insertions non-OOV
            # n'étaient pas comptées (sous-comptage systématique).
            for w in hyp_words[j1:j2]:
                position = min(i1, n_gt - 1) if n_gt else 0
                bin_idx = _bin_for_position(position, n_gt, n_bins)
                cls = "oov_character" if _is_oov_word(w) else "hapax"
                per_class[cls][bin_idx] += 1
                totals_per_bin[bin_idx] += 1
                total_errors += 1
        elif tag == "replace":
            for offset, (gt_w, hyp_w) in enumerate(
                zip(gt_words[i1:i2], hyp_words[j1:j2]),
            ):
                if gt_w == hyp_w:
                    continue
                position = i1 + offset
                bin_idx = _bin_for_position(position, n_gt, n_bins)
                cls = _classify_word_pair(gt_w, hyp_w)
                per_class[cls][bin_idx] += 1
                totals_per_bin[bin_idx] += 1
                total_errors += 1

    return {
        "n_bins": n_bins,
        "n_words_gt": n_gt,
        "total_errors": total_errors,
        "per_class": per_class,
        "totals_per_bin": totals_per_bin,
    }


__all__ = [
    "compute_taxonomy_position_heatmap",
]
