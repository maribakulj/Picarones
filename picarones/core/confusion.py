"""Matrice de confusion unicode pour l'analyse fine des erreurs OCR.

Pour chaque moteur, on calcule quels caractères du GT sont transcrits par
quels caractères OCR (substitutions). Cette "empreinte d'erreur" est
caractéristique de chaque moteur ou pipeline.

Méthode
-------
L'alignement caractère par caractère utilise les opérations d'édition
de la distance de Levenshtein (via difflib.SequenceMatcher), ce qui permet
d'identifier les substitutions, insertions et suppressions.

La matrice est stockée comme un dict de dict :
    ``{gt_char: {ocr_char: count}}``

La valeur spéciale ``"∅"`` (U+2205) représente un caractère vide :
- ``{"a": {"∅": 3}}`` → 'a' supprimé 3 fois dans l'OCR
- ``{"∅": {"x": 2}}`` → 'x' inséré 2 fois dans l'OCR (absent du GT)
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# Symbole représentant un caractère absent (insertion / suppression)
EMPTY_CHAR = "∅"

# Caractères non pertinents à ignorer dans la matrice (espaces, sauts de ligne)
_WHITESPACE = set(" \t\n\r")


@dataclass
class ConfusionMatrix:
    """Matrice de confusion unicode pour une paire (GT, OCR)."""

    matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    """Clé externe = char GT ; clé interne = char OCR ; valeur = count."""

    total_substitutions: int = 0
    total_insertions: int = 0
    total_deletions: int = 0

    @property
    def total_errors(self) -> int:
        return self.total_substitutions + self.total_insertions + self.total_deletions

    def top_confusions(self, n: int = 20) -> list[dict]:
        """Retourne les n confusions les plus fréquentes (substitutions uniquement)."""
        pairs: list[tuple[str, str, int]] = []
        for gt_char, ocr_counts in self.matrix.items():
            if gt_char == EMPTY_CHAR:
                continue  # insertions
            for ocr_char, count in ocr_counts.items():
                if ocr_char == EMPTY_CHAR:
                    continue  # suppressions
                if gt_char != ocr_char:
                    pairs.append((gt_char, ocr_char, count))
        pairs.sort(key=lambda x: -x[2])
        return [
            {"gt": gt, "ocr": ocr, "count": cnt}
            for gt, ocr, cnt in pairs[:n]
        ]

    def as_compact_dict(self, min_count: int = 1) -> dict:
        """Sérialise la matrice en éliminant les entrées rares."""
        compact: dict[str, dict[str, int]] = {}
        for gt_char, ocr_counts in self.matrix.items():
            filtered = {
                oc: cnt for oc, cnt in ocr_counts.items()
                if cnt >= min_count
            }
            if filtered:
                compact[gt_char] = filtered
        return {
            "matrix": compact,
            "total_substitutions": self.total_substitutions,
            "total_insertions": self.total_insertions,
            "total_deletions": self.total_deletions,
        }

    def as_dict(self) -> dict:
        return self.as_compact_dict(min_count=1)


def build_confusion_matrix(
    ground_truth: str,
    hypothesis: str,
    ignore_whitespace: bool = True,
    ignore_correct: bool = True,
) -> ConfusionMatrix:
    """Construit la matrice de confusion unicode pour une paire GT/OCR.

    Parameters
    ----------
    ground_truth:
        Texte de référence (vérité terrain).
    hypothesis:
        Texte produit par l'OCR.
    ignore_whitespace:
        Si True, ignore les espaces, tabulations et sauts de ligne.
    ignore_correct:
        Si True, n'enregistre pas les paires identiques (gt_char == ocr_char).
        Par défaut True pour réduire la taille de la matrice.

    Returns
    -------
    ConfusionMatrix
    """
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_subs = n_ins = n_dels = 0

    if not ground_truth and not hypothesis:
        return ConfusionMatrix(dict(matrix), 0, 0, 0)

    # SequenceMatcher sur listes de chars pour un alignement précis
    matcher = difflib.SequenceMatcher(None, ground_truth, hypothesis, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            if not ignore_correct:
                for ch in ground_truth[i1:i2]:
                    if ignore_whitespace and ch in _WHITESPACE:
                        continue
                    matrix[ch][ch] += 1
        elif tag == "replace":
            # Aligner char par char les séquences de longueurs différentes
            gt_seg = ground_truth[i1:i2]
            oc_seg = hypothesis[j1:j2]
            _align_segments(gt_seg, oc_seg, matrix, ignore_whitespace)
            # Comptabiliser grossièrement (alignement sous-optimal possible)
            n_subs += max(len(gt_seg), len(oc_seg))
        elif tag == "delete":
            for ch in ground_truth[i1:i2]:
                if ignore_whitespace and ch in _WHITESPACE:
                    continue
                matrix[ch][EMPTY_CHAR] += 1
                n_dels += 1
        elif tag == "insert":
            for ch in hypothesis[j1:j2]:
                if ignore_whitespace and ch in _WHITESPACE:
                    continue
                matrix[EMPTY_CHAR][ch] += 1
                n_ins += 1

    # Convertir defaultdict en dict normal
    result_matrix: dict[str, dict[str, int]] = {
        k: dict(v) for k, v in matrix.items()
    }

    return ConfusionMatrix(
        matrix=result_matrix,
        total_substitutions=n_subs,
        total_insertions=n_ins,
        total_deletions=n_dels,
    )


def _align_segments(
    gt_seg: str,
    oc_seg: str,
    matrix: dict,
    ignore_whitespace: bool,
) -> None:
    """Aligne deux segments de longueurs potentiellement différentes."""
    if not gt_seg:
        for ch in oc_seg:
            if ignore_whitespace and ch in _WHITESPACE:
                continue
            matrix[EMPTY_CHAR][ch] += 1
        return
    if not oc_seg:
        for ch in gt_seg:
            if ignore_whitespace and ch in _WHITESPACE:
                continue
            matrix[ch][EMPTY_CHAR] += 1
        return

    if len(gt_seg) == len(oc_seg):
        # Substitutions 1-pour-1
        for g, o in zip(gt_seg, oc_seg):
            if ignore_whitespace and (g in _WHITESPACE or o in _WHITESPACE):
                continue
            matrix[g][o] += 1
    else:
        # Longueurs différentes : utiliser SequenceMatcher récursif sur segments courts
        sub = difflib.SequenceMatcher(None, gt_seg, oc_seg, autojunk=False)
        for tag2, i1, i2, j1, j2 in sub.get_opcodes():
            if tag2 == "equal":
                pass
            elif tag2 == "replace":
                # Régression simple : aligner par troncature
                for g, o in zip(gt_seg[i1:i2], oc_seg[j1:j2]):
                    if ignore_whitespace and (g in _WHITESPACE or o in _WHITESPACE):
                        continue
                    matrix[g][o] += 1
            elif tag2 == "delete":
                for g in gt_seg[i1:i2]:
                    if ignore_whitespace and g in _WHITESPACE:
                        continue
                    matrix[g][EMPTY_CHAR] += 1
            elif tag2 == "insert":
                for o in oc_seg[j1:j2]:
                    if ignore_whitespace and o in _WHITESPACE:
                        continue
                    matrix[EMPTY_CHAR][o] += 1


def aggregate_confusion_matrices(matrices: list[ConfusionMatrix]) -> ConfusionMatrix:
    """Agrège plusieurs matrices de confusion en une seule.

    Utile pour obtenir la matrice agrégée sur l'ensemble du corpus.
    """
    combined: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_subs = total_ins = total_dels = 0

    for cm in matrices:
        for gt_char, ocr_counts in cm.matrix.items():
            for ocr_char, count in ocr_counts.items():
                combined[gt_char][ocr_char] += count
        total_subs += cm.total_substitutions
        total_ins += cm.total_insertions
        total_dels += cm.total_deletions

    return ConfusionMatrix(
        matrix={k: dict(v) for k, v in combined.items()},
        total_substitutions=total_subs,
        total_insertions=total_ins,
        total_deletions=total_dels,
    )


def top_confused_chars(
    matrix: ConfusionMatrix,
    n: int = 15,
    exclude_empty: bool = True,
) -> list[dict]:
    """Retourne les caractères GT les plus souvent confondus.

    Retourne une liste triée par nombre total d'erreurs décroissant :
    ``[{"char": "ſ", "total_errors": 47, "top_substitutes": [...]}, ...]``
    """
    char_stats: dict[str, dict] = {}
    for gt_char, ocr_counts in matrix.matrix.items():
        if exclude_empty and gt_char == EMPTY_CHAR:
            continue
        error_count = sum(
            cnt for oc, cnt in ocr_counts.items()
            if (oc != gt_char) and (not exclude_empty or oc != EMPTY_CHAR or True)
        )
        if error_count > 0:
            top_subs = sorted(
                [{"ocr": oc, "count": cnt} for oc, cnt in ocr_counts.items() if oc != gt_char],
                key=lambda x: -x["count"],
            )[:5]
            char_stats[gt_char] = {
                "char": gt_char,
                "total_errors": error_count,
                "top_substitutes": top_subs,
            }

    return sorted(char_stats.values(), key=lambda x: -x["total_errors"])[:n]
