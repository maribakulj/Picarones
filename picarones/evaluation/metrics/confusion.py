"""Matrice de confusion unicode pour l'analyse fine des erreurs OCR.

Pour chaque moteur, on calcule quels caractères du GT sont transcrits par
quels caractères OCR (substitutions). Cette "empreinte d'erreur" est
caractéristique de chaque moteur ou pipeline.

Méthode
-------
L'alignement caractère par caractère utilise la distance de
**Levenshtein** (``rapidfuzz.distance.Levenshtein``, coûts
substitution = insertion = suppression = 1) — le même modèle d'édition
que le CER (jiwer).  Audit scientifique F4 : auparavant l'alignement
passait par ``difflib.SequenceMatcher`` (Ratcliff–Obershelp), qui
maximise les blocs communs et **ne minimise pas** le nombre
d'éditions ; les comptes substitutions/insertions/suppressions et
l'empreinte d'erreur affichés divergeaient alors du CER montré à côté.
L'alignement minimal garantit aussi que tout bloc ``replace`` est de
longueur égale côté GT et côté OCR (substitutions 1-pour-1), ce qui
supprime l'heuristique d'alignement positionnel des segments inégaux.

La matrice est stockée comme un dict de dict :
    ``{gt_char: {ocr_char: count}}``

La valeur spéciale ``"∅"`` (U+2205) représente un caractère vide :
- ``{"a": {"∅": 3}}`` → 'a' supprimé 3 fois dans l'OCR
- ``{"∅": {"x": 2}}`` → 'x' inséré 2 fois dans l'OCR (absent du GT)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein

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

    # Alignement minimal de Levenshtein (audit F4) — cohérent avec le
    # CER.  Sous ce modèle, un bloc ``replace`` est une suite de
    # substitutions 1-pour-1 : longueurs GT et OCR égales, alignement
    # positionnel exact (plus d'heuristique sur segments inégaux).
    for op in Levenshtein.opcodes(ground_truth, hypothesis):
        tag = op.tag
        i1, i2, j1, j2 = (
            op.src_start, op.src_end, op.dest_start, op.dest_end,
        )
        if tag == "equal":
            if not ignore_correct:
                for ch in ground_truth[i1:i2]:
                    if ignore_whitespace and ch in _WHITESPACE:
                        continue
                    matrix[ch][ch] += 1
        elif tag == "replace":
            for g, o in zip(ground_truth[i1:i2], hypothesis[j1:j2]):
                if ignore_whitespace and (g in _WHITESPACE or o in _WHITESPACE):
                    continue
                matrix[g][o] += 1
                n_subs += 1
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
            if (oc != gt_char) and (not exclude_empty or oc != EMPTY_CHAR)
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
