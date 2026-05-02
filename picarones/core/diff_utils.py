"""Calcul du diff mot-à-mot entre vérité terrain et sortie OCR.

Produit une liste d'opérations sérialisables en JSON, consommée
par le rendu JS dans le rapport HTML.

Opérations possibles
--------------------
{"op": "equal",   "text": "mot"}
{"op": "insert",  "text": "mot"}    -- présent dans l'OCR mais pas dans la GT
{"op": "delete",  "text": "mot"}    -- présent dans la GT mais pas dans l'OCR
{"op": "replace", "old": "…", "new": "…"}  -- substitution (orange)
"""

from __future__ import annotations

import difflib
import re
from typing import Any


def _tokenize(text: str) -> list[str]:
    """Découpe le texte en tokens (mots + ponctuation + espaces)."""
    # Conserver les espaces comme tokens pour un rendu fidèle
    return re.split(r"(\s+)", text)


def compute_word_diff(reference: str, hypothesis: str) -> list[dict[str, Any]]:
    """Calcule un diff mot-à-mot entre deux textes.

    Parameters
    ----------
    reference:
        Texte de vérité terrain.
    hypothesis:
        Texte produit par le moteur OCR.

    Returns
    -------
    list of dict
        Séquence d'opérations : equal, insert, delete, replace.
    """
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens, autojunk=False)
    ops: list[dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_chunk = " ".join(ref_tokens[i1:i2])
        hyp_chunk = " ".join(hyp_tokens[j1:j2])

        if tag == "equal":
            ops.append({"op": "equal", "text": ref_chunk})
        elif tag == "insert":
            ops.append({"op": "insert", "text": hyp_chunk})
        elif tag == "delete":
            ops.append({"op": "delete", "text": ref_chunk})
        elif tag == "replace":
            ops.append({"op": "replace", "old": ref_chunk, "new": hyp_chunk})

    return ops


def compute_char_diff(reference: str, hypothesis: str) -> list[dict[str, Any]]:
    """Diff caractère par caractère — utile pour les tokens courts."""
    matcher = difflib.SequenceMatcher(None, list(reference), list(hypothesis), autojunk=False)
    ops: list[dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_chunk = reference[i1:i2]
        hyp_chunk = hypothesis[j1:j2]
        if tag == "equal":
            ops.append({"op": "equal", "text": ref_chunk})
        elif tag == "insert":
            ops.append({"op": "insert", "text": hyp_chunk})
        elif tag == "delete":
            ops.append({"op": "delete", "text": ref_chunk})
        elif tag == "replace":
            ops.append({"op": "replace", "old": ref_chunk, "new": hyp_chunk})

    return ops


def diff_stats(ops: list[dict[str, Any]]) -> dict[str, int]:
    """Compte le nombre d'insertions, suppressions et substitutions."""
    stats = {"equal": 0, "insert": 0, "delete": 0, "replace": 0}
    for op in ops:
        stats[op["op"]] += 1
    return stats
