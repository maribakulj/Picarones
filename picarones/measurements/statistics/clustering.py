"""Clustering des patterns d'erreurs (Sprint 7).

Regroupe les substitutions OCR/HTR fréquentes en clusters lisibles
(« confusion rn/m », « ligature œ/æ », etc.) pour le rapport HTML.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from picarones.core.diff_utils import compute_word_diff

# Patterns d'erreurs fréquentes (OCR + HTR documents patrimoniaux)
_ERROR_PATTERNS = [
    # (pattern_re, label)
    (r"\brn\b.*\bm\b|\bm\b.*\brn\b|rn→m|m→rn",       "confusion rn/m"),
    (r"[lI]→1|1→[lI]|l→1|1→l|I→1|1→I",               "confusion l/1/I"),
    (r"u→n|n→u|v→u|u→v",                              "confusion u/n/v"),
    (r"[oO]→0|0→[oO]",                                "confusion O/0"),
    (r"ſ→[fs]|[fs]→ſ",                                "confusion ſ/f/s"),
    (r"é→e|è→e|ê→e|e→[éèê]",                          "erreur diacritique é/e"),
    (r"œ→oe|oe→œ|æ→ae|ae→æ",                          "ligature œ/æ"),
    (r"[fF]i→fi|fi→[fF]i",                            "ligature fi"),
    (r"[fF]l→fl|fl→[fF]l",                            "ligature fl"),
    (r"\s+→''|''→\s+",                                "segmentation espace"),
]


def _extract_error_pairs(gt: str, hyp: str) -> list[tuple[str, str]]:
    """Extrait les paires (gt_char_seq, hyp_char_seq) d'erreurs de substitution.

    L'import de ``compute_word_diff`` est au top-level du module
    (cercle 1 → cercle 2, sens autorisé). Il était paresseux historiquement
    pour contourner une violation de cercle (Sprint A3) qui n'existe plus.
    """
    ops = compute_word_diff(gt, hyp)
    pairs = []
    for op in ops:
        if op["op"] == "replace":
            pairs.append((op["old"], op["new"]))
        elif op["op"] == "delete":
            pairs.append((op["text"], ""))
        elif op["op"] == "insert":
            pairs.append(("", op["text"]))
    return pairs


@dataclass
class ErrorCluster:
    """Un cluster d'erreurs similaires."""
    cluster_id: int
    label: str
    """Description humaine du pattern (ex. 'confusion rn/m')."""
    count: int
    examples: list[dict]
    """Liste de {engine, gt_fragment, ocr_fragment}."""

    def as_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "count": self.count,
            "examples": self.examples[:5],  # 5 exemples max
        }


def cluster_errors(
    error_data: list[dict],
    max_clusters: int = 8,
) -> list[ErrorCluster]:
    """Regroupe les erreurs en clusters avec labels lisibles.

    Parameters
    ----------
    error_data : liste de dicts {engine, gt, hypothesis}
    max_clusters : nombre max de clusters à retourner

    Returns
    -------
    Liste de ErrorCluster triée par count décroissant.
    """
    # Collecter tous les patterns d'erreur avec contexte
    # Clé : catégorie d'erreur → liste d'exemples
    bucket: dict[str, list[dict]] = defaultdict(list)
    other_pairs: list[dict] = []

    for item in error_data:
        engine = item.get("engine", "")
        gt = item.get("gt", "")
        hyp = item.get("hypothesis", "")
        pairs = _extract_error_pairs(gt, hyp)

        for old, new in pairs:
            if not old and not new:
                continue
            matched = False
            # Essayer de matcher un pattern connu
            probe = f"{old}→{new}"
            for _pat, label in _ERROR_PATTERNS:
                try:
                    if re.search(_pat, probe, re.IGNORECASE):
                        bucket[label].append({
                            "engine": engine,
                            "gt_fragment": old,
                            "ocr_fragment": new,
                        })
                        matched = True
                        break
                except re.error:
                    pass

            if not matched:
                # Regrouper les substitutions restantes par paire de caractères
                if len(old) <= 3 and len(new) <= 3:
                    key = f"{old}→{new}" if (old and new) else (f"—→{new}" if new else f"{old}→—")
                    bucket[key].append({
                        "engine": engine,
                        "gt_fragment": old,
                        "ocr_fragment": new,
                    })
                else:
                    other_pairs.append({
                        "engine": engine,
                        "gt_fragment": old,
                        "ocr_fragment": new,
                    })

    # Construire les clusters triés par fréquence
    clusters: list[ErrorCluster] = []
    cluster_id = 1
    sorted_buckets = sorted(bucket.items(), key=lambda x: -len(x[1]))

    for label, examples in sorted_buckets[:max_clusters - 1]:
        clusters.append(ErrorCluster(
            cluster_id=cluster_id,
            label=label,
            count=len(examples),
            examples=examples,
        ))
        cluster_id += 1

    # Cluster "autres"
    if other_pairs:
        clusters.append(ErrorCluster(
            cluster_id=cluster_id,
            label="autres substitutions",
            count=len(other_pairs),
            examples=other_pairs,
        ))

    # Trier par count décroissant et limiter
    clusters.sort(key=lambda c: -c.count)
    return clusters[:max_clusters]


__all__ = ["ErrorCluster", "cluster_errors"]
