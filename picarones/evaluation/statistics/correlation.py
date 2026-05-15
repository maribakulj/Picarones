"""Matrice de corrélation entre métriques (Sprint 7).

Coefficient de Pearson entre toutes les métriques numériques d'un
DocumentResult — montre les redondances (CER ↔ WER ≈ 1) et les
dimensions indépendantes (CER ↔ image_quality ≈ 0.5).
"""

from __future__ import annotations

import math
from typing import Optional


def _pearson(x: list[float], y: list[float]) -> float:
    """Coefficient de corrélation de Pearson."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(
        sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)
    )
    return num / den if den > 0 else 0.0


def compute_correlation_matrix(
    metrics_per_doc: list[dict],
    metric_keys: Optional[list[str]] = None,
) -> dict:
    """Calcule la matrice de corrélation entre toutes les métriques numériques.

    Parameters
    ----------
    metrics_per_doc : liste de dicts, un par document, contenant les métriques
    metric_keys     : clés à inclure (None → toutes les clés numériques)

    Returns
    -------
    {
      "labels": [...],
      "matrix": [[r_ij, ...], ...]   // coefficients de Pearson
    }
    """
    if not metrics_per_doc:
        return {"labels": [], "matrix": []}

    if metric_keys is None:
        # Déduire les clés numériques
        sample = metrics_per_doc[0]
        metric_keys = [k for k, v in sample.items() if isinstance(v, (int, float))]

    # Conserver les valeurs brutes (``None`` préservé) — audit
    # scientifique F13 : imputer 0.0 pour une métrique absente
    # déformait le coefficient de Pearson (un document sans CER était
    # traité comme un CER de 0).  On calcule désormais en
    # **pairwise-complete** : pour chaque paire (i, j), seuls les
    # documents où *les deux* métriques sont présentes contribuent.
    raw: dict[str, list[Optional[float]]] = {k: [] for k in metric_keys}
    for doc in metrics_per_doc:
        for k in metric_keys:
            v = doc.get(k)
            raw[k].append(float(v) if isinstance(v, (int, float)) else None)

    labels = metric_keys
    n = len(labels)
    matrix = []
    for i in range(n):
        row = []
        xi = raw[labels[i]]
        for j in range(n):
            yj = raw[labels[j]]
            xs: list[float] = []
            ys: list[float] = []
            for a, b in zip(xi, yj):
                if a is not None and b is not None:
                    xs.append(a)
                    ys.append(b)
            row.append(round(_pearson(xs, ys), 4))
        matrix.append(row)

    return {"labels": labels, "matrix": matrix}


__all__ = ["compute_correlation_matrix"]
