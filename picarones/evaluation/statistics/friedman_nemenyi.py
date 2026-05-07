"""Test de Friedman + post-hoc Nemenyi (Sprint 17).

Référence : Demšar, J. (2006), "Statistical Comparisons of Classifiers
over Multiple Data Sets", Journal of Machine Learning Research 7:1-30.
Standard de facto pour comparer plusieurs systèmes sur plusieurs
datasets — ici plusieurs moteurs OCR sur plusieurs documents.

Le rendu visuel canonique (Critical Difference Diagram) vit dans
:mod:`picarones.measurements.statistics.cdd_render` pour séparer
calcul (ce module) et présentation (l'autre).
"""

from __future__ import annotations

import math
from typing import Optional

from picarones.evaluation.statistics.wilcoxon import _normal_sf

# Valeurs critiques de la distribution du Studentized Range divisées par √2,
# pour df = ∞ (approximation usuelle pour Nemenyi). Source : tables de Tukey.
# Clé : nombre de traitements k ; valeur : q_α pour α ∈ {0.05, 0.01}.
_NEMENYI_Q_TABLE = {
    # k   q_0.05   q_0.01
    2:  (1.960, 2.576),
    3:  (2.343, 2.913),
    4:  (2.569, 3.113),
    5:  (2.728, 3.255),
    6:  (2.850, 3.364),
    7:  (2.949, 3.452),
    8:  (3.031, 3.526),
    9:  (3.102, 3.590),
    10: (3.164, 3.646),
    11: (3.219, 3.696),
    12: (3.268, 3.741),
    13: (3.313, 3.781),
    14: (3.354, 3.818),
    15: (3.391, 3.853),
    16: (3.426, 3.886),
    17: (3.458, 3.916),
    18: (3.489, 3.944),
    19: (3.517, 3.970),
    20: (3.544, 3.995),
    25: (3.658, 4.095),
    30: (3.739, 4.167),
    40: (3.858, 4.272),
    50: (3.945, 4.349),
}


def _chi_square_sf(x: float, df: int) -> float:
    """Survival function de la loi chi², 1 - CDF(x).

    Utilise scipy si disponible (méthode exacte), sinon Wilson-Hilferty
    (approximation normale précise dès df ≥ 3).
    """
    if x <= 0 or df <= 0:
        return 1.0
    try:
        from scipy.stats import chi2 as _chi2  # type: ignore[import-untyped]
        return float(_chi2.sf(x, df))
    except ImportError:
        pass
    # Wilson-Hilferty : transforme chi² en approximation normale
    z = (((x / df) ** (1.0 / 3.0)) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return _normal_sf(z)


def _rank_row(values: list[float]) -> list[float]:
    """Rangs d'une ligne — petit = rang 1. Ex-aequo : rangs moyens."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j
    return ranks


def _aligned_cer_matrix(
    engine_cer_map: dict[str, list[float]],
) -> tuple[list[str], list[list[float]]]:
    """Construit la matrice (k moteurs × n documents) alignée sur la longueur
    minimale. Retourne ``(noms, matrice_colonne_par_moteur)``.

    Friedman exige des blocs (documents) complets : si les moteurs n'ont pas
    tous été exécutés sur les mêmes documents, on tronque à la longueur
    minimale, documentée dans le résultat via ``n_blocks``.
    """
    names = list(engine_cer_map.keys())
    if not names:
        return [], []
    min_len = min(len(v) for v in engine_cer_map.values())
    if min_len == 0:
        return names, []
    matrix = [engine_cer_map[n][:min_len] for n in names]
    return names, matrix


def friedman_test(engine_cer_map: dict[str, list[float]]) -> dict:
    """Test de Friedman — k moteurs sur n documents appariés.

    Test non-paramétrique équivalent à l'ANOVA à mesures répétées pour des
    données ordinales. Hypothèse nulle : tous les moteurs ont la même
    performance moyenne. Rejet → au moins un moteur diffère des autres.

    Parameters
    ----------
    engine_cer_map:
        Dict ``{engine_name → [cer_doc1, cer_doc2, ...]}``. Tous les moteurs
        doivent avoir été évalués sur les mêmes documents (dans le même ordre).

    Returns
    -------
    dict avec :
      - ``statistic``     : Q corrigé pour les ex-aequo
      - ``p_value``       : p-value (scipy si dispo, sinon Wilson-Hilferty)
      - ``significant``   : bool, p < 0.05
      - ``df``            : degrés de liberté = k - 1
      - ``n_blocks``      : nombre de documents (blocs) utilisés
      - ``n_engines``     : nombre de moteurs (k)
      - ``mean_ranks``    : dict ``{engine: rang_moyen}``
      - ``interpretation``: phrase lisible
      - ``error``         : message si le test n'est pas applicable
    """
    names, matrix = _aligned_cer_matrix(engine_cer_map)
    k = len(names)
    n = len(matrix[0]) if matrix else 0

    if k < 2:
        return {
            "statistic": 0.0, "p_value": 1.0, "significant": False,
            "df": 0, "n_blocks": n, "n_engines": k,
            "mean_ranks": {names[0]: 1.0} if k == 1 else {},
            "interpretation": "Test de Friedman non applicable : il faut au moins 2 moteurs.",
            "error": "not_enough_engines",
        }
    if n < 2:
        return {
            "statistic": 0.0, "p_value": 1.0, "significant": False,
            "df": k - 1, "n_blocks": n, "n_engines": k,
            "mean_ranks": {name: 1.0 for name in names},
            "interpretation": "Test de Friedman non applicable : il faut au moins 2 documents communs.",
            "error": "not_enough_blocks",
        }

    # Rangs par bloc (document) : pour chaque doc, ranger les k moteurs
    ranks_by_engine: list[list[float]] = [[] for _ in range(k)]
    for j in range(n):
        row = [matrix[i][j] for i in range(k)]
        row_ranks = _rank_row(row)
        for i in range(k):
            ranks_by_engine[i].append(row_ranks[i])

    rank_sums = [sum(r) for r in ranks_by_engine]
    mean_ranks = {names[i]: rank_sums[i] / n for i in range(k)}

    # Statistique Q non-corrigée (sans ex-aequo)
    #   Q = 12 / (n·k·(k+1)) · Σ R_j² − 3·n·(k+1)
    Q = (12.0 / (n * k * (k + 1))) * sum(rs ** 2 for rs in rank_sums) - 3.0 * n * (k + 1)

    # Correction pour les ex-aequo (ties factor) — ajuste si des rangs sont
    # partagés dans certains blocs. Formule : Q_corr = Q / (1 - T/(n·(k³−k)))
    # où T = Σ (tⱼ³ − tⱼ) sur tous les groupes d'ex-aequo.
    tie_correction = 0.0
    for j in range(n):
        row = [matrix[i][j] for i in range(k)]
        sorted_row = sorted(row)
        i = 0
        while i < len(sorted_row):
            count = 1
            while i + count < len(sorted_row) and sorted_row[i + count] == sorted_row[i]:
                count += 1
            if count > 1:
                tie_correction += count ** 3 - count
            i += count
    denom = 1.0 - tie_correction / (n * (k ** 3 - k)) if k >= 2 else 1.0
    if denom > 0:
        Q = Q / denom

    df = k - 1
    p_value = _chi_square_sf(Q, df)
    significant = p_value < 0.05

    if significant:
        interpretation = (
            f"Test de Friedman significatif (Q = {Q:.3f}, df = {df}, p = {p_value:.4f}). "
            f"Au moins un moteur diffère des autres — utiliser le post-hoc Nemenyi "
            f"pour identifier les paires distinguables."
        )
    else:
        interpretation = (
            f"Test de Friedman non significatif (Q = {Q:.3f}, df = {df}, p = {p_value:.4f}). "
            f"Aucune différence globale détectée entre les moteurs sur ce corpus."
        )

    return {
        "statistic": round(Q, 4),
        "p_value": round(p_value, 6),
        "significant": significant,
        "df": df,
        "n_blocks": n,
        "n_engines": k,
        "mean_ranks": {k_: round(v, 4) for k_, v in mean_ranks.items()},
        "interpretation": interpretation,
    }


def _nemenyi_critical_value(k: int, alpha: float = 0.05) -> Optional[float]:
    """Valeur critique q_α pour k traitements, df = ∞.

    Retourne ``None`` si k est hors table (< 2 ou > 50).
    """
    if k < 2:
        return None
    if k in _NEMENYI_Q_TABLE:
        q05, q01 = _NEMENYI_Q_TABLE[k]
        return q05 if alpha == 0.05 else q01 if alpha == 0.01 else q05
    # Au-delà de la table : borne supérieure (conservateur)
    max_k = max(_NEMENYI_Q_TABLE.keys())
    if k > max_k:
        q05, q01 = _NEMENYI_Q_TABLE[max_k]
        return q05 if alpha == 0.05 else q01
    # Entre deux clés : interpolation linéaire
    keys = sorted(_NEMENYI_Q_TABLE.keys())
    for i in range(len(keys) - 1):
        if keys[i] < k < keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            q_lo = _NEMENYI_Q_TABLE[lo][0 if alpha == 0.05 else 1]
            q_hi = _NEMENYI_Q_TABLE[hi][0 if alpha == 0.05 else 1]
            frac = (k - lo) / (hi - lo)
            return q_lo + frac * (q_hi - q_lo)
    return None


def nemenyi_posthoc(
    engine_cer_map: dict[str, list[float]],
    alpha: float = 0.05,
) -> dict:
    """Post-hoc de Nemenyi — identifie les paires de moteurs statistiquement
    indiscernables après un test de Friedman.

    Calcule la *critical distance* CD = q_α · √(k·(k+1) / (6·n)). Deux moteurs
    dont les rangs moyens diffèrent de moins que CD ne sont **pas**
    statistiquement distinguables au seuil α.

    Returns
    -------
    dict avec :
      - ``alpha``               : seuil utilisé
      - ``critical_distance``   : CD calculée
      - ``q_alpha``             : valeur critique q_α issue de la table
      - ``n_blocks``, ``n_engines``
      - ``mean_ranks``          : rangs moyens par moteur (dict)
      - ``engines_sorted``      : liste des moteurs triés par rang croissant
      - ``significant_matrix``  : matrice bool (list[list[bool]]),
                                  ``True`` = paire significativement différente
      - ``tied_groups``         : liste de listes de moteurs indiscernables
                                  (groupes maximaux d'ex-aequo pratiques)
      - ``error``               : présent si le test n'est pas applicable
    """
    names, matrix = _aligned_cer_matrix(engine_cer_map)
    k = len(names)
    n = len(matrix[0]) if matrix else 0

    if k < 2 or n < 2:
        return {
            "alpha": alpha,
            "critical_distance": 0.0,
            "q_alpha": 0.0,
            "n_blocks": n,
            "n_engines": k,
            "mean_ranks": {name: 1.0 for name in names},
            "engines_sorted": list(names),
            "significant_matrix": [[False] * k for _ in range(k)],
            "tied_groups": [list(names)] if names else [],
            "error": "not_enough_data",
        }

    # Friedman fournit les rangs moyens — on les recalcule ici pour rester
    # autonome (sans forcer l'utilisateur à chaîner les deux appels).
    ranks_by_engine: list[list[float]] = [[] for _ in range(k)]
    for j in range(n):
        row = [matrix[i][j] for i in range(k)]
        row_ranks = _rank_row(row)
        for i in range(k):
            ranks_by_engine[i].append(row_ranks[i])

    mean_ranks_list = [sum(r) / n for r in ranks_by_engine]
    mean_ranks = {names[i]: round(mean_ranks_list[i], 4) for i in range(k)}

    q_alpha = _nemenyi_critical_value(k, alpha) or 0.0
    critical_distance = q_alpha * math.sqrt(k * (k + 1) / (6.0 * n))

    # Matrice de significativité : paire (i,j) significative si |R_i - R_j| > CD
    significant_matrix = [
        [
            (i != j) and (abs(mean_ranks_list[i] - mean_ranks_list[j]) > critical_distance)
            for j in range(k)
        ]
        for i in range(k)
    ]

    # Groupes d'ex-aequo pratiques : fenêtre glissante sur les rangs triés.
    # Deux moteurs sont dans le même groupe si leur écart ≤ CD.
    order = sorted(range(k), key=lambda i: mean_ranks_list[i])
    sorted_names = [names[i] for i in order]
    sorted_ranks = [mean_ranks_list[i] for i in order]

    tied_groups: list[list[str]] = []
    i = 0
    while i < len(sorted_names):
        # étendre le groupe tant que le moteur suivant est à ≤ CD du premier du groupe
        j = i
        while j + 1 < len(sorted_names) and (sorted_ranks[j + 1] - sorted_ranks[i]) <= critical_distance:
            j += 1
        tied_groups.append(sorted_names[i:j + 1])
        i = j + 1 if j > i else i + 1

    return {
        "alpha": alpha,
        "critical_distance": round(critical_distance, 4),
        "q_alpha": round(q_alpha, 4),
        "n_blocks": n,
        "n_engines": k,
        "mean_ranks": mean_ranks,
        "engines_sorted": sorted_names,
        "significant_matrix": significant_matrix,
        "tied_groups": tied_groups,
    }


__all__ = [
    # Symboles publics.
    "friedman_test",
    "nemenyi_posthoc",
    # Symboles privés ré-exportés (consommés par les tests Sprint 18).
    # Note : ``_aligned_cer_matrix`` reste strictement interne au module
    # (utilisé seulement par friedman_test et nemenyi_posthoc) ; il n'est
    # ni dans __all__ ni ré-exporté par le __init__.py du sous-package.
    "_chi_square_sf",
    "_nemenyi_critical_value",
    "_rank_row",
]
