"""Tests statistiques et clustering d'erreurs pour Picarones.

Fonctions fournies
------------------
- wilcoxon_test(a, b)                  : Wilcoxon signé-rangé (2 moteurs appariés)
- bootstrap_ci(values, ...)            : intervalle de confiance à 95 % par bootstrap
- compute_pairwise_stats(...)          : matrice de Wilcoxon entre toutes les paires
- friedman_test(engine_cer_map)        : Friedman (k moteurs, n documents)       [Sprint 17]
- nemenyi_posthoc(engine_cer_map)      : post-hoc Nemenyi avec critical distance [Sprint 17]
- build_critical_difference_svg(...)   : rendu SVG du CDD (Demšar 2006)          [Sprint 17]
- compute_pareto_front(points, ...)    : frontière de Pareto multi-objectifs     [Sprint 19]
- cluster_errors(...)                  : regroupement des patterns d'erreurs
- compute_correlation_matrix(...)      : matrice de corrélation des métriques
- compute_reliability_curve(...)       : courbe CER vs. % docs les plus faciles
- compute_venn_data(...)               : diagramme de Venn 2/3 moteurs
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

# Import optionnel de scipy — utilisé pour le test de Wilcoxon si disponible
# (méthode exacte pour n ≤ 25, approximation normale pour n > 25).
# En son absence, l'implémentation native (approximation normale pour n ≥ 10)
# est utilisée automatiquement.
try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon  # type: ignore[import-untyped]
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Intervalle de confiance par bootstrap.

    Parameters
    ----------
    values : liste des valeurs (ex. CER par document)
    n_iter : nombre d'itérations bootstrap (défaut 1000)
    ci     : niveau de confiance (défaut 0.95 → 95 %)
    seed   : graine RNG pour reproductibilité

    Returns
    -------
    (lower, upper) — les bornes de l'IC à ``ci`` %
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_iter):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = max(0, int(alpha * n_iter))
    hi_idx = min(n_iter - 1, int((1.0 - alpha) * n_iter))
    return (means[lo_idx], means[hi_idx])


# ---------------------------------------------------------------------------
# Test de Wilcoxon signé-rangé (implémentation pure Python)
# ---------------------------------------------------------------------------

def wilcoxon_test(
    a: list[float],
    b: list[float],
    zero_method: str = "wilcox",
) -> dict:
    """Test de Wilcoxon signé-rangé entre deux séries de CER appariées.

    Retourne un dict avec :
      - statistic     : W = min(W⁺, W⁻)
      - p_value       : p-value bilatérale
      - significant   : bool (p < 0.05)
      - interpretation : phrase lisible
      - n_pairs       : nombre de paires utilisées (après retrait des zéros)
      - W_plus        : somme des rangs des différences positives
      - W_minus       : somme des rangs des différences négatives

    Hypothèses et limites
    ---------------------
    * Les observations sont appariées (même corpus, deux moteurs différents).
    * Le test est non-paramétrique : aucune hypothèse de normalité des CER.
    * ``zero_method="wilcox"`` (défaut) : les paires sans différence (aᵢ = bᵢ)
      sont simplement exclues.  Les autres méthodes (``"pratt"``, ``"zsplit"``)
      nécessitent scipy.
    * **Approximation normale** (implémentation native, n ≥ 10) :
      L'approximation est raisonnable pour n ≥ 10 et converge vers la
      distribution exacte.  Pour n < 10, une table critique simplifiée est
      utilisée (p ∈ {0.04, 0.20}) — résultat **conservateur**.
    * **scipy** (si installé) : ``scipy.stats.wilcoxon`` est utilisé à la place
      de l'approximation native.  scipy utilise la méthode exacte pour n ≤ 25
      et l'approximation normale pour n > 25, ce qui est plus précis.
    * **Validité** : le test suppose la symétrie de la distribution des
      différences.  Avec de très petits n (< 5), les résultats sont peu fiables
      quelle que soit la méthode.

    Parameters
    ----------
    a, b : séries de CER (même longueur, même ordre de documents)
    zero_method : gestion des paires nulles (défaut : ``"wilcox"``)
    """
    if len(a) != len(b):
        raise ValueError("Les deux listes doivent avoir la même longueur")

    diffs = [x - y for x, y in zip(a, b)]

    # Retirer les zéros (méthode "wilcox")
    if zero_method == "wilcox":
        diffs = [d for d in diffs if d != 0.0]

    n = len(diffs)
    if n == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "interpretation": "Aucune différence entre les deux concurrents.",
            "n_pairs": 0,
        }

    # Rangs des valeurs absolues
    abs_diffs = [abs(d) for d in diffs]
    indexed = sorted(enumerate(abs_diffs), key=lambda x: x[1])

    # Gestion des ex-aequo : rang moyen
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs_diffs[indexed[j][0]] == abs_diffs[indexed[i][0]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # rang moyen (1-based)
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j

    W_plus  = sum(ranks[k] for k in range(n) if diffs[k] > 0)
    W_minus = sum(ranks[k] for k in range(n) if diffs[k] < 0)
    W = min(W_plus, W_minus)

    # Calcul de la p-value : scipy si disponible, sinon approximation native
    if _SCIPY_AVAILABLE:
        try:
            scipy_res = _scipy_wilcoxon(diffs, zero_method=zero_method)
            p_value = float(scipy_res.pvalue)
        except Exception:
            # Repli sur l'implémentation native en cas d'erreur scipy
            p_value = _native_p_value(n, W)
    else:
        p_value = _native_p_value(n, W)

    significant = p_value < 0.05

    if significant:
        better = "premier" if W_plus < W_minus else "second"
        interpretation = (
            f"Différence statistiquement significative (p = {p_value:.4f} < 0.05). "
            f"Le {better} concurrent obtient de meilleurs scores."
        )
    else:
        interpretation = (
            f"Différence non significative (p = {p_value:.4f} ≥ 0.05). "
            "On ne peut pas conclure que l'un surpasse l'autre."
        )

    return {
        "statistic": round(W, 4),
        "p_value": round(p_value, 6),
        "significant": significant,
        "interpretation": interpretation,
        "n_pairs": n,
        "W_plus": round(W_plus, 4),
        "W_minus": round(W_minus, 4),
    }


def _normal_sf(z: float) -> float:
    """Survival function de la loi normale standard (1 - CDF)."""
    # Approximation Abramowitz & Stegun 26.2.17
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
           + t * (-1.821255978 + t * 1.330274429))))
    phi_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    p = phi_z * poly
    return p if z >= 0 else 1.0 - p


# Table des valeurs critiques de W pour α=0.05 bilatéral (test exact, source : tables de Wilcoxon)
_W_CRITICAL = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 2, 8: 3, 9: 5}


def _wilcoxon_exact_p(n: int, w: float) -> float:
    """P-value approximée pour petits n (< 10) via table critique simplifiée.

    Note : résultat **conservateur** — seules deux valeurs sont retournées :
    0.04 (significatif à 5 %) ou 0.20 (non significatif).
    Préférer scipy pour des p-values exactes.
    """
    critical = _W_CRITICAL.get(n, 0)
    if w <= critical:
        return 0.04  # significatif à 5 %
    return 0.20      # non significatif (approximation conservative)


def _native_p_value(n: int, W: float) -> float:
    """Calcule la p-value via l'approximation normale (n ≥ 10) ou la table exacte (n < 10)."""
    if n >= 10:
        mu = n * (n + 1) / 4.0
        sigma2 = n * (n + 1) * (2 * n + 1) / 24.0
        if sigma2 <= 0:
            return 1.0
        z = abs((W + 0.5) - mu) / math.sqrt(sigma2)  # correction de continuité
        return 2.0 * _normal_sf(z)  # test bilatéral
    return _wilcoxon_exact_p(n, W)


# ---------------------------------------------------------------------------
# Matrice des tests pairwise
# ---------------------------------------------------------------------------

def compute_pairwise_stats(
    engine_cer_map: dict[str, list[float]],
) -> list[dict]:
    """Calcule les tests de Wilcoxon entre toutes les paires de concurrents.

    Parameters
    ----------
    engine_cer_map : dict {engine_name → [cer_doc1, cer_doc2, ...]}

    Returns
    -------
    Liste de dicts, un par paire :
      - engine_a, engine_b, statistic, p_value, significant, interpretation
    """
    names = list(engine_cer_map.keys())
    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a_vals = engine_cer_map[a_name]
            b_vals = engine_cer_map[b_name]
            # Aligner les longueurs
            min_len = min(len(a_vals), len(b_vals))
            if min_len < 2:
                continue
            res = wilcoxon_test(a_vals[:min_len], b_vals[:min_len])
            results.append({
                "engine_a": a_name,
                "engine_b": b_name,
                **res,
            })
    return results


# ---------------------------------------------------------------------------
# Test de Friedman + post-hoc Nemenyi (Sprint 17)
# ---------------------------------------------------------------------------
#
# Référence : Demšar, J. (2006), "Statistical Comparisons of Classifiers over
# Multiple Data Sets", Journal of Machine Learning Research 7:1-30. Standard
# de facto pour comparer plusieurs systèmes sur plusieurs datasets — ici :
# plusieurs moteurs OCR sur plusieurs documents. Le CDD (critical difference
# diagram) issu de Nemenyi est le rendu canonique.

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


# ---------------------------------------------------------------------------
# Critical Difference Diagram — rendu SVG (Sprint 17)
# ---------------------------------------------------------------------------

def build_critical_difference_svg(
    nemenyi_result: dict,
    width: int = 780,
    row_height: int = 22,
) -> str:
    """Génère le SVG du Critical Difference Diagram (Demšar 2006).

    Le diagramme montre :
      * un axe horizontal des rangs moyens (1 à k),
      * chaque moteur positionné sur l'axe à son rang moyen,
      * des barres horizontales épaisses reliant les moteurs statistiquement
        indiscernables (distance ≤ CD),
      * la longueur de CD affichée au-dessus de l'axe en référence.

    Parameters
    ----------
    nemenyi_result:
        Résultat de ``nemenyi_posthoc``.
    width:
        Largeur totale du SVG en pixels.
    row_height:
        Hauteur de chaque ligne d'étiquette moteur (auto-adaptatif).

    Returns
    -------
    Chaîne contenant le SVG (balise racine ``<svg>…</svg>``).
    """
    k = nemenyi_result.get("n_engines", 0)
    if k < 2 or nemenyi_result.get("error"):
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="40" '
            'role="img" aria-label="Critical Difference Diagram indisponible">'
            '<text x="10" y="24" font-family="sans-serif" font-size="12" fill="#666">'
            'Critical Difference Diagram non calculable — données insuffisantes.'
            '</text></svg>'
        )

    engines_sorted: list[str] = list(nemenyi_result.get("engines_sorted", []))
    mean_ranks: dict[str, float] = dict(nemenyi_result.get("mean_ranks", {}))
    tied_groups: list[list[str]] = list(nemenyi_result.get("tied_groups", []))
    cd: float = float(nemenyi_result.get("critical_distance", 0.0))

    # Dimensions
    left_pad, right_pad = 40, 40
    top_pad = 50   # espace pour l'affichage CD
    axis_y = top_pad + 10
    bars_start_y = axis_y + 20  # première barre d'ex-aequo sous l'axe
    # Empiler une ligne par groupe + une ligne par étiquette
    label_rows = k  # chaque moteur a sa propre ligne de label
    bars_count = len(tied_groups)
    total_h = bars_start_y + bars_count * 10 + label_rows * row_height + 20

    axis_x0, axis_x1 = left_pad, width - right_pad
    axis_width = axis_x1 - axis_x0

    def x_for_rank(r: float) -> float:
        # Rang 1 à gauche, rang k à droite
        if k <= 1:
            return axis_x0
        return axis_x0 + (r - 1.0) / (k - 1.0) * axis_width

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {width} {total_h}" '
        f'role="img" aria-label="Critical Difference Diagram (Friedman-Nemenyi)" '
        f'font-family="system-ui, -apple-system, sans-serif">'
    )
    parts.append('<style>.cd-axis{stroke:#334155;stroke-width:1.5}.cd-tick{stroke:#334155;stroke-width:1}'
                 '.cd-label{fill:#0f172a;font-size:11px}'
                 '.cd-tie{stroke:#0f172a;stroke-width:4;stroke-linecap:round}'
                 '.cd-cd-bar{stroke:#dc2626;stroke-width:2}'
                 '.cd-cd-txt{fill:#dc2626;font-size:11px;font-weight:600}'
                 '.cd-name{fill:#0f172a;font-size:12px}'
                 '.cd-rank{fill:#64748b;font-size:10px}'
                 '</style>')

    # Barre CD de référence (en haut, à gauche de l'axe)
    if cd > 0 and k >= 2:
        cd_bar_x0 = axis_x0
        cd_bar_x1 = axis_x0 + (cd / max(1, k - 1)) * axis_width
        cd_y = top_pad - 20
        parts.append(f'<line class="cd-cd-bar" x1="{cd_bar_x0:.1f}" y1="{cd_y}" '
                     f'x2="{cd_bar_x1:.1f}" y2="{cd_y}"/>')
        parts.append(f'<line class="cd-cd-bar" x1="{cd_bar_x0:.1f}" y1="{cd_y - 4}" '
                     f'x2="{cd_bar_x0:.1f}" y2="{cd_y + 4}"/>')
        parts.append(f'<line class="cd-cd-bar" x1="{cd_bar_x1:.1f}" y1="{cd_y - 4}" '
                     f'x2="{cd_bar_x1:.1f}" y2="{cd_y + 4}"/>')
        parts.append(f'<text class="cd-cd-txt" x="{(cd_bar_x0 + cd_bar_x1)/2:.1f}" y="{cd_y - 8}" '
                     f'text-anchor="middle">CD = {cd:.3f}</text>')

    # Axe principal
    parts.append(f'<line class="cd-axis" x1="{axis_x0}" y1="{axis_y}" '
                 f'x2="{axis_x1}" y2="{axis_y}"/>')
    # Ticks entiers
    for r in range(1, k + 1):
        xt = x_for_rank(r)
        parts.append(f'<line class="cd-tick" x1="{xt:.1f}" y1="{axis_y - 5}" '
                     f'x2="{xt:.1f}" y2="{axis_y + 5}"/>')
        parts.append(f'<text class="cd-label" x="{xt:.1f}" y="{axis_y - 9}" '
                     f'text-anchor="middle">{r}</text>')

    # Barres reliant les groupes indiscernables
    for i, group in enumerate(tied_groups):
        if len(group) < 2:
            continue
        rs = [mean_ranks[n] for n in group]
        x0 = x_for_rank(min(rs))
        x1 = x_for_rank(max(rs))
        y_bar = bars_start_y + i * 10
        parts.append(f'<line class="cd-tie" x1="{x0 - 3:.1f}" y1="{y_bar}" '
                     f'x2="{x1 + 3:.1f}" y2="{y_bar}"/>')

    # Étiquettes des moteurs : la moitié la plus basse à gauche, l'autre à droite
    labels_y_base = bars_start_y + bars_count * 10 + 15
    half = (len(engines_sorted) + 1) // 2
    left_engines = engines_sorted[:half]
    right_engines = engines_sorted[half:]

    for idx, name in enumerate(left_engines):
        r = mean_ranks[name]
        x = x_for_rank(r)
        y_label = labels_y_base + idx * row_height
        # Ligne du moteur vers axe
        parts.append(f'<line class="cd-tick" x1="{x:.1f}" y1="{axis_y + 6}" '
                     f'x2="{x:.1f}" y2="{y_label - 4}"/>')
        parts.append(f'<line class="cd-tick" x1="{x:.1f}" y1="{y_label - 4}" '
                     f'x2="{axis_x0 - 4:.1f}" y2="{y_label - 4}"/>')
        parts.append(f'<text class="cd-name" x="{axis_x0 - 6:.1f}" y="{y_label}" '
                     f'text-anchor="end">{_svg_escape(name)} '
                     f'<tspan class="cd-rank">({r:.2f})</tspan></text>')

    for idx, name in enumerate(right_engines):
        r = mean_ranks[name]
        x = x_for_rank(r)
        y_label = labels_y_base + idx * row_height
        parts.append(f'<line class="cd-tick" x1="{x:.1f}" y1="{axis_y + 6}" '
                     f'x2="{x:.1f}" y2="{y_label - 4}"/>')
        parts.append(f'<line class="cd-tick" x1="{x:.1f}" y1="{y_label - 4}" '
                     f'x2="{axis_x1 + 4:.1f}" y2="{y_label - 4}"/>')
        parts.append(f'<text class="cd-name" x="{axis_x1 + 6:.1f}" y="{y_label}" '
                     f'text-anchor="start">{_svg_escape(name)} '
                     f'<tspan class="cd-rank">({r:.2f})</tspan></text>')

    parts.append('</svg>')
    return "".join(parts)


def _svg_escape(text: str) -> str:
    """Échappe un texte pour inclusion sûre dans un nœud SVG/XML."""
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))


# ---------------------------------------------------------------------------
# Frontière de Pareto (Sprint 19)
# ---------------------------------------------------------------------------

def compute_pareto_front(
    points: list[dict],
    objectives: tuple[str, ...] = ("cer", "cost"),
    name_key: str = "engine",
    minimize: Optional[tuple[bool, ...]] = None,
) -> list[str]:
    """Calcule la frontière de Pareto sur ``len(objectives)`` dimensions.

    Un point ``p`` est Pareto-dominant si aucun autre point n'a, pour TOUS
    les objectifs, une valeur au moins aussi bonne ET au moins une valeur
    strictement meilleure.

    Parameters
    ----------
    points:
        Liste de dicts. Chaque dict doit contenir ``name_key`` et toutes les
        clés de ``objectives``. Les points dont une valeur d'objectif est
        ``None`` sont ignorés (pas de comparaison possible).
    objectives:
        Clés des objectifs à minimiser/maximiser.
    name_key:
        Clé identifiant le point (par défaut ``"engine"``).
    minimize:
        Pour chaque objectif, ``True`` = minimiser (ex. CER, coût),
        ``False`` = maximiser (ex. ancrage). Doit avoir la même longueur
        que ``objectives``.

    Returns
    -------
    Liste des ``name`` des points sur le front Pareto, ordre stable depuis
    ``points``.
    """
    if minimize is None:
        minimize = tuple(True for _ in objectives)
    if len(minimize) != len(objectives):
        raise ValueError("`minimize` doit avoir la même longueur que `objectives`")

    valid = []
    for p in points:
        try:
            vals = tuple(float(p[k]) for k in objectives)
        except (KeyError, TypeError, ValueError):
            continue
        valid.append((p[name_key], vals))

    front: list[str] = []
    for name_a, vals_a in valid:
        dominated = False
        for name_b, vals_b in valid:
            if name_a == name_b:
                continue
            # B domine A si B est ≥ aussi bon partout ET strictement meilleur quelque part
            better_or_equal_everywhere = True
            strictly_better_somewhere = False
            for va, vb, mini in zip(vals_a, vals_b, minimize):
                if mini:
                    if vb > va:
                        better_or_equal_everywhere = False
                        break
                    if vb < va:
                        strictly_better_somewhere = True
                else:  # maximiser
                    if vb < va:
                        better_or_equal_everywhere = False
                        break
                    if vb > va:
                        strictly_better_somewhere = True
            if better_or_equal_everywhere and strictly_better_somewhere:
                dominated = True
                break
        if not dominated:
            front.append(name_a)
    return front


# ---------------------------------------------------------------------------
# Clustering des patterns d'erreurs
# ---------------------------------------------------------------------------

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
    """Extrait les paires (gt_char_seq, hyp_char_seq) d'erreurs de substitution."""
    # Sprint A3 (B-1) : import depuis Cercle 1, plus de violation Cercle 2→3.
    from picarones.core.diff_utils import compute_word_diff
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


# ---------------------------------------------------------------------------
# Matrice de corrélation entre métriques
# ---------------------------------------------------------------------------

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

    # Construire les vecteurs
    vectors: dict[str, list[float]] = {k: [] for k in metric_keys}
    for doc in metrics_per_doc:
        for k in metric_keys:
            v = doc.get(k)
            vectors[k].append(float(v) if v is not None else 0.0)

    # Calculer la matrice
    labels = metric_keys
    n = len(labels)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            r = _pearson(vectors[labels[i]], vectors[labels[j]])
            row.append(round(r, 4))
        matrix.append(row)

    return {"labels": labels, "matrix": matrix}


# ---------------------------------------------------------------------------
# Courbe de fiabilité (reliability curve)
# ---------------------------------------------------------------------------

def compute_reliability_curve(
    cer_values: list[float],
    steps: int = 20,
) -> list[dict]:
    """Pour les X% documents les plus faciles, quel est le CER moyen ?

    Returns
    -------
    Liste de {pct_docs: float, mean_cer: float}
    """
    if not cer_values:
        return []
    sorted_cer = sorted(cer_values)
    n = len(sorted_cer)
    points = []
    for step in range(1, steps + 1):
        pct = step / steps
        cutoff = max(1, int(pct * n))
        subset = sorted_cer[:cutoff]
        mean_cer = sum(subset) / len(subset)
        points.append({"pct_docs": round(pct * 100, 1), "mean_cer": round(mean_cer, 6)})
    return points


# ---------------------------------------------------------------------------
# Données pour le diagramme de Venn (erreurs communes / exclusives)
# ---------------------------------------------------------------------------

def compute_venn_data(
    engine_error_sets: dict[str, set[str]],
) -> dict:
    """Calcule les cardinalités pour un diagramme de Venn entre 2 ou 3 concurrents.

    Parameters
    ----------
    engine_error_sets : {engine_name → set of doc_id:error_token_pair strings}

    Returns
    -------
    Pour 2 concurrents :
      {only_a, only_b, both, label_a, label_b}
    Pour 3 concurrents :
      {only_a, only_b, only_c, ab, ac, bc, abc, label_a, label_b, label_c}
    """
    names = list(engine_error_sets.keys())[:3]  # max 3 pour Venn lisible
    if len(names) < 2:
        return {}

    sets = {n: engine_error_sets[n] for n in names}

    if len(names) == 2:
        a, b = names
        sa, sb = sets[a], sets[b]
        return {
            "type": "venn2",
            "label_a": a,
            "label_b": b,
            "only_a": len(sa - sb),
            "only_b": len(sb - sa),
            "both": len(sa & sb),
        }
    else:
        a, b, c = names
        sa, sb, sc = sets[a], sets[b], sets[c]
        return {
            "type": "venn3",
            "label_a": a,
            "label_b": b,
            "label_c": c,
            "only_a": len(sa - sb - sc),
            "only_b": len(sb - sa - sc),
            "only_c": len(sc - sa - sb),
            "ab": len((sa & sb) - sc),
            "ac": len((sa & sc) - sb),
            "bc": len((sb & sc) - sa),
            "abc": len(sa & sb & sc),
        }
