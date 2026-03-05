"""Tests statistiques et clustering d'erreurs pour Picarones.

Fonctions fournies
------------------
- wilcoxon_test(a, b)          : test de Wilcoxon signé-rangé entre deux séries de CER
- bootstrap_ci(values, ...)    : intervalle de confiance à 95 % par bootstrap
- compute_pairwise_stats(...)  : matrice de tests de Wilcoxon entre tous les concurrents
- cluster_errors(...)          : regroupement des patterns d'erreurs en clusters
- compute_correlation_matrix(...)  : matrice de corrélation entre toutes les métriques
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional


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
      - statistic : W+
      - p_value   : p-value approximée (distribution normale pour n ≥ 10)
      - significant : bool (p < 0.05)
      - interpretation : phrase lisible
      - n_pairs   : nombre de paires utilisées

    Pour n < 10, on utilise la table exacte simplifée.
    Pour n ≥ 10, on utilise l'approximation normale.
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

    # Approximation normale (valide pour n ≥ 10)
    if n >= 10:
        mu = n * (n + 1) / 4.0
        # Correction pour les ex-aequo (simplifiée)
        sigma2 = n * (n + 1) * (2 * n + 1) / 24.0
        if sigma2 <= 0:
            p_value = 1.0
        else:
            z = abs((W + 0.5) - mu) / math.sqrt(sigma2)  # correction de continuité
            p_value = 2.0 * _normal_sf(z)  # test bilatéral
    else:
        # Table exacte approximée pour petits n
        p_value = _wilcoxon_exact_p(n, W)

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


# Table des valeurs critiques de W pour α=0.05 bilatéral
_W_CRITICAL = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 2, 8: 3, 9: 5}

def _wilcoxon_exact_p(n: int, w: float) -> float:
    """P-value approximée pour petits n (< 10) via table critique simplifiée."""
    critical = _W_CRITICAL.get(n, 0)
    if w <= critical:
        return 0.04  # significatif à 5 %
    return 0.20      # non significatif (approximation conservative)


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
    from picarones.report.diff_utils import compute_word_diff
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
