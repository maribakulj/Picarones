"""Distribution des erreurs CER par ligne — Sprint 10.

Métriques calculées
-------------------
- CER par ligne    : distance d'édition caractère/longueur GT sur chaque paire de lignes
- Percentiles      : p50, p75, p90, p95, p99 sur la distribution des CER ligne
- Taux catastrophiques : % de lignes dépassant des seuils configurables (30 %, 50 %, 100 %)
- Coefficient de Gini  : concentration des erreurs (0 = uniformes, 1 = toutes concentrées)
- Carte thermique      : CER moyen par tranche de position dans le document
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# CER d'une paire de lignes (distance d'édition Levenshtein normalisée)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Distance de Levenshtein entre deux chaînes."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _line_cer(ref_line: str, hyp_line: str) -> float:
    """CER pour une paire de lignes.  Retourne 1.0 si le GT est vide et que l'hyp ne l'est pas."""
    ref = unicodedata.normalize("NFC", ref_line.strip())
    hyp = unicodedata.normalize("NFC", hyp_line.strip())
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = _edit_distance(ref, hyp)
    return dist / len(ref)


# ---------------------------------------------------------------------------
# Percentiles (implémentation pur-Python, sans numpy)
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[float], p: float) -> float:
    """Retourne le p-ième percentile (0 ≤ p ≤ 100) d'une liste triée."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    index = p / 100 * (n - 1)
    lo = int(index)
    hi = min(lo + 1, n - 1)
    frac = index - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


# ---------------------------------------------------------------------------
# Coefficient de Gini
# ---------------------------------------------------------------------------

def _gini(values: list[float]) -> float:
    """Coefficient de Gini des erreurs (0 = uniformes, 1 = toutes concentrées).

    Formule : G = (2 * Σ i*x_i) / (n * Σ x_i) - (n+1)/n
    sur les valeurs triées par ordre croissant.
    """
    if not values:
        return 0.0
    xs = sorted(max(v, 0.0) for v in values)
    n = len(xs)
    total = sum(xs)
    if total == 0.0:
        return 0.0
    weighted_sum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2.0 * weighted_sum) / (n * total) - (n + 1) / n


# ---------------------------------------------------------------------------
# Résultat structuré
# ---------------------------------------------------------------------------

@dataclass
class LineMetrics:
    """Distribution des erreurs CER par ligne pour une paire (GT, hypothèse)."""

    cer_per_line: list[float]
    """CER de chaque ligne (longueur = nombre de lignes GT)."""

    percentiles: dict[str, float]
    """Percentiles : p50, p75, p90, p95, p99."""

    catastrophic_rate: dict[str, float]
    """Taux de lignes catastrophiques pour chaque seuil (ex. {0.3: 0.12, 0.5: 0.07, 1.0: 0.02})."""

    gini: float
    """Coefficient de Gini des erreurs (0 → uniforme, 1 → concentrées)."""

    heatmap: list[float]
    """CER moyen par tranche de position dans le document (longueur = heatmap_bins)."""

    line_count: int
    """Nombre de lignes GT traitées."""

    mean_cer: float
    """CER moyen sur l'ensemble des lignes."""

    def as_dict(self) -> dict:
        return {
            "cer_per_line": [round(v, 6) for v in self.cer_per_line],
            "percentiles": {k: round(v, 6) for k, v in self.percentiles.items()},
            "catastrophic_rate": {str(k): round(v, 6) for k, v in self.catastrophic_rate.items()},
            "gini": round(self.gini, 6),
            "heatmap": [round(v, 6) for v in self.heatmap],
            "line_count": self.line_count,
            "mean_cer": round(self.mean_cer, 6),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LineMetrics":
        return cls(
            cer_per_line=d.get("cer_per_line", []),
            percentiles=d.get("percentiles", {}),
            catastrophic_rate={float(k): v for k, v in d.get("catastrophic_rate", {}).items()},
            gini=d.get("gini", 0.0),
            heatmap=d.get("heatmap", []),
            line_count=d.get("line_count", 0),
            mean_cer=d.get("mean_cer", 0.0),
        )


# ---------------------------------------------------------------------------
# Calcul principal
# ---------------------------------------------------------------------------

def compute_line_metrics(
    reference: str,
    hypothesis: str,
    thresholds: Optional[list[float]] = None,
    heatmap_bins: int = 10,
) -> LineMetrics:
    """Calcule la distribution des erreurs CER ligne par ligne.

    Parameters
    ----------
    reference:
        Texte de vérité terrain (GT) avec sauts de ligne.
    hypothesis:
        Texte produit par le moteur OCR.
    thresholds:
        Seuils CER pour le taux catastrophique. Défaut : [0.30, 0.50, 1.00].
    heatmap_bins:
        Nombre de tranches de position pour la carte thermique.

    Returns
    -------
    LineMetrics
    """
    if thresholds is None:
        thresholds = [0.30, 0.50, 1.00]

    ref_lines = reference.splitlines()
    hyp_lines = hypothesis.splitlines()

    # Aligner les lignes GT / hypothèse — on prend au moins autant de lignes que le GT
    n = len(ref_lines)
    if n == 0:
        # Pas de lignes : retourner des métriques neutres
        return LineMetrics(
            cer_per_line=[],
            percentiles={f"p{p}": 0.0 for p in (50, 75, 90, 95, 99)},
            catastrophic_rate={t: 0.0 for t in thresholds},
            gini=0.0,
            heatmap=[0.0] * heatmap_bins,
            line_count=0,
            mean_cer=0.0,
        )

    # Aligner en ignorant les lignes d'hypothèse supplémentaires
    # Si l'hypothèse a moins de lignes, les lignes manquantes comptent comme supprimées (CER = 1.0)
    cer_per_line: list[float] = []
    for i, ref_line in enumerate(ref_lines):
        hyp_line = hyp_lines[i] if i < len(hyp_lines) else ""
        cer_per_line.append(min(_line_cer(ref_line, hyp_line), 1.0))

    sorted_cer = sorted(cer_per_line)

    # Percentiles
    percentiles = {
        f"p{p}": _percentile(sorted_cer, p)
        for p in (50, 75, 90, 95, 99)
    }

    # Taux catastrophiques
    catastrophic_rate: dict[float, float] = {}
    for t in thresholds:
        count = sum(1 for v in cer_per_line if v > t)
        catastrophic_rate[t] = count / n

    # Gini
    gini = _gini(cer_per_line)

    # Carte thermique par tranche de position
    bins = heatmap_bins
    heatmap: list[float] = []
    for b in range(bins):
        start = int(b * n / bins)
        end = int((b + 1) * n / bins)
        slice_ = cer_per_line[start:end]
        heatmap.append(sum(slice_) / len(slice_) if slice_ else 0.0)

    mean_cer = sum(cer_per_line) / n

    return LineMetrics(
        cer_per_line=cer_per_line,
        percentiles=percentiles,
        catastrophic_rate=catastrophic_rate,
        gini=gini,
        heatmap=heatmap,
        line_count=n,
        mean_cer=mean_cer,
    )


# ---------------------------------------------------------------------------
# Agrégation sur un corpus
# ---------------------------------------------------------------------------

def aggregate_line_metrics(results: list[LineMetrics]) -> dict:
    """Agrège les métriques de distribution par ligne sur un corpus.

    Returns
    -------
    dict
        Statistiques agrégées : Gini moyen, percentiles moyens, taux catastrophiques moyens.
    """
    if not results:
        return {}

    import statistics as _stats

    gini_values = [r.gini for r in results]
    mean_cer_values = [r.mean_cer for r in results]

    # Percentiles moyens
    pct_keys = ["p50", "p75", "p90", "p95", "p99"]
    avg_percentiles = {}
    for k in pct_keys:
        vals = [r.percentiles.get(k, 0.0) for r in results]
        avg_percentiles[k] = round(sum(vals) / len(vals), 6) if vals else 0.0

    # Taux catastrophiques moyens (union des seuils)
    all_thresholds: set[float] = set()
    for r in results:
        all_thresholds.update(r.catastrophic_rate.keys())
    avg_catastrophic: dict[str, float] = {}
    for t in sorted(all_thresholds):
        vals = [r.catastrophic_rate.get(t, 0.0) for r in results]
        avg_catastrophic[str(t)] = round(sum(vals) / len(vals), 6) if vals else 0.0

    # Heatmap moyenne (longueur = max des longueurs)
    if results and results[0].heatmap:
        n_bins = len(results[0].heatmap)
        heatmap_avg = []
        for b in range(n_bins):
            vals = [r.heatmap[b] for r in results if b < len(r.heatmap)]
            heatmap_avg.append(round(sum(vals) / len(vals), 6) if vals else 0.0)
    else:
        heatmap_avg = []

    return {
        "gini_mean": round(sum(gini_values) / len(gini_values), 6),
        "gini_stdev": round(_stats.stdev(gini_values), 6) if len(gini_values) > 1 else 0.0,
        "mean_cer_mean": round(sum(mean_cer_values) / len(mean_cer_values), 6),
        "percentiles": avg_percentiles,
        "catastrophic_rate": avg_catastrophic,
        "heatmap": heatmap_avg,
        "document_count": len(results),
    }
