"""Métriques longitudinales — Sprint 92 (A.II.9).

A.II.9 du plan d'évolution 2026.

Pourquoi ce module
------------------
L'historique SQLite (`core/history.py`, Sprint 8) collecte les
résultats de chaque run de benchmark, mais aucune métrique
n'en sortait dans le rapport.  Ce module exploite la série
temporelle des CER d'un moteur pour répondre à deux
questions :

1. **Y a-t-il une tendance ?**  Régression linéaire simple
   (méthode des moindres carrés) sur ``(t, CER)`` —  pente,
   ordonnée à l'origine, R², n_runs.  Une pente > 0 signale
   une régression progressive ; une pente < 0 une amélioration.

2. **Y a-t-il un point de rupture ?**  Algorithme de
   change-point pur Python (différence de moyennes maximale,
   variante de Pettitt simplifiée).  Identifie l'index où la
   série se sépare en deux segments avec moyennes les plus
   différentes — typiquement le run où un modèle a changé de
   comportement.

Pas de scipy
------------
Pour rester sans dépendance lourde, on implémente :
- la régression linéaire en pur Python (closed-form OLS) ;
- le change-point par balayage exhaustif (O(N) pour de petits
  N — l'historique d'une institution dépasse rarement quelques
  centaines de runs).
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class LinearTrend:
    """Résultat d'une régression linéaire sur une série CER."""
    slope: float
    """Pente (CER par jour). Positif = régression."""
    intercept: float
    """Ordonnée à l'origine."""
    r_squared: float
    """Qualité de l'ajustement, ∈ [0, 1]."""
    n_runs: int
    """Nombre de points utilisés."""

    def as_dict(self) -> dict:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "n_runs": self.n_runs,
        }


@dataclass
class ChangePointResult:
    """Résultat d'une détection de point de rupture."""
    index: int
    """Index de la rupture (0-based, le segment 1 est [0:index],
    le segment 2 est [index:N])."""
    timestamp: str
    """Timestamp du run à la rupture."""
    mean_before: float
    mean_after: float
    delta: float
    """``mean_after - mean_before``. Positif = régression."""
    n_before: int
    n_after: int

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "mean_before": self.mean_before,
            "mean_after": self.mean_after,
            "delta": self.delta,
            "n_before": self.n_before,
            "n_after": self.n_after,
        }


def _parse_timestamp(ts: str) -> Optional[float]:
    """Parse un ISO timestamp en jour ordinal float.

    Tolère ``YYYY-MM-DD`` et ``YYYY-MM-DDTHH:MM:SS``.  Retourne
    ``None`` si non parsable.
    """
    if not ts:
        return None
    formats = (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    )
    for fmt in formats:
        try:
            dt = datetime.strptime(ts.split("+")[0].split("Z")[0], fmt)
            return dt.toordinal() + (
                dt.hour * 3600 + dt.minute * 60 + dt.second
            ) / 86400.0
        except ValueError:
            continue
    return None


def compute_linear_trend(
    cer_series: Iterable[tuple[str, float]],
) -> Optional[LinearTrend]:
    """Régression linéaire OLS sur une série temporelle de CER.

    Parameters
    ----------
    cer_series:
        Itérable de ``(timestamp_iso, cer)``.  Au moins 2 points
        valides requis.

    Returns
    -------
    LinearTrend | None
        ``None`` si moins de 2 points ou si tous les timestamps
        sont identiques (variance nulle sur t).
    """
    points: list[tuple[float, float]] = []
    for ts, cer in cer_series:
        t = _parse_timestamp(ts)
        if t is None or cer is None:
            continue
        try:
            cer_f = float(cer)
        except (TypeError, ValueError):
            continue
        points.append((t, cer_f))
    n = len(points)
    if n < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    sxx = sum((x - x_mean) ** 2 for x in xs)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    if sxx == 0:
        return None
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    syy = sum((y - y_mean) ** 2 for y in ys)
    if syy == 0:
        # Tous les CER sont égaux → R² mathématiquement indéfini ;
        # on retourne 1.0 (parfaite "non-tendance").
        r_squared = 1.0
    else:
        ss_res = sum(
            (y - (slope * x + intercept)) ** 2
            for x, y in zip(xs, ys)
        )
        r_squared = max(0.0, 1.0 - ss_res / syy)
    return LinearTrend(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        n_runs=n,
    )


def detect_change_point(
    cer_series: Iterable[tuple[str, float]],
    min_segment_size: int = 3,
) -> Optional[ChangePointResult]:
    """Détecte le point de rupture maximisant l'écart de moyennes.

    Algorithme : balayage des indices ``i`` où la série se
    sépare en deux segments d'au moins ``min_segment_size``
    points chacun ; on retient l'index où ``|mean_after -
    mean_before|`` est maximal.  Variante simplifiée de Pettitt.

    Parameters
    ----------
    cer_series:
        Itérable de ``(timestamp_iso, cer)``.
    min_segment_size:
        Taille minimale des deux segments.  Défaut 3.

    Returns
    -------
    ChangePointResult | None
        ``None`` si la série a moins de ``2 × min_segment_size``
        points valides.
    """
    points: list[tuple[str, float, float]] = []
    for ts, cer in cer_series:
        t = _parse_timestamp(ts)
        if t is None or cer is None:
            continue
        try:
            cer_f = float(cer)
        except (TypeError, ValueError):
            continue
        points.append((ts, t, cer_f))
    if len(points) < 2 * min_segment_size:
        return None
    points.sort(key=lambda p: p[1])
    n = len(points)
    best_index = -1
    best_abs_delta = -1.0
    best_delta = 0.0
    best_mean_before = 0.0
    best_mean_after = 0.0
    for i in range(min_segment_size, n - min_segment_size + 1):
        before = [p[2] for p in points[:i]]
        after = [p[2] for p in points[i:]]
        mean_b = statistics.fmean(before)
        mean_a = statistics.fmean(after)
        delta = mean_a - mean_b
        abs_delta = abs(delta)
        if abs_delta > best_abs_delta:
            best_abs_delta = abs_delta
            best_index = i
            best_delta = delta
            best_mean_before = mean_b
            best_mean_after = mean_a
    if best_index < 0:
        return None
    return ChangePointResult(
        index=best_index,
        timestamp=points[best_index][0],
        mean_before=best_mean_before,
        mean_after=best_mean_after,
        delta=best_delta,
        n_before=best_index,
        n_after=n - best_index,
    )


def compute_engine_longitudinal(
    history_entries: Iterable,
    engine_name: str,
    corpus_name: Optional[str] = None,
    *,
    min_runs_for_trend: int = 3,
    min_segment_size: int = 3,
    change_point_threshold: float = 0.01,
) -> Optional[dict]:
    """Calcule trend + change_point pour un moteur.

    Parameters
    ----------
    history_entries:
        Liste de ``HistoryEntry`` (ou dicts compatibles).
    engine_name:
        Filtre sur le nom du moteur.
    corpus_name:
        Filtre optionnel sur le corpus.  ``None`` (défaut) : tous
        les corpus.
    min_runs_for_trend:
        Minimum de runs pour calculer une tendance.
    min_segment_size:
        Taille minimale des segments pour le change-point.
    change_point_threshold:
        Magnitude absolue minimale du delta (en CER) pour
        retenir le change-point.  Défaut 0.01 (1 point de CER).

    Returns
    -------
    dict | None
        ``{
            "engine_name", "corpus_name", "n_runs", "trend",
            "change_point",  # ou None
            "first_timestamp", "last_timestamp",
            "first_cer", "last_cer", "absolute_delta_pct",
        }`` ou ``None`` si moins de ``min_runs_for_trend`` runs.
    """
    series: list[tuple[str, float]] = []
    for entry in history_entries:
        if hasattr(entry, "as_dict"):
            data = entry.as_dict()
        else:
            data = entry
        if data.get("engine_name") != engine_name:
            continue
        if corpus_name is not None and data.get("corpus_name") != corpus_name:
            continue
        cer = data.get("cer_mean")
        ts = data.get("timestamp")
        if cer is None or ts is None:
            continue
        series.append((ts, float(cer)))
    if len(series) < min_runs_for_trend:
        return None
    series.sort(key=lambda p: _parse_timestamp(p[0]) or 0.0)
    trend = compute_linear_trend(series)
    cp = detect_change_point(series, min_segment_size=min_segment_size)
    if cp is not None and abs(cp.delta) < change_point_threshold:
        cp = None
    first_ts, first_cer = series[0]
    last_ts, last_cer = series[-1]
    return {
        "engine_name": engine_name,
        "corpus_name": corpus_name,
        "n_runs": len(series),
        "trend": trend.as_dict() if trend else None,
        "change_point": cp.as_dict() if cp else None,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "first_cer": first_cer,
        "last_cer": last_cer,
        "absolute_delta": last_cer - first_cer,
        "absolute_delta_pct": round((last_cer - first_cer) * 100, 2),
    }


def compute_corpus_longitudinal(
    history_entries: Iterable,
    corpus_name: Optional[str] = None,
    *,
    min_runs_for_trend: int = 3,
    min_segment_size: int = 3,
    change_point_threshold: float = 0.01,
) -> list[dict]:
    """Pour chaque moteur présent dans l'historique sur ``corpus_name``,
    calcule trend + change_point.

    Returns
    -------
    list[dict]
        Une entrée par moteur (filtrée), liste vide si rien.
    """
    entries = list(history_entries)
    engines: set[str] = set()
    for entry in entries:
        data = entry.as_dict() if hasattr(entry, "as_dict") else entry
        if corpus_name is not None and data.get("corpus_name") != corpus_name:
            continue
        name = data.get("engine_name")
        if name:
            engines.add(name)
    out: list[dict] = []
    for engine in sorted(engines):
        result = compute_engine_longitudinal(
            entries, engine, corpus_name=corpus_name,
            min_runs_for_trend=min_runs_for_trend,
            min_segment_size=min_segment_size,
            change_point_threshold=change_point_threshold,
        )
        if result is not None:
            out.append(result)
    return out


__all__ = [
    "LinearTrend",
    "ChangePointResult",
    "compute_linear_trend",
    "detect_change_point",
    "compute_engine_longitudinal",
    "compute_corpus_longitudinal",
]


# Marqueur d'évitement d'import inutilisé (math)
_ = math
