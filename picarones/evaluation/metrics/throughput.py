"""Throughput effectif (Sprint 91 — A.II.6).

A.II.6 du plan d'évolution 2026.

Pourquoi ce module
------------------
Le throughput brut (pages/heure d'OCR pur) ment quand un moteur
est rapide mais imprécis : la correction humaine *post hoc*
absorbe le gain.  La **vraie** vitesse opérationnelle inclut
le temps de correction.  Cette métrique discrimine fortement
entre un cloud rapide à 30 % de timeouts/erreurs et un local
lent à 100 % de fiabilité.

Formule
-------
.. code::

    pages_par_heure_utilisable =
        pages_traitées / (durée_totale + temps_correction_humaine)

Le temps de correction est estimé linéairement :
``temps_par_erreur × nombre_d_erreurs``.  Le défaut
``time_per_error_seconds=5.0`` correspond aux études HTR-United
(saisie manuelle d'une correction de mot par un opérateur
formé : ≈ 5 s par erreur).  L'utilisateur peut le surcharger
pour son institution.

Sortie
------
``compute_effective_throughput(n_pages, duration_seconds,
n_errors, time_per_error_seconds=5.0)`` retourne ``{n_pages,
duration_seconds, n_errors, time_per_error_seconds,
correction_time_seconds, total_seconds, pages_per_hour_raw,
pages_per_hour_effective, drag_ratio}``.

``aggregate_effective_throughput(per_engine_data)`` agrège par
moteur sur l'ensemble du corpus.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


_DEFAULT_TIME_PER_ERROR_SECONDS = 5.0


def compute_effective_throughput(
    n_pages: int,
    duration_seconds: float,
    n_errors: int,
    *,
    time_per_error_seconds: float = _DEFAULT_TIME_PER_ERROR_SECONDS,
) -> Optional[dict]:
    """Throughput effectif (pages/heure utilisables).

    Parameters
    ----------
    n_pages:
        Nombre de pages traitées.
    duration_seconds:
        Durée totale de l'OCR (somme des durées par doc).
    n_errors:
        Nombre d'erreurs (au niveau mot, typiquement
        ``WER × n_words_total``).
    time_per_error_seconds:
        Temps moyen de correction humaine par erreur.  Défaut
        5 s (HTR-United).  Doit être ≥ 0.

    Returns
    -------
    dict | None
        ``None`` si ``n_pages == 0`` ou ``total_seconds == 0``
        (pas de division par zéro).
    """
    if n_pages <= 0:
        return None
    if duration_seconds < 0 or n_errors < 0 or time_per_error_seconds < 0:
        raise ValueError(
            "duration_seconds, n_errors et time_per_error_seconds "
            "doivent être ≥ 0",
        )
    correction_seconds = float(n_errors) * float(time_per_error_seconds)
    total_seconds = float(duration_seconds) + correction_seconds
    if total_seconds <= 0:
        # Aucun temps écoulé : impossible de définir un throughput
        return None
    pages_per_hour_raw = (
        n_pages / duration_seconds * 3600.0
        if duration_seconds > 0 else None
    )
    pages_per_hour_effective = n_pages / total_seconds * 3600.0
    drag_ratio = (
        correction_seconds / total_seconds if total_seconds > 0 else 0.0
    )
    return {
        "n_pages": int(n_pages),
        "duration_seconds": float(duration_seconds),
        "n_errors": int(n_errors),
        "time_per_error_seconds": float(time_per_error_seconds),
        "correction_time_seconds": correction_seconds,
        "total_seconds": total_seconds,
        "pages_per_hour_raw": pages_per_hour_raw,
        "pages_per_hour_effective": pages_per_hour_effective,
        "drag_ratio": drag_ratio,
    }


def aggregate_effective_throughput(
    per_engine: Iterable[dict],
    *,
    time_per_error_seconds: float = _DEFAULT_TIME_PER_ERROR_SECONDS,
) -> Optional[dict]:
    """Agrège le throughput effectif par moteur.

    Parameters
    ----------
    per_engine:
        Itérable de dicts ``{engine_name, n_pages,
        duration_seconds, n_errors}``.

    Returns
    -------
    dict | None
        ``{
            "engines": [
                {"engine_name", ..., compute_effective_throughput
                fields},
                ...
            ],
            "time_per_error_seconds": float,
        }`` ou ``None`` si aucun moteur exploitable.
    """
    rows: list[dict] = []
    for entry in per_engine:
        if not isinstance(entry, dict):
            continue
        name = entry.get("engine_name") or entry.get("engine")
        if not name:
            continue
        result = compute_effective_throughput(
            int(entry.get("n_pages") or 0),
            float(entry.get("duration_seconds") or 0.0),
            int(entry.get("n_errors") or 0),
            time_per_error_seconds=time_per_error_seconds,
        )
        if result is None:
            continue
        result["engine_name"] = str(name)
        rows.append(result)
    if not rows:
        return None
    return {
        "engines": rows,
        "time_per_error_seconds": float(time_per_error_seconds),
    }


__all__ = [
    "compute_effective_throughput",
    "aggregate_effective_throughput",
]
