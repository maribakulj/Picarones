"""Comparaison à la baseline historique — Sprint 73 (A.I.3).

Sprint 73 — chantier 2 d'A.I.3 du plan d'évolution 2026.

Pourquoi ce module
------------------
L'historique SQLite (``picarones/core/history.py``, Sprint 8)
existe mais aucun détecteur narratif ne le lit.  Ce module fournit
la couche de calcul qui répond à *« comment ce moteur se
comporte-t-il sur ce corpus, **par rapport à ses runs précédents
de mon institution** ? »*.

Sortie typique
--------------
Un dict par moteur :

.. code-block:: python

    {
        "engine_name": "tesseract",
        "cer_current": 0.052,
        "cer_historical_mean": 0.041,
        "cer_historical_median": 0.040,
        "n_runs": 12,
        "absolute_delta": 0.011,
        "relative_delta": 0.268,        # +26,8 % vs moyenne
        "off_baseline": True,
    }

Le détecteur narratif ``engine_off_baseline`` (Sprint 73)
consomme cette structure pour émettre des Facts.

Garde-fous
----------
- ``min_runs`` (défaut 5) : si l'historique pour le moteur×corpus
  contient moins de runs, on retourne ``None`` plutôt que de
  comparer à un échantillon trop petit.
- ``corpus_name`` est utilisé pour ne comparer qu'aux runs **du
  même corpus** (sinon on compare des pommes et des oranges :
  registres paroissiaux vs imprimés modernes).
- Le run courant lui-même n'est pas inclus dans la baseline (on
  passe le ``current_run_id`` à exclure).
"""

from __future__ import annotations

import logging
import statistics
from typing import Optional

logger = logging.getLogger(__name__)


def compute_engine_baseline(
    history,
    engine_name: str,
    corpus_name: str,
    current_cer: float,
    *,
    current_run_id: Optional[str] = None,
    min_runs: int = 5,
    relative_delta_threshold: float = 0.20,
) -> Optional[dict]:
    """Compare le CER courant d'un moteur à sa moyenne historique
    sur le **même corpus**.

    Parameters
    ----------
    history:
        Instance de ``BenchmarkHistory`` (ou compatible : doit
        exposer une méthode ``query(engine, corpus, limit)``
        retournant une liste d'``HistoryEntry`` avec attribut
        ``cer_mean`` et ``run_id``).
    engine_name:
        Nom du moteur dont on calcule la baseline.
    corpus_name:
        Nom du corpus — limite la comparaison aux runs antérieurs
        sur ce même corpus.
    current_cer:
        CER moyen observé dans le run courant.
    current_run_id:
        Si fourni, le run portant cet identifiant est exclu de la
        baseline (utile quand le run courant est déjà enregistré
        dans l'historique avant d'appeler ce calcul).
    min_runs:
        Nombre minimum de runs historiques pour que la
        comparaison soit considérée fiable.  Sous ce seuil, on
        retourne ``None``.
    relative_delta_threshold:
        Seuil au-delà duquel ``off_baseline`` vaut ``True``
        (défaut : 0,20 = 20 % d'écart relatif).

    Returns
    -------
    Optional[dict]
        ``None`` si :
        - moins de ``min_runs`` runs historiques disponibles
        - ``current_cer`` est ``None`` ou négatif
        - tous les CER historiques sont ``None``

        Sinon, dict avec les champs documentés dans le module.
    """
    if current_cer is None or current_cer < 0:
        return None
    try:
        entries = history.query(
            engine=engine_name, corpus=corpus_name, limit=1000,
        )
    except Exception as exc:  # pragma: no cover — défense
        logger.warning(
            "[baseline_comparison] query history a levé : %s", exc,
        )
        return None

    historical_cers: list[float] = []
    for entry in entries:
        if current_run_id is not None and entry.run_id == current_run_id:
            continue
        cer = entry.cer_mean
        if cer is None or cer < 0:
            continue
        historical_cers.append(float(cer))

    if len(historical_cers) < min_runs:
        return None

    mean = statistics.fmean(historical_cers)
    median = statistics.median(historical_cers)
    absolute_delta = current_cer - mean
    if mean > 0:
        relative_delta = absolute_delta / mean
    elif current_cer == 0:
        relative_delta = 0.0
    else:
        # Baseline à 0 mais CER courant > 0 : écart infini —
        # convention : on signale comme off_baseline avec
        # relative_delta = None.
        relative_delta = None

    off_baseline = (
        relative_delta is not None
        and abs(relative_delta) > relative_delta_threshold
    )

    return {
        "engine_name": engine_name,
        "corpus_name": corpus_name,
        "cer_current": float(current_cer),
        "cer_historical_mean": mean,
        "cer_historical_median": median,
        "n_runs": len(historical_cers),
        "absolute_delta": absolute_delta,
        "relative_delta": relative_delta,
        "off_baseline": off_baseline,
    }


def compute_corpus_difficulty_percentile(
    history,
    current_difficulty: float,
    *,
    min_runs: int = 5,
) -> Optional[dict]:
    """Place la difficulté du corpus courant dans la distribution
    des difficultés historiques.

    Lit les difficultés stockées dans ``HistoryEntry.metadata``
    sous la clé ``difficulty`` (convention de
    ``picarones/core/difficulty.py``).

    Returns
    -------
    Optional[dict]
        ``{
            "current_difficulty": float,
            "percentile": float,            # 0..100
            "n_runs": int,
            "median_historical": float,
            "harder_than_usual": bool,      # percentile > 75
            "easier_than_usual": bool,      # percentile < 25
        }``
        ou ``None`` si moins de ``min_runs`` runs historiques ont
        une difficulté enregistrée.
    """
    if current_difficulty is None:
        return None
    try:
        entries = history.query(limit=1000)
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "[baseline_comparison] query history a levé : %s", exc,
        )
        return None

    historical_difficulties: list[float] = []
    for entry in entries:
        diff = entry.metadata.get("difficulty") if entry.metadata else None
        if diff is None:
            continue
        try:
            historical_difficulties.append(float(diff))
        except (TypeError, ValueError):
            continue

    if len(historical_difficulties) < min_runs:
        return None

    sorted_diff = sorted(historical_difficulties)
    n = len(sorted_diff)
    # Percentile = % de corpus historiques de difficulté ≤
    # current_difficulty.  Convention courante (P_i = i/n × 100).
    n_below = sum(1 for d in sorted_diff if d <= current_difficulty)
    percentile = (n_below / n) * 100.0
    median = statistics.median(sorted_diff)

    return {
        "current_difficulty": float(current_difficulty),
        "percentile": percentile,
        "n_runs": n,
        "median_historical": median,
        "harder_than_usual": percentile > 75.0,
        "easier_than_usual": percentile < 25.0,
    }


__all__ = [
    "compute_engine_baseline",
    "compute_corpus_difficulty_percentile",
]
