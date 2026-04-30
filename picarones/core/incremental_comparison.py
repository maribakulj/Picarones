"""Comparaison incrémentale de pipelines composées — Sprint 96 (B.5).

Sprint 96 — B.5 du plan d'évolution 2026.

Pourquoi ce module
------------------
Avec 5 OCR × 3 reconstructeurs × 4 post-correcteurs × 3
mappeurs = 180 pipelines à comparer, le rapport noie
l'information.  Il faut un mécanisme de **comparaison
contrôlée** type design d'expérience.

Méthode
-------
Pour mesurer l'effet isolé d'un slot ``varying`` :

1. Fixer les valeurs des autres slots (``fixed``).
2. Pour chaque combinaison des fixed, comparer les pipelines
   qui ne diffèrent que sur le slot varying.
3. Agréger : pour chaque valeur du slot varying, calculer
   sa moyenne, son écart-type, son rang moyen sur les groupes.

C'est presque un Latin square automatisé.  Sans ça, le
rapport sur 180 pipelines est inutilisable.

Pas de tests statistiques scipy
-------------------------------
On ne reconstruit pas Friedman/Nemenyi (déjà dans Sprint 18) ;
on agrège ici les données nécessaires pour qu'un
tests statistique externe puisse les consommer.  Le rapport
existant reste libre de brancher
``picarones.core.statistics.friedman_test`` sur la sortie de
ce module.

Sortie
------
``compare_isolated_effect(runs, varying_slot)`` retourne :

.. code-block:: text

    {
        "varying_slot": str,
        "n_runs": int,
        "n_groups": int,                    # combinaisons fixed distinctes
        "values": list[str],                # valeurs distinctes du slot
        "per_value": {value: {
            "n_observations": int,
            "mean": float | None,
            "stdev": float | None,
            "min": float, "max": float,
            "mean_rank": float | None,
        }},
        "best_value": str | None,
        "worst_value": str | None,
        "groups": list[dict],               # détail par groupe
    }
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineRun:
    """Un run de pipeline composée pour la comparaison contrôlée.

    Attributes
    ----------
    name:
        Nom du run (libre — informatif uniquement).
    slots:
        Map ``{slot_name: module_name}`` décrivant la pipeline
        (ex. ``{"ocr": "tess", "llm": "gpt-4o"}``).
    score:
        Métrique numérique à comparer (CER moyen typiquement).
        Plus bas = meilleur par convention sauf si
        ``higher_is_better=True`` est passé à
        ``compare_isolated_effect``.
    """

    name: str
    slots: dict[str, str]
    score: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "slots": dict(self.slots),
            "score": self.score,
        }


def _normalise_runs(runs) -> list[PipelineRun]:
    """Accepte une liste de ``PipelineRun`` ou de dicts compatibles."""
    out: list[PipelineRun] = []
    for r in runs:
        if isinstance(r, PipelineRun):
            out.append(r)
            continue
        if not isinstance(r, dict):
            continue
        slots = r.get("slots") or {}
        if not isinstance(slots, dict):
            continue
        try:
            score = float(r.get("score"))
        except (TypeError, ValueError):
            continue
        out.append(PipelineRun(
            name=str(r.get("name") or ""),
            slots={str(k): str(v) for k, v in slots.items()},
            score=score,
        ))
    return out


def compare_isolated_effect(
    runs,
    varying_slot: str,
    *,
    higher_is_better: bool = False,
) -> Optional[dict]:
    """Mesure l'effet isolé du slot ``varying_slot``.

    Parameters
    ----------
    runs:
        Liste de ``PipelineRun`` (ou dicts compatibles).
    varying_slot:
        Nom du slot dont on veut isoler l'effet.  Les autres
        slots constituent les groupes de contrôle.
    higher_is_better:
        Si ``True``, on inverse la convention de classement
        (rang 1 = score le plus haut).  Défaut ``False`` =
        rang 1 = score le plus bas (CER).

    Returns
    -------
    dict | None
        ``None`` si moins de 2 runs ou si ``varying_slot``
        n'est présent dans aucun run.
    """
    runs_list = _normalise_runs(runs)
    if len(runs_list) < 2:
        return None
    runs_list = [r for r in runs_list if varying_slot in r.slots]
    if not runs_list:
        return None

    # Constitue les groupes par valeurs des slots fixed
    groups: dict[tuple, list[PipelineRun]] = {}
    fixed_slot_names: list[str] = []
    for r in runs_list:
        other_slots = sorted(k for k in r.slots if k != varying_slot)
        if not fixed_slot_names:
            fixed_slot_names = other_slots
        # Skip runs avec un schéma de slots incompatible
        if other_slots != fixed_slot_names:
            continue
        key = tuple((k, r.slots[k]) for k in other_slots)
        groups.setdefault(key, []).append(r)

    if not groups:
        return None

    # Pour chaque groupe : ranking des runs par score
    per_value: dict[str, dict] = {}
    group_details: list[dict] = []
    for key, members in groups.items():
        members_sorted = sorted(
            members, key=lambda x: x.score, reverse=higher_is_better,
        )
        # Rangs : runs ex aequo partagent la moyenne des rangs
        ranks: dict[str, float] = {}
        i = 0
        while i < len(members_sorted):
            j = i
            while (
                j + 1 < len(members_sorted)
                and members_sorted[j + 1].score == members_sorted[i].score
            ):
                j += 1
            avg_rank = (i + 1 + j + 1) / 2
            for k in range(i, j + 1):
                value = members_sorted[k].slots[varying_slot]
                ranks[value] = avg_rank
            i = j + 1

        for r in members:
            value = r.slots[varying_slot]
            slot = per_value.setdefault(value, {
                "scores": [],
                "ranks": [],
            })
            slot["scores"].append(r.score)
            slot["ranks"].append(ranks[value])
        group_details.append({
            "fixed_slots": dict(key),
            "n_members": len(members),
            "values": [r.slots[varying_slot] for r in members_sorted],
            "scores": [r.score for r in members_sorted],
        })

    # Calcul mean/stdev/min/max + rang moyen par valeur
    summary: dict[str, dict] = {}
    for value, slot in per_value.items():
        scores = slot["scores"]
        ranks = slot["ranks"]
        summary[value] = {
            "n_observations": len(scores),
            "mean": statistics.fmean(scores) if scores else None,
            "stdev": (
                statistics.stdev(scores) if len(scores) >= 2 else None
            ),
            "min": min(scores),
            "max": max(scores),
            "mean_rank": (
                statistics.fmean(ranks) if ranks else None
            ),
        }

    # Best/worst : sur la mean (convention CER : plus bas = meilleur)
    by_mean = sorted(
        ((v, d["mean"]) for v, d in summary.items()
         if d["mean"] is not None),
        key=lambda kv: kv[1],
        reverse=higher_is_better,
    )
    best_value = by_mean[0][0] if by_mean else None
    worst_value = by_mean[-1][0] if by_mean else None

    return {
        "varying_slot": varying_slot,
        "n_runs": len(runs_list),
        "n_groups": len(groups),
        "values": sorted(per_value.keys()),
        "per_value": summary,
        "best_value": best_value,
        "worst_value": worst_value,
        "groups": group_details,
        "higher_is_better": higher_is_better,
    }


__all__ = [
    "PipelineRun",
    "compare_isolated_effect",
]
