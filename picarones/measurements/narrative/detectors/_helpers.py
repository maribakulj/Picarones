"""Helpers internes partagés par les détecteurs narratifs.

Chantier 5 du plan d'évolution post-Sprint 97 — découpage de
``picarones/core/narrative/detectors.py`` (1229 lignes, 18 détecteurs)
en 6 sous-modules thématiques + ce module d'helpers communs.

Ces fonctions étaient privées (préfixe ``_``) au module historique.
Elles sont conservées telles quelles ici ; les sous-modules les
importent.
"""

from __future__ import annotations

from typing import Optional


def _engines_summary(data: dict) -> list[dict]:
    """Accès normalisé à la liste des résumés moteur."""
    return data.get("engines", []) or []


def _engine_by_name(data: dict, name: str) -> Optional[dict]:
    for e in _engines_summary(data):
        if e.get("name") == name:
            return e
    return None


def _n_docs(data: dict) -> int:
    meta = data.get("meta", {}) or {}
    return int(meta.get("document_count") or 0)


def _mean_duration_per_engine(data: dict) -> dict[str, float]:
    """Retourne ``{engine_name: mean_duration_seconds}`` quand disponible.

    Lit ``benchmark_data["engines"][i]["mean_duration"]`` (renseigné
    par le runner depuis ``durations_by_engine`` Sprint 4). Filtre les
    durées non-numériques.
    """
    out: dict[str, float] = {}
    for e in _engines_summary(data):
        name = e.get("name")
        if not name:
            continue
        dur = e.get("mean_duration")
        if dur is None:
            continue
        try:
            dur_f = float(dur)
        except (TypeError, ValueError):
            continue
        if dur_f > 0:
            out[name] = dur_f
    return out
