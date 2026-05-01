"""Helpers internes partagés par les détecteurs narratifs."""

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
    """Durée moyenne d'exécution par moteur (en secondes par document).

    Lit ``benchmark_data["documents"][i]["engine_results"][j]["duration"]``
    (format runner). Filtre les durées non-numériques.
    """
    durations: dict[str, list[float]] = {}
    for doc in data.get("documents") or []:
        for er in doc.get("engine_results") or []:
            engine_name = er.get("engine")
            d = er.get("duration")
            if engine_name is None or d is None:
                continue
            try:
                d_f = float(d)
            except (TypeError, ValueError):
                continue
            durations.setdefault(engine_name, []).append(d_f)
    return {k: sum(v) / len(v) for k, v in durations.items() if v}
