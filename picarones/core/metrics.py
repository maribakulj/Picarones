"""Modèle de données des métriques (Cercle 1).

Abstractions pures pour représenter les métriques calculées sur une
paire (référence, hypothèse) — pas de dépendance externe (pas de
jiwer, pas de scipy).

Le calcul effectif via jiwer vit en cercle 2 dans
:mod:`picarones.measurements.metrics` (``compute_metrics``).
L'agrégation statistique vit ici car elle n'utilise que la stdlib
(``statistics``).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional


@dataclass
class MetricsResult:
    """Ensemble des métriques calculées pour une paire (référence, hypothèse)."""

    cer: float
    cer_nfc: float
    cer_caseless: float
    wer: float
    wer_normalized: float
    mer: float
    wil: float
    reference_length: int
    hypothesis_length: int
    error: Optional[str] = None
    cer_diplomatic: Optional[float] = None
    """CER calculé après normalisation diplomatique (ſ=s, u=v, i=j…).
    None si aucun profil diplomatique n'a été fourni à compute_metrics.
    """
    diplomatic_profile_name: Optional[str] = None
    """Nom du profil de normalisation diplomatique utilisé."""

    def as_dict(self) -> dict:
        d = {
            "cer": round(self.cer, 6),
            "cer_nfc": round(self.cer_nfc, 6),
            "cer_caseless": round(self.cer_caseless, 6),
            "wer": round(self.wer, 6),
            "wer_normalized": round(self.wer_normalized, 6),
            "mer": round(self.mer, 6),
            "wil": round(self.wil, 6),
            "reference_length": self.reference_length,
            "hypothesis_length": self.hypothesis_length,
            "error": self.error,
        }
        if self.cer_diplomatic is not None:
            d["cer_diplomatic"] = round(self.cer_diplomatic, 6)
            d["diplomatic_profile_name"] = self.diplomatic_profile_name
        return d

    @property
    def cer_percent(self) -> float:
        return round(self.cer * 100, 2)

    @property
    def wer_percent(self) -> float:
        return round(self.wer * 100, 2)


def aggregate_metrics(results: list[MetricsResult]) -> dict:
    """Calcule les statistiques agrégées sur un ensemble de résultats.

    Parameters
    ----------
    results:
        Liste de MetricsResult correspondant à plusieurs documents.

    Returns
    -------
    dict
        Statistiques : moyenne, médiane, min, max, std pour chaque métrique.
    """
    if not results:
        return {}

    def _stats(values: list[float]) -> dict:
        if not values:
            return {}
        return {
            "mean": round(statistics.mean(values), 6),
            "median": round(statistics.median(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        }

    metric_names = ["cer", "cer_nfc", "cer_caseless", "wer", "wer_normalized", "mer", "wil"]
    aggregated: dict = {}
    for metric in metric_names:
        values = [getattr(r, metric) for r in results if r.error is None]
        aggregated[metric] = _stats(values)

    # CER diplomatique (optionnel — présent seulement si calculé)
    diplo_values = [
        r.cer_diplomatic for r in results
        if r.error is None and r.cer_diplomatic is not None
    ]
    if diplo_values:
        aggregated["cer_diplomatic"] = _stats(diplo_values)
        # Nom du profil (même pour tous les docs d'un corpus)
        profile_name = next(
            (r.diplomatic_profile_name for r in results if r.diplomatic_profile_name),
            None,
        )
        if profile_name:
            aggregated["cer_diplomatic"]["profile"] = profile_name

    aggregated["document_count"] = len(results)
    aggregated["failed_count"] = sum(1 for r in results if r.error is not None)

    return aggregated
