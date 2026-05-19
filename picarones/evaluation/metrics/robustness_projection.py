"""Projection de robustesse synthétique sur le corpus réel —
Sprint 81 (A.I.8).

A.I.8 du plan d'évolution 2026.

Pourquoi ce module
------------------
Le module ``picarones/core/robustness.py`` (Sprint 8) génère des
courbes CER vs niveau de dégradation **synthétique** (bruit, flou,
rotation, résolution).  ``picarones/core/image_quality.py`` mesure
le bruit/flou/contraste **réels** des images du corpus.  Ce
sprint **projette** les caractéristiques réelles sur les courbes
synthétiques pour estimer le **déficit attendu de CER** sur le
corpus dans son état actuel.

Lecture concrète
----------------
*« 30 % de vos documents ont un bruit équivalent à σ=15 où
Tesseract perd 8 points de CER — soit un déficit attendu global
de 2,4 points (30 % × 8 points). »*

Méthode
-------
1. Pour chaque document, on extrait la valeur de qualité réelle
   (``noise_level``, ``blur_score``, ``contrast_score``…) depuis
   ``ImageQualityResult``.
2. Pour chaque type de dégradation, on interpole linéairement la
   ``DegradationCurve`` synthétique : CER attendu à ce niveau.
3. On agrège : CER moyen attendu, % docs au-dessus du seuil
   critique de la courbe, déficit projeté = CER_attendu -
   CER_baseline (niveau nul).

Sortie
------
``project_robustness_on_corpus(curves, image_qualities)`` retourne
``{engine_name: {degradation_type: {expected_cer_mean,
deficit_vs_baseline, n_docs_above_critical, n_docs}}}``.

Limites
-------
- Mapping ``image_quality → degradation level`` : on suppose que
  ``noise_level`` (ImageQualityResult) correspond à σ
  (DegradationCurve), et idem pour ``blur_score`` ↔ rayon de
  flou.  Si un corpus expose ces valeurs avec une échelle
  différente, le mapping est documenté et l'utilisateur peut
  passer ``quality_to_level`` custom.
- Interpolation **linéaire** entre les points de la courbe.  Au-
  delà des bornes, on **clip** au point extrême (pas
  d'extrapolation hasardeuse).
"""

from __future__ import annotations

import logging
import statistics
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)


# Mapping par défaut entre attributs ImageQualityResult et types
# de dégradation synthétique.  L'utilisateur peut passer un dict
# custom pour modifier ce mapping.
_DEFAULT_QUALITY_FIELD: dict[str, str] = {
    "noise":      "noise_level",       # σ
    "blur":       "blur_score",        # Variance laplacienne (inverse)
    "contrast":   "contrast_score",
    "rotation":   "rotation_angle",
    "resolution": "resolution_score",  # peut être absent
}


def _interpolate_cer(
    levels: list[float],
    cer_values: list[Optional[float]],
    target_level: float,
) -> Optional[float]:
    """Interpolation linéaire : retourne CER attendu à
    ``target_level``.

    - Si ``target_level`` est en-dessous du minimum de levels,
      retourne le CER au minimum (clip).
    - Si au-dessus du maximum, retourne le CER au maximum.
    - Sinon, interpolation linéaire entre les deux points
      encadrants.
    - Retourne ``None`` si aucun ``cer_value`` valide.
    """
    if not levels:
        return None
    # Filtrer les paires (level, cer) où cer est None
    pairs = [
        (lvl, cer) for lvl, cer in zip(levels, cer_values)
        if cer is not None
    ]
    if not pairs:
        return None
    pairs.sort(key=lambda p: p[0])
    # Clip
    if target_level <= pairs[0][0]:
        return pairs[0][1]
    if target_level >= pairs[-1][0]:
        return pairs[-1][1]
    # Interpolation
    for i in range(len(pairs) - 1):
        lo_lvl, lo_cer = pairs[i]
        hi_lvl, hi_cer = pairs[i + 1]
        if lo_lvl <= target_level <= hi_lvl:
            if hi_lvl == lo_lvl:
                return lo_cer
            ratio = (target_level - lo_lvl) / (hi_lvl - lo_lvl)
            return lo_cer + (hi_cer - lo_cer) * ratio
    return None  # ne devrait pas arriver


def _extract_quality_value(
    quality: dict, degradation_type: str,
    custom_mapping: Optional[dict[str, str]] = None,
) -> Optional[float]:
    """Extrait la valeur de qualité pertinente pour un type de
    dégradation depuis un ``ImageQualityResult.as_dict()``."""
    mapping = custom_mapping or _DEFAULT_QUALITY_FIELD
    field = mapping.get(degradation_type)
    if field is None:
        return None
    value = quality.get(field)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def project_robustness_on_corpus(
    curves: Iterable,
    image_qualities: list[dict],
    *,
    quality_to_level: Optional[Callable[[dict, str], Optional[float]]] = None,
    critical_threshold: Optional[float] = None,
) -> dict:
    """Projette les courbes de robustesse sur les qualités réelles.

    Parameters
    ----------
    curves:
        Itérable de ``DegradationCurve`` (ou dicts compatibles
        avec ``engine_name``, ``degradation_type``, ``levels``,
        ``cer_values``, ``critical_threshold_level``).
    image_qualities:
        Liste de dicts ``ImageQualityResult.as_dict()`` (un par
        document).  Si vide, retourne une projection vide.
    quality_to_level:
        Fonction custom ``(quality_dict, degradation_type) →
        Optional[float]`` pour adapter le mapping qualité→niveau.
        Par défaut, utilise ``_DEFAULT_QUALITY_FIELD``.
    critical_threshold:
        Override pour le seuil critique de CER (défaut : utilise
        ``DegradationCurve.cer_threshold``).

    Returns
    -------
    dict
        ``{
            engine_name: {
                degradation_type: {
                    "n_docs": int,
                    "n_docs_with_data": int,    # qualité disponible
                    "expected_cer_mean": float, # moyenne CER attendu
                    "expected_cer_median": float,
                    "baseline_cer": float,      # CER à niveau min
                    "deficit_vs_baseline": float,
                    "n_docs_above_critical": int,
                    "critical_threshold_level": float | None,
                    "critical_threshold_cer": float,
                },
            },
        }``
    """
    extractor = quality_to_level or (
        lambda q, dt: _extract_quality_value(q, dt)
    )
    out: dict[str, dict] = {}

    for curve in curves:
        # Accepter dict ou DegradationCurve
        if hasattr(curve, "as_dict"):
            data = curve.as_dict()
        else:
            data = curve
        engine = data.get("engine_name")
        deg_type = data.get("degradation_type")
        levels = data.get("levels") or []
        cer_values = data.get("cer_values") or []
        crit_lvl = data.get("critical_threshold_level")
        crit_cer = (
            critical_threshold
            if critical_threshold is not None
            else data.get("cer_threshold", 0.20)
        )
        if not engine or not deg_type:
            continue

        per_doc_cer: list[float] = []
        n_docs_with_data = 0
        n_above_critical = 0
        for quality in image_qualities:
            level = extractor(quality, deg_type)
            if level is None:
                continue
            n_docs_with_data += 1
            cer = _interpolate_cer(levels, cer_values, level)
            if cer is None:
                continue
            per_doc_cer.append(cer)
            if cer > crit_cer:
                n_above_critical += 1

        if not per_doc_cer:
            continue

        # Baseline = CER au niveau minimum (sans dégradation)
        baseline = _interpolate_cer(
            levels, cer_values,
            min(levels) if levels else 0.0,
        )
        expected_mean = statistics.fmean(per_doc_cer)
        expected_median = statistics.median(per_doc_cer)
        deficit = (
            expected_mean - baseline
            if baseline is not None else None
        )

        out.setdefault(engine, {})[deg_type] = {
            "n_docs": len(image_qualities),
            "n_docs_with_data": n_docs_with_data,
            "expected_cer_mean": expected_mean,
            "expected_cer_median": expected_median,
            "baseline_cer": baseline,
            "deficit_vs_baseline": deficit,
            "n_docs_above_critical": n_above_critical,
            "critical_threshold_level": crit_lvl,
            "critical_threshold_cer": crit_cer,
        }
    return out


def aggregate_projection_per_engine(projection: dict) -> dict:
    """Pour chaque moteur, agrège le déficit projeté en sommant
    sur tous les types de dégradation.

    Lecture : *« déficit total attendu pour Tesseract = 5,2 points
    de CER si on considère les 4 dégradations indépendamment »*.

    Note : la sommation **suppose l'indépendance** des
    dégradations, ce qui n'est pas strictement vrai mais reste
    une approximation utile pour le diagnostic.
    """
    out: dict[str, dict] = {}
    for engine, per_type in projection.items():
        total_deficit = 0.0
        n_types_with_data = 0
        max_deficit_type: Optional[tuple[str, float]] = None
        for deg_type, stats in per_type.items():
            deficit = stats.get("deficit_vs_baseline")
            if deficit is None:
                continue
            total_deficit += deficit
            n_types_with_data += 1
            if max_deficit_type is None or deficit > max_deficit_type[1]:
                max_deficit_type = (deg_type, deficit)
        out[engine] = {
            "total_expected_deficit": total_deficit,
            "n_degradation_types": n_types_with_data,
            "worst_degradation_type": (
                max_deficit_type[0] if max_deficit_type else None
            ),
            "worst_degradation_deficit": (
                max_deficit_type[1] if max_deficit_type else None
            ),
        }
    return out


__all__ = [
    "project_robustness_on_corpus",
    "aggregate_projection_per_engine",
]
