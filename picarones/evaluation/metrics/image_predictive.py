"""Métriques d'image prédictives — Sprint 93 (A.II.7).

A.II.7 du plan d'évolution 2026.

Pourquoi ce module
------------------
``image_quality`` (Sprint 5) mesure des features d'image
indépendamment ; ce module **les combine** pour produire deux
indicateurs corpus-level :

1. **Score de complexité paléographique** ∈ [0, 1].  Combine
   bruit, faible netteté, faible contraste et rotation en un
   indicateur unique de la difficulté intrinsèque pour un OCR.
   0 = document trivial, 1 = document extrême.  Permet
   d'expliquer une partie du CER observé.

2. **Score d'homogénéité du corpus** ∈ [0, 1].  Variance des
   features entre documents.  0 = corpus uniforme (la moyenne
   globale du benchmark est fiable), 1 = corpus hétérogène
   (la moyenne ment, il faut stratifier).  Couplé au détecteur
   ``stratification_recommended`` (Sprint 46) qui agit sur
   ``script_type``.

Pondérations
------------
La roadmap propose une combinaison **pondérée** sans fixer les
poids — on adopte une convention éditoriale documentée :

- ``noise_level``        : poids 0.30 (bruit franc → CER ↑)
- ``1 - sharpness_score`` : poids 0.30 (flou → CER ↑)
- ``1 - contrast_score``  : poids 0.20 (faible contraste → CER ↑)
- ``|rotation_degrees|/30``  : poids 0.20 (rotation > 30° = pire)

Les poids somment à 1.  L'utilisateur peut surcharger via
``weights={...}``.

Pas de prédiction CER absolue
-----------------------------
On ne prétend **pas** prédire une valeur CER en pourcentage —
ça demanderait un modèle entraîné par moteur, ce que la
philosophie banc d'essai exclut.  On fournit un score relatif
qui se corrèle au CER observé pour une **lecture
diagnostique** : *« le document A est ~3× plus complexe que le
document B, ce qui est cohérent avec le CER observé. »*
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# Poids éditoriaux par défaut.
DEFAULT_COMPLEXITY_WEIGHTS = {
    "noise_level": 0.30,
    "blur": 0.30,           # 1 - sharpness_score
    "low_contrast": 0.20,   # 1 - contrast_score
    "rotation": 0.20,       # |rotation_degrees| / 30
}


# Plage de saturation pour la rotation.  Au-delà de 30°, on
# considère que c'est aussi pire que pire.
_ROTATION_SATURATION_DEG = 30.0


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _extract_feature(
    quality: dict, key: str, default: float = 0.0,
) -> float:
    val = quality.get(key, default)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def compute_paleographic_complexity(
    quality: dict,
    *,
    weights: Optional[dict[str, float]] = None,
) -> Optional[dict]:
    """Score de complexité paléographique d'une image.

    Parameters
    ----------
    quality:
        Dict ``ImageQualityResult.as_dict()`` ou compatible.
        Champs lus : ``noise_level``, ``sharpness_score``,
        ``contrast_score``, ``rotation_degrees``.
    weights:
        Poids surchargeant les défauts.  Doit contenir les
        4 clés ``noise_level``, ``blur``, ``low_contrast``,
        ``rotation``.  Les poids sont normalisés (somme = 1).

    Returns
    -------
    dict | None
        ``{
            "score": float,                 # ∈ [0, 1]
            "components": {
                "noise": float, "blur": float,
                "low_contrast": float, "rotation": float,
            },
            "weights_used": dict,
        }`` ou ``None`` si ``quality`` est falsy.
    """
    if not quality:
        return None
    w = dict(DEFAULT_COMPLEXITY_WEIGHTS)
    if weights:
        for k in w:
            if k in weights:
                w[k] = float(weights[k])
    total = sum(w.values())
    if total <= 0:
        return None
    w = {k: v / total for k, v in w.items()}
    noise = _clip01(_extract_feature(quality, "noise_level"))
    sharpness = _clip01(_extract_feature(quality, "sharpness_score"))
    contrast = _clip01(_extract_feature(quality, "contrast_score"))
    rotation_deg = abs(_extract_feature(quality, "rotation_degrees"))
    blur = 1.0 - sharpness
    low_contrast = 1.0 - contrast
    rotation = _clip01(rotation_deg / _ROTATION_SATURATION_DEG)
    score = (
        w["noise_level"] * noise
        + w["blur"] * blur
        + w["low_contrast"] * low_contrast
        + w["rotation"] * rotation
    )
    return {
        "score": _clip01(score),
        "components": {
            "noise": noise,
            "blur": blur,
            "low_contrast": low_contrast,
            "rotation": rotation,
        },
        "weights_used": w,
    }


def compute_corpus_homogeneity(
    image_qualities: Iterable[dict],
) -> Optional[dict]:
    """Score d'homogénéité du corpus ∈ [0, 1].

    0 = corpus uniforme (faible variance entre documents),
    1 = corpus hétérogène.

    Méthode : pour chaque feature dans ``noise_level``,
    ``sharpness_score``, ``contrast_score``, ``rotation_degrees``,
    on calcule l'écart-type *normalisé* sur les documents (par
    une plage de référence), puis on prend la moyenne des 4.

    Plages de normalisation :
    - ``noise_level``, ``sharpness_score``, ``contrast_score``
      ∈ [0, 1] → écart-type / 0.5 (max théorique de l'écart-type
      d'une distribution sur [0,1]) borné à 1.
    - ``rotation_degrees`` → écart-type / 10°.

    Parameters
    ----------
    image_qualities:
        Itérable de dicts ``ImageQualityResult.as_dict()``.

    Returns
    -------
    dict | None
        ``{
            "score": float,                 # ∈ [0, 1]
            "n_docs": int,
            "per_feature": {
                feature: {"mean": float, "stdev": float,
                          "normalised": float},
            },
        }`` ou ``None`` si moins de 2 documents.
    """
    docs = [q for q in image_qualities if q]
    if len(docs) < 2:
        return None
    features = (
        ("noise_level", 0.5),
        ("sharpness_score", 0.5),
        ("contrast_score", 0.5),
        ("rotation_degrees", 10.0),
    )
    per_feature: dict[str, dict] = {}
    norm_stdevs: list[float] = []
    for key, divisor in features:
        values = [
            _extract_feature(q, key)
            for q in docs
        ]
        if not values:
            continue
        mean = statistics.fmean(values)
        try:
            stdev = statistics.stdev(values) if len(values) >= 2 else 0.0
        except statistics.StatisticsError:
            stdev = 0.0
        normalised = _clip01(stdev / divisor) if divisor > 0 else 0.0
        per_feature[key] = {
            "mean": mean,
            "stdev": stdev,
            "normalised": normalised,
        }
        norm_stdevs.append(normalised)
    if not norm_stdevs:
        return None
    score = statistics.fmean(norm_stdevs)
    return {
        "score": _clip01(score),
        "n_docs": len(docs),
        "per_feature": per_feature,
    }


def aggregate_corpus_predictive(
    image_qualities: Iterable[dict],
    *,
    weights: Optional[dict[str, float]] = None,
) -> Optional[dict]:
    """Synthèse corpus-wide : complexité moyenne + homogénéité.

    Returns
    -------
    dict | None
        ``{
            "n_docs": int,
            "complexity_mean": float,
            "complexity_median": float,
            "complexity_min": float,
            "complexity_max": float,
            "complexity_stdev": float,
            "homogeneity": dict,            # sortie de
                                            # compute_corpus_homogeneity
        }`` ou ``None`` si moins d'un document.
    """
    docs = [q for q in image_qualities if q]
    if not docs:
        return None
    scores: list[float] = []
    for q in docs:
        result = compute_paleographic_complexity(q, weights=weights)
        if result is not None:
            scores.append(float(result["score"]))
    if not scores:
        return None
    homogeneity = compute_corpus_homogeneity(docs)
    return {
        "n_docs": len(docs),
        "complexity_mean": statistics.fmean(scores),
        "complexity_median": statistics.median(scores),
        "complexity_min": min(scores),
        "complexity_max": max(scores),
        "complexity_stdev": (
            statistics.stdev(scores) if len(scores) >= 2 else 0.0
        ),
        "homogeneity": homogeneity,
    }


__all__ = [
    "DEFAULT_COMPLEXITY_WEIGHTS",
    "compute_paleographic_complexity",
    "compute_corpus_homogeneity",
    "aggregate_corpus_predictive",
]


# Évite warning import inutilisé
_ = math
