"""Score de difficulté intrinsèque par document.

Le score est indépendant des moteurs OCR : il mesure la difficulté
*objective* d'un document, indépendamment de la qualité des transcriptions.

Formule
-------
  difficulty = w_variance * variance_norm
             + w_quality  * (1 - image_quality_score)
             + w_density  * special_char_density

où :
  - variance_norm   : variance inter-moteurs du CER, normalisée [0, 1]
  - image_quality   : score de qualité image [0, 1] (netteté, contraste…)
  - special_chars   : densité de caractères spéciaux dans la GT [0, 1]

Les poids sont configurables (défaut : 0.4 / 0.35 / 0.25).

Score final : [0, 1] — 0 = document facile, 1 = très difficile.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional


# Poids par défaut
_W_VARIANCE = 0.40
_W_QUALITY  = 0.35
_W_DENSITY  = 0.25

# Caractères spéciaux patrimoniaux (ligatures, abréviations, diacritiques rares)
_SPECIAL_CHARS_RE = re.compile(
    r"[ſœæꝑꝓ&]"              # ligatures / abréviations médiévales
    r"|[ḁ-ỿ]"                # Latin Étendu Additionnel (diacritiques rares)
    r"|[\u0300-\u036f]"       # Diacritiques combinants
    r"|[\ufb00-\ufb06]"       # Formes de présentation latines (fi, fl…)
    r"|[IVXLCDM]{3,}"         # Chiffres romains (3+ caractères)
)


@dataclass
class DifficultyScore:
    """Score de difficulté intrinsèque d'un document."""
    doc_id: str
    score: float
    """Score global [0, 1] — plus élevé = plus difficile."""
    variance_component: float
    """Composante variance inter-moteurs [0, 1]."""
    quality_component: float
    """Composante qualité image inversée [0, 1]."""
    density_component: float
    """Composante densité caractères spéciaux [0, 1]."""
    cer_variance: float
    """Variance brute du CER entre moteurs."""
    image_quality_score: float
    """Score de qualité image (si disponible, sinon 0.5)."""
    special_char_ratio: float
    """Ratio caractères spéciaux / longueur GT."""

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 4),
            "variance_component": round(self.variance_component, 4),
            "quality_component": round(self.quality_component, 4),
            "density_component": round(self.density_component, 4),
            "cer_variance": round(self.cer_variance, 6),
            "image_quality_score": round(self.image_quality_score, 4),
            "special_char_ratio": round(self.special_char_ratio, 4),
        }


def _special_char_density(text: str) -> float:
    """Ratio de caractères spéciaux patrimoniaux dans le texte."""
    if not text:
        return 0.0
    matches = len(_SPECIAL_CHARS_RE.findall(text))
    return min(1.0, matches / len(text))


def _variance(values: list[float]) -> float:
    """Variance d'une liste de valeurs."""
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    return sum((v - mu) ** 2 for v in values) / len(values)


def compute_difficulty_score(
    doc_id: str,
    ground_truth: str,
    cer_per_engine: list[float],
    image_quality_score: Optional[float] = None,
    weights: tuple[float, float, float] = (_W_VARIANCE, _W_QUALITY, _W_DENSITY),
) -> DifficultyScore:
    """Calcule le score de difficulté intrinsèque pour un document.

    Parameters
    ----------
    doc_id             : identifiant du document
    ground_truth       : texte de référence
    cer_per_engine     : liste des CER (un par moteur concurrent)
    image_quality_score: score de qualité image [0, 1] (None → 0.5 neutre)
    weights            : (w_variance, w_quality, w_density)

    Returns
    -------
    DifficultyScore
    """
    w_var, w_qual, w_den = weights

    # 1. Variance inter-moteurs (normalisée sur [0, 1] — variance max ≈ 0.25)
    cer_var = _variance(cer_per_engine)
    variance_norm = min(1.0, cer_var / 0.25)

    # 2. Qualité image inversée
    iq = image_quality_score if image_quality_score is not None else 0.5
    iq = max(0.0, min(1.0, iq))
    quality_component = 1.0 - iq

    # 3. Densité de caractères spéciaux
    density = _special_char_density(ground_truth)
    # Amplifier légèrement (la densité brute est souvent faible)
    density_component = min(1.0, density * 3.0)

    # Score combiné
    score = (
        w_var  * variance_norm
        + w_qual * quality_component
        + w_den  * density_component
    )
    score = max(0.0, min(1.0, score))

    return DifficultyScore(
        doc_id=doc_id,
        score=score,
        variance_component=variance_norm,
        quality_component=quality_component,
        density_component=density_component,
        cer_variance=cer_var,
        image_quality_score=iq,
        special_char_ratio=density,
    )


def compute_all_difficulties(
    doc_ids: list[str],
    ground_truths: dict[str, str],
    cer_map: dict[str, dict[str, float]],
    image_quality_map: Optional[dict[str, float]] = None,
) -> dict[str, DifficultyScore]:
    """Calcule les scores de difficulté pour tous les documents d'un corpus.

    Parameters
    ----------
    doc_ids            : liste des identifiants de documents
    ground_truths      : {doc_id → gt_text}
    cer_map            : {doc_id → {engine_name → cer}}
    image_quality_map  : {doc_id → quality_score} (facultatif)

    Returns
    -------
    {doc_id → DifficultyScore}
    """
    result = {}
    for doc_id in doc_ids:
        gt = ground_truths.get(doc_id, "")
        engine_cers = list(cer_map.get(doc_id, {}).values())
        iq = (image_quality_map or {}).get(doc_id)
        result[doc_id] = compute_difficulty_score(
            doc_id=doc_id,
            ground_truth=gt,
            cer_per_engine=engine_cers,
            image_quality_score=iq,
        )
    return result


def difficulty_label(score: float) -> str:
    """Retourne un label lisible pour un score de difficulté."""
    if score < 0.25:
        return "Facile"
    if score < 0.50:
        return "Modéré"
    if score < 0.75:
        return "Difficile"
    return "Très difficile"


def difficulty_color(score: float) -> str:
    """Retourne une couleur CSS pour un score de difficulté."""
    if score < 0.25:
        return "#16a34a"   # vert
    if score < 0.50:
        return "#ca8a04"   # jaune
    if score < 0.75:
        return "#ea580c"   # orange
    return "#dc2626"       # rouge
