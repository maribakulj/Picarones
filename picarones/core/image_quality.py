"""Analyse automatique de la qualité des images de documents numérisés.

Métriques
---------
- **Score de netteté** : variance du laplacien (plus élevé = plus net)
- **Niveau de bruit** : écart-type des résidus haute-fréquence
- **Angle de rotation résiduel** : estimé par projection horizontale
- **Score de contraste** : ratio Michelson entre zones sombres (encre) et claires (fond)
- **Score de qualité global** : combinaison normalisée des métriques ci-dessus

Ces calculs sont réalisés en pur Python + bibliothèques stdlib ou Pillow.
NumPy est utilisé si disponible (calculs plus rapides), mais les méthodes
de fallback n'en dépendent pas.

Note
----
Pour les images placeholder (fixtures), des valeurs fictives cohérentes
sont générées via `generate_mock_quality_scores()`.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ImageQualityResult:
    """Métriques de qualité d'une image de document."""

    sharpness_score: float = 0.0
    """Score de netteté [0, 1]. Basé sur la variance du laplacien normalisée."""

    noise_level: float = 0.0
    """Niveau de bruit [0, 1]. 0 = pas de bruit, 1 = très bruité."""

    rotation_degrees: float = 0.0
    """Angle de rotation résiduel estimé en degrés (positif = sens horaire)."""

    contrast_score: float = 0.0
    """Score de contraste [0, 1]. Ratio Michelson encre/fond."""

    quality_score: float = 0.0
    """Score de qualité global [0, 1]. Combinaison pondérée des autres métriques."""

    analysis_method: str = "none"
    """Méthode d'analyse utilisée : 'pillow', 'numpy', 'mock'."""

    error: Optional[str] = None
    """Erreur si l'analyse a échoué."""

    @property
    def is_good_quality(self) -> bool:
        """Vrai si le score de qualité global est ≥ 0.7."""
        return self.quality_score >= 0.7

    @property
    def quality_tier(self) -> str:
        """Catégorie de qualité : 'good', 'medium', 'poor'."""
        if self.quality_score >= 0.7:
            return "good"
        elif self.quality_score >= 0.4:
            return "medium"
        return "poor"

    def as_dict(self) -> dict:
        d = {
            "sharpness_score": round(self.sharpness_score, 4),
            "noise_level": round(self.noise_level, 4),
            "rotation_degrees": round(self.rotation_degrees, 2),
            "contrast_score": round(self.contrast_score, 4),
            "quality_score": round(self.quality_score, 4),
            "quality_tier": self.quality_tier,
            "analysis_method": self.analysis_method,
        }
        if self.error:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ImageQualityResult":
        return cls(
            sharpness_score=data.get("sharpness_score", 0.0),
            noise_level=data.get("noise_level", 0.0),
            rotation_degrees=data.get("rotation_degrees", 0.0),
            contrast_score=data.get("contrast_score", 0.0),
            quality_score=data.get("quality_score", 0.0),
            analysis_method=data.get("analysis_method", "none"),
            error=data.get("error"),
        )


def analyze_image_quality(image_path: str | Path) -> ImageQualityResult:
    """Analyse la qualité d'une image de document numérisé.

    Essaie successivement :
    1. Pillow + NumPy (méthode complète)
    2. Pillow seul (méthode simplifiée)
    3. Fallback : retourne un résultat vide avec erreur

    Parameters
    ----------
    image_path:
        Chemin vers l'image (JPG, PNG, TIFF…).

    Returns
    -------
    ImageQualityResult
    """
    path = Path(image_path)
    if not path.exists():
        return ImageQualityResult(
            error=f"Fichier image introuvable : {image_path}",
            analysis_method="none",
        )

    # Essai avec Pillow + NumPy
    try:
        import numpy as np
        from PIL import Image
        return _analyze_with_numpy(path, np, Image)
    except ImportError:
        pass

    # Essai avec Pillow seul
    try:
        from PIL import Image
        return _analyze_with_pillow(path, Image)
    except ImportError:
        pass

    return ImageQualityResult(
        error="Pillow non disponible (pip install Pillow)",
        analysis_method="none",
        quality_score=0.5,  # valeur neutre
    )


def _analyze_with_numpy(path: Path, np, Image) -> ImageQualityResult:
    """Analyse complète avec NumPy."""
    img = Image.open(path).convert("L")  # niveaux de gris
    arr = np.array(img, dtype=np.float32)

    # 1. Netteté : variance du laplacien
    laplacian = _laplacian_variance_numpy(arr, np)
    # Normalisation empirique : variance > 500 = très net, < 50 = flou
    sharpness = min(1.0, laplacian / 500.0)

    # 2. Bruit : écart-type des résidus (différence image - image lissée)
    noise = _noise_level_numpy(arr, np)

    # 3. Rotation : angle d'inclinaison estimé
    rotation = _estimate_rotation_numpy(arr, np)

    # 4. Contraste : ratio Michelson
    contrast = _contrast_score_numpy(arr, np)

    # 5. Score global pondéré
    quality = _global_quality_score(sharpness, noise, abs(rotation), contrast)

    return ImageQualityResult(
        sharpness_score=float(sharpness),
        noise_level=float(noise),
        rotation_degrees=float(rotation),
        contrast_score=float(contrast),
        quality_score=float(quality),
        analysis_method="numpy",
    )


def _analyze_with_pillow(path: Path, Image) -> ImageQualityResult:
    """Analyse simplifiée avec Pillow seul (sans NumPy)."""
    img = Image.open(path).convert("L")
    pixels = list(img.getdata())
    w, h = img.size

    if not pixels:
        return ImageQualityResult(quality_score=0.5, analysis_method="pillow")

    # Contraste : étendue des valeurs
    min_val = min(pixels)
    max_val = max(pixels)
    if max_val + min_val > 0:
        contrast = (max_val - min_val) / (max_val + min_val)
    else:
        contrast = 0.0

    # Netteté approximée : variance globale des pixels
    mean_pix = statistics.mean(pixels)
    try:
        variance = statistics.variance(pixels)
    except statistics.StatisticsError:
        variance = 0.0
    sharpness = min(1.0, math.sqrt(variance) / 128.0)

    # Bruit : approximation grossière
    noise = min(1.0, statistics.stdev(pixels[:min(1000, len(pixels))]) / 64.0) if len(pixels) > 1 else 0.0

    quality = _global_quality_score(sharpness, noise, 0.0, contrast)

    return ImageQualityResult(
        sharpness_score=sharpness,
        noise_level=noise,
        rotation_degrees=0.0,  # non calculé sans NumPy
        contrast_score=contrast,
        quality_score=quality,
        analysis_method="pillow",
    )


def _laplacian_variance_numpy(arr, np) -> float:
    """Calcule la variance du laplacien (mesure de netteté)."""
    # Filtre laplacien 3x3
    laplacian_kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0],
    ], dtype=np.float32)

    # Convolution manuelle simplifiée (bordures ignorées)
    h, w = arr.shape
    if h < 3 or w < 3:
        return float(np.var(arr))

    # Utiliser une convolution rapide avec slicing
    center = arr[1:-1, 1:-1]
    top    = arr[:-2,  1:-1]
    bottom = arr[2:,   1:-1]
    left   = arr[1:-1, :-2]
    right  = arr[1:-1, 2:]
    lap = top + bottom + left + right - 4 * center

    return float(np.var(lap))


def _noise_level_numpy(arr, np) -> float:
    """Estime le niveau de bruit par la MAD (Median Absolute Deviation) des gradients."""
    h, w = arr.shape
    if h < 2 or w < 2:
        return 0.0
    # Différences horizontales et verticales
    diff_h = np.abs(arr[:, 1:] - arr[:, :-1])
    diff_v = np.abs(arr[1:, :] - arr[:-1, :])
    noise_std = float(np.median(np.concatenate([diff_h.ravel(), diff_v.ravel()])))
    # Normaliser : 0 = pas de bruit, 1 = très bruité (seuil à ~30)
    return min(1.0, noise_std / 30.0)


def _estimate_rotation_numpy(arr, np) -> float:
    """Estime l'angle de rotation par projection horizontale simplifiée.

    Retourne l'angle estimé en degrés [-45, 45].
    """
    # Méthode simplifiée : analyse de la variance des projections à différents angles
    # Limiter à quelques angles pour la performance
    h, w = arr.shape
    if h < 20 or w < 20:
        return 0.0

    # Sous-échantillonnage pour la performance
    step = max(1, h // 100)
    sample = arr[::step, :]

    best_angle = 0.0
    best_var = -1.0

    for angle_deg in range(-5, 6):  # ±5 degrés, pas de 1°
        angle_rad = math.radians(angle_deg)
        # Projection horizontale après rotation approximative
        # (approximation linéaire rapide)
        offsets = np.round(
            np.arange(sample.shape[0]) * math.tan(angle_rad)
        ).astype(int)
        offsets = np.clip(offsets, 0, w - 1)

        # Variance des sommes de lignes décalées
        try:
            row_sums = np.array([
                float(np.sum(sample[i, max(0, offsets[i]):min(w, offsets[i]+w)]))
                for i in range(sample.shape[0])
            ])
            var = float(np.var(row_sums))
            if var > best_var:
                best_var = var
                best_angle = float(angle_deg)
        except Exception:
            pass

    return best_angle


def _contrast_score_numpy(arr, np) -> float:
    """Score de contraste Michelson [0, 1]."""
    p5 = float(np.percentile(arr, 5))   # fond clair
    p95 = float(np.percentile(arr, 95))  # encre sombre
    if p5 + p95 == 0:
        return 0.0
    # Michelson : (Imax - Imin) / (Imax + Imin)
    return float((p95 - p5) / (p95 + p5))


def _global_quality_score(
    sharpness: float,
    noise: float,
    rotation_abs: float,
    contrast: float,
) -> float:
    """Calcule le score de qualité global pondéré."""
    # Poids : netteté (40%), contraste (30%), bruit (20%), rotation (10%)
    score = (
        0.40 * sharpness
        + 0.30 * contrast
        + 0.20 * (1.0 - noise)  # moins de bruit = mieux
        + 0.10 * max(0.0, 1.0 - rotation_abs / 10.0)  # ±10° max
    )
    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Données fictives pour les fixtures de démo
# ---------------------------------------------------------------------------

def generate_mock_quality_scores(
    doc_id: str,
    seed: Optional[int] = None,
) -> ImageQualityResult:
    """Génère des métriques de qualité fictives mais cohérentes pour un document.

    Utilisé par les fixtures de démo pour simuler une diversité réaliste
    de qualités d'image (bonne, moyenne, dégradée).

    Parameters
    ----------
    doc_id:
        Identifiant du document (utilisé pour la reproductibilité).
    seed:
        Graine aléatoire optionnelle.
    """
    import random
    rng = random.Random(seed or hash(doc_id) % 2**32)

    # Générer une qualité cohérente : certains docs sont plus difficiles
    # doc_id finissant par un chiffre impair → qualité variable
    last_char = doc_id[-1] if doc_id else "0"
    base_quality = 0.3 + rng.random() * 0.6  # 0.3 à 0.9

    sharpness = max(0.1, min(1.0, base_quality + rng.gauss(0, 0.1)))
    noise = max(0.0, min(1.0, (1.0 - base_quality) * 0.8 + rng.gauss(0, 0.05)))
    rotation = rng.gauss(0, 1.5)  # ±1.5° typique
    contrast = max(0.2, min(1.0, base_quality + rng.gauss(0, 0.15)))

    quality = _global_quality_score(sharpness, noise, abs(rotation), contrast)

    return ImageQualityResult(
        sharpness_score=round(sharpness, 4),
        noise_level=round(noise, 4),
        rotation_degrees=round(rotation, 2),
        contrast_score=round(contrast, 4),
        quality_score=round(quality, 4),
        analysis_method="mock",
    )


def aggregate_image_quality(results: list[ImageQualityResult]) -> dict:
    """Agrège les métriques de qualité image sur un corpus."""
    if not results:
        return {}

    valid = [r for r in results if r.error is None]
    if not valid:
        return {"error": "Aucune analyse réussie"}

    def _mean(vals: list[float]) -> float:
        return round(statistics.mean(vals), 4) if vals else 0.0

    quality_scores = [r.quality_score for r in valid]
    sharpness_scores = [r.sharpness_score for r in valid]
    noise_levels = [r.noise_level for r in valid]

    # Distribution par tier
    tiers = {"good": 0, "medium": 0, "poor": 0}
    for r in valid:
        tiers[r.quality_tier] += 1

    return {
        "mean_quality_score": _mean(quality_scores),
        "mean_sharpness": _mean(sharpness_scores),
        "mean_noise_level": _mean(noise_levels),
        "quality_distribution": tiers,
        "document_count": len(valid),
        "scores": [r.quality_score for r in valid],  # pour scatter plot
    }
