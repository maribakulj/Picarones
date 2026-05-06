"""Calibration des moteurs : ECE, MCE, reliability diagram.

Sprint 39 — A.II.1.b du plan d'évolution 2026 : couche de calcul pure.

Pourquoi ce module
------------------
Tous les moteurs OCR cibles fournissent une confidence par token ou par
ligne (Tesseract via le ``tsv``, Pero OCR via le ``PageLayout``,
Mistral OCR via ``confidence``, Google Vision via ``Word.confidence``).
La question naturelle pour un workflow patrimonial est : *« quand le
moteur dit qu'il est sûr, est-il vraiment sûr ? »*.  Pour une équipe
qui doit vérifier humainement un corpus de 50 000 pages, la différence
entre vérifier 100 % vs 15 % du volume est l'effet de la calibration.

Ce module fournit les trois mesures classiques :

- **Expected Calibration Error (ECE)** — moyenne pondérée par bin de
  l'écart absolu entre confiance moyenne et précision moyenne.
  ``ECE = 0`` ↔ moteur parfaitement calibré ; ``ECE`` élevé ↔ écart
  systématique entre confiance affichée et fiabilité réelle.
- **Maximum Calibration Error (MCE)** — max de cet écart sur les bins.
  Utile pour repérer le pire mensonge du moteur (ex. il dit toujours
  95 % de confiance et il a tort une fois sur deux).
- **Reliability diagram** — table ``[(bin_low, bin_high, avg_conf,
  accuracy, count)]`` qui peut être rendue en SVG côté serveur ou en
  Chart.js côté navigateur dans un sprint suivant.

Stratégie de découpage
----------------------
Comme pour le NER (Sprint 38) et la divergence (Sprints 35-37),
on découpe :

- **Sprint 39** (ici) — couche de calcul pure : entrée = deux listes
  parallèles ``confidences`` (∈ [0, 1]) et ``is_correct`` (bool/0-1).
  Aucune dépendance externe.
- **Sprint à venir** — exposition de ``token_confidences`` sur
  ``EngineResult``, alignement caractère/token avec la GT pour produire
  ``is_correct``, intégration dans le runner et vue HTML reliability.

Ce qui est explicitement hors scope
-----------------------------------
Ce sprint ne touche **aucun adaptateur OCR**.  Aucune confiance n'est
extraite ; on calcule uniquement à partir de séquences de prédictions
fournies en entrée.  C'est ce qui permet de tester rigoureusement les
invariants mathématiques (ECE = 0 ↔ calibré, ECE = |bias| pour bias
constant, etc.) sans dépendre d'un backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Modèle de données
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationBin:
    """Un bin du reliability diagram.

    Attributs
    ---------
    bin_low, bin_high:
        Bornes du bin sur l'axe de confiance (``[bin_low, bin_high)`` —
        sauf le dernier bin qui inclut ``1.0``).
    avg_confidence:
        Moyenne des confidences des prédictions tombées dans le bin.
        ``None`` si le bin est vide.
    accuracy:
        Fraction de prédictions correctes dans le bin (``∈ [0, 1]``).
        ``None`` si le bin est vide.
    count:
        Nombre de prédictions dans le bin.
    """

    bin_low: float
    bin_high: float
    avg_confidence: float | None
    accuracy: float | None
    count: int

    @property
    def gap(self) -> float | None:
        """Écart absolu ``|confidence - accuracy|`` ou ``None`` si vide."""
        if self.avg_confidence is None or self.accuracy is None:
            return None
        return abs(self.avg_confidence - self.accuracy)


# ──────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────


def _validate_inputs(
    confidences: list[float],
    is_correct: list[bool | int],
) -> None:
    if len(confidences) != len(is_correct):
        raise ValueError(
            f"Longueurs incompatibles : confidences={len(confidences)} "
            f"vs is_correct={len(is_correct)}"
        )
    for i, c in enumerate(confidences):
        if not (0.0 <= float(c) <= 1.0):
            raise ValueError(
                f"Confiance hors [0, 1] à l'index {i} : {c!r}"
            )


# ──────────────────────────────────────────────────────────────────────────
# Reliability diagram (binning)
# ──────────────────────────────────────────────────────────────────────────


def reliability_diagram(
    confidences: Iterable[float],
    is_correct: Iterable[bool | int],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Découpe les prédictions en ``n_bins`` bins équidistants par confiance
    et calcule pour chacun la confiance moyenne, la précision et le compte.

    Parameters
    ----------
    confidences:
        Confidences des prédictions, ``∈ [0, 1]``.
    is_correct:
        Indicateur booléen (1 = prédiction correcte, 0 = incorrecte).
    n_bins:
        Nombre de bins (défaut : 10).  Bornes : ``[k/n_bins, (k+1)/n_bins)``
        sauf le dernier bin qui inclut ``1.0``.

    Returns
    -------
    list[CalibrationBin]
        Liste de ``n_bins`` bins, dans l'ordre croissant des confidences.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins doit être ≥ 1 — reçu {n_bins}")

    confs = [float(c) for c in confidences]
    correct = [int(bool(x)) for x in is_correct]
    _validate_inputs(confs, correct)

    bin_width = 1.0 / n_bins
    sums: list[float] = [0.0] * n_bins
    correct_counts: list[int] = [0] * n_bins
    counts: list[int] = [0] * n_bins

    for c, ok in zip(confs, correct):
        # Calcul du bin index par multiplication ``c * n_bins`` plutôt que
        # division ``c / bin_width`` pour éviter les pièges de
        # représentation flottante (ex. ``0.6 / 0.1 = 5.999…`` en IEEE 754
        # qui placerait 0.6 dans le bin [0.5, 0.6) au lieu de [0.6, 0.7)).
        if c >= 1.0:
            idx = n_bins - 1
        else:
            idx = int(c * n_bins)
            # Garde-fou en cas d'arrondi flottant
            if idx >= n_bins:
                idx = n_bins - 1
            elif idx < 0:
                idx = 0
        sums[idx] += c
        correct_counts[idx] += ok
        counts[idx] += 1

    bins: list[CalibrationBin] = []
    for k in range(n_bins):
        low = k * bin_width
        high = (k + 1) * bin_width
        n = counts[k]
        if n == 0:
            bins.append(CalibrationBin(low, high, None, None, 0))
        else:
            bins.append(CalibrationBin(
                bin_low=low,
                bin_high=high,
                avg_confidence=sums[k] / n,
                accuracy=correct_counts[k] / n,
                count=n,
            ))
    return bins


# ──────────────────────────────────────────────────────────────────────────
# ECE et MCE
# ──────────────────────────────────────────────────────────────────────────


def expected_calibration_error(
    confidences: Iterable[float],
    is_correct: Iterable[bool | int],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error : moyenne pondérée par bin de l'écart
    absolu confiance ↔ précision.

    ``ECE = sum_k (n_k / N) * |avg_conf_k - accuracy_k|``

    où la somme porte sur les bins non vides.

    Returns
    -------
    float
        ``∈ [0, 1]``.  ``0`` ↔ calibration parfaite.
    """
    bins = reliability_diagram(confidences, is_correct, n_bins=n_bins)
    total = sum(b.count for b in bins)
    if total == 0:
        return 0.0
    ece = 0.0
    for b in bins:
        if b.count == 0 or b.gap is None:
            continue
        ece += (b.count / total) * b.gap
    return ece


def maximum_calibration_error(
    confidences: Iterable[float],
    is_correct: Iterable[bool | int],
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error : pire écart confiance ↔ précision sur
    tous les bins non vides.

    Utile pour repérer un mensonge ponctuel du moteur (ex. il dit 95 %
    de confiance et il a tort une fois sur deux dans ce bin).

    Returns
    -------
    float
        ``∈ [0, 1]``.  ``0`` ↔ calibration parfaite.
    """
    bins = reliability_diagram(confidences, is_correct, n_bins=n_bins)
    gaps = [b.gap for b in bins if b.gap is not None]
    return max(gaps) if gaps else 0.0


# ──────────────────────────────────────────────────────────────────────────
# Vue agrégée
# ──────────────────────────────────────────────────────────────────────────


def compute_calibration_metrics(
    confidences: Iterable[float],
    is_correct: Iterable[bool | int],
    n_bins: int = 10,
) -> dict:
    """Calcule l'ensemble des métriques de calibration en un appel.

    Returns
    -------
    dict
        ``{
            "ece":   float,
            "mce":   float,
            "n_bins": int,
            "n_predictions": int,
            "overall_accuracy": float,
            "overall_confidence": float,
            "bins": [
                {"bin_low", "bin_high", "avg_confidence",
                 "accuracy", "count", "gap"},
                ...
            ],
        }``
    """
    confs = list(confidences)
    correct = list(is_correct)
    bins = reliability_diagram(confs, correct, n_bins=n_bins)
    total = sum(b.count for b in bins)
    overall_acc = (
        sum(int(bool(x)) for x in correct) / total if total > 0 else 0.0
    )
    overall_conf = (
        sum(float(c) for c in confs) / total if total > 0 else 0.0
    )

    ece = 0.0
    if total > 0:
        for b in bins:
            if b.gap is None:
                continue
            ece += (b.count / total) * b.gap
    mce = max((b.gap for b in bins if b.gap is not None), default=0.0)

    return {
        "ece": ece,
        "mce": mce,
        "n_bins": n_bins,
        "n_predictions": total,
        "overall_accuracy": overall_acc,
        "overall_confidence": overall_conf,
        "bins": [
            {
                "bin_low": b.bin_low,
                "bin_high": b.bin_high,
                "avg_confidence": b.avg_confidence,
                "accuracy": b.accuracy,
                "count": b.count,
                "gap": b.gap,
            }
            for b in bins
        ],
    }


__all__ = [
    "CalibrationBin",
    "reliability_diagram",
    "expected_calibration_error",
    "maximum_calibration_error",
    "compute_calibration_metrics",
]
