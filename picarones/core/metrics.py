"""Calcul des métriques CER et WER via jiwer.

Métriques implémentées
----------------------
- CER brut                : distance d'édition caractère / longueur GT
- CER normalisé NFC       : après normalisation Unicode NFC
- CER sans casse          : insensible aux majuscules/minuscules
- WER brut                : word error rate standard
- WER normalisé           : après normalisation des espaces
- MER                     : Match Error Rate (jiwer)
- WIL                     : Word Information Lost (jiwer)
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Optional

try:
    import jiwer

    _JIWER_AVAILABLE = True
except ImportError:
    _JIWER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Transformations / normalisations
# ---------------------------------------------------------------------------

def _normalize_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _normalize_caseless(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


# Transformations jiwer pour le CER (chaque char devient un "mot")
_CHAR_TRANSFORM = jiwer.transforms.Compose([]) if _JIWER_AVAILABLE else None

# Transformations jiwer pour le WER (normalisation légère des espaces)
_WER_TRANSFORM = (
    jiwer.transforms.Compose(
        [
            jiwer.transforms.RemoveMultipleSpaces(),
            jiwer.transforms.Strip(),
            jiwer.transforms.ReduceToListOfListOfWords(),
        ]
    )
    if _JIWER_AVAILABLE
    else None
)


def _cer_from_strings(reference: str, hypothesis: str) -> float:
    """CER brut : distance d'édition sur les caractères."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    # jiwer.cer traite chaque caractère comme un token
    return jiwer.cer(reference, hypothesis)


# ---------------------------------------------------------------------------
# Résultat structuré
# ---------------------------------------------------------------------------

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

    def as_dict(self) -> dict:
        return {
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

    @property
    def cer_percent(self) -> float:
        return round(self.cer * 100, 2)

    @property
    def wer_percent(self) -> float:
        return round(self.wer * 100, 2)


def compute_metrics(reference: str, hypothesis: str) -> MetricsResult:
    """Calcule l'ensemble des métriques CER/WER pour une paire de textes.

    Parameters
    ----------
    reference:
        Texte de vérité terrain (ground truth).
    hypothesis:
        Texte produit par le moteur OCR.

    Returns
    -------
    MetricsResult
        Objet contenant toutes les métriques calculées.
    """
    if not _JIWER_AVAILABLE:
        return MetricsResult(
            cer=0.0, cer_nfc=0.0, cer_caseless=0.0,
            wer=0.0, wer_normalized=0.0, mer=0.0, wil=0.0,
            reference_length=len(reference),
            hypothesis_length=len(hypothesis),
            error="jiwer n'est pas installé (pip install jiwer)",
        )

    try:
        # CER variants
        cer_raw = _cer_from_strings(reference, hypothesis)
        cer_nfc = _cer_from_strings(
            _normalize_nfc(reference), _normalize_nfc(hypothesis)
        )
        cer_caseless = _cer_from_strings(
            _normalize_caseless(reference), _normalize_caseless(hypothesis)
        )

        # WER variants
        ref_norm = _normalize_whitespace(reference)
        hyp_norm = _normalize_whitespace(hypothesis)

        wer_raw = jiwer.wer(reference, hypothesis)
        wer_normalized = jiwer.wer(ref_norm, hyp_norm)
        mer = jiwer.mer(reference, hypothesis)
        wil = jiwer.wil(reference, hypothesis)

        return MetricsResult(
            cer=cer_raw,
            cer_nfc=cer_nfc,
            cer_caseless=cer_caseless,
            wer=wer_raw,
            wer_normalized=wer_normalized,
            mer=mer,
            wil=wil,
            reference_length=len(reference),
            hypothesis_length=len(hypothesis),
        )

    except Exception as exc:  # noqa: BLE001
        return MetricsResult(
            cer=0.0, cer_nfc=0.0, cer_caseless=0.0,
            wer=0.0, wer_normalized=0.0, mer=0.0, wil=0.0,
            reference_length=len(reference),
            hypothesis_length=len(hypothesis),
            error=str(exc),
        )


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
    import statistics

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

    aggregated["document_count"] = len(results)
    aggregated["failed_count"] = sum(1 for r in results if r.error is not None)

    return aggregated
