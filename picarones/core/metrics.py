"""Calcul des métriques CER et WER via jiwer.

Métriques implémentées
----------------------
- CER brut                : distance d'édition caractère / longueur GT
- CER normalisé NFC       : après normalisation Unicode NFC
- CER sans casse          : insensible aux majuscules/minuscules
- CER diplomatique        : après application d'une table de correspondances
                            historiques (ſ=s, u=v, i=j…) — configurable
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


def compute_metrics(
    reference: str,
    hypothesis: str,
    normalization_profile: "Optional[NormalizationProfile]" = None,  # noqa: F821
    char_exclude: "Optional[frozenset]" = None,
) -> MetricsResult:
    """Calcule l'ensemble des métriques CER/WER pour une paire de textes.

    Parameters
    ----------
    reference:
        Texte de vérité terrain (ground truth).
    hypothesis:
        Texte produit par le moteur OCR.
    normalization_profile:
        Profil de normalisation diplomatique optionnel.
        Si fourni, calcule ``cer_diplomatic`` en plus des métriques standard.
        Si None, utilise le profil medieval_french par défaut.
    char_exclude:
        Ensemble de caractères à supprimer des deux textes avant tout calcul
        (CER, WER, MER, WIL). Appliqué également au CER diplomatique.

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
        # Exclusion de caractères avant tout calcul
        if char_exclude:
            reference  = "".join(c for c in reference  if c not in char_exclude)
            hypothesis = "".join(c for c in hypothesis if c not in char_exclude)

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

        # CER diplomatique — utilise le profil fourni ou le profil médiéval par défaut
        cer_diplomatic: Optional[float] = None
        diplomatic_profile_name: Optional[str] = None
        try:
            from picarones.core.normalization import DEFAULT_DIPLOMATIC_PROFILE
            profile = normalization_profile or DEFAULT_DIPLOMATIC_PROFILE
            ref_diplo = profile.normalize(reference)
            hyp_diplo = profile.normalize(hypothesis)
            cer_diplomatic = _cer_from_strings(ref_diplo, hyp_diplo)
            diplomatic_profile_name = profile.name
        except Exception:  # noqa: BLE001
            pass  # CER diplomatique non critique

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
            cer_diplomatic=cer_diplomatic,
            diplomatic_profile_name=diplomatic_profile_name,
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


# Import paresseux pour éviter les imports circulaires
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from picarones.core.normalization import NormalizationProfile
