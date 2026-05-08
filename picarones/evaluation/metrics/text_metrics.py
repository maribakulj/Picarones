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

Modèle de données
-----------------
``MetricsResult`` (dataclass pure) et ``aggregate_metrics`` (stats
moyenne/médiane via ``statistics`` stdlib) vivent en couche 3 dans
:mod:`picarones.evaluation.metric_result`. Ils sont ré-exportés ici pour la
commodité — un module qui consomme déjà ``compute_metrics`` n'a
qu'à en faire ``from picarones.evaluation.metrics.text_metrics import …``.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Optional

from picarones.evaluation.metric_result import MetricsResult, aggregate_metrics

logger = logging.getLogger(__name__)

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
        # Sprint A14-S1 — A.I.0 P0 : ne pas retourner 0.0 en erreur
        # (indistinguable d'un score parfait pour un lecteur qui ne
        # vérifie pas ``error``).  None = absence de mesure.
        return MetricsResult(
            cer=None, cer_nfc=None, cer_caseless=None,
            wer=None, wer_normalized=None, mer=None, wil=None,
            reference_length=len(reference),
            hypothesis_length=len(hypothesis),
            error="jiwer n'est pas installé (pip install jiwer)",
        )

    # Hypothèse vide avec référence non vide = erreur totale (toutes les
    # métriques jiwer lèvent une ZeroDivisionError sur hypothèse vide).
    ref_stripped = reference.strip()
    hyp_stripped = hypothesis.strip() if hypothesis else ""
    if ref_stripped and not hyp_stripped:
        return MetricsResult(
            cer=1.0, cer_nfc=1.0, cer_caseless=1.0,
            wer=1.0, wer_normalized=1.0, mer=1.0, wil=1.0,
            reference_length=len(reference),
            hypothesis_length=0,
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
            from picarones.evaluation.metrics.normalization import DEFAULT_DIPLOMATIC_PROFILE
            profile = normalization_profile or DEFAULT_DIPLOMATIC_PROFILE
            ref_diplo = profile.normalize(reference)
            hyp_diplo = profile.normalize(hypothesis)
            cer_diplomatic = _cer_from_strings(ref_diplo, hyp_diplo)
            diplomatic_profile_name = profile.name
        except Exception as e:  # noqa: BLE001
            logger.warning("[metrics] CER diplomatique dégradé : %s", e)

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
        logger.warning("[metrics] calcul métriques échoué : %s", exc)
        # Sprint A14-S1 — A.I.0 P0 : None plutôt que 0.0 (cf. cas
        # ``not _JIWER_AVAILABLE`` plus haut pour le rationale).
        return MetricsResult(
            cer=None, cer_nfc=None, cer_caseless=None,
            wer=None, wer_normalized=None, mer=None, wil=None,
            reference_length=len(reference),
            hypothesis_length=len(hypothesis),
            error=str(exc),
        )


__all__ = ["MetricsResult", "aggregate_metrics", "compute_metrics"]


# Import paresseux pour éviter les imports circulaires
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from picarones.evaluation.metrics.normalization import NormalizationProfile
