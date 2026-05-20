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
        # A.I.0 P0 : ne pas retourner 0.0 en erreur
        # (indistinguable d'un score parfait pour un lecteur qui ne
        # vérifie pas ``error``).  None = absence de mesure.
        return MetricsResult(
            cer=None, cer_nfc=None, cer_caseless=None,
            wer=None, wer_normalized=None, mer=None, wil=None,
            reference_length=len(reference),
            hypothesis_length=len(hypothesis),
            error="jiwer n'est pas installé (pip install jiwer)",
        )

    # Audit scientifique (F10) — l'exclusion de caractères est appliquée
    # **avant** le court-circuit des cas vides : si ``char_exclude`` vide
    # entièrement un texte, le cas est traité par les conventions
    # "texte vide" ci-dessous (résultat déterministe) plutôt que de
    # tomber dans le ``except`` et de renvoyer une erreur / des None.
    if char_exclude:
        reference  = "".join(c for c in reference  if c not in char_exclude)
        hypothesis = "".join(c for c in hypothesis if c not in char_exclude)

    # Cas dégénérés des inputs vides — jiwer 3.x lève sur ces cas
    # (4.x les gère mais on ne dépend plus d'une majeure spécifique).
    # Convention :
    # - vide vs vide → 0.0 (rien à corriger, score parfait par défaut).
    # - vide ref vs hyp non vide → 1.0 (toute l'hypothèse est une
    #   insertion, error rate = 1.0).
    # - ref non vide vs hyp vide → 1.0 (toute la GT manque).
    # Dans ces trois cas, les comptes bruts (cer_errors/cer_ref_chars…)
    # restent ``None`` : le dénominateur micro n'est pas défini sur une
    # référence vide, l'agrégateur micro saute donc le document.
    ref_stripped = reference.strip()
    hyp_stripped = hypothesis.strip() if hypothesis else ""
    if not ref_stripped and not hyp_stripped:
        return MetricsResult(
            cer=0.0, cer_nfc=0.0, cer_caseless=0.0,
            wer=0.0, wer_normalized=0.0, mer=0.0, wil=0.0,
            reference_length=len(reference),
            hypothesis_length=len(hypothesis),
        )
    if not ref_stripped and hyp_stripped:
        return MetricsResult(
            cer=1.0, cer_nfc=1.0, cer_caseless=1.0,
            wer=1.0, wer_normalized=1.0, mer=1.0, wil=1.0,
            reference_length=len(reference),
            hypothesis_length=len(hypothesis),
        )
    if ref_stripped and not hyp_stripped:
        return MetricsResult(
            cer=1.0, cer_nfc=1.0, cer_caseless=1.0,
            wer=1.0, wer_normalized=1.0, mer=1.0, wil=1.0,
            reference_length=len(reference),
            hypothesis_length=0,
        )

    try:
        # CER : un seul appel ``process_characters`` fournit la valeur
        # (``co.cer`` est bit-identique à ``jiwer.cer``) ET les comptes
        # de l'alignement minimal (= Levenshtein) nécessaires au
        # micro-CER corpus (audit scientifique F1).
        co = jiwer.process_characters(reference, hypothesis)
        cer_raw = co.cer
        cer_errors = co.substitutions + co.deletions + co.insertions
        cer_ref_chars = co.substitutions + co.deletions + co.hits

        cer_nfc = _cer_from_strings(
            _normalize_nfc(reference), _normalize_nfc(hypothesis)
        )
        cer_caseless = _cer_from_strings(
            _normalize_caseless(reference), _normalize_caseless(hypothesis)
        )

        # WER : idem via ``process_words`` (``wo.wer/mer/wil`` identiques
        # aux fonctions jiwer, même tokenisation par espaces).
        ref_norm = _normalize_whitespace(reference)
        hyp_norm = _normalize_whitespace(hypothesis)

        wo = jiwer.process_words(reference, hypothesis)
        wer_raw = wo.wer
        wer_errors = wo.substitutions + wo.deletions + wo.insertions
        wer_ref_words = wo.substitutions + wo.deletions + wo.hits
        wer_normalized = jiwer.wer(ref_norm, hyp_norm)
        mer = wo.mer
        wil = wo.wil

        # CER diplomatique — le profil peut arriver soit comme objet
        # NormalizationProfile, soit comme **nom** (str).  Le runner et
        # l'interface web sérialisent le profil en nom dans
        # ``RunSpec.normalization_profile`` (str) puis le passent tel
        # quel jusqu'ici : il faut donc le **résoudre** en
        # NormalizationProfile.  Sans cette résolution, ``profile`` est
        # un str, ``profile.normalize(...)`` lève « 'str' object has no
        # attribute 'normalize' », l'``except`` l'avale en warning et le
        # CER diplomatique est silencieusement None pour TOUT le corpus
        # web (donnée scientifique manquante, non signalée à l'usager).
        cer_diplomatic: Optional[float] = None
        diplomatic_profile_name: Optional[str] = None
        try:
            from picarones.evaluation.metrics.normalization import (
                DEFAULT_DIPLOMATIC_PROFILE,
                get_builtin_profile,
            )
            profile = normalization_profile
            if profile is None or profile == "":
                profile = DEFAULT_DIPLOMATIC_PROFILE
            elif isinstance(profile, str):
                try:
                    profile = get_builtin_profile(profile)
                except KeyError:
                    logger.warning(
                        "[metrics] profil de normalisation inconnu "
                        "'%s' — repli sur le profil médiéval par défaut",
                        profile,
                    )
                    profile = DEFAULT_DIPLOMATIC_PROFILE
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
            cer_errors=cer_errors,
            cer_ref_chars=cer_ref_chars,
            wer_errors=wer_errors,
            wer_ref_words=wer_ref_words,
            cer_diplomatic=cer_diplomatic,
            diplomatic_profile_name=diplomatic_profile_name,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("[metrics] calcul métriques échoué : %s", exc)
        # A.I.0 P0 : None plutôt que 0.0 (cf. cas
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
