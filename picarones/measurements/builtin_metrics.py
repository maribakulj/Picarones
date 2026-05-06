"""Métriques natives enregistrées dans le registre typé (Sprint 34).

Ce module est un démonstrateur d'enregistrement : il expose les
métriques scalaires existantes (CER, WER, MER, WIL) sous une forme
unitaire dans le registre, plus un stub typé hétérogène pour les
jonctions ``(TEXT, ALTO)``.

L'import du module suffit à peupler le registre — le décorateur
``@register_metric`` s'exécute à l'import.  Les sprints suivants (axe A
du plan d'évolution) ajouteront ici les métriques structurelles
(``reading_order_f1``, ``layout_f1``), philologiques (``unicode_block_*``,
``mufi_coverage``), et de fiabilité (``ece``, ``mce``).

Important — pas de double calcul
-------------------------------
Ces wrappers ne **remplacent pas** ``compute_metrics`` du module
``metrics.py``.  Ils existent pour les nouveaux chemins (pipelines
composées qui calculent par jonction).  Le rapport HTML existant
continue à passer par ``compute_metrics`` et reste donc strictement
identique octet par octet (critère de la Phase 0.3).
"""

from __future__ import annotations

import logging

from picarones.core.metric_registry import register_metric
from picarones.domain.artifacts import ArtifactType

logger = logging.getLogger(__name__)


try:
    import jiwer
    _JIWER_AVAILABLE = True
except ImportError:
    _JIWER_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────
# Métriques scalaires (TEXT, TEXT) — wrappers fins autour de jiwer
# ──────────────────────────────────────────────────────────────────────────


def _safe_jiwer_call(fn, reference: str, hypothesis: str) -> float:
    """Wrapper qui gère les cas dégénérés (références ou hypothèses vides)."""
    if not _JIWER_AVAILABLE:
        raise RuntimeError(
            "jiwer n'est pas installé — installer avec `pip install jiwer`"
        )
    if not reference:
        return 0.0 if not hypothesis else 1.0
    if not hypothesis:
        return 1.0
    return fn(reference, hypothesis)


@register_metric(
    name="cer",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description="Character Error Rate (distance d'édition normalisée par la longueur de la GT).",
    higher_is_better=False,
    tags={"text", "edit_distance", "error_rate"},
)
def cer(reference: str, hypothesis: str) -> float:
    """CER brut sur les caractères, via jiwer."""
    return _safe_jiwer_call(jiwer.cer, reference, hypothesis)


@register_metric(
    name="wer",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description="Word Error Rate.",
    higher_is_better=False,
    tags={"text", "edit_distance", "error_rate"},
)
def wer(reference: str, hypothesis: str) -> float:
    """WER brut, via jiwer."""
    return _safe_jiwer_call(jiwer.wer, reference, hypothesis)


@register_metric(
    name="mer",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description="Match Error Rate (jiwer).",
    higher_is_better=False,
    tags={"text", "error_rate"},
)
def mer(reference: str, hypothesis: str) -> float:
    return _safe_jiwer_call(jiwer.mer, reference, hypothesis)


@register_metric(
    name="wil",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description="Word Information Lost (jiwer).",
    higher_is_better=False,
    tags={"text", "error_rate"},
)
def wil(reference: str, hypothesis: str) -> float:
    return _safe_jiwer_call(jiwer.wil, reference, hypothesis)


# ──────────────────────────────────────────────────────────────────────────
# Métrique typée hétérogène (TEXT, ALTO) — stub démonstrateur
# ──────────────────────────────────────────────────────────────────────────


@register_metric(
    name="text_preservation_after_reconstruction",
    input_types=(ArtifactType.TEXT, ArtifactType.ALTO),
    description=(
        "Taux de tokens de la GT texte présents dans le texte extrait de "
        "l'ALTO produit (preuve de concept ; remplaçable par une mesure "
        "alignée par les sprints futurs)."
    ),
    higher_is_better=True,
    tags={"structure", "preservation", "stub"},
)
def text_preservation_after_reconstruction(
    reference_text: str,
    hypothesis_alto: str,
) -> float:
    """Stub démonstrateur d'une jonction texte → ALTO.

    Sprints à venir (axe A du plan d'évolution) remplaceront cette
    implémentation par une vraie mesure de préservation : extraction
    structurée du texte ALTO via le parser dédié, alignement, calcul
    déterministe.  Pour l'instant la mesure est volontairement simple
    pour démontrer le mécanisme.

    Parameters
    ----------
    reference_text:
        Texte GT (niveau ``GTLevel.TEXT``).
    hypothesis_alto:
        ALTO XML brut produit par un module de reconstruction (niveau
        ``ArtifactType.ALTO``).

    Returns
    -------
    float
        Taux de tokens uniques de ``reference_text`` apparaissant dans
        ``hypothesis_alto`` (case-insensitive).  ``1.0`` = tous les
        tokens préservés.
    """
    if not reference_text:
        return 1.0
    ref_tokens = {tok.lower() for tok in reference_text.split() if tok}
    if not ref_tokens:
        return 1.0
    alto_text = hypothesis_alto.lower()
    preserved = sum(1 for tok in ref_tokens if tok in alto_text)
    return preserved / len(ref_tokens)


__all__ = [
    "cer",
    "wer",
    "mer",
    "wil",
    "text_preservation_after_reconstruction",
]
