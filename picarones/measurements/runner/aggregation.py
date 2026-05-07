"""Délégations rétrocompat vers ``builtin_hooks._aggregate_*``.

Chantier 2 (post-Sprint 97) : la logique d'agrégation par-engine de
toutes les métriques (confusion, taxonomy, structure, image_quality,
line_metrics, hallucination, calibration, char_scores) vit désormais
dans :mod:`picarones.measurements.builtin_hooks` (single source of truth,
exposé via le registre :mod:`picarones.evaluation.metric_hooks`).

Les noms ci-dessous restent disponibles depuis
``picarones.measurements.runner`` pour la rétrocompat des tests
Sprint 13 / 42 qui les importent directement.
"""

from __future__ import annotations

from typing import Optional


def _aggregate_confusion(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_confusion`."""
    from picarones.measurements.builtin_hooks import _aggregate_confusion as _impl
    return _impl(doc_results)


def _aggregate_char_scores(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_char_scores`."""
    from picarones.measurements.builtin_hooks import _aggregate_char_scores as _impl
    return _impl(doc_results)


def _aggregate_taxonomy(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_taxonomy`."""
    from picarones.measurements.builtin_hooks import _aggregate_taxonomy as _impl
    return _impl(doc_results)


def _aggregate_structure(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_structure`."""
    from picarones.measurements.builtin_hooks import _aggregate_structure as _impl
    return _impl(doc_results)


def _aggregate_image_quality(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_image_quality`."""
    from picarones.measurements.builtin_hooks import _aggregate_image_quality as _impl
    return _impl(doc_results)


def _aggregate_line_metrics(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_line_metrics`."""
    from picarones.measurements.builtin_hooks import _aggregate_line_metrics as _impl
    return _impl(doc_results)


def _aggregate_hallucination(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_hallucination`."""
    from picarones.measurements.builtin_hooks import _aggregate_hallucination as _impl
    return _impl(doc_results)


def _aggregate_calibration(doc_results: list) -> Optional[dict]:
    """Délégation vers :func:`builtin_hooks._aggregate_calibration`.

    Conservé pour la rétrocompat du test ``test_sprint42_calibration_runner``
    qui importe directement depuis ``picarones.measurements.runner``. La
    logique réelle vit dans :mod:`picarones.measurements.builtin_hooks`
    (chantier 2 post-Sprint 97).
    """
    from picarones.measurements.builtin_hooks import _aggregate_calibration as _impl
    return _impl(doc_results)


__all__ = [
    "_aggregate_calibration",
    "_aggregate_char_scores",
    "_aggregate_confusion",
    "_aggregate_hallucination",
    "_aggregate_image_quality",
    "_aggregate_line_metrics",
    "_aggregate_structure",
    "_aggregate_taxonomy",
]
