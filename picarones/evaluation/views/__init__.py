"""Vues d'évaluation — Sprints S13-S16.

Une vue d'évaluation répond à une question précise : "lequel des
pipelines disponibles produit la meilleure sortie selon cet angle ?"

Vues canoniques cibles (rewrite ciblé) :

- ``TextView`` (S14) — qualité textuelle finale.  Accepte RAW_TEXT,
  CORRECTED_TEXT, ALTO_XML, PAGE_XML, CANONICAL_DOCUMENT, projette
  tout vers texte brut.  Métriques : CER, WER, insertions, omissions.
- ``AltoView`` (S15) — fidélité documentaire.  Exige ALTO_XML.
  Métriques : validité, alignement lignes/mots, ordre de lecture.
- ``SearchView`` (S16) — recherchabilité plein-texte.  Métriques :
  rappel fuzzy, séquences numériques préservées, noms propres
  retrouvés.

Reporté post-livraison : ``LayoutView``, ``HallucinationView``,
``CostView``, ``PhilologicalView``, ``ProductionView``.
"""

from __future__ import annotations

from picarones.evaluation.views.alto_view import (
    DEFAULT_ALTO_CANDIDATE_TYPES,
    DEFAULT_ALTO_IGNORED_DIMENSIONS,
    DEFAULT_ALTO_METRICS,
    DEFAULT_ALTO_WARNINGS,
    build_alto_view,
)
from picarones.evaluation.views.base import EvaluationViewExecutor, ViewResult
from picarones.evaluation.views.executor import (
    DefaultEvaluationViewExecutor,
    PayloadLoader,
)
from picarones.evaluation.views.text_view import (
    DEFAULT_TEXT_CANDIDATE_TYPES,
    DEFAULT_TEXT_IGNORED_DIMENSIONS,
    DEFAULT_TEXT_METRICS,
    DEFAULT_TEXT_PROJECTIONS,
    DEFAULT_TEXT_WARNINGS,
    build_text_view,
)

__all__ = [
    # Protocol + result
    "EvaluationViewExecutor",
    "ViewResult",
    # Executor
    "DefaultEvaluationViewExecutor",
    "PayloadLoader",
    # TextView (S14)
    "build_text_view",
    "DEFAULT_TEXT_METRICS",
    "DEFAULT_TEXT_CANDIDATE_TYPES",
    "DEFAULT_TEXT_PROJECTIONS",
    "DEFAULT_TEXT_IGNORED_DIMENSIONS",
    "DEFAULT_TEXT_WARNINGS",
    # AltoView (S15)
    "build_alto_view",
    "DEFAULT_ALTO_METRICS",
    "DEFAULT_ALTO_CANDIDATE_TYPES",
    "DEFAULT_ALTO_IGNORED_DIMENSIONS",
    "DEFAULT_ALTO_WARNINGS",
]
