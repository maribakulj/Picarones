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

from picarones.evaluation.views.base import EvaluationViewExecutor, ViewResult
from picarones.evaluation.views.executor import (
    DefaultEvaluationViewExecutor,
    PayloadLoader,
)

__all__ = [
    "EvaluationViewExecutor",
    "ViewResult",
    "DefaultEvaluationViewExecutor",
    "PayloadLoader",
]
