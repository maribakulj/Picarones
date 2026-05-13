"""Shim de re-export — ``RunResult`` vit désormais dans ``pipeline/``.

Phase 5.1 audit code-quality (2026-05) : ``RunResult``,
``RunDocumentResult`` et ``ReportRenderer`` ont été déplacés vers
``picarones.pipeline.run_result``.  Raison : la couche ``reports/``
(couche 7) consomme ces types ; le manifeste interdit
``reports/ → app/`` (couche 7 → 6 est anti-orientation).  En les
plaçant en couche 4 (``pipeline/``), ``reports/`` peut les importer
légalement.

Ce module reste comme **shim de compatibilité interne** pour les
callers historiques côté ``app/services/``.  Toute nouvelle code
doit importer directement depuis ``picarones.pipeline.run_result``.
"""

from __future__ import annotations

from picarones.pipeline.run_result import (
    ReportRenderer,
    RunDocumentResult,
    RunResult,
)

__all__ = ["ReportRenderer", "RunDocumentResult", "RunResult"]
