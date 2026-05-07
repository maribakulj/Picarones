"""Modules officiels Picarones (legacy ``BaseModule``-based).

Phase 7.A.4 — package relocalisé depuis ``picarones.modules`` vers
``picarones.adapters.legacy_modules``.  Le chemin legacy reste
disponible via un shim avec ``DeprecationWarning`` ; suppression
prévue en 2.0.

Pendant un module officiel pré-rewrite : extension de
``BaseModule`` (Sprint 33) avec contrat
``process(inputs) → outputs``.  La convergence vers le contrat
canonique ``StepExecutor`` (``execute(inputs, params, context)``)
est documentée dans
``docs/migration/pipeline-convergence-plan.md`` (sub-phases
7.A-7.D, stratégie 4.B).

Modules disponibles
-------------------
- ``alto_text_to_mono_region.TextToAltoMonoRegion`` — baseline
  reconstructeur ``RAW_TEXT → ALTO_XML`` mono-région (310 LOC).
"""

from __future__ import annotations

from picarones.adapters.legacy_modules.alto_text_to_mono_region import (
    TextToAltoMonoRegion,
)

__all__ = ["TextToAltoMonoRegion"]
