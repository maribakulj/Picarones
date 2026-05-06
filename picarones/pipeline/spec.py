"""``picarones.pipeline.spec`` — shim de compatibilité descendante (déprécié).

Le module canonique est ``picarones.domain.pipeline_spec`` depuis le
sprint S40.  Ce module a été supprimé temporairement au S57 puis
restauré au S59 avec ``DeprecationWarning`` pour respecter une
deprecation period propre vis-à-vis des callers externes (espaces
HuggingFace tiers, scripts archivistiques, notebooks de chercheurs).

Suppression effective prévue en version majeure suivante (1.x → 2.0).

::

    # Migration : remplacer
    from picarones.pipeline.spec import PipelineSpec
    # par
    from picarones.domain import PipelineSpec
"""

from __future__ import annotations

import warnings

from picarones.domain.pipeline_spec import (
    INITIAL_STEP_ID,
    PipelineSpec,
    PipelineStep,
)

warnings.warn(
    "picarones.pipeline.spec is deprecated and will be removed in 2.0. "
    "Import from picarones.domain instead "
    "(`from picarones.domain import PipelineSpec, PipelineStep, "
    "INITIAL_STEP_ID`).",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["INITIAL_STEP_ID", "PipelineSpec", "PipelineStep"]
