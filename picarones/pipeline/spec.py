"""``PipelineStep`` et ``PipelineSpec`` — re-export depuis ``domain``.

Sprint A14-S40 a migré le module canonique vers
``picarones.domain.pipeline_spec`` (cercle 1, types purs).  Ce
module reste un alias de chemin pour ne pas casser les callers
existants — ce n'est pas un shim au sens architectural
(adaptation d'une API incompatible) mais une convenance de chemin.

Les nouveaux callers doivent importer directement depuis
``picarones.domain`` :

::

    from picarones.domain import PipelineSpec, PipelineStep, INITIAL_STEP_ID
"""

from __future__ import annotations

from picarones.domain.pipeline_spec import (
    INITIAL_STEP_ID,
    PipelineSpec,
    PipelineStep,
)

__all__ = ["PipelineStep", "PipelineSpec", "INITIAL_STEP_ID"]
