"""``PipelineStep`` et ``PipelineSpec`` — re-export depuis ``domain`` (déprécié).

Sprint A14-S40 a migré le module canonique vers
``picarones.domain.pipeline_spec`` (cercle 1, types purs).  Ce
module reste un alias de chemin pour ne pas casser les callers
existants — ce n'est pas un shim au sens architectural
(adaptation d'une API incompatible) mais une convenance de chemin.

Sprint A14-S57 (audit #26) : émission d'un ``DeprecationWarning``
à l'import de ce module pour signaler aux callers que le chemin
canonique est ``picarones.domain``.  Le module sera supprimé au
sprint S60.

Les nouveaux callers doivent importer directement depuis
``picarones.domain`` :

::

    from picarones.domain import PipelineSpec, PipelineStep, INITIAL_STEP_ID
"""

from __future__ import annotations

import warnings

from picarones.domain.pipeline_spec import (
    INITIAL_STEP_ID,
    PipelineSpec,
    PipelineStep,
)

warnings.warn(
    "picarones.pipeline.spec is deprecated since S57; "
    "import from picarones.domain instead "
    "(`from picarones.domain import PipelineSpec, PipelineStep, "
    "INITIAL_STEP_ID`).  This re-export will be removed in S60.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PipelineStep", "PipelineSpec", "INITIAL_STEP_ID"]
