"""``picarones.core.modules`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.domain.module_protocol` pour ``BaseModule``
et ``ExecutionMode`` ; :mod:`picarones.domain.artifacts` pour
``ArtifactType``.  Phase 4-bis du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.domain.artifacts import ArtifactType
from picarones.domain.module_protocol import BaseModule, ExecutionMode

warnings.warn(
    "picarones.core.modules is deprecated and will be removed in 2.0.  "
    "Import ArtifactType from picarones.domain and BaseModule from "
    "picarones.domain.module_protocol instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ArtifactType", "BaseModule", "ExecutionMode"]
