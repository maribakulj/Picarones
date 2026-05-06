"""``picarones.core.modules`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.domain.module_protocol` pour ``BaseModule``
et ``ExecutionMode`` ; :mod:`picarones.domain.artifacts` pour
``ArtifactType``.

Migration ::

    # Avant
    from picarones.core.modules import ArtifactType, BaseModule

    # Après
    from picarones.domain import ArtifactType
    from picarones.domain.module_protocol import BaseModule

L'``ArtifactType`` canonique a 10 valeurs (vs 6 legacy).  Les noms
legacy ``TEXT``/``ALTO``/``PAGE`` restent disponibles comme aliases
de ``RAW_TEXT``/``ALTO_XML``/``PAGE_XML`` ; le hook ``_missing_``
accepte aussi les valeurs string legacy (``"text"``, ``"alto"``,
``"page"``).  Les dicts indexés par ``ArtifactType.value``
(junction_metrics) sont enrichis automatiquement par
``expand_legacy_keys`` au moment de leur production pour conserver
les clés legacy en parallèle.
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
