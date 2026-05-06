"""``picarones.core.metric_hooks`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation.metric_hooks`.  Phase 4-ter
du retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.evaluation.metric_hooks import *  # noqa: F401, F403
from picarones.evaluation.metric_hooks import (  # noqa: F401
    _CORPUS_AGGREGATORS,
    _DOCUMENT_HOOKS,
    _all_corpus_aggregator_names,
    _all_document_hook_names,
    _reset_for_tests,
)

warnings.warn(
    "picarones.core.metric_hooks is deprecated and will be removed in 2.0.  "
    "Import from picarones.evaluation.metric_hooks instead.",
    DeprecationWarning,
    stacklevel=2,
)
