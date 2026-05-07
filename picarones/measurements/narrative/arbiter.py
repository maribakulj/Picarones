"""``picarones.measurements.narrative.arbiter`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.reports_v2.narrative.arbiter`.
"""

from __future__ import annotations

import warnings

from picarones.reports_v2.narrative.arbiter import (  # noqa: F401
    DEFAULT_TYPE_ORDER,
    select_facts,
    _compute_default_type_order,
    _FALLBACK_TYPE_ORDER,
    _TYPE_ORDER,
    _TYPE_INDEX,
    _COMPLEMENTARY_PAIRS,
    _sort_key,
    _is_redundant,
    _remove_contradictions,
)

warnings.warn(
    "picarones.measurements.narrative.arbiter is deprecated and will be removed in 2.0.  "
    "Import from picarones.reports_v2.narrative.arbiter instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['DEFAULT_TYPE_ORDER', 'select_facts']
