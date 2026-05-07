"""``picarones.report.diff_utils`` — shim re-export (déprécié, suppression 2.0).

Canonique : :mod:`picarones.evaluation` (qui ré-exporte depuis
``picarones.evaluation._diff_utils``).  Phase 1 puis Phase 5 du
retrait du legacy.
"""

from __future__ import annotations

import warnings

from picarones.evaluation._diff_utils import (  # noqa: F401
    compute_char_diff,
    compute_word_diff,
    diff_stats,
)

warnings.warn(
    "picarones.report.diff_utils is deprecated and will be removed in 2.0.  "
    "Import {compute_word_diff, compute_char_diff, diff_stats} from "
    "picarones.evaluation instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["compute_word_diff", "compute_char_diff", "diff_stats"]
