"""``picarones.core.diff_utils`` — shim re-export (déprécié).

Le module canonique est :mod:`picarones.evaluation._diff_utils`,
ré-exporté publiquement sous :mod:`picarones.evaluation` (Phase 1
du retrait du legacy, cf.
``docs/migration/legacy-retirement-plan.md``).

Suppression effective : version 2.0.

Migration ::

    # Avant
    from picarones.core.diff_utils import compute_word_diff, compute_char_diff

    # Après
    from picarones.evaluation import compute_word_diff, compute_char_diff
"""

from __future__ import annotations

import warnings

from picarones.evaluation._diff_utils import (
    compute_char_diff,
    compute_word_diff,
    diff_stats,
)

warnings.warn(
    "picarones.core.diff_utils is deprecated and will be removed in "
    "2.0.  Import {compute_word_diff, compute_char_diff, diff_stats} "
    "from picarones.evaluation instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["compute_word_diff", "compute_char_diff", "diff_stats"]
