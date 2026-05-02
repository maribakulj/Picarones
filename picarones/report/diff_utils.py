"""Ré-export rétrocompat — la canonique est :mod:`picarones.core.diff_utils`.

Sprint A3 (item B-1 de l'audit institutional-readiness-2026-05) :
``compute_word_diff`` et consorts ont été déplacés dans Cercle 1 pour
respecter la règle de dépendance (Cercle 2 → Cercle 1 uniquement).

Ce module reste pour les consommateurs externes existants (scripts,
notebooks, plug-ins). Suppression planifiée v1.3.0.
"""

from __future__ import annotations

import warnings as _warnings

from picarones.core.diff_utils import (  # noqa: F401
    compute_char_diff,
    compute_word_diff,
    diff_stats,
)

_warnings.warn(
    "picarones.report.diff_utils est déprécié — utiliser "
    "picarones.core.diff_utils. Ce ré-export sera retiré en v1.3.0.",
    DeprecationWarning,
    stacklevel=2,
)
