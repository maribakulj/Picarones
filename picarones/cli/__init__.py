"""Shim de compatibilité — CLI legacy.

Sprint G du plan v2.0 (mai 2026) — la CLI legacy a été déplacée
vers :mod:`picarones.interfaces.cli._legacy`.  Ce package racine
``picarones.cli`` reste exécutable (rétrocompat des imports
``from picarones.cli import cli``) et sera supprimé en 2.0.

Le script ``picarones`` installé via ``pyproject.toml`` pointe
désormais directement sur ``picarones.interfaces.cli._legacy:cli``
— ce shim n'est utilisé que par les imports Python des appelants
qui font ``from picarones.cli import cli`` (notamment les tests
Sprint 8 et autres).
"""

from __future__ import annotations

import warnings

warnings.warn(
    "picarones.cli est obsolète et sera supprimé en 2.0.  "
    "Utiliser picarones.interfaces.cli._legacy à la place.",
    DeprecationWarning,
    stacklevel=2,
)

from picarones.interfaces.cli._legacy import (  # noqa: E402
    cli,
    _engine_from_name,
    _setup_logging,
)

__all__ = ["cli", "_engine_from_name", "_setup_logging"]
