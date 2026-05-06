"""``picarones.core.xml_utils`` — shim re-export (déprécié).

Le module canonique est :mod:`picarones.formats._xml_utils`,
ré-exporté publiquement sous :mod:`picarones.formats` (Phase 1 du
retrait du legacy, cf.
``docs/migration/legacy-retirement-plan.md``).

Suppression effective : version 2.0.

Migration ::

    # Avant
    from picarones.core.xml_utils import safe_parse_xml

    # Après
    from picarones.formats import safe_parse_xml
"""

from __future__ import annotations

import warnings

from picarones.formats._xml_utils import safe_parse_xml

warnings.warn(
    "picarones.core.xml_utils is deprecated and will be removed in "
    "2.0.  Import safe_parse_xml from picarones.formats instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["safe_parse_xml"]
