"""Parsing XML sécurisé — anti-XXE / Billion Laughs / DTD retrieval.

Helper transverse appliqué partout où Picarones parse du XML reçu
depuis une source externe (corpus uploadé via le web, manifeste
Gallica, ALTO produit par un module ``BaseModule`` tiers, etc.).

Délègue à :mod:`defusedxml` (dépendance dure du projet) qui durcit
le parser stdlib contre :

- **XXE** (``XML External Entity``) — résolution d'entités vers
  des fichiers locaux ou des URL distantes.
- **Billion Laughs** — expansion exponentielle d'entités.
- **DTD retrieval** — fetch d'une DTD distante.

Discipline : tout module qui parse du XML doit utiliser
``safe_parse_xml`` plutôt que ``xml.etree.ElementTree.fromstring``
directement.

Module nommé avec un ``_`` initial : c'est un détail
d'implémentation du package ``formats`` ; les callers passent par
``picarones.formats.xml.safe_parse_xml`` (re-export public au
niveau du package).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

import defusedxml
import defusedxml.ElementTree as _SafeET


def safe_parse_xml(xml_bytes: bytes) -> Optional[ET.Element]:
    """Parse du XML en bloquant les entités externes.

    Retourne ``None`` si le payload n'est pas un XML valide ou si
    ``defusedxml`` détecte une attaque
    (``EntitiesForbidden``, ``ExternalReferenceForbidden``,
    ``DTDForbidden``, ``NotSupportedError``).
    """
    try:
        return _SafeET.fromstring(xml_bytes)
    except (ET.ParseError, defusedxml.DefusedXmlException):
        return None


__all__ = ["safe_parse_xml"]
