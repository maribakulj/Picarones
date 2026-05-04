"""Sprint A14-S9 — ALTO parser, writer, projector.

Tests minimaux mais couvrant les invariants critiques :

- Round-trip ``parse → write → parse`` préserve la structure.
- Détection auto v2 / v3 / v4 / sans namespace.
- Extraction texte respecte ``Page → Block → Line → String``.
- Césure ``HypPart1`` / ``HypPart2`` (même ligne ET cross-ligne).
- ``defusedxml`` bloque les attaques XXE.
"""

from __future__ import annotations

import pytest

from picarones.domain import Artifact, ArtifactType
from picarones.domain.errors import ProjectionError
from picarones.formats.alto import (
    AltoBBox,
    AltoDocument,
    AltoLine,
    AltoPage,
    AltoParseError,
    AltoString,
    AltoTextBlock,
    AltoToText,
    alto_document_to_text,
    parse_alto,
    write_alto,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures synthétiques
# ──────────────────────────────────────────────────────────────────────


def _simple_doc() -> AltoDocument:
    return AltoDocument(
        pages=(AltoPage(
            id="p1", width=1000, height=1500,
            blocks=(AltoTextBlock(
                id="b1",
                lines=(
                    AltoLine(id="l1", strings=(
                        AltoString(content="Hello", id="s1"),
                        AltoString(content="world", id="s2"),
                    )),
                    AltoLine(id="l2", strings=(
                        AltoString(content="second", id="s3"),
                        AltoString(content="line", id="s4"),
                    )),
                ),
            ),),
        ),),
    )


# ──────────────────────────────────────────────────────────────────────
# Parser — détection de namespaces
# ──────────────────────────────────────────────────────────────────────


class TestParserVersions:
    def test_v4_namespace_detected(self) -> None:
        xml = b'''<?xml version="1.0"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Layout><Page ID="p" WIDTH="100" HEIGHT="200">
    <PrintSpace>
      <TextBlock ID="b">
        <TextLine ID="l">
          <String CONTENT="hi"/>
        </TextLine>
      </TextBlock>
    </PrintSpace>
  </Page></Layout>
</alto>'''
        doc = parse_alto(xml)
        assert doc.source_version == "v4"
        assert len(doc.pages) == 1

    def test_v3_namespace_detected(self) -> None:
        xml = b'''<?xml version="1.0"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
  <Layout><Page ID="p"><PrintSpace>
    <TextBlock><TextLine><String CONTENT="x"/></TextLine></TextBlock>
  </PrintSpace></Page></Layout>
</alto>'''
        doc = parse_alto(xml)
        assert doc.source_version == "v3"

    def test_v2_namespace_detected(self) -> None:
        xml = b'''<?xml version="1.0"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Layout><Page><PrintSpace>
    <TextBlock><TextLine><String CONTENT="x"/></TextLine></TextBlock>
  </PrintSpace></Page></Layout>
</alto>'''
        doc = parse_alto(xml)
        assert doc.source_version == "v2"

    def test_no_namespace_accepted(self) -> None:
        xml = b'''<?xml version="1.0"?>
<alto>
  <Layout><Page><PrintSpace>
    <TextBlock><TextLine><String CONTENT="x"/></TextLine></TextBlock>
  </PrintSpace></Page></Layout>
</alto>'''
        doc = parse_alto(xml)
        assert doc.source_version == "none"

    def test_invalid_xml_raises(self) -> None:
        with pytest.raises(AltoParseError, match="invalide"):
            parse_alto(b"<not closed")

    def test_empty_xml_raises(self) -> None:
        with pytest.raises(AltoParseError, match="vide"):
            parse_alto(b"")

    def test_xxe_blocked(self) -> None:
        """defusedxml doit bloquer les attaques XXE."""
        xml = b'''<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<alto>&xxe;</alto>'''
        with pytest.raises(AltoParseError):
            parse_alto(xml)


# ──────────────────────────────────────────────────────────────────────
# Round-trip writer/parser
# ──────────────────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_simple_doc_roundtrip(self) -> None:
        doc = _simple_doc()
        xml = write_alto(doc)
        doc2 = parse_alto(xml)
        # Les structures internes sont équivalentes (sans
        # tenir compte de source_version qui peut différer).
        assert len(doc2.pages) == len(doc.pages)
        assert len(doc2.pages[0].blocks) == len(doc.pages[0].blocks)
        assert doc2.pages[0].width == doc.pages[0].width
        assert doc2.pages[0].height == doc.pages[0].height

    def test_string_content_preserved(self) -> None:
        doc = _simple_doc()
        xml = write_alto(doc)
        doc2 = parse_alto(xml)
        block = doc2.pages[0].blocks[0]
        assert block.lines[0].strings[0].content == "Hello"
        assert block.lines[1].strings[1].content == "line"

    def test_bbox_preserved(self) -> None:
        doc = AltoDocument(
            pages=(AltoPage(
                blocks=(AltoTextBlock(
                    lines=(AltoLine(strings=(
                        AltoString(
                            content="x",
                            bbox=AltoBBox(hpos=10, vpos=20, width=30, height=40),
                        ),
                    ),),),
                ),),
            ),),
        )
        doc2 = parse_alto(write_alto(doc))
        bbox = doc2.pages[0].blocks[0].lines[0].strings[0].bbox
        assert bbox is not None
        assert bbox.hpos == 10 and bbox.vpos == 20
        assert bbox.width == 30 and bbox.height == 40

    def test_byte_deterministic(self) -> None:
        """Même structure → mêmes octets."""
        doc1 = _simple_doc()
        doc2 = _simple_doc()
        assert write_alto(doc1) == write_alto(doc2)

    def test_write_in_v3(self) -> None:
        xml = write_alto(_simple_doc(), version="v3")
        doc = parse_alto(xml)
        assert doc.source_version == "v3"

    def test_write_no_namespace(self) -> None:
        xml = write_alto(_simple_doc(), version="none")
        doc = parse_alto(xml)
        assert doc.source_version == "none"

    def test_invalid_version_rejected(self) -> None:
        from picarones.domain.errors import PicaronesError
        with pytest.raises(PicaronesError, match="version ALTO invalide"):
            write_alto(_simple_doc(), version="v9")


# ──────────────────────────────────────────────────────────────────────
# Projector — extraction texte + césure
# ──────────────────────────────────────────────────────────────────────


class TestExtractText:
    def test_simple_text(self) -> None:
        text = alto_document_to_text(_simple_doc())
        assert text == "Hello world\nsecond line"

    def test_multi_block_separated_by_blank_line(self) -> None:
        doc = AltoDocument(pages=(AltoPage(
            blocks=(
                AltoTextBlock(lines=(
                    AltoLine(strings=(AltoString(content="A"),)),
                ),),
                AltoTextBlock(lines=(
                    AltoLine(strings=(AltoString(content="B"),)),
                ),),
            ),
        ),),)
        assert alto_document_to_text(doc) == "A\n\nB"

    def test_hyphenation_same_line_with_subs_content(self) -> None:
        """HypPart1 + HypPart2 sur la même ligne, SUBS_CONTENT fourni."""
        doc = AltoDocument(pages=(AltoPage(
            blocks=(AltoTextBlock(lines=(
                AltoLine(strings=(
                    AltoString(content="Bonjour"),
                    AltoString(
                        content="est-",
                        subs_type="HypPart1",
                        subs_content="est-il",
                    ),
                    AltoString(content="il", subs_type="HypPart2"),
                    AltoString(content="clair"),
                )),
            ),),),
        ),),)
        # "est-il" reconstruit, "il" suivant skippé.
        assert alto_document_to_text(doc) == "Bonjour est-il clair"

    def test_hyphenation_cross_line(self) -> None:
        """HypPart1 fin d'une ligne, HypPart2 début ligne suivante.

        C'est l'usage standard ALTO (la césure visuelle correspond à
        un saut de ligne réel).
        """
        doc = AltoDocument(pages=(AltoPage(
            blocks=(AltoTextBlock(lines=(
                AltoLine(strings=(
                    AltoString(content="ceci"),
                    AltoString(
                        content="est-",
                        subs_type="HypPart1",
                        subs_content="est-il",
                    ),
                )),
                AltoLine(strings=(
                    AltoString(content="il", subs_type="HypPart2"),
                    AltoString(content="clair"),
                )),
            ),),),
        ),),)
        # Ligne 1 : "ceci est-il" (mot complet placé en fin de ligne 1).
        # Ligne 2 : "clair" (le HypPart2 "il" est skippé).
        assert alto_document_to_text(doc) == "ceci est-il\nclair"

    def test_hyphenation_no_subs_content_concatenates(self) -> None:
        doc = AltoDocument(pages=(AltoPage(
            blocks=(AltoTextBlock(lines=(
                AltoLine(strings=(
                    AltoString(content="lec-", subs_type="HypPart1"),
                    AltoString(content="ture", subs_type="HypPart2"),
                )),
            ),),),
        ),),)
        assert alto_document_to_text(doc) == "lec-ture"


# ──────────────────────────────────────────────────────────────────────
# AltoToText projector (protocole)
# ──────────────────────────────────────────────────────────────────────


class TestAltoToTextProjector:
    def test_protocol_satisfied(self) -> None:
        from picarones.evaluation.projectors import Projector
        assert isinstance(AltoToText(), Projector)

    def test_project_from_filesystem(self, tmp_path) -> None:
        xml = write_alto(_simple_doc())
        path = tmp_path / "doc.alto.xml"
        path.write_bytes(xml)

        artifact = Artifact(
            id="d1:ocr:alto",
            document_id="d1",
            type=ArtifactType.ALTO_XML,
            uri=str(path),
        )
        projector = AltoToText()
        target, report = projector.project(artifact, {})
        assert target.type == ArtifactType.RAW_TEXT
        assert report.lossy is True
        assert "geometry" in report.ignored_dimensions

    def test_project_wrong_type_raises(self) -> None:
        artifact = Artifact(
            id="d1:image", document_id="d1",
            type=ArtifactType.IMAGE,
        )
        with pytest.raises(ProjectionError, match="ALTO_XML"):
            AltoToText().project(artifact, {})

    def test_project_missing_uri_raises(self) -> None:
        artifact = Artifact(
            id="d1:alto", document_id="d1",
            type=ArtifactType.ALTO_XML,
        )
        with pytest.raises(ProjectionError, match="URI"):
            AltoToText().project(artifact, {})
