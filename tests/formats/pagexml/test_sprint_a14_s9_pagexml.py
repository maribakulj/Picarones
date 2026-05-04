"""Sprint A14-S9 — PAGE XML parser, projector."""

from __future__ import annotations

import pytest

from picarones.domain import Artifact, ArtifactType
from picarones.domain.errors import ProjectionError
from picarones.formats.pagexml import (
    PageDocument,
    PageParseError,
    PagePage,
    PageTextLine,
    PageTextRegion,
    PageToText,
    page_document_to_text,
    parse_pagexml,
)


_SAMPLE_PAGE_XML = '''<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageFilename="folio_001.png" imageWidth="1200" imageHeight="1800">
    <TextRegion id="r1" type="paragraph">
      <Coords points="100,100 1100,100 1100,400 100,400"/>
      <TextLine id="l1">
        <Coords points="100,100 1100,100 1100,150 100,150"/>
        <Baseline points="100,140 1100,140"/>
        <TextEquiv><Unicode>Premier ligne</Unicode></TextEquiv>
      </TextLine>
      <TextLine id="l2">
        <TextEquiv><Unicode>deuxième ligne</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
    <TextRegion id="r2" type="heading">
      <TextLine id="l3">
        <TextEquiv><Unicode>Titre</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
'''.encode("utf-8")


class TestParser:
    def test_parse_simple_page(self) -> None:
        doc = parse_pagexml(_SAMPLE_PAGE_XML)
        assert len(doc.pages) == 1
        page = doc.pages[0]
        assert page.image_filename == "folio_001.png"
        assert page.image_width == 1200
        assert page.image_height == 1800
        assert len(page.text_regions) == 2

    def test_text_lines_extracted(self) -> None:
        doc = parse_pagexml(_SAMPLE_PAGE_XML)
        r1 = doc.pages[0].text_regions[0]
        assert len(r1.text_lines) == 2
        assert r1.text_lines[0].text == "Premier ligne"
        assert r1.text_lines[0].coords is not None
        assert r1.text_lines[0].baseline is not None

    def test_region_type_preserved(self) -> None:
        doc = parse_pagexml(_SAMPLE_PAGE_XML)
        assert doc.pages[0].text_regions[0].region_type == "paragraph"
        assert doc.pages[0].text_regions[1].region_type == "heading"

    def test_namespace_detected(self) -> None:
        doc = parse_pagexml(_SAMPLE_PAGE_XML)
        assert doc.source_namespace is not None
        assert "primaresearch" in doc.source_namespace

    def test_empty_raises(self) -> None:
        with pytest.raises(PageParseError, match="vide"):
            parse_pagexml(b"")

    def test_invalid_xml_raises(self) -> None:
        with pytest.raises(PageParseError, match="invalide"):
            parse_pagexml(b"<not closed")

    def test_xxe_blocked(self) -> None:
        xml = b'''<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<PcGts>&xxe;</PcGts>'''
        with pytest.raises(PageParseError):
            parse_pagexml(xml)


class TestExtractText:
    def test_full_extraction(self) -> None:
        doc = parse_pagexml(_SAMPLE_PAGE_XML)
        text = page_document_to_text(doc)
        # 2 régions séparées par ligne vide, lignes par \n.
        assert text == "Premier ligne\ndeuxième ligne\n\nTitre"

    def test_empty_document(self) -> None:
        doc = PageDocument()
        assert page_document_to_text(doc) == ""

    def test_region_without_lines_skipped(self) -> None:
        doc = PageDocument(pages=(PagePage(
            text_regions=(
                PageTextRegion(id="empty"),
                PageTextRegion(
                    id="full",
                    text_lines=(PageTextLine(text="hello"),),
                ),
            ),
        ),),)
        assert page_document_to_text(doc) == "hello"


class TestProjector:
    def test_protocol_satisfied(self) -> None:
        from picarones.evaluation.projectors import Projector
        assert isinstance(PageToText(), Projector)

    def test_project_from_filesystem(self, tmp_path) -> None:
        path = tmp_path / "doc.page.xml"
        path.write_bytes(_SAMPLE_PAGE_XML)
        artifact = Artifact(
            id="d:page",
            document_id="d",
            type=ArtifactType.PAGE_XML,
            uri=str(path),
        )
        target, report = PageToText().project(artifact, {})
        assert target.type == ArtifactType.RAW_TEXT
        assert "geometry" in report.ignored_dimensions

    def test_wrong_type_rejected(self) -> None:
        artifact = Artifact(
            id="d:alto", document_id="d", type=ArtifactType.ALTO_XML,
        )
        with pytest.raises(ProjectionError, match="PAGE_XML"):
            PageToText().project(artifact, {})
