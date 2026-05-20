"""Sprint A14-S4 — ``DocumentRef`` et ``GroundTruthRef`` multi-niveaux."""

from __future__ import annotations

import pytest

from picarones.domain import (
    ArtifactType,
    CorpusSpecError,
    DocumentRef,
    GroundTruthRef,
)


class TestDocumentRefBasics:
    def test_minimal_document(self) -> None:
        d = DocumentRef(id="folio_001")
        assert d.id == "folio_001"
        assert d.image_uri is None
        assert d.ground_truths == ()

    def test_document_with_image_and_text_gt(self) -> None:
        d = DocumentRef(
            id="folio_001",
            image_uri="/corpus/folio_001.png",
            ground_truths=(
                GroundTruthRef(type=ArtifactType.RAW_TEXT, uri="/corpus/folio_001.gt.txt"),
            ),
        )
        assert d.image_uri == "/corpus/folio_001.png"
        assert len(d.ground_truths) == 1

    def test_id_validation_rejects_spaces(self) -> None:
        with pytest.raises(CorpusSpecError, match="document id invalide"):
            DocumentRef(id="bad id")


class TestMultiLevelGT:
    def test_multi_level_gt(self) -> None:
        d = DocumentRef(
            id="folio_001",
            ground_truths=(
                GroundTruthRef(type=ArtifactType.RAW_TEXT, uri="/x.gt.txt"),
                GroundTruthRef(type=ArtifactType.ALTO_XML, uri="/x.gt.alto.xml"),
                GroundTruthRef(type=ArtifactType.READING_ORDER, uri="/x.ro.json"),
            ),
        )
        assert len(d.ground_truths) == 3
        assert d.available_gt_types == (
            ArtifactType.RAW_TEXT,
            ArtifactType.ALTO_XML,
            ArtifactType.READING_ORDER,
        )

    def test_gt_for_returns_matching_level(self) -> None:
        d = DocumentRef(
            id="x",
            ground_truths=(
                GroundTruthRef(type=ArtifactType.RAW_TEXT, uri="/x.txt"),
                GroundTruthRef(type=ArtifactType.ALTO_XML, uri="/x.xml"),
            ),
        )
        gt = d.gt_for(ArtifactType.ALTO_XML)
        assert gt is not None
        assert gt.uri == "/x.xml"

    def test_gt_for_returns_none_when_absent(self) -> None:
        d = DocumentRef(id="x")
        assert d.gt_for(ArtifactType.RAW_TEXT) is None

    def test_duplicate_gt_type_rejected(self) -> None:
        with pytest.raises(CorpusSpecError, match="GT dupliquée"):
            DocumentRef(
                id="x",
                ground_truths=(
                    GroundTruthRef(type=ArtifactType.RAW_TEXT, uri="/a.txt"),
                    GroundTruthRef(type=ArtifactType.RAW_TEXT, uri="/b.txt"),
                ),
            )


class TestDocumentRefImmutability:
    def test_frozen_blocks_mutation(self) -> None:
        from pydantic import ValidationError

        d = DocumentRef(id="x")
        with pytest.raises(ValidationError):
            d.id = "y"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        d = DocumentRef(
            id="vol_a/folio_001",
            image_uri="/c/folio_001.png",
            ground_truths=(
                GroundTruthRef(type=ArtifactType.ALTO_XML, uri="/x.xml"),
            ),
        )
        j = d.model_dump_json()
        d2 = DocumentRef.model_validate_json(j)
        assert d == d2
