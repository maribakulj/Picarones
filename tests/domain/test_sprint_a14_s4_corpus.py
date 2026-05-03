"""Sprint A14-S4 — ``CorpusSpec`` immuable."""

from __future__ import annotations

import pytest

from picarones.domain import ArtifactType, CorpusSpec, CorpusSpecError, DocumentRef, GroundTruthRef


def _doc(doc_id: str) -> DocumentRef:
    return DocumentRef(id=doc_id)


class TestCorpusSpec:
    def test_empty_corpus(self) -> None:
        c = CorpusSpec(name="empty")
        assert len(c) == 0
        assert c.documents == ()

    def test_corpus_with_documents(self) -> None:
        c = CorpusSpec(
            name="bnf_demo",
            documents=(_doc("a"), _doc("b"), _doc("c")),
        )
        assert len(c) == 3

    def test_doc_by_id_finds_document(self) -> None:
        c = CorpusSpec(name="x", documents=(_doc("a"), _doc("b")))
        assert c.doc_by_id("a") is not None
        assert c.doc_by_id("b") is not None
        assert c.doc_by_id("missing") is None

    def test_duplicate_doc_ids_rejected(self) -> None:
        with pytest.raises(CorpusSpecError, match="dupliqué"):
            CorpusSpec(
                name="x",
                documents=(_doc("a"), _doc("b"), _doc("a")),
            )

    def test_metadata_is_free_dict(self) -> None:
        c = CorpusSpec(
            name="x",
            metadata={"language": "fr", "period": "early_modern"},
        )
        assert c.metadata["language"] == "fr"

    def test_name_validation(self) -> None:
        with pytest.raises(Exception):  # pydantic ValidationError
            CorpusSpec(name="")  # min_length=1


class TestCorpusSpecImmutability:
    def test_frozen_blocks_mutation(self) -> None:
        c = CorpusSpec(name="x")
        with pytest.raises(Exception):
            c.name = "y"  # type: ignore[misc]

    def test_json_roundtrip_with_multilevel_gt(self) -> None:
        c = CorpusSpec(
            name="philological",
            documents=(
                DocumentRef(
                    id="folio_001",
                    image_uri="/c/folio_001.png",
                    ground_truths=(
                        GroundTruthRef(type=ArtifactType.RAW_TEXT, uri="/x.txt"),
                        GroundTruthRef(type=ArtifactType.ALTO_XML, uri="/x.xml"),
                    ),
                ),
            ),
            metadata={"language": "lat"},
        )
        j = c.model_dump_json()
        c2 = CorpusSpec.model_validate_json(j)
        assert c == c2
