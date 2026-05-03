"""Sprint A14-S4 — ``Artifact`` et ``ArtifactType``.

Vérifie les invariants des artefacts du nouveau domain : validation
des id, hash, immutabilité, sérialisation JSON déterministe.

Note : pas de test "logique métier" ici — un Artifact ne fait rien,
il décrit.  Les tests qui valident le comportement viendront avec
le pipeline executor (S7) qui produit et consomme des artefacts.
"""

from __future__ import annotations

import hashlib

import pytest

from picarones.domain import (
    Artifact,
    ArtifactType,
    ArtifactValidationError,
    ProvenanceRecord,
    compute_content_hash,
)


def _prov() -> ProvenanceRecord:
    return ProvenanceRecord(code_version="1.0.0", parameters_hash="a" * 64)


# ──────────────────────────────────────────────────────────────────────
# ArtifactType
# ──────────────────────────────────────────────────────────────────────


class TestArtifactType:
    def test_nine_canonical_values(self) -> None:
        """Sprint A14-S4 — 9 valeurs canoniques."""
        expected = {
            "image", "raw_text", "corrected_text",
            "alto_xml", "page_xml", "canonical_document",
            "entities", "reading_order", "alignment",
        }
        assert {t.value for t in ArtifactType} == expected

    def test_string_enum_serializes_as_value(self) -> None:
        """``ArtifactType`` hérite de ``str`` → JSON en string brute."""
        assert ArtifactType.RAW_TEXT == "raw_text"
        assert ArtifactType("alto_xml") is ArtifactType.ALTO_XML


# ──────────────────────────────────────────────────────────────────────
# compute_content_hash
# ──────────────────────────────────────────────────────────────────────


class TestComputeContentHash:
    def test_returns_64_char_hex(self) -> None:
        h = compute_content_hash(b"hello")
        assert len(h) == 64
        assert int(h, 16) >= 0  # hex valide

    def test_deterministic(self) -> None:
        assert compute_content_hash(b"abc") == compute_content_hash(b"abc")

    def test_matches_sha256(self) -> None:
        h = compute_content_hash(b"picarones")
        assert h == hashlib.sha256(b"picarones").hexdigest()


# ──────────────────────────────────────────────────────────────────────
# Artifact — création et validation
# ──────────────────────────────────────────────────────────────────────


class TestArtifactCreation:
    def test_minimal_artifact(self) -> None:
        a = Artifact(id="x", document_id="d1", type=ArtifactType.RAW_TEXT)
        assert a.id == "x"
        assert a.uri is None
        assert a.content_hash is None
        assert a.produced_by_step is None
        assert a.provenance is None

    def test_full_artifact(self) -> None:
        a = Artifact(
            id="d1:ocr:raw_text",
            document_id="d1",
            type=ArtifactType.RAW_TEXT,
            uri="/tmp/x.txt",
            content_hash="b" * 64,
            produced_by_step="ocr",
            provenance=_prov(),
        )
        assert a.produced_by_step == "ocr"

    def test_id_validation_rejects_spaces(self) -> None:
        with pytest.raises(ArtifactValidationError, match="id invalide"):
            Artifact(id="bad id", document_id="d1", type=ArtifactType.RAW_TEXT)

    def test_id_validation_rejects_null_byte(self) -> None:
        with pytest.raises(ArtifactValidationError):
            Artifact(id="x\x00y", document_id="d1", type=ArtifactType.RAW_TEXT)

    def test_id_accepts_filesystem_safe_chars(self) -> None:
        # alphanum + ``_.-:/`` selon le regex.
        a = Artifact(
            id="vol_a:folio.001-r/raw_text",
            document_id="vol_a/folio.001-r",
            type=ArtifactType.RAW_TEXT,
        )
        assert a.id == "vol_a:folio.001-r/raw_text"

    def test_content_hash_must_be_64_hex(self) -> None:
        # Trop court
        with pytest.raises(Exception):  # pydantic ValidationError
            Artifact(
                id="x", document_id="d1", type=ArtifactType.RAW_TEXT,
                content_hash="abc",
            )
        # Bonne longueur mais pas hex
        with pytest.raises(ArtifactValidationError, match="hex SHA-256"):
            Artifact(
                id="x", document_id="d1", type=ArtifactType.RAW_TEXT,
                content_hash="z" * 64,
            )

    def test_content_hash_lowercased(self) -> None:
        a = Artifact(
            id="x", document_id="d1", type=ArtifactType.RAW_TEXT,
            content_hash="A" * 64,
        )
        assert a.content_hash == "a" * 64


# ──────────────────────────────────────────────────────────────────────
# Artifact — immutabilité
# ──────────────────────────────────────────────────────────────────────


class TestArtifactImmutability:
    def test_frozen_blocks_attribute_mutation(self) -> None:
        a = Artifact(id="x", document_id="d1", type=ArtifactType.RAW_TEXT)
        with pytest.raises(Exception):  # pydantic ValidationError
            a.id = "y"  # type: ignore[misc]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(Exception):  # pydantic ValidationError
            Artifact(  # type: ignore[call-arg]
                id="x", document_id="d1", type=ArtifactType.RAW_TEXT,
                bogus_field="oops",
            )


# ──────────────────────────────────────────────────────────────────────
# Artifact — sérialisation déterministe
# ──────────────────────────────────────────────────────────────────────


class TestArtifactSerialization:
    def test_json_roundtrip_preserves_equality(self) -> None:
        a = Artifact(
            id="d1:ocr:raw_text", document_id="d1",
            type=ArtifactType.RAW_TEXT, content_hash="c" * 64,
            produced_by_step="ocr", provenance=_prov(),
        )
        j = a.model_dump_json()
        a2 = Artifact.model_validate_json(j)
        assert a == a2

    def test_json_is_byte_deterministic(self) -> None:
        """Même contenu → mêmes octets exacts.  Indispensable au cache
        d'artefacts du Sprint S7."""
        a1 = Artifact(
            id="x", document_id="d1", type=ArtifactType.RAW_TEXT,
            content_hash="d" * 64,
        )
        a2 = Artifact(
            id="x", document_id="d1", type=ArtifactType.RAW_TEXT,
            content_hash="d" * 64,
        )
        assert a1.model_dump_json() == a2.model_dump_json()

    def test_artifacts_are_hashable(self) -> None:
        """Frozen pydantic models sont hashables — on peut les mettre
        dans un set ou utiliser comme clé de dict."""
        a = Artifact(id="x", document_id="d1", type=ArtifactType.RAW_TEXT)
        s = {a}
        assert a in s
