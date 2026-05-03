"""Sprint A14-S4 — ``ProvenanceRecord`` + hiérarchie d'erreurs."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from picarones.domain import (
    ArtifactValidationError,
    CorpusSpecError,
    PicaronesError,
    ProjectionError,
    ProvenanceRecord,
)


class TestProvenanceRecord:
    def test_minimal_provenance(self) -> None:
        p = ProvenanceRecord(code_version="1.0.0")
        assert p.code_version == "1.0.0"
        assert p.parameters_hash is None
        assert isinstance(p.timestamp, datetime)
        assert p.timestamp.tzinfo == timezone.utc

    def test_with_parameters_hash(self) -> None:
        p = ProvenanceRecord(code_version="1.0.0", parameters_hash="a" * 64)
        assert p.parameters_hash == "a" * 64

    def test_compatibility_check(self) -> None:
        p1 = ProvenanceRecord(code_version="1.0.0", parameters_hash="x" * 64)
        p2 = ProvenanceRecord(code_version="1.0.0", parameters_hash="x" * 64)
        assert p1.is_compatible_with(p2)

        p3 = ProvenanceRecord(code_version="1.0.1", parameters_hash="x" * 64)
        assert not p1.is_compatible_with(p3)  # code_version diffère

        p4 = ProvenanceRecord(code_version="1.0.0", parameters_hash="y" * 64)
        assert not p1.is_compatible_with(p4)  # parameters_hash diffère

    def test_frozen(self) -> None:
        p = ProvenanceRecord(code_version="1.0.0")
        with pytest.raises(Exception):
            p.code_version = "1.0.1"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        p = ProvenanceRecord(code_version="1.0.0", parameters_hash="x" * 64)
        p2 = ProvenanceRecord.model_validate_json(p.model_dump_json())
        assert p == p2


class TestErrorHierarchy:
    def test_all_errors_inherit_picarones_error(self) -> None:
        for cls in (
            ArtifactValidationError,
            ProjectionError,
            CorpusSpecError,
        ):
            assert issubclass(cls, PicaronesError), (
                f"{cls.__name__} doit hériter de PicaronesError pour "
                "permettre un `except PicaronesError` global au niveau "
                "de la couche transport."
            )

    def test_picarones_error_is_exception(self) -> None:
        assert issubclass(PicaronesError, Exception)

    def test_can_raise_and_catch_via_base(self) -> None:
        with pytest.raises(PicaronesError):
            raise ArtifactValidationError("x")
        with pytest.raises(PicaronesError):
            raise ProjectionError("y")
        with pytest.raises(PicaronesError):
            raise CorpusSpecError("z")
