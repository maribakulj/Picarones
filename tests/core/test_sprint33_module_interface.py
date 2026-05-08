"""Tests Sprint 33 — Interface module générique (Phase 0.2).

Vérifie :

1. ``BaseModule`` est instanciable via une sous-classe minimale qui
   déclare ses ``input_types`` / ``output_types`` et implémente
   ``process``.
2. La validation des entrées/sorties (``validate_inputs`` /
   ``validate_outputs``) lève ``ValueError`` quand un type déclaré est
   manquant.
3. Un ``MockModule`` qui consomme ``TEXT`` et produit ``ALTO`` peut
   exister — l'interface n'est pas restreinte aux OCR (critère
   explicite du plan).
4. ``BaseOCREngine`` hérite de ``BaseModule`` et expose
   ``input_types=(IMAGE,)``, ``output_types=(TEXT,)``.
5. La méthode ``process`` d'un moteur OCR existant délègue correctement
   à ``run``/``_run_ocr`` et retourne le bon type d'artefact.
6. Les valeurs string de ``ArtifactType`` correspondent à celles de
   ``GTLevel`` pour permettre la conversion triviale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from picarones.domain.artifacts import ArtifactType
from picarones.domain.module_protocol import BaseModule
from picarones.adapters.legacy_engines.base import BaseOCREngine, EngineResult


# ──────────────────────────────────────────────────────────────────────────
# Fixtures de modules de test
# ──────────────────────────────────────────────────────────────────────────


class UpperCaseTextModule(BaseModule):
    """Module trivial TEXT → TEXT pour valider le contrat de base."""

    input_types = (ArtifactType.TEXT,)
    output_types = (ArtifactType.TEXT,)
    execution_mode = "cpu"

    @property
    def name(self) -> str:
        return "uppercase"

    def process(self, inputs: dict[ArtifactType, Any]) -> dict[ArtifactType, Any]:
        self.validate_inputs(inputs)
        return {ArtifactType.TEXT: inputs[ArtifactType.TEXT].upper()}


class TextToAltoMock(BaseModule):
    """Mock TEXT → ALTO : le critère de réussite explicite du plan.

    Un cas d'école pour le futur ``alto_reconstructor`` BnF (cf. plan
    d'évolution, Sprint B.1).
    """

    input_types = (ArtifactType.TEXT,)
    output_types = (ArtifactType.ALTO,)
    execution_mode = "cpu"

    @property
    def name(self) -> str:
        return "text_to_alto_mock"

    def process(self, inputs: dict[ArtifactType, Any]) -> dict[ArtifactType, Any]:
        self.validate_inputs(inputs)
        text = inputs[ArtifactType.TEXT]
        # Génère un ALTO trivial qui contient le texte en CONTENT
        alto = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">'
            f'<Layout><Page><PrintSpace><TextBlock><TextLine>'
            f'<String CONTENT="{text}"/>'
            f'</TextLine></TextBlock></PrintSpace></Page></Layout>'
            '</alto>'
        )
        return {ArtifactType.ALTO: alto}

    def metadata(self) -> dict:
        return {"strategy": "trivial_single_string"}


class FaultyModule(BaseModule):
    """Module qui prétend produire ALTO mais ne le fait pas — pour tester
    la validation des sorties."""

    input_types = (ArtifactType.TEXT,)
    output_types = (ArtifactType.ALTO,)

    @property
    def name(self) -> str:
        return "faulty"

    def process(self, inputs: dict[ArtifactType, Any]) -> dict[ArtifactType, Any]:
        return {ArtifactType.TEXT: "oops"}  # mauvais type de sortie


class FakeOCREngine(BaseOCREngine):
    """Moteur OCR factice pour tester la délégation BaseOCREngine.process."""

    @property
    def name(self) -> str:
        return "fake_ocr"

    def version(self) -> str:
        return "0.1.0"

    def _run_ocr(self, image_path: Path) -> str:
        return f"transcription de {image_path.name}"


# ──────────────────────────────────────────────────────────────────────────
# 1 & 2. Contrat BaseModule : instanciation et validation
# ──────────────────────────────────────────────────────────────────────────


class TestBaseModuleContract:
    def test_minimal_module_runs(self) -> None:
        m = UpperCaseTextModule()
        out = m.process({ArtifactType.TEXT: "bonjour"})
        assert out == {ArtifactType.TEXT: "BONJOUR"}

    def test_validate_inputs_missing_raises(self) -> None:
        m = UpperCaseTextModule()
        with pytest.raises(ValueError, match="entrées manquantes"):
            m.validate_inputs({})

    def test_validate_outputs_missing_raises(self) -> None:
        m = UpperCaseTextModule()
        with pytest.raises(ValueError, match="sorties manquantes"):
            m.validate_outputs({})

    def test_validate_outputs_passes_when_complete(self) -> None:
        m = UpperCaseTextModule()
        # Doit passer sans lever
        m.validate_outputs({ArtifactType.TEXT: "hello"})

    def test_default_metadata_is_empty(self) -> None:
        assert UpperCaseTextModule().metadata() == {}

    def test_repr_shows_io_types(self) -> None:
        m = UpperCaseTextModule()
        r = repr(m)
        assert "uppercase" in r
        # Phase 4-bis : ``ArtifactType.TEXT.value`` est désormais
        # ``"raw_text"`` (alias canonique vers ``RAW_TEXT``).
        assert "raw_text→raw_text" in r

    def test_default_execution_mode(self) -> None:
        # UpperCaseTextModule a forcé "cpu" ; un module qui ne déclare
        # rien hérite de "io".
        class IOModule(BaseModule):
            input_types = (ArtifactType.TEXT,)
            output_types = (ArtifactType.TEXT,)

            @property
            def name(self) -> str:
                return "io"

            def process(self, inputs):
                return {ArtifactType.TEXT: inputs[ArtifactType.TEXT]}

        assert IOModule.execution_mode == "io"


# ──────────────────────────────────────────────────────────────────────────
# 3. MockModule TEXT → ALTO (critère explicite du plan)
# ──────────────────────────────────────────────────────────────────────────


class TestMockTextToAlto:
    def test_text_to_alto_runs(self) -> None:
        m = TextToAltoMock()
        out = m.process({ArtifactType.TEXT: "Hello"})

        assert ArtifactType.ALTO in out
        assert "Hello" in out[ArtifactType.ALTO]
        assert "alto" in out[ArtifactType.ALTO]

    def test_text_to_alto_declares_correct_types(self) -> None:
        assert TextToAltoMock.input_types == (ArtifactType.TEXT,)
        assert TextToAltoMock.output_types == (ArtifactType.ALTO,)

    def test_text_to_alto_metadata_exposed(self) -> None:
        assert TextToAltoMock().metadata() == {"strategy": "trivial_single_string"}

    def test_validate_inputs_catches_missing_text(self) -> None:
        m = TextToAltoMock()
        with pytest.raises(ValueError):
            # Donne une IMAGE alors qu'on attend TEXT
            m.process({ArtifactType.IMAGE: Path("/tmp/x.png")})


# ──────────────────────────────────────────────────────────────────────────
# 4 & 5. BaseOCREngine est rétrocompatible et respecte BaseModule
# ──────────────────────────────────────────────────────────────────────────


class TestOCREngineAsModule:
    def test_baseocrengine_is_basemodule(self) -> None:
        assert issubclass(BaseOCREngine, BaseModule)

    def test_baseocrengine_io_types(self) -> None:
        assert BaseOCREngine.input_types == (ArtifactType.IMAGE,)
        assert BaseOCREngine.output_types == (ArtifactType.TEXT,)

    def test_fake_engine_run_unchanged(self, tmp_path: Path) -> None:
        """L'API historique ``run`` retourne un ``EngineResult`` intact."""
        image = tmp_path / "doc.png"
        image.write_bytes(b"\x89PNG")
        engine = FakeOCREngine()

        result = engine.run(image)

        assert isinstance(result, EngineResult)
        assert result.success
        assert result.text == "transcription de doc.png"
        assert result.engine_name == "fake_ocr"

    def test_fake_engine_process_returns_text_artifact(self, tmp_path: Path) -> None:
        """``process`` délègue à ``run`` et retourne ``{TEXT: ...}``."""
        image = tmp_path / "doc.png"
        image.write_bytes(b"\x89PNG")
        engine = FakeOCREngine()

        outputs = engine.process({ArtifactType.IMAGE: image})

        assert outputs == {ArtifactType.TEXT: "transcription de doc.png"}

    def test_fake_engine_process_validates_missing_image(self) -> None:
        engine = FakeOCREngine()
        with pytest.raises(ValueError, match="entrées manquantes"):
            engine.process({ArtifactType.TEXT: "wrong artifact"})

    def test_fake_engine_metadata_exposes_version(self) -> None:
        meta = FakeOCREngine().metadata()
        assert meta == {"engine_version": "0.1.0"}


# ──────────────────────────────────────────────────────────────────────────
# 6. Cohérence ArtifactType — niveaux de GT
# ──────────────────────────────────────────────────────────────────────────
# Phase 4 leftover : l'ancien test de cohérence GTLevel ↔ ArtifactType
# a été supprimé en même temps que ``GTLevel`` (mai 2026) au profit
# d'un usage direct d'``ArtifactType`` partout.
