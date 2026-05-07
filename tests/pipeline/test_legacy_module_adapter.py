"""Tests Sub-phase 7.B — adaptateur ``BaseModule`` → ``StepExecutor``.

Couvre :

1. ``_PayloadRegistry`` : store / get / __contains__ / clear.
2. ``_BaseModuleAdapter`` : satisfaction du Protocol
   ``StepExecutor`` (props + execute).
3. ``wrap_initial_inputs`` : conversion payloads bruts →
   ``dict[ArtifactType, Artifact]`` avec registre.
4. Bout-en-bout : enchaînement de 2 modules legacy via
   ``PipelineExecutor`` canonique grâce à l'adapter.
"""

from __future__ import annotations

from typing import Any

import pytest

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.module_protocol import BaseModule
from picarones.pipeline._legacy_module_adapter import (
    _BaseModuleAdapter,
    _PayloadRegistry,
    wrap_initial_inputs,
)
from picarones.pipeline.protocols import StepExecutor
from picarones.pipeline.types import RunContext


# ──────────────────────────────────────────────────────────────────
# Modules legacy mocks
# ──────────────────────────────────────────────────────────────────


class _UpperCaseTextModule(BaseModule):
    """Mock : RAW_TEXT → RAW_TEXT (uppercase)."""

    input_types = (ArtifactType.RAW_TEXT,)
    output_types = (ArtifactType.RAW_TEXT,)
    execution_mode = "cpu"

    @property
    def name(self) -> str:
        return "upper-case"

    def process(self, inputs: dict[ArtifactType, Any]) -> dict[ArtifactType, Any]:
        text = inputs[ArtifactType.RAW_TEXT]
        return {ArtifactType.RAW_TEXT: text.upper()}


class _ImageOCRModule(BaseModule):
    """Mock : IMAGE → RAW_TEXT (constant)."""

    input_types = (ArtifactType.IMAGE,)
    output_types = (ArtifactType.RAW_TEXT,)
    execution_mode = "io"

    def __init__(self, fixed_output: str) -> None:
        self._out = fixed_output

    @property
    def name(self) -> str:
        return "mock-ocr"

    def process(self, inputs: dict[ArtifactType, Any]) -> dict[ArtifactType, Any]:
        # On vérifie qu'on reçoit bien un chemin string pour IMAGE
        image_path = inputs[ArtifactType.IMAGE]
        assert isinstance(image_path, str)
        return {ArtifactType.RAW_TEXT: self._out}


# ──────────────────────────────────────────────────────────────────
# 1. _PayloadRegistry
# ──────────────────────────────────────────────────────────────────


class TestPayloadRegistry:
    def test_store_and_get(self) -> None:
        reg = _PayloadRegistry()
        reg.store("a:1", "hello")
        assert reg.get("a:1") == "hello"

    def test_get_missing_raises(self) -> None:
        reg = _PayloadRegistry()
        with pytest.raises(KeyError, match="introuvable"):
            reg.get("missing")

    def test_contains(self) -> None:
        reg = _PayloadRegistry()
        reg.store("k", 42)
        assert "k" in reg
        assert "other" not in reg

    def test_clear(self) -> None:
        reg = _PayloadRegistry()
        reg.store("k", 1)
        reg.clear()
        assert "k" not in reg


# ──────────────────────────────────────────────────────────────────
# 2. _BaseModuleAdapter
# ──────────────────────────────────────────────────────────────────


class TestAdapterProperties:
    def test_satisfies_step_executor_protocol(self) -> None:
        """Le wrapper doit être ``isinstance``-compatible avec ``StepExecutor``."""
        adapter = _BaseModuleAdapter(_UpperCaseTextModule(), _PayloadRegistry())
        assert isinstance(adapter, StepExecutor)

    def test_name_proxies_module_name(self) -> None:
        adapter = _BaseModuleAdapter(_UpperCaseTextModule(), _PayloadRegistry())
        assert adapter.name == "upper-case"

    def test_input_types_is_frozenset(self) -> None:
        adapter = _BaseModuleAdapter(_UpperCaseTextModule(), _PayloadRegistry())
        assert adapter.input_types == frozenset({ArtifactType.RAW_TEXT})

    def test_output_types_is_frozenset(self) -> None:
        adapter = _BaseModuleAdapter(_UpperCaseTextModule(), _PayloadRegistry())
        assert adapter.output_types == frozenset({ArtifactType.RAW_TEXT})

    def test_execution_mode_proxied(self) -> None:
        adapter = _BaseModuleAdapter(_UpperCaseTextModule(), _PayloadRegistry())
        assert adapter.execution_mode == "cpu"


class TestAdapterExecute:
    def _make_context(self, document_id: str = "doc1") -> RunContext:
        return RunContext(
            document_id=document_id,
            code_version="1.0.0",
            pipeline_name="test",
        )

    def test_execute_text_to_text(self) -> None:
        """Un module RAW_TEXT → RAW_TEXT s'exécute via l'adapter."""
        registry = _PayloadRegistry()
        # Initial input : payload inline
        registry.store("doc1:__initial__:raw_text", "hello")
        in_artifact = Artifact(
            id="doc1:__initial__:raw_text",
            document_id="doc1",
            type=ArtifactType.RAW_TEXT,
        )
        adapter = _BaseModuleAdapter(_UpperCaseTextModule(), registry)
        outputs = adapter.execute(
            {ArtifactType.RAW_TEXT: in_artifact},
            params={},
            context=self._make_context(),
        )
        # Output type bien présent
        assert ArtifactType.RAW_TEXT in outputs
        out_artifact = outputs[ArtifactType.RAW_TEXT]
        # Artifact bien construit
        assert out_artifact.document_id == "doc1"
        assert out_artifact.produced_by_step == "upper-case"
        assert out_artifact.id == "doc1:upper-case:raw_text"
        # Payload bien stocké dans le registre
        assert registry.get(out_artifact.id) == "HELLO"

    def test_execute_image_to_text(self) -> None:
        """Un module IMAGE → RAW_TEXT lit le ``uri`` de l'Artifact."""
        registry = _PayloadRegistry()
        in_artifact = Artifact(
            id="doc1:__initial__:image",
            document_id="doc1",
            type=ArtifactType.IMAGE,
            uri="/tmp/test.png",
        )
        adapter = _BaseModuleAdapter(
            _ImageOCRModule("ocr-output"), registry,
        )
        outputs = adapter.execute(
            {ArtifactType.IMAGE: in_artifact},
            params={},
            context=self._make_context(),
        )
        # Le module a reçu un chemin str, pas un Artifact
        assert outputs[ArtifactType.RAW_TEXT].produced_by_step == "mock-ocr"
        # Le payload de sortie est bien dans le registre
        out_id = outputs[ArtifactType.RAW_TEXT].id
        assert registry.get(out_id) == "ocr-output"

    def test_chain_two_modules(self) -> None:
        """Sortie d'un step alimente l'input du suivant via le registre."""
        registry = _PayloadRegistry()
        ctx = self._make_context()

        # Step 1 : OCR
        ocr_adapter = _BaseModuleAdapter(_ImageOCRModule("hello"), registry)
        image_artifact = Artifact(
            id="doc1:__initial__:image",
            document_id="doc1",
            type=ArtifactType.IMAGE,
            uri="/tmp/x.png",
        )
        ocr_out = ocr_adapter.execute(
            {ArtifactType.IMAGE: image_artifact}, params={}, context=ctx,
        )
        # Step 2 : uppercase consume l'output de step 1
        upper_adapter = _BaseModuleAdapter(_UpperCaseTextModule(), registry)
        upper_out = upper_adapter.execute(
            {ArtifactType.RAW_TEXT: ocr_out[ArtifactType.RAW_TEXT]},
            params={}, context=ctx,
        )
        # Final payload : "HELLO"
        final_id = upper_out[ArtifactType.RAW_TEXT].id
        assert registry.get(final_id) == "HELLO"


# ──────────────────────────────────────────────────────────────────
# 3. wrap_initial_inputs
# ──────────────────────────────────────────────────────────────────


class TestWrapInitialInputs:
    def test_image_input_uri_set(self) -> None:
        registry = _PayloadRegistry()
        out = wrap_initial_inputs(
            {ArtifactType.IMAGE: "/tmp/x.png"},
            registry,
            document_id="doc1",
        )
        assert ArtifactType.IMAGE in out
        artifact = out[ArtifactType.IMAGE]
        assert artifact.uri == "/tmp/x.png"
        assert artifact.id == "doc1:__initial__:image"
        # IMAGE : pas de payload dans le registre (le module lit le uri)
        assert artifact.id not in registry

    def test_text_input_registered(self) -> None:
        registry = _PayloadRegistry()
        out = wrap_initial_inputs(
            {ArtifactType.RAW_TEXT: "hello"},
            registry,
            document_id="doc1",
        )
        artifact = out[ArtifactType.RAW_TEXT]
        # Le payload inline est dans le registre, pas dans uri
        assert artifact.uri is None
        assert registry.get(artifact.id) == "hello"
