"""Sprint A14-S54 — garde-fou MRO BaseVLMAdapter (fix audit #6).

Avant S54, l'ordre des parents dans :

    class AnthropicVLMAdapter(BaseVLMAdapter, AnthropicAdapter)

était critique mais non vérifié.  Un swap accidentel à
``(AnthropicAdapter, BaseVLMAdapter)`` aurait silencieusement donné
output_types = {CORRECTED_TEXT} (depuis LLM) au lieu de {RAW_TEXT}
(depuis VLM) — l'erreur ne se serait manifestée qu'au runtime sur
une jonction de type incompatible.

S54 ajoute ``__init_subclass__`` qui lève ``TypeError`` à la
définition de la classe si l'ordre est incorrect.
"""

from __future__ import annotations

import pytest

from picarones.adapters.llm.anthropic_adapter import AnthropicAdapter
from picarones.adapters.llm.openai_adapter import OpenAIAdapter
from picarones.adapters.vlm import (
    AnthropicVLMAdapter,
    BaseVLMAdapter,
    OpenAIVLMAdapter,
)
from picarones.domain.artifacts import ArtifactType


class TestExistingAdaptersStillValid:
    """Les 4 VLM adapters concrets définis correctement passent."""

    def test_anthropic_vlm_defined(self) -> None:
        # Si l'ordre était mauvais, l'import aurait planté.
        adapter = AnthropicVLMAdapter()
        assert adapter.input_types == frozenset({ArtifactType.IMAGE})
        assert adapter.output_types == frozenset({ArtifactType.RAW_TEXT})

    def test_openai_vlm_defined(self) -> None:
        adapter = OpenAIVLMAdapter()
        assert adapter.input_types == frozenset({ArtifactType.IMAGE})


class TestWrongOrderRejected:
    def test_llm_first_then_vlm_rejected(self) -> None:
        """Définir une classe avec LLM avant VLM doit lever TypeError."""
        with pytest.raises(TypeError, match="ordre MRO"):
            # Définition dynamique d'une classe avec mauvais ordre.
            type(
                "BadOrderVLM",
                (AnthropicAdapter, BaseVLMAdapter),
                {"name": property(lambda self: "bad")},
            )

    def test_correct_order_accepted(self) -> None:
        """L'ordre correct (VLM en premier) est accepté."""
        # Test propriété : aucun TypeError levé.
        type(
            "GoodOrderVLM",
            (BaseVLMAdapter, OpenAIAdapter),
            {"name": property(lambda self: "good")},
        )


class TestErrorMessageHelpful:
    def test_message_explains_the_fix(self) -> None:
        with pytest.raises(TypeError) as exc_info:
            type(
                "BadVLM",
                (AnthropicAdapter, BaseVLMAdapter),
                {"name": property(lambda self: "x")},
            )
        msg = str(exc_info.value)
        # Le message doit suggérer la correction concrète.
        assert "BaseVLMAdapter" in msg
        assert "AnthropicAdapter" in msg
        assert "Corrigez" in msg or "correct" in msg.lower()
