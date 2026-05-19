"""Sprint S4.8 — couverture des 4 adapters VLM.

Avant S4 : ``adapters/vlm/{anthropic,mistral,ollama,openai}_vlm.py``
à 0% direct (testés transitivement).

Cible : 80%+ — vérifie le contrat MRO + ``input_types`` /
``output_types`` + ``name`` propre à chaque adapter, sans appeler
les SDK réels (qui exigeraient des clés API et du réseau).
"""

from __future__ import annotations

import pytest

from picarones.domain.artifacts import ArtifactType


# ──────────────────────────────────────────────────────────────────────
# Liste des adapters à tester avec leur identifiant attendu
# ──────────────────────────────────────────────────────────────────────


_VLM_CASES = [
    ("anthropic_vlm", "picarones.adapters.vlm.anthropic_vlm",
     "AnthropicVLMAdapter"),
    ("mistral_vlm", "picarones.adapters.vlm.mistral_vlm",
     "MistralVLMAdapter"),
    ("ollama_vlm", "picarones.adapters.vlm.ollama_vlm",
     "OllamaVLMAdapter"),
    ("openai_vlm", "picarones.adapters.vlm.openai_vlm",
     "OpenAIVLMAdapter"),
]


# ──────────────────────────────────────────────────────────────────────
# 1. Contrat de base : input/output types, name, MRO
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "expected_name,module_path,class_name", _VLM_CASES,
)
class TestVLMAdapterContract:
    def test_input_types_is_image(
        self, expected_name: str, module_path: str, class_name: str,
    ) -> None:
        import importlib

        module = importlib.import_module(module_path)
        adapter_cls = getattr(module, class_name)
        adapter = adapter_cls(model="any-model", config={})

        assert ArtifactType.IMAGE in adapter.input_types

    def test_output_types_is_raw_text(
        self, expected_name: str, module_path: str, class_name: str,
    ) -> None:
        import importlib

        module = importlib.import_module(module_path)
        adapter_cls = getattr(module, class_name)
        adapter = adapter_cls(model="any-model", config={})

        assert ArtifactType.RAW_TEXT in adapter.output_types

    def test_name_is_distinct_per_adapter(
        self, expected_name: str, module_path: str, class_name: str,
    ) -> None:
        import importlib

        module = importlib.import_module(module_path)
        adapter_cls = getattr(module, class_name)
        adapter = adapter_cls(model="any-model", config={})

        assert adapter.name == expected_name

    def test_mro_baseVLMAdapter_first(
        self, expected_name: str, module_path: str, class_name: str,
    ) -> None:
        """Le garde-fou ``__init_subclass__`` exige
        ``BaseVLMAdapter`` AVANT le LLM sibling dans le MRO.  On
        vérifie qu'une instance correctement définie a bien
        ``BaseVLMAdapter`` parmi ses ancêtres et que ``input_types``
        vient bien de lui (et pas du LLM)."""
        import importlib

        from picarones.adapters.vlm.base import BaseVLMAdapter

        module = importlib.import_module(module_path)
        adapter_cls = getattr(module, class_name)
        assert issubclass(adapter_cls, BaseVLMAdapter)
        # MRO : BaseVLMAdapter doit venir avant BaseLLMAdapter
        # (à travers la chaîne d'héritage, on vérifie indirectement
        # que ``input_types`` est l'IMAGE ; déjà testé plus haut).


# ──────────────────────────────────────────────────────────────────────
# 2. Transcription prompt configurable
# ──────────────────────────────────────────────────────────────────────


class TestTranscriptionPromptConfigurable:
    def test_custom_prompt_via_config(self) -> None:
        from picarones.adapters.vlm.openai_vlm import OpenAIVLMAdapter

        adapter = OpenAIVLMAdapter(
            model="gpt-4o",
            config={"transcription_prompt": "Custom prompt for testing."},
        )
        # Doit pouvoir instancier sans erreur ; le prompt est consommé
        # par ``execute``.
        assert adapter.name == "openai_vlm"

    def test_default_prompt_used_when_none_provided(self) -> None:
        from picarones.adapters.vlm.openai_vlm import OpenAIVLMAdapter

        adapter = OpenAIVLMAdapter(model="gpt-4o", config={})
        # Pas de plantage à l'init — le défaut est utilisé.
        assert adapter is not None


# ──────────────────────────────────────────────────────────────────────
# 3. MRO guard — ordre incorrect → TypeError
# ──────────────────────────────────────────────────────────────────────


class TestMROGuardRaisesOnSwap:
    """Le garde-fou ``__init_subclass__`` doit lever ``TypeError``
    quand on déclare le LLM sibling AVANT ``BaseVLMAdapter``.

    Reproduction du bug que le garde protège : si l'ordre est
    inversé, ``input_types`` viendrait du LLM (= RAW_TEXT) au
    lieu de IMAGE, et le pipeline silencieusement passerait du
    texte au VLM."""

    def test_swapped_parents_raises_typeerror(self) -> None:
        from picarones.adapters.llm.openai_adapter import OpenAIAdapter
        from picarones.adapters.vlm.base import BaseVLMAdapter

        with pytest.raises(TypeError):
            # Ordre INVERSE — BaseVLMAdapter en deuxième.
            class _BadVLM(OpenAIAdapter, BaseVLMAdapter):  # type: ignore[misc]
                @property
                def name(self) -> str:
                    return "bad"
