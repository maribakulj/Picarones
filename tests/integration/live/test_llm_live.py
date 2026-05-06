"""Tests live des 4 LLM adapters (skip si SDK + clé API absent).

Chaque test valide qu'un appel minimal ``complete(prompt, None)``
retourne du texte non-vide.  Pas d'assertion de qualité — on
détecte uniquement les régressions de schéma API / SDK.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.live
def test_anthropic_live() -> None:
    pytest.importorskip("anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY absent — skip live test")
    from picarones.adapters.llm import AnthropicAdapter
    adapter = AnthropicAdapter()
    result = adapter.complete(
        "Say 'OK' and nothing else.", image_b64=None,
    )
    assert result.success, f"Anthropic call failed: {result.error}"
    assert result.text


@pytest.mark.live
def test_openai_live() -> None:
    pytest.importorskip("openai")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY absent — skip live test")
    from picarones.adapters.llm import OpenAIAdapter
    adapter = OpenAIAdapter()
    result = adapter.complete(
        "Say 'OK' and nothing else.", image_b64=None,
    )
    assert result.success, f"OpenAI call failed: {result.error}"
    assert result.text


@pytest.mark.live
def test_mistral_live() -> None:
    pytest.importorskip("mistralai")
    if not os.environ.get("MISTRAL_API_KEY"):
        pytest.skip("MISTRAL_API_KEY absent — skip live test")
    from picarones.adapters.llm import MistralAdapter
    adapter = MistralAdapter()
    result = adapter.complete(
        "Say 'OK' and nothing else.", image_b64=None,
    )
    assert result.success, f"Mistral call failed: {result.error}"
    assert result.text


@pytest.mark.live
def test_ollama_live() -> None:
    """Ollama est local — skip si serveur indisponible."""
    pytest.importorskip("requests")
    import requests
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        r = requests.get(f"{base}/api/tags", timeout=2)
        if r.status_code != 200:
            pytest.skip(f"Ollama indisponible à {base}")
    except Exception:
        pytest.skip(f"Ollama indisponible à {base}")
    from picarones.adapters.llm import OllamaAdapter
    adapter = OllamaAdapter()
    result = adapter.complete(
        "Say 'OK' and nothing else.", image_b64=None,
    )
    # On ne réclame pas success — Ollama peut ne pas avoir le modèle
    # par défaut installé ; on vérifie juste que l'adapter ne plante
    # pas sur une cassure d'API.
    assert isinstance(result.text, str)
