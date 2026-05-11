"""Sprint S8.7 — couverture des fonctions utilitaires de
``picarones/adapters/llm/base.py``.

Cible (avant) : 94% avec 8 lignes manquantes (68, 72, 160,
353-358, 447, 488).  Toutes représentent des contrats fonctionnels
réels accessibles directement (pas de mock SDK requis) :

- ``normalize_llm_content`` fallbacks sur formats inconnus ;
- ``_substitute_prompt_variables`` détection rewrite/legacy ;
- ``BaseLLMAdapter.__repr__`` format stable pour debug/logs ;
- ``DEFAULT_CORRECTION_PROMPTS`` warning sur langue non supportée.
"""

from __future__ import annotations

import pytest

from picarones.adapters.llm.base import (
    _substitute_prompt_variables,
    normalize_llm_content,
)


class TestNormalizeLLMContentFallbacks:
    """Le helper doit absorber tous les formats de contenu LLM
    (str, list de str/dict/objet, objet avec ``.text``, valeurs
    inattendues) sans planter — les SDK retournent des structures
    variées selon le provider et la version."""

    def test_none_returns_empty_string(self) -> None:
        assert normalize_llm_content(None) == ""

    def test_plain_string_passthrough(self) -> None:
        assert normalize_llm_content("hello") == "hello"

    def test_list_of_strings_concatenated(self) -> None:
        assert normalize_llm_content(["a", "b", "c"]) == "abc"

    def test_list_with_none_chunks_skipped(self) -> None:
        assert normalize_llm_content(["a", None, "b"]) == "ab"

    def test_list_with_text_attribute_objects(self) -> None:
        class Chunk:
            def __init__(self, text: str) -> None:
                self.text = text

        result = normalize_llm_content([Chunk("hello"), Chunk(" world")])
        assert result == "hello world"

    def test_list_with_dict_text_key(self) -> None:
        result = normalize_llm_content([{"text": "x"}, {"text": "y"}])
        assert result == "xy"

    def test_list_with_unknown_chunk_falls_back_to_str(self) -> None:
        """Couvre la ligne 68 — chunk sans ``.text`` ni dict
        ``{"text": ...}`` → ``str(chunk)`` en dernier recours."""

        class Opaque:
            def __str__(self) -> str:
                return "opaque-content"

        result = normalize_llm_content([Opaque()])
        assert result == "opaque-content"

    def test_object_with_text_attribute_returns_text(self) -> None:
        class Reply:
            text = "single response"

        assert normalize_llm_content(Reply()) == "single response"

    def test_unknown_object_falls_back_to_str(self) -> None:
        """Couvre la ligne 72 — objet sans ``.text`` ni list/str
        → ``str(raw)`` en dernier recours."""

        class Opaque:
            def __str__(self) -> str:
                return "opaque"

        assert normalize_llm_content(Opaque()) == "opaque"


class TestSubstitutePromptVariables:
    """Le helper doit détecter automatiquement la convention de
    variables (rewrite ``{text}`` ou legacy
    ``{ocr_output}``/``{image_b64}``) et router vers la bonne
    stratégie de substitution."""

    def test_rewrite_format_substitutes_text(self) -> None:
        """Couvre la ligne 165 (``return template.format(text=text)``).
        Convention rewrite Sprint A14-S44."""
        result = _substitute_prompt_variables(
            "Corriger : {text}", text="hello", image_b64=None,
        )
        assert result == "Corriger : hello"

    def test_legacy_format_substitutes_ocr_output(self) -> None:
        """Convention legacy ``OCRLLMPipeline``."""
        result = _substitute_prompt_variables(
            "OCR result: {ocr_output}", text="raw", image_b64=None,
        )
        assert result == "OCR result: raw"

    def test_legacy_format_with_image_b64(self) -> None:
        result = _substitute_prompt_variables(
            "Text {ocr_output} Image {image_b64}",
            text="t", image_b64="iVBOR...",
        )
        assert result == "Text t Image iVBOR..."

    def test_legacy_image_b64_none_becomes_empty(self) -> None:
        """``image_b64=None`` → chaîne vide (mode texte-seul)."""
        result = _substitute_prompt_variables(
            "{ocr_output}|{image_b64}", text="x", image_b64=None,
        )
        assert result == "x|"

    def test_template_without_known_placeholder_raises(self) -> None:
        """Sprint S9 — la défense au niveau substitution refuse
        tout template qui n'a aucun placeholder connu
        (``{ocr_output}``, ``{text}``, ``{image_b64}``).  Avant
        S9, ce test attendait un ``KeyError`` de ``str.format`` ;
        le contrat est maintenant strict en amont : pas de
        placeholder = template invalide."""
        with pytest.raises(ValueError, match="placeholder|filename"):
            _substitute_prompt_variables(
                "{unknown_var}", text="x", image_b64=None,
            )

    def test_legacy_detection_takes_precedence(self) -> None:
        """Si le template contient AUSSI ``{text}``, la détection
        legacy l'emporte (présence d'``{ocr_output}`` détectée
        d'abord) — ``{text}`` reste littéral."""
        result = _substitute_prompt_variables(
            "{ocr_output} et {text}", text="OCR", image_b64=None,
        )
        # ``{text}`` n'est PAS substitué en mode legacy.
        assert result == "OCR et {text}"


class TestBaseLLMAdapterRepr:
    """``__repr__`` stable pour les logs et le debug."""

    def test_repr_includes_class_and_model(self) -> None:
        """Couvre la ligne 488."""
        from picarones.adapters.llm.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(model="gpt-4o")
        repr_str = repr(adapter)
        assert "OpenAIAdapter" in repr_str
        assert "gpt-4o" in repr_str


#
# Pas couvert ici : ligne 447 (``lang non supportée → fallback FR``)
# est dans ``BaseLLMAdapter.execute`` après plusieurs niveaux de
# priorité (param_prompt → custom_prompt → langue).  La déclencher
# proprement exige de construire un RunContext + Artifact OCR + de
# mocker l'appel LLM lui-même — coût nettement supérieur à
# ``adapter._effective_correction_prompt()``.  Tests dédiés
# d'``execute`` au sprint A.2 (``test_sprint_a2_*``).
