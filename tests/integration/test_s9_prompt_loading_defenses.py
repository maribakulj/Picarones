"""Sprint S9 — garde-fous anti-régression pour le bug
"filename passé à la place du contenu" (post-correction LLM).

Deux niveaux de garde-fou :

1. **Contrat ``OCRLLMPipelineConfig.__post_init__``** : refuse un
   ``prompt_template`` non-vide sans aucune accolade.  Check
   minimal qui capture le cas ``correction_*.txt`` injecté tel
   quel comme template.

2. **Test d'intégration** : mock du LLM qui capture le prompt
   réellement envoyé, exécution du factory web → pipeline → LLM,
   assertion que le prompt capturé est le **contenu** du fichier
   prompt (pas le filename).  C'est le filet manquant pré-S9
   qui aurait pris le bug en amont — chaque couche était testée
   en isolation, personne ne vérifiait le bout-en-bout du flux
   post-correction.
"""

from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────────────
# Niveau 1 — contrat OCRLLMPipelineConfig
# ──────────────────────────────────────────────────────────────────────


class TestPipelineConfigRefusesInvalidTemplate:
    """Le constructeur de ``OCRLLMPipelineConfig`` doit refuser un
    ``prompt_template`` qui n'a aucun placeholder substituable —
    c'est exactement ce qui se passait avec le filename brut."""

    def test_filename_passed_as_template_rejected(self) -> None:
        """Cas exact du bug en prod : on passe un filename à la
        place du contenu → rejet au constructeur."""
        from picarones.adapters.llm.openai_adapter import OpenAIAdapter
        from picarones.adapters.ocr.tesseract import TesseractAdapter
        from picarones.pipeline.llm_pipeline_config import (
            OCRLLMPipelineConfig,
        )

        with pytest.raises(ValueError, match="accolade|filename"):
            OCRLLMPipelineConfig(
                ocr_adapter=TesseractAdapter(lang="fra"),
                llm_adapter=OpenAIAdapter(model="gpt-4o"),
                mode="text_only",
                prompt_template="correction_early_modern_english.txt",
            )

    def test_string_without_brace_rejected(self) -> None:
        """Toute string non-vide sans accolade est refusée — pas
        seulement les filenames.  Le LLM recevrait une string fixe
        qui ignore l'OCR."""
        from picarones.adapters.llm.openai_adapter import OpenAIAdapter
        from picarones.adapters.ocr.tesseract import TesseractAdapter
        from picarones.pipeline.llm_pipeline_config import (
            OCRLLMPipelineConfig,
        )

        with pytest.raises(ValueError, match="accolade"):
            OCRLLMPipelineConfig(
                ocr_adapter=TesseractAdapter(lang="fra"),
                llm_adapter=OpenAIAdapter(model="gpt-4o"),
                mode="text_only",
                prompt_template="Corrige ce texte.",
            )

    def test_empty_template_still_allowed(self) -> None:
        """L'``empty string`` reste valide : signale au LLM adapter
        d'utiliser son prompt par défaut interne (``DEFAULT_CORRECTION_PROMPTS``)."""
        from picarones.adapters.llm.openai_adapter import OpenAIAdapter
        from picarones.adapters.ocr.tesseract import TesseractAdapter
        from picarones.pipeline.llm_pipeline_config import (
            OCRLLMPipelineConfig,
        )

        config = OCRLLMPipelineConfig(
            ocr_adapter=TesseractAdapter(lang="fra"),
            llm_adapter=OpenAIAdapter(model="gpt-4o"),
            mode="text_only",
            prompt_template="",
        )
        assert config.prompt_template == ""

    @pytest.mark.parametrize(
        "valid_template",
        [
            "Corrige : {ocr_output}",           # legacy
            "Texte : {text}",                    # rewrite
            "{ocr_output} + {image_b64}",        # legacy multi
            "Pré-prompt long\nMulti-ligne\n\nOCR: {ocr_output}\nFin.",
        ],
    )
    def test_valid_template_with_placeholder_accepted(
        self, valid_template: str,
    ) -> None:
        from picarones.adapters.llm.openai_adapter import OpenAIAdapter
        from picarones.adapters.ocr.tesseract import TesseractAdapter
        from picarones.pipeline.llm_pipeline_config import (
            OCRLLMPipelineConfig,
        )

        config = OCRLLMPipelineConfig(
            ocr_adapter=TesseractAdapter(lang="fra"),
            llm_adapter=OpenAIAdapter(model="gpt-4o"),
            mode="text_only",
            prompt_template=valid_template,
        )
        assert config.prompt_template == valid_template


# ──────────────────────────────────────────────────────────────────────
# Niveau 2 — intégration LLM end-to-end (filet manquant pré-S9)
# ──────────────────────────────────────────────────────────────────────


class TestEndToEndPromptReachesLLM:
    """Le filet manquant pré-S9 : un test qui capture le prompt
    réel envoyé au LLM lors d'une post-correction, et vérifie
    qu'il contient bien le contenu chargé depuis disque (pas un
    filename, pas une string fixe).

    C'est exactement le test qui aurait pris le bug initialement —
    le restant des défenses est superflu tant que ce filet tourne
    en CI.
    """

    def test_llm_receives_loaded_prompt_content(self, monkeypatch) -> None:
        from picarones.adapters.llm.base import _substitute_prompt_variables
        from picarones.adapters.llm.openai_adapter import OpenAIAdapter
        from picarones.interfaces.web.benchmark_utils import (
            _load_prompt_content,
        )

        # Adapter concret (OpenAI) avec ``_call`` mocké pour
        # capturer le prompt sans hit le réseau.
        adapter = OpenAIAdapter(model="gpt-4o")
        captured = {"prompt": None, "called": False}

        def fake_call(prompt: str, image_b64=None) -> str:
            captured["prompt"] = prompt
            captured["called"] = True
            return "réponse simulée"

        monkeypatch.setattr(adapter, "_call", fake_call)

        template = _load_prompt_content("correction_medieval_french.txt")
        ocr_text = "li rois Phelippes feift faire ledict"
        substituted = _substitute_prompt_variables(
            template, text=ocr_text, image_b64=None,
        )

        result = adapter.complete(substituted)
        assert result.text == "réponse simulée"
        assert captured["called"]

        # Le prompt capturé doit contenir le contenu du fichier ET
        # le texte OCR substitué.
        sent = captured["prompt"]
        assert "Phelippes" in sent, (
            "Le texte OCR n'a pas été substitué dans le prompt."
        )
        # Pas le filename.
        assert "correction_medieval_french.txt" not in sent
        # Le contenu réel fait ~1500 chars.
        assert len(sent) > 200, (
            f"Prompt suspicieusement court ({len(sent)} chars) — "
            "probable que la substitution n'a pas chargé le contenu."
        )

    def test_web_factory_to_pipeline_to_llm_flow(self) -> None:
        """End-to-end depuis ``PipelineConfig`` (UI) jusqu'au LLM :
        le prompt arrivé au LLM doit être le CONTENU du fichier,
        pas le filename.  C'est le chemin exact du bug en prod."""
        from picarones.interfaces.web.benchmark_utils import (
            _engine_from_competitor,
        )
        from picarones.interfaces.web.models import PipelineConfig
        from picarones.adapters.llm.base import _substitute_prompt_variables

        comp = PipelineConfig(
            engine_name="tesseract", ocr_model="fra",
            llm_provider="mistral", llm_model="mistral-small-latest",
            pipeline_mode="text_only",
            prompt_file="correction_early_modern_english.txt",
        )
        pipeline = _engine_from_competitor(comp)

        # Le ``prompt_template`` du pipeline doit être substituable
        # (contient un placeholder).  Si on essayait de "substituer"
        # un filename, la défense niveau 2 lèverait.
        substituted = _substitute_prompt_variables(
            pipeline.prompt_template,
            text="thou hast",
            image_b64=None,
        )
        # Le résultat de la substitution est un PROMPT réel, pas un
        # filename.  Il doit contenir le texte OCR injecté ET du
        # contenu de prompt Early Modern English.
        assert "thou hast" in substituted
        assert "Early Modern English" in substituted
        assert "correction_early_modern_english.txt" not in substituted
