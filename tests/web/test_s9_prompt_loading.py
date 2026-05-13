"""Sprint S9 — régression critique de la post-correction LLM.

Bug observé en prod (interface web, 2026-05-11) : un benchmark
avec post-correction LLM produisait, au lieu d'un texte OCR
corrigé, un **méta-discours** du LLM décrivant le contenu
hypothétique du fichier prompt :

    It looks like you're referring to a file named
    "correction_early_modern_english.txt"...

Cause racine
------------
``_engine_from_competitor`` (couche web) passait à
``OCRLLMPipelineConfig.prompt_template`` le **nom du fichier**
(``"correction_early_modern_english.txt"``) au lieu de son
**contenu**.  Le pipeline canonique n'a pas de logique de
chargement disque — il s'attend à recevoir une string brute.

Avant le sprint H.2.c-d (mai 2026), l'``OCRLLMPipeline`` legacy
lisait elle-même le fichier depuis ``picarones/prompts/``.  Au
moment de la migration, ce chargement n'a pas été reporté côté
factory web.

Symptôme côté LLM : le template substitué via
``_substitute_prompt_variables(template, text, image_b64)``
ne contenait ni ``{ocr_output}`` ni ``{image_b64}`` ni
``{text}`` — donc ``template.format(text=text)`` retournait la
chaîne ``"correction_early_modern_english.txt"`` inchangée.
Le LLM recevait ce filename comme prompt complet → réponse
méta-discursive.

Le fix charge le contenu du prompt côté ``_engine_from_competitor``
via le helper ``_load_prompt_content``.
"""

from __future__ import annotations

import pytest

from picarones.interfaces.web.benchmark_utils import (
    _engine_from_competitor,
    _load_prompt_content,
)
from picarones.interfaces.web.models import PipelineConfig


class TestLoadPromptContent:
    """Le helper doit retourner le CONTENU du fichier prompt
    embarqué, pas le filename."""

    def test_loads_real_content_not_filename(self) -> None:
        content = _load_prompt_content("correction_medieval_french.txt")
        assert content != "correction_medieval_french.txt"
        assert len(content) > 100, (
            "Le prompt embarqué fait plusieurs centaines de chars — "
            "si on en lit < 100 c'est probablement un fichier vide ou "
            "le filename brut."
        )

    @pytest.mark.parametrize(
        "filename",
        [
            "correction_medieval_french.txt",
            "correction_medieval_english.txt",
            "correction_early_modern_english.txt",
            "correction_imprime_ancien.txt",
            "correction_image_medieval_french.txt",
            "zero_shot_medieval_french.txt",
            "zero_shot_medieval_english.txt",
            "zero_shot_imprime_ancien.txt",
        ],
    )
    def test_all_embedded_prompts_loadable(self, filename: str) -> None:
        """Tous les prompts livrés avec le package doivent être
        chargeables — garde-fou contre la suppression accidentelle
        d'un fichier référencé par défaut."""
        content = _load_prompt_content(filename)
        assert content
        # Heuristique : un vrai prompt LLM contient soit
        # ``{ocr_output}`` (convention legacy) soit ``{text}``
        # (convention rewrite) — pas le nom du fichier.
        assert (
            "{ocr_output}" in content
            or "{text}" in content
            or "{image_b64}" in content
        ), (
            f"Prompt {filename!r} n'a pas de placeholder variable — "
            "il ne pourra pas être substitué par le LLM, c'est un "
            "fichier inerte."
        )

    def test_traversal_attempt_rejected(self) -> None:
        """Le loader refuse de remonter hors du dossier prompts —
        défense en profondeur contre un caller qui aurait court-
        circuité ``validated_prompt_filename``."""
        with pytest.raises(ValueError, match="hors de la bibliothèque"):
            _load_prompt_content("../../../etc/passwd")

    def test_unknown_filename_raises_with_listing(self) -> None:
        """Filename inconnu → ``FileNotFoundError`` avec la liste
        des fichiers disponibles, utile pour debug ops."""
        with pytest.raises(FileNotFoundError) as exc_info:
            _load_prompt_content("never_existed.txt")
        msg = str(exc_info.value)
        # La liste des fichiers réels doit apparaître pour guider
        # l'utilisateur.
        assert "correction_medieval_french.txt" in msg


class TestEngineFromCompetitorPassesPromptContent:
    """Régression : ``_engine_from_competitor`` doit injecter le
    CONTENU du prompt dans ``OCRLLMPipelineConfig.prompt_template``,
    pas le filename brut."""

    def test_pipeline_template_contains_file_content(self) -> None:
        comp = PipelineConfig(
            name="t",
            ocr_engine="tesseract",
            ocr_model="fra",
            llm_provider="mistral",
            llm_model="mistral-small-latest",
            pipeline_mode="text_only",
            prompt_file="correction_early_modern_english.txt",
        )
        pipeline = _engine_from_competitor(comp)

        # Le contenu réel commence par "You are an expert" (vérifié
        # dans le fichier embarqué).
        assert pipeline.prompt_template != "correction_early_modern_english.txt"
        assert len(pipeline.prompt_template) > 100
        assert "Early Modern English" in pipeline.prompt_template

    def test_default_prompt_loaded_when_none_specified(self) -> None:
        """``prompt_file`` vide → default
        ``correction_medieval_french.txt`` chargé."""
        comp = PipelineConfig(
            ocr_engine="tesseract", ocr_model="fra",
            llm_provider="mistral", llm_model="m",
            pipeline_mode="text_only", prompt_file="",
        )
        pipeline = _engine_from_competitor(comp)
        assert pipeline.prompt_template != "correction_medieval_french.txt"
        assert "{ocr_output}" in pipeline.prompt_template or "{text}" in pipeline.prompt_template

    def test_unknown_prompt_file_raises(self) -> None:
        """Si le frontend envoie un filename qui n'existe pas, le
        factory doit lever proprement (pas continuer avec le filename
        comme prompt — c'est le bug d'origine)."""
        comp = PipelineConfig(
            ocr_engine="tesseract", ocr_model="fra",
            llm_provider="mistral", llm_model="m",
            pipeline_mode="text_only",
            prompt_file="prompt_que_personne_na_jamais_cree.txt",
        )
        with pytest.raises(FileNotFoundError):
            _engine_from_competitor(comp)
