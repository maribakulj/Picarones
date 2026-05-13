"""Sprint S8.7 — couverture réelle des factories de
``benchmark_utils.py`` (avant : 51.51% patch coverage).

Pourquoi ce fichier
-------------------
``_build_llm_adapter`` et ``_engine_from_competitor`` sont les
points de **routage** entre la config web (``PipelineConfig``)
et les adapters concrets : si une régression silencieusement
fait passer ``mistral`` au lieu de ``openai``, ou ``tesseract``
au lieu de ``mistral_ocr``, le benchmark tourne mais avec le
mauvais moteur — tests fonctionnels classiques ne le verraient
pas.

Pattern
-------
Les adapters LLM lazy-importent leurs SDK (cf. ``__init__``
sans ``import openai``), donc ``OpenAIAdapter()`` etc.
s'instancient sans erreur même hors environnement de prod —
on peut donc tester directement le routing sans mocker les SDK.

Pour les adapters OCR cloud (mistral_ocr, google_vision,
azure_doc_intel) qui exigent un SDK à l'import du wrapper,
on réutilise le pattern ``patch.dict(sys.modules, {... : None})``
de ``test_s8_factory_branches.py``.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from picarones.interfaces.web.benchmark_utils import (
    _build_llm_adapter,
    _engine_from_competitor,
    sse_format,
)
from picarones.interfaces.web.models import PipelineConfig


# ──────────────────────────────────────────────────────────────────────
# _build_llm_adapter — routing par provider
# ──────────────────────────────────────────────────────────────────────


class TestBuildLLMAdapterRouting:
    """Chaque provider de la config doit retourner exactement
    l'adapter correspondant — pas un autre, pas une instance
    fallback silencieuse."""

    @pytest.mark.parametrize(
        ("provider", "expected_class_name"),
        [
            ("openai", "OpenAIAdapter"),
            ("anthropic", "AnthropicAdapter"),
            ("mistral", "MistralAdapter"),
            ("ollama", "OllamaAdapter"),
        ],
    )
    def test_provider_routes_to_expected_adapter(
        self, provider: str, expected_class_name: str,
    ) -> None:
        comp = PipelineConfig(
            name="t", ocr_engine="", llm_provider=provider, llm_model="m",
        )
        adapter = _build_llm_adapter(comp)
        assert type(adapter).__name__ == expected_class_name, (
            f"provider={provider!r} doit instancier "
            f"{expected_class_name}, reçu {type(adapter).__name__}"
        )

    def test_unknown_provider_raises_value_error(self) -> None:
        comp = PipelineConfig(
            name="t", ocr_engine="",
            llm_provider="some_made_up_provider", llm_model="x",
        )
        with pytest.raises(ValueError, match="inconnu|unknown"):
            _build_llm_adapter(comp)

    def test_empty_llm_model_uses_adapter_default(self) -> None:
        """Quand ``llm_model`` est vide, on passe ``None`` à
        l'adapter (qui utilise son default interne) — pas une
        chaîne vide qui serait rejetée par l'API."""
        comp = PipelineConfig(
            name="t", ocr_engine="", llm_provider="openai", llm_model="",
        )
        adapter = _build_llm_adapter(comp)
        # L'adapter doit être instancié sans planter sur llm_model="".
        assert adapter is not None


# ──────────────────────────────────────────────────────────────────────
# _engine_from_competitor — routing OCR / pipeline / corpus-only
# ──────────────────────────────────────────────────────────────────────


class TestEngineFromCompetitorOCROnly:
    """OCR seul (pas de ``llm_provider``) → retourne un
    ``BaseOCRAdapter`` directement, prêt à être enregistré."""

    def test_tesseract_only_returns_adapter(self) -> None:
        """Le ``name`` est dérivé de ``(engine_id, ocr_model)`` pour
        que deux configs distinctes obtiennent automatiquement des
        identifiants différents au resolver (cf. S9 fix)."""
        comp = PipelineConfig(
            name="t", ocr_engine="tesseract", llm_provider="",
            ocr_model="fra",
        )
        engine = _engine_from_competitor(comp)
        assert engine.name == "tesseract_fra"

    def test_tesseract_only_different_lang_distinct_name(self) -> None:
        """Garantie anti-collision : ``lang=eng`` et ``lang=fra``
        produisent des ``name`` distincts au resolver."""
        comp_fra = PipelineConfig(
            ocr_engine="tesseract", llm_provider="", ocr_model="fra",
        )
        comp_eng = PipelineConfig(
            ocr_engine="tesseract", llm_provider="", ocr_model="eng",
        )
        assert _engine_from_competitor(comp_fra).name == "tesseract_fra"
        assert _engine_from_competitor(comp_eng).name == "tesseract_eng"

    def test_unknown_engine_raises_runtime_error(self) -> None:
        """``RuntimeError`` (et pas ``ValueError`` brut) — c'est le
        contrat documenté pour que le worker thread puisse
        loguer ``warning`` et passer au concurrent suivant."""
        comp = PipelineConfig(
            name="t", ocr_engine="not_an_engine", llm_provider="",
        )
        with pytest.raises(RuntimeError, match="inconnu"):
            _engine_from_competitor(comp)


class TestEngineFromCompetitorPipeline:
    """OCR + LLM → retourne un ``OCRLLMPipelineConfig`` (rewrite)
    avec le bon mode selon ``pipeline_mode``."""

    @pytest.mark.parametrize(
        ("pipeline_mode", "expected_mode"),
        [
            ("text_only", "text_only"),
            ("text_and_image", "text_and_image"),
        ],
    )
    def test_pipeline_mode_passes_through_with_ocr(
        self, pipeline_mode: str, expected_mode: str,
    ) -> None:
        """Modes canoniques qui exigent un OCR amont — Phase 2 du
        chantier post-rewrite : plus de mapping/alias.  Les 3 valeurs
        de :class:`PipelineMode` traversent telles quelles vers le
        ``OCRLLMPipelineConfig`` (``zero_shot`` testé séparément car
        il refuse l'OCR amont)."""
        comp = PipelineConfig(
            name="t", ocr_engine="tesseract", llm_provider="mistral",
            llm_model="m", ocr_model="fra", pipeline_mode=pipeline_mode,
        )
        pipeline = _engine_from_competitor(comp)
        assert pipeline.mode == expected_mode

    @pytest.mark.parametrize(
        "deprecated_mode",
        ["post_correction_text", "post_correction_image", "POST_CORRECTION_TEXT"],
    )
    def test_legacy_aliases_rejected_at_pydantic_level(
        self, deprecated_mode: str,
    ) -> None:
        """Phase 2 rupture API : les anciens alias
        (``post_correction_text``/``post_correction_image``) sont
        rejetés par Pydantic au niveau ``PipelineConfig`` — plus de
        mapping silencieux vers ``text_only`` / ``text_and_image``."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PipelineConfig(
                name="t", ocr_engine="tesseract", llm_provider="mistral",
                llm_model="m", ocr_model="fra",
                pipeline_mode=deprecated_mode,
            )

    def test_empty_pipeline_mode_with_llm_raises(self) -> None:
        """Phase 2 rupture API : un client qui combine ``llm_provider``
        non vide avec ``pipeline_mode=""`` reçoit désormais une
        ``ValueError`` claire — l'ancien fallback silencieux vers
        ``text_only`` masquait la config incomplète."""
        comp = PipelineConfig(
            name="t", ocr_engine="tesseract", llm_provider="mistral",
            llm_model="m", ocr_model="fra", pipeline_mode="",
        )
        with pytest.raises(ValueError, match="pipeline_mode invalide"):
            _engine_from_competitor(comp)

    def test_zero_shot_mode_requires_corpus_ocr(self) -> None:
        """Le mode ``zero_shot`` exige ``ocr_adapter=None`` au niveau
        du pipeline (le VLM lit l'image directement) — donc côté
        factory web, il doit être combiné avec ``ocr_engine=corpus``
        ou ``""``, pas avec un moteur live."""
        comp = PipelineConfig(
            name="t", ocr_engine="corpus", llm_provider="mistral",
            llm_model="m", pipeline_mode="zero_shot",
        )
        pipeline = _engine_from_competitor(comp)
        assert pipeline.mode == "zero_shot"
        assert pipeline.ocr_adapter is None

    def test_pipeline_name_from_explicit_name(self) -> None:
        comp = PipelineConfig(
            name="my-pipeline", ocr_engine="tesseract",
            llm_provider="mistral", llm_model="m", ocr_model="fra",
            pipeline_mode="text_only",
        )
        pipeline = _engine_from_competitor(comp)
        assert pipeline.pipeline_name == "my-pipeline"

    def test_pipeline_name_default_format(self) -> None:
        """Sans ``name`` explicite, format ``{engine} → {model}``."""
        comp = PipelineConfig(
            name="", ocr_engine="tesseract", llm_provider="mistral",
            llm_model="ministral-3b-latest", ocr_model="fra",
            pipeline_mode="text_only",
        )
        pipeline = _engine_from_competitor(comp)
        assert "tesseract" in pipeline.pipeline_name
        assert "ministral" in pipeline.pipeline_name

    def test_default_prompt_file_when_not_specified(self) -> None:
        """``prompt_file`` vide → chargement du contenu du prompt
        par défaut (``correction_medieval_french.txt``).  Cf. S9 :
        ``prompt_template`` contient désormais le CONTENU lu sur
        disque, pas le filename brut."""
        comp = PipelineConfig(
            name="t", ocr_engine="tesseract", llm_provider="mistral",
            llm_model="m", ocr_model="fra", prompt_file="",
            pipeline_mode="text_only",
        )
        pipeline = _engine_from_competitor(comp)
        # Le template ne doit PAS être le filename littéral.
        assert pipeline.prompt_template != "correction_medieval_french.txt"
        # Et doit être un vrai prompt substituable (placeholder
        # ``{ocr_output}`` ou ``{text}``).
        assert (
            "{ocr_output}" in pipeline.prompt_template
            or "{text}" in pipeline.prompt_template
        )


class TestEngineFromCompetitorCorpusOCR:
    """Mode ``corpus`` : utilise OCR pré-calculé (fichiers
    ``.ocr.txt``) au lieu d'un moteur live — exige un
    ``llm_provider`` car le pipeline a forcément besoin d'un
    LLM (post-correction ou zero-shot)."""

    @pytest.mark.parametrize("ocr_engine", ["corpus", ""])
    def test_corpus_or_empty_without_llm_raises(
        self, ocr_engine: str,
    ) -> None:
        comp = PipelineConfig(
            name="t", ocr_engine=ocr_engine, llm_provider="",
        )
        with pytest.raises(ValueError, match="llm_provider"):
            _engine_from_competitor(comp)

    @pytest.mark.parametrize("ocr_engine", ["corpus", ""])
    def test_corpus_with_llm_returns_pipeline(
        self, ocr_engine: str,
    ) -> None:
        """Mode corpus + LLM → pipeline ``zero_shot`` (le LLM/VLM
        traite l'image ou l'OCR pré-calculé, l'``ocr_adapter`` est
        ``None``)."""
        comp = PipelineConfig(
            name="post-corr", ocr_engine=ocr_engine,
            llm_provider="mistral", llm_model="m",
            pipeline_mode="zero_shot",
        )
        pipeline = _engine_from_competitor(comp)
        assert pipeline.ocr_adapter is None, (
            "en mode corpus, l'OCR adapter doit être None — "
            "le pipeline lit l'OCR pré-calculé du corpus."
        )
        assert pipeline.llm_adapter is not None

    def test_corpus_pipeline_name_format(self) -> None:
        """Sans ``name``, format ``corpus_ocr → {model}``."""
        comp = PipelineConfig(
            name="", ocr_engine="corpus", llm_provider="mistral",
            llm_model="ministral-3b-latest",
            pipeline_mode="zero_shot",
        )
        pipeline = _engine_from_competitor(comp)
        assert "corpus_ocr" in pipeline.pipeline_name
        assert "ministral" in pipeline.pipeline_name


class TestEngineFromCompetitorCloudWithoutSDK:
    """Pour les adapters OCR cloud, le wrapper module est
    importé conditionnellement — un SDK absent doit être
    transformé en ``RuntimeError`` propre côté factory web."""

    @pytest.mark.parametrize(
        ("engine", "module_path"),
        [
            ("mistral_ocr", "picarones.adapters.ocr.mistral_ocr"),
            ("google_vision", "picarones.adapters.ocr.google_vision"),
            ("azure_doc_intel", "picarones.adapters.ocr.azure_doc_intel"),
        ],
    )
    def test_cloud_engine_without_sdk_runtime_error(
        self, engine: str, module_path: str,
    ) -> None:
        comp = PipelineConfig(
            name="t", ocr_engine=engine, llm_provider="",
        )
        with patch.dict(sys.modules, {module_path: None}):
            with pytest.raises(RuntimeError, match="indisponible"):
                _engine_from_competitor(comp)


# ──────────────────────────────────────────────────────────────────────
# sse_format — sérialisation Server-Sent Events
# ──────────────────────────────────────────────────────────────────────


class TestSSEFormat:
    """Le format SSE doit respecter la spec WHATWG : ``id:`` (si
    seq fourni), ``event:``, ``data:``, double newline final."""

    def test_basic_event_no_seq(self) -> None:
        out = sse_format("log", {"message": "hello"})
        assert "event: log\n" in out
        # ``json.dumps`` par défaut → séparateurs avec espace.
        assert '"message": "hello"' in out
        assert out.endswith("\n\n")
        assert not out.startswith("id:")

    def test_event_with_seq(self) -> None:
        out = sse_format("progress", {"pct": 0.5}, seq=42)
        assert out.startswith("id: 42\n")
        assert "event: progress\n" in out

    def test_unicode_preserved(self) -> None:
        """``ensure_ascii=False`` — les accents passent en clair."""
        out = sse_format("log", {"message": "événement"})
        assert "événement" in out

    def test_seq_zero_not_skipped(self) -> None:
        """``seq=0`` est valide (premier événement) — ne doit pas
        être traité comme None."""
        out = sse_format("start", {}, seq=0)
        assert out.startswith("id: 0\n")
