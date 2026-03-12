"""Pipeline OCR+LLM — présenté comme un concurrent normal dans les benchmarks.

Un pipeline compose un moteur OCR et un LLM de correction selon trois modes :

  text_only      → OCR brut ──► LLM (texte seul)
  text_and_image → OCR brut + image ──► LLM multimodal
  zero_shot      → image ──► LLM (pas d'OCR amont)

La classe ``OCRLLMPipeline`` étend ``BaseOCREngine`` : un pipeline est
un concurrent comme un autre dans ``run_benchmark``, avec les mêmes métriques
CER/WER. Les métadonnées spécifiques (étapes, prompt, OCR intermédiaire) sont
exposées via ``EngineResult.metadata``.
"""

from __future__ import annotations

import base64
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from picarones.engines.base import BaseOCREngine, EngineResult
from picarones.llm.base import BaseLLMAdapter

logger = logging.getLogger(__name__)


class PipelineMode(str, Enum):
    """Mode d'appel LLM dans le pipeline."""

    TEXT_ONLY = "text_only"
    """Le LLM reçoit uniquement le texte OCR brut."""

    TEXT_AND_IMAGE = "text_and_image"
    """Le LLM reçoit le texte OCR ET l'image (mode multimodal)."""

    ZERO_SHOT = "zero_shot"
    """Le LLM reçoit uniquement l'image — aucun OCR amont."""


# Répertoire de la bibliothèque de prompts intégrée
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(prompt_path: str | Path) -> str:
    """Charge un prompt depuis un chemin absolu, relatif ou depuis la bibliothèque intégrée."""
    p = Path(prompt_path)
    if p.is_absolute() and p.exists():
        return p.read_text(encoding="utf-8")
    # Chemin relatif : chercher d'abord dans le CWD, puis dans la bibliothèque
    if p.exists():
        return p.read_text(encoding="utf-8")
    builtin = _PROMPTS_DIR / p
    if builtin.exists():
        return builtin.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Prompt introuvable : '{prompt_path}'. "
        f"Bibliothèque disponible dans : {_PROMPTS_DIR}"
    )


def _image_to_b64(image_path: Path) -> str:
    """Encode une image en base64 pur (sans préfixe data URI)."""
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


class OCRLLMPipeline(BaseOCREngine):
    """Pipeline OCR+LLM, interchangeable avec n'importe quel moteur OCR.

    Parameters
    ----------
    llm_adapter:
        Adaptateur LLM (OpenAI, Anthropic, Mistral, Ollama…).
    mode:
        Mode de correction — text_only, text_and_image, ou zero_shot.
    prompt:
        Chemin vers un fichier .txt de prompt, ou nom d'un fichier de la
        bibliothèque intégrée (ex : ``"correction_medieval_french.txt"``).
        Variables disponibles dans le fichier : ``{ocr_output}`` et ``{image_b64}``.
    ocr_engine:
        Moteur OCR amont. Obligatoire pour text_only et text_and_image.
        Non utilisé en mode zero_shot.
    pipeline_name:
        Nom affiché dans le rapport (ex : ``"tesseract → gpt-4o"``).
        Généré automatiquement si non fourni.
    config:
        Paramètres supplémentaires passés à la classe de base.

    Examples
    --------
    >>> from picarones.llm import OpenAIAdapter
    >>> from picarones.engines.tesseract import TesseractEngine
    >>> pipeline = OCRLLMPipeline(
    ...     ocr_engine=TesseractEngine({"lang": "fra"}),
    ...     llm_adapter=OpenAIAdapter(model="gpt-4o"),
    ...     mode=PipelineMode.TEXT_AND_IMAGE,
    ...     prompt="correction_medieval_french.txt",
    ... )
    """

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        mode: PipelineMode | str = PipelineMode.TEXT_ONLY,
        prompt: str | Path = "correction_medieval_french.txt",
        ocr_engine: Optional[BaseOCREngine] = None,
        pipeline_name: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(config)
        self.ocr_engine = ocr_engine
        self.llm_adapter = llm_adapter
        self.mode = PipelineMode(mode)
        self.prompt_path = str(prompt)
        self._prompt_template = _load_prompt(prompt)

        # Nom affiché dans le rapport
        if pipeline_name:
            self._name = pipeline_name
        elif self.mode == PipelineMode.ZERO_SHOT:
            self._name = f"{llm_adapter.model} (zero-shot)"
        elif ocr_engine:
            self._name = f"{ocr_engine.name} → {llm_adapter.model}"
        else:
            self._name = f"pipeline → {llm_adapter.model}"


    # ------------------------------------------------------------------
    # Interface BaseOCREngine
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    def version(self) -> str:
        ocr_v = self.ocr_engine._safe_version() if self.ocr_engine else "—"
        return f"ocr={ocr_v}; llm={self.llm_adapter.model}"

    def _run_ocr(self, image_path: Path) -> tuple[str, Optional[str]]:
        """Logique interne du pipeline — appelée par ``run()``.

        Returns
        -------
        tuple[str, Optional[str]]
            (llm_text, ocr_intermediate) — ocr_intermediate est None en mode zero_shot.
        """
        ocr_text = ""

        if self.mode == PipelineMode.ZERO_SHOT:
            image_b64 = _image_to_b64(image_path)
            prompt = self._build_prompt(image_b64=image_b64)
            logger.debug(
                "[%s] zero-shot — longueur prompt : %d car.", self._name, len(prompt)
            )
            logger.info("[Pipeline] appel LLM pour doc %s (zero-shot)", image_path.name)
            result = self.llm_adapter.complete(prompt, image_b64=image_b64)
            logger.info("[Pipeline] LLM retourné pour doc %s", image_path.name)

        elif self.mode == PipelineMode.TEXT_ONLY:
            if self.ocr_engine is None:
                raise ValueError("ocr_engine est requis pour le mode text_only")
            ocr_result = self.ocr_engine.run(image_path)
            ocr_text = ocr_result.text
            logger.debug(
                "[%s] texte OCR : %d car. → envoi au LLM.",
                self._name, len(ocr_text),
            )
            if not ocr_text.strip():
                logger.warning(
                    "[%s] le moteur OCR a produit un texte vide pour '%s'. "
                    "Le LLM recevra un prompt sans texte OCR ({ocr_output} vide).",
                    self._name, image_path.name,
                )
            prompt = self._build_prompt(ocr_text=ocr_text)
            logger.info("[Pipeline] appel LLM pour doc %s (text_only, ocr=%d chars)", image_path.name, len(ocr_text))
            result = self.llm_adapter.complete(prompt)
            logger.info("[Pipeline] LLM retourné pour doc %s", image_path.name)

        else:  # TEXT_AND_IMAGE
            if self.ocr_engine is None:
                raise ValueError("ocr_engine est requis pour le mode text_and_image")
            ocr_result = self.ocr_engine.run(image_path)
            ocr_text = ocr_result.text
            logger.debug(
                "[%s] texte OCR : %d car. + image → envoi au LLM.",
                self._name, len(ocr_text),
            )
            if not ocr_text.strip():
                logger.warning(
                    "[%s] le moteur OCR a produit un texte vide pour '%s'. "
                    "Le LLM recevra un prompt sans texte OCR ({ocr_output} vide).",
                    self._name, image_path.name,
                )
            image_b64 = _image_to_b64(image_path)
            prompt = self._build_prompt(ocr_text=ocr_text, image_b64=image_b64)
            logger.info("[Pipeline] appel LLM pour doc %s (text_and_image, ocr=%d chars)", image_path.name, len(ocr_text))
            result = self.llm_adapter.complete(prompt, image_b64=image_b64)
            logger.info("[Pipeline] LLM retourné pour doc %s", image_path.name)

        if not result.success:
            raise RuntimeError(f"Erreur LLM ({self.llm_adapter.model}): {result.error}")

        llm_text = result.text
        # INFO — bilan OCR→LLM visible sur HuggingFace (niveau INFO)
        logger.info(
            "[Pipeline] %s — OCR: %d chars → LLM: %d chars",
            image_path.name, len(ocr_text), len(llm_text),
        )
        if not llm_text or not llm_text.strip():
            logger.warning(
                "[%s] le LLM ('%s') a retourné un texte vide pour '%s'. "
                "CER sera calculé à 1.0 (100%%). "
                "Vérifier : (1) le prompt contient-il {ocr_output} ? "
                "(2) le modèle supporte-t-il ce mode d'appel ? "
                "(3) la réponse n'est-elle pas tronquée (max_tokens) ?",
                self._name, self.llm_adapter.model, image_path.name,
            )
        else:
            logger.debug(
                "[%s] réponse LLM : %d car., extrait : %r",
                self._name, len(llm_text), llm_text[:120],
            )

        ocr_intermediate = ocr_text if self.mode != PipelineMode.ZERO_SHOT else None
        return llm_text, ocr_intermediate

    # ------------------------------------------------------------------
    # Override run() pour injecter les métadonnées pipeline
    # ------------------------------------------------------------------

    def run(self, image_path: str | Path) -> EngineResult:
        """Exécute le pipeline et retourne un EngineResult enrichi de métadonnées."""
        image_path = Path(image_path)
        start = time.perf_counter()

        ocr_intermediate: Optional[str] = None
        try:
            text, ocr_intermediate = self._run_ocr(image_path)
            error = None
        except Exception as exc:  # noqa: BLE001
            text = ""
            error = str(exc)
            logger.warning(
                "[%s] erreur pipeline pour '%s' : %s",
                self._name, image_path.name, exc,
            )

        duration = time.perf_counter() - start

        metadata: dict = {
            "engine_version": self._safe_version(),
            "pipeline_mode": self.mode.value,
            "prompt_file": self.prompt_path,
            "prompt_template": self._prompt_template,
            "llm_model": self.llm_adapter.model,
            "llm_provider": self.llm_adapter.name,
            "pipeline_steps": self._build_steps_info(),
            "is_pipeline": True,
        }
        if ocr_intermediate is not None:
            metadata["ocr_intermediate"] = ocr_intermediate

        return EngineResult(
            engine_name=self.name,
            image_path=str(image_path),
            text=text,
            duration_seconds=round(duration, 4),
            error=error,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, ocr_text: str = "", image_b64: str = "") -> str:
        """Substitue {ocr_output} et {image_b64} dans le template de prompt."""
        return (
            self._prompt_template
            .replace("{ocr_output}", ocr_text)
            .replace("{image_b64}", image_b64)
        )

    def _build_steps_info(self) -> list[dict]:
        steps: list[dict] = []
        if self.ocr_engine:
            steps.append({
                "type": "ocr",
                "engine": self.ocr_engine.name,
                "version": self.ocr_engine._safe_version(),
            })
        steps.append({
            "type": "llm",
            "model": self.llm_adapter.model,
            "provider": self.llm_adapter.name,
            "mode": self.mode.value,
            "prompt_file": self.prompt_path,
        })
        return steps
