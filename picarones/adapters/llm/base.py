"""Interface abstraite commune à tous les adaptateurs LLM."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


from picarones.adapters._retry import (
    DEFAULT_BACKOFF_BASE as _DEFAULT_BACKOFF_BASE,
)
from picarones.adapters._retry import (
    DEFAULT_MAX_RETRIES as _DEFAULT_MAX_RETRIES,
)
from picarones.adapters._retry import (
    is_retryable as _is_retryable,
)


def normalize_llm_content(raw: Any) -> str:
    """Normalise une réponse LLM en chaîne plate.

    Chantier 4 (post-Sprint 97) — propagation du fix Mistral
    Sprint 15 à tous les providers. Le SDK Mistral peut retourner
    une liste de ``ContentChunk`` au lieu d'une chaîne pour certains
    modèles/versions ; le SDK OpenAI peut faire de même quand on
    active des features de structuration. Ce helper applique la même
    discipline pour les 4 adapters :

    - ``str``                          → renvoyée telle quelle (ou ``""``).
    - ``None``                         → ``""``.
    - ``list[ContentChunk]``           → concaténation des ``.text``.
    - ``list[dict]`` avec clé ``text`` → concaténation des ``["text"]``.
    - ``list[str]``                    → concaténation directe.
    - autre objet avec ``.text``       → ``obj.text``.
    - autre                            → ``str(obj)`` (best-effort).

    Le résultat est garanti être une ``str`` ; ``""`` quand la réponse
    est vide. La fonction est idempotente : ``normalize_llm_content(s)
    == s`` pour toute chaîne ``s``.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for chunk in raw:
            if chunk is None:
                continue
            if isinstance(chunk, str):
                parts.append(chunk)
                continue
            if hasattr(chunk, "text"):
                txt = getattr(chunk, "text", None)
                if isinstance(txt, str):
                    parts.append(txt)
                    continue
            if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                parts.append(chunk["text"])
                continue
            # Dernier recours — convertit le chunk en chaîne
            parts.append(str(chunk))
        return "".join(parts)
    if hasattr(raw, "text") and isinstance(getattr(raw, "text", None), str):
        return raw.text  # type: ignore[no-any-return]
    return str(raw)


def log_http_error(
    adapter_name: str,
    model: str,
    exc: Exception,
    *,
    env_var: Optional[str] = None,
) -> None:
    """Log standardisé des erreurs HTTP des SDK LLM.

    Chantier 4 (post-Sprint 97) — propagation du log discriminant
    Mistral/OpenAI à tous les providers. Inspecte ``status_code`` et
    ``http_status`` puis émet un warning ciblé selon le code :

    - 401 : clé API invalide/expirée (mention de la variable
      d'environnement à vérifier si fournie).
    - 429 : rate limit / quota dépassé.
    - 5xx : problème serveur côté provider.
    - autre / pas de status_code : log générique.

    L'exception n'est pas levée — l'appelant doit ``raise``
    explicitement après ce log s'il veut propager (le retry est géré
    par ``BaseLLMAdapter.complete`` selon ``_is_retryable``).
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status == 401:
        suffix = f" Vérifier {env_var}." if env_var else ""
        logger.warning(
            "[%s] erreur HTTP 401 — clé API invalide ou expirée "
            "(modèle=%s).%s",
            adapter_name, model, suffix,
        )
    elif status == 429:
        logger.warning(
            "[%s] erreur HTTP 429 — quota dépassé ou rate-limit "
            "(modèle=%s). Réessayer plus tard.",
            adapter_name, model,
        )
    elif status is not None and status >= 500:
        logger.warning(
            "[%s] erreur HTTP %d — problème serveur (modèle=%s) : %s",
            adapter_name, status, model, exc,
        )
    else:
        logger.warning(
            "[%s] erreur lors de l'appel API (modèle=%s) : %s",
            adapter_name, model, exc,
        )


from picarones.domain.errors import AdapterStepError


def _substitute_prompt_variables(
    template: str,
    text: str,
    image_b64: str | None,
) -> str:
    """Substitue les variables d'un template de prompt LLM.

    Supporte deux conventions de nommage des variables :

    - **Rewrite** (Sprint A14-S44) : ``{text}``.  Substitué par
      ``str.format(text=text)``.
    - **Legacy** (``OCRLLMPipeline``, Sprint A.2 du plan v2.0) :
      ``{ocr_output}`` et ``{image_b64}``.  Substitués par
      ``str.replace(...)`` — tolérant si une variable est absente
      du template.

    La convention est détectée automatiquement.  Si le template
    contient ``{ocr_output}`` ou ``{image_b64}``, on applique le
    format legacy ; sinon, on applique le format rewrite (qui
    lèvera ``KeyError`` si une variable inattendue est utilisée,
    comportement strict d'origine).

    Parameters
    ----------
    template:
        Template de prompt (chaîne avec variables ``{...}``).
    text:
        Texte OCR à injecter (substitue ``{text}`` ou ``{ocr_output}``).
    image_b64:
        Image encodée base64 sans préfixe (substitue ``{image_b64}``).
        ``None`` → chaîne vide pour les modes texte-seul.
    """
    if "{ocr_output}" in template or "{image_b64}" in template:
        return (
            template
            .replace("{ocr_output}", text)
            .replace("{image_b64}", image_b64 or "")
        )
    # Convention rewrite : ``{text}`` est l'unique placeholder.
    # Défense en profondeur (Sprint S9) : si la string n'a aucun
    # placeholder de substitution, ``template.format(text=text)``
    # retournerait la string inchangée sans erreur — ce qui faisait
    # passer un filename (``correction_*.txt``) au LLM en prod.
    # On lève maintenant explicitement : un template sans
    # placeholder est sémantiquement vide (le LLM ignorerait l'OCR).
    if "{text}" not in template:
        raise ValueError(
            "Prompt template invalide : aucun placeholder "
            "``{ocr_output}``, ``{text}`` ou ``{image_b64}`` "
            "trouvé.  Le LLM recevrait une string fixe.  "
            "Probable cause : un filename a été injecté au "
            "lieu du contenu du fichier prompt.",
        )
    return template.format(text=text)


class LLMAdapterError(AdapterStepError):
    """Erreur typée pour un échec d'adapter LLM.

    Hérite de ``AdapterStepError`` (racine commune avec OCR et VLM)
    → un caller peut catcher ``AdapterStepError`` pour toute erreur
    d'adapter sans connaître la sous-classe.

    Avant S52, ``BaseLLMAdapter.execute`` levait ``OCRAdapterError``
    par confusion sémantique — c'était noté dans l'audit comme issue
    #11 (hiérarchie incohérente).
    """


@dataclass
class LLMResult:
    """Résultat produit par un appel LLM."""

    model_id: str
    text: str
    duration_seconds: float
    tokens_used: Optional[int] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class BaseLLMAdapter(ABC):
    """Classe de base pour tous les adaptateurs LLM.

    Chaque adaptateur doit implémenter :
    - ``name``         : identifiant du provider (ex : 'openai')
    - ``default_model``: modèle par défaut du provider
    - ``_call()``      : appel API effectif, retourne le texte brut

    Les clés API sont lues depuis les variables d'environnement uniquement.

    Retry automatique
    -----------------
    Les erreurs retryables (HTTP 429, 5xx, timeout réseau) sont automatiquement
    retentées avec backoff exponentiel (2s, 4s, 8s par défaut). Configurable
    via ``config["max_retries"]`` et ``config["retry_backoff"]``.

    Normalisation des réponses (chantier 4)
    ---------------------------------------
    Les sous-classes utilisent :func:`normalize_llm_content` sur la
    réponse SDK avant de la retourner — garantit qu'une réponse de
    type ``list[ContentChunk]`` (Mistral, parfois OpenAI) est
    convertie en ``str`` plate.

    Logging d'erreurs HTTP (chantier 4)
    -----------------------------------
    Les sous-classes utilisent :func:`log_http_error` pour produire
    un log discriminant par ``status_code`` (401 → clé invalide,
    429 → rate limit, 5xx → serveur).  Auparavant ce log était
    dupliqué chez Mistral/OpenAI et absent chez Anthropic.

    Sprint A14-S44 — intégration pipeline native
    ---------------------------------------------
    ``BaseLLMAdapter`` implémente désormais le contrat ``StepExecutor``
    du pipeline (``input_types``, ``output_types``, ``execution_mode``,
    ``execute(inputs, params, context)``) — un adapter LLM est
    directement utilisable comme step de pipeline pour la post-correction
    de texte OCR.  Pas de wrapper / shim : la méthode ``execute`` vit
    dans la base et est partagée par les 4 adapters concrets.

    Convention par défaut : un LLM consomme ``RAW_TEXT`` (depuis l'OCR
    en amont) et produit ``CORRECTED_TEXT``.  Une sous-classe peut
    surcharger ``input_types`` / ``output_types`` si elle implémente un
    autre contrat (ex : ALTO → ALTO pour un module de remappage).
    """

    # Variable d'environnement portant la clé API.  Sous-classes
    # surchargent (ex. ``"OPENAI_API_KEY"``) ; mention utilisée par
    # :func:`log_http_error` quand un 401 est rencontré.  ``None``
    # pour les providers sans clé (Ollama).
    api_key_env_var: Optional[str] = None

    # ──────────────────────────────────────────────────────────────────
    # Sprint A14-S44 — contrat StepExecutor du pipeline
    # ──────────────────────────────────────────────────────────────────

    #: Types d'artefacts consommés par défaut.  Surchargeable par
    #: une sous-classe qui consommerait des artefacts différents
    #: (ex : ALTO_XML pour un remappeur ALTO LLM).
    @property
    def input_types(self) -> "frozenset":
        from picarones.domain.artifacts import ArtifactType
        return frozenset({ArtifactType.RAW_TEXT})

    @property
    def output_types(self) -> "frozenset":
        from picarones.domain.artifacts import ArtifactType
        return frozenset({ArtifactType.CORRECTED_TEXT})

    #: Mode d'exécution : LLM via API → IO-bound → ThreadPool dans le
    #: runner.  Une sous-classe locale (Ollama CPU-bound) peut
    #: surcharger en ``"cpu"``.
    execution_mode: str = "io"

    #: Prompts de post-correction par défaut, indexés par code langue
    #: ISO-639-1 (``fr``, ``en``, ``la``).  Sélection via
    #: ``config["lang"]`` ; fallback FR si la langue est absente.
    DEFAULT_CORRECTION_PROMPTS: dict[str, str] = {
        "fr": (
            "Corrige les erreurs OCR dans le texte suivant en "
            "conservant fidèlement la langue, l'orthographe "
            "historique et la ponctuation. Retourne uniquement le "
            "texte corrigé, sans commentaire :\n\n{text}"
        ),
        "en": (
            "Fix OCR errors in the following text while preserving "
            "the original language, historical spelling, and "
            "punctuation. Return only the corrected text, with no "
            "commentary:\n\n{text}"
        ),
        "la": (
            "Corrige errores OCR in textu sequenti, fideliter "
            "servans linguam, orthographiam historicam et "
            "interpunctionem. Redde solum textum correctum, sine "
            "ulla glossa:\n\n{text}"
        ),
    }

    def __init__(
        self,
        model: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        self.config: dict = config or {}
        self.model: str = model or self.default_model

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifiant du provider (ex : 'openai', 'anthropic')."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Modèle utilisé si aucun n'est fourni explicitement."""

    @abstractmethod
    def _call(self, prompt: str, image_b64: Optional[str] = None) -> str:
        """Appel LLM effectif.

        Parameters
        ----------
        prompt:
            Texte du prompt final (variables déjà substituées).
        image_b64:
            Image encodée en base64 (sans préfixe data URI).
            None pour les appels texte-uniquement.

        Returns
        -------
        str
            Texte généré par le LLM.
        """

    def complete(
        self,
        prompt: str,
        image_b64: Optional[str] = None,
    ) -> LLMResult:
        """Point d'entrée public : appelle le LLM avec retry automatique."""
        max_retries = int(self.config.get("max_retries", _DEFAULT_MAX_RETRIES))
        backoff_base = float(self.config.get("retry_backoff", _DEFAULT_BACKOFF_BASE))

        start = time.perf_counter()
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                text = self._call(prompt, image_b64)
                duration = time.perf_counter() - start
                return LLMResult(
                    model_id=self.model,
                    text=text,
                    duration_seconds=round(duration, 4),
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_retries and _is_retryable(exc):
                    wait = backoff_base ** (attempt + 1)
                    logger.warning(
                        "[%s] erreur retryable (tentative %d/%d, attente %.1fs) : %s",
                        self.name, attempt + 1, max_retries + 1, wait, exc,
                    )
                    time.sleep(wait)
                else:
                    break

        duration = time.perf_counter() - start
        return LLMResult(
            model_id=self.model,
            text="",
            duration_seconds=round(duration, 4),
            error=str(last_exc),
        )

    # ──────────────────────────────────────────────────────────────────
    # Sprint A14-S44 — execute() pour le pipeline
    # ──────────────────────────────────────────────────────────────────

    def execute(
        self,
        inputs: dict,
        params: dict,
        context: Any,
    ) -> dict:
        """Exécute la post-correction LLM en tant que step de pipeline.

        Convention par défaut : lit ``inputs[RAW_TEXT]`` (Artifact),
        charge son contenu UTF-8 depuis l'URI, appelle ``self.complete``
        avec le ``correction_prompt`` formaté, écrit le résultat dans
        un fichier ``<input_stem>.<adapter_name>.corrected.txt``, et
        retourne ``{CORRECTED_TEXT: Artifact}``.

        Le caller (``PipelineExecutor``) catch les exceptions ; on les
        propage telles quelles.

        Optionnel : si ``inputs[IMAGE]`` est présent, l'image est
        encodée en base64 et passée au LLM (mode VLM).  Les sous-classes
        qui ne supportent pas la vision (ex. ollama texte) ignorent
        silencieusement.
        """
        from pathlib import Path
        import base64

        from picarones.domain.artifacts import Artifact, ArtifactType

        if ArtifactType.RAW_TEXT not in inputs:
            raise LLMAdapterError(
                f"{self.name} : input RAW_TEXT manquant.",
            )
        text_artifact = inputs[ArtifactType.RAW_TEXT]
        if text_artifact.uri is None:
            raise LLMAdapterError(
                f"{self.name} : artefact RAW_TEXT "
                f"{text_artifact.id!r} sans URI.",
            )
        text_path = Path(text_artifact.uri)
        if not text_path.exists():
            raise LLMAdapterError(
                f"{self.name} : fichier texte introuvable {text_path!r}.",
            )

        original_text = text_path.read_text(encoding="utf-8")

        # Image optionnelle (VLM-style si supporté).
        image_b64: Optional[str] = None
        image_artifact = inputs.get(ArtifactType.IMAGE)
        if image_artifact is not None and image_artifact.uri is not None:
            image_path = Path(image_artifact.uri)
            if image_path.exists():
                image_b64 = base64.b64encode(
                    image_path.read_bytes(),
                ).decode("ascii")

        # Priorité (Sprint A.2 du plan v2.0) :
        # 1. ``params["prompt_template"]`` (override par le step lui-même —
        #    permet à un caller qui construit une PipelineSpec d'injecter
        #    un prompt personnalisé sans toucher à la config de l'adapter).
        # 2. ``self.config["correction_prompt"]`` (override au constructeur
        #    de l'adapter — pattern historique).
        # 3. Prompt par langue selon ``self.config["lang"]``.
        # 4. Fallback FR.
        # ``""`` est traité comme "pas fourni" (au même titre que
        # ``None``) — on tombe sur le défaut de l'adapter.  Avant
        # Sprint S9, ``""`` était propagé jusqu'à
        # ``_substitute_prompt_variables`` qui retournait ``""``
        # silencieusement, laissant le LLM voir une string vide.
        param_prompt = params.get("prompt_template") if params else None
        if param_prompt:
            prompt_template = param_prompt
        else:
            custom_prompt = self.config.get("correction_prompt")
            if custom_prompt is not None:
                prompt_template = custom_prompt
            else:
                lang = (self.config.get("lang") or "fr").lower()
                if lang not in self.DEFAULT_CORRECTION_PROMPTS:
                    logger.warning(
                        "[%s] lang=%r non supportée par "
                        "DEFAULT_CORRECTION_PROMPTS (%s) — fallback FR. "
                        "Pour un corpus dans cette langue, fournir "
                        "config['correction_prompt'] explicite.",
                        self.name, lang,
                        sorted(self.DEFAULT_CORRECTION_PROMPTS.keys()),
                    )
                prompt_template = self.DEFAULT_CORRECTION_PROMPTS.get(
                    lang, self.DEFAULT_CORRECTION_PROMPTS["fr"],
                )
        prompt = _substitute_prompt_variables(
            prompt_template, original_text, image_b64,
        )

        result = self.complete(prompt, image_b64=image_b64)
        if not result.success:
            raise LLMAdapterError(
                f"{self.name} : LLM a échoué ({result.error}).",
            )

        from picarones.adapters.output_paths import resolve_output_path
        out_path = resolve_output_path(
            input_path=text_path,
            adapter_name=self.name,
            suffix="corrected.txt",
            context=context,
        )
        out_path.write_text(result.text, encoding="utf-8")

        return {
            ArtifactType.CORRECTED_TEXT: Artifact(
                id=f"{context.document_id}:{self.name}:corrected_text",
                document_id=context.document_id,
                type=ArtifactType.CORRECTED_TEXT,
                produced_by_step="post_correction",
                uri=str(out_path),
            ),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


__all__ = [
    "BaseLLMAdapter",
    "LLMAdapterError",
    "LLMResult",
    "log_http_error",
    "normalize_llm_content",
]
