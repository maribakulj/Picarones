"""Engine → ``PipelineSpec`` + adapter resolver pour ``PipelineExecutor``.

Module extrait du god-module ``benchmark_runner.py`` lors de la
Phase 6 (complète) de l'audit code-quality (2026-05).

Surface publique (rééxportée par ``benchmark_runner.py`` pour compat
des appels internes) :

- :func:`engine_to_pipeline_spec` — convertit un engine en
  ``PipelineSpec`` (canonique adapter ou OCR+LLM pipeline).
- :func:`build_adapter_resolver` — construit un resolver de
  ``StepExecutor`` à partir d'une liste d'engines.

Helpers internes :

- ``_is_canonical_adapter`` — détecte un ``BaseOCRAdapter``.
- ``_canonical_adapter_to_spec`` — spec mono-step OCR.
- ``_ocr_llm_pipeline_to_spec`` — spec composée OCR+LLM.
- ``_llm_adapter_name`` / ``_safe_pipeline_name`` — naming.
"""

from __future__ import annotations

from typing import Any, Callable

from picarones.domain.artifacts import ArtifactType
from picarones.domain.errors import PicaronesError
from picarones.domain.pipeline_spec import (
    INITIAL_STEP_ID,
    PipelineSpec,
    PipelineStep,
)
from picarones.pipeline.llm_pipeline_builder import make_ocr_llm_pipeline_spec


def _is_canonical_adapter(engine: Any) -> bool:
    """Détecte si ``engine`` est un ``BaseOCRAdapter`` canonique
    (par opposition aux modèles riches en behavior).

    Duck-typing tolérant : un objet est canonical s'il expose
    ``execute``, ``input_types``, ``output_types`` (les trois
    attributs requis par le contrat ``StepExecutor``) ET n'a pas
    le marker ``is_pipeline``.
    """
    from picarones.adapters.ocr.base import BaseOCRAdapter
    return isinstance(engine, BaseOCRAdapter)


def _llm_adapter_name(llm_adapter: Any) -> str:
    """Identifiant ``provider:model`` stable pour un adapter LLM/VLM."""
    return f"{llm_adapter.name}:{llm_adapter.model}"


def _safe_pipeline_name(name: str) -> str:
    """Convertit un ``engine.name`` quelconque en suffixe identifiant
    valide pour ``PipelineSpec.name`` (alphanum + ``_-``)."""
    out: list[str] = []
    for ch in name:
        if ch.isalnum() or ch in "_-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "engine"


def engine_to_pipeline_spec(engine: Any) -> PipelineSpec:
    """Convertit un engine en ``PipelineSpec`` rewrite.

    Deux cas (le path historique ``BaseOCREngine`` a été retiré) :

    - **BaseOCRAdapter** (canonique) : spec mono-step consommant
      ``engine.input_types`` et produisant
      ``engine.effective_output_types`` (sous-ensemble garanti de
      ``output_types`` — exclut les extras opt-in/best-effort).
    - **OCRLLMPipelineConfig** (``engine.is_pipeline = True``) : la
      spec composée est construite via ``make_ocr_llm_pipeline_spec``
      avec le mode (``text_only`` / ``text_and_image`` /
      ``zero_shot``), l'OCR amont (s'il existe), le LLM, et le
      template de prompt en ``llm_params``.
    """
    if _is_canonical_adapter(engine):
        return _canonical_adapter_to_spec(engine)
    if getattr(engine, "is_pipeline", False):
        return _ocr_llm_pipeline_to_spec(engine)
    raise PicaronesError(
        f"Type d'engine non supporté : {type(engine).__name__}.  "
        "Attendu : ``BaseOCRAdapter`` ou ``OCRLLMPipelineConfig``.  "
        "Le support historique ``BaseOCREngine`` / ``OCRLLMPipeline`` "
        "a été retiré au sprint H.2.c.",
    )


def _canonical_adapter_to_spec(adapter: Any) -> PipelineSpec:
    """Spec mono-step pour un ``BaseOCRAdapter`` canonique."""
    name = adapter.name
    safe_name = _safe_pipeline_name(name)
    input_types = tuple(adapter.input_types)
    # ``effective_output_types`` (et non ``output_types``) : on déclare
    # uniquement les artefacts *garantis* par cette instance.  Sinon un
    # adapter dont ``output_types`` annonce des extras opt-in/best-effort
    # (Tesseract : ``CONFIDENCES`` / ``ALTO_XML``) ferait échouer tout le
    # step (``missing_output``) quand l'extra n'est pas produit — et
    # l'``engine_error`` qui en résulte sauterait les hooks
    # ``requires_success`` (analyse caractères vide alors que l'OCR est
    # valide).  Défaut = ``output_types`` pour les adapters simples.
    output_types = tuple(adapter.effective_output_types)
    if ArtifactType.IMAGE not in input_types:
        raise PicaronesError(
            f"Adapter {name!r} ne déclare pas IMAGE en input_types "
            f"({input_types!r}) — incompatible avec "
            "``run_benchmark_via_service`` qui fournit toujours IMAGE.",
        )
    return PipelineSpec(
        name=f"ocr_only_{safe_name}",
        description=f"OCR step seul ({name}, canonique) — IMAGE → "
                    f"{','.join(t.value for t in output_types)}.",
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(
            PipelineStep(
                id="ocr",
                kind="ocr",
                adapter_name=name,
                input_types=input_types,
                output_types=output_types,
                inputs_from={ArtifactType.IMAGE: INITIAL_STEP_ID},
            ),
        ),
    )


def _ocr_llm_pipeline_to_spec(pipeline: Any) -> PipelineSpec:
    """Spec composée pour un ``OCRLLMPipelineConfig`` canonique (3 modes).

    Tolère ``pipeline.mode`` en enum (``PipelineMode.TEXT_ONLY``)
    ou en string (canonique ``"text_only"``).
    """
    mode_attr = pipeline.mode
    mode = mode_attr.value if hasattr(mode_attr, "value") else mode_attr
    llm_name = _llm_adapter_name(pipeline.llm_adapter)
    llm_params: dict[str, str | int | float | bool] = {
        "prompt_template": pipeline.prompt_template,
    }
    if mode == "zero_shot":
        return make_ocr_llm_pipeline_spec(
            mode="zero_shot",
            llm_adapter_name=llm_name,
            llm_params=llm_params,
        )
    if pipeline.ocr_engine is None:
        raise PicaronesError(
            f"OCRLLMPipeline mode {mode!r} requiert un ocr_engine — "
            "valeur None inattendue.",
        )
    return make_ocr_llm_pipeline_spec(
        mode=mode,
        ocr_adapter_name=pipeline.ocr_engine.name,
        llm_adapter_name=llm_name,
        llm_params=llm_params,
    )


def build_adapter_resolver(
    engines: list[Any],
) -> Callable[[str], Any]:
    """Construit un adapter resolver pour ``PipelineExecutor``.

    Parcourt les engines fournis et associe leur ``name`` à un
    ``StepExecutor`` valide :

    - **BaseOCRAdapter** : enregistré directement (déjà ``StepExecutor``).
    - **OCRLLMPipelineConfig** → enregistre les deux sous-composants :
      ``ocr_adapter`` (canonique, direct) et ``llm_adapter`` (déjà
      ``StepExecutor`` natif).  Le pipeline lui-même n'est pas
      enregistré directement — sa spec référence ses sous-steps par
      leur ``adapter_name``.

    Le resolver retourné lève ``KeyError`` si un nom inconnu est
    demandé.

    Raises
    ------
    PicaronesError
        Si deux engines partagent le même ``name`` (collision) avec
        des configurations distinctes.
    """
    name_to_executor: dict[str, Any] = {}

    def _is_equivalent_executor(a: Any, b: Any) -> bool:
        """Deux executors fonctionnellement équivalents : même type
        + même état (``__dict__`` complet).

        Cas concret : deux ``PipelineConfig`` qui utilisent
        ``tesseract`` avec la même langue — l'un en mode OCR seul,
        l'autre encapsulé dans un pipeline OCR+LLM.  Le factory web
        leur donne le même ``name`` (dérivé de la config) → la 2e
        registration ici est trivialement idempotente.

        Sécurité : la comparaison ``__dict__`` inclut TOUS les
        attributs (privés ``_name``/``_lang``/``_psm`` ou publics).
        Une config différente (lang≠, psm≠) → ``__dict__`` différents
        → équivalence False → collision réelle remontée.
        """
        if type(a) is not type(b):
            return False
        try:
            return a.__dict__ == b.__dict__
        except AttributeError:
            return False

    def _register(name: str, executor: Any) -> None:
        existing = name_to_executor.get(name)
        if existing is None:
            name_to_executor[name] = executor
            return
        if existing is executor:
            return
        if _is_equivalent_executor(existing, executor):
            return
        raise PicaronesError(
            f"Adapter resolver : nom {name!r} enregistré deux fois "
            f"avec des configurations différentes "
            f"({type(existing).__name__} vs "
            f"{type(executor).__name__}, états distincts).  "
            "Probable régression dans le factory : deux engines "
            "logiquement distincts doivent recevoir des ``name`` "
            "distincts.",
        )

    for engine in engines:
        if _is_canonical_adapter(engine):
            _register(engine.name, engine)
        elif getattr(engine, "is_pipeline", False):
            ocr_engine = getattr(engine, "ocr_engine", None)
            llm_adapter = getattr(engine, "llm_adapter", None)
            if ocr_engine is not None:
                _register(ocr_engine.name, ocr_engine)
            if llm_adapter is not None:
                _register(_llm_adapter_name(llm_adapter), llm_adapter)
        else:
            raise PicaronesError(
                f"Type d'engine non supporté pour le resolver : "
                f"{type(engine).__name__}.  Attendu : ``BaseOCRAdapter`` "
                "ou ``OCRLLMPipelineConfig``.",
            )

    def resolver(name: str) -> Any:
        if name not in name_to_executor:
            raise KeyError(
                f"adapter inconnu pour le resolver : {name!r}.  "
                f"Enregistrés : {sorted(name_to_executor.keys())!r}."
            )
        return name_to_executor[name]

    return resolver


__all__ = [
    "build_adapter_resolver",
    "engine_to_pipeline_spec",
]
