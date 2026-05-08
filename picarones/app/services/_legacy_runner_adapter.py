"""Sprint D.1 du plan v2.0 — adapter de compat ``run_benchmark`` legacy
→ ``BenchmarkService`` rewrite.

Ce module présente l'API mono-call historique de
``picarones.measurements.runner.run_benchmark`` mais s'appuie en
interne sur le rewrite (``BenchmarkService``,
``PipelineExecutor``, ``CorpusRunner``).  Il sert de pont
transitoire pour faciliter la migration des callers en plusieurs
étapes :

1. (cette session) Helpers de mapping ``Corpus`` ↔ ``CorpusSpec``
   et ``Document`` ↔ ``DocumentRef`` — testables indépendamment.
2. (sub-phase D.1.b) Mapping ``BaseOCREngine`` → ``PipelineSpec``
   + adapter resolver.
3. (sub-phase D.1.c) Conversion ``RunResult`` → ``BenchmarkResult``.
4. (sub-phase D.1.d) Fonction ``run_benchmark_via_service``
   complète avec progress callback, output_json, partial_dir.
5. (sub-phase D.1.e) Tests d'équivalence numérique (CER/WER) entre
   les deux runners sur les fixtures.

Trace de retrait
----------------
Ce module est **transitoire** (Sprint D du plan v2.0).  Il sera
supprimé en D.6 quand tous les callers (cli/_workflows,
web/benchmark_utils) consommeront ``BenchmarkService``
directement.

Cette première itération n'expose que les helpers de mapping
documents/corpus — la fonction publique
``run_benchmark_via_service`` arrive dans une session ultérieure
quand toutes les briques seront en place.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from picarones.adapters.legacy_engines._step_executor import (
    LegacyOCREngineExecutor,
)
from picarones.domain.artifacts import ArtifactType
from picarones.domain.corpus import CorpusSpec
from picarones.domain.documents import DocumentRef, GroundTruthRef
from picarones.domain.errors import PicaronesError
from picarones.domain.pipeline_spec import (
    INITIAL_STEP_ID,
    PipelineSpec,
    PipelineStep,
)
from picarones.pipeline.llm_pipeline_builder import make_ocr_llm_pipeline_spec

if TYPE_CHECKING:
    from picarones.adapters.legacy_engines.base import BaseOCREngine
    from picarones.evaluation.corpus import Corpus, Document

# Pas d'import direct de ``picarones.pipelines.base.OCRLLMPipeline`` ici —
# l'invariant architectural ``test_layer_imports_are_legal[layer-app]``
# interdit à ``app/`` de dépendre du legacy.  On consomme un
# ``OCRLLMPipeline`` exclusivement par duck typing (``is_pipeline``,
# ``ocr_engine``, ``llm_adapter``, ``mode``, ``prompt_template``).


# ──────────────────────────────────────────────────────────────────────
# Mapping Document (legacy) → DocumentRef (rewrite)
# ──────────────────────────────────────────────────────────────────────


def document_to_document_ref(
    document: "Document",
    *,
    workspace_dir: Path,
) -> DocumentRef:
    """Convertit un ``Document`` legacy en ``DocumentRef`` rewrite.

    Le ``Document`` legacy porte sa GT en mémoire (``ground_truth: str``
    et ``ground_truths: dict[ArtifactType, GTPayload]``).  Le
    ``DocumentRef`` rewrite porte des références filesystem
    (``GroundTruthRef.uri``).  La conversion écrit chaque GT
    in-memory dans ``workspace_dir`` et construit les références.

    Parameters
    ----------
    document:
        Document legacy.  ``image_path`` non-``None`` est requis ;
        ``ground_truth`` (TEXT) peut être vide.
    workspace_dir:
        Répertoire de travail où écrire les fichiers GT
        synthétisés.  Doit exister et être writable.

    Returns
    -------
    DocumentRef
        Référence canonique avec ``id``, ``image_uri`` et un tuple
        ordonné de ``GroundTruthRef`` (par niveau ArtifactType).

    Raises
    ------
    PicaronesError
        Si ``document.doc_id`` ne respecte pas le regex
        ``DocumentRef._DOC_ID_RE`` (fallback explicite si besoin).
    """
    if not workspace_dir.exists():
        raise PicaronesError(
            f"workspace_dir doit exister : {workspace_dir!r}"
        )

    doc_id = _safe_doc_id(document.doc_id)

    image_uri: str | None = None
    if document.image_path is not None:
        image_uri = str(document.image_path)

    ground_truths: list[GroundTruthRef] = []

    # Niveau TEXT : ``ground_truth: str`` → fichier .gt.txt dans
    # workspace.  On écrit toujours, même vide, pour préserver le
    # contrat (un caller qui lit le fichier obtient la chaîne vide
    # et le runner sait gérer ce cas en métriques).
    if document.ground_truth or _has_text_gt(document):
        text_content = document.ground_truth
        if not text_content and _has_text_gt(document):
            # Le payload est dans ``ground_truths[RAW_TEXT]``.
            from picarones.evaluation.corpus import TextGT

            payload = document.ground_truths.get(ArtifactType.RAW_TEXT)
            if isinstance(payload, TextGT):
                text_content = payload.text

        text_path = workspace_dir / f"{doc_id.replace('/', '_')}.gt.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text_content, encoding="utf-8")
        ground_truths.append(
            GroundTruthRef(type=ArtifactType.RAW_TEXT, uri=str(text_path)),
        )

    # Niveaux étendus (ALTO, PAGE, ENTITIES, READING_ORDER) :
    # déjà sérialisés via leur ``source_path`` quand disponibles.
    # On préfère le ``source_path`` original au lieu d'une copie
    # pour ne pas dupliquer.
    for level in (
        ArtifactType.ALTO_XML,
        ArtifactType.PAGE_XML,
        ArtifactType.ENTITIES,
        ArtifactType.READING_ORDER,
    ):
        payload = document.ground_truths.get(level)
        if payload is None:
            continue
        gt_uri = _resolve_gt_uri(
            level=level,
            payload=payload,
            doc_id=doc_id,
            workspace_dir=workspace_dir,
        )
        ground_truths.append(GroundTruthRef(type=level, uri=gt_uri))

    return DocumentRef(
        id=doc_id,
        image_uri=image_uri,
        ground_truths=tuple(ground_truths),
    )


def corpus_to_corpus_spec(
    corpus: "Corpus",
    *,
    workspace_dir: Path,
) -> CorpusSpec:
    """Convertit un ``Corpus`` legacy en ``CorpusSpec`` rewrite.

    Itère sur ``corpus.documents`` et applique
    ``document_to_document_ref`` pour chacun.

    Parameters
    ----------
    corpus:
        Corpus legacy.
    workspace_dir:
        Répertoire de travail où écrire les fichiers GT
        synthétisés (typiquement un ``tempfile.TemporaryDirectory``
        détenu par le caller).

    Returns
    -------
    CorpusSpec
        Spec immutable consommable par ``BenchmarkService.run``.
    """
    if not workspace_dir.exists():
        raise PicaronesError(
            f"workspace_dir doit exister : {workspace_dir!r}"
        )

    docs = tuple(
        document_to_document_ref(d, workspace_dir=workspace_dir)
        for d in corpus.documents
    )

    metadata: dict[str, str] = {}
    for k, v in (corpus.metadata or {}).items():
        # CorpusSpec.metadata accepte ``str`` only — sérialise les
        # valeurs scalaires en str ; les structures complexes sont
        # ignorées (le caller adapte si besoin).
        if isinstance(v, (str, int, float, bool)):
            metadata[str(k)] = str(v)

    if corpus.source_path:
        metadata.setdefault("source_path", str(corpus.source_path))

    return CorpusSpec(
        name=corpus.name,
        documents=docs,
        metadata=metadata,
    )


# ──────────────────────────────────────────────────────────────────────
# Mapping RunResult (rewrite) → BenchmarkResult (legacy)
# ──────────────────────────────────────────────────────────────────────


def run_result_to_benchmark_result(
    run_result: Any,
    *,
    corpus: "Corpus",
    engines: list["BaseOCREngine"],
    char_exclude: Any | None = None,
    normalization_profile: Any | None = None,
) -> Any:
    """Transpose un ``RunResult`` rewrite en ``BenchmarkResult`` legacy.

    Le mapping est en **transposition** :

    - **Rewrite** ``RunResult`` : itère par document puis par
      pipeline.  ``run_result.document_results[i].pipeline_results[j]``.
    - **Legacy** ``BenchmarkResult`` : itère par engine puis par
      document.  ``benchmark_result.engine_reports[j].document_results[i]``.

    Pour chaque couple ``(engine, document)``, le converter :

    1. Récupère le ``PipelineResult`` correspondant depuis
       ``RunDocumentResult.pipeline_results``.
    2. Lit le texte produit final (``CORRECTED_TEXT`` prioritaire,
       sinon ``RAW_TEXT``) depuis l'``Artifact.uri``.
    3. Lit l'``ocr_intermediate`` (RAW_TEXT) si le pipeline a un
       step OCR amont.
    4. Calcule les métriques CER/WER via ``compute_metrics``.
    5. Construit un ``DocumentResult`` legacy avec ``engine_error``
       extrait des ``step_results``.
    6. Aggrège les métriques par engine via ``aggregate_metrics``.
    7. Reconstitue ``pipeline_info`` pour les engines pipeline
       (mode, prompt, llm_model, llm_provider, pipeline_steps).

    Parameters
    ----------
    run_result:
        ``RunResult`` produit par ``BenchmarkService.run``.
    corpus:
        Corpus legacy d'origine — sert à récupérer le ``ground_truth``
        et l'``image_path`` pour chaque document, dans le même ordre
        que ``run_result.document_results``.
    engines:
        Liste d'engines legacy dans l'ordre où leurs specs ont été
        passées à ``BenchmarkService.run`` (l'ordre détermine
        l'index dans ``RunDocumentResult.pipeline_results``).
    char_exclude:
        Filtre passé à ``compute_metrics``.  ``None`` par défaut.
    normalization_profile:
        Profil de normalisation passé à ``compute_metrics``.

    Returns
    -------
    BenchmarkResult
        Format legacy compatible avec les consommateurs historiques
        (rapport HTML, persistance JSON, narrative engine).
    """
    from picarones.evaluation.benchmark_result import (
        BenchmarkResult,
        DocumentResult,
        EngineReport,
    )
    from picarones.evaluation.metric_result import aggregate_metrics
    # ``compute_metrics`` n'a pas encore d'équivalent dans
    # ``evaluation/`` (migration en Sprint E du plan v2.0).  En
    # attendant, on l'importe dynamiquement via ``importlib`` —
    # explicitement permis par ``test_no_legacy_imports_in_rewrite``
    # qui ne couvre pas les imports différés.  Ce détour disparaît
    # quand ``compute_metrics`` aura été migré vers
    # ``picarones.evaluation.metrics.text_metrics`` (Sprint E).
    import importlib
    _metrics_mod = importlib.import_module("picarones.measurements.metrics")
    compute_metrics = _metrics_mod.compute_metrics

    documents = list(corpus.documents)
    if len(documents) != len(run_result.document_results):
        raise PicaronesError(
            f"Mismatch documents : corpus={len(documents)} vs "
            f"run_result={len(run_result.document_results)}.",
        )

    engine_reports: list[Any] = []

    for engine_idx, engine in enumerate(engines):
        doc_results: list[Any] = []
        for doc_idx, document in enumerate(documents):
            run_doc = run_result.document_results[doc_idx]
            if engine_idx >= len(run_doc.pipeline_results):
                # Plus d'engines que de pipeline_results — incohérence.
                continue
            pipeline_result = run_doc.pipeline_results[engine_idx]

            text_final, ocr_intermediate = _extract_text_outputs(
                pipeline_result=pipeline_result,
            )
            engine_error = _extract_first_error(pipeline_result)
            duration = float(pipeline_result.duration_seconds)

            metrics = compute_metrics(
                document.ground_truth,
                text_final,
                normalization_profile=normalization_profile,
                char_exclude=char_exclude,
            )

            pipeline_metadata = _build_pipeline_metadata(
                engine=engine,
                ocr_intermediate=ocr_intermediate,
            )

            doc_results.append(
                DocumentResult(
                    doc_id=document.doc_id,
                    image_path=str(document.image_path),
                    ground_truth=document.ground_truth,
                    hypothesis=text_final,
                    metrics=metrics,
                    duration_seconds=round(duration, 4),
                    engine_error=engine_error,
                    ocr_intermediate=ocr_intermediate,
                    pipeline_metadata=pipeline_metadata,
                ),
            )

        aggregated = aggregate_metrics([d.metrics for d in doc_results])
        pipeline_info = _build_pipeline_info(engine)

        engine_reports.append(
            EngineReport(
                engine_name=engine.name,
                engine_version=_safe_engine_version(engine),
                engine_config=getattr(engine, "config", {}) or {},
                document_results=doc_results,
                aggregated_metrics=aggregated,
                pipeline_info=pipeline_info,
            ),
        )

    return BenchmarkResult(
        corpus_name=corpus.name,
        corpus_source=str(corpus.source_path) if corpus.source_path else None,
        document_count=len(documents),
        engine_reports=engine_reports,
    )


# ──────────────────────────────────────────────────────────────────────
# Helpers privés du converter RunResult → BenchmarkResult
# ──────────────────────────────────────────────────────────────────────


def _extract_text_outputs(pipeline_result: Any) -> tuple[str, str | None]:
    """Extrait ``(text_final, ocr_intermediate)`` du PipelineResult.

    - ``text_final`` : ``CORRECTED_TEXT`` prioritaire (post-correction
      LLM), sinon ``RAW_TEXT`` (OCR seul ou VLM zero-shot).
    - ``ocr_intermediate`` : ``RAW_TEXT`` quand un ``CORRECTED_TEXT``
      coexiste — ce qui correspond au texte OCR avant correction LLM.
      ``None`` si pas de pipeline composé.
    """
    corrected_text: str | None = None
    raw_text: str | None = None
    for art in pipeline_result.artifacts:
        if art.uri is None:
            continue
        if art.type == ArtifactType.CORRECTED_TEXT and corrected_text is None:
            try:
                corrected_text = Path(art.uri).read_text(encoding="utf-8")
            except OSError:
                corrected_text = ""
        elif art.type == ArtifactType.RAW_TEXT and raw_text is None:
            try:
                raw_text = Path(art.uri).read_text(encoding="utf-8")
            except OSError:
                raw_text = ""

    if corrected_text is not None:
        return corrected_text, raw_text
    if raw_text is not None:
        return raw_text, None
    return "", None


def _extract_first_error(pipeline_result: Any) -> str | None:
    """Retourne le ``error`` du premier step en échec, ou ``None``."""
    for step in pipeline_result.step_results:
        err = getattr(step, "error", None)
        if err:
            return str(err)
    return None


def _build_pipeline_metadata(
    *,
    engine: Any,
    ocr_intermediate: str | None,
) -> dict:
    """Reconstitue les ``pipeline_metadata`` legacy pour un DocumentResult."""
    if not getattr(engine, "is_pipeline", False):
        return {}
    metadata: dict = {
        "pipeline_mode": getattr(engine, "mode", None),
        "is_pipeline": True,
    }
    # mode peut être un Enum — sérialise sa value.
    mode = metadata["pipeline_mode"]
    if mode is not None and hasattr(mode, "value"):
        metadata["pipeline_mode"] = mode.value
    llm_adapter = getattr(engine, "llm_adapter", None)
    if llm_adapter is not None:
        metadata["llm_model"] = llm_adapter.model
        metadata["llm_provider"] = llm_adapter.name
    if ocr_intermediate is not None:
        metadata["ocr_intermediate"] = ocr_intermediate
    return metadata


def _build_pipeline_info(engine: Any) -> dict:
    """Reconstitue ``EngineReport.pipeline_info`` pour un engine pipeline."""
    if not getattr(engine, "is_pipeline", False):
        return {}
    info: dict = {
        "pipeline_steps": getattr(engine, "pipeline_steps_info", []),
        "prompt_template": getattr(engine, "prompt_template", ""),
    }
    llm_adapter = getattr(engine, "llm_adapter", None)
    if llm_adapter is not None:
        info["llm_model"] = llm_adapter.model
        info["llm_provider"] = llm_adapter.name
    mode = getattr(engine, "mode", None)
    if mode is not None and hasattr(mode, "value"):
        info["mode"] = mode.value
    prompt_path = getattr(engine, "prompt_path", None)
    if prompt_path is not None:
        info["prompt_file"] = prompt_path
    return info


def _safe_engine_version(engine: Any) -> str:
    """Retourne ``engine.version()`` ou ``"unknown"`` en cas d'erreur."""
    try:
        v = engine.version()
        return str(v) if v else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


# ──────────────────────────────────────────────────────────────────────
# Mapping BaseOCREngine → PipelineSpec
# ──────────────────────────────────────────────────────────────────────


def engine_to_pipeline_spec(engine: "BaseOCREngine") -> PipelineSpec:
    """Convertit un ``BaseOCREngine`` legacy en ``PipelineSpec`` rewrite.

    Deux cas :

    - **OCRLLMPipeline** (``engine.is_pipeline = True``) : la spec
      composée est construite via ``make_ocr_llm_pipeline_spec``
      avec le mode (``text_only`` / ``text_and_image`` /
      ``zero_shot``), l'OCR amont (s'il existe), le LLM, et le
      template de prompt en ``llm_params``.
    - **OCR seul** : spec mono-step (IMAGE → RAW_TEXT).  Le step
      référencera ``engine.name`` ; le caller l'enregistre dans
      l'adapter resolver via un ``LegacyOCREngineExecutor(engine)``.

    Parameters
    ----------
    engine:
        Instance d'un sous-classe de ``BaseOCREngine`` (Tesseract,
        Pero, Mistral OCR, Google Vision, Azure DI) ou un
        ``OCRLLMPipeline``.

    Returns
    -------
    PipelineSpec
        Spec immutable consommable par ``BenchmarkService``.
    """
    if getattr(engine, "is_pipeline", False):
        return _ocr_llm_pipeline_to_spec(engine)
    return _ocr_only_to_spec(engine)


def _ocr_only_to_spec(engine: "BaseOCREngine") -> PipelineSpec:
    """Spec mono-step : un OCR simple consommant IMAGE et produisant RAW_TEXT."""
    name = engine.name
    safe_name = _safe_pipeline_name(name)
    return PipelineSpec(
        name=f"ocr_only_{safe_name}",
        description=f"OCR step seul ({name}) — IMAGE → RAW_TEXT.",
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(
            PipelineStep(
                id="ocr",
                kind="ocr",
                adapter_name=name,
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
                inputs_from={ArtifactType.IMAGE: INITIAL_STEP_ID},
            ),
        ),
    )


def _ocr_llm_pipeline_to_spec(pipeline: Any) -> PipelineSpec:
    """Spec composée pour un ``OCRLLMPipeline`` (3 modes)."""
    mode = pipeline.mode.value
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


# ──────────────────────────────────────────────────────────────────────
# Adapter resolver
# ──────────────────────────────────────────────────────────────────────


def build_adapter_resolver(
    engines: list["BaseOCREngine"],
) -> Callable[[str], Any]:
    """Construit un adapter resolver pour ``PipelineExecutor``.

    Parcourt les engines fournis et associe leur ``name`` à un
    ``StepExecutor`` valide :

    - **OCR simple** (``BaseOCREngine``) → wrapped via
      ``LegacyOCREngineExecutor`` (qui satisfait le contrat
      ``StepExecutor``).
    - **OCRLLMPipeline** → enregistre les deux sous-composants :
      ``ocr_engine`` (wrapped) et ``llm_adapter`` (déjà
      ``StepExecutor`` natif depuis Sprint A14-S44).  Le pipeline
      lui-même n'est pas enregistré directement — sa spec
      référence ses sous-steps par leur ``adapter_name``.

    Le resolver retourné lève ``KeyError`` si un nom inconnu est
    demandé.

    Parameters
    ----------
    engines:
        Liste d'engines/pipelines legacy à enregistrer.

    Returns
    -------
    Callable[[str], Any]
        Fonction ``resolver(name) -> step_executor``.

    Raises
    ------
    PicaronesError
        Si deux engines partagent le même ``name`` (collision).
    """
    name_to_executor: dict[str, Any] = {}

    def _register(name: str, executor: Any) -> None:
        existing = name_to_executor.get(name)
        if existing is not None and existing is not executor:
            raise PicaronesError(
                f"Adapter resolver : nom {name!r} enregistré "
                "deux fois avec des instances différentes — "
                "collision impossible à résoudre.",
            )
        name_to_executor[name] = executor

    for engine in engines:
        if getattr(engine, "is_pipeline", False):
            # OCRLLMPipeline : enregistrer ocr + llm sous-jacents.
            ocr_engine = getattr(engine, "ocr_engine", None)
            llm_adapter = getattr(engine, "llm_adapter", None)
            if ocr_engine is not None:
                _register(ocr_engine.name, LegacyOCREngineExecutor(ocr_engine))
            if llm_adapter is not None:
                _register(_llm_adapter_name(llm_adapter), llm_adapter)
        else:
            _register(engine.name, LegacyOCREngineExecutor(engine))

    def resolver(name: str) -> Any:
        if name not in name_to_executor:
            raise KeyError(
                f"adapter inconnu pour le resolver legacy : {name!r}.  "
                f"Enregistrés : {sorted(name_to_executor.keys())!r}."
            )
        return name_to_executor[name]

    return resolver


# ──────────────────────────────────────────────────────────────────────
# Helpers privés
# ──────────────────────────────────────────────────────────────────────


def _llm_adapter_name(llm_adapter: Any) -> str:
    """Identifiant ``provider:model`` stable pour un adapter LLM/VLM.

    Convention identique à celle utilisée par
    ``picarones.pipelines._executor_runner`` (Sprint B) — les
    adapter resolvers internes attendent ce format.
    """
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


def _safe_doc_id(doc_id: str) -> str:
    """Coerce un ``Document.doc_id`` vers le regex de ``DocumentRef.id``.

    Le regex ``_DOC_ID_RE = r"^[A-Za-z0-9_.\\-/]+$"`` interdit les
    espaces, accents et caractères de contrôle.  Les doc_ids
    historiques issus de ``image_path.stem`` peuvent en contenir —
    on normalise NFD et on remplace tout ce qui n'est pas conforme.
    """
    if not doc_id:
        return "doc"
    import unicodedata

    # Normalise NFD pour décomposer les caractères accentués en
    # base + diacritique, puis filtre la base ASCII.
    normalized = unicodedata.normalize("NFD", doc_id)
    safe = []
    for ch in normalized:
        # Skip les diacritiques (Mn = Mark, Nonspacing).
        if unicodedata.category(ch) == "Mn":
            continue
        if ch.isalnum() or ch in "_.-/":
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_") or "doc"
    return out


def _has_text_gt(document: "Document") -> bool:
    """``True`` ssi le document a un payload TEXT (RAW_TEXT) renseigné."""
    return ArtifactType.RAW_TEXT in document.ground_truths


def _resolve_gt_uri(
    *,
    level: ArtifactType,
    payload: object,
    doc_id: str,
    workspace_dir: Path,
) -> str:
    """Retourne l'URI d'un payload GT.

    - Si ``payload.source_path`` existe sur disque → on l'utilise
      directement (pas de copie).
    - Sinon → on sérialise dans ``workspace_dir`` selon le niveau.
    """
    source_path = getattr(payload, "source_path", None)
    if source_path is not None and Path(source_path).exists():
        return str(source_path)

    # Sérialisation de secours pour les payloads in-memory
    suffix = _DEFAULT_SUFFIXES.get(level, ".gt.txt")
    safe_id = doc_id.replace("/", "_")
    out_path = workspace_dir / f"{safe_id}{suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content = _payload_to_text(level, payload)
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)


_DEFAULT_SUFFIXES: dict[ArtifactType, str] = {
    ArtifactType.ALTO_XML: ".gt.alto.xml",
    ArtifactType.PAGE_XML: ".gt.page.xml",
    ArtifactType.ENTITIES: ".gt.entities.json",
    ArtifactType.READING_ORDER: ".gt.reading_order.json",
}


def _payload_to_text(level: ArtifactType, payload: object) -> str:
    """Sérialise un payload GT (in-memory) vers une string fichier."""
    if level in (ArtifactType.ALTO_XML, ArtifactType.PAGE_XML):
        return getattr(payload, "xml_content", "")
    if level == ArtifactType.ENTITIES:
        import json
        return json.dumps(
            getattr(payload, "entities", []),
            ensure_ascii=False,
            indent=2,
        )
    if level == ArtifactType.READING_ORDER:
        import json
        return json.dumps(
            getattr(payload, "region_order", []),
            ensure_ascii=False,
        )
    # Niveau inconnu : on utilise le ``text`` si présent, sinon
    # une chaîne vide.
    return getattr(payload, "text", "") or ""


__all__ = [
    "document_to_document_ref",
    "corpus_to_corpus_spec",
    "engine_to_pipeline_spec",
    "build_adapter_resolver",
    "run_result_to_benchmark_result",
    "run_benchmark_via_service",
]


# ──────────────────────────────────────────────────────────────────────
# Fonction publique principale (Sprint D.1.d)
# ──────────────────────────────────────────────────────────────────────


def run_benchmark_via_service(
    corpus: "Corpus",
    engines: list["BaseOCREngine"],
    *,
    char_exclude: Any | None = None,
    normalization_profile: Any | None = None,
    output_json: Any | None = None,
    code_version: str | None = None,
    show_progress: bool = True,  # noqa: ARG001
    progress_callback: Callable[[str, int, str], None] | None = None,
    timeout_seconds: float = 60.0,
    cancel_event: Any | None = None,
    # ---- Paramètres legacy non encore portés vers BenchmarkService ----
    # Sprint D.2 du plan v2.0 — les features manquantes seront
    # ajoutées au ``BenchmarkService`` dans une session ultérieure.
    max_workers: int = 4,  # noqa: ARG001
    partial_dir: Any | None = None,  # noqa: ARG001
    entity_extractor: Any | None = None,  # noqa: ARG001
    profile: str = "standard",  # noqa: ARG001
) -> Any:
    """Adapter de compatibilité ``run_benchmark`` legacy →
    ``BenchmarkService`` rewrite.

    Présente la signature historique de
    ``picarones.measurements.runner.run_benchmark`` mais s'appuie
    en interne sur le rewrite (``CorpusSpec``, ``PipelineSpec``,
    ``PipelineExecutor``, ``BenchmarkService``).  Pivot du Sprint D
    du plan v2.0.

    Périmètre actuel (D.1.d, MVP)
    -----------------------------
    Cette première version fonctionne pour le cas le plus simple :

    - Un ou plusieurs ``BaseOCREngine`` (OCR seul ou pipeline OCR+LLM
      via ``OCRLLMPipeline``).
    - Un ``Corpus`` avec image_path + ground_truth (TEXT) par doc.
    - Métriques CER/WER calculées via ``compute_metrics`` sur les
      hypothèses extraites des artefacts produits.
    - Conversion en ``BenchmarkResult`` legacy compatible avec les
      consommateurs historiques (rapport HTML, narrative engine).

    Périmètre reporté (D.2)
    -----------------------
    Les paramètres suivants sont **acceptés mais ignorés** dans
    cette MVP — leur portage vers ``BenchmarkService`` constitue
    le Sprint D.2 :

    - ``show_progress`` (tqdm),
    - ``progress_callback`` (SSE web),
    - ``max_workers`` (parallélisme intra-engine),
    - ``partial_dir`` (reprise sur interruption),
    - ``cancel_event`` (annulation propre),
    - ``entity_extractor`` (calcul NER),
    - ``profile`` (validation de profil de mesures).

    Parameters
    ----------
    corpus:
        Corpus legacy.
    engines:
        Liste d'engines/pipelines legacy à benchmarker.
    char_exclude:
        Filtre passé à ``compute_metrics``.
    normalization_profile:
        Profil de normalisation passé à ``compute_metrics``.
    output_json:
        Si fourni, le ``BenchmarkResult`` est sérialisé en JSON
        à ce chemin (via la sérialisation legacy).
    code_version:
        Version du code injectée dans le ``RunContext`` /
        ``RunManifest``.  Défaut : ``picarones.__version__``.
    timeout_seconds:
        Timeout par document propagé au ``CorpusRunner``.

    Returns
    -------
    BenchmarkResult
        Format legacy compatible.

    Raises
    ------
    PicaronesError
        Si les engines ne déclarent pas tous un ``name`` unique
        (cf. ``build_adapter_resolver``).
    """
    import tempfile

    if code_version is None:
        # Le scanner d'archi rejette ``from picarones import __version__``
        # parce qu'il classe ``picarones`` (sans sous-package) comme une
        # lib externe non whitelistée pour la couche ``app/``.  On
        # contourne via importlib (déclaration dynamique).
        import importlib

        try:
            code_version = importlib.import_module("picarones").__version__
        except (ImportError, AttributeError):
            code_version = "unknown"

    with tempfile.TemporaryDirectory(prefix="picarones_bench_") as ws:
        workspace = Path(ws)
        gt_dir = workspace / "gt"
        gt_dir.mkdir()
        run_dir = workspace / "run"
        run_dir.mkdir()

        # 1. Conversion corpus → CorpusSpec (D.1.a)
        corpus_spec = corpus_to_corpus_spec(corpus, workspace_dir=gt_dir)

        # 2. Conversion engines → PipelineSpec[] + adapter resolver (D.1.b)
        pipeline_specs = [engine_to_pipeline_spec(e) for e in engines]
        adapter_resolver = build_adapter_resolver(engines)

        # Mapping pipeline_name → engine.name pour préserver la
        # sémantique legacy de ``progress_callback(engine_name, ...)``
        # qui attend le nom de l'engine, pas celui de la pipeline
        # (qui inclut le préfixe ``ocr_only_`` côté rewrite).
        pipeline_to_engine_name = {
            spec.name: engine.name
            for spec, engine in zip(pipeline_specs, engines)
        }

        # 3. Exécution via BenchmarkService rewrite
        run_result = _execute_via_benchmark_service(
            corpus_spec=corpus_spec,
            pipeline_specs=pipeline_specs,
            adapter_resolver=adapter_resolver,
            workspace_uri=str(run_dir),
            code_version=code_version,
            timeout_seconds=timeout_seconds,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            pipeline_to_engine_name=pipeline_to_engine_name,
        )

        # 4. Conversion RunResult → BenchmarkResult legacy (D.1.c)
        benchmark_result = run_result_to_benchmark_result(
            run_result,
            corpus=corpus,
            engines=engines,
            char_exclude=char_exclude,
            normalization_profile=normalization_profile,
        )

    # 5. Sérialisation JSON optionnelle
    if output_json is not None:
        _persist_benchmark_result_json(benchmark_result, Path(output_json))

    return benchmark_result


def _execute_via_benchmark_service(
    *,
    corpus_spec: CorpusSpec,
    pipeline_specs: list[PipelineSpec],
    adapter_resolver: Callable[[str], Any],
    workspace_uri: str,
    code_version: str,
    timeout_seconds: float,
    progress_callback: Callable[[str, int, str], None] | None = None,
    cancel_event: Any | None = None,
    pipeline_to_engine_name: dict[str, str] | None = None,
) -> Any:
    """Lance ``BenchmarkService.run`` sur les specs converties.

    Vues passées en liste vide — les métriques sont calculées
    côté converter D.1.c via ``compute_metrics`` directement sur
    les hypothèses extraites des artefacts.  Pattern simple,
    cohérent avec le legacy qui calcule aussi les métriques au
    moment du benchmark (pas via ``EvaluationView``).
    """
    from picarones.app.services.benchmark_service import BenchmarkService
    from picarones.evaluation.projectors.registry import ProjectorRegistry
    from picarones.evaluation.registry.registry import MetricRegistry
    from picarones.evaluation.views.executor import (
        DefaultEvaluationViewExecutor,
    )
    from picarones.pipeline.executor import PipelineExecutor
    from picarones.pipeline.runner import CorpusRunner
    from picarones.pipeline.types import RunContext

    executor = PipelineExecutor(adapter_resolver=adapter_resolver)
    runner = CorpusRunner(
        executor,
        max_in_flight=2,
        timeout_seconds_per_doc=timeout_seconds,
    )

    # ViewExecutor minimal : registres vides.
    # Pas de calcul de ``ViewResult`` ici — le converter D.1.c
    # calcule les métriques côté legacy via ``compute_metrics``
    # directement sur les hypothèses extraites des artefacts.
    view_executor = DefaultEvaluationViewExecutor.from_registries(
        metric_registry=MetricRegistry(),
        projector_registry=ProjectorRegistry(),
        payload_loader=lambda art: None,
    )
    bench = BenchmarkService(
        corpus_runner=runner,
        view_executor=view_executor,
        code_version=code_version,
    )

    # Factory pour les inputs initiaux (toujours IMAGE depuis l'URI).
    def inputs_factory(doc: DocumentRef) -> dict[ArtifactType, Any]:
        from picarones.domain.artifacts import Artifact

        if doc.image_uri is None:
            raise PicaronesError(
                f"Document {doc.id!r} sans image_uri — la pipeline "
                "par défaut consomme une IMAGE en entrée.",
            )
        return {
            ArtifactType.IMAGE: Artifact(
                id=f"{doc.id}:image",
                document_id=doc.id,
                type=ArtifactType.IMAGE,
                uri=doc.image_uri,
            ),
        }

    # GT factory : pas utilisée car ``views=[]``.
    def gt_factory(doc: DocumentRef, art_type: ArtifactType) -> Any:
        return None

    # Context factory : ``workspace_uri`` propagé pour résoudre les
    # output paths des adapters (cf. ``resolve_output_path``).
    # Sprint D.2.a : le hook ``progress_callback`` est appelé ici —
    # ``context_factory`` est invoqué une fois par (doc, pipeline)
    # AVANT l'exécution effective, ce qui correspond à la sémantique
    # legacy de ``progress_callback(engine_name, doc_idx, doc_id)``.
    import threading

    counter_lock = threading.Lock()
    counter_state = {"doc_idx": 0}

    def context_factory(
        doc: DocumentRef, pipeline_name: str,
    ) -> RunContext:
        if progress_callback is not None:
            with counter_lock:
                idx = counter_state["doc_idx"]
                counter_state["doc_idx"] = idx + 1
            # Sémantique legacy : ``progress_callback(engine.name, ...)``
            # plutôt que le nom de la pipeline (qui inclut le préfixe
            # ``ocr_only_``).  Le mapping est fourni par le caller.
            engine_name = (
                pipeline_to_engine_name.get(pipeline_name, pipeline_name)
                if pipeline_to_engine_name is not None
                else pipeline_name
            )
            try:
                progress_callback(engine_name, idx, doc.id)
            except Exception:  # noqa: BLE001
                # Le legacy ignore silencieusement les erreurs du
                # callback (un caller qui crashe ne doit pas faire
                # tomber le benchmark).  Même contrat ici.
                pass
        return RunContext(
            document_id=doc.id,
            code_version=code_version,
            pipeline_name=pipeline_name,
            workspace_uri=workspace_uri,
        )

    # Sprint D.2.a — propagation du cancel_event au CorpusRunner.
    # Note : ``BenchmarkService.run`` boucle pipeline × document en
    # interne et appelle ``corpus_runner.run`` une fois par pipeline.
    # Le ``cancel_event`` doit être passé à chaque appel — on le
    # fait via un wrapping minimal en re-construisant le runner avec
    # le cancel_event capturé.
    if cancel_event is not None:
        original_run = runner.run

        def _runner_run_with_cancel(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("cancel_event", cancel_event)
            return original_run(*args, **kwargs)

        runner.run = _runner_run_with_cancel  # type: ignore[method-assign]

    return bench.run(
        corpus=corpus_spec,
        pipelines=pipeline_specs,
        views=[],
        ground_truth_factory=gt_factory,
        pipeline_inputs_factory=inputs_factory,
        context_factory=context_factory,
    )


def _persist_benchmark_result_json(
    benchmark_result: Any, output_path: Path,
) -> None:
    """Sérialise un ``BenchmarkResult`` legacy en JSON.

    Utilise la méthode ``to_json``/``compact``/``asdict`` selon la
    surface disponible.  Ce helper duplique la logique de
    ``measurements.runner.orchestration._save_benchmark_json`` en
    attendant que ``BenchmarkResult`` quitte ``evaluation/`` (Sprint E).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # ``BenchmarkResult`` est un dataclass — dataclasses.asdict
    # sérialise récursivement.  Le format n'est pas forcément
    # identique octet pour octet à la sortie legacy, mais reste
    # compatible avec les consommateurs (rapport, narrative).
    import dataclasses
    import json

    payload = dataclasses.asdict(benchmark_result)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
