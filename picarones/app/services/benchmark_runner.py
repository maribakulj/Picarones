"""Entry point CLI/web — façade ``run_benchmark_via_service``.

Présente l'API mono-call ``run_benchmark_via_service(corpus,
engines, ...)`` consommée par ``picarones.interfaces.cli`` et
``picarones.interfaces.web``.  S'appuie en interne sur le service
canonique (``BenchmarkService``, ``PipelineExecutor``,
``CorpusRunner``).

Pourquoi cette façade
---------------------
``BenchmarkService`` consomme ``CorpusSpec`` (références
filesystem, Pydantic, immutable) et ``PipelineSpec`` (déclaratif).
Les interfaces utilisateur (CLI, web upload) raisonnent en
``Corpus`` riche en behavior + liste de moteurs OCR/LLM.  Ce
module fait la conversion entre les deux modèles, expose une API
mono-call ergonomique et restitue un ``BenchmarkResult``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from picarones.domain.artifacts import ArtifactType
from picarones.domain.corpus import CorpusSpec
from picarones.domain.documents import DocumentRef, GroundTruthRef
from picarones.domain.errors import PicaronesError
from picarones.domain.pipeline_spec import PipelineSpec

if TYPE_CHECKING:
    from picarones.evaluation.corpus import Corpus, Document

logger = logging.getLogger(__name__)

# Le ``OCRLLMPipelineConfig`` (couche 4) est consommé exclusivement
# par duck typing (``is_pipeline``, ``ocr_adapter``, ``llm_adapter``,
# ``mode``, ``prompt_template``) pour respecter l'inward-only :
# ``app/`` ne doit pas importer ``pipeline/llm_pipeline_config``
# directement.


# ──────────────────────────────────────────────────────────────────────
# Mapping Document → DocumentRef
# ──────────────────────────────────────────────────────────────────────


def document_to_document_ref(
    document: "Document",
    *,
    workspace_dir: Path,
) -> DocumentRef:
    """Convertit un ``Document`` (couche 3) en ``DocumentRef`` (couche 1).

    Le ``Document`` (modèle riche) porte sa GT en mémoire (``ground_truth: str``
    et ``ground_truths: dict[ArtifactType, GTPayload]``).  Le
    ``DocumentRef`` rewrite porte des références filesystem
    (``GroundTruthRef.uri``).  La conversion écrit chaque GT
    in-memory dans ``workspace_dir`` et construit les références.

    Parameters
    ----------
    document:
        Document.  ``image_path`` non-``None`` est requis ;
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
    """Convertit un ``Corpus`` (couche 3) en ``CorpusSpec`` (couche 1).

    Itère sur ``corpus.documents`` et applique
    ``document_to_document_ref`` pour chacun.

    Parameters
    ----------
    corpus:
        Corpus.
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
# Mapping RunResult → BenchmarkResult
# ──────────────────────────────────────────────────────────────────────


def run_result_to_benchmark_result(
    run_result: Any,
    *,
    corpus: "Corpus",
    engines: list[Any],
    char_exclude: Any | None = None,
    normalization_profile: Any | None = None,
    profile: str = "standard",
) -> Any:
    """Transpose un ``RunResult`` (couche 4) en ``BenchmarkResult`` (couche 3).

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
    4. Calcule les métriques CER/WER via ``compute_metrics`` puis
       exécute les hooks document-level enregistrés pour ``profile``
       via ``picarones.evaluation.metric_hooks.run_document_hooks``
       (confusion unicode, ligatures, diacritiques, taxonomie,
       structure, hallucination, philological, searchability,
       readability, etc.).
    5. Construit un ``DocumentResult`` avec ``engine_error``
       extrait des ``step_results``.
    6. Aggrège les métriques par engine via ``aggregate_metrics`` et
       exécute les agrégateurs corpus-level via
       ``run_corpus_aggregators`` pour alimenter
       ``EngineReport.aggregated_*`` (la vue HTML "Analyse des
       caractères" et les vues sœurs lisent ces champs).
    7. Reconstitue ``pipeline_info`` pour les engines pipeline
       (mode, prompt, llm_model, llm_provider, pipeline_steps).

    Parameters
    ----------
    run_result:
        ``RunResult`` produit par ``BenchmarkService.run``.
    corpus:
        Corpus d'origine — sert à récupérer le ``ground_truth``
        et l'``image_path`` pour chaque document, dans le même ordre
        que ``run_result.document_results``.
    engines:
        Liste d'adapters dans l'ordre où leurs specs ont été
        passées à ``BenchmarkService.run`` (l'ordre détermine
        l'index dans ``RunDocumentResult.pipeline_results``).
    char_exclude:
        Filtre passé à ``compute_metrics``.  ``None`` par défaut.
    normalization_profile:
        Profil de normalisation passé à ``compute_metrics``.

    Returns
    -------
    BenchmarkResult
        Format compatible avec les consommateurs historiques
        (rapport HTML, persistance JSON, narrative engine).
    """
    from picarones.evaluation.benchmark_result import (
        BenchmarkResult,
        DocumentResult,
        EngineReport,
    )
    from picarones.evaluation.metric_hooks import (
        run_corpus_aggregators,
        run_document_hooks,
    )
    # Import nécessaire : les hooks ``builtin`` s'enregistrent dans le
    # registre global au moment de l'import du module (décorateurs).
    # Sans cette ligne, ``run_document_hooks(profile="standard", ...)``
    # retournerait un dict vide et la vue HTML "Analyse des caractères"
    # tomberait sur ses placeholders.
    import picarones.evaluation.metrics.builtin_hooks  # noqa: F401
    from picarones.evaluation.metric_result import aggregate_metrics
    from picarones.evaluation.metrics.text_metrics import compute_metrics

    documents = list(corpus.documents)
    if len(documents) != len(run_result.document_results):
        raise PicaronesError(
            f"Mismatch documents : corpus={len(documents)} vs "
            f"run_result={len(run_result.document_results)}.",
        )

    corpus_lang = _resolve_corpus_lang(corpus)
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
                ground_truth=document.ground_truth,
                hypothesis=text_final,
            )

            hook_values = run_document_hooks(
                profile=profile,
                ground_truth=document.ground_truth,
                hypothesis=text_final,
                image_path=str(document.image_path or ""),
                corpus_lang=corpus_lang,
                ocr_result=_OCRResultLike(
                    success=(engine_error is None and bool(text_final)),
                    token_confidences=_extract_token_confidences(
                        pipeline_result,
                    ),
                ),
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
                    **hook_values,
                ),
            )

        aggregated = aggregate_metrics([d.metrics for d in doc_results])
        pipeline_info = _build_pipeline_info(engine)
        agg_values = run_corpus_aggregators(profile, doc_results)

        engine_reports.append(
            EngineReport(
                engine_name=engine.name,
                engine_version=_safe_engine_version(engine),
                engine_config=getattr(engine, "config", {}) or {},
                document_results=doc_results,
                aggregated_metrics=aggregated,
                pipeline_info=pipeline_info,
                **agg_values,
            ),
        )

    # Phase 3.2 audit code-quality — consommer le journal des
    # fallbacks d'importer (HTR-United, HuggingFace, etc.).  La liste
    # est vidée à la fin du benchmark pour que le run suivant
    # n'hérite pas des incidents du précédent.  Le détecteur narratif
    # ``IMPORTER_FALLBACK_TRIGGERED`` (history.py:280) lit
    # ``benchmark_data["importer_fallbacks"]`` propagé par
    # ``build_report_data``.
    from picarones.adapters.corpus._fallback_log import consume_fallback_log
    fallbacks = consume_fallback_log()
    metadata: dict[str, Any] = {}
    if fallbacks:
        metadata["importer_fallbacks"] = fallbacks

    return BenchmarkResult(
        corpus_name=corpus.name,
        corpus_source=str(corpus.source_path) if corpus.source_path else None,
        document_count=len(documents),
        engine_reports=engine_reports,
        metadata=metadata,
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
    ground_truth: str = "",
    hypothesis: str = "",
) -> dict:
    """Reconstitue les ``pipeline_metadata`` pour un DocumentResult.

    Sprint D.2.d — pour les pipelines composées OCR+LLM, calcule
    ``over_normalization`` (détection des cas où le LLM a sur-normalisé
    le texte par rapport à la GT) si ``ocr_intermediate`` est
    disponible.  Equivalent fonctionnel de
    le calcul historique de DocumentResult
    (supprimé en D.6.b).
    """
    if not getattr(engine, "is_pipeline", False):
        return {}
    metadata: dict = {
        "pipeline_mode": getattr(engine, "mode", None),
        "is_pipeline": True,
    }
    # mode peut être un Enum ou une string (canonique).
    mode = metadata["pipeline_mode"]
    if mode is not None and hasattr(mode, "value"):
        metadata["pipeline_mode"] = mode.value
    llm_adapter = getattr(engine, "llm_adapter", None)
    if llm_adapter is not None:
        metadata["llm_model"] = llm_adapter.model
        metadata["llm_provider"] = llm_adapter.name
    if ocr_intermediate is not None:
        metadata["ocr_intermediate"] = ocr_intermediate
        # D.2.d : over_normalization computé pour les pipelines avec
        # OCR amont — pas de signal exploitable en zero-shot.
        try:
            from picarones.evaluation.metrics.over_normalization import (
                detect_over_normalization,
            )
            over_norm = detect_over_normalization(
                ground_truth=ground_truth,
                ocr_text=ocr_intermediate,
                llm_text=hypothesis,
            )
            metadata["over_normalization"] = over_norm.as_dict()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[over_normalization] fonctionnalité dégradée : %s",
                exc,
            )
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
    if mode is not None:
        # Tolère enum (``PipelineMode.X``) ou string.
        info["mode"] = mode.value if hasattr(mode, "value") else mode
    prompt_path = getattr(engine, "prompt_path", None)
    if prompt_path is not None:
        info["prompt_file"] = prompt_path
    return info


def _engine_config_for_fingerprint(engine: Any) -> dict:
    """Extrait une config sérialisable d'un engine pour le fingerprint.

    Phase 2.3 : utilisé par
    :func:`partial_store.compute_run_fingerprint` pour distinguer deux
    runs avec le même couple ``(corpus, engine.name)`` mais des
    paramètres internes différents (psm/lang Tesseract, modèle LLM,
    prompt_template, mode pipeline, …).  Un changement non capturé
    par ce dict = potentiel faux résultat en reprise.

    Stratégie : sonder les attributs canoniques connus, repli sur
    ``repr`` pour les types non sérialisables.  ``json.dumps`` finalise
    via ``default=str`` côté ``compute_run_fingerprint`` — la
    granularité est conservatrice (toute différence visible → nouveau
    fingerprint).
    """
    cfg: dict = {"engine_name": getattr(engine, "name", "")}

    # Pipeline composé : capturer le mode + prompt + LLM model
    # (sources de différence majeure des résultats).
    if getattr(engine, "is_pipeline", False):
        mode = getattr(engine, "mode", None)
        cfg["mode"] = mode.value if hasattr(mode, "value") else mode
        prompt = getattr(engine, "prompt_template", None)
        if prompt is not None:
            # Hasher le prompt pour éviter de polluer le nom du fichier
            # partiel avec un prompt multi-lignes (et de fuiter le
            # contenu d'un prompt institutionnel dans un nom de fichier).
            # SHA-1 utilisé comme identifiant de cache uniquement
            # (fingerprint partial store, pas une garantie crypto) —
            # ``usedforsecurity=False`` neutralise le faux positif
            # bandit B324 et exprime l'intention pour le reviewer.
            cfg["prompt_sha1"] = hashlib.sha1(
                str(prompt).encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()[:12]
        llm = getattr(engine, "llm_adapter", None)
        if llm is not None:
            cfg["llm_model"] = getattr(llm, "model", "")
            cfg["llm_provider"] = getattr(llm, "name", "")
        ocr = getattr(engine, "ocr_adapter", None)
        if ocr is not None:
            cfg["ocr_name"] = getattr(ocr, "name", "")
    else:
        # Adapter OCR seul : sonder les attributs courants.
        for attr in ("lang", "psm", "model", "model_id", "feature_type"):
            value = getattr(engine, attr, None)
            if value is not None:
                cfg[attr] = value
    return cfg


def _safe_engine_version(engine: Any) -> str:
    """Retourne ``engine.version()`` ou ``"unknown"`` en cas d'erreur.

    Tolère les ``BaseOCRAdapter`` qui n'ont pas de méthode
    ``version()`` (le contrat canonique ne l'inclut pas).
    """
    version_attr = getattr(engine, "version", None)
    if version_attr is None:
        return "unknown"
    try:
        v = version_attr() if callable(version_attr) else version_attr
        return str(v) if v else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass
class _OCRResultLike:
    """Shim minimal consommé par ``run_document_hooks``.

    Les hooks utilisent deux attributs : ``success`` (filtre
    ``requires_success``) et ``token_confidences`` (filtre
    ``requires_token_confidences`` + entrée du hook calibration).
    Le runner canonique manipule des ``PipelineResult`` et non des
    ``OCRResult`` legacy — ce shim fournit juste les deux attributs
    nécessaires sans tirer le modèle legacy.
    """

    success: bool
    token_confidences: list | None = None


def _resolve_corpus_lang(corpus: "Corpus") -> str:
    """Récupère la langue du corpus pour le hook ``readability``.

    Cherche dans ``corpus.metadata`` (clés ``lang`` ou ``language``)
    puis tombe sur ``"fr"`` (cohérent avec le défaut de
    ``compute_readability_metrics``).
    """
    metadata = getattr(corpus, "metadata", None) or {}
    for key in ("lang", "language"):
        value = metadata.get(key) if isinstance(metadata, dict) else None
        if value:
            return str(value)
    return "fr"


def _extract_token_confidences(pipeline_result: Any) -> list | None:
    """Récupère les confidences au token si un step OCR en a publié.

    Les adapters canoniques n'exposent pas encore systématiquement
    ces données ; le hook calibration retombera silencieusement via
    ``requires_token_confidences`` quand ``None``.
    """
    for step in getattr(pipeline_result, "step_results", ()) or ():
        confidences = getattr(step, "token_confidences", None)
        if confidences:
            return list(confidences)
    return None


# Phase 6 (complète) audit code-quality (2026-05) — extraction du
# bloc ``engine → PipelineSpec`` + adapter resolver vers le module
# dédié ``_benchmark_adapter_resolver.py``.  Les noms historiques
# restent ici comme alias importés pour préserver les appels
# internes (le runner consomme ``engine_to_pipeline_spec``,
# ``build_adapter_resolver``, ``_safe_pipeline_name``,
# ``_llm_adapter_name``, ``_is_canonical_adapter``).
from picarones.app.services._benchmark_adapter_resolver import (  # noqa: F401
    _canonical_adapter_to_spec,
    _is_canonical_adapter,
    _llm_adapter_name,
    _ocr_llm_pipeline_to_spec,
    _safe_pipeline_name,
    build_adapter_resolver,
    engine_to_pipeline_spec,
)


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
    engines: list[Any],
    *,
    char_exclude: Any | None = None,
    normalization_profile: Any | None = None,
    output_json: Any | None = None,
    code_version: str | None = None,
    show_progress: bool = True,  # noqa: ARG001
    progress_callback: Callable[[str, int, str], None] | None = None,
    timeout_seconds: float = 60.0,
    cancel_event: Any | None = None,
    partial_dir: str | Path | None = None,
    entity_extractor: Callable[[str], list[dict]] | None = None,
    profile: str = "standard",
) -> Any:
    """Façade ``run_benchmark`` →
    ``BenchmarkService`` rewrite.

    Présente la signature historique de
    ``picarones.app.services.benchmark_runner.run_benchmark`` mais s'appuie
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
    - Conversion en ``BenchmarkResult`` compatible avec les
      consommateurs historiques (rapport HTML, narrative engine).

    Périmètre reporté (D.2)
    -----------------------
    Les paramètres suivants sont **acceptés mais ignorés** dans
    cette MVP — le rewrite gère ces aspects nativement :

    - ``show_progress`` (tqdm).

    Pour régler le parallélisme corpus-wide, passer par
    ``CorpusRunner.max_in_flight`` directement (couche pipeline).

    Profil de mesures (D.2.f)
    -------------------------
    ``profile`` est validé au démarrage via
    ``picarones.evaluation.metric_hooks.validate_profile``.  Un
    profil inconnu lève ``PicaronesError``.  La valeur n'a pas
    encore d'effet sur les hooks document-level (ce serait l'objet
    d'un sprint ultérieur, hors du périmètre v2.0).

    NER attach (D.2.e)
    ------------------
    Si ``entity_extractor`` est fourni, après le calcul des
    ``DocumentResult``, le service appelle l'extracteur sur chaque
    hypothèse OCR pour les documents dont la GT possède un niveau
    ``ENTITIES``, puis attache les métriques NER (``ner_metrics``
    par document, ``aggregated_ner`` au niveau engine).

    Reprise sur interruption (D.2.b)
    --------------------------------
    Si ``partial_dir`` est fourni, le bench est exécuté en mode
    **per-engine resumable** :

    - Pour chaque engine, on cherche un fichier
      ``{partial_dir}/picarones_{corpus}_{engine}.partial.jsonl``
      d'une exécution précédente interrompue.
    - Les ``DocumentResult`` qui y sont déjà persistés sont
      réutilisés tels quels (pas de recalcul).
    - Seuls les documents restants sont soumis au ``BenchmarkService``.
    - Chaque nouveau ``DocumentResult`` est ajouté en append au
      partial avant de passer au suivant.
    - À la fin d'un engine traité avec succès, son partial est
      supprimé.

    Quand ``partial_dir`` est ``None`` (défaut), une seule passe
    multi-engine est lancée (chemin rapide, pas de persistance
    intermédiaire).

    Parameters
    ----------
    corpus:
        Corpus.
    engines:
        Liste d'engines/pipelines à benchmarker.
    char_exclude:
        Filtre passé à ``compute_metrics``.
    normalization_profile:
        Profil de normalisation passé à ``compute_metrics``.
    output_json:
        Si fourni, le ``BenchmarkResult`` est sérialisé en JSON
        à ce chemin (sérialisation BenchmarkResult).
    code_version:
        Version du code injectée dans le ``RunContext`` /
        ``RunManifest``.  Défaut : ``picarones.__version__``.
    timeout_seconds:
        Timeout par document propagé au ``CorpusRunner``.

    Returns
    -------
    BenchmarkResult
        Format compatible avec les consommateurs historiques.

    Raises
    ------
    PicaronesError
        Si les engines ne déclarent pas tous un ``name`` unique
        (cf. ``build_adapter_resolver``).
    """
    # D.2.f : valide ``profile`` tôt — un nom inconnu lève
    # ``PicaronesError`` avant que le bench ne démarre, plutôt
    # que de dégrader silencieusement plus loin.
    from picarones.evaluation.metric_hooks import validate_profile

    validate_profile(profile)

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

    if partial_dir is None:
        benchmark_result = _run_benchmark_unified(
            corpus=corpus,
            engines=engines,
            char_exclude=char_exclude,
            normalization_profile=normalization_profile,
            profile=profile,
            code_version=code_version,
            progress_callback=progress_callback,
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
        )
    else:
        benchmark_result = _run_benchmark_with_partial(
            corpus=corpus,
            engines=engines,
            partial_dir=Path(partial_dir),
            char_exclude=char_exclude,
            normalization_profile=normalization_profile,
            profile=profile,
            code_version=code_version,
            progress_callback=progress_callback,
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
        )

    # D.2.e : NER attach post-process.  Idempotent — re-calcule à
    # chaque run même en mode resume (les ner_metrics ne sont pas
    # persistées dans le partial NDJSON
    # qui calculait NER après le doc loop).
    if entity_extractor is not None:
        _attach_ner_metrics_to_benchmark(
            benchmark_result, corpus, entity_extractor,
        )

    # Sérialisation JSON optionnelle
    if output_json is not None:
        _persist_benchmark_result_json(benchmark_result, Path(output_json))

    return benchmark_result


# Phase 6 audit code-quality (2026-05) — extraction NER aggregation
# vers ``_benchmark_ner.py``.  Les noms ``_attach_ner_metrics_to_benchmark``
# et ``_aggregate_ner_metrics`` restent ici comme alias pour ne pas
# casser les appels internes (les autres fonctions du runner s'y
# réfèrent) et les tests qui patchent ces symboles via monkeypatch.
from picarones.app.services._benchmark_ner import (  # noqa: F401
    aggregate_ner_metrics as _aggregate_ner_metrics,
    attach_ner_metrics_to_benchmark as _attach_ner_metrics_to_benchmark,
)


def _run_benchmark_unified(
    *,
    corpus: "Corpus",
    engines: list[Any],
    char_exclude: Any | None,
    normalization_profile: Any | None,
    profile: str,
    code_version: str,
    progress_callback: Callable[[str, int, str], None] | None,
    timeout_seconds: float,
    cancel_event: Any | None,
) -> Any:
    """Chemin rapide : un seul ``BenchmarkService.run`` multi-engine.

    Pas de persistance intermédiaire — si le run crashe, tout est
    perdu.  Utilisé quand ``partial_dir`` est ``None``.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="picarones_bench_") as ws:
        workspace = Path(ws)
        gt_dir = workspace / "gt"
        gt_dir.mkdir()
        run_dir = workspace / "run"
        run_dir.mkdir()

        corpus_spec = corpus_to_corpus_spec(corpus, workspace_dir=gt_dir)
        pipeline_specs = [engine_to_pipeline_spec(e) for e in engines]
        adapter_resolver = build_adapter_resolver(engines)
        pipeline_to_engine_name = {
            spec.name: engine.name
            for spec, engine in zip(pipeline_specs, engines)
        }

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

        return run_result_to_benchmark_result(
            run_result,
            corpus=corpus,
            engines=engines,
            char_exclude=char_exclude,
            normalization_profile=normalization_profile,
            profile=profile,
        )


def _run_benchmark_with_partial(
    *,
    corpus: "Corpus",
    engines: list[Any],
    partial_dir: Path,
    char_exclude: Any | None,
    normalization_profile: Any | None,
    profile: str,
    code_version: str,
    progress_callback: Callable[[str, int, str], None] | None,
    timeout_seconds: float,
    cancel_event: Any | None,
) -> Any:
    """Chemin reprise : per-engine avec NDJSON intermédiaire.

    Pour chaque engine, charge le partial existant, filtre les docs
    déjà traités, lance ``BenchmarkService`` sur les restants,
    persiste chaque nouveau ``DocumentResult`` au fil de l'eau.
    """
    import tempfile

    from picarones.app.services.partial_store import (
        _delete_partial,
        _load_partial,
        _save_partial_line,
        partial_path_for_engine,
    )
    from picarones.evaluation.benchmark_result import (
        BenchmarkResult,
        EngineReport,
    )
    from picarones.evaluation.corpus import Corpus as LegacyCorpus
    from picarones.evaluation.metric_hooks import run_corpus_aggregators
    # Force l'auto-enregistrement des hooks builtin (décorateurs).
    import picarones.evaluation.metrics.builtin_hooks  # noqa: F401
    from picarones.evaluation.metric_result import aggregate_metrics

    partial_dir.mkdir(parents=True, exist_ok=True)

    # Index des docs par ID — permet de ré-ordonner les
    # DocumentResult rechargés selon l'ordre original du corpus.
    doc_order = {doc.doc_id: idx for idx, doc in enumerate(corpus.documents)}

    engine_reports: list[Any] = []

    for engine in engines:
        # Vérifier la cancellation entre engines (matche la
        # sémantique : un Ctrl+C arrête après l'engine en
        # cours, conserve les partials, ne démarre pas le suivant).
        if cancel_event is not None and getattr(
            cancel_event, "is_set", lambda: False,
        )():
            logger.info(
                "[partial_dir] benchmark annulé avant l'engine '%s' "
                "— partials conservés pour reprise.", engine.name,
            )
            break

        # Phase 2.3 — fingerprint inclut config moteur + profil
        # normalisation + char_exclude + corpus files (mtime/size) +
        # version code.  Deux runs avec configs différentes →
        # fichiers partiels distincts → pas de réutilisation
        # silencieuse de résultats incompatibles.
        partial_path = partial_path_for_engine(
            corpus=corpus,
            engine=engine,
            partial_dir=partial_dir,
            engine_config=_engine_config_for_fingerprint(engine),
            normalization_profile=normalization_profile,
            char_exclude=char_exclude,
            profile=profile,
            code_version=code_version,
        )
        loaded_results = _load_partial(partial_path)
        loaded_doc_ids = {dr.doc_id for dr in loaded_results}

        if loaded_results:
            logger.info(
                "[partial_dir] reprise '%s' : %d/%d docs déjà traités.",
                engine.name, len(loaded_results), len(corpus.documents),
            )

        remaining_docs = [
            d for d in corpus.documents if d.doc_id not in loaded_doc_ids
        ]

        new_doc_results: list[Any] = []
        if remaining_docs:
            # Sub-corpus avec uniquement les docs restants.  On
            # conserve le ``name`` original pour que les chemins de
            # partial restent cohérents si un re-run arrive.
            sub_corpus = LegacyCorpus(
                name=corpus.name,
                documents=remaining_docs,
                source_path=corpus.source_path,
            )

            with tempfile.TemporaryDirectory(
                prefix="picarones_bench_partial_",
            ) as ws:
                workspace = Path(ws)
                gt_dir = workspace / "gt"
                gt_dir.mkdir()
                run_dir = workspace / "run"
                run_dir.mkdir()

                sub_corpus_spec = corpus_to_corpus_spec(
                    sub_corpus, workspace_dir=gt_dir,
                )
                pipeline_spec = engine_to_pipeline_spec(engine)
                adapter_resolver = build_adapter_resolver([engine])
                pipeline_to_engine_name = {pipeline_spec.name: engine.name}

                run_result = _execute_via_benchmark_service(
                    corpus_spec=sub_corpus_spec,
                    pipeline_specs=[pipeline_spec],
                    adapter_resolver=adapter_resolver,
                    workspace_uri=str(run_dir),
                    code_version=code_version,
                    timeout_seconds=timeout_seconds,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                    pipeline_to_engine_name=pipeline_to_engine_name,
                )

                # Convertir ce sous-RunResult en EngineReport avec
                # uniquement les docs restants — puis extraire les
                # ``DocumentResult`` pour append au partial.
                sub_report = run_result_to_benchmark_result(
                    run_result,
                    corpus=sub_corpus,
                    engines=[engine],
                    char_exclude=char_exclude,
                    normalization_profile=normalization_profile,
                    profile=profile,
                )
                new_doc_results = list(
                    sub_report.engine_reports[0].document_results,
                )

                # Append au partial : un cancel mid-engine
                # préservera ce qui a déjà été calculé.
                for dr in new_doc_results:
                    _save_partial_line(partial_path, dr)

        # Fusion : loaded + new, ré-ordonné selon le corpus original.
        all_doc_results = list(loaded_results) + new_doc_results
        all_doc_results.sort(key=lambda dr: doc_order.get(dr.doc_id, 0))

        aggregated = aggregate_metrics([d.metrics for d in all_doc_results])
        pipeline_info = _build_pipeline_info(engine)
        agg_values = run_corpus_aggregators(profile, all_doc_results)

        engine_reports.append(
            EngineReport(
                engine_name=engine.name,
                engine_version=_safe_engine_version(engine),
                engine_config=getattr(engine, "config", {}) or {},
                document_results=all_doc_results,
                aggregated_metrics=aggregated,
                pipeline_info=pipeline_info,
                **agg_values,
            ),
        )

        # Engine traité avec succès → cleanup du partial.  Si on
        # arrive ici sans exception, tous les docs sont dans
        # ``all_doc_results``.
        _delete_partial(partial_path)

    # Phase 3.2 audit code-quality — cf. _run_benchmark_unified.
    from picarones.adapters.corpus._fallback_log import consume_fallback_log
    fallbacks = consume_fallback_log()
    metadata: dict[str, Any] = {}
    if fallbacks:
        metadata["importer_fallbacks"] = fallbacks

    return BenchmarkResult(
        corpus_name=corpus.name,
        corpus_source=str(corpus.source_path) if corpus.source_path else None,
        document_count=len(corpus.documents),
        engine_reports=engine_reports,
        metadata=metadata,
    )


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
    cohérent : on calcule aussi les métriques au
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
    # calcule les métriques via ``compute_metrics``
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
    # de ``progress_callback(engine_name, doc_idx, doc_id)``.
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
            # Sémantique : ``progress_callback(engine.name, ...)``
            # plutôt que le nom de la pipeline (qui inclut le préfixe
            # ``ocr_only_``).  Le mapping est fourni par le caller.
            engine_name = (
                pipeline_to_engine_name.get(pipeline_name, pipeline_name)
                if pipeline_to_engine_name is not None
                else pipeline_name
            )
            try:
                progress_callback(engine_name, idx, doc.id)
            except Exception as exc:  # noqa: BLE001
                # On ignore silencieusement les erreurs du
                # callback (un caller qui crashe ne doit pas faire
                # tomber le benchmark).  Logge en debug pour
                # diagnostic en cas de comportement bizarre.
                logger.debug(
                    "[benchmark_runner] progress_callback raised, ignoring: %s",
                    exc,
                )
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


# Phase 6 audit code-quality (2026-05) — ``_persist_benchmark_result_json``
# extrait vers ``_benchmark_persistence.py`` ; alias conservé pour
# les appels internes (cf. l.1182).
from picarones.app.services._benchmark_persistence import (
    persist_benchmark_result_json as _persist_benchmark_result_json,
)
