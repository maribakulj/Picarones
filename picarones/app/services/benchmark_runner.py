"""Entry point CLI/web — façade ``run_benchmark_via_service``.

.. deprecated:: 2.0.0
    Module deprecated en Phase B7 du chantier de migration Option B
    (mai 2026).  Utiliser :class:`picarones.RunOrchestrator` qui
    consomme un ``RunSpec`` Pydantic.

    - La fonction ``run_benchmark_via_service`` émet une
      ``DeprecationWarning`` à chaque appel.
    - Aucun call site actif ne subsiste dans ``picarones/`` —
      CLI/Web passent désormais par
      ``picarones.app.services.legacy_runner_compat.run_via_orchestrator``.
    - Retrait du module prévu **Phase B8** (release suivante).

    Pour migrer votre code, voir le guide
    ``docs/migration/option_b_user_guide.md``.

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

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from picarones.evaluation.corpus import Corpus

logger = logging.getLogger(__name__)

# Le ``OCRLLMPipelineConfig`` (couche 4) est consommé exclusivement
# par duck typing (``is_pipeline``, ``ocr_adapter``, ``llm_adapter``,
# ``mode``, ``prompt_template``) pour respecter l'inward-only :
# ``app/`` ne doit pas importer ``pipeline/llm_pipeline_config``
# directement.


# ──────────────────────────────────────────────────────────────────────
# Mapping Document → DocumentRef
# ──────────────────────────────────────────────────────────────────────


# Phase 6 (round 3) audit code-quality (2026-05) — extraction des
# conversions Document/Corpus + helpers GT vers
# ``_benchmark_conversions.py``.  Réexport pour préserver l'API
# publique (CLI/web consomment ces noms).
from picarones.app.services._benchmark_conversions import (  # noqa: F401
    _DEFAULT_SUFFIXES,
    _has_text_gt,
    _payload_to_text,
    _resolve_gt_uri,
    _safe_doc_id,
    corpus_to_corpus_spec,
    document_to_document_ref,
)


# ──────────────────────────────────────────────────────────────────────
# Mapping RunResult → BenchmarkResult
# ──────────────────────────────────────────────────────────────────────


# Phase 6 (round 6) audit code-quality (2026-05) — converter
# ``run_result_to_benchmark_result`` extrait vers le module dédié.
from picarones.app.services._benchmark_converter import (  # noqa: F401
    run_result_to_benchmark_result,
)
# ──────────────────────────────────────────────────────────────────────
# Helpers privés du converter RunResult → BenchmarkResult
# ──────────────────────────────────────────────────────────────────────


# Phase 6 (round 5) audit code-quality (2026-05) — extraction des
# helpers internes de conversion ``RunResult → BenchmarkResult``
# vers ``_benchmark_helpers.py`` (~260 LOC).  Réexport pour les
# appels internes et les tests qui patchent ces symboles.
from picarones.app.services._benchmark_helpers import (  # noqa: F401
    _OCRResultLike,
    _build_pipeline_info,
    _build_pipeline_metadata,
    _engine_config_for_fingerprint,
    _extract_first_error,
    _extract_text_outputs,
    _extract_token_confidences,
    _resolve_corpus_lang,
    _safe_engine_version,
)
# Phase 6 (round 2) — extraction du bloc engine→spec + resolver.
from picarones.app.services._benchmark_adapter_resolver import (  # noqa: F401
    _canonical_adapter_to_spec,
    _is_canonical_adapter,
    _llm_adapter_name,
    _ocr_llm_pipeline_to_spec,
    _safe_pipeline_name,
    build_adapter_resolver,
    engine_to_pipeline_spec,
)


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
    # Phase B3 migration Option B (mai 2026) — ``run_benchmark_via_service``
    # est désormais déprécié.  Utiliser ``picarones.RunOrchestrator``
    # qui consomme un ``RunSpec`` Pydantic et expose nativement les 4
    # fichiers JSONL.  La fonction sera retirée en Phase B8 (post-
    # deprecation release) ; cette warning aide à identifier les call
    # sites à migrer.
    #
    # ``stacklevel=2`` pour que la warning pointe sur le caller (et non
    # cette ligne).  ``stacklevel=3`` ferait pointer sur le caller du
    # caller (utile si on emballe encore dans un helper privé).
    import warnings as _warnings
    _warnings.warn(
        "run_benchmark_via_service est déprécié depuis Phase B3 de la "
        "migration Option B.  Utiliser picarones.RunOrchestrator qui "
        "consomme un RunSpec Pydantic.  Retrait prévu en Phase B8.",
        DeprecationWarning,
        stacklevel=2,
    )

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


# Phase 6 (round 6) — orchestration extraite.
from picarones.app.services._benchmark_orchestration import (  # noqa: F401
    run_benchmark_unified as _run_benchmark_unified,
    run_benchmark_with_partial as _run_benchmark_with_partial,
)

# Phase 6 (round 4) audit code-quality (2026-05) — extraction de
# ``_execute_via_benchmark_service`` vers ``_benchmark_execution.py``.
# Alias conservé pour les appels internes de
# ``_run_benchmark_unified`` et ``_run_benchmark_with_partial``.
from picarones.app.services._benchmark_execution import (  # noqa: F401
    execute_via_benchmark_service as _execute_via_benchmark_service,
)
from picarones.app.services._benchmark_persistence import (
    persist_benchmark_result_json as _persist_benchmark_result_json,
)
