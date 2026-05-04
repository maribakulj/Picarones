"""``BenchmarkService`` — orchestration runner + vues + persistance.

Sprint A14-S17 du rewrite ciblé.

Premier service applicatif du rewrite.  Assemble :

- ``CorpusRunner`` (S8) qui exécute N pipelines sur le corpus,
- ``DefaultEvaluationViewExecutor`` (S13) qui applique chaque vue
  aux artefacts produits par les pipelines éligibles,
- ``RunManifest`` + ``RunResult`` (S17) pour la structure
  d'agrégation,
- Persistance optionnelle sur disque en JSONL.

Périmètre S17 (assumé minimal)
------------------------------
- ``run(corpus, pipelines, views, ...)`` orchestre tout en
  séquentiel pour une exécution simple.
- Pattern d'omission explicite : pour chaque (pipeline, view), si
  les artefacts produits par le pipeline ne sont pas dans
  ``view.candidate_types``, le pipeline est OMIS de cette vue
  (pas de ``ViewResult`` factice).
- ``persist(result, output_dir)`` écrit 3 fichiers :
  - ``run_manifest.json`` — métadonnées du run.
  - ``pipeline_results.jsonl`` — un ``PipelineResult`` par ligne.
  - ``view_results.jsonl`` — un ``ViewResult`` par ligne, avec
    ``document_id`` ajouté pour reconnaître l'origine.

Reportés au S19+
----------------
- WorkspaceManager pour isoler les chemins par session
  (validation chemin, sandbox).
- Job queue / async / cancel via threading.Event.
- Cache d'artefacts entre runs.
- Recovery sur interruption.

Le S17 livre la structure d'intégration complète mais utilisable
en mode "simple call" pour démontrer la définition de done.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Iterable

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.corpus import CorpusSpec
from picarones.domain.documents import DocumentRef
from picarones.domain.evaluation_spec import EvaluationView
from picarones.domain.run_manifest import RunManifest, utcnow
from picarones.app.results import RunDocumentResult, RunResult
from picarones.evaluation.views.base import ViewResult
from picarones.evaluation.views.executor import DefaultEvaluationViewExecutor
from picarones.pipeline.runner import CorpusRunner
from picarones.pipeline.spec import PipelineSpec
from picarones.pipeline.types import PipelineResult, RunContext

logger = logging.getLogger(__name__)


#: Factory qui produit l'artefact GT d'un document pour un type donné.
#: Le caller injecte cette factory pour découpler le service de la
#: manière dont les GT sont stockées (filesystem direct, dict in-memory,
#: GT lazy-loaded depuis IIIF, ...).
GroundTruthFactory = Callable[
    [DocumentRef, ArtifactType],
    "Artifact | None",
]

#: Factory qui produit les inputs initiaux d'un pipeline pour un doc
#: (typiquement ``{IMAGE: artifact_image}``).
PipelineInputsFactory = Callable[
    [DocumentRef],
    dict[ArtifactType, Artifact],
]

#: Factory qui produit le ``RunContext`` d'un doc pour un pipeline.
ContextFactory = Callable[[DocumentRef, str], RunContext]


class BenchmarkService:
    """Orchestre l'exécution complète d'un benchmark.

    Parameters
    ----------
    corpus_runner:
        ``CorpusRunner`` injecté.  Le service ne le crée pas lui-même
        pour permettre au caller de configurer ``max_in_flight`` /
        ``timeout_seconds_per_doc`` selon son contexte.
    view_executor:
        ``DefaultEvaluationViewExecutor`` injecté avec son propre
        ``payload_loader``.  Le service ne fournit pas de loader par
        défaut.
    code_version:
        Version du code à inscrire dans le ``RunManifest``.
    """

    def __init__(
        self,
        corpus_runner: CorpusRunner,
        view_executor: DefaultEvaluationViewExecutor,
        code_version: str,
    ) -> None:
        self._runner = corpus_runner
        self._view_executor = view_executor
        self._code_version = code_version

    # ──────────────────────────────────────────────────────────────────
    # Run
    # ──────────────────────────────────────────────────────────────────

    def run(
        self,
        *,
        corpus: CorpusSpec,
        pipelines: Iterable[PipelineSpec],
        views: Iterable[EvaluationView],
        ground_truth_factory: GroundTruthFactory,
        pipeline_inputs_factory: PipelineInputsFactory,
        context_factory: ContextFactory,
        run_id: str | None = None,
        dependencies_lock: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> RunResult:
        """Exécute un benchmark complet et retourne le ``RunResult``.

        Pattern d'orchestration :

        1. Pour chaque ``pipeline`` × chaque ``document`` du corpus :
           - lance ``corpus_runner.run(spec, [doc], ...)``,
           - récupère le ``PipelineResult``.
        2. Pour chaque ``view`` :
           - pour chaque pipeline_result du doc, identifier les
             artefacts produits dont le type est dans
             ``view.candidate_types``,
           - pour chaque artefact éligible, lancer
             ``view_executor.evaluate(view, candidate, gt)`` où ``gt``
             est l'artefact GT du niveau correspondant (récupéré via
             ``ground_truth_factory``),
           - collecter les ``ViewResult`` produits.
        3. Construire ``RunManifest`` avec timestamps + version + lock.
        4. Construire ``RunResult`` avec un ``RunDocumentResult`` par
           document.
        """
        pipelines_list = list(pipelines)
        views_list = list(views)
        documents = list(corpus.documents)

        started_at = utcnow()

        # 1. Exécution séquentielle pipeline × document.
        # On boucle pipeline-par-pipeline pour bénéficier de la
        # backpressure du CorpusRunner sur les documents.
        pipeline_results_by_doc: dict[str, list[PipelineResult]] = {
            doc.id: [] for doc in documents
        }
        for spec in pipelines_list:
            corpus_result = self._runner.run(
                spec=spec,
                documents=documents,
                initial_inputs_factory=pipeline_inputs_factory,
                context_factory=lambda d, _spec_name=spec.name:
                    context_factory(d, _spec_name),
                corpus_name=corpus.name,
            )
            for outcome in corpus_result.outcomes:
                if outcome.pipeline_result is not None:
                    pipeline_results_by_doc[outcome.document_id].append(
                        outcome.pipeline_result,
                    )

        # 2. Application des vues.
        view_results_by_doc: dict[str, list[ViewResult]] = {
            doc.id: [] for doc in documents
        }
        for doc in documents:
            for vr in self._evaluate_document_in_views(
                document=doc,
                pipeline_results=pipeline_results_by_doc[doc.id],
                views=views_list,
                ground_truth_factory=ground_truth_factory,
            ):
                view_results_by_doc[doc.id].append(vr)

        # 3. Manifest.
        completed_at = utcnow()
        manifest = RunManifest(
            run_id=run_id or _default_run_id(corpus.name, started_at),
            corpus_name=corpus.name,
            n_documents=len(documents),
            pipeline_names=tuple(spec.name for spec in pipelines_list),
            view_specs=tuple(views_list),
            code_version=self._code_version,
            started_at=started_at,
            completed_at=completed_at,
            dependencies_lock=dependencies_lock or {},
            metadata=metadata or {},
        )

        # 4. RunResult.
        document_results = tuple(
            RunDocumentResult(
                document_id=doc.id,
                pipeline_results=tuple(pipeline_results_by_doc[doc.id]),
                view_results=tuple(view_results_by_doc[doc.id]),
            )
            for doc in documents
        )

        return RunResult(
            manifest=manifest,
            document_results=document_results,
        )

    # ──────────────────────────────────────────────────────────────────
    # Persistance JSONL
    # ──────────────────────────────────────────────────────────────────

    def persist(
        self,
        result: RunResult,
        output_dir: Path | str,
    ) -> dict[str, Path]:
        """Persiste un ``RunResult`` en 3 fichiers dans ``output_dir``.

        Returns
        -------
        dict[str, Path]
            Map ``{kind: path}`` des fichiers écrits.  Kinds :
            ``"manifest"``, ``"pipeline_results"``, ``"view_results"``.

        Notes
        -----
        Le format JSONL pour les results permet à un consommateur
        (rapport HTML S22) de streamer la lecture sans charger tout
        le run en RAM.  Pour de gros corpus (1000+ docs × N pipelines
        × M vues), c'est précieux.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = out_dir / "run_manifest.json"
        manifest_path.write_text(
            result.manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

        pipeline_path = out_dir / "pipeline_results.jsonl"
        with pipeline_path.open("w", encoding="utf-8") as f:
            for doc_result in result.document_results:
                for pr in doc_result.pipeline_results:
                    payload = {
                        "document_id": doc_result.document_id,
                        **pr.model_dump(mode="json"),
                    }
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        view_path = out_dir / "view_results.jsonl"
        with view_path.open("w", encoding="utf-8") as f:
            for doc_result in result.document_results:
                for vr in doc_result.view_results:
                    payload = {
                        "document_id": doc_result.document_id,
                        **vr.model_dump(mode="json"),
                    }
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        return {
            "manifest": manifest_path,
            "pipeline_results": pipeline_path,
            "view_results": view_path,
        }

    # ──────────────────────────────────────────────────────────────────
    # Helpers internes
    # ──────────────────────────────────────────────────────────────────

    def _evaluate_document_in_views(
        self,
        *,
        document: DocumentRef,
        pipeline_results: list[PipelineResult],
        views: list[EvaluationView],
        ground_truth_factory: GroundTruthFactory,
    ) -> list[ViewResult]:
        """Pour un document, applique chaque vue à chaque artefact
        éligible (pattern d'omission explicite)."""
        out: list[ViewResult] = []
        for view in views:
            for pr in pipeline_results:
                # Trouve les artefacts du pipeline_result éligibles à
                # cette vue.  Pattern d'omission : si aucun artefact
                # éligible, le pipeline n'est PAS dans le ViewResult
                # de cette vue.
                eligible = [
                    a for a in pr.artifacts
                    if view.accepts(a.type)
                ]
                if not eligible:
                    continue
                # Pour chaque artefact éligible, on cherche la GT du
                # type adapté.  Un projecteur dans la vue peut
                # transformer le type ; la GT doit correspondre au
                # type cible APRÈS projection.
                for cand in eligible:
                    gt = ground_truth_factory(
                        document, _gt_type_for_candidate(view, cand.type),
                    )
                    if gt is None:
                        # Pas de GT disponible → omis silencieusement
                        # (le caller verra l'absence dans view_results).
                        continue
                    try:
                        vr = self._view_executor.evaluate(view, cand, gt)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[benchmark_service] evaluate %s/%s/%s a "
                            "levé : %s",
                            view.name, document.id, cand.id, exc,
                        )
                        continue
                    out.append(vr)
        return out


def _gt_type_for_candidate(
    view: EvaluationView,
    candidate_type: ArtifactType,
) -> ArtifactType:
    """Détermine le type de GT à charger pour évaluer un candidat
    dans une vue donnée.

    Si la vue projette le candidat avant comparaison, la GT doit
    être au type **cible** de la projection.  Sinon, elle est au
    type du candidat.
    """
    projection = view.projection_for(candidate_type)
    if projection is not None and not projection.is_identity:
        return projection.target_type
    return candidate_type


def _default_run_id(corpus_name: str, started_at) -> str:
    """Construit un run_id par défaut filesystem-safe."""
    ts = started_at.strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in corpus_name)
    return f"{safe_name}_{ts}"


__all__ = [
    "BenchmarkService",
    "GroundTruthFactory",
    "PipelineInputsFactory",
    "ContextFactory",
]
