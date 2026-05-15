"""``RunOrchestrator`` — exécute un benchmark complet depuis un ``RunSpec``.

Service applicatif qui assemble :

- ``CorpusService`` (import du corpus depuis ZIP ou dir extrait),
- ``RegistryService`` (bootstrap des registres),
- ``BenchmarkService`` (orchestration runner + vues + persistance).

Le rendu de rapport (HTML, JSON, CSV) est **injecté par le caller**
via le paramètre ``report_renderer`` — le service ``app/`` ne peut
pas importer ``reports/`` car cette couche est plus externe
(``domain → … → app → reports → interfaces``).  Cette inversion
de dépendance garantit que :

- L'orchestrateur n'est pas couplé à un format de sortie spécifique.
- Une nouvelle couche de rapport (CSV, JSON) s'ajoute sans modifier
  l'orchestrateur.
- L'ordre des couches reste inviolable (test d'architecture).

Anti-bricolage
--------------
Pas de fonction-helper privée éparpillée dans la CLI.  L'interface
``picarones-rewrite run`` est désormais un thin wrapper Click qui
appelle ``RunOrchestrator.execute(spec, report_renderer=…)`` et
formate la sortie.

Anti-sur-ingénierie
-------------------
- Pas de hooks d'extension (avant/après chaque étape) — quand un
  caller en aura besoin, on ajoutera des callbacks explicites.
- Pas de logique de retry / cache / batching.  Le runner sous-jacent
  les gère déjà s'ils sont configurés.
- Le ``RunOrchestrator`` est sans état entre deux ``execute()`` —
  on peut en créer un par invocation, c'est fait pour.
"""

from __future__ import annotations

import io
import logging
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

from picarones.app.results import ReportRenderer, RunResult
from picarones.app.schemas import RunSpec, resolve_adapter_class
from picarones.app.services.benchmark_service import BenchmarkService
from picarones.app.services.dependencies import (
    capture_dependencies_lock,
    capture_system_binaries_lock,
)
from picarones.app.services.corpus_service import (
    CorpusImportError,
    CorpusService,
)
from picarones.app.services.path_security import WorkspaceManager
from picarones.app.services.registry_service import RegistryService
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.corpus import CorpusSpec
from picarones.domain.documents import DocumentRef
from picarones.evaluation.views import (
    DefaultEvaluationViewExecutor,
    build_alto_view,
    build_search_view,
    build_text_view,
)
from picarones.formats.alto.parser import parse_alto
from picarones.pipeline import (
    CorpusRunner,
    PipelineExecutor,
    PipelineSpec,
    PipelineStep,
    RunContext,
)


# ──────────────────────────────────────────────────────────────────────
# Résultat structuré d'un run orchestré
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OrchestrationResult:
    """Tout ce qu'un caller (CLI, HTTP, script) doit savoir d'un run.

    Attributs
    ---------
    run_result:
        Le ``RunResult`` agrégé produit par le ``BenchmarkService``.
    extracted_corpus_dir:
        Chemin du dossier où le corpus a été extrait (sous le
        workspace).
    persisted_files:
        Map ``{kind: path}`` des 3 fichiers persistés
        (``run_manifest.json``, ``pipeline_results.jsonl``,
        ``view_results.jsonl``).
    report_path:
        Chemin du rapport effectivement écrit par le
        ``report_renderer`` injecté, ou ``None`` si aucun renderer
        n'a été fourni ou si ``spec.report_html`` est vide.
    """

    run_result: RunResult
    extracted_corpus_dir: Path
    persisted_files: dict[str, Path] = field(default_factory=dict)
    report_path: Path | None = None


# ──────────────────────────────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────────────────────────────


class RunOrchestrator:
    """Service applicatif qui exécute un benchmark complet depuis un
    ``RunSpec``.

    Un orchestrateur est lié à un ``output_dir`` (où il créera le
    workspace, le dossier d'extraction et les fichiers de résultats).
    Il ne crée rien tant qu'on n'appelle pas :meth:`execute`.

    Parameters
    ----------
    output_dir:
        Répertoire racine de sortie.  Créé s'il n'existe pas.
    """

    def __init__(self, output_dir: Path | str) -> None:
        self._output_dir = Path(output_dir)

    # ──────────────────────────────────────────────────────────────────
    # API publique
    # ──────────────────────────────────────────────────────────────────

    def execute(
        self,
        spec: RunSpec,
        *,
        report_renderer: ReportRenderer | None = None,
        progress_callback: Callable[[str, int, str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> OrchestrationResult:
        """Exécute le run complet et retourne tout ce qu'on en sait.

        Parameters
        ----------
        spec:
            ``RunSpec`` validée (pydantic).
        report_renderer:
            Callable optionnel ``(run_result, output_path, lang) →
            written_path`` qui rend le rapport.  Si ``None`` (défaut)
            OU si ``spec.report_html`` est vide, aucun rapport n'est
            émis.  L'inversion de dépendance évite à
            ``app/services/`` d'importer ``reports/`` (couche plus
            externe — interdit par l'architecture).
        progress_callback:
            Phase B1.2 — kwarg d'exécution non-sérialisable. Callable
            invoqué ``(engine_name, doc_idx, doc_id)`` à chaque
            document traité.  Le branchement concret au runner est
            fait en Phase B2.1.  Pour l'instant, le kwarg est accepté
            et stocké sur l'instance mais ignoré au runtime — il sera
            consommé quand B2.1 portera le pattern verrou+compteur
            depuis ``_benchmark_execution.py:109-139``.
        cancel_event:
            Phase B1.2 — kwarg d'exécution non-sérialisable.
            ``threading.Event`` qui, quand ``set()``, demande l'arrêt
            propre du run en cours.  Phase B2.2 le branchera au
            ``CorpusRunner`` (pattern existant dans
            ``_benchmark_execution.py:142-149``).

        Raises
        ------
        CorpusImportError
            Si le corpus ne peut pas être chargé.
        RunSpecLoadError
            Si la résolution dotted-path d'un ``adapter_class``
            échoue.
        """
        # Phase B1.2 — kwargs d'exécution stockés temporairement sur
        # l'instance.  Phase B2.1/B2.2 les consommera depuis ici.
        # Volontairement public-protected (un underscore) : ce sont
        # des paramètres d'exécution, pas une configuration durable.
        self._progress_callback = progress_callback
        self._cancel_event = cancel_event

        self._output_dir.mkdir(parents=True, exist_ok=True)
        workspace = WorkspaceManager(self._output_dir)

        # 1. Corpus.
        corpus_spec, extracted_dir = self._load_corpus(spec, workspace)

        # 2. Registres.
        registries = RegistryService.bootstrap_defaults()

        # 3. Pipelines + resolver d'adapters + dump des kwargs pour le manifest.
        pipeline_specs, adapter_resolver, adapter_kwargs = (
            self._build_pipelines(spec)
        )

        # 4. Vues canoniques.  Phase B2.5 — propage normalization +
        # char_exclude aux vues text_final/searchability.
        views = self._build_views(
            spec.views,
            normalization_profile=spec.normalization_profile,
            char_exclude=spec.char_exclude,
        )

        # 5. BenchmarkService.
        bench = self._build_benchmark_service(
            registries=registries,
            adapter_resolver=adapter_resolver,
            code_version=spec.code_version,
            cancel_event=self._cancel_event,
            timeout_seconds_per_doc=spec.timeout_seconds_per_doc,
        )

        # 6. Capture du verrou de dépendances pour la reproductibilité.
        # Sprint S8.5 — capture aussi les binaires système (Tesseract,
        # etc.) qui ne sont pas couverts par le wheel ``pytesseract``.
        deps_lock = capture_dependencies_lock()
        bin_lock = capture_system_binaries_lock()

        # Phase B2.3 — si ``spec.partial_dir`` est fourni, on pivote
        # par pipeline avec reprise sur interruption.  Sinon, chemin
        # rapide en un seul ``bench.run`` multi-pipeline.
        # Phase B4 — workspace_uri dédié au runtime des adapters
        # (artefacts intermédiaires).  Distinct du extracted_dir
        # qui porte les images source du corpus.
        runtime_dir = self._output_dir / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)

        if spec.partial_dir:
            result = self._execute_with_partial(
                spec=spec,
                bench=bench,
                corpus_spec=corpus_spec,
                pipeline_specs=pipeline_specs,
                views=views,
                adapter_kwargs=adapter_kwargs,
                deps_lock=deps_lock,
                bin_lock=bin_lock,
                runtime_dir=runtime_dir,
            )
        else:
            result = bench.run(
                corpus=corpus_spec,
                pipelines=pipeline_specs,
                views=views,
                ground_truth_factory=_default_gt_factory,
                pipeline_inputs_factory=_default_inputs_factory,
                context_factory=_make_context_factory(
                    spec.code_version,
                    progress_callback=self._progress_callback,
                    workspace_uri=str(runtime_dir),
                ),
                adapter_kwargs=adapter_kwargs,
                dependencies_lock=deps_lock,
                system_binaries_lock=bin_lock,
                metadata={
                    "orchestrator":
                    "picarones.app.services.run_orchestrator",
                },
            )

        # 6. Persistance JSONL.
        persist_dir = self._output_dir / "results"
        persisted = bench.persist(result, persist_dir)

        # 6.bis. Phase B2.7 — persistance optionnelle du
        # ``BenchmarkResult`` legacy (format ``run_benchmark_via_service``).
        # Cohabite avec les 4 fichiers JSONL natifs (run_manifest,
        # pipeline_results, artifacts_index, view_results) — utile pour
        # les consommateurs historiques (rapport HTML legacy, narrative
        # engine) tant que la migration n'est pas terminée.
        if spec.output_json:
            self._persist_legacy_benchmark_json(
                run_result=result,
                extracted_dir=extracted_dir,
                pipeline_specs=pipeline_specs,
                corpus_name=corpus_spec.name,
                output_json=Path(spec.output_json),
                char_exclude=spec.char_exclude,
                normalization_profile=spec.normalization_profile,
                profile=spec.profile,
                entity_extractor=spec.entity_extractor,
            )

        # 7. Rapport optionnel — délégué au renderer injecté.
        # Inversion de dépendance : ``app/`` ne peut pas importer
        # ``reports/`` (plus externe).  Le caller fournit un
        # callable.
        report_path: Path | None = None
        if report_renderer is not None and spec.report_html:
            target = Path(spec.report_html)
            target.parent.mkdir(parents=True, exist_ok=True)
            report_path = report_renderer(result, target, spec.report_lang)

        return OrchestrationResult(
            run_result=result,
            extracted_corpus_dir=extracted_dir,
            persisted_files=persisted,
            report_path=report_path,
        )

    def execute_preset(
        self,
        spec: RunSpec,
        *,
        corpus_spec: Any,
        extracted_dir: Path,
        pipeline_specs: list[Any],
        adapter_resolver: Callable[[str], Any],
        adapter_kwargs: dict[str, Any] | None = None,
        report_renderer: ReportRenderer | None = None,
        progress_callback: Callable[[str, int, str], None] | None = None,
        cancel_event: threading.Event | None = None,
        corpus_legacy: Any | None = None,
    ) -> OrchestrationResult:
        """Phase B4 — variante d'``execute()`` pour objets domain pré-construits.

        Utilisée par les call sites (tests, CLI/Web post-B3) qui ont
        déjà instancié leurs engines et chargé leur corpus avant de
        connaître le ``RunOrchestrator``.  Plus simple à câbler qu'un
        ``RunSpec`` YAML complet car on n'a pas à reconstruire les
        dotted paths d'``adapter_class`` ni à re-extraire un
        ``corpus_zip``.

        Le ``spec`` fourni est utilisé pour les **paramètres**
        (``views``, ``output_json``, ``partial_dir``, ``char_exclude``,
        ``normalization_profile``, ``profile``, ``entity_extractor``,
        ``code_version``, etc.) mais ``spec.corpus_dir`` /
        ``spec.corpus_zip`` et ``spec.pipelines`` sont **ignorés** au
        profit des objets fournis en kwargs.

        Parameters
        ----------
        spec:
            ``RunSpec`` qui porte les paramètres d'exécution.  Sa partie
            corpus/pipelines est ignorée — le caller a déjà résolu ces
            objets.
        corpus_spec:
            ``CorpusSpec`` (couche 1) déjà construit.  Typiquement obtenu
            via ``corpus_to_corpus_spec(corpus_legacy, workspace_dir=...)``.
        extracted_dir:
            Dossier où le corpus est physiquement disponible (pour le
            ``_persist_legacy_benchmark_json`` Phase B2.7 et pour le
            ``corpus_name`` propagé au converter).
        pipeline_specs:
            Liste de ``PipelineSpec`` (couche 1) déjà construits depuis
            les engines en mémoire (typiquement via
            ``engine_to_pipeline_spec``).
        adapter_resolver:
            Resolver ``name → StepExecutor`` déjà construit (typiquement
            via ``build_adapter_resolver``).
        adapter_kwargs:
            Map ``adapter_name → kwargs dict`` pour le manifest.  Peut
            être ``{}``.
        report_renderer, progress_callback, cancel_event:
            Identiques à ``execute()``.
        """
        self._progress_callback = progress_callback
        self._cancel_event = cancel_event
        self._output_dir.mkdir(parents=True, exist_ok=True)

        registries = RegistryService.bootstrap_defaults()
        views = self._build_views(
            spec.views,
            normalization_profile=spec.normalization_profile,
            char_exclude=spec.char_exclude,
        )
        bench = self._build_benchmark_service(
            registries=registries,
            adapter_resolver=adapter_resolver,
            code_version=spec.code_version,
            cancel_event=self._cancel_event,
            timeout_seconds_per_doc=spec.timeout_seconds_per_doc,
        )
        deps_lock = capture_dependencies_lock()
        bin_lock = capture_system_binaries_lock()
        adapter_kwargs_clean = adapter_kwargs or {}

        # Phase B4 — workspace_uri pour les adapters.
        runtime_dir = self._output_dir / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)

        if spec.partial_dir:
            result = self._execute_with_partial(
                spec=spec,
                bench=bench,
                corpus_spec=corpus_spec,
                pipeline_specs=pipeline_specs,
                views=views,
                adapter_kwargs=adapter_kwargs_clean,
                deps_lock=deps_lock,
                bin_lock=bin_lock,
                runtime_dir=runtime_dir,
            )
        else:
            result = bench.run(
                corpus=corpus_spec,
                pipelines=pipeline_specs,
                views=views,
                ground_truth_factory=_default_gt_factory,
                pipeline_inputs_factory=_default_inputs_factory,
                context_factory=_make_context_factory(
                    spec.code_version,
                    progress_callback=self._progress_callback,
                    workspace_uri=str(runtime_dir),
                ),
                adapter_kwargs=adapter_kwargs_clean,
                dependencies_lock=deps_lock,
                system_binaries_lock=bin_lock,
                metadata={
                    "orchestrator":
                    "picarones.app.services.run_orchestrator",
                    "mode": "preset",
                },
            )

        persist_dir = self._output_dir / "results"
        persisted = bench.persist(result, persist_dir)

        if spec.output_json:
            self._persist_legacy_benchmark_json(
                run_result=result,
                extracted_dir=extracted_dir,
                pipeline_specs=pipeline_specs,
                corpus_name=corpus_spec.name,
                output_json=Path(spec.output_json),
                char_exclude=spec.char_exclude,
                normalization_profile=spec.normalization_profile,
                profile=spec.profile,
                entity_extractor=spec.entity_extractor,
                corpus_legacy=corpus_legacy,
            )

        report_path: Path | None = None
        if report_renderer is not None and spec.report_html:
            target = Path(spec.report_html)
            target.parent.mkdir(parents=True, exist_ok=True)
            report_path = report_renderer(result, target, spec.report_lang)

        return OrchestrationResult(
            run_result=result,
            extracted_corpus_dir=extracted_dir,
            persisted_files=persisted,
            report_path=report_path,
        )

    # ──────────────────────────────────────────────────────────────────
    # Étapes individuelles (publiques pour permettre la composition
    # depuis un caller qui veut surcharger une étape).
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_corpus(
        spec: RunSpec, workspace: WorkspaceManager,
    ) -> tuple[CorpusSpec, Path]:
        """Charge le corpus selon ``corpus_zip`` ou ``corpus_dir``."""
        corpus_service = CorpusService(workspace)
        if spec.corpus_zip is not None:
            zip_path = Path(spec.corpus_zip)
            zip_bytes = zip_path.read_bytes()
            report = corpus_service.import_zip(
                zip_bytes,
                corpus_name=spec.corpus_name or zip_path.stem,
                metadata=spec.corpus_metadata,
            )
            return report.spec, report.extracted_dir

        # corpus_dir : on zippe à la volée le contenu du dir et on
        # délègue à ``CorpusService`` — réutilise toute la détection
        # sans dupliquer la logique de classification image / GT.
        assert spec.corpus_dir is not None  # garanti par RunSpec validator
        src_dir = Path(spec.corpus_dir)
        if not src_dir.is_dir():
            raise CorpusImportError(
                f"corpus_dir n'est pas un répertoire : {src_dir!r}.",
            )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            for file_path in src_dir.rglob("*"):
                if file_path.is_file():
                    arc = file_path.relative_to(src_dir).as_posix()
                    zf.write(file_path, arcname=arc)
        report = corpus_service.import_zip(
            buf.getvalue(),
            corpus_name=spec.corpus_name or src_dir.name,
            metadata=spec.corpus_metadata,
        )
        return report.spec, report.extracted_dir

    @staticmethod
    def _build_pipelines(
        spec: RunSpec,
    ) -> tuple[
        list[PipelineSpec],
        Callable[[str], Any],
        dict[str, dict[str, Any]],
    ]:
        """Construit les ``PipelineSpec`` + un resolver d'adapters.

        Disambiguation des steps :

        - Deux steps avec la même ``(class, kwargs)`` partagent la
          même instance d'adapter (cache).
        - Deux steps avec la même ``id`` mais une ``class`` ou des
          ``kwargs`` différents reçoivent des ``adapter_name``
          distincts (préfixés par le nom de pipeline).

        C'est essentiel pour le cas où plusieurs pipelines utilisent
        la **même classe** avec des **kwargs différents** (ex :
        ``PrecomputedTextAdapter`` instancié N fois avec
        ``source_label`` distincts).
        """
        instance_cache: dict[str, Any] = {}
        registered: dict[str, tuple[type, str]] = {}
        name_to_class: dict[str, type] = {}
        name_to_kwargs: dict[str, dict[str, Any]] = {}

        pipeline_specs: list[PipelineSpec] = []
        for p in spec.pipelines:
            steps: list[PipelineStep] = []
            for s in p.steps:
                cls = resolve_adapter_class(s.adapter_class)
                kwargs_sig = _kwargs_signature(s.adapter_kwargs)
                adapter_name = s.id
                existing = registered.get(adapter_name)
                if existing is not None and existing != (cls, kwargs_sig):
                    adapter_name = f"{p.name}__{s.id}"
                registered[adapter_name] = (cls, kwargs_sig)
                name_to_class[adapter_name] = cls
                name_to_kwargs[adapter_name] = s.adapter_kwargs
                # ``inputs_from`` du StepSpec YAML doit être propagé au
                # ``domain.PipelineSpec`` pour que le DAG branchant soit
                # honoré ; sans ce passage, un DAG branchant déclaré dans
                # le YAML serait silencieusement exécuté en linéaire.
                steps.append(PipelineStep(
                    id=s.id,
                    kind="step",
                    adapter_name=adapter_name,
                    input_types=s.input_types,
                    output_types=s.output_types,
                    inputs_from=dict(s.inputs_from),
                ))
            pipeline_specs.append(PipelineSpec(
                name=p.name,
                initial_inputs=p.initial_inputs,
                steps=tuple(steps),
            ))

        def resolver(name: str) -> Any:
            if name not in instance_cache:
                cls = name_to_class[name]
                kwargs = name_to_kwargs[name]
                instance_cache[name] = cls(**kwargs)
            return instance_cache[name]

        # Copie défensive — le manifest doit recevoir un snapshot
        # immuable, pas la map vivante du resolver.
        adapter_kwargs_dump = {
            name: dict(kwargs) for name, kwargs in name_to_kwargs.items()
        }
        return pipeline_specs, resolver, adapter_kwargs_dump

    def _execute_with_partial(
        self,
        *,
        spec: Any,
        bench: Any,
        corpus_spec: Any,
        pipeline_specs: list[Any],
        views: list[Any],
        adapter_kwargs: dict[str, Any],
        deps_lock: dict[str, Any],
        bin_lock: dict[str, Any],
        runtime_dir: Path | None = None,
    ) -> Any:
        """Phase B2.3 — exécution pivotée par pipeline avec reprise.

        Pour chaque ``pipeline_spec`` :

        1. Calcule un fingerprint SHA-256 du run (pipeline structure +
           normalization + char_exclude + profile + corpus
           mtime/size + code_version).
        2. Cherche un fichier partial existant matchant ce fingerprint.
        3. Charge les ``PipelineResult`` déjà calculés.
        4. Filtre le corpus pour ne soumettre au ``BenchmarkService``
           que les documents manquants.
        5. Append chaque nouveau ``PipelineResult`` au fichier partial
           au fil de l'eau (un crash mid-run préserve ce qui a été
           calculé).
        6. À la fin d'une pipeline traitée intégralement, supprime
           le partial (cleanup).

        Le résultat final est un ``RunResult`` reconstruit à partir de
        tous les ``PipelineResult`` (chargés + nouveaux), réorganisés
        par document selon l'ordre du corpus original.

        Limitations volontaires (scope B2.3) : les ``ViewResult`` ne
        sont conservés que pour les ``PipelineResult`` calculés dans
        le run courant (pas pour ceux rechargés depuis partial).
        Pour relancer les vues sur l'ensemble, le caller doit relancer
        sans ``partial_dir`` ou pré-supprimer les partials.
        """
        from picarones.app.results import RunResult
        from picarones.app.services._orchestrator_partial import (
            append_pipeline_result,
            compute_pipeline_fingerprint,
            delete_partial,
            filter_remaining_documents,
            load_partial_pipeline_results,
            partial_path_for_pipeline,
        )
        from picarones.domain.corpus import CorpusSpec
        from picarones.domain.run_manifest import RunManifest
        from picarones.pipeline.run_result import RunDocumentResult

        partial_dir = Path(spec.partial_dir)
        partial_dir.mkdir(parents=True, exist_ok=True)

        # Map : pipeline_name → (partial_path, list[PipelineResult])
        per_pipeline_state: dict[str, tuple[Path, list[Any]]] = {}
        for pipeline_spec in pipeline_specs:
            fingerprint = compute_pipeline_fingerprint(
                pipeline_spec=pipeline_spec,
                corpus_spec=corpus_spec,
                normalization_profile=spec.normalization_profile,
                char_exclude=spec.char_exclude,
                profile=spec.profile,
                code_version=spec.code_version,
            )
            path = partial_path_for_pipeline(
                partial_dir=partial_dir,
                corpus_name=corpus_spec.name,
                pipeline_name=pipeline_spec.name,
                fingerprint=fingerprint,
            )
            loaded = load_partial_pipeline_results(path)
            if loaded:
                logger.info(
                    "[run_orchestrator] reprise pipeline %r : %d/%d "
                    "documents déjà persistés.",
                    pipeline_spec.name,
                    len(loaded), len(corpus_spec.documents),
                )
            per_pipeline_state[pipeline_spec.name] = (path, loaded)

        # Lance un sub-run par pipeline avec uniquement les docs
        # manquants.  Sub-RunResult séparés ; on agrège ensuite.
        sub_run_results: list[Any] = []
        for pipeline_spec in pipeline_specs:
            partial_path, loaded_results = per_pipeline_state[pipeline_spec.name]

            remaining_docs, deduplicated_loaded = filter_remaining_documents(
                corpus_spec.documents, loaded_results,
            )
            per_pipeline_state[pipeline_spec.name] = (
                partial_path, deduplicated_loaded,
            )

            if not remaining_docs:
                logger.info(
                    "[run_orchestrator] pipeline %r déjà complet — "
                    "skip exécution.", pipeline_spec.name,
                )
                # Cleanup du partial : le pipeline est entièrement
                # rechargé, plus besoin de garder le fichier sur disque.
                delete_partial(partial_path)
                continue

            sub_corpus_spec = CorpusSpec(
                name=corpus_spec.name,
                documents=tuple(remaining_docs),
                metadata=dict(corpus_spec.metadata),
            )

            sub_result = bench.run(
                corpus=sub_corpus_spec,
                pipelines=[pipeline_spec],
                views=views,
                ground_truth_factory=_default_gt_factory,
                pipeline_inputs_factory=_default_inputs_factory,
                context_factory=_make_context_factory(
                    spec.code_version,
                    progress_callback=self._progress_callback,
                    workspace_uri=str(runtime_dir) if runtime_dir else None,
                ),
                adapter_kwargs=adapter_kwargs,
                dependencies_lock=deps_lock,
                system_binaries_lock=bin_lock,
                metadata={
                    "orchestrator":
                    "picarones.app.services.run_orchestrator",
                    "partial_pipeline": pipeline_spec.name,
                },
            )
            sub_run_results.append(sub_result)

            # Persiste chaque nouveau PipelineResult au partial.
            new_count = 0
            for doc_result in sub_result.document_results:
                for pr in doc_result.pipeline_results:
                    if pr.pipeline_name == pipeline_spec.name:
                        append_pipeline_result(partial_path, pr)
                        new_count += 1

            # Si tous les docs du corpus original ont été traités
            # (loaded + new) → cleanup du partial.
            loaded_doc_ids = {pr.document_id for pr in deduplicated_loaded}
            new_doc_ids = {
                pr.document_id
                for doc_result in sub_result.document_results
                for pr in doc_result.pipeline_results
                if pr.pipeline_name == pipeline_spec.name
            }
            all_doc_ids = {d.id for d in corpus_spec.documents}
            if (loaded_doc_ids | new_doc_ids) >= all_doc_ids:
                delete_partial(partial_path)
                logger.info(
                    "[run_orchestrator] pipeline %r complet (%d docs) "
                    "— partial supprimé.",
                    pipeline_spec.name, len(all_doc_ids),
                )

        # Reconstruit le RunResult final : pour chaque doc du corpus
        # original, agrège les PipelineResult de tous les pipelines.
        # Map (doc_id, pipeline_name) → PipelineResult
        pr_index: dict[tuple[str, str], Any] = {}
        # Map (doc_id, pipeline_name) → list[ViewResult]
        vr_index: dict[tuple[str, str], list[Any]] = {}

        # Charge les pipeline_results depuis les partials (rechargés).
        for pipeline_name, (_, loaded_list) in per_pipeline_state.items():
            for pr in loaded_list:
                pr_index[(pr.document_id, pipeline_name)] = pr

        # Charge les pipeline_results et view_results depuis les sub-runs.
        for sub_result in sub_run_results:
            for sub_doc in sub_result.document_results:
                for pr in sub_doc.pipeline_results:
                    pr_index[(sub_doc.document_id, pr.pipeline_name)] = pr
                for vr in sub_doc.view_results:
                    # ``ViewResult.pipeline_name`` n'existe pas ; on
                    # regroupe par doc seulement (pas suffisamment
                    # granulaire mais OK pour la sortie).
                    vr_index.setdefault(
                        (sub_doc.document_id, ""), [],
                    ).append(vr)

        # Construit les RunDocumentResult dans l'ordre du corpus.
        final_doc_results: list[Any] = []
        for doc in corpus_spec.documents:
            doc_pipeline_results = tuple(
                pr_index[(doc.id, ps.name)]
                for ps in pipeline_specs
                if (doc.id, ps.name) in pr_index
            )
            doc_view_results = tuple(vr_index.get((doc.id, ""), []))
            final_doc_results.append(RunDocumentResult(
                document_id=doc.id,
                pipeline_results=doc_pipeline_results,
                view_results=doc_view_results,
            ))

        # Synthétise un RunManifest minimal (on prend celui d'un
        # sub-run s'il y en a eu, sinon on synthétise from scratch).
        if sub_run_results:
            # Fusionne les pipeline_specs de tous les sub-runs.
            base_manifest = sub_run_results[0].manifest
            manifest = RunManifest(
                run_id=base_manifest.run_id,
                corpus_name=corpus_spec.name,
                n_documents=len(corpus_spec.documents),
                pipeline_specs=tuple(pipeline_specs),
                adapter_kwargs=adapter_kwargs,
                view_specs=tuple(views),
                code_version=spec.code_version,
                started_at=base_manifest.started_at,
                completed_at=base_manifest.completed_at,
                dependencies_lock=deps_lock,
                system_binaries_lock=bin_lock,
                metadata={
                    "orchestrator":
                    "picarones.app.services.run_orchestrator",
                    "partial_dir": str(partial_dir),
                },
            )
        else:
            # Tous les pipelines ont été chargés depuis partial — pas
            # de sub-run.  On synthétise un manifest from scratch.
            from picarones.app.services.benchmark_service import (
                _default_run_id,
            )
            from picarones.domain.run_manifest import utcnow
            now = utcnow()
            manifest = RunManifest(
                run_id=_default_run_id(corpus_spec.name, now),
                corpus_name=corpus_spec.name,
                n_documents=len(corpus_spec.documents),
                pipeline_specs=tuple(pipeline_specs),
                adapter_kwargs=adapter_kwargs,
                view_specs=tuple(views),
                code_version=spec.code_version,
                started_at=now,
                completed_at=now,
                dependencies_lock=deps_lock,
                system_binaries_lock=bin_lock,
                metadata={
                    "orchestrator":
                    "picarones.app.services.run_orchestrator",
                    "partial_dir": str(partial_dir),
                    "fully_resumed": "true",
                },
            )

        return RunResult(
            manifest=manifest,
            document_results=tuple(final_doc_results),
        )

    @staticmethod
    def _persist_legacy_benchmark_json(
        *,
        run_result: Any,
        extracted_dir: Path,
        pipeline_specs: list[Any],
        corpus_name: str,
        output_json: Path,
        char_exclude: str | None,
        normalization_profile: str | None,
        profile: str,
        entity_extractor: str | None = None,
        corpus_legacy: Any | None = None,
    ) -> None:
        """Phase B2.7 — converti ``RunResult`` → ``BenchmarkResult`` legacy
        et persiste en JSON.

        Délègue à
        :func:`picarones.app.services._benchmark_converter.run_result_to_benchmark_result`
        (utilisé aussi par ``run_benchmark_via_service``) pour
        garantir l'équivalence numérique du format de sortie.

        Le caller fournit :

        - ``run_result`` : le ``RunResult`` produit par le ``BenchmarkService``.
        - ``extracted_dir`` : où le corpus a été extrait — sert à
          recharger un ``Corpus`` legacy via
          ``load_corpus_from_directory`` **quand** ``corpus_legacy``
          n'est pas fourni (mode ``execute()`` avec extraction réelle
          d'un zip/dir).  Le converter attend des ``Document`` legacy
          avec ``image_path`` et ``ground_truth``.
        - ``pipeline_specs`` : la liste des pipelines exécutées, dans
          l'ordre soumis à ``BenchmarkService.run``.  Chaque spec est
          wrappée en ``_PipelineEngineProxy`` qui expose le contrat
          minimal attendu par le converter (``name``, ``config``).
        - ``output_json`` : chemin de sortie ; les répertoires parents
          sont créés.
        - ``char_exclude``, ``normalization_profile``, ``profile`` :
          paramètres legacy propagés au converter (qui les passe à
          ``compute_metrics`` et aux hooks document-level).
        - ``corpus_legacy`` *(Phase B3-final hotfix mai 2026)* : Corpus
          legacy déjà en mémoire.  Quand fourni, court-circuite le
          ``load_corpus_from_directory(extracted_dir)`` qui échoue dans
          le path ``execute_preset`` : en mode preset, ``extracted_dir``
          pointe vers le ``workspace_dir`` qui ne contient que les
          ``.gt.txt`` synthétisés par ``document_to_document_ref``, pas
          les images sources — ``load_corpus_from_directory`` itère
          alors sur zéro image et lève ``ValueError: Aucun document
          valide trouvé``.  Symptôme observé en prod : le benchmark
          web/CLI échouait silencieusement après la 1re exécution OCR
          avec ce message trompeur.

        Notes
        -----
        Le format produit est strictement identique à celui de
        ``run_benchmark_via_service(output_json=...)`` (testé via le
        snapshot d'invariance ``test_migration_invariance.py``).
        """
        from picarones.app.services._benchmark_converter import (
            run_result_to_benchmark_result,
        )
        from picarones.app.services._benchmark_persistence import (
            persist_benchmark_result_json,
        )

        if corpus_legacy is not None:
            # Mode preset : le caller a déjà le ``Corpus`` en mémoire
            # (typiquement chargé depuis ``uploads/`` côté web ou via
            # ``load_corpus_from_directory(corpus_arg)`` côté CLI).
            # Pas de reload — évite la divergence ``extracted_dir`` ≠
            # vrai source dir documentée plus haut.
            corpus = corpus_legacy
        else:
            from picarones.evaluation.corpus import load_corpus_from_directory

            # Mode ``execute()`` classique : le corpus est physiquement
            # disponible dans ``extracted_dir`` (zip extrait ou dossier
            # source).  ``name`` passé explicitement pour matcher
            # ``corpus_spec.name`` (sinon le loader retourne
            # ``"Corpus"`` par défaut, ce qui casserait le snapshot
            # d'invariance).
            try:
                corpus = load_corpus_from_directory(
                    extracted_dir, name=corpus_name,
                )
            except (ValueError, FileNotFoundError) as exc:
                # Audit B3-final mai 2026, trou #9 : si ``extracted_dir``
                # est en fait un ``workspace_dir`` synthétisé par
                # ``prepare_preset_args`` (= gt-only, pas d'images), le
                # reload lève ``ValueError: Aucun document valide
                # trouvé`` — message cryptique qui masque le vrai
                # problème (caller direct à ``execute_preset(...,
                # output_json=set)`` sans passer ``corpus_legacy``).
                # On enrichit le message pour pointer le caller.
                raise ValueError(
                    "_persist_legacy_benchmark_json : impossible de "
                    f"reloader le corpus depuis {extracted_dir!r}.\n"
                    "Si vous êtes en mode preset (corpus chargé en "
                    "mémoire avant ``execute_preset()``), passer "
                    "``corpus_legacy=corpus`` à ``execute_preset()`` "
                    "pour éviter ce reload — le ``workspace_dir`` "
                    "synthétisé par ``prepare_preset_args`` ne "
                    "contient que les .gt.txt, pas les images "
                    f"sources.\nErreur originale : {exc}",
                ) from exc

        # Wrappe chaque PipelineSpec en proxy minimal pour le converter.
        # Le converter ne consomme que ``.name``, ``.config`` et tolère
        # l'absence de ``.version`` (cf. ``_safe_engine_version``).
        engines = [_PipelineEngineProxy(spec) for spec in pipeline_specs]

        # Phase B2.5 — le converter legacy passe ``normalization_profile``
        # à ``compute_metrics`` qui attend un objet ``NormalizationProfile``,
        # pas une string.  Résolution explicite ici pour aligner avec ce que
        # font les call sites legacy (CLI ``_workflows.py`` via
        # ``resolve_normalization_profile``).  ``char_exclude`` reste string —
        # ``compute_metrics`` le traite comme un set/frozenset implicite.
        resolved_profile: Any = None
        if normalization_profile:
            from picarones.formats.text.normalization import get_builtin_profile
            try:
                resolved_profile = get_builtin_profile(normalization_profile)
            except KeyError:
                # Profil inconnu — on laisse ``None`` (le converter
                # tombera dans son default ``DEFAULT_DIPLOMATIC_PROFILE``).
                # Cohérent avec le legacy qui logge un warning sans
                # casser le run.
                logger.warning(
                    "[run_orchestrator] profil normalisation %r inconnu "
                    "pour output_json — fallback default diplomatique.",
                    normalization_profile,
                )

        benchmark_result = run_result_to_benchmark_result(
            run_result,
            corpus=corpus,
            engines=engines,
            char_exclude=frozenset(char_exclude) if char_exclude else None,
            normalization_profile=resolved_profile,
            profile=profile,
        )

        # Phase B2.4 — NER attach post-process si un entity_extractor
        # est fourni.  Pattern identique à
        # ``run_benchmark_via_service:261-264`` :  on résout le dotted
        # path, on instancie la factory, on attache au BenchmarkResult.
        if entity_extractor:
            extractor_callable = _resolve_entity_extractor(entity_extractor)
            if extractor_callable is not None:
                from picarones.app.services._benchmark_ner import (
                    attach_ner_metrics_to_benchmark,
                )
                attach_ner_metrics_to_benchmark(
                    benchmark_result, corpus, extractor_callable,
                )

        persist_benchmark_result_json(benchmark_result, output_json)

    @staticmethod
    def _build_views(
        view_names: tuple[str, ...],
        *,
        normalization_profile: str | None = None,
        char_exclude: str | None = None,
    ) -> list[Any]:
        """Map noms canoniques → vues construites.

        Phase B2.5 — ``normalization_profile`` et ``char_exclude``
        sont propagés aux vues qui les supportent (``text_final`` et
        ``searchability``).  ``alto_documentary`` les ignore : ses
        métriques structurelles n'opèrent pas sur du texte.
        """
        text_view_kwargs = {
            "normalization_profile": normalization_profile,
            "char_exclude": char_exclude,
        }
        builders: dict[str, Callable[[], Any]] = {
            "text_final": lambda: build_text_view(**text_view_kwargs),
            "alto_documentary": build_alto_view,
            "searchability": lambda: build_search_view(**text_view_kwargs),
        }
        return [builders[name]() for name in view_names]

    @staticmethod
    def _build_benchmark_service(
        *,
        registries: RegistryService,
        adapter_resolver: Callable[[str], Any],
        code_version: str,
        cancel_event: threading.Event | None = None,
        timeout_seconds_per_doc: float = 300.0,
    ) -> BenchmarkService:
        """Assemble ``BenchmarkService`` avec un loader filesystem.

        Phase B2.2 — quand ``cancel_event`` est fourni, le
        ``CorpusRunner.run`` est wrappé pour injecter l'event dans
        chaque appel.  Pattern strictement copié de
        ``_benchmark_execution.py:142-149`` (legacy).
        """
        pipeline_executor = PipelineExecutor(
            adapter_resolver=adapter_resolver,
        )
        corpus_runner = CorpusRunner(
            pipeline_executor,
            max_in_flight=2,
            timeout_seconds_per_doc=timeout_seconds_per_doc,
            poll_interval_seconds=0.05,
        )

        if cancel_event is not None:
            original_run = corpus_runner.run

            def _runner_run_with_cancel(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("cancel_event", cancel_event)
                return original_run(*args, **kwargs)

            corpus_runner.run = _runner_run_with_cancel  # type: ignore[method-assign]

        view_executor = DefaultEvaluationViewExecutor.from_registries(
            registries.metrics,
            registries.projectors,
            _filesystem_payload_loader,
        )
        return BenchmarkService(
            corpus_runner=corpus_runner,
            view_executor=view_executor,
            code_version=code_version,
        )


# ──────────────────────────────────────────────────────────────────────
# Helpers privés (factories canoniques)
# ──────────────────────────────────────────────────────────────────────


class _PipelineEngineProxy:
    """Proxy léger ``PipelineSpec → engine`` pour le converter legacy.

    Phase B2.7 — quand on persiste un ``BenchmarkResult`` legacy via
    le converter ``run_result_to_benchmark_result``, ce dernier attend
    une liste d'``engines`` qui exposent ``.name`` et ``.config`` (le
    modèle mental legacy est OCR adapter / OCRLLMPipeline).

    Le ``RunOrchestrator`` raisonne en termes de ``PipelineSpec``
    (déclaration YAML), pas d'instances d'engines.  Ce proxy fournit
    juste le contrat minimal pour que le converter produise un
    ``EngineReport`` cohérent — sans introduire de couplage entre
    ``run_orchestrator`` et le code legacy.

    Le ``.name`` exposé est celui de la pipeline (``"ocr_then_correct"``)
    et non du premier step (``"tesseract"``) — pour que l'``EngineReport``
    porte le nom canonique du pipeline.
    """

    __slots__ = ("_spec",)

    def __init__(self, pipeline_spec: Any) -> None:
        self._spec = pipeline_spec

    @property
    def name(self) -> str:
        return str(self._spec.name)

    @property
    def config(self) -> dict[str, Any]:
        """Config sérialisable : noms des steps + leurs types I/O.

        Permet aux consommateurs du ``BenchmarkResult`` legacy de
        savoir quel pipeline a produit chaque ``EngineReport`` sans
        connaître les détails d'implémentation des adapters.
        """
        return {
            "pipeline_name": self._spec.name,
            "steps": [
                {
                    "id": step.id,
                    "input_types": sorted(t.value for t in step.input_types),
                    "output_types": sorted(t.value for t in step.output_types),
                }
                for step in self._spec.steps
            ],
        }


def _resolve_entity_extractor(
    dotted_path: str,
) -> Callable[[str], list[dict]] | None:
    """Phase B2.4 — résout un dotted path vers un extracteur d'entités.

    Format attendu (validé en B1.1 via ``_DOTTED_PATH_RE`` du
    ``RunSpec``) :

    - ``module.submodule:Symbol`` (PEP 621 entry points / setuptools)
    - ``module.submodule.Symbol`` (import classique)

    Le symbole résolu doit être soit :

    - une **factory zéro-arg** qui retourne un callable ``(text: str)
      -> list[dict]`` (pattern legacy CLI : ``SpacyEntityExtractor``
      avec config par défaut),
    - soit directement un callable ``(text: str) -> list[dict]``
      (pattern test : fonction mock).

    On essaie d'abord d'appeler le symbole sans argument ; si ça
    renvoie un callable, on l'utilise.  Sinon, on suppose que le
    symbole est déjà un callable.

    Returns
    -------
    Callable ou ``None`` si la résolution échoue.  Un échec ne
    casse pas le bench (warning loggé, NER skippé) — cohérent avec
    le legacy ``_attach_ner_metrics_to_benchmark`` qui dégrade
    proprement.
    """
    import importlib

    # Normalise le séparateur final : ``:`` ou ``.`` indifféremment.
    if ":" in dotted_path:
        module_path, _, symbol_name = dotted_path.rpartition(":")
    else:
        module_path, _, symbol_name = dotted_path.rpartition(".")

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        logger.warning(
            "[run_orchestrator] entity_extractor : module %r introuvable "
            "(%s) — NER sauté pour ce run.",
            module_path, exc,
        )
        return None

    symbol = getattr(module, symbol_name, None)
    if symbol is None:
        logger.warning(
            "[run_orchestrator] entity_extractor : symbole %r absent de %r "
            "— NER sauté pour ce run.",
            symbol_name, module_path,
        )
        return None

    # Pattern legacy : si ``symbol`` est une factory (classe ou
    # fonction zéro-arg), l'instancier.  Sinon, l'utiliser tel quel.
    if callable(symbol):
        try:
            candidate = symbol()
            if callable(candidate):
                return candidate
            # ``symbol()`` retourne autre chose qu'un callable —
            # ``symbol`` est probablement déjà la fonction d'extraction.
            return symbol
        except TypeError:
            # ``symbol`` n'accepte pas zéro-arg : c'est probablement
            # la fonction d'extraction directe.
            return symbol

    logger.warning(
        "[run_orchestrator] entity_extractor : %r n'est pas callable.",
        dotted_path,
    )
    return None


def _kwargs_signature(kwargs: dict[str, Any]) -> str:
    """Signature stable d'un dict de kwargs (ordre tri-stable)."""
    return "|".join(f"{k}={kwargs[k]!r}" for k in sorted(kwargs))


def _default_gt_factory(
    doc: DocumentRef, art_type: ArtifactType,
) -> Artifact | None:
    """Factory GT par défaut.

    Convention : un candidat ``CORRECTED_TEXT`` est comparé contre
    la GT ``RAW_TEXT`` (les deux sont du texte plat — la distinction
    de type ne porte que sur le côté candidat).  Cas typique : un
    pipeline OCR + post-correction LLM produit un ``CORRECTED_TEXT``
    qu'on compare au ``.gt.txt`` original.
    """
    effective_type = (
        ArtifactType.RAW_TEXT
        if art_type == ArtifactType.CORRECTED_TEXT
        else art_type
    )
    gt_ref = doc.gt_for(effective_type)
    if gt_ref is None:
        return None
    return Artifact(
        id=f"{doc.id}:gt:{effective_type.value}",
        document_id=doc.id,
        type=effective_type,
        uri=gt_ref.uri,
    )


def _default_inputs_factory(doc: DocumentRef) -> dict[ArtifactType, Artifact]:
    """``{IMAGE: artifact_image}``.  Lève si ``doc.image_uri`` absent."""
    if doc.image_uri is None:
        raise CorpusImportError(
            f"Document {doc.id!r} sans ``image_uri`` — la pipeline "
            "par défaut consomme une IMAGE en entrée.",
        )
    return {ArtifactType.IMAGE: Artifact(
        id=f"{doc.id}:image",
        document_id=doc.id,
        type=ArtifactType.IMAGE,
        uri=doc.image_uri,
    )}


def _make_context_factory(
    code_version: str,
    *,
    progress_callback: Callable[[str, int, str], None] | None = None,
    workspace_uri: str | None = None,
) -> Callable[[DocumentRef, str], RunContext]:
    """Phase B2.1 — factory de ``RunContext`` avec callback de progression.

    Pattern strictement copié de
    ``_benchmark_execution.py:109-139`` (legacy) pour garantir
    l'équivalence numérique du compteur ``doc_idx`` à
    ``run_benchmark_via_service``.

    Le ``counter_lock`` partagé empêche les race conditions quand le
    ``CorpusRunner`` traite plusieurs documents en parallèle
    (``max_in_flight > 1``).  Le compteur est **global au run**, pas
    par pipeline — un benchmark à 2 pipelines × 5 docs émet 10
    notifications ``doc_idx ∈ {0..9}``.

    Phase B4 — ``workspace_uri`` propagé au ``RunContext`` pour que les
    adapters qui en ont besoin (PrecomputedTextAdapter,
    TesseractAdapter, etc.) puissent écrire leurs artefacts
    intermédiaires.  Cohérent avec ``_benchmark_execution.py:134-139``.
    """
    counter_lock = threading.Lock()
    counter_state = {"doc_idx": 0}

    def _factory(doc: DocumentRef, pipeline_name: str) -> RunContext:
        if progress_callback is not None:
            with counter_lock:
                idx = counter_state["doc_idx"]
                counter_state["doc_idx"] = idx + 1
            try:
                progress_callback(pipeline_name, idx, doc.id)
            except Exception:
                # On ignore silencieusement les erreurs du callback :
                # un caller qui crashe ne doit pas faire tomber le
                # benchmark.  Cohérent avec
                # ``_benchmark_execution.py:126-133``.
                import logging
                logging.getLogger(__name__).debug(
                    "[run_orchestrator] progress_callback levé — "
                    "ignoré pour ne pas tomber le bench.",
                    exc_info=True,
                )
        return RunContext(
            document_id=doc.id,
            code_version=code_version,
            pipeline_name=pipeline_name,
            workspace_uri=workspace_uri,
        )
    return _factory


def _filesystem_payload_loader(art: Artifact) -> Any:
    """Loader filesystem : lit RAW_TEXT/CORRECTED_TEXT depuis le
    fichier pointé par l'URI, parse ALTO_XML depuis le fichier pointé.

    Les artefacts projetés (sans URI) ne passent pas par ce loader —
    l'executor utilise directement le payload retourné par le
    projecteur.
    """
    if art.uri is None:
        raise FileNotFoundError(
            f"Loader filesystem : artifact {art.id!r} sans URI ; "
            "un projecteur aurait dû fournir le payload.",
        )
    path = Path(art.uri)
    if art.type == ArtifactType.ALTO_XML:
        return parse_alto(path.read_bytes())
    if art.type in (ArtifactType.RAW_TEXT, ArtifactType.CORRECTED_TEXT):
        return path.read_text(encoding="utf-8")
    raise ValueError(
        f"Loader filesystem : type {art.type.value!r} non géré.",
    )


__all__ = [
    "OrchestrationResult",
    "RunOrchestrator",
]
