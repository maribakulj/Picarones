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

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

from picarones.app.results import ReportRenderer, RunResult
from picarones.app.schemas import RunSpec
from picarones.app.services.dependencies import (
    capture_dependencies_lock,
    capture_system_binaries_lock,
)
from picarones.app.services.path_security import WorkspaceManager
from picarones.app.services.registry_service import RegistryService
from picarones.domain.corpus import CorpusSpec
from picarones.pipeline import (
    PipelineSpec,
)

# Helpers stateless extraits (audit prod P1 — dégonflage god-module).
# Réimportés ici pour préserver l'API : ``from
# picarones.app.services.run_orchestrator import _default_gt_factory``
# reste valide, les call sites internes utilisent le nom module-global
# (donc ``monkeypatch.setattr(run_orchestrator, …)`` fonctionne aussi).
from picarones.app.services.run_orchestrator_helpers import (
    _PipelineEngineProxy as _PipelineEngineProxy,
    _build_benchmark_service as _build_benchmark_service,
    _build_pipelines as _build_pipelines,
    _build_views as _build_views,
    _default_gt_factory as _default_gt_factory,
    _default_inputs_factory as _default_inputs_factory,
    _filesystem_payload_loader as _filesystem_payload_loader,
    _kwargs_signature as _kwargs_signature,
    _load_corpus as _load_corpus,
    _make_context_factory as _make_context_factory,
    _persist_legacy_benchmark_json as _persist_legacy_benchmark_json,
    _resolve_entity_extractor as _resolve_entity_extractor,
)

# Phase B — gros bloc stateful extrait (ex-_execute_with_partial,
# ~283 l).  Réimporté ici : call sites internes basculés sur le nom
# module-global (cohérent avec _persist/_load_corpus/...).
from picarones.app.services.run_orchestrator_execution import (
    execute_with_partial as _execute_with_partial,
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
        corpus_spec, extracted_dir = _load_corpus(spec, workspace)

        # 2. Registres.
        registries = RegistryService.bootstrap_defaults()

        # 3. Pipelines + resolver d'adapters + dump des kwargs pour le manifest.
        pipeline_specs, adapter_resolver, adapter_kwargs = (
            self._build_pipelines(spec)
        )

        # 4. Vues canoniques.  Phase B2.5 — propage normalization +
        # char_exclude aux vues text_final/searchability.
        views = _build_views(
            spec.views,
            normalization_profile=spec.normalization_profile,
            char_exclude=spec.char_exclude,
        )

        # 5. BenchmarkService.
        bench = _build_benchmark_service(
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
            result = _execute_with_partial(
                spec=spec,
                bench=bench,
                corpus_spec=corpus_spec,
                pipeline_specs=pipeline_specs,
                views=views,
                adapter_kwargs=adapter_kwargs,
                deps_lock=deps_lock,
                bin_lock=bin_lock,
                runtime_dir=runtime_dir,
                progress_callback=self._progress_callback,
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
            _persist_legacy_benchmark_json(
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
        views = _build_views(
            spec.views,
            normalization_profile=spec.normalization_profile,
            char_exclude=spec.char_exclude,
        )
        bench = _build_benchmark_service(
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
            result = _execute_with_partial(
                spec=spec,
                bench=bench,
                corpus_spec=corpus_spec,
                pipeline_specs=pipeline_specs,
                views=views,
                adapter_kwargs=adapter_kwargs_clean,
                deps_lock=deps_lock,
                bin_lock=bin_lock,
                runtime_dir=runtime_dir,
                progress_callback=self._progress_callback,
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
            _persist_legacy_benchmark_json(
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
        """Wrapper mince — délègue à
        :func:`run_orchestrator_helpers.builders._load_corpus`
        (audit Phase A).  Conservé comme ``@staticmethod`` car un test
        de parité l'appelle via ``orch._load_corpus(...)`` pour
        recalculer un fingerprint de partial cohérent."""
        return _load_corpus(spec, workspace)

    @staticmethod
    def _build_pipelines(
        spec: RunSpec,
    ) -> tuple[list[PipelineSpec], Callable[[str], Any], dict[str, dict[str, Any]]]:
        """Wrapper mince — délègue à
        :func:`run_orchestrator_helpers.builders._build_pipelines`
        (audit Phase A : corps extrait hors du god-module).  Conservé
        comme ``@staticmethod`` car un test l'appelle via
        ``orch._build_pipelines(spec)`` ; ``_build_pipelines`` réfère
        ici le nom module-global réimporté (pas de récursion : la
        méthode de classe n'est pas dans les globals de la fonction)."""
        return _build_pipelines(spec)


__all__ = [
    "OrchestrationResult",
    "RunOrchestrator",
]
