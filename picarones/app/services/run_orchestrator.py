"""``RunOrchestrator`` — exécute un benchmark complet depuis un ``RunSpec``.

Service applicatif qui assemble :

- ``CorpusService`` (import du corpus depuis ZIP ou dir extrait),
- ``RegistryService`` (bootstrap des registres),
- ``BenchmarkService`` (orchestration runner + vues + persistance),
- ``ReportService`` (rendu HTML optionnel).

C'est le « workflow par défaut » d'un run YAML.  Il vit dans
``app/services/`` (couche métier, pas couche d'interface) pour que
toutes les interfaces (CLI Click, futur HTTP, scripts Python tiers)
puissent l'invoquer sans dupliquer la logique d'orchestration.

Anti-bricolage
--------------
Pas de fonction-helper privée éparpillée dans la CLI.  L'interface
``picarones-rewrite run`` est désormais un thin wrapper Click qui
appelle ``RunOrchestrator.execute(spec)`` et formate la sortie.

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
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from picarones.app.results import RunResult
from picarones.app.schemas import RunSpec, resolve_adapter_class
from picarones.app.services.benchmark_service import BenchmarkService
from picarones.app.services.corpus_service import (
    CorpusImportError,
    CorpusService,
)
from picarones.app.services.path_security import WorkspaceManager
from picarones.app.services.registry_service import RegistryService
from picarones.app.services.report_service import ReportService
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
    report_html_path:
        Chemin du rapport HTML écrit, ou ``None`` si pas demandé.
    """

    run_result: RunResult
    extracted_corpus_dir: Path
    persisted_files: dict[str, Path] = field(default_factory=dict)
    report_html_path: Path | None = None


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
        emit_report: bool = True,
    ) -> OrchestrationResult:
        """Exécute le run complet et retourne tout ce qu'on en sait.

        Parameters
        ----------
        spec:
            ``RunSpec`` validée (pydantic).
        emit_report:
            Si ``True`` (défaut) ET que ``spec.report_html`` est
            renseigné, génère le rapport HTML.  Sinon, retourne
            ``OrchestrationResult.report_html_path = None``.

        Raises
        ------
        CorpusImportError
            Si le corpus ne peut pas être chargé.
        RunSpecLoadError
            Si la résolution dotted-path d'un ``adapter_class``
            échoue.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        workspace = WorkspaceManager(self._output_dir)

        # 1. Corpus.
        corpus_spec, extracted_dir = self._load_corpus(spec, workspace)

        # 2. Registres.
        registries = RegistryService.bootstrap_defaults()

        # 3. Pipelines + resolver d'adapters.
        pipeline_specs, adapter_resolver = self._build_pipelines(spec)

        # 4. Vues canoniques.
        views = self._build_views(spec.views)

        # 5. BenchmarkService.
        bench = self._build_benchmark_service(
            registries=registries,
            adapter_resolver=adapter_resolver,
            code_version=spec.code_version,
        )

        result = bench.run(
            corpus=corpus_spec,
            pipelines=pipeline_specs,
            views=views,
            ground_truth_factory=_default_gt_factory,
            pipeline_inputs_factory=_default_inputs_factory,
            context_factory=_make_context_factory(spec.code_version),
            metadata={"orchestrator": "picarones.app.services.run_orchestrator"},
        )

        # 6. Persistance JSONL.
        persist_dir = self._output_dir / "results"
        persisted = bench.persist(result, persist_dir)

        # 7. Rapport HTML optionnel.
        report_path: Path | None = None
        if emit_report and spec.report_html:
            report_service = ReportService(lang=spec.report_lang)
            html = report_service.render(result)
            report_path = Path(spec.report_html)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(html, encoding="utf-8")

        return OrchestrationResult(
            run_result=result,
            extracted_corpus_dir=extracted_dir,
            persisted_files=persisted,
            report_html_path=report_path,
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
    ) -> tuple[list[PipelineSpec], Callable[[str], Any]]:
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
                steps.append(PipelineStep(
                    id=s.id,
                    kind="step",
                    adapter_name=adapter_name,
                    input_types=s.input_types,
                    output_types=s.output_types,
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

        return pipeline_specs, resolver

    @staticmethod
    def _build_views(view_names: tuple[str, ...]) -> list[Any]:
        """Map noms canoniques → vues construites."""
        builders = {
            "text_final": build_text_view,
            "alto_documentary": build_alto_view,
            "searchability": build_search_view,
        }
        return [builders[name]() for name in view_names]

    @staticmethod
    def _build_benchmark_service(
        *,
        registries: RegistryService,
        adapter_resolver: Callable[[str], Any],
        code_version: str,
    ) -> BenchmarkService:
        """Assemble ``BenchmarkService`` avec un loader filesystem."""
        pipeline_executor = PipelineExecutor(
            adapter_resolver=adapter_resolver,
        )
        corpus_runner = CorpusRunner(
            pipeline_executor,
            max_in_flight=2,
            timeout_seconds_per_doc=300.0,
            poll_interval_seconds=0.05,
        )
        view_executor = DefaultEvaluationViewExecutor(
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
) -> Callable[[DocumentRef, str], RunContext]:
    def _factory(doc: DocumentRef, pipeline_name: str) -> RunContext:
        return RunContext(
            document_id=doc.id,
            code_version=code_version,
            pipeline_name=pipeline_name,
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
