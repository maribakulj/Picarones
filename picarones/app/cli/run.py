"""``picarones-rewrite run`` — workflow benchmark complet via YAML.

Sprint A14-S24.

Orchestre tous les services applicatifs livrés en S17-S23 pour
exécuter un benchmark bout-en-bout depuis une spec YAML déclarative,
sans écrire de Python :

::

    python -m picarones.app.cli run --spec ./run.yaml

Étapes
------
1. Charger ``run.yaml`` via :func:`load_run_spec_from_yaml`.
2. Créer un ``WorkspaceManager`` (S19) sous ``output_dir``.
3. Importer le corpus (S20) — depuis ZIP ou depuis un dir
   pré-extrait.
4. Bootstrap du ``RegistryService`` (S23).
5. Construire les ``PipelineSpec`` à partir de la YAML (résolution
   dotted-path des adapters via :func:`resolve_adapter_class`).
6. Construire les vues canoniques (TextView/AltoView/SearchView)
   demandées dans ``views``.
7. Construire le ``BenchmarkService`` (S17) et lancer ``run()``.
8. Persister les 3 fichiers JSONL.
9. (Optionnel) Générer le rapport HTML via ``ReportService`` (S21).

Limitations MVP S24 (mises à jour S25)
--------------------------------------
- Vues : seulement les 3 canoniques (``text_final``,
  ``alto_documentary``, ``searchability``).
- ~~Projection ALTO → texte non câblée bout-en-bout~~ — **levée
  au S25** : le projecteur retourne désormais le payload calculé
  via ``(Artifact, payload, ProjectionReport)``, l'executor
  l'utilise directement sans repasser par le loader.  Un pipeline
  produisant ALTO_XML évalué via TextView projeté fonctionne
  bout-en-bout.
- ``ground_truth_factory`` / ``pipeline_inputs_factory`` /
  ``context_factory`` : versions filesystem-by-default minimales
  (cf. helpers privés en bas du module).

Codes de sortie : 0 succès, 1 erreur typée, 2 erreur d'usage Click.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from picarones.app.services import (
    BenchmarkService,
    CorpusImportError,
    CorpusService,
    RegistryService,
    ReportService,
    WorkspaceManager,
)
from picarones.app.services.run_spec import (
    RunSpec,
    RunSpecLoadError,
    load_run_spec_from_yaml,
    resolve_adapter_class,
)
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.corpus import CorpusSpec
from picarones.domain.documents import DocumentRef
from picarones.domain.run_result import RunResult
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


@click.command()
@click.option(
    "--spec",
    "spec_path",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, path_type=Path,
    ),
    required=True,
    help="Chemin du fichier YAML décrivant le run.",
)
@click.option(
    "--no-report",
    is_flag=True,
    default=False,
    help=(
        "Ne génère pas le rapport HTML, même si ``report_html`` "
        "est défini dans la spec."
    ),
)
def run_command(spec_path: Path, no_report: bool) -> None:
    """Exécute un benchmark complet depuis une spec YAML."""
    # 1. Charger la spec.
    try:
        spec = load_run_spec_from_yaml(spec_path.read_text(encoding="utf-8"))
    except RunSpecLoadError as exc:
        click.echo(f"erreur : spec invalide : {exc}", err=True)
        sys.exit(1)

    # 2. Workspace.
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = WorkspaceManager(output_dir)

    # 3. Corpus.
    try:
        corpus_spec, extracted_dir = _load_corpus(spec, workspace)
    except CorpusImportError as exc:
        click.echo(f"erreur : import corpus : {exc}", err=True)
        sys.exit(1)
    click.echo(
        f"Corpus chargé : {corpus_spec.name} "
        f"({len(corpus_spec.documents)} docs, {extracted_dir})",
    )

    # 4. Registries.
    registries = RegistryService.bootstrap_defaults()

    # 5. Pipelines.
    try:
        pipeline_specs, adapter_resolver = _build_pipelines(spec)
    except RunSpecLoadError as exc:
        click.echo(f"erreur : résolution pipeline : {exc}", err=True)
        sys.exit(1)

    # 6. Vues.
    views = _build_views(spec.views)

    # 7. BenchmarkService.
    pipeline_executor = PipelineExecutor(adapter_resolver=adapter_resolver)
    corpus_runner = CorpusRunner(
        pipeline_executor,
        max_in_flight=2,
        timeout_seconds_per_doc=300.0,
        poll_interval_seconds=0.05,
    )
    view_executor = DefaultEvaluationViewExecutor(
        registries.metrics,
        registries.projectors,
        _make_filesystem_loader(),
    )
    bench = BenchmarkService(
        corpus_runner=corpus_runner,
        view_executor=view_executor,
        code_version=spec.code_version,
    )

    click.echo(
        f"Lancement du run : {len(pipeline_specs)} pipeline(s) × "
        f"{len(views)} vue(s) × {len(corpus_spec.documents)} doc(s)…",
    )
    result = bench.run(
        corpus=corpus_spec,
        pipelines=pipeline_specs,
        views=views,
        ground_truth_factory=_default_gt_factory,
        pipeline_inputs_factory=_default_inputs_factory,
        context_factory=_make_context_factory(spec.code_version),
        metadata={"operator_cli": "picarones-rewrite-s24"},
    )

    # 8. Persistance.
    persist_dir = output_dir / "results"
    files = bench.persist(result, persist_dir)
    click.echo(f"Run persisté dans : {persist_dir}")
    for kind, path in files.items():
        click.echo(f"  {kind}: {path}")

    # 9. Rapport HTML optionnel.
    if spec.report_html and not no_report:
        report_service = ReportService(lang=spec.report_lang)
        html = report_service.render(result)
        report_path = Path(spec.report_html)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html, encoding="utf-8")
        click.echo(f"Rapport HTML : {report_path}")

    click.echo("OK")


# ──────────────────────────────────────────────────────────────────────
# Helpers privés
# ──────────────────────────────────────────────────────────────────────


def _load_corpus(
    spec: RunSpec, workspace: WorkspaceManager,
) -> tuple[CorpusSpec, Path]:
    """Charge le corpus selon ``corpus_zip`` ou ``corpus_dir``.

    Pour ``corpus_dir``, on construit directement un ``CorpusSpec``
    depuis les fichiers présents (réutilise la détection de patterns
    de ``CorpusService`` via la même classification interne).
    """
    if spec.corpus_zip is not None:
        zip_path = Path(spec.corpus_zip)
        zip_bytes = zip_path.read_bytes()
        corpus_service = CorpusService(workspace)
        report = corpus_service.import_zip(
            zip_bytes,
            corpus_name=spec.corpus_name or zip_path.stem,
            metadata=spec.corpus_metadata,
        )
        return report.spec, report.extracted_dir
    # corpus_dir : on délègue à un import en mode "déjà extrait".
    # MVP S24 : on zippe à la volée le contenu du dir et on délègue
    # à CorpusService — réutilise toute la détection sans dupliquer.
    import io
    import zipfile

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
    corpus_service = CorpusService(workspace)
    report = corpus_service.import_zip(
        buf.getvalue(),
        corpus_name=spec.corpus_name or src_dir.name,
        metadata=spec.corpus_metadata,
    )
    return report.spec, report.extracted_dir


def _build_pipelines(spec: RunSpec) -> tuple[
    list[PipelineSpec], "callable",
]:
    """Construit les ``PipelineSpec`` + un ``adapter_resolver`` qui
    instancie les adapters au besoin.

    Le resolver maintient un cache instance-par-nom (un adapter est
    instancié une seule fois pour tout le run).
    """
    instance_cache: dict[str, object] = {}
    name_to_class: dict[str, type] = {}
    name_to_kwargs: dict[str, dict] = {}

    pipeline_specs: list[PipelineSpec] = []
    for p in spec.pipelines:
        steps = []
        for s in p.steps:
            cls = resolve_adapter_class(s.adapter_class)
            adapter_name = s.id
            # Si le même step.id apparaît dans deux pipelines avec
            # des classes différentes, on disambiguë par la pipeline.
            if (
                adapter_name in name_to_class
                and name_to_class[adapter_name] is not cls
            ):
                adapter_name = f"{p.name}__{s.id}"
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

    def resolver(name: str):
        if name not in instance_cache:
            cls = name_to_class[name]
            kwargs = name_to_kwargs[name]
            instance_cache[name] = cls(**kwargs)
        return instance_cache[name]

    return pipeline_specs, resolver


def _build_views(view_names: tuple[str, ...]):
    """Map noms canoniques → vues construites."""
    builders = {
        "text_final": build_text_view,
        "alto_documentary": build_alto_view,
        "searchability": build_search_view,
    }
    return [builders[name]() for name in view_names]


def _default_gt_factory(doc: DocumentRef, art_type: ArtifactType):
    """Factory GT par défaut : retourne un Artifact pointant sur la
    GT du type demandé si présente, ou tombe sur RAW_TEXT pour
    CORRECTED_TEXT (les deux sont du texte plat — convention S18)."""
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


def _default_inputs_factory(doc: DocumentRef):
    """``{IMAGE: artifact_image}``."""
    return {ArtifactType.IMAGE: Artifact(
        id=f"{doc.id}:image",
        document_id=doc.id,
        type=ArtifactType.IMAGE,
        uri=doc.image_uri,
    )}


def _make_context_factory(code_version: str):
    def _factory(doc: DocumentRef, pipeline_name: str) -> RunContext:
        return RunContext(
            document_id=doc.id,
            code_version=code_version,
            pipeline_name=pipeline_name,
        )
    return _factory


def _make_filesystem_loader():
    """Loader filesystem MVP : lit RAW_TEXT depuis le fichier
    pointé par l'URI, parse ALTO_XML depuis le fichier pointé.

    Sprint S25 : les artefacts projetés (sans URI) ne sont plus un
    problème — l'executor utilise directement le payload retourné
    par le projecteur, le loader n'est plus appelé pour ces cas.
    Le loader ne gère donc que les artefacts avec URI (candidats
    directs et GT).
    """

    def loader(art: Artifact):
        if art.uri is None:
            raise FileNotFoundError(
                f"Loader CLI : artifact {art.id!r} sans URI ; "
                "appelez le projecteur d'abord pour produire le "
                "payload (S25)."
            )
        path = Path(art.uri)
        if art.type == ArtifactType.ALTO_XML:
            return parse_alto(path.read_bytes())
        if art.type in (
            ArtifactType.RAW_TEXT,
            ArtifactType.CORRECTED_TEXT,
        ):
            return path.read_text(encoding="utf-8")
        raise ValueError(
            f"Loader CLI : type {art.type.value!r} non géré.",
        )

    return loader


# Réexpose une fonction utile aux tests d'intégration.
def _execute_run_for_tests(spec: RunSpec, output_dir: Path) -> RunResult:
    """Exécute le run sans passer par Click — utilisé par les tests
    d'intégration qui veulent inspecter le ``RunResult`` directement."""
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = WorkspaceManager(output_dir)
    corpus_spec, _ = _load_corpus(spec, workspace)
    registries = RegistryService.bootstrap_defaults()
    pipeline_specs, adapter_resolver = _build_pipelines(spec)
    views = _build_views(spec.views)
    pipeline_executor = PipelineExecutor(adapter_resolver=adapter_resolver)
    corpus_runner = CorpusRunner(
        pipeline_executor,
        max_in_flight=2,
        timeout_seconds_per_doc=300.0,
        poll_interval_seconds=0.05,
    )
    view_executor = DefaultEvaluationViewExecutor(
        registries.metrics,
        registries.projectors,
        _make_filesystem_loader(),
    )
    bench = BenchmarkService(
        corpus_runner=corpus_runner,
        view_executor=view_executor,
        code_version=spec.code_version,
    )
    return bench.run(
        corpus=corpus_spec,
        pipelines=pipeline_specs,
        views=views,
        ground_truth_factory=_default_gt_factory,
        pipeline_inputs_factory=_default_inputs_factory,
        context_factory=_make_context_factory(spec.code_version),
    )


__all__ = ["run_command"]
