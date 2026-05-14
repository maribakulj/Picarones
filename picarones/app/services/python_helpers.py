"""Helpers pour invoquer ``RunOrchestrator`` depuis du code Python qui
instancie ses adapters en mémoire (par opposition au chargement depuis
un YAML via :class:`RunSpec`).

API publique
------------
- :class:`PresetArgs` — dataclass qui agrège les objets domain prêts
  à passer à :meth:`RunOrchestrator.execute_preset`.
- :func:`prepare_preset_args` — convertit ``(Corpus legacy + liste
  d'instances d'adapters)`` en :class:`PresetArgs`.

Pattern d'usage canonique
-------------------------

::

    from picarones import RunOrchestrator
    from picarones.app.services import (
        prepare_preset_args,
        run_result_to_benchmark_result,
    )
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as ws:
        ws_path = Path(ws)
        args = prepare_preset_args(
            corpus, engines,
            workspace_dir=ws_path / "gt",
            views=("text_final", "alto_documentary"),
            normalization_profile="caseless",
            profile="standard",
        )
        orch_result = RunOrchestrator(ws_path / "run").execute_preset(
            spec=args.spec,
            corpus_spec=args.corpus_spec,
            extracted_dir=args.extracted_dir,
            pipeline_specs=args.pipeline_specs,
            adapter_resolver=args.adapter_resolver,
            adapter_kwargs=args.adapter_kwargs,
            progress_callback=cb,  # optionnel
            cancel_event=ev,       # optionnel
        )
        # Si l'on veut un BenchmarkResult legacy (rapport HTML, etc.) :
        benchmark = run_result_to_benchmark_result(
            orch_result.run_result,
            corpus=corpus, engines=engines,
            normalization_profile="caseless", profile="standard",
        )

Pourquoi 3 étapes et pas une seule fonction ?
---------------------------------------------
Volontairement explicite : chaque étape (préparation → exécution →
conversion legacy) est visible dans le call site et testable
isolément.  Un caller qui n'a pas besoin du ``BenchmarkResult``
legacy peut sauter la 3e étape et consommer directement le
``RunResult`` typé du :class:`OrchestrationResult`.

Pour les callers YAML (CI, scripts reproductibles), passer par
:meth:`RunOrchestrator.execute(spec)` avec un :class:`RunSpec`
sérialisable plutôt que par ce helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from picarones.app.schemas.run_spec import RunSpec
    from picarones.domain.corpus import CorpusSpec
    from picarones.domain.pipeline_spec import PipelineSpec
    from picarones.evaluation.corpus import Corpus


@dataclass(frozen=True)
class PresetArgs:
    """Objets domain pré-construits pour
    :meth:`RunOrchestrator.execute_preset`.

    Attributs
    ---------
    spec:
        ``RunSpec`` qui porte les paramètres (views, char_exclude,
        normalization_profile, partial_dir, entity_extractor,
        profile, output_json, timeout, code_version).  Sa partie
        ``corpus_dir`` + ``pipelines`` est **ignorée** par
        ``execute_preset`` (placeholders Pydantic).
    corpus_spec:
        ``CorpusSpec`` (couche 1, domain) construit depuis le
        ``Corpus`` legacy via ``corpus_to_corpus_spec``.
    extracted_dir:
        Dossier où les images source du corpus sont accessibles
        (utilisé par le converter legacy si ``output_json`` est
        renseigné).
    pipeline_specs:
        Liste de ``PipelineSpec`` (couche 1) construite via
        ``engine_to_pipeline_spec`` pour chaque engine fourni.
    adapter_resolver:
        Resolver ``name → StepExecutor`` construit via
        ``build_adapter_resolver`` qui mappe chaque adapter à son
        instance pour ``PipelineExecutor``.
    adapter_kwargs:
        Map ``adapter_name → kwargs dict`` pour le manifest.  Vide
        par défaut.
    """

    spec: "RunSpec"
    corpus_spec: "CorpusSpec"
    extracted_dir: Path
    pipeline_specs: list["PipelineSpec"]
    adapter_resolver: Callable[[str], Any]
    adapter_kwargs: dict[str, Any]


def _dummy_pipeline_yaml(name: str = "preset_pipeline") -> Any:
    """``PipelineSpecYaml`` minimaliste pour passer le validator
    Pydantic de ``RunSpec.pipelines`` (min_length=1).  Le contenu
    est **ignoré** par ``execute_preset`` qui utilise les
    ``pipeline_specs`` du :class:`PresetArgs`.
    """
    from picarones.app.schemas.run_spec import PipelineSpecYaml, StepSpec
    from picarones.domain.artifacts import ArtifactType
    return PipelineSpecYaml(
        name=name,
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(StepSpec(
            id="ocr",
            adapter_class="picarones.app.services.python_helpers.IgnoredByPreset",
            adapter_kwargs={},
            input_types=(ArtifactType.IMAGE,),
            output_types=(ArtifactType.RAW_TEXT,),
        ),),
    )


def prepare_preset_args(
    corpus: "Corpus",
    engines: list[Any],
    *,
    workspace_dir: Path,
    views: tuple[str, ...] = ("text_final",),
    char_exclude: Any | None = None,
    normalization_profile: Any | None = None,
    partial_dir: str | Path | None = None,
    entity_extractor: str | None = None,
    profile: str = "standard",
    output_json: str | Path | None = None,
    timeout_seconds_per_doc: float = 60.0,
    code_version: str | None = None,
    output_dir: str | Path | None = None,
) -> PresetArgs:
    """Convertit ``(Corpus legacy + instances d'adapters)`` en
    objets domain prêts pour :meth:`RunOrchestrator.execute_preset`.

    Parameters
    ----------
    corpus:
        ``picarones.evaluation.corpus.Corpus`` legacy (en mémoire,
        avec ``Document.image_path`` et ``ground_truth``).
    engines:
        Liste d'instances ``BaseOCRAdapter`` ou
        ``OCRLLMPipelineConfig``.  Chaque instance doit exposer
        ``.name`` unique.
    workspace_dir:
        Dossier où sérialiser les GT pour ``corpus_to_corpus_spec``.
        Typiquement ``Path(tmp).joinpath("gt")``.  Doit exister.
    views:
        Noms canoniques des vues à appliquer.  Défaut :
        ``("text_final",)``.  Valeurs valides : ``"text_final"``,
        ``"alto_documentary"``, ``"searchability"``.
    char_exclude, normalization_profile, partial_dir,
    entity_extractor, profile, output_json, timeout_seconds_per_doc,
    code_version:
        Paramètres propagés au ``RunSpec``.  Voir
        :class:`picarones.RunSpec` pour les contrats.

        - ``char_exclude`` accepte ``str`` ou ``frozenset[str]``
          (auto-converti en string).
        - ``normalization_profile`` accepte ``str`` ou objet
          ``NormalizationProfile`` (le nom est extrait).
    output_dir:
        Dossier où ``RunOrchestrator`` écrira ses 4 fichiers JSONL.
        Si ``None``, défaut ``workspace_dir.parent / "run"``.

    Returns
    -------
    :class:`PresetArgs`

    Notes
    -----
    Aucune ressource externe n'est créée par cette fonction (pas
    de tempdir, pas de fichier).  Le caller est responsable du
    cycle de vie du ``workspace_dir`` (typiquement via
    ``tempfile.TemporaryDirectory``).

    Limite reproductibilité manifest (audit B3-final mai 2026,
    trou #2) : ``PresetArgs.adapter_kwargs`` est retourné vide
    (``{}``) car le mode preset reçoit des instances d'adapters
    déjà construites en mémoire ; les kwargs de construction ne
    sont pas réintrospectables génériquement.  Conséquence :
    ``RunManifest.adapter_kwargs`` sera vide dans le manifest
    persisté.  Pour reproduire un run preset à l'identique,
    relancer le code Python qui a construit les instances — le
    manifest seul n'est qu'un audit log informatif côté preset.
    Le mode ``execute(spec_yaml)`` capture lui les vraies kwargs
    et permet la reproductibilité complète depuis le manifest.
    """
    from picarones.app.schemas.run_spec import RunSpec
    from picarones.app.services._benchmark_adapter_resolver import (
        build_adapter_resolver,
        engine_to_pipeline_spec,
    )
    from picarones.app.services._benchmark_conversions import (
        corpus_to_corpus_spec,
    )

    if code_version is None:
        import importlib
        try:
            code_version = importlib.import_module("picarones").__version__
        except (ImportError, AttributeError):
            code_version = "unknown"

    workspace_dir = Path(workspace_dir)
    if not workspace_dir.exists():
        workspace_dir.mkdir(parents=True, exist_ok=True)
    effective_output_dir = (
        Path(output_dir) if output_dir
        else workspace_dir.parent / "run"
    )

    corpus_spec = corpus_to_corpus_spec(corpus, workspace_dir=workspace_dir)
    pipeline_specs = [engine_to_pipeline_spec(e) for e in engines]
    adapter_resolver = build_adapter_resolver(engines)

    # Normalisation des params hétérogènes legacy → RunSpec string.
    char_exclude_str: str | None = None
    if char_exclude is not None:
        if isinstance(char_exclude, str):
            char_exclude_str = char_exclude
        else:
            char_exclude_str = "".join(sorted(char_exclude))

    norm_profile_str = normalization_profile
    if normalization_profile is not None and not isinstance(
        normalization_profile, str,
    ):
        norm_profile_str = getattr(normalization_profile, "name", None)

    spec = RunSpec(
        corpus_dir=str(workspace_dir.parent),  # ignoré par execute_preset
        pipelines=(_dummy_pipeline_yaml(),),
        views=views,
        output_dir=str(effective_output_dir),
        char_exclude=char_exclude_str,
        normalization_profile=norm_profile_str,
        partial_dir=str(partial_dir) if partial_dir else None,
        entity_extractor=entity_extractor,
        profile=profile,
        output_json=str(output_json) if output_json else None,
        code_version=code_version,
        timeout_seconds_per_doc=timeout_seconds_per_doc,
    )

    return PresetArgs(
        spec=spec,
        corpus_spec=corpus_spec,
        extracted_dir=workspace_dir,
        pipeline_specs=pipeline_specs,
        adapter_resolver=adapter_resolver,
        adapter_kwargs={},
    )


__all__ = ["PresetArgs", "prepare_preset_args"]
