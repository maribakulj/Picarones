"""``RunSpec`` — déclaration YAML d'un run benchmark.

Sprint A14-S24 / S39 du rewrite ciblé.

Format qui décrit un run complet en YAML : corpus, pipelines
hétérogènes (potentiellement avec DAG branchant), vues canoniques à
appliquer, sortie HTML.  Permet à l'utilisateur BnF de lancer un
benchmark via la CLI sans écrire de Python.

Format
------

::

    corpus_zip: ./bnf.zip                       # OU corpus_dir
    corpus_dir: ./extracted/                    # mutuellement exclusif
    corpus_name: bnf_xviiie                     # optionnel (défaut : stem)
    corpus_metadata:
      language: fr
      period: early_modern

    pipelines:
      - name: ocr_then_correct
        initial_inputs: [image]
        # output symbolique préféré pour le texte.
        # Référence un (step_id).(output_type) qui sera utilisé par
        # les vues TextView / SearchView quand plusieurs steps
        # produisent du RAW_TEXT.  Optionnel.
        preferred_text_output: corrector.corrected_text
        steps:
          - id: ocr
            adapter_class: my_pkg.adapters.TesseractAdapter
            adapter_kwargs: {lang: fra}
            input_types: [image]
            output_types: [raw_text]
          - id: corrector
            adapter_class: my_pkg.adapters.LLMCorrector
            adapter_kwargs: {model: gpt-4o}
            input_types: [raw_text]
            output_types: [corrected_text]
            # DAG branchant.  Si plusieurs steps
            # produisent le même type, on désigne explicitement la
            # source.  Sans inputs_from : dernier producteur.
            inputs_from:
              raw_text: ocr

    views: [text_final, searchability]          # noms canoniques

    output_dir: ./runs/r1
    report_html: ./runs/r1/rapport.html         # optionnel
    report_lang: fr
    code_version: "1.0.0-rewrite"

Conventions
-----------
- ``corpus_zip`` ou ``corpus_dir`` est requis (pas les deux).
- ``views`` accepte uniquement les noms canoniques :
  ``text_final``, ``alto_documentary``, ``searchability``.  Le
  caller qui veut des vues custom passe par l'API Python directe.
- ``adapter_class`` est un dotted path Python.  La classe doit être
  importable au moment du run (l'utilisateur installe ses propres
  packages dans le venv courant).
- ``adapter_kwargs`` est passé tel quel au constructeur.
- ``inputs_from`` (S39) : map ``ArtifactType → step_id`` qui désigne
  explicitement la source d'un input.  ``__initial__`` désigne les
  entrées initiales du runner.  Sans ``inputs_from``, l'executor
  prend le dernier producteur de chaque type.
- ``preferred_text_output`` (S39) : référence symbolique
  ``step_id.output_type`` qui désigne quelle sortie de pipeline est
  préférée pour les vues textuelles (utile quand plusieurs steps
  produisent du RAW_TEXT ou du CORRECTED_TEXT).  Optionnel.

Anti-sur-ingénierie
-------------------
- Pas de templating Jinja2 dans le YAML (variables d'env, includes).
  Si un caller veut composer plusieurs YAMLs, il les concatène en
  Python.
- Pas de schéma JSON publié — pydantic est l'autorité.  Le format
  évoluera avec le rewrite ; la stabilité sera tagguée à la
  livraison BnF.
- Pas de validation des dépendances de package — si la classe n'est
  pas importable au runtime, on échoue lisiblement.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from picarones.domain.artifacts import ArtifactType
from picarones.domain.errors import PicaronesError


#: Format autorisé pour ``entity_extractor`` (Phase B1 migration Option B).
#: Accepte les deux conventions de dotted path Python :
#:
#:  - ``module.submodule:Symbol`` (PEP 621 entry points / setuptools)
#:  - ``module.submodule.Symbol`` (import classique)
#:
#: La validation est purement structurelle ici — l'importabilité est
#: vérifiée plus tard, au moment de :meth:`RunOrchestrator.execute`
#: (lazy resolve).  Cohérent avec ``adapter_class`` (cf. ``StepSpec``).
_DOTTED_PATH_RE = re.compile(
    r"^[a-zA-Z_][a-zA-Z0-9_]*"           # premier composant
    r"(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*"     # composants intermédiaires
    r"(?:[:.][a-zA-Z_][a-zA-Z0-9_]*)$"   # séparateur final (``:`` ou ``.``) + symbole
)


#: Vues canoniques supportées par la CLI.
CANONICAL_VIEW_NAMES: frozenset[str] = frozenset({
    "text_final",
    "alto_documentary",
    "searchability",
})


# ──────────────────────────────────────────────────────────────────────
# Schéma pydantic
# ──────────────────────────────────────────────────────────────────────


class StepSpec(BaseModel):
    """Description d'un step de pipeline dans la spec YAML."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    adapter_class: str = Field(
        min_length=1, max_length=512,
        description="Dotted path Python vers la classe adapter.",
    )
    adapter_kwargs: dict[str, Any] = Field(default_factory=dict)
    input_types: tuple[ArtifactType, ...] = Field(...)
    output_types: tuple[ArtifactType, ...] = Field(...)
    inputs_from: dict[ArtifactType, str] = Field(
        default_factory=dict,
        description=(
            "Sprint S39 — DAG branchant : map ``ArtifactType → step_id`` "
            "qui désigne explicitement la source d'un input. "
            "``__initial__`` pour les entrées initiales du runner. "
            "Sans ``inputs_from``, l'executor prend le dernier producteur."
        ),
    )


class PipelineSpecYaml(BaseModel):
    """Description d'une pipeline dans la spec YAML."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    initial_inputs: tuple[ArtifactType, ...] = Field(...)
    steps: tuple[StepSpec, ...] = Field(min_length=1)
    preferred_text_output: str | None = Field(
        default=None,
        max_length=256,
        description=(
            "Sprint S39 — référence ``step_id.output_type`` qui désigne "
            "quelle sortie de la pipeline est préférée pour les vues "
            "textuelles (utile quand plusieurs steps produisent du "
            "RAW_TEXT ou CORRECTED_TEXT). Format ``<step_id>.<artifact_type>`` "
            "(ex : ``corrector.corrected_text``). Optionnel — sans, les "
            "vues prennent la dernière sortie textuelle observée."
        ),
    )

    @model_validator(mode="after")
    def _validate_preferred_text_output(self) -> "PipelineSpecYaml":
        """Vérifie que ``preferred_text_output`` (si défini) référence
        un step existant dont les ``output_types`` contiennent le
        type cité."""
        ref = self.preferred_text_output
        if ref is None:
            return self
        if "." not in ref:
            raise ValueError(
                f"preferred_text_output {ref!r} : format attendu "
                "``step_id.output_type`` (ex : ``corrector.corrected_text``).",
            )
        step_id, _, output_type_value = ref.partition(".")
        if not step_id or not output_type_value:
            raise ValueError(
                f"preferred_text_output {ref!r} : step_id ou output_type vide.",
            )
        # Vérifier que le step existe.
        target_step = next(
            (s for s in self.steps if s.id == step_id), None,
        )
        if target_step is None:
            raise ValueError(
                f"preferred_text_output {ref!r} : step "
                f"{step_id!r} introuvable dans la pipeline "
                f"{self.name!r}.",
            )
        # Vérifier que le step produit bien ce type.
        try:
            output_enum = ArtifactType(output_type_value)
        except ValueError as exc:
            raise ValueError(
                f"preferred_text_output {ref!r} : "
                f"output_type {output_type_value!r} inconnu.",
            ) from exc
        if output_enum not in target_step.output_types:
            raise ValueError(
                f"preferred_text_output {ref!r} : step {step_id!r} "
                f"ne produit pas {output_type_value!r} "
                f"(produit : {[t.value for t in target_step.output_types]}).",
            )
        return self

    @model_validator(mode="after")
    def _validate_inputs_from(self) -> "PipelineSpecYaml":
        """Vérifie que chaque ``inputs_from[type] = ref`` désigne soit
        ``__initial__``, soit un step antérieur qui produit le type."""
        from picarones.domain.pipeline_spec import INITIAL_STEP_ID

        # Set des steps déjà vus pour vérifier l'antériorité.
        seen_step_ids: set[str] = set()
        # Map des outputs produits par chaque step (pour vérification
        # des types).
        outputs_by_step: dict[str, set[ArtifactType]] = {}

        for step in self.steps:
            for input_type, source in step.inputs_from.items():
                if source == INITIAL_STEP_ID:
                    if input_type not in self.initial_inputs:
                        raise ValueError(
                            f"step {step.id!r} : inputs_from[{input_type.value!r}] "
                            f"= {INITIAL_STEP_ID!r} mais ce type n'est pas dans "
                            f"initial_inputs (= {[t.value for t in self.initial_inputs]}).",
                        )
                    continue
                if source not in seen_step_ids:
                    raise ValueError(
                        f"step {step.id!r} : inputs_from[{input_type.value!r}] "
                        f"= {source!r} ne désigne pas une étape antérieure "
                        f"connue (déjà vues : {sorted(seen_step_ids)}).",
                    )
                if input_type not in outputs_by_step.get(source, set()):
                    raise ValueError(
                        f"step {step.id!r} : inputs_from[{input_type.value!r}] "
                        f"= {source!r} mais cette étape ne produit pas ce type.",
                    )
            seen_step_ids.add(step.id)
            outputs_by_step[step.id] = set(step.output_types)
        return self


class RunSpec(BaseModel):
    """Déclaration complète d'un run benchmark.

    Tous les chemins (``corpus_zip``, ``corpus_dir``, ``output_dir``,
    ``report_html``) sont relatifs au répertoire courant au moment de
    l'invocation CLI, ou absolus.  Pas de résolution magique
    (``$HOME``, env vars) — le caller passe ce qu'il veut voir.
    """

    model_config = ConfigDict(extra="forbid")

    corpus_zip: str | None = Field(default=None, max_length=2048)
    corpus_dir: str | None = Field(default=None, max_length=2048)
    corpus_name: str | None = Field(default=None, max_length=128)
    corpus_metadata: dict[str, str] = Field(default_factory=dict)

    pipelines: tuple[PipelineSpecYaml, ...] = Field(min_length=1)
    views: tuple[str, ...] = Field(min_length=1)

    output_dir: str = Field(min_length=1, max_length=2048)
    report_html: str | None = Field(default=None, max_length=2048)
    report_lang: str = Field(default="fr")
    code_version: str = Field(default="0.0.0-unset", max_length=128)

    # ──────────────────────────────────────────────────────────────────
    # migration Option B (run_benchmark_via_service →
    # RunOrchestrator).  Les 7 champs ci-dessous portent les
    # paramètres legacy de ``run_benchmark_via_service`` dans la
    # spec déclarative.  À ce stade (B1) ils sont validés mais pas
    # encore consommés par l'orchestrateur — les phases B2.1-B2.7
    # branchent chaque champ à son comportement.
    #
    # Les paramètres d'exécution **non-sérialisables**
    # (``progress_callback``, ``cancel_event``) restent kwargs de
    # ``RunOrchestrator.execute()`` — un YAML ne peut pas porter un
    # callable Python.
    # ──────────────────────────────────────────────────────────────────

    char_exclude: str | None = Field(
        default=None,
        max_length=512,
        description=(
            "Caractères à exclure du calcul CER/WER (Phase B2.5). "
            "Chaîne de caractères Unicode, traitée comme un set par "
            "``compute_metrics``.  Cas typique : ``'!?.,;:'`` pour "
            "ignorer la ponctuation."
        ),
    )

    normalization_profile: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Profil de normalisation texte appliqué avant CER/WER "
            "(Phase B2.5).  Valeurs canoniques : ``caseless``, "
            "``medieval_french``, ``sans_apostrophes``, etc.  Voir "
            "``picarones.formats.text.normalization``."
        ),
    )

    partial_dir: str | None = Field(
        default=None,
        max_length=2048,
        description=(
            "Répertoire où persister les ``DocumentResult`` intermédiaires "
            "pour la reprise sur interruption (Phase B2.3).  Format : "
            "JSONL par pipeline (``picarones_{corpus}_{pipeline}.partial.jsonl``)."
        ),
    )

    entity_extractor: str | None = Field(
        default=None,
        max_length=512,
        description=(
            "Dotted path Python vers une factory d'extracteur d'entités "
            "nommées (Phase B2.4).  Format accepté : ``module.submodule:"
            "Symbol`` ou ``module.submodule.Symbol``.  La factory doit "
            "retourner un callable ``(text: str) -> list[dict]`` compatible "
            "avec ``_attach_ner_metrics_to_benchmark``.  L'importabilité "
            "est vérifiée lazy à ``execute()``."
        ),
    )

    profile: str = Field(
        default="standard",
        max_length=64,
        description=(
            "Profil de hooks document-level / corpus aggregators "
            "(Phase B2.6).  Sélectionne quels hooks "
            "``@register_document_metric`` / ``@register_corpus_aggregator`` "
            "s'exécutent.  Profils canoniques : ``standard``, ``diagnostics``, "
            "``economics``, ``pipeline``, ``full``."
        ),
    )

    output_json: str | None = Field(
        default=None,
        max_length=2048,
        description=(
            "Chemin facultatif où sérialiser le ``BenchmarkResult`` "
            "legacy en JSON (Phase B2.7).  Cohabite avec les 4 fichiers "
            "JSONL natifs persistés sous ``output_dir/results/``."
        ),
    )

    timeout_seconds_per_doc: float = Field(
        default=60.0,
        gt=0.0,
        le=86400.0,
        description=(
            "Timeout par document propagé au ``CorpusRunner``.  "
            "Cohérent avec ``run_benchmark_via_service.timeout_seconds``."
        ),
    )

    @model_validator(mode="after")
    def _validate_corpus_source(self) -> "RunSpec":
        if (self.corpus_zip is None) == (self.corpus_dir is None):
            raise ValueError(
                "RunSpec : il faut renseigner exactement l'un de "
                "``corpus_zip`` ou ``corpus_dir`` (pas les deux, pas "
                "aucun).",
            )
        return self

    @model_validator(mode="after")
    def _validate_views_are_canonical(self) -> "RunSpec":
        unknown = [v for v in self.views if v not in CANONICAL_VIEW_NAMES]
        if unknown:
            raise ValueError(
                f"RunSpec : vue(s) inconnue(s) {unknown!r}.  "
                f"Seules les vues canoniques sont supportées par la "
                f"CLI : {sorted(CANONICAL_VIEW_NAMES)}.",
            )
        return self

    @model_validator(mode="after")
    def _validate_unique_pipeline_names(self) -> "RunSpec":
        names = [p.name for p in self.pipelines]
        if len(set(names)) != len(names):
            raise ValueError(
                f"RunSpec : noms de pipeline dupliqués dans {names!r}.",
            )
        return self

    @model_validator(mode="after")
    def _validate_profile_is_known(self) -> "RunSpec":
        """Phase B1.1 — rejet précoce des profils inconnus.

        Délégué à ``evaluation.metric_hooks.validate_profile``, le
        même validator que ``run_benchmark_via_service`` utilise au
        démarrage du bench legacy.
        """
        from picarones.evaluation.metric_hooks import validate_profile

        validate_profile(self.profile)
        return self

    @model_validator(mode="after")
    def _validate_entity_extractor_format(self) -> "RunSpec":
        """Phase B1.1 — valide le format dotted path.

        L'**importabilité** est vérifiée lazy à
        ``RunOrchestrator.execute()`` (cf. Phase B2.4), parce qu'un
        YAML peut être validé sur une machine où le package
        contenant l'extracteur n'est pas installé.
        """
        if self.entity_extractor is None:
            return self
        if not _DOTTED_PATH_RE.match(self.entity_extractor):
            raise ValueError(
                f"entity_extractor invalide : {self.entity_extractor!r}. "
                "Format attendu : ``module.submodule:Symbol`` ou "
                "``module.submodule.Symbol`` (composants alphanumériques "
                "+ ``_``).",
            )
        return self


# ──────────────────────────────────────────────────────────────────────
# Loader YAML + résolution dotted path
# ──────────────────────────────────────────────────────────────────────


class RunSpecLoadError(PicaronesError):
    """Échec de chargement / validation d'une spec YAML."""


def load_run_spec_from_yaml(yaml_text: str) -> RunSpec:
    """Parse + valide une chaîne YAML.

    Raises
    ------
    RunSpecLoadError
        Si le YAML est mal formé, si pydantic rejette le schéma, ou
        si une contrainte du model_validator échoue.
    """
    import yaml

    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise RunSpecLoadError(f"YAML mal formé : {exc}") from exc

    if data is None:
        raise RunSpecLoadError(
            "RunSpec : YAML vide (attendu un mapping racine).",
        )
    if not isinstance(data, dict):
        raise RunSpecLoadError(
            f"RunSpec : YAML racine doit être un mapping, reçu "
            f"{type(data).__name__}.",
        )

    try:
        return RunSpec.model_validate(data)
    except Exception as exc:  # noqa: BLE001 — re-typer en exception métier
        raise RunSpecLoadError(f"RunSpec invalide : {exc}") from exc


def resolve_adapter_class(dotted_path: str) -> type:
    """Importe et retourne la classe désignée par un dotted path.

    Format attendu : ``module.sub.ClassName``.  ``module.sub:ClassName``
    accepté aussi (séparateur ``:`` style entry-point).

    Raises
    ------
    RunSpecLoadError
        Si le module est introuvable, si l'attribut n'existe pas,
        ou si l'attribut n'est pas une classe instanciable.
    """
    if not dotted_path or "." not in dotted_path and ":" not in dotted_path:
        raise RunSpecLoadError(
            f"adapter_class invalide : {dotted_path!r} — attendu "
            f"``module.sub.ClassName`` ou ``module.sub:ClassName``.",
        )
    if ":" in dotted_path:
        module_path, _, class_name = dotted_path.rpartition(":")
    else:
        module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path or not class_name:
        raise RunSpecLoadError(
            f"adapter_class mal formé : {dotted_path!r}.",
        )

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise RunSpecLoadError(
            f"Module introuvable pour {dotted_path!r} : {exc}",
        ) from exc

    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise RunSpecLoadError(
            f"Attribut {class_name!r} absent du module "
            f"{module_path!r}.",
        ) from exc

    if not isinstance(cls, type):
        raise RunSpecLoadError(
            f"adapter_class {dotted_path!r} n'est pas une classe "
            f"(c'est un {type(cls).__name__}).",
        )
    return cls


__all__ = [
    "CANONICAL_VIEW_NAMES",
    "PipelineSpecYaml",
    "RunSpec",
    "RunSpecLoadError",
    "StepSpec",
    "load_run_spec_from_yaml",
    "resolve_adapter_class",
]
