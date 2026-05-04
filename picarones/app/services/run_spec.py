"""``RunSpec`` — déclaration YAML d'un run benchmark.

Sprint A14-S24 du rewrite ciblé.

Format minimal qui décrit un run complet en YAML : corpus, pipelines
hétérogènes, vues canoniques à appliquer, sortie HTML.  Permet à
l'utilisateur BnF de lancer un benchmark via la CLI sans écrire de
Python.

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
      - name: tesseract_only
        initial_inputs: [image]
        steps:
          - id: ocr
            adapter_class: my_pkg.adapters.TesseractAdapter
            adapter_kwargs: {lang: fra}
            input_types: [image]
            output_types: [raw_text]

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
  caller qui veut des vues custom passe par l'API Python directe
  (la CLI MVP reste sur les 3 canoniques).
- ``adapter_class`` est un dotted path Python.  La classe doit être
  importable au moment du run (l'utilisateur installe ses propres
  packages dans le venv courant).
- ``adapter_kwargs`` est passé tel quel au constructeur.

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
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from picarones.domain.artifacts import ArtifactType


#: Vues canoniques supportées par la CLI MVP.
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


class PipelineSpecYaml(BaseModel):
    """Description d'une pipeline dans la spec YAML."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    initial_inputs: tuple[ArtifactType, ...] = Field(...)
    steps: tuple[StepSpec, ...] = Field(min_length=1)


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
                f"CLI MVP : {sorted(CANONICAL_VIEW_NAMES)}.",
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


# ──────────────────────────────────────────────────────────────────────
# Loader YAML + résolution dotted path
# ──────────────────────────────────────────────────────────────────────


class RunSpecLoadError(Exception):
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
