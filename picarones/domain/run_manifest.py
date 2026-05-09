"""``RunManifest`` — empreinte immuable d'un run de benchmark.

Sprint A14-S17 du rewrite ciblé.

Le ``RunManifest`` est la **source de vérité** d'un run :

- **Quoi** a été exécuté (corpus + pipelines + vues).
- **Avec quelle version du code**.
- **Quand** (timestamp UTC de début et fin).
- **Quelles dépendances** étaient en place (snapshot du lock file).

Cette structure est sérialisée en ``run_manifest.json`` à la
racine du répertoire du run.  Combinée à ``view_results.jsonl``
et ``pipeline_results.jsonl``, elle permet à un caller (rapport
HTML, CLI ``picarones report``) de **reconstituer entièrement**
un run sans recourir à des objets Python live.

Garantie de reproductibilité
----------------------------
À ``code_version`` + ``corpus_name`` + ``pipeline_specs`` +
``view_specs`` + ``dependencies_lock`` identiques, ré-exécuter
doit donner les mêmes résultats (à la déterministe près des
adapters externes — un appel LLM cloud peut varier).

C'est ce qui permet à la BnF de citer un commit + un
``run_manifest.json`` dans une publication scientifique et à un
relecteur de re-vérifier.

Anti-sur-ingénierie
-------------------
- Pas de signature cryptographique du manifest pour S17.  Si la
  BnF veut une preuve d'intégrité, elle peut hasher le fichier et
  le citer (le contenu est byte-déterministe via
  ``model_dump_json(indent=2, sort_keys=True)``).
- Pas de versioning du schéma RunManifest.  Si le schéma évolue,
  on rebump pydantic — les anciens manifests pourront être
  interprétés via un convertisseur explicite, pas via un système
  de migration automatique.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from picarones.domain.evaluation_spec import EvaluationView
from picarones.domain.pipeline_spec import PipelineSpec


class RunManifest(BaseModel):
    """Empreinte immuable d'un run de benchmark.

    Tous les champs sont déterministes à entrée constante.
    ``started_at`` / ``completed_at`` capturent le wall-clock du
    run mais n'entrent pas dans les comparaisons de
    reproductibilité (deux runs identiques doivent donner les
    mêmes résultats même si exécutés à des moments différents).

    Attributs
    ---------
    run_id:
        Identifiant unique du run.  Convention :
        ``"<corpus_name>_<isoformat_compact>"`` (ex :
        ``"bnf_xviiie_20260503T144012Z"``).  Filesystem-safe.
    corpus_name:
        Nom du corpus traité (cf. ``CorpusSpec.name``).
    n_documents:
        Nombre de documents du corpus.
    pipeline_specs:
        Spécifications **complètes** des pipelines exécutées (steps,
        adapter_name par step, params, inputs_from, output_types).
        Inclus intégralement dans le manifest pour reproductibilité —
        un relecteur peut reconstituer le DAG sans accès au YAML
        d'origine.
    adapter_kwargs:
        Map ``{adapter_name: kwargs}`` capturée pour chaque adapter
        instancié.  Permet de reconstituer ``OpenAIAdapter(model=
        "gpt-4o-2024-08-06", temperature=0.0)`` à l'identique.
        Les valeurs sensibles (``api_key``) ne doivent pas y figurer
        — elles viennent toujours de variables d'environnement.
    view_specs:
        Vues d'évaluation appliquées.  Portées intégralement
        (frozen pydantic) parce qu'elles sont déclaratives et
        compactes.
    code_version:
        Version du code Picarones (typiquement
        ``picarones.__version__``).
    started_at, completed_at:
        Wall-clock UTC de début et fin du run.
    dependencies_lock:
        Snapshot ``{package: version}`` de l'environnement Python
        au moment du run.  Capturé via
        ``picarones.app.services.dependencies.capture_dependencies_lock``.
        Indispensable pour la promesse de reproductibilité — sans
        lui, un changement de version d'un parser XML ou d'une
        lib statistique fait diverger les résultats sans qu'on
        puisse l'attribuer.
    metadata:
        Dict libre pour notes utilisateur, etc.  Ne doit pas
        contenir d'info qui devrait être dans un autre champ.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1, max_length=256)
    corpus_name: str = Field(min_length=1, max_length=128)
    n_documents: int = Field(ge=0)
    pipeline_specs: tuple[PipelineSpec, ...] = Field(default_factory=tuple)
    adapter_kwargs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    view_specs: tuple[EvaluationView, ...] = Field(default_factory=tuple)
    code_version: str = Field(min_length=1, max_length=128)
    started_at: datetime
    completed_at: datetime
    dependencies_lock: dict[str, str] = Field(default_factory=dict)
    #: Sprint S8.5 — versions des binaires système critiques
    #: (Tesseract, etc.).  Capturé via
    #: ``picarones.app.services.dependencies.capture_system_binaries_lock``.
    #: Ferme le trou laissé par ``dependencies_lock`` qui ne couvre
    #: que les paquets Python — sans cette capture, deux runs avec
    #: le même ``dependencies_lock`` peuvent produire des CER
    #: différents si la version Tesseract change entre temps.
    #: Default empty dict (rétro-compat manifests pré-S8.5).
    system_binaries_lock: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pipeline_names(self) -> tuple[str, ...]:
        """Liste compacte des noms de pipelines (sérialisée dans le
        JSON pour les lecteurs qui ne traitent pas le DAG complet).

        Dérivée de ``pipeline_specs`` ; la liste authoritative pour
        la reproductibilité est ``pipeline_specs`` qui porte les DAG
        complets avec params et inputs_from.
        """
        return tuple(spec.name for spec in self.pipeline_specs)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_pipeline_names(
        cls,
        data: Any,
    ) -> Any:
        """Accepte ``pipeline_names`` au constructeur comme alias
        déprécié de ``pipeline_specs``.

        Trois cas :

        1. ``pipeline_names`` seul → convertit chaque nom en
           ``PipelineSpec(name=n, steps=())`` + ``DeprecationWarning``.
        2. ``pipeline_specs`` + ``pipeline_names`` cohérents → cas du
           round-trip JSON (``pipeline_names`` est un computed_field
           sérialisé) : on ignore silencieusement le doublon.
        3. ``pipeline_specs`` + ``pipeline_names`` incohérents →
           ``ValueError`` (incohérence sémantique).
        """
        if not isinstance(data, dict):
            return data
        if "pipeline_names" not in data:
            return data
        names = data["pipeline_names"]
        if "pipeline_specs" in data:
            specs = data["pipeline_specs"]
            spec_names = tuple(
                s.name if hasattr(s, "name") else s.get("name")
                for s in specs
            )
            if tuple(names) != spec_names:
                raise ValueError(
                    "RunManifest : ``pipeline_names`` et "
                    "``pipeline_specs`` désignent des pipelines "
                    f"distinctes (names={tuple(names)!r}, "
                    f"specs={spec_names!r}).",
                )
            # Round-trip JSON : computed_field re-sérialisé puis
            # re-parsé.  On ignore le doublon, ``pipeline_specs``
            # est authoritative.
            data = dict(data)
            data.pop("pipeline_names")
            return data
        warnings.warn(
            "RunManifest(pipeline_names=...) is deprecated and will "
            "be removed in 2.0.  Use pipeline_specs=tuple(PipelineSpec"
            "(name=n, steps=()) for n in names) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        data = dict(data)
        data.pop("pipeline_names")
        data["pipeline_specs"] = tuple(
            PipelineSpec(name=n, steps=()) for n in names
        )
        return data

    @property
    def duration_seconds(self) -> float:
        """Durée wall-clock du run en secondes."""
        delta = self.completed_at - self.started_at
        return delta.total_seconds()


def utcnow() -> datetime:
    """Helper pour timestamp UTC (utile pour les fixtures)."""
    return datetime.now(tz=timezone.utc)


__all__ = ["RunManifest", "utcnow"]
