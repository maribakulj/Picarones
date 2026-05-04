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

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from picarones.domain.evaluation_spec import EvaluationView


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
    pipeline_names:
        Noms des pipelines exécutées (un par pipeline).  Ne porte
        PAS la spec complète pour rester compact dans le manifest
        — la spec YAML est citée par référence
        (``pipeline_specs_uri``).
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
        Snapshot des dépendances installées au moment du run
        (typiquement ``pip freeze`` ou ``poetry lock`` digéré).
        Format libre — un dict ``{package: version}`` est
        idiomatique mais pas imposé.
    metadata:
        Dict libre pour notes utilisateur, etc.  Ne doit pas
        contenir d'info qui devrait être dans un autre champ.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1, max_length=256)
    corpus_name: str = Field(min_length=1, max_length=128)
    n_documents: int = Field(ge=0)
    pipeline_names: tuple[str, ...] = Field(default_factory=tuple)
    view_specs: tuple[EvaluationView, ...] = Field(default_factory=tuple)
    code_version: str = Field(min_length=1, max_length=128)
    started_at: datetime
    completed_at: datetime
    dependencies_lock: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Durée wall-clock du run en secondes."""
        delta = self.completed_at - self.started_at
        return delta.total_seconds()


def utcnow() -> datetime:
    """Helper pour timestamp UTC (utile pour les fixtures)."""
    return datetime.now(tz=timezone.utc)


__all__ = ["RunManifest", "utcnow"]
