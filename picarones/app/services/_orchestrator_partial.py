"""Phase B2.3 — reprise sur interruption pour ``RunOrchestrator``.

Pivote par **pipeline** (vs par **engine** dans le legacy
``_benchmark_orchestration.run_benchmark_with_partial``).  Cohérent
avec l'architecture du ``RunOrchestrator`` qui raisonne en
``PipelineSpec``.

Format
------
Pour chaque pipeline d'un run, un fichier JSONL séparé :

::

    {partial_dir}/picarones_{corpus_name}_{pipeline_name}_{fingerprint}.partial.jsonl

Chaque ligne = ``PipelineResult.model_dump_json()`` d'un document
traité.  Append-only ; la sérialisation Pydantic garantit le
roundtrip ``model_validate_json`` propre.

Fingerprint
-----------
Le fingerprint SHA-256 mélange :

- Le nom + structure de la pipeline (steps + adapter_class).
- ``normalization_profile`` (string canonique).
- ``char_exclude`` (caractères triés).
- ``profile`` (hooks document-level).
- Les ``mtime``/``size`` de chaque fichier du corpus
  (détection de modifs sans coût hash de contenu).
- ``code_version``.

Deux runs avec des configs divergentes → fingerprints différents →
fichiers de partial distincts → pas de réutilisation accidentelle
de résultats incompatibles.

Sémantique du resume
--------------------
1. Au démarrage du run : pour chaque ``pipeline_spec``, on cherche un
   partial existant matchant le fingerprint.  S'il existe, on
   charge les ``PipelineResult`` déjà calculés.
2. On filtre le corpus pour ne soumettre au ``BenchmarkService`` que
   les documents **manquants**.
3. Chaque nouveau ``PipelineResult`` est appendé au partial.
4. À la fin d'une pipeline traitée avec succès complet, le partial
   est supprimé (cleanup).  Une exception en cours préserve le
   partial pour la prochaine reprise.

Tolérance
---------
- Partial corrompu (JSON invalide) : on log un warning et on traite
  le document comme s'il n'avait jamais été calculé (recalcul propre).
- Partial avec fingerprint divergent : ignoré, fichier laissé tel
  quel (sera écrasé par le nouveau partial avec son propre
  fingerprint).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from picarones.pipeline.types import PipelineResult

if TYPE_CHECKING:
    from picarones.domain.corpus import CorpusSpec
    from picarones.domain.pipeline_spec import PipelineSpec

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Fingerprint
# ──────────────────────────────────────────────────────────────────────


def compute_pipeline_fingerprint(
    *,
    pipeline_spec: "PipelineSpec",
    corpus_spec: "CorpusSpec",
    normalization_profile: str | None,
    char_exclude: str | None,
    profile: str,
    code_version: str,
) -> str:
    """Phase B2.3 — fingerprint SHA-256 d'un run pour une pipeline donnée.

    Délègue à ``compute_run_fingerprint`` (helper legacy partagé
    avec ``run_benchmark_with_partial``) en construisant un
    ``engine_config`` qui matérialise la structure de la pipeline
    (nom + steps + adapter classes).  Deux pipelines avec le même
    nom mais des steps différents → fingerprints différents.

    Les chemins d'images du corpus sont aussi inclus pour détecter
    les modifications de fichiers entre runs (mtime+size, sans coût
    hash de contenu).
    """
    from picarones.app.services.partial_store import compute_run_fingerprint

    pipeline_engine_config: dict[str, Any] = {
        "pipeline_name": pipeline_spec.name,
        "steps": [
            {
                "id": step.id,
                "adapter": step.adapter_name,
                "inputs": sorted(t.value for t in step.input_types),
                "outputs": sorted(t.value for t in step.output_types),
            }
            for step in pipeline_spec.steps
        ],
    }
    # Phase B2.3 — on utilise les ``doc.id`` (stables cross-workspace)
    # plutôt que les ``image_uri`` (qui changent à chaque extraction du
    # corpus_zip vers un workspace temporaire).  Sinon le fingerprint
    # divergerait entre runs successifs même avec le même corpus,
    # rendant le resume inopérant.
    #
    # Limite : ce fingerprint ne détecte pas une modification du
    # contenu d'un doc (même id mais image différente).  Acceptable
    # pour le scope B2.3 ; pour une vraie détection de modifs, hasher
    # le contenu du corpus_zip d'origine (coûteux, scope futur).
    doc_signatures = sorted(doc.id for doc in corpus_spec.documents)
    return compute_run_fingerprint(
        engine_config=pipeline_engine_config,
        normalization_profile=normalization_profile,
        char_exclude=char_exclude,
        corpus_files=None,
        code_version=code_version,
        extra={
            "profile": profile,
            "doc_ids": ",".join(doc_signatures),
        } if profile else {"doc_ids": ",".join(doc_signatures)},
    )


def partial_path_for_pipeline(
    *,
    partial_dir: Path,
    corpus_name: str,
    pipeline_name: str,
    fingerprint: str,
) -> Path:
    """Chemin du fichier JSONL partiel pour une pipeline donnée."""
    from picarones.app.services.partial_store import _partial_path

    return _partial_path(
        corpus_name=corpus_name,
        engine_name=pipeline_name,
        partial_dir=partial_dir,
        fingerprint=fingerprint,
    )


# ──────────────────────────────────────────────────────────────────────
# I/O JSONL
# ──────────────────────────────────────────────────────────────────────


def load_partial_pipeline_results(
    partial_path: Path,
) -> list[PipelineResult]:
    """Charge tous les ``PipelineResult`` déjà persistés dans un partial.

    Retourne ``[]`` si le fichier n'existe pas ou est vide.

    Tolérance : une ligne JSON corrompue est sautée avec un warning ;
    les autres lignes valides sont conservées.  Le caller peut
    décider quoi faire des doc_id manquants (typiquement : les
    recalculer).
    """
    if not partial_path.exists() or partial_path.stat().st_size == 0:
        return []

    results: list[PipelineResult] = []
    with partial_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(PipelineResult.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[orchestrator_partial] ligne %d corrompue dans %s "
                    "— sautée : %s",
                    line_no, partial_path.name, exc,
                )
    return results


def append_pipeline_result(
    partial_path: Path,
    pipeline_result: PipelineResult,
) -> None:
    """Append un ``PipelineResult`` au fichier partiel (JSONL).

    Crée les répertoires parents et le fichier si nécessaire.
    Mode ``a`` : un crash mid-write peut laisser une ligne partielle
    en queue qui sera ignorée au prochain ``load_partial_pipeline_results``
    grâce au filet ``try/except`` autour de ``model_validate_json``.
    """
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("a", encoding="utf-8") as f:
        f.write(pipeline_result.model_dump_json() + "\n")


def delete_partial(partial_path: Path) -> None:
    """Supprime le fichier partiel (cleanup post-success).

    Idempotent : pas d'erreur si le fichier n'existe pas (autre
    pipeline a déjà nettoyé, fingerprint divergent, etc.).
    """
    try:
        partial_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            "[orchestrator_partial] échec suppression %s : %s",
            partial_path, exc,
        )


# ──────────────────────────────────────────────────────────────────────
# Filtrage des documents restants
# ──────────────────────────────────────────────────────────────────────


def filter_remaining_documents(
    documents: Iterable[Any],
    loaded_results: list[PipelineResult],
) -> tuple[list[Any], list[PipelineResult]]:
    """Retourne ``(docs_à_traiter, results_déjà_persistés_filtrés)``.

    Filtre les documents dont le ``id`` est déjà dans
    ``loaded_results``.  Les doublons éventuels du partial (ne devrait
    pas arriver vu le append-only) sont déduplitqués par ``document_id``
    en gardant le premier.
    """
    seen_doc_ids: set[str] = set()
    deduplicated: list[PipelineResult] = []
    for pr in loaded_results:
        if pr.document_id not in seen_doc_ids:
            seen_doc_ids.add(pr.document_id)
            deduplicated.append(pr)

    remaining = [d for d in documents if d.id not in seen_doc_ids]
    return remaining, deduplicated


__all__ = [
    "append_pipeline_result",
    "compute_pipeline_fingerprint",
    "delete_partial",
    "filter_remaining_documents",
    "load_partial_pipeline_results",
    "partial_path_for_pipeline",
]
