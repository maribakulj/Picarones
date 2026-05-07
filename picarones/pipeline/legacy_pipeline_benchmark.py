"""Orchestration corpus-wide d'une pipeline composée — Sprint 64
(axe B).

Phase 5.C.batch7 — module relocalisé depuis
``picarones.measurements.pipeline_benchmark`` vers
``picarones.evaluation.pipeline_benchmark``.  Le chemin legacy
reste disponible via un shim avec ``DeprecationWarning`` ;
suppression prévue en 2.0.

Sprint 64 — Étape 4 / axe B du plan d'évolution 2026 : suite directe
du Sprint 63.  Le ``PipelineRunner`` exécute une pipeline sur **un**
document ; ce module fournit l'orchestration sur un **corpus
complet** et l'agrégation des résultats par étape.

Philosophie inchangée
---------------------
Picarones reste un **banc d'essai**.  Aucun module métier n'est
fourni — l'utilisateur amène ses propres ``BaseModule`` (Sprint 33).
Cette infrastructure se contente d'orchestrer leur exécution sur un
corpus, de mesurer le temps, de capturer les erreurs gracieusement,
et d'agréger les métriques calculées aux jonctions GT-vs-sortie.

Périmètre Sprint 64
-------------------
Inclus :

- ``run_pipeline_benchmark(spec, corpus, initial_inputs_factory)``
  qui itère séquentiellement sur les documents.
- Agrégation par étape : ``StepAggregate`` avec n_succeeded /
  n_failed, durées (total / mean / median), failing_doc_ids,
  métriques agrégées par type d'artefact (mean / median sur les
  métriques numériques uniquement), breakdown des types d'erreur.
- ``PipelineBenchmarkResult`` : conteneur global avec liste des
  ``PipelineResult`` par doc + liste des ``StepAggregate``.
- Helper ``default_initial_inputs`` qui couvre le cas standard
  ``IMAGE`` depuis ``Document.image_path``.

Reporté à des sprints suivants :

- Comparaison de N pipelines sur le même corpus (Sprint 65).
- DAG branchant non séquentiel (Sprint 66).
- Vue HTML dédiée aux pipelines composées (Sprint 67).
- Parallélisation inter-documents (à arbitrer selon les besoins).
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from picarones.evaluation.corpus import Corpus, Document
from picarones.domain.artifacts import ArtifactType
from picarones.pipeline.legacy_runner import (
    PipelineResult,
    PipelineRunner,
    PipelineSpec,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Helpers : factory d'entrées initiales
# ──────────────────────────────────────────────────────────────────────────

InitialInputsFactory = Callable[[Document], dict[ArtifactType, Any]]


def default_initial_inputs(document: Document) -> dict[ArtifactType, Any]:
    """Factory d'entrées initiales par défaut : couvre le cas
    « la pipeline démarre par un module qui consomme l'image ».

    Retourne ``{ArtifactType.IMAGE: document.image_path}`` si
    ``image_path`` est présent, sinon dict vide (la première étape
    devra alors signaler « entrée manquante »).
    """
    if document.image_path:
        return {ArtifactType.IMAGE: document.image_path}
    return {}


# ──────────────────────────────────────────────────────────────────────────
# Agrégats
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class StepAggregate:
    """Agrégat des résultats d'une étape sur tout le corpus.

    Champs
    ------
    step_name:
        Nom de l'étape (cf. ``PipelineStep.name``).
    n_docs:
        Nombre de documents pour lesquels l'étape a été tentée.
    n_succeeded:
        Nombre de documents pour lesquels l'étape s'est terminée
        sans erreur (``StepResult.error is None``).
    n_failed:
        Nombre de documents pour lesquels l'étape a renvoyé une
        erreur.
    duration_seconds_total / mean / median:
        Statistiques de durée sur les **étapes ayant réussi**
        uniquement (les étapes en erreur peuvent avoir une durée
        artificielle).
    failing_doc_ids:
        Liste des ``doc_id`` pour lesquels cette étape a échoué.
    junction_metrics:
        ``{artifact_type_value: {metric_name: {"mean": float,
        "median": float, "n": int}}}`` — agrégé sur les documents
        où la métrique a été calculée (n peut différer de
        ``n_succeeded`` si la GT du type n'est pas portée par tous
        les docs).
    error_breakdown:
        ``{type_d_erreur: count}`` où ``type_d_erreur`` est extrait
        en heuristique depuis le message (``"missing_input"``,
        ``"raised_exception"``, ``"missing_output"``,
        ``"other"``).
    """

    step_name: str
    n_docs: int = 0
    n_succeeded: int = 0
    n_failed: int = 0
    duration_seconds_total: float = 0.0
    duration_seconds_mean: float = 0.0
    duration_seconds_median: float = 0.0
    failing_doc_ids: list[str] = field(default_factory=list)
    junction_metrics: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=dict,
    )
    error_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.n_docs == 0:
            return 0.0
        return self.n_succeeded / self.n_docs


@dataclass
class PipelineBenchmarkResult:
    """Résultat d'un benchmark de pipeline sur un corpus complet.

    On capture la durée totale, les résultats par document
    (utiles pour le rapport HTML par-doc des sprints suivants), et
    l'agrégation par étape.
    """

    pipeline_name: str
    corpus_name: str
    n_docs: int = 0
    per_doc_results: list[PipelineResult] = field(default_factory=list)
    per_step_aggregates: list[StepAggregate] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    @property
    def n_pipelines_succeeded(self) -> int:
        return sum(1 for r in self.per_doc_results if r.succeeded)

    @property
    def n_pipelines_failed(self) -> int:
        return sum(1 for r in self.per_doc_results if not r.succeeded)

    def aggregate_for_step(self, step_name: str) -> Optional[StepAggregate]:
        for agg in self.per_step_aggregates:
            if agg.step_name == step_name:
                return agg
        return None


# ──────────────────────────────────────────────────────────────────────────
# Classification des erreurs
# ──────────────────────────────────────────────────────────────────────────


_ERROR_PATTERNS: tuple[tuple[str, str], ...] = (
    ("entrée manquante",  "missing_input"),
    ("sortie manquante",  "missing_output"),
    ("Error",             "raised_exception"),  # RuntimeError, ValueError…
)


def _classify_error(message: str) -> str:
    """Heuristique simple pour catégoriser une erreur d'étape.

    On regarde des marqueurs lexicaux dans le message (les messages
    sont produits par ``pipeline_runner._run_step`` qui les contrôle
    entièrement, donc cette heuristique est stable).
    """
    if not message:
        return "other"
    for pattern, label in _ERROR_PATTERNS:
        if pattern in message:
            return label
    return "other"


# ──────────────────────────────────────────────────────────────────────────
# Agrégation
# ──────────────────────────────────────────────────────────────────────────


def _aggregate_step(
    step_name: str, per_doc: list[tuple[str, Any]],
) -> StepAggregate:
    """Construit le ``StepAggregate`` pour une étape donnée.

    ``per_doc`` est une liste de tuples ``(doc_id, step_result)`` où
    ``step_result`` peut être ``None`` (cas où la pipeline a été
    arrêtée en amont avant cette étape) ou un ``StepResult``.
    """
    agg = StepAggregate(step_name=step_name)
    durations_succeeded: list[float] = []
    metrics_by_type: dict[str, dict[str, list[float]]] = {}

    for doc_id, sr in per_doc:
        if sr is None:
            # L'étape n'a même pas été exécutée (validation amont
            # invalide, ou exécutée n'a pas atteint l'index — ne se
            # produit pas en séquentiel mais peut arriver avec un
            # DAG plus tard).  On compte ce cas comme échec
            # explicite avec un type dédié.
            agg.n_docs += 1
            agg.n_failed += 1
            agg.failing_doc_ids.append(doc_id)
            agg.error_breakdown["pipeline_aborted"] = (
                agg.error_breakdown.get("pipeline_aborted", 0) + 1
            )
            continue
        agg.n_docs += 1
        if sr.error is None:
            agg.n_succeeded += 1
            durations_succeeded.append(sr.duration_seconds)
            # Collecte des métriques pour agrégation moyenne/médiane
            for at_value, metrics in sr.junction_metrics.items():
                slot = metrics_by_type.setdefault(at_value, {})
                for mname, mvalue in metrics.items():
                    if isinstance(mvalue, (int, float)) and not isinstance(
                        mvalue, bool,
                    ):
                        slot.setdefault(mname, []).append(float(mvalue))
        else:
            agg.n_failed += 1
            agg.failing_doc_ids.append(doc_id)
            label = _classify_error(sr.error)
            agg.error_breakdown[label] = (
                agg.error_breakdown.get(label, 0) + 1
            )

    if durations_succeeded:
        agg.duration_seconds_total = sum(durations_succeeded)
        agg.duration_seconds_mean = statistics.fmean(durations_succeeded)
        agg.duration_seconds_median = statistics.median(durations_succeeded)

    for at_value, metrics in metrics_by_type.items():
        agg.junction_metrics[at_value] = {
            mname: {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "n": len(values),
            }
            for mname, values in metrics.items()
        }
    # Phase 4-bis : double-clé legacy/canonique pour rétrocompat.
    from picarones.domain.artifacts import expand_legacy_keys
    expand_legacy_keys(agg.junction_metrics)
    return agg


# ──────────────────────────────────────────────────────────────────────────
# Orchestrateur principal
# ──────────────────────────────────────────────────────────────────────────


def run_pipeline_benchmark(
    spec: PipelineSpec,
    corpus: Corpus,
    initial_inputs_factory: InitialInputsFactory = default_initial_inputs,
) -> PipelineBenchmarkResult:
    """Exécute ``spec`` sur tous les documents de ``corpus``.

    Parameters
    ----------
    spec:
        Spécification de la pipeline composée.  Toutes les étapes
        sont des ``BaseModule`` fournis par l'utilisateur.
    corpus:
        Corpus chargé via ``Corpus.from_directory`` ou équivalent.
    initial_inputs_factory:
        Fonction qui produit, pour chaque document, les artefacts
        d'entrée de la pipeline.  Par défaut : ``IMAGE`` depuis
        ``document.image_path``.  L'utilisateur peut fournir une
        factory personnalisée pour brancher d'autres sources
        (par exemple ``ALTO`` pré-existant pour évaluer un
        pipeline qui démarre par un re-segmenteur).

    Returns
    -------
    PipelineBenchmarkResult
        Résultat global avec ``per_doc_results``,
        ``per_step_aggregates``, durée totale.

    Comportement
    ------------
    L'orchestration est **séquentielle** par document.  Pour chaque
    document, ``PipelineRunner.run`` est appelé ; quel que soit le
    résultat (réussi, partiellement échoué, totalement invalide),
    le résultat est ajouté à ``per_doc_results`` et le benchmark
    continue avec le document suivant.

    Si la spec est statiquement invalide (cf.
    ``PipelineSpec.validate``), tous les documents auront un
    ``PipelineResult.error`` non vide et aucune étape ne sera
    exécutée — le résultat reste cohérent.
    """
    result = PipelineBenchmarkResult(
        pipeline_name=spec.name, corpus_name=corpus.name,
    )
    documents = list(corpus.documents)
    result.n_docs = len(documents)

    benchmark_t0 = time.monotonic()
    for doc in documents:
        try:
            initial = initial_inputs_factory(doc)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[pipeline_benchmark] factory a levé sur %s : %s",
                doc.doc_id, exc,
            )
            # On crée un PipelineResult portant l'erreur factory
            failed = PipelineResult(
                pipeline_name=spec.name, doc_id=doc.doc_id,
                error=f"initial_inputs_factory: {type(exc).__name__}: {exc}",
            )
            result.per_doc_results.append(failed)
            continue
        per_doc = PipelineRunner.run(spec, doc, initial)
        result.per_doc_results.append(per_doc)
    result.total_duration_seconds = time.monotonic() - benchmark_t0

    # Agrégation par étape
    step_names = [step.name for step in spec.steps]
    for idx, step_name in enumerate(step_names):
        per_doc_step: list[tuple[str, Any]] = []
        for pr in result.per_doc_results:
            if idx < len(pr.steps):
                per_doc_step.append((pr.doc_id, pr.steps[idx]))
            else:
                # Pipeline a été arrêtée en amont : aucune étape de
                # cet index n'existe.  On compte ça comme une
                # absence d'étape (cf. ``_aggregate_step`` qui gère
                # le ``None``).
                per_doc_step.append((pr.doc_id, None))
        result.per_step_aggregates.append(
            _aggregate_step(step_name, per_doc_step),
        )

    return result


__all__ = [
    "InitialInputsFactory",
    "PipelineBenchmarkResult",
    "StepAggregate",
    "default_initial_inputs",
    "run_pipeline_benchmark",
]
