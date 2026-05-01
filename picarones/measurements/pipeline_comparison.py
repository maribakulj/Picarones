"""Comparaison de N pipelines sur le même corpus — Sprint 65 (axe B).

Sprint 65 — Étape 4 / axe B du plan d'évolution 2026 : suite directe
des Sprints 63-64.  Le runner mono-document (Sprint 63) et
l'orchestration corpus-wide (Sprint 64) permettent d'évaluer **une**
pipeline composée ; ce sprint répond à la question typique BnF :

    « OCR seul vs OCR+correcteur A vs OCR+correcteur B :
      laquelle est la meilleure sur mon corpus, et de combien ? »

Philosophie inchangée
---------------------
Picarones reste un **banc d'essai** — on juge des pipelines tierces
sur le **même corpus** avec la **même GT**, en exposant des chiffres
bruts comparatifs.  Aucun verdict imposé : le chercheur lit le
ranking et la table de gain et conclut selon ses critères.

Périmètre Sprint 65
-------------------
Inclus :

- ``compare_pipelines(specs, corpus, factories=None)`` qui exécute
  séquentiellement N pipelines sur le même corpus.
- ``PipelineComparisonResult`` : conteneur avec
  ``per_pipeline: dict[name → PipelineBenchmarkResult]``,
  ``ranking_by_final_metric(artifact_type, metric_name,
  higher_is_better)`` qui retourne ``[(pipeline_name, score), ...]``
  trié, et ``gain_table(artifact_type, metric_name,
  baseline_pipeline)`` qui retourne pour chaque pipeline le
  ``{absolute, relative}`` vs baseline.
- ``factories``: dict ``{pipeline_name: InitialInputsFactory}`` pour
  personnaliser les entrées initiales par pipeline (utile pour
  comparer une pipeline qui démarre par IMAGE et une qui démarre
  par TEXT).
- Garde-fou : noms de pipelines uniques exigés.

Reporté à des sprints suivants :

- DAG branchant non séquentiel (Sprint 66).
- Vue HTML dédiée à la comparaison de pipelines (Sprint 67+).
- Tests statistiques (Wilcoxon, Friedman, Nemenyi) sur les
  pipelines composées — déjà disponibles côté OCR (Sprint 18) ;
  l'application au cadre pipeline arrive plus tard.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from picarones.core.corpus import Corpus
from picarones.core.modules import ArtifactType
from picarones.measurements.pipeline_benchmark import (
    InitialInputsFactory,
    PipelineBenchmarkResult,
    default_initial_inputs,
    run_pipeline_benchmark,
)
from picarones.core.pipeline import PipelineSpec

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Conteneur de résultats
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class PipelineComparisonResult:
    """Résultat de la comparaison de N pipelines sur un corpus.

    Champs
    ------
    corpus_name:
        Nom du corpus (commun à toutes les pipelines comparées).
    n_docs:
        Nombre de documents du corpus.
    per_pipeline:
        Map ``{pipeline_name: PipelineBenchmarkResult}``.  L'ordre
        d'insertion suit l'ordre des ``specs`` passées à
        ``compare_pipelines`` ; on s'appuie sur le ``dict`` ordonné
        de Python 3.7+.
    total_duration_seconds:
        Durée totale de la comparaison (sommes des durées par
        pipeline + petit overhead).
    """

    corpus_name: str
    n_docs: int = 0
    per_pipeline: dict[str, PipelineBenchmarkResult] = field(
        default_factory=dict,
    )
    total_duration_seconds: float = 0.0

    def pipeline_names(self) -> list[str]:
        """Retourne la liste des noms de pipelines dans leur ordre
        d'insertion (= ordre de la comparaison initiale)."""
        return list(self.per_pipeline.keys())

    def _final_metric_value(
        self,
        pipeline_name: str,
        artifact_type: ArtifactType,
        metric_name: str,
    ) -> Optional[float]:
        """Retourne le ``mean`` de la métrique demandée à la
        **dernière étape** de la pipeline qui a produit
        ``artifact_type`` (avec succès sur ≥ 1 doc), ou ``None``
        si la métrique n'est pas disponible.

        Cohérent avec ``PipelineResult.junction_metrics_for`` du
        Sprint 63 mais au niveau corpus-wide.
        """
        bench = self.per_pipeline.get(pipeline_name)
        if bench is None:
            return None
        for agg in reversed(bench.per_step_aggregates):
            type_metrics = agg.junction_metrics.get(artifact_type.value)
            if not type_metrics:
                continue
            stats = type_metrics.get(metric_name)
            if stats is None:
                continue
            return stats["mean"]
        return None

    def ranking_by_final_metric(
        self,
        artifact_type: ArtifactType,
        metric_name: str,
        higher_is_better: bool = False,
    ) -> list[tuple[str, Optional[float]]]:
        """Classe les pipelines par la valeur **finale** de
        ``metric_name`` à la jonction ``artifact_type``.

        Returns
        -------
        list[tuple[str, Optional[float]]]
            Liste ``[(pipeline_name, mean_value)]`` triée :

            - Les pipelines avec une valeur définie viennent en
              premier, triées selon ``higher_is_better``.
            - Les pipelines sans valeur (métrique absente) viennent
              en queue, dans leur ordre d'insertion.
        """
        with_value: list[tuple[str, float]] = []
        without_value: list[tuple[str, Optional[float]]] = []
        for name in self.pipeline_names():
            value = self._final_metric_value(name, artifact_type, metric_name)
            if value is None:
                without_value.append((name, None))
            else:
                with_value.append((name, value))
        with_value.sort(
            key=lambda pair: pair[1],
            reverse=higher_is_better,
        )
        return [*with_value, *without_value]

    def gain_table(
        self,
        artifact_type: ArtifactType,
        metric_name: str,
        baseline_pipeline: str,
    ) -> dict[str, dict[str, Optional[float]]]:
        """Calcule l'écart de chaque pipeline vs la baseline.

        Returns
        -------
        dict
            Map ``{pipeline_name: {"value", "absolute", "relative"}}``
            où :

            - ``value`` : valeur finale de la métrique pour cette
              pipeline (``None`` si absente).
            - ``absolute`` : ``value - baseline_value``
              (``None`` si l'une des deux est absente).
            - ``relative`` : ``(value - baseline_value) /
              baseline_value`` (``None`` si baseline absente ou
              égale à 0).

        La baseline elle-même apparaît avec ``absolute == 0`` et
        ``relative == 0``.
        """
        if baseline_pipeline not in self.per_pipeline:
            raise KeyError(
                f"baseline {baseline_pipeline!r} absente de la comparaison",
            )
        baseline_value = self._final_metric_value(
            baseline_pipeline, artifact_type, metric_name,
        )
        out: dict[str, dict[str, Optional[float]]] = {}
        for name in self.pipeline_names():
            value = self._final_metric_value(
                name, artifact_type, metric_name,
            )
            absolute: Optional[float]
            relative: Optional[float]
            if value is None or baseline_value is None:
                absolute = None
                relative = None
            else:
                absolute = value - baseline_value
                relative = (
                    (value - baseline_value) / baseline_value
                    if baseline_value != 0 else None
                )
            out[name] = {
                "value": value,
                "absolute": absolute,
                "relative": relative,
            }
        return out


# ──────────────────────────────────────────────────────────────────────────
# Orchestrateur
# ──────────────────────────────────────────────────────────────────────────


def compare_pipelines(
    specs: list[PipelineSpec],
    corpus: Corpus,
    factories: Optional[dict[str, InitialInputsFactory]] = None,
) -> PipelineComparisonResult:
    """Exécute N ``PipelineSpec`` sur le **même** ``corpus``.

    Parameters
    ----------
    specs:
        Liste de ``PipelineSpec``.  Les noms de pipelines doivent
        être uniques (sinon ``ValueError``).
    corpus:
        Corpus partagé entre toutes les pipelines comparées —
        c'est le point fort du sprint : même corpus, même GT, on
        peut comparer apple-to-apple.
    factories:
        Optionnel.  Si fourni, dict ``{pipeline_name:
        InitialInputsFactory}`` pour personnaliser les entrées
        initiales par pipeline.  Les pipelines absentes du dict
        utilisent ``default_initial_inputs`` (cas standard
        ``IMAGE`` depuis ``Document.image_path``).

    Returns
    -------
    PipelineComparisonResult
        Conteneur avec ``per_pipeline`` indexé par nom et
        utilitaires comparatifs (``ranking_by_final_metric``,
        ``gain_table``).

    Raises
    ------
    ValueError
        Si deux ``PipelineSpec`` ont le même nom (impossible alors
        de les distinguer dans le résultat).
    """
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        seen: set[str] = set()
        duplicates: list[str] = []
        for n in names:
            if n in seen:
                duplicates.append(n)
            seen.add(n)
        raise ValueError(
            f"noms de pipelines non uniques : {sorted(set(duplicates))}",
        )

    factories = factories or {}
    result = PipelineComparisonResult(
        corpus_name=corpus.name,
        n_docs=len(list(corpus.documents)),
    )

    t0 = time.monotonic()
    for spec in specs:
        factory = factories.get(spec.name, default_initial_inputs)
        bench = run_pipeline_benchmark(spec, corpus, factory)
        result.per_pipeline[spec.name] = bench
    result.total_duration_seconds = time.monotonic() - t0
    return result


__all__ = [
    "PipelineComparisonResult",
    "compare_pipelines",
]
