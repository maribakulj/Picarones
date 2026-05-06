"""``RunResult`` et ``RunDocumentResult`` — agrégats applicatifs d'un run.

Sprint A14-S17 (créé) / S26 (déplacé depuis ``domain/`` car
agrège des objets de ``evaluation/`` et ``pipeline/`` — la couche
``domain`` n'a pas le droit d'importer de ces couches plus
externes).

Structure
---------
Un ``RunResult`` est l'agrégat complet d'un run :

::

    RunResult
      ├── manifest: RunManifest
      └── document_results: tuple[RunDocumentResult, ...]
            ├── document_id: str
            ├── pipeline_results: tuple[PipelineResult, ...]
            │     (un par pipeline du run)
            └── view_results: tuple[ViewResult, ...]
                  (un par couple (vue, pipeline_éligible_à_la_vue))

Le ``RunResult`` est sérialisable JSON pour persistance
(typiquement éclaté en plusieurs fichiers : ``run_manifest.json``,
``pipeline_results.jsonl``, ``view_results.jsonl`` — cf.
``picarones.app.services.benchmark_service``).

Anti-sur-ingénierie
-------------------
Pas d'agrégation pré-calculée (rang par vue, moyennes par
pipeline, etc.) dans le ``RunResult`` lui-même — c'est de la
**présentation**, pas du domain.  Le rapport HTML (S22) calcule
ses agrégats à la volée depuis les ``ViewResult`` listés.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from picarones.domain.run_manifest import RunManifest
from picarones.evaluation.views.base import ViewResult
from picarones.pipeline.types import PipelineResult


class RunDocumentResult(BaseModel):
    """Tous les résultats d'un run pour un seul document.

    Agrège :
    - Les ``PipelineResult`` (un par pipeline exécutée).  Permet
      de reconstituer ce qui a été produit (artefacts, durées,
      erreurs).
    - Les ``ViewResult`` (un par couple ``(view, pipeline)`` où le
      pipeline a produit un artefact éligible à la vue).  Les
      pipelines OMIS d'une vue n'ont PAS de ``ViewResult`` pour
      cette vue (pattern d'omission explicite — cf. AltoView S15).

    Le caller (typiquement le rapport HTML) reconstruit les
    associations ``pipeline ↔ view_result`` via le champ
    ``ViewResult.candidate_artifact_id`` qui pointe vers
    ``Artifact.produced_by_step`` (lui-même corrélé au pipeline).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1, max_length=256)
    pipeline_results: tuple[PipelineResult, ...] = Field(default_factory=tuple)
    view_results: tuple[ViewResult, ...] = Field(default_factory=tuple)


class RunResult(BaseModel):
    """Agrégat complet d'un run de benchmark.

    Sérialisable JSON.  En pratique, persisté en plusieurs
    fichiers (cf. ``BenchmarkService.persist``) pour permettre
    une lecture sélective et un streaming jsonl.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: RunManifest
    document_results: tuple[RunDocumentResult, ...] = Field(default_factory=tuple)

    @property
    def n_documents(self) -> int:
        return len(self.document_results)

    def view_results_for(self, view_name: str) -> tuple[ViewResult, ...]:
        """Retourne tous les ``ViewResult`` du run pour une vue donnée.

        Utile pour l'agrégation par vue (rangs, moyennes) côté
        rapport HTML.  Préserve l'ordre d'apparition.
        """
        out: list[ViewResult] = []
        for doc in self.document_results:
            for vr in doc.view_results:
                if vr.view_name == view_name:
                    out.append(vr)
        return tuple(out)

    def pipeline_results_for(self, pipeline_name: str) -> tuple[PipelineResult, ...]:
        """Retourne tous les ``PipelineResult`` d'un pipeline donné."""
        out: list[PipelineResult] = []
        for doc in self.document_results:
            for pr in doc.pipeline_results:
                if pr.pipeline_name == pipeline_name:
                    out.append(pr)
        return tuple(out)


#: Type alias d'un renderer de rapport injecté par le caller.
#:
#: Signature canonique partagée par le ``RunOrchestrator`` (qui
#: l'invoque) et le ``JobRunner`` (qui le transmet).  Reçoit
#: ``(run_result, output_path, lang)``, écrit le fichier et retourne
#: le ``Path`` effectivement écrit (généralement identique à
#: ``output_path``, mais le renderer peut changer l'extension).
ReportRenderer = Callable[["RunResult", Path, str], Path]


__all__ = ["ReportRenderer", "RunDocumentResult", "RunResult"]
