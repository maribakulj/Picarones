"""DTO de transport pour API web et CLI — Sprint S19.

Schemas Pydantic strictement orientés "request/response".  Ils ne
remontent jamais à un service métier — ce sont les frontières
entre HTTP/CLI et la logique applicative.

Pattern : un endpoint reçoit un schema (validation Pydantic),
appelle un service avec les paramètres extraits + validés du
schema, retourne un autre schema.

Exemple cible :

.. code-block:: python

    # app/schemas/benchmark.py
    class StartRunRequest(BaseModel):
        corpus_path: str
        pipelines: list[PipelineSpecDTO]
        views: list[str]
        normalization_profile: NormalizationProfileId

    # interfaces/web/routers/benchmark.py
    @router.post("/api/runs")
    def start_run(req: StartRunRequest) -> StartRunResponse:
        run_id = benchmark_service.start_run(req.to_domain())
        return StartRunResponse(run_id=run_id)
"""

from __future__ import annotations

from picarones.app.schemas.run_spec import (
    CANONICAL_VIEW_NAMES,
    PipelineSpecYaml,
    RunSpec,
    RunSpecLoadError,
    StepSpec,
    load_run_spec_from_yaml,
    resolve_adapter_class,
)

__all__ = [
    "CANONICAL_VIEW_NAMES",
    "PipelineSpecYaml",
    "RunSpec",
    "RunSpecLoadError",
    "StepSpec",
    "load_run_spec_from_yaml",
    "resolve_adapter_class",
]
