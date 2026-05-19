"""Helpers stateless du ``RunOrchestrator`` (sous-package cohésif).

Audit prod P1.1 — l'ancien module plat ``run_orchestrator_helpers.py``
était un signal de dette (« poubelle propre » fourre-tout).  Éclaté
en sous-modules cohésifs :

- :mod:`.factories`  — GT / inputs / RunContext (stateless)
- :mod:`.loaders`    — payload filesystem + signature kwargs
- :mod:`.builders`   — corpus / pipelines / vues / BenchmarkService
  (ex-``@staticmethod`` du god-module, Phase A)
- :mod:`.legacy`     — pont converter ``BenchmarkResult`` + résolution
  NER + persistance JSON legacy

Ce ``__init__`` ré-exporte l'API : ``from
picarones.app.services.run_orchestrator_helpers import
_default_gt_factory`` reste valide, et ``run_orchestrator`` réimporte
ces noms dans son namespace (donc ``monkeypatch.setattr(
run_orchestrator, …)`` continue de fonctionner).
"""

from picarones.app.services.run_orchestrator_helpers.factories import (
    _default_gt_factory as _default_gt_factory,
    _default_inputs_factory as _default_inputs_factory,
    _make_context_factory as _make_context_factory,
)
from picarones.app.services.run_orchestrator_helpers.loaders import (
    _filesystem_payload_loader as _filesystem_payload_loader,
    _kwargs_signature as _kwargs_signature,
)
from picarones.app.services.run_orchestrator_helpers.builders import (
    _build_benchmark_service as _build_benchmark_service,
    _build_pipelines as _build_pipelines,
    _build_views as _build_views,
    _load_corpus as _load_corpus,
)
from picarones.app.services.run_orchestrator_helpers.legacy import (
    _PipelineEngineProxy as _PipelineEngineProxy,
    _persist_legacy_benchmark_json as _persist_legacy_benchmark_json,
    _resolve_entity_extractor as _resolve_entity_extractor,
)

__all__ = [
    "_PipelineEngineProxy",
    "_build_benchmark_service",
    "_build_pipelines",
    "_build_views",
    "_default_gt_factory",
    "_default_inputs_factory",
    "_filesystem_payload_loader",
    "_kwargs_signature",
    "_load_corpus",
    "_make_context_factory",
    "_persist_legacy_benchmark_json",
    "_resolve_entity_extractor",
]
