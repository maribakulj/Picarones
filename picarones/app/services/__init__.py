"""Services applicatifs — couche ``app/`` du rewrite ciblé.

Un service = une responsabilité fonctionnelle, testable sans
démarrer FastAPI.

Services livrés
---------------
- ``benchmark_service.py`` (S17) — orchestre ``CorpusRunner`` +
  ``DefaultEvaluationViewExecutor`` + persistance JSONL.
- ``corpus_service.py`` (S20) — upload ZIP sandboxé + détection
  des paires image / GT (``.gt.alto.xml``, ``.gt.txt``, etc.).
- ``path_security.py`` (S19) — ``WorkspaceManager`` (sandbox
  par session) + helpers ``validated_path``, ``safe_report_name``,
  ``validated_prompt_filename``.
- ``registry_service.py`` (S23) — bootstrap explicite du
  ``MetricRegistry`` et du ``ProjectorRegistry`` au démarrage.
- ``report_service.py`` (S21) — rendu HTML autonome depuis un
  ``RunResult``.

Schemas (DTO de transport CLI/web) : voir ``picarones.app.schemas``.
Agrégats applicatifs (``RunResult``) : voir ``picarones.app.results``.
"""

from __future__ import annotations

from picarones.app.services.benchmark_service import (
    BenchmarkService,
    ContextFactory,
    GroundTruthFactory,
    PipelineInputsFactory,
)
from picarones.app.services.corpus_service import (
    CorpusImportError,
    CorpusImportReport,
    CorpusService,
)
from picarones.app.services.path_security import (
    PathValidationError,
    WorkspaceManager,
    safe_report_name,
    validated_path,
    validated_prompt_filename,
)
from picarones.app.services.registry_service import (
    RegistriesBundle,
    RegistryService,
    bootstrap_default_registries,
)
from picarones.app.services.run_orchestrator import (
    OrchestrationResult,
    RunOrchestrator,
)

# Le rendu HTML vit dans la couche ``reports_v2/`` (cible documentée
# du rewrite — un rapport est un format de sortie, pas un service).
# Un caller qui veut juste générer un HTML l'importe directement
# depuis là.

__all__ = [
    "BenchmarkService",
    "ContextFactory",
    "CorpusImportError",
    "CorpusImportReport",
    "CorpusService",
    "GroundTruthFactory",
    "OrchestrationResult",
    "PathValidationError",
    "PipelineInputsFactory",
    "RegistriesBundle",
    "RegistryService",
    "RunOrchestrator",
    "WorkspaceManager",
    "bootstrap_default_registries",
    "safe_report_name",
    "validated_path",
    "validated_prompt_filename",
]
