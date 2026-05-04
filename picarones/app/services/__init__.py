"""Services applicatifs — Sprint S19.

Un service = une responsabilité fonctionnelle, testable sans
démarrer FastAPI.

Cibles :

- ``benchmark_service.py`` — ``BenchmarkService.start_run(spec)``,
  ``cancel_run(run_id)``, ``get_status(run_id)``.
- ``corpus_service.py`` — upload ZIP sandboxé, analyse de structure
  (pairs image/GT, détection des patterns ``.gt.alto.xml`` etc.).
- ``path_security.py`` — ``WorkspaceManager`` qui crée un dossier
  isolé par session et garantit que toute écriture/lecture y reste
  confinée.  Foyer définitif des helpers ``validated_path``,
  ``safe_report_name``, ``validated_prompt_filename`` du S1.
- ``registry_service.py`` — construit explicitement le
  ``MetricRegistry`` au démarrage (remplace l'import par effet de
  bord ``import picarones.measurements as _trigger``).
- ``report_service.py`` — produit le rapport HTML depuis un
  ``RunResult`` persisté.
- ``job_service.py`` — orchestration des jobs en arrière-plan
  (queue, workers, persistance).
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
from picarones.app.services.report_service import ReportService
from picarones.app.services.run_spec import (
    CANONICAL_VIEW_NAMES,
    PipelineSpecYaml,
    RunSpec,
    RunSpecLoadError,
    StepSpec,
    load_run_spec_from_yaml,
    resolve_adapter_class,
)

__all__ = [
    "BenchmarkService",
    "CANONICAL_VIEW_NAMES",
    "ContextFactory",
    "CorpusImportError",
    "CorpusImportReport",
    "CorpusService",
    "GroundTruthFactory",
    "PathValidationError",
    "PipelineInputsFactory",
    "PipelineSpecYaml",
    "RegistriesBundle",
    "RegistryService",
    "ReportService",
    "RunSpec",
    "RunSpecLoadError",
    "StepSpec",
    "WorkspaceManager",
    "bootstrap_default_registries",
    "load_run_spec_from_yaml",
    "resolve_adapter_class",
    "safe_report_name",
    "validated_path",
    "validated_prompt_filename",
]
