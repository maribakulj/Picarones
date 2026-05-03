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

__all__: list[str] = []
