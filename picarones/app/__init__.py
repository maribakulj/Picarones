"""Couche 6 — Application services.

Couche d'orchestration : reçoit des requêtes (DTO Pydantic) depuis
``interfaces/``, valide tout (chemins sandboxés, quotas, mode
public/dev), assemble adapters + pipeline + evaluation, retourne
des résultats sérialisables.

C'est ici que les **6 P0 du S1** trouvent leur foyer définitif au
S19 : ``WorkspaceManager`` qui isole les chemins par session,
``BenchmarkService`` qui orchestre run + projections + persistance,
``RegistryService`` qui construit les registres explicitement.

Sous-packages :

- ``services/`` — un service par domaine fonctionnel
  (BenchmarkService, CorpusService, ReportService, JobService,
  RegistryService, WorkspaceManager).
- ``schemas/`` — DTO Pydantic pour API et CLI.  **Séparés** des
  modèles de domaine pour éviter le couplage transport ↔ métier.

Règle d'import : peut importer domain/, evaluation/, pipeline/,
formats/, adapters/.  Ne doit **jamais** importer interfaces/.
"""

from __future__ import annotations

__all__: list[str] = []
