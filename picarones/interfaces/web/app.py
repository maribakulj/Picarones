"""``create_app`` — Sprint A14-S35.

Squelette FastAPI du nouveau monde.  **Pas un shim** sur le legacy
``picarones.web.app`` — c'est une app neuve, écrite pour consommer
directement les services du Sprint S17+ (``BenchmarkService``,
``RegistryService``, ``RunOrchestrator``, ``WorkspaceManager``,
``CorpusService``).

Le legacy ``picarones.web.app`` reste en place jusqu'au S46.

Architecture
------------
- ``create_app(app_state) → FastAPI`` : factory qui construit l'app
  avec les services injectés. Pas de singleton global — chaque
  ``create_app`` produit une instance indépendante.
- ``WebAppState`` : container immuable des services injectés
  (services + workspace root + version).
- Endpoint ``GET /health`` : liveness probe pour Docker / k8s.
- Endpoint ``GET /version`` : version + flags (mode public, etc.).
- Endpoints corpus/benchmark/jobs : ajoutés aux S36-S37 via routers
  dédiés.

Anti-sur-ingénierie
-------------------
- Pas de middleware CSP/CSRF dans S35 — ajoutés au S38 quand on
  servira des templates HTML (le squelette S35 est API-only).
- Pas de lifespan (rien à initialiser au démarrage — les services
  sont injectés déjà construits).
- Pas de mount static (S38).
- Pas de jobs queue (S37).

Chaque sprint S36-S38 ajoute incrémentalement sans toucher au
squelette : on monte des routers, on attache des middlewares, on
mount des fichiers statiques.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from pydantic import BaseModel

from picarones.app.services import (
    BenchmarkService,
    CorpusService,
    RegistryService,
    RunOrchestrator,
    WorkspaceManager,
)


@dataclass(frozen=True)
class WebAppState:
    """Container immuable des services injectés dans l'app web.

    Attributes
    ----------
    workspace:
        ``WorkspaceManager`` du run en cours.
    registry:
        ``RegistryService`` (registres de métriques + projecteurs
        pré-bootstrap).
    corpus:
        ``CorpusService`` (import ZIP, détection patterns).
    benchmark:
        ``BenchmarkService`` (orchestration runner + vues +
        persistance).
    orchestrator:
        ``RunOrchestrator`` (workflow YAML → bench → HTML report).
    version:
        Version du code Picarones à afficher dans
        ``GET /version``.

    Notes
    -----
    Frozen : aucun service ne change de référence après le démarrage
    de l'app.  Pour reconstruire l'état (test isolé), créer une
    nouvelle ``WebAppState``.
    """

    workspace: WorkspaceManager
    registry: RegistryService
    corpus: CorpusService
    benchmark: BenchmarkService
    orchestrator: RunOrchestrator
    version: str = "1.0.0"


class HealthResponse(BaseModel):
    """Schéma JSON pour ``GET /health``."""

    status: str = "ok"


class VersionResponse(BaseModel):
    """Schéma JSON pour ``GET /version``."""

    version: str
    workspace_root: str
    n_metrics: int
    n_projectors: int


def create_app(state: WebAppState) -> FastAPI:
    """Construit une instance FastAPI consommant l'``WebAppState``.

    Pas de singleton global : chaque appel produit une nouvelle app
    indépendante.  Permet aux tests d'instancier des apps avec des
    services mockés sans interférence avec d'autres tests.

    Parameters
    ----------
    state:
        ``WebAppState`` immuable injectée dans tous les endpoints
        via ``Request.app.state.picarones``.

    Returns
    -------
    FastAPI
        Instance prête à être lancée par ``uvicorn`` ou consommée
        par ``TestClient``.
    """
    if not isinstance(state, WebAppState):
        raise TypeError(
            f"create_app : state doit être WebAppState, "
            f"reçu {type(state).__name__}.",
        )

    app = FastAPI(
        title="Picarones",
        description=(
            "Plateforme de benchmark OCR/HTR pour documents patrimoniaux. "
            "API du nouveau monde (Sprint A14-S35)."
        ),
        version=state.version,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # On stocke l'état dans app.state.picarones pour permettre aux
    # endpoints (S36+) d'y accéder via Request.app.state.picarones
    # — namespace explicite pour ne pas collisionner avec d'autres
    # extensions FastAPI.
    app.state.picarones = state

    # ──────────────────────────────────────────────────────────────
    # Routers métier (S36+)
    # ──────────────────────────────────────────────────────────────
    # Import paresseux pour éviter les cycles : `routers/__init__.py`
    # importe les routers individuels, qui n'ont pas besoin de
    # `WebAppState` au moment de leur définition (ils consomment via
    # `request.app.state.picarones` à chaque appel).
    from picarones.interfaces.web.routers import (
        benchmark_router,
        corpus_router,
    )
    app.include_router(corpus_router)
    app.include_router(benchmark_router)

    # ──────────────────────────────────────────────────────────────
    # Endpoints squelette (sondes santé/version)
    # ──────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness probe — toujours ``200 OK`` si l'app a démarré.

        Pas de dépendance aux services backends : on veut détecter
        un crash de l'app, pas un crash transitoire d'un service.
        """
        return HealthResponse(status="ok")

    @app.get("/version", response_model=VersionResponse)
    async def version() -> VersionResponse:
        """Affiche la version du code et un compte rapide des
        registres pour vérifier que le bootstrap a bien eu lieu."""
        return VersionResponse(
            version=state.version,
            workspace_root=str(state.workspace.root),
            n_metrics=len(state.registry.metrics),
            n_projectors=len(state.registry.projectors),
        )

    return app


__all__ = [
    "HealthResponse",
    "VersionResponse",
    "WebAppState",
    "create_app",
]
