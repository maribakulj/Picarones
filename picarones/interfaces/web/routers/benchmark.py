"""Router benchmark — Sprint A14-S36.

Endpoints de listing/lecture des runs persistés dans le workspace.
Le **lancement** d'un run (asynchrone) est dans le router ``jobs``
au S37 — ici, on lit uniquement les manifests d'archive.

Convention de stockage
----------------------
``<workspace.root>/runs/<run_id>/`` contient :

- ``run_manifest.json`` (métadonnées du run)
- ``pipeline_results.jsonl``
- ``view_results.jsonl``

(cf. ``BenchmarkService.persist`` au S17.)

Endpoints
---------
- ``GET /api/runs`` : liste des run_ids disponibles avec leur
  manifest (corpus, pipeline_names, n_documents, started_at,
  completed_at).
- ``GET /api/runs/{run_id}`` : retourne le manifest complet d'un
  run.

Anti-sur-ingénierie
-------------------
- Pas de pagination — un workspace utilisateur a typiquement < 100
  runs.  Si un caller en a besoin, on l'ajoutera.
- Pas de delete — un caller peut supprimer le sous-dossier
  manuellement.
- Pas de search/filter par corpus_name — facile à ajouter mais on
  attend qu'un caller le demande.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/runs", tags=["benchmark"])

#: Sous-dossier sous ``WorkspaceManager.root`` où les runs sont
#: persistés.  Convention partagée avec ``BenchmarkService.persist``
#: lorsque le caller ne précise pas de répertoire.  Pour l'instant,
#: le caller peut tout aussi bien persister ailleurs — l'API web
#: regarde uniquement ici.  Au S37, ``RunOrchestrator`` garantira
#: cette convention.
RUNS_SUBDIR = "runs"


# ──────────────────────────────────────────────────────────────────────
# Schémas de réponse
# ──────────────────────────────────────────────────────────────────────


class RunSummary(BaseModel):
    """Résumé d'un run pour la liste."""

    run_id: str
    corpus_name: str | None = None
    n_documents: int | None = None
    pipeline_names: list[str] = Field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None


class RunListResponse(BaseModel):
    """Réponse JSON pour ``GET /api/runs``."""

    runs: list[RunSummary]


class RunManifestResponse(BaseModel):
    """Réponse JSON pour ``GET /api/runs/{run_id}``.

    Manifest complet — ``raw`` est le contenu JSON exact du
    ``run_manifest.json`` persisté.  L'utilisateur web peut faire
    son propre rendu sans qu'on impose une représentation.
    """

    run_id: str
    raw: dict


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _runs_dir(state) -> Path:
    """Retourne le dossier des runs sous le workspace de l'état."""
    return Path(state.workspace.root) / RUNS_SUBDIR


def _read_manifest(manifest_path: Path) -> dict | None:
    """Lit un ``run_manifest.json`` et retourne le dict ; ``None`` en
    cas d'échec (warning loggé)."""
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[benchmark] échec de lecture du manifest %s : %s",
            manifest_path, exc,
        )
        return None


def _summarize(manifest: dict, run_id: str) -> RunSummary:
    """Construit un ``RunSummary`` à partir d'un manifest."""
    return RunSummary(
        run_id=run_id,
        corpus_name=manifest.get("corpus_name"),
        n_documents=manifest.get("n_documents"),
        pipeline_names=list(manifest.get("pipeline_names", [])),
        started_at=manifest.get("started_at"),
        completed_at=manifest.get("completed_at"),
    )


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────


@router.get("", response_model=RunListResponse)
async def list_runs(request: Request) -> RunListResponse:
    """Liste les runs persistés dans le workspace.

    Scan le sous-dossier ``runs/`` du workspace et lit chaque
    ``run_manifest.json``.  Les manifests illisibles (corruption,
    permission) sont loggés en warning et omis du résultat.
    """
    state = request.app.state.picarones
    runs_dir = _runs_dir(state)
    if not runs_dir.exists():
        return RunListResponse(runs=[])

    summaries: list[RunSummary] = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        manifest_path = entry / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_manifest(manifest_path)
        if manifest is None:
            continue
        summaries.append(_summarize(manifest, run_id=entry.name))

    return RunListResponse(runs=summaries)


@router.get("/{run_id}", response_model=RunManifestResponse)
async def get_run(request: Request, run_id: str) -> RunManifestResponse:
    """Retourne le manifest complet d'un run."""
    state = request.app.state.picarones
    runs_dir = _runs_dir(state)
    run_dir = runs_dir / run_id
    manifest_path = run_dir / "run_manifest.json"

    # Validation : le run_id ne doit pas s'évader du workspace.
    try:
        run_dir_resolved = run_dir.resolve()
        runs_dir_resolved = runs_dir.resolve()
        if not str(run_dir_resolved).startswith(str(runs_dir_resolved)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="run_id invalide.",
            )
    except (OSError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"run_id invalide : {exc}",
        ) from exc

    if not manifest_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"run {run_id!r} introuvable.",
        )

    manifest = _read_manifest(manifest_path)
    if manifest is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"manifest du run {run_id!r} illisible.",
        )

    return RunManifestResponse(run_id=run_id, raw=manifest)


__all__ = ["router"]
