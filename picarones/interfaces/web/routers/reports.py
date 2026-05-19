"""Router des rapports HTML générés."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from picarones.interfaces.web.security import compute_browse_roots
from picarones.interfaces.web.state import UPLOADS_DIR

router = APIRouter()

# Dossiers de sortie canoniques de l'app — toujours scannés, quelle
# que soit la racine demandée (ce ne sont pas des entrées utilisateur).
_DEFAULT_REPORT_DIRS = [Path(".").resolve(), Path("./rapports").resolve()]


def _reports_dir_allowed(target: Path) -> bool:
    """Confine ``reports_dir`` aux racines autorisées.

    Sans ce garde-fou, ``/api/reports?reports_dir=/etc`` énumère les
    ``*.html`` (noms, tailles, mtime) de n'importe quel dossier
    lisible par le process — fuite de structure filesystem.  On
    réutilise :func:`compute_browse_roots` (même modèle que
    ``/api/corpus/browse``) + les dossiers de rapport canoniques.
    """
    allowed = list(_DEFAULT_REPORT_DIRS)
    allowed.extend(compute_browse_roots(UPLOADS_DIR))
    for root in allowed:
        try:
            if target == root or target.is_relative_to(root):
                return True
        except (ValueError, TypeError):
            continue
    return False


@router.get("/api/reports")
async def api_reports(
    reports_dir: str = Query(default=".", description="Dossier rapports"),
) -> dict:
    """Liste les rapports HTML disponibles dans ``reports_dir``, ``.`` et ``./rapports``.

    ``reports_dir`` est confiné aux racines autorisées (cf.
    :func:`_reports_dir_allowed`) : un chemin hors périmètre est
    refusé en 403 plutôt que d'énumérer un dossier arbitraire.

    Le scan disque (``glob``, ``stat``) est exécuté dans un thread
    pour ne pas bloquer l'event loop si le dossier contient des
    centaines de rapports.
    """
    target = Path(reports_dir).resolve()
    if not _reports_dir_allowed(target):
        raise HTTPException(
            status_code=403,
            detail="Accès refusé : reports_dir hors des répertoires autorisés",
        )
    return await asyncio.to_thread(_list_reports_sync, target)


def _list_reports_sync(target: Path) -> dict:
    reports: list[dict] = []

    search_dirs = [target, *_DEFAULT_REPORT_DIRS]
    seen: set[str] = set()

    for d in search_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.html"), key=lambda x: x.stat().st_mtime, reverse=True):
            if str(f) not in seen:
                seen.add(str(f))
                stat = f.stat()
                reports.append({
                    "filename": f.name,
                    "path": str(f),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc,
                    ).isoformat(),
                    "url": f"/reports/{f.name}",
                })

    return {"reports": reports}


@router.get("/reports/{filename}", response_class=HTMLResponse)
async def serve_report(filename: str) -> HTMLResponse:
    """Sert un rapport HTML par son nom de fichier.

    Lit le contenu et le renvoie en ``text/html`` plutôt que de
    rediriger vers un fichier statique : indispensable pour que ça
    fonctionne depuis un Codespace ou tout reverse-proxy distant.

    La lecture est déléguée à un thread car les rapports peuvent
    atteindre plusieurs Mo (Chart.js inline, données denses).
    """
    # Sécurité : interdire les path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Nom de fichier invalide")
    content = await asyncio.to_thread(_read_report_sync, filename)
    return HTMLResponse(content=content)


def _read_report_sync(filename: str) -> str:
    for d in [Path("."), Path("./rapports")]:
        f = d / filename
        if f.exists() and f.suffix == ".html":
            return f.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail=f"Rapport non trouvé : {filename}")
