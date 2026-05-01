"""Router de gestion du corpus : navigation, upload, listing, suppression."""

from __future__ import annotations

import io
import logging
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from picarones.web.corpus_utils import analyze_corpus_dir, flatten_zip_to_dir
from picarones.web.security import compute_browse_roots, validate_image_safe
from picarones.web.state import IMAGE_EXTS, UPLOADS_DIR

router = APIRouter()
_logger = logging.getLogger(__name__)


# Sprint 24 — racines configurables via PICARONES_BROWSE_ROOTS, sinon
# défaut restreint en mode public, défaut historique en mode dev.
_BROWSE_ROOTS = compute_browse_roots(UPLOADS_DIR)


def _is_path_allowed(target: Path) -> bool:
    """Vérifie qu'un chemin résolu est sous un des répertoires autorisés."""
    for root in _BROWSE_ROOTS:
        try:
            if target == root or target.is_relative_to(root):
                return True
        except (ValueError, TypeError):
            continue
    return False


@router.get("/api/corpus/browse")
async def api_corpus_browse(
    path: str = Query(default=".", description="Chemin à explorer"),
) -> dict:
    """Parcourt un dossier autorisé et retourne ses entrées."""
    target = Path(path).resolve()
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Dossier non trouvé : {path}")
    if not _is_path_allowed(target):
        raise HTTPException(
            status_code=403,
            detail="Accès refusé : chemin hors des répertoires autorisés",
        )

    items = []
    try:
        for entry in sorted(target.iterdir()):
            item: dict[str, Any] = {
                "name": entry.name,
                "path": str(entry),
                "is_dir": entry.is_dir(),
            }
            if entry.is_dir():
                gt_count = sum(
                    1 for f in entry.iterdir()
                    if f.suffix == ".txt" and f.stem.endswith(".gt")
                )
                item["gt_count"] = gt_count
                item["has_corpus"] = gt_count > 0
            items.append(item)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    return {
        "current_path": str(target),
        "parent_path": str(target.parent) if target.parent != target else None,
        "items": items,
    }


@router.post("/api/corpus/upload")
async def api_corpus_upload(files: list[UploadFile] = File(...)) -> dict:
    """Upload un corpus : soit un ``.zip``, soit une sélection d'images + ``.gt.txt``."""
    corpus_id = str(uuid.uuid4())
    corpus_dir = UPLOADS_DIR / corpus_id
    corpus_dir.mkdir(parents=True, exist_ok=True)

    try:
        for uf in files:
            filename = uf.filename or "upload"
            # Sprint 24 — empêcher la traversée via le nom de fichier reçu
            # depuis le client (multipart). On garde uniquement le basename.
            safe_name = Path(filename).name
            data = await uf.read()
            suffix = Path(safe_name).suffix.lower()

            if suffix == ".zip":
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    flatten_zip_to_dir(zf, corpus_dir)
            elif suffix in IMAGE_EXTS:
                # Sprint 24 — valider l'image avant écriture (Pillow.verify,
                # taille max, rejet des bombes de décompression).
                try:
                    validate_image_safe(data, filename=safe_name)
                except ValueError as exc:
                    raise HTTPException(status_code=415, detail=str(exc))
                (corpus_dir / safe_name).write_bytes(data)
            elif (
                safe_name.endswith(".gt.txt")
                or safe_name.endswith(".ocr.txt")
                or suffix in (".txt", ".xml")
            ):
                (corpus_dir / safe_name).write_bytes(data)
            # Ignorer les autres types

        summary = analyze_corpus_dir(corpus_dir)
        if not summary["usable"]:
            shutil.rmtree(corpus_dir, ignore_errors=True)
            raise HTTPException(
                status_code=422,
                detail="Aucune paire image/.gt.txt valide trouvée dans les fichiers uploadés.",
            )

        return {
            "corpus_id": corpus_id,
            "corpus_path": str(corpus_dir),
            **summary,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(corpus_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/corpus/uploads")
async def api_corpus_uploads() -> dict:
    """Liste les corpus uploadés disponibles."""
    if not UPLOADS_DIR.exists():
        return {"uploads": []}

    uploads = []
    for d in sorted(UPLOADS_DIR.iterdir()):
        if not d.is_dir():
            continue
        try:
            summary = analyze_corpus_dir(d)
            uploads.append({
                "corpus_id": d.name,
                "corpus_path": str(d),
                "doc_count": summary["doc_count"],
                "has_missing_gt": summary["has_missing_gt"],
            })
        except Exception as e:  # noqa: BLE001
            _logger.warning(
                "[api_corpus_uploads] upload '%s' ignoré — inspection impossible : %s",
                d.name, e,
            )
    return {"uploads": uploads}


@router.get("/api/corpus/image/{upload_id}/{filename}")
async def api_corpus_image(upload_id: str, filename: str) -> FileResponse:
    """Sert une image depuis le dossier d'upload."""
    # Sécurité : interdire les path traversal
    if "/" in upload_id or "\\" in upload_id or ".." in upload_id:
        raise HTTPException(status_code=400, detail="upload_id invalide")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="filename invalide")
    image_path = UPLOADS_DIR / upload_id / filename
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image non trouvée")
    suffix = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff", ".tiff": "image/tiff",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    return FileResponse(str(image_path), media_type=media_type)


@router.delete("/api/corpus/uploads/{corpus_id}")
async def api_corpus_delete(corpus_id: str) -> dict:
    """Supprime un corpus uploadé."""
    if "/" in corpus_id or "\\" in corpus_id or ".." in corpus_id:
        raise HTTPException(status_code=400, detail="corpus_id invalide")
    corpus_dir = UPLOADS_DIR / corpus_id
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Corpus non trouvé : {corpus_id}")
    shutil.rmtree(corpus_dir)
    return {"deleted": corpus_id}
