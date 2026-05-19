"""Router de gestion du corpus : navigation, upload, listing, suppression."""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from picarones.interfaces.web.corpus_utils import analyze_corpus_dir, flatten_zip_to_dir
from picarones.interfaces.web.security import (
    UPLOAD_CHUNK_SIZE,
    compute_browse_roots,
    get_max_total_upload_mb,
    get_max_upload_mb,
    validate_image_file_safe,
)
from picarones.interfaces.web.state import IMAGE_EXTS, UPLOADS_DIR

router = APIRouter()
_logger = logging.getLogger(__name__)


# Racines configurables via PICARONES_BROWSE_ROOTS, sinon
# défaut restreint en mode public, défaut historique en mode dev.
_BROWSE_ROOTS = compute_browse_roots(UPLOADS_DIR)


def _dedupe_name(raw: str, seen: set[str]) -> str:
    """Évite l'écrasement silencieux de deux fichiers multipart de
    même basename (audit P0.5).

    ``photo.png`` envoyé 2× ⇒ ``photo.png`` puis ``photo_1.png`` —
    pas de perte de données silencieuse, pas de mauvaise association
    image/GT.
    """
    if raw not in seen:
        return raw
    stem, dot, ext = raw.partition(".")
    i = 1
    while True:
        candidate = f"{stem}_{i}{dot}{ext}"
        if candidate not in seen:
            return candidate
        i += 1


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
    """Upload un corpus : soit un ``.zip``, soit une sélection d'images + ``.gt.txt``.

    Chaque fichier est *streamé* vers le disque par blocs bornés
    (``UPLOAD_CHUNK_SIZE``) — jamais matérialisé en un seul ``bytes``
    en RAM.  Deux plafonds durs sont appliqués pendant le streaming :
    par fichier (``PICARONES_MAX_UPLOAD_MB``) et cumulé sur la requête
    (``PICARONES_MAX_TOTAL_UPLOAD_MB``).  Le dossier est purgé sur
    *toute* sortie anormale (violation de quota, déconnexion client,
    erreur) — pas de résidu disque.  La finalisation (validation
    image, extraction ZIP, analyse) est déléguée à un thread.
    """
    corpus_id = str(uuid.uuid4())
    corpus_dir = UPLOADS_DIR / corpus_id
    corpus_dir.mkdir(parents=True, exist_ok=True)

    max_file_bytes = get_max_upload_mb() * 1024 * 1024
    max_total_bytes = get_max_total_upload_mb() * 1024 * 1024

    try:
        total = 0
        staged: list[str] = []
        seen_names: set[str] = set()
        for uf in files:
            try:
                # Empêcher la traversée via le nom reçu du client
                # (multipart) : basename seul ; puis dédup pour ne pas
                # écraser silencieusement un fichier de même nom.
                raw_name = Path(uf.filename or "upload").name
                safe_name = _dedupe_name(raw_name, seen_names)
                seen_names.add(safe_name)
                dest = corpus_dir / safe_name
                written = 0
                with dest.open("wb") as fh:
                    while True:
                        chunk = await uf.read(UPLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        written += len(chunk)
                        total += len(chunk)
                        if written > max_file_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"Fichier '{safe_name}' dépasse "
                                    f"{get_max_upload_mb()} Mo "
                                    "(PICARONES_MAX_UPLOAD_MB)."
                                ),
                            )
                        if total > max_total_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    "Upload total dépasse "
                                    f"{get_max_total_upload_mb()} Mo "
                                    "(PICARONES_MAX_TOTAL_UPLOAD_MB)."
                                ),
                            )
                        fh.write(chunk)
                staged.append(safe_name)
            finally:
                # Fermeture explicite du SpooledTemporaryFile sous-jacent
                # (libère le fd / le fichier temp même en cas d'erreur).
                await uf.close()

        try:
            summary = await asyncio.to_thread(
                _finalize_uploaded_dir, corpus_dir, staged,
            )
        except ValueError as exc:
            # Image invalide / ZIP corrompu ou bombe de décompression.
            raise HTTPException(status_code=415, detail=str(exc))

        if not summary["usable"]:
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
        # Nettoyage garanti : violation de quota, 422, déconnexion.
        shutil.rmtree(corpus_dir, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(corpus_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))


def _finalize_uploaded_dir(corpus_dir: Path, staged: list[str]) -> dict:
    """Valide et normalise les fichiers déjà streamés sur disque.

    Exécuté dans un thread (lib appelante : ``api_corpus_upload``).
    Lève ``ValueError`` sur image invalide ou ZIP corrompu —
    l'appelant doit traduire en HTTP 415.

    - ``.zip`` : extrait via ``flatten_zip_to_dir`` (ouvert depuis le
      chemin disque, pas de chargement RAM intégral), puis supprimé.
    - image : validée depuis le fichier (``validate_image_file_safe``,
      Pillow lit en flux — pas de ``read_bytes()`` intégral).
    - ``.txt``/``.xml``/``.gt.txt`` : conservés tels quels.
    - autre : supprimé (type ignoré).
    """
    for name in staged:
        p = corpus_dir / name
        if not p.exists():
            continue
        suffix = p.suffix.lower()
        if suffix == ".zip":
            with zipfile.ZipFile(p) as zf:
                flatten_zip_to_dir(zf, corpus_dir)
            p.unlink(missing_ok=True)
        elif suffix in IMAGE_EXTS:
            validate_image_file_safe(p, filename=name)
        elif (
            name.endswith(".gt.txt")
            or name.endswith(".ocr.txt")
            or suffix in (".txt", ".xml")
        ):
            continue  # conservé
        else:
            p.unlink(missing_ok=True)  # type ignoré

    return analyze_corpus_dir(corpus_dir)


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
