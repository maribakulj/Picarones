"""Router corpus — Sprint A14-S36.

Endpoints d'import et d'analyse de corpus, adossés à
``CorpusService`` (S20).  **Pas un shim** sur le legacy
``picarones.web.routers.corpus`` — c'est un router neuf, mince,
qui délègue toute la logique à ``CorpusService``.

Endpoints
---------
- ``POST /api/corpus/import``  : multipart upload d'un ZIP, retourne
  un ``CorpusImportResponse`` avec stats et warnings.
- ``GET  /api/corpus/{name}``  : retourne les métadonnées d'un
  corpus déjà importé (lit le manifest depuis le workspace).

Anti-sur-ingénierie
-------------------
- Pas de listing exhaustif des corpora.  Si un caller a besoin de
  lister, on l'ajoutera (typiquement S37+).
- Pas de browse arbitraire du filesystem (legacy
  ``/api/corpus/browse`` est une exposition risquée — la cible
  documentée demande un workflow plus contraint).
- Pas de delete — un caller peut supprimer manuellement le
  ``WorkspaceManager.root`` ou attendre la session expiration.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from picarones.app.services.corpus_service import CorpusImportError

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/corpus", tags=["corpus"])


# ──────────────────────────────────────────────────────────────────────
# Schémas de réponse
# ──────────────────────────────────────────────────────────────────────


class CorpusImportResponse(BaseModel):
    """Réponse JSON pour ``POST /api/corpus/import``."""

    corpus_name: str = Field(description="Nom du corpus importé.")
    extracted_dir: str = Field(description="Répertoire d'extraction.")
    n_documents: int
    n_images_without_gt: int
    n_gt_without_image: int
    n_skipped_noise: int
    warnings: list[str] = Field(default_factory=list)
    skipped_paths: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# POST /api/corpus/import
# ──────────────────────────────────────────────────────────────────────


@router.post(
    "/import",
    response_model=CorpusImportResponse,
    status_code=status.HTTP_201_CREATED,
)
async def import_corpus(
    request: Request,
    corpus_name: str,
    file: UploadFile = File(...),
) -> CorpusImportResponse:
    """Importe un corpus depuis un ZIP uploadé.

    Le service ``CorpusService.import_zip`` valide le ZIP (taille,
    nombre d'entrées, taille décompressée), l'extrait dans le
    workspace, et construit un ``CorpusSpec`` listant les paires
    image+GT détectées.

    Retourne un ``CorpusImportResponse`` avec stats et warnings.
    """
    state = request.app.state.picarones
    corpus_service = state.corpus

    # Validation rapide du nom : on délègue la validation stricte au
    # service mais on rejette tout de suite les noms vides.
    if not corpus_name or not corpus_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="corpus_name est requis et ne peut pas être vide.",
        )

    zip_bytes = await file.read()
    if not zip_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fichier ZIP vide.",
        )

    try:
        report = corpus_service.import_zip(
            zip_bytes=zip_bytes,
            corpus_name=corpus_name.strip(),
        )
    except CorpusImportError as exc:
        # Erreurs métier (ZIP mal formé, bombe, paths unsafe, ...).
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        # Erreurs inattendues — log + 500.
        logger.error(
            "[corpus] import inattendu en échec : %s", exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Échec d'import : {type(exc).__name__}",
        ) from exc

    return CorpusImportResponse(
        corpus_name=report.spec.name,
        extracted_dir=str(report.extracted_dir),
        n_documents=report.n_documents,
        n_images_without_gt=report.n_images_without_gt,
        n_gt_without_image=report.n_gt_without_image,
        n_skipped_noise=report.n_skipped_noise,
        warnings=list(report.warnings),
        skipped_paths=list(report.skipped_paths),
    )


__all__ = ["router"]
