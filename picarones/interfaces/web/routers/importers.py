"""Router des importers de corpus distants : HTR-United et HuggingFace."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from picarones.interfaces.web.models import HTRUnitedImportRequest, HuggingFaceImportRequest

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────
# HTR-United
# ──────────────────────────────────────────────────────────────────────────

@router.get("/api/htr-united/catalogue")
async def api_htr_united_catalogue(
    query: str = Query(default="", description="Recherche textuelle"),
    language: str = Query(default="", description="Filtre langue"),
    script: str = Query(default="", description="Filtre type d'écriture"),
) -> dict:
    """Catalogue HTR-United filtrable."""
    from picarones.adapters.corpus.htr_united import HTRUnitedCatalogue

    cat = HTRUnitedCatalogue.from_demo()
    results = cat.search(
        query=query,
        language=language or None,
        script=script or None,
    )
    return {
        "source": cat.source,
        "total": len(results),
        "entries": [e.as_dict() for e in results],
        "available_languages": cat.available_languages(),
        "available_scripts": cat.available_scripts(),
    }


@router.post("/api/htr-united/import")
async def api_htr_united_import(req: HTRUnitedImportRequest) -> dict:
    """Importe une entrée HTR-United dans ``req.output_dir``."""
    from picarones.adapters.corpus.htr_united import (
        HTRUnitedCatalogue,
        import_htr_united_corpus,
    )

    cat = HTRUnitedCatalogue.from_demo()
    entry = cat.get_by_id(req.entry_id)
    if not entry:
        raise HTTPException(
            status_code=404, detail=f"Entrée non trouvée : {req.entry_id}",
        )

    return import_htr_united_corpus(
        entry=entry,
        output_dir=req.output_dir,
        max_samples=req.max_samples,
    )


# ──────────────────────────────────────────────────────────────────────────
# HuggingFace Datasets
# ──────────────────────────────────────────────────────────────────────────

@router.get("/api/huggingface/search")
async def api_huggingface_search(
    query: str = Query(default="", description="Requête de recherche"),
    language: str = Query(default="", description="Filtre langue"),
    tags: str = Query(default="", description="Tags séparés par des virgules"),
    limit: int = Query(default=20, ge=1, le=50),
) -> dict:
    """Recherche de datasets sur HuggingFace Hub."""
    from picarones.adapters.corpus.huggingface import HuggingFaceImporter

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    importer = HuggingFaceImporter()
    results = importer.search(
        query=query,
        tags=tag_list,
        language=language or None,
        limit=limit,
    )
    return {
        "total": len(results),
        "datasets": [ds.as_dict() for ds in results],
    }


@router.post("/api/huggingface/import")
async def api_huggingface_import(req: HuggingFaceImportRequest) -> dict:
    """Importe un dataset HuggingFace dans ``req.output_dir``."""
    from picarones.adapters.corpus.huggingface import HuggingFaceImporter

    importer = HuggingFaceImporter()
    return importer.import_dataset(
        dataset_id=req.dataset_id,
        output_dir=req.output_dir,
        split=req.split,
        max_samples=req.max_samples,
    )
