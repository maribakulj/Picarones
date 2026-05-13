"""Router des importers de corpus distants : HTR-United et HuggingFace."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Query

from picarones.interfaces.web.models import HTRUnitedImportRequest, HuggingFaceImportRequest

router = APIRouter()


def _htr_united_catalogue():
    """Récupère le catalogue HTR-United (remote ou demo).

    Phase 4.4 du chantier post-rewrite : auparavant le router
    appelait ``HTRUnitedCatalogue.from_demo()`` exclusivement —
    l'UI annonçait "catalogue HTR-United" alors qu'on chargeait
    un échantillon embarqué.  Désormais ``from_remote()`` est
    utilisé (avec fallback automatique sur demo si offline), et
    le champ ``source`` (``"remote" | "demo"``) est exposé dans
    la réponse pour que l'UI puisse signaler clairement le mode
    actif.

    En CI / déploiement sans réseau, exporter
    ``PICARONES_HTR_UNITED_OFFLINE=1`` force le mode démo et
    évite un timeout de 10s à chaque GET catalogue.
    """
    from picarones.adapters.corpus.htr_united import HTRUnitedCatalogue

    if os.environ.get("PICARONES_HTR_UNITED_OFFLINE", "").strip() in (
        "1", "true", "yes",
    ):
        return HTRUnitedCatalogue.from_demo()
    return HTRUnitedCatalogue.from_remote(timeout=5)


# Phase 7.2 audit code-quality (2026-05) : la fonction
# ``_validated_output_dir`` (helper local historique) est désormais
# importée depuis ``interfaces/web/_path_helpers.py`` sous le nom
# ``validated_user_output_dir`` (factorisation de 3 sites web).  Alias
# conservé pour ne pas casser les usages locaux.
from picarones.interfaces.web._path_helpers import (
    validated_user_output_dir as _validated_output_dir,
)


# ──────────────────────────────────────────────────────────────────────────
# HTR-United
# ──────────────────────────────────────────────────────────────────────────

@router.get("/api/htr-united/catalogue")
async def api_htr_united_catalogue(
    query: str = Query(default="", description="Recherche textuelle"),
    language: str = Query(default="", description="Filtre langue"),
    script: str = Query(default="", description="Filtre type d'écriture"),
) -> dict:
    """Catalogue HTR-United filtrable (remote, fallback demo si offline)."""
    cat = _htr_united_catalogue()
    results = cat.search(
        query=query,
        language=language or None,
        script=script or None,
    )
    return {
        "source": cat.source,
        # Indication explicite du mode pour l'UI : "demo" si on charge
        # le catalogue embarqué (réseau indisponible ou variable
        # ``PICARONES_HTR_UNITED_OFFLINE=1`` exportée).
        "is_demo": cat.source == "demo",
        "total": len(results),
        "entries": [e.as_dict() for e in results],
        "available_languages": cat.available_languages(),
        "available_scripts": cat.available_scripts(),
    }


@router.post("/api/htr-united/import")
async def api_htr_united_import(req: HTRUnitedImportRequest) -> dict:
    """Importe une entrée HTR-United dans ``req.output_dir``."""
    from picarones.adapters.corpus.htr_united import import_htr_united_corpus

    output_dir = _validated_output_dir(req.output_dir)
    cat = _htr_united_catalogue()
    entry = cat.get_by_id(req.entry_id)
    if not entry:
        raise HTTPException(
            status_code=404, detail=f"Entrée non trouvée : {req.entry_id}",
        )

    return import_htr_united_corpus(
        entry=entry,
        output_dir=output_dir,
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

    output_dir = _validated_output_dir(req.output_dir)
    importer = HuggingFaceImporter()
    return importer.import_dataset(
        dataset_id=req.dataset_id,
        output_dir=output_dir,
        split=req.split,
        max_samples=req.max_samples,
    )
