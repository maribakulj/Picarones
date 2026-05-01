"""Router des profils de normalisation Unicode."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/api/normalization/profiles")
async def api_normalization_profiles() -> dict:
    """Liste les profils de normalisation disponibles avec leurs caractéristiques."""
    from picarones.measurements.normalization import NORMALIZATION_PROFILES

    profiles = [
        {
            "id": pid,
            "name": p.name,
            "description": p.description or p.name,
            "caseless": p.caseless,
            "diplomatic_rules": len(p.diplomatic_table),
            "exclude_chars": sorted(p.exclude_chars),
        }
        for pid, p in NORMALIZATION_PROFILES.items()
    ]
    return {"profiles": profiles}
