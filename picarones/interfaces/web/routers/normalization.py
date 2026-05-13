"""Router des profils de normalisation Unicode."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


#: Limite la taille d'un YAML utilisateur — au-delà c'est suspect
#: (un profil typique fait quelques dizaines de lignes).  Phase 3.3
#: audit code-quality.
_MAX_YAML_BYTES = 64 * 1024


@router.get("/api/normalization/profiles")
async def api_normalization_profiles() -> dict:
    """Liste les profils de normalisation disponibles avec leurs caractéristiques."""
    from picarones.evaluation.metrics.normalization import NORMALIZATION_PROFILES

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


class _PreviewYAMLRequest(BaseModel):
    """Corps de ``POST /api/normalization/profiles/preview``.

    Le client envoie le contenu textuel d'un fichier YAML — la
    validation passe par :meth:`NormalizationProfile.from_yaml`,
    ce qui garantit l'unicité du chemin de parsing (CLI + web).
    """

    yaml: str = Field(
        ...,
        description=(
            "Contenu textuel d'un fichier YAML décrivant un profil de "
            "normalisation custom (clés : name, description, caseless, "
            "nfc, exclude_chars, diplomatic)."
        ),
        max_length=_MAX_YAML_BYTES,
    )


@router.post("/api/normalization/profiles/preview")
async def api_normalization_profile_preview(req: _PreviewYAMLRequest) -> dict:
    """Valide un YAML de profil custom et retourne le profil sérialisé.

    Phase 3.3 audit code-quality — permet aux chercheurs de
    pré-visualiser leur YAML versionné avant de l'utiliser dans un
    benchmark via la CLI (``--normalization-profile /path/to.yaml``).

    Le YAML n'est **pas** persisté côté serveur ; il est écrit dans
    un fichier temporaire le temps du parsing puis effacé.  Le
    profil retourné n'est pas non plus enregistré dans
    :data:`NORMALIZATION_PROFILES` — il sert uniquement de preview
    pour confirmer le format au client.

    Raises
    ------
    HTTPException(400)
        Si le YAML est mal formé ou ne respecte pas le schéma.
    """
    payload_bytes = req.yaml.encode("utf-8")
    if len(payload_bytes) > _MAX_YAML_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"YAML trop volumineux ({len(payload_bytes)} octets) — "
                f"limite {_MAX_YAML_BYTES}."
            ),
        )

    from picarones.formats.text.normalization import NormalizationProfile

    # Écriture dans un fichier temporaire — ``from_yaml`` attend un
    # chemin, par cohérence avec l'usage CLI où la source est un
    # fichier git-versionné.
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".yaml", delete=False,
    ) as tmp:
        tmp.write(payload_bytes)
        tmp_path = Path(tmp.name)

    try:
        try:
            profile = NormalizationProfile.from_yaml(tmp_path)
        except Exception as exc:  # noqa: BLE001 — surface yaml + parsing
            raise HTTPException(
                status_code=400,
                detail=f"YAML invalide : {exc}",
            ) from exc
    finally:
        try:
            tmp_path.unlink()
        except OSError:  # pragma: no cover — best-effort cleanup
            pass

    return {
        "name": profile.name,
        "description": profile.description or profile.name,
        "caseless": profile.caseless,
        "nfc": profile.nfc,
        "diplomatic_rules": len(profile.diplomatic_table),
        "diplomatic_table": dict(profile.diplomatic_table),
        "exclude_chars": sorted(profile.exclude_chars),
    }
