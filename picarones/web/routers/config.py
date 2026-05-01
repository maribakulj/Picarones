"""Router de sauvegarde / chargement des configs utilisateur."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Response

from picarones.web.config_utils import (
    CONFIG_SCHEMA_VERSION,
    filter_config,
    upgrade_config,
)

router = APIRouter()


@router.post("/api/config/save")
async def api_config_save(payload: dict) -> Response:
    """Sérialise un dict de config en JSON téléchargeable.

    Supprime la friction *« reconfigurer chaque session »*.
    Le client envoie sa config courante (engines, profil, options),
    le serveur retourne un fichier JSON à télécharger ; un autre
    utilisateur peut le réimporter via ``/api/config/load``.
    """
    cleaned = filter_config(payload or {})
    cleaned["schema_version"] = CONFIG_SCHEMA_VERSION
    cleaned["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    body = json.dumps(cleaned, ensure_ascii=False, indent=2, sort_keys=True)
    label = str(cleaned.get("label") or "picarones-config")
    # Sanitisation du nom : pas de "/" ni "..", longueur bornée
    safe_label = (
        "".join(c for c in label if c.isalnum() or c in "-_")
        or "picarones-config"
    )[:80]
    filename = f"{safe_label}-v{CONFIG_SCHEMA_VERSION}.json"
    return Response(
        content=body,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@router.post("/api/config/load")
async def api_config_load(payload: dict) -> dict:
    """Valide et normalise une config uploadée.

    Le client envoie le contenu JSON déjà parsé (le frontend lit le
    fichier via ``FileReader``). On filtre les champs autorisés,
    applique l'upgrade path éventuel, et retourne le dict normalisé.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Le corps doit être un objet JSON.")

    schema = payload.get("schema_version")
    if not isinstance(schema, int) or schema < 1 or schema > CONFIG_SCHEMA_VERSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"schema_version invalide ({schema!r}) — "
                f"attendu entre 1 et {CONFIG_SCHEMA_VERSION}."
            ),
        )

    upgraded = upgrade_config(payload)
    return {
        "config": filter_config(upgraded),
        "schema_version": CONFIG_SCHEMA_VERSION,
    }
