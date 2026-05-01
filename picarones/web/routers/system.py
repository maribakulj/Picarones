"""Router système : statut applicatif et langue de l'interface."""

from __future__ import annotations

from fastapi import APIRouter, Cookie, HTTPException, Response

from picarones import __version__
from picarones.web.state import LANG_COOKIE, SUPPORTED_LANGS, iso_now

router = APIRouter()


@router.get("/api/status")
async def api_status() -> dict:
    """Version et état de l'application."""
    return {
        "app": "Picarones",
        "version": __version__,
        "status": "ok",
        "timestamp": iso_now(),
    }


@router.get("/api/lang")
async def api_get_lang(picarones_lang: str = Cookie(default="fr")) -> dict:
    """Retourne la langue courante (lue depuis le cookie de session)."""
    lang = picarones_lang if picarones_lang in SUPPORTED_LANGS else "fr"
    return {"lang": lang, "supported": list(SUPPORTED_LANGS)}


@router.post("/api/lang/{lang_code}")
async def api_set_lang(lang_code: str, response: Response) -> dict:
    """Définit la langue de l'interface et la persiste dans un cookie de session.

    Langues supportées : ``fr`` (français), ``en`` (anglais patrimonial).
    """
    if lang_code not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Langue non supportée : '{lang_code}'. "
                f"Disponibles : {', '.join(SUPPORTED_LANGS)}"
            ),
        )
    response.set_cookie(
        key=LANG_COOKIE,
        value=lang_code,
        max_age=60 * 60 * 24 * 365,  # 1 an
        httponly=False,
        samesite="lax",
    )
    return {"lang": lang_code, "message": f"Langue définie : {lang_code}"}
