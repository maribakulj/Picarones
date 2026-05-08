"""Router système : statut applicatif, /health, langue, jeton CSRF."""

from __future__ import annotations

from fastapi import APIRouter, Cookie, HTTPException, Response

# Sprint F (plan v2.0) — interfaces/ ne peut pas importer ``picarones`` racine.
_picarones = __import__("importlib").import_module("picarones")
__version__ = getattr(_picarones, "__version__", "unknown")
from picarones.interfaces.web._legacy.security import (
    CSRF_COOKIE,
    generate_csrf_token,
    is_csrf_required,
)
from picarones.interfaces.web._legacy.state import LANG_COOKIE, SUPPORTED_LANGS, iso_now

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """Endpoint *liveness* minimal pour orchestrateurs (Docker, Kubernetes).

    Sprint A4 (item M-3 de l'audit institutional-readiness-2026-05) :
    le ``HEALTHCHECK`` du Dockerfile pointe vers ``/health`` ; sans
    cet endpoint dédié, le check Docker échouait. Le contenu est
    volontairement minimal pour répondre en < 50 ms même sous charge :

    - **pas** d'accès à la base SQLite ``jobs.sqlite`` (qui peut être
      verrouillée pendant un benchmark long) ;
    - **pas** d'introspection des engines (qui peut tomber sur un
      timeout réseau pour les adapters cloud) ;
    - **pas** de calcul ni d'I/O.

    Pour un état applicatif riche (versions des engines, charge
    courante, mode public, etc.), utiliser ``/api/status``.
    """
    return {"status": "ok", "version": __version__}


@router.get("/api/csrf/token")
async def api_csrf_token(response: Response) -> dict:
    """Force la rotation du jeton CSRF — Sprint A4 (item B-11).

    Pose un cookie ``picarones_csrf`` frais sur la réponse, indépendant
    du middleware de rotation paresseuse. Utile :

    - au démarrage du frontend (single-page app), avant le premier POST ;
    - après un logout / changement d'utilisateur ;
    - depuis ``curl`` pour scripter un client en mode institutionnel.

    En mode public (``PICARONES_CSRF_REQUIRED`` désactivé), retourne
    ``enabled=false`` et ne pose pas de cookie — le frontend peut alors
    décider de ne pas injecter le header sur ses POST.
    """
    if not is_csrf_required():
        return {"enabled": False, "token": None}
    token = generate_csrf_token()
    response.set_cookie(
        key=CSRF_COOKIE,
        value=token,
        httponly=False,
        samesite="strict",
        secure=False,
    )
    return {"enabled": True, "token": token}


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
    # ``httponly=False`` est volontaire : le frontend lit ce cookie en
    # JS pour adapter l'UI sans round-trip serveur. ``samesite="strict"``
    # car aucun usage légitime ne demande que ce cookie soit envoyé
    # depuis une navigation cross-site — on resserre le cran qu'on
    # peut sans casser l'UX.
    response.set_cookie(
        key=LANG_COOKIE,
        value=lang_code,
        max_age=60 * 60 * 24 * 365,  # 1 an
        httponly=False,
        samesite="strict",
    )
    return {"lang": lang_code, "message": f"Langue définie : {lang_code}"}
