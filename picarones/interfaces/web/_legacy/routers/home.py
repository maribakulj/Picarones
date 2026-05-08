"""Router de la page d'accueil (Single Page Application).

Le HTML/CSS/JS qui vivait inline dans le code Python (variable
``_HTML_TEMPLATE`` historique de 3000+ lignes) a été découpé en :

- ``picarones/web/templates/`` (base + partials Jinja2)
- ``picarones/web/static/web-app.js`` (toute la logique JS)

Ce découpage permet :

1. de tester chaque vue indépendamment ;
2. de durcir la CSP à ``script-src 'self'`` (le JS n'est plus inline) ;
3. de toucher l'UI sans relire un fichier de 3000 lignes.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Cookie
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Sprint F (plan v2.0) — interfaces/ ne peut pas importer ``picarones`` racine.
_picarones = __import__("importlib").import_module("picarones")
__version__ = getattr(_picarones, "__version__", "unknown")
from picarones.interfaces.web._legacy.state import SUPPORTED_LANGS

router = APIRouter()


_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=False,
    lstrip_blocks=False,
)


def render_index(lang: str) -> str:
    """Rend la SPA depuis ``base.html.j2``.

    Déterministe pour un même couple ``(lang, version)`` — utilisé par
    le test de non-régression Sprint 25.
    """
    return _jinja_env.get_template("base.html.j2").render(
        lang=lang,
        version=__version__,
    )


@router.get("/", response_class=HTMLResponse)
async def index(picarones_lang: str = Cookie(default="fr")) -> HTMLResponse:
    """Page principale (Single Page Application)."""
    lang = picarones_lang if picarones_lang in SUPPORTED_LANGS else "fr"
    return HTMLResponse(content=render_index(lang))
