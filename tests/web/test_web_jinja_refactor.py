"""Tests Sprint 25 — refactor web frontend miroir du Sprint 17.

Sprint 17 a découpé le rapport HTML monolithique en 10 fichiers Jinja2.
Sprint 25 fait pareil pour la SPA web : l'ancien ``_HTML_TEMPLATE`` de
~1500 lignes string Python (3000+ lignes au total avec le JS) dans
``picarones/web/app.py`` est remplacé par :

    picarones/web/templates/
      ├── base.html.j2
      ├── _ascii_banner.html
      ├── _header_nav.html
      ├── _view_benchmark.html
      ├── _view_reports.html
      ├── _view_engines.html
      ├── _view_import.html
      └── _modals.html

    picarones/web/static/
      ├── picarones.css    (refonte XerOCR — Bauhaus/Xerox-Star)
      └── web-app.js       (extrait du <script> inline)

Ce module vérifie :

1. Les fichiers attendus existent et ne sont pas vides.
2. ``_render_index`` est déterministe (Sprint 17 imposait la même règle).
3. Les éléments structurants critiques sont présents (vues, nav, modals).
4. Pas de balise dupliquée (ex. deux ``id="view-benchmark"``).
5. Pas de bloc ``<script>...</script>`` inline avec du code dans la page
   rendue — uniquement des ``<script src="...">``.
6. ``picarones/web/app.py`` est passé sous la barre des 2000 lignes
   (était 3163 ; cible Sprint 25 long terme : ≤ 400, mais on commence
   par mesurer la victoire de l'extraction des templates).
7. Le rendu HTML reflète bien le cookie de langue (FR vs EN).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent.parent
WEB_DIR = ROOT / "picarones" / "interfaces" / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"
APP_PY = WEB_DIR / "app.py"


# ---------------------------------------------------------------------------
# 1. Présence et taille des fichiers extraits
# ---------------------------------------------------------------------------

EXPECTED_TEMPLATES = [
    "base.html.j2",
    "_rail.html",
    "_view_benchmark.html",
    "_view_reports.html",
    "_view_engines.html",
    "_view_import.html",
    "_modals.html",
]


class TestTemplateFilesExist:
    @pytest.mark.parametrize("name", EXPECTED_TEMPLATES)
    def test_template_present_and_non_empty(self, name):
        path = TEMPLATES_DIR / name
        assert path.is_file(), f"Template manquant : {path}"
        assert path.stat().st_size > 30, f"Template suspect (vide ?) : {path}"

    def test_web_app_js_extracted(self):
        path = STATIC_DIR / "web-app.js"
        assert path.is_file(), "web-app.js doit être extrait dans static/"
        # L'ancien <script> inline pesait ~1131 lignes
        line_count = sum(1 for _ in path.read_text(encoding="utf-8").splitlines())
        assert line_count > 500, (
            f"web-app.js semble trop court ({line_count} lignes) — "
            "extraction incomplète ?"
        )

    def test_picarones_css_present(self):
        # Sanity : la feuille de style XerOCR est en place.
        assert (STATIC_DIR / "picarones.css").is_file()


# ---------------------------------------------------------------------------
# 2. Déterminisme du rendu
# ---------------------------------------------------------------------------

class TestRenderIndexDeterminism:
    def test_same_inputs_same_output(self):
        from picarones.interfaces.web.routers.home import render_index as _render_index

        a = _render_index("fr")
        b = _render_index("fr")
        assert a == b, "Rendu non déterministe sur lang=fr"

    def test_lang_change_changes_output(self):
        from picarones.interfaces.web.routers.home import render_index as _render_index

        fr = _render_index("fr")
        en = _render_index("en")
        assert fr != en, "Le rendu doit dépendre de la langue"
        assert 'name="picarones-lang" content="fr"' in fr
        assert 'name="picarones-lang" content="en"' in en

    def test_html_lang_attribute_set(self):
        from picarones.interfaces.web.routers.home import render_index as _render_index
        assert '<html lang="en">' in _render_index("en")
        assert '<html lang="fr">' in _render_index("fr")


# ---------------------------------------------------------------------------
# 3. Éléments structurants présents
# ---------------------------------------------------------------------------

class TestStructuralElementsPresent:
    @pytest.fixture(scope="class")
    def html(self) -> str:
        from picarones.interfaces.web.routers.home import render_index as _render_index
        return _render_index("fr")

    @pytest.mark.parametrize("view_id", [
        "view-benchmark",
        "view-reports",
        "view-engines",
        "view-import",
    ])
    def test_each_view_present(self, html, view_id):
        assert f'id="{view_id}"' in html, (
            f"Vue '{view_id}' manquante dans la page rendue"
        )

    def test_nav_buttons_present(self, html):
        # Sidebar XerOCR : benchmark + library + reports + engines.
        # La vue est toujours `view-import` côté DOM mais le label nav
        # est `nav_library` (l'ancien import est fusionné dans la
        # bibliothèque).
        for label in ("nav_benchmark", "nav_library", "nav_reports", "nav_engines"):
            assert f'data-i18n="{label}"' in html

    def test_import_modal_present(self, html):
        assert 'id="import-modal"' in html

    def test_external_js_referenced(self, html):
        # Le bundle JS doit être chargé via <script src="...">
        assert re.search(r'<script\s+src="/static/web-app\.js', html), (
            "La balise <script src='/static/web-app.js'> doit être présente"
        )

    def test_picarones_css_referenced(self, html):
        assert re.search(r'<link\s+rel="stylesheet"\s+href="/static/picarones\.css', html)


# ---------------------------------------------------------------------------
# 4. Pas de balise dupliquée (garde-fou contre {% include %} en double)
# ---------------------------------------------------------------------------

_ID_RE = re.compile(r'\sid="([a-zA-Z0-9_\-]+)"')


class TestNoDuplicateIds:
    def test_no_duplicate_ids_in_rendered_page(self):
        from picarones.interfaces.web.routers.home import render_index as _render_index
        html = _render_index("fr")
        ids = _ID_RE.findall(html)
        # Les `id` HTML doivent être uniques (W3C). Une duplication signe un
        # double-include accidentel ou un copier-coller raté.
        seen: dict[str, int] = {}
        for i in ids:
            seen[i] = seen.get(i, 0) + 1
        dupes = {k: v for k, v in seen.items() if v > 1}
        assert not dupes, f"IDs dupliqués dans la SPA rendue : {dupes}"


# ---------------------------------------------------------------------------
# 5. Pas de gros bloc <script>...</script> inline avec du code
# ---------------------------------------------------------------------------

class TestNoInlineScriptCode:
    """Sprint 25 a extrait tout le JS dans /static/web-app.js. La page
    rendue ne doit plus contenir un bloc ``<script>...</script>`` qui
    embarque du code (les ``<script src="..."></script>`` restent
    autorisés)."""

    def test_no_large_inline_script_block(self):
        from picarones.interfaces.web.routers.home import render_index as _render_index
        html = _render_index("fr")
        # Capture tout le contenu entre <script> sans src= et </script>.
        pattern = re.compile(
            r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
            re.DOTALL,
        )
        for body in pattern.findall(html):
            # Quelques bytes blancs sont tolérés (ex. <script>\n</script>)
            stripped = body.strip()
            assert len(stripped) < 200, (
                "Un bloc <script> inline contient encore du code "
                f"({len(stripped)} caractères). Doit vivre dans /static/web-app.js."
            )


# ---------------------------------------------------------------------------
# 6. Mesure du dégonflement de app.py
# ---------------------------------------------------------------------------

class TestAppPyShrunk:
    def test_app_py_below_target_threshold(self):
        # Sprint 25 a fait passer app.py de 3163 à 1690 lignes en
        # extrayant ``_HTML_TEMPLATE`` (HTML + CSS + 1131 lignes de JS).
        # Sprint 28 a ajouté ~370 lignes de nouveaux endpoints
        # (config/save, config/load, synthesis_preview, history/regressions).
        # La cible long terme reste ≤ 400 lignes et sera atteinte au
        # Sprint 31 quand on découpera en ``picarones/web/routes/``.
        # En attendant, on borne à 2400 pour détecter une re-régression
        # (ex. quelqu'un qui réintroduit un gros template inline).
        n = sum(1 for _ in APP_PY.read_text(encoding="utf-8").splitlines())
        assert n < 2400, (
            f"web/app.py fait {n} lignes — sortie de la borne haute. "
            "Le découpage en routes/ est-il toujours sur la roadmap ?"
        )

    def test_html_template_string_removed(self):
        src = APP_PY.read_text(encoding="utf-8")
        assert "_HTML_TEMPLATE = r" not in src, (
            "Le monolithe _HTML_TEMPLATE doit être supprimé de app.py"
        )


# ---------------------------------------------------------------------------
# 7. Smoke test bout-en-bout via TestClient
# ---------------------------------------------------------------------------

class TestEndpointStillServesPage:
    @pytest.fixture
    def client(self):
        from picarones.interfaces.web.app import app
        return TestClient(app)

    def test_root_returns_200_and_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_root_respects_cookie_lang(self, client):
        r = client.get("/", cookies={"picarones_lang": "en"})
        assert 'content="en"' in r.text
        r2 = client.get("/", cookies={"picarones_lang": "fr"})
        assert 'content="fr"' in r2.text

    def test_root_falls_back_on_unsupported_lang(self, client):
        r = client.get("/", cookies={"picarones_lang": "ne-pas-exister"})
        # Doit retomber sur fr (cf. ``_SUPPORTED_LANGS``)
        assert 'content="fr"' in r.text

    def test_static_js_served(self, client):
        r = client.get("/static/web-app.js")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith(("application/javascript", "text/javascript"))
