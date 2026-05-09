"""Tests Sprint A6 — accessibilité WCAG niveau A bloquant.

Items B-9, B-10, m-3, m-4 de l'audit institutional-readiness-2026-05.

Ce fichier valide le **socle a11y bloquant** pour une déclaration de
conformité RGAA / WCAG 2.1 niveau A :

- WCAG 2.4.1 (Bypass Blocks) — skip-to-content link (B-10)
- WCAG 1.1.1 (Non-text Content) — Canvas charts → aria-label + table
  jumelle accessible aux AT (B-9)
- WCAG 1.3.1 (Info and Relationships) — ``scope="col"`` sur les
  ``<th>`` (m-4)
- Pas de chaîne hardcodée FR/EN dans la nav (m-3)

Ces tests se contentent de vérifier la présence des marqueurs HTML
attendus dans le rapport généré. L'audit sémantique complet (NVDA /
JAWS / VoiceOver) reste manuel et tracé dans
``docs/audits/external-audits-2026/`` (Sprint A15).
"""

from __future__ import annotations

import re

import pytest

from picarones.evaluation.synthetic import generate_sample_benchmark
from picarones.reports.html.generator import ReportGenerator


@pytest.fixture(scope="module")
def demo_html(tmp_path_factory) -> str:
    """Rapport démo (FR) généré une fois pour tous les tests du module."""
    out = tmp_path_factory.mktemp("a11y") / "report.html"
    bench = generate_sample_benchmark(n_docs=4)
    ReportGenerator(bench, lang="fr").generate(out)
    return out.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def demo_html_en(tmp_path_factory) -> str:
    """Rapport démo (EN) — pour vérifier que les libellés a11y sont
    bilingues."""
    out = tmp_path_factory.mktemp("a11y_en") / "report_en.html"
    bench = generate_sample_benchmark(n_docs=4)
    ReportGenerator(bench, lang="en").generate(out)
    return out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# B-10 — Skip-to-content (WCAG 2.4.1)
# ---------------------------------------------------------------------------


def test_skip_link_present(demo_html: str) -> None:
    """Un lien ``href="#main"`` avec class ``skip-link`` doit exister."""
    assert 'class="skip-link"' in demo_html
    assert 'href="#main"' in demo_html


def test_skip_link_first_focusable_in_body(demo_html: str) -> None:
    """Le skip-link doit être le **premier** élément focusable du body
    (sinon Tab depuis l'URL bar atteint d'abord la nav, ce qui défait
    le but)."""
    body_start = demo_html.find("<body>")
    assert body_start > 0
    body_part = demo_html[body_start : body_start + 1500]
    skip_pos = body_part.find('class="skip-link"')
    nav_pos = body_part.find("<nav")
    assert skip_pos > 0 and nav_pos > 0
    assert skip_pos < nav_pos, (
        "Le skip-link doit précéder le <nav> dans le DOM."
    )


def test_main_has_id_main(demo_html: str) -> None:
    """Le ``<main>`` doit avoir ``id="main"`` pour que le skip-link
    pointe vers une cible existante."""
    assert re.search(r'<main[^>]*\bid="main"', demo_html), (
        '<main id="main"> attendu pour la cible du skip-link.'
    )


def test_skip_link_label_is_i18n(demo_html: str, demo_html_en: str) -> None:
    """Le libellé du skip-link doit être en français en mode FR et en
    anglais en mode EN (pas de chaîne hardcodée)."""
    # FR : "Aller au contenu"
    assert "Aller au contenu" in demo_html
    # EN : "Skip to content"
    assert "Skip to content" in demo_html_en


# ---------------------------------------------------------------------------
# B-9 — Canvas charts accessibles (WCAG 1.1.1)
# ---------------------------------------------------------------------------


def test_all_canvases_have_aria_label(demo_html: str) -> None:
    """Tout ``<canvas>`` Chart.js (avec ``id="chart-..."`` ou
    ``pareto-chart``) doit avoir ``aria-label`` non vide.

    Tolérance : un ``<canvas>`` créé dynamiquement côté JS sans id
    pré-déclaré reste possible (Chart.js peut en générer pour des
    sub-charts). Le test ne valide que les canvas que les templates
    Jinja2 produisent — pas ceux du DOM dynamique."""
    html = _strip_inline_scripts(demo_html)
    canvases = re.findall(r"<canvas[^>]*>", html)
    chart_canvases = [
        c for c in canvases
        if 'id="chart-' in c or 'id="pareto-chart"' in c
    ]
    canvases_no_label = [
        c for c in chart_canvases
        if 'aria-label="' not in c and "data-a11y-label" not in c
    ]
    assert not canvases_no_label, (
        f"Canvas Chart.js sans aria-label : {canvases_no_label}"
    )


def test_canvases_have_role_img(demo_html: str) -> None:
    """``role="img"`` doit être posé sur les canvas pour les annoncer
    comme images aux AT."""
    canvases = re.findall(r"<canvas[^>]*>", demo_html)
    chart_canvases = [c for c in canvases if "chart-" in c]
    if not chart_canvases:
        pytest.skip("Aucun canvas Chart.js dans le rapport démo")
    canvases_no_role = [c for c in chart_canvases if 'role="img"' not in c]
    assert not canvases_no_role, (
        f"Canvas Chart.js sans role=img : {canvases_no_role[:3]}"
    )


def test_data_table_helpers_present(demo_html: str) -> None:
    """La fonction ``attachChartA11y`` qui génère les tables jumelles
    doit être incluse dans le JS embarqué."""
    assert "attachChartA11y" in demo_html
    assert "_populateChartDataTable" in demo_html


def test_view_data_button_label_localized(
    demo_html: str, demo_html_en: str
) -> None:
    """Les libellés du bouton « Voir les données » doivent être dans
    l'objet I18N côté JS (pas hardcodés en français)."""
    assert "Voir les données" in demo_html
    assert "View data" in demo_html_en


# ---------------------------------------------------------------------------
# m-4 — scope="col" sur les <th>
# ---------------------------------------------------------------------------


def _strip_inline_scripts(html: str) -> str:
    """Retire les blocs ``<script>...</script>`` et ``<style>...</style>``
    avant d'analyser les balises HTML.

    Nécessaire car Chart.js minifié contient des chaînes comme
    ``<this._cachedMeta`` qui matchent le regex ``<th[\\s>]`` faussement
    (sequence ``<t`` + word boundary). On limite l'analyse au HTML rendu
    par les templates Jinja2, pas au JS embarqué.
    """
    cleaned = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    cleaned = re.sub(r"<style\b[^>]*>.*?</style>", "", cleaned, flags=re.DOTALL)
    return cleaned


def test_table_headers_have_scope(demo_html: str) -> None:
    """Tout ``<th>`` rendu par les templates doit avoir ``scope="col"``
    ou ``scope="row"``."""
    html = _strip_inline_scripts(demo_html)
    # Regex strict : <th suivi d'un espace ou >, qui n'a PAS d'attribut scope=
    th_no_scope = re.findall(
        r"<th(?:\s+(?![^>]*\bscope=)[^>]*)?>",
        html,
    )
    # On filtre faux positifs : <thead, <tbody, <tfoot etc. ne doivent pas matcher.
    th_no_scope = [t for t in th_no_scope if re.match(r"<th(\s|>)", t)]
    total_th = len(re.findall(r"<th(\s|>)", html))
    if total_th == 0:
        pytest.skip("Pas de <th> dans le rapport démo")
    assert not th_no_scope, (
        f"{len(th_no_scope)}/{total_th} <th> sans scope= "
        f"dans le HTML rendu (hors <script>/<style>). "
        f"Premiers : {th_no_scope[:3]}"
    )


# ---------------------------------------------------------------------------
# m-3 — Bouton Reset i18n
# ---------------------------------------------------------------------------


def test_reset_button_uses_i18n_key(demo_html: str) -> None:
    """Le bouton « Réinitialiser » du bandeau d'exclusion doit avoir
    ``data-i18n="reset_all"`` (pas de chaîne FR hardcodée sans
    mécanisme i18n)."""
    # Le bouton apparaît avec data-i18n="reset_all"
    assert 'data-i18n="reset_all"' in demo_html


def test_reset_label_in_i18n_dicts(demo_html: str, demo_html_en: str) -> None:
    """Les clés ``reset_all`` doivent exister dans les deux
    dictionnaires i18n embarqués."""
    # Le JSON I18N est embarqué inline dans le HTML.
    # On cherche un fragment JSON ``"reset_all":"..."``
    assert re.search(r'"reset_all"\s*:\s*"R[ée]initialiser"', demo_html)
    assert re.search(r'"reset_all"\s*:\s*"Reset"', demo_html_en)


# ---------------------------------------------------------------------------
# Synthèse
# ---------------------------------------------------------------------------


def test_html_has_lang_attribute(demo_html: str, demo_html_en: str) -> None:
    """``<html lang="...">`` doit être posé pour les AT (déjà cas mais
    on renforce)."""
    assert 'lang="fr"' in demo_html
    assert 'lang="en"' in demo_html_en


def test_global_a11y_smoke(demo_html: str) -> None:
    """Méta-test : tous les marqueurs a11y de niveau A sont présents
    dans un rapport démo standard."""
    markers = [
        'class="skip-link"',
        'href="#main"',
        'id="main"',
        'role="img"',
        "attachChartA11y",
        'scope="col"',
        'data-i18n="reset_all"',
    ]
    missing = [m for m in markers if m not in demo_html]
    assert not missing, f"Marqueurs WCAG niveau A manquants : {missing}"
