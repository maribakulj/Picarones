"""Tests Sprint 17 — refactor du générateur HTML en templates Jinja2.

Objectif : garantir que le découpage de ``_HTML_TEMPLATE`` (3100 lignes
monolithiques) en templates séparés (``base.html.j2`` + 9 partials) n'a pas
altéré la sortie du rapport. Après ce sprint, toute modification future doit
conserver ces invariants.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from picarones import fixtures
from picarones.report.generator import (
    ReportGenerator,
    _build_jinja_env,
    _TEMPLATES_DIR,
)


# ---------------------------------------------------------------------------
# Structure des fichiers attendus
# ---------------------------------------------------------------------------

EXPECTED_TEMPLATE_FILES = {
    "base.html.j2",
    "_header.html",
    "_footer.html",
    "_styles.css",
    "_app.js",
    "view_ranking.html",
    "view_gallery.html",
    "view_document.html",
    "view_analyses.html",
    "view_characters.html",
}


class TestTemplateStructure:
    def test_all_expected_template_files_exist(self):
        present = {p.name for p in _TEMPLATES_DIR.iterdir() if p.is_file()}
        missing = EXPECTED_TEMPLATE_FILES - present
        assert not missing, f"Templates manquants : {missing}"

    def test_jinja_env_can_load_base_template(self):
        env = _build_jinja_env()
        tpl = env.get_template("base.html.j2")
        assert tpl is not None

    def test_no_dangling_format_placeholders_in_templates(self):
        """Aucun {placeholder} style .format() ne doit traîner — tout doit être
        en syntaxe Jinja2 {{ variable }}."""
        suspicious_pattern = re.compile(r"(?<!\{)\{[a-z_]+\}(?!\})")
        for tpl_file in _TEMPLATES_DIR.iterdir():
            if tpl_file.suffix in (".html", ".j2", ".css"):
                content = tpl_file.read_text(encoding="utf-8")
                matches = suspicious_pattern.findall(content)
                assert not matches, (
                    f"{tpl_file.name} contient des placeholders style .format() : {matches}"
                )


# ---------------------------------------------------------------------------
# Génération et validité du rapport
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def benchmark_result():
    return fixtures.generate_sample_benchmark(n_docs=3)


class TestReportGeneration:
    def test_generate_produces_file(self, benchmark_result, tmp_path):
        out = tmp_path / "rapport.html"
        gen = ReportGenerator(benchmark_result)
        result_path = gen.generate(out)
        assert result_path.exists()
        assert result_path.stat().st_size > 10_000  # Chart.js inline à lui seul

    def test_report_contains_expected_markers(self, benchmark_result, tmp_path):
        out = tmp_path / "rapport.html"
        ReportGenerator(benchmark_result).generate(out)
        html = out.read_text(encoding="utf-8")

        # Structure HTML attendue
        assert "<!DOCTYPE html>" in html
        assert "<html lang=\"fr\">" in html
        assert "Picarones" in html
        # Les 5 vues doivent être présentes
        for view in ("view-ranking", "view-gallery", "view-document",
                     "view-analyses", "view-characters"):
            assert f'id="{view}"' in html, f"Vue '{view}' absente du rapport"
        # Données embarquées
        assert "const DATA =" in html
        assert "const I18N =" in html
        # Chart.js inline
        assert "Chart.js" in html

    def test_report_has_no_nested_script_tags(self, benchmark_result, tmp_path):
        """Un bug classique du refactor : les `<script>` dupliqués quand on
        oublie de les retirer du contenu extrait."""
        out = tmp_path / "rapport.html"
        ReportGenerator(benchmark_result).generate(out)
        html = out.read_text(encoding="utf-8")

        # Chaque bloc script doit avoir un fermeture correspondante
        opens = html.count("<script>")
        # On tolère aussi `<script type="...">` mais on n'en utilise pas actuellement
        closes = html.count("</script>")
        assert opens == closes, f"Script tags déséquilibrés : {opens} ouvertures vs {closes} fermetures"

    def test_report_deterministic_given_same_data(self, benchmark_result, tmp_path):
        """Deux générations sur le MÊME benchmark produisent du HTML identique
        (garde-fou pour le moteur narratif Sprint 4 qui doit être déterministe)."""
        out1 = tmp_path / "r1.html"
        out2 = tmp_path / "r2.html"
        ReportGenerator(benchmark_result).generate(out1)
        ReportGenerator(benchmark_result).generate(out2)
        h1 = hashlib.sha256(out1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(out2.read_bytes()).hexdigest()
        assert h1 == h2, "La génération du rapport doit être déterministe"

    def test_english_locale_renders(self, benchmark_result, tmp_path):
        out = tmp_path / "report_en.html"
        ReportGenerator(benchmark_result, lang="en").generate(out)
        html = out.read_text(encoding="utf-8")
        assert '<html lang="en">' in html


# ---------------------------------------------------------------------------
# Chargement i18n depuis JSON
# ---------------------------------------------------------------------------

class TestI18nFromJSON:
    def test_i18n_directory_exists_and_has_json(self):
        i18n_dir = Path(__file__).parent.parent / "picarones" / "report" / "i18n"
        assert i18n_dir.is_dir()
        files = {p.name for p in i18n_dir.glob("*.json")}
        assert "fr.json" in files
        assert "en.json" in files

    def test_all_i18n_files_parse_as_json(self):
        i18n_dir = Path(__file__).parent.parent / "picarones" / "report" / "i18n"
        for f in i18n_dir.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            assert isinstance(data, dict)
            assert len(data) > 50  # raisonnable : on a 101 clés

    def test_fr_and_en_have_same_keys(self):
        """Garde-fou contre les traductions manquantes."""
        from picarones.i18n import TRANSLATIONS
        fr_keys = set(TRANSLATIONS.get("fr", {}).keys())
        en_keys = set(TRANSLATIONS.get("en", {}).keys())
        missing_in_en = fr_keys - en_keys
        missing_in_fr = en_keys - fr_keys
        assert not missing_in_en, f"Clés manquantes en anglais : {missing_in_en}"
        assert not missing_in_fr, f"Clés manquantes en français : {missing_in_fr}"

    def test_translations_load_via_public_api(self):
        from picarones.i18n import get_labels, SUPPORTED_LANGS
        assert "fr" in SUPPORTED_LANGS
        assert "en" in SUPPORTED_LANGS
        fr = get_labels("fr")
        en = get_labels("en")
        assert fr["html_lang"] == "fr"
        assert en["html_lang"] == "en"
        # Fallback sur fr si langue inconnue
        assert get_labels("xx") == fr


# ---------------------------------------------------------------------------
# Validation du contenu extrait (pas de régression sur le HTML rendu)
# ---------------------------------------------------------------------------

class TestTemplateContent:
    def test_css_file_contains_expected_rules(self):
        css = (_TEMPLATES_DIR / "_styles.css").read_text(encoding="utf-8")
        # Quelques règles canoniques du rapport qui doivent rester
        for marker in ("nav", ".cer-badge", ".gallery-card", ".tab-btn"):
            assert marker in css, f"Règle CSS '{marker}' manquante"

    def test_app_js_starts_with_use_strict(self):
        js = (_TEMPLATES_DIR / "_app.js").read_text(encoding="utf-8")
        first_nonblank = next((l for l in js.splitlines() if l.strip()), "")
        assert "'use strict'" in first_nonblank

    def test_app_js_has_no_residual_script_tag(self):
        """Garde-fou contre un futur refactor qui ré-inclurait par erreur."""
        js = (_TEMPLATES_DIR / "_app.js").read_text(encoding="utf-8")
        assert "<script" not in js
        assert "</script>" not in js

    def test_view_files_contain_root_section_element(self):
        """Chaque vue HTML doit avoir un élément racine avec id='view-<nom>'."""
        view_ids = {
            "view_ranking.html": "view-ranking",
            "view_gallery.html": "view-gallery",
            "view_document.html": "view-document",
            "view_analyses.html": "view-analyses",
            "view_characters.html": "view-characters",
        }
        for fname, expected_id in view_ids.items():
            content = (_TEMPLATES_DIR / fname).read_text(encoding="utf-8")
            assert f'id="{expected_id}"' in content, (
                f"{fname} devrait contenir id='{expected_id}'"
            )
