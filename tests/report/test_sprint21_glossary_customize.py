"""Tests Sprint 21 — glossaire contextuel + panneau personnalisation.

Sprint 6 du plan rapport. Couvre :
  1. Le loader YAML du glossaire (FR/EN, cache, fallback).
  2. La complétude des entrées (chaque langue a les mêmes clés que FR).
  3. La structure des entrées (chaque entrée a definition/measures/usage…).
  4. L'intégration au rapport HTML : GLOSSARY embarqué, boutons `?` injectés,
     panneaux latéraux présents, bouton "Avancé".
  5. Les libellés i18n des deux nouveaux groupes (glossary_*, customize_*).
"""

from __future__ import annotations

import json
import re

import pytest

from picarones.reports_v2.glossary import SUPPORTED_LANGS, load_glossary


# ---------------------------------------------------------------------------
# 1. Loader
# ---------------------------------------------------------------------------

class TestLoadGlossary:
    def test_default_lang_loads_with_entries(self):
        g = load_glossary("fr")
        assert isinstance(g, dict)
        assert len(g) >= 20  # objectif Sprint 6 : 25 entrées
        for key, entry in g.items():
            assert isinstance(entry, dict), f"Entrée {key} doit être un dict"

    def test_english_loads(self):
        g = load_glossary("en")
        assert len(g) >= 20

    def test_unknown_lang_falls_back_to_fr(self):
        g_fr = load_glossary("fr")
        g_xx = load_glossary("xx")
        assert g_xx == g_fr

    def test_cache_is_used(self):
        # Deux appels successifs renvoient le MÊME objet (cache hit)
        a = load_glossary("fr")
        b = load_glossary("fr")
        assert a is b

    def test_supported_langs_includes_fr_and_en(self):
        assert "fr" in SUPPORTED_LANGS
        assert "en" in SUPPORTED_LANGS


# ---------------------------------------------------------------------------
# 2. Complétude FR/EN
# ---------------------------------------------------------------------------

class TestGlossaryCompleteness:
    def test_fr_and_en_have_same_keys(self):
        fr = set(load_glossary("fr").keys())
        en = set(load_glossary("en").keys())
        missing_in_en = fr - en
        missing_in_fr = en - fr
        assert not missing_in_en, f"Entrées manquantes en anglais : {sorted(missing_in_en)}"
        assert not missing_in_fr, f"Entrées manquantes en français : {sorted(missing_in_fr)}"

    def test_each_entry_has_required_fields(self):
        required = {"title", "definition", "measures", "usage", "limits", "reference"}
        for lang in ("fr", "en"):
            entries = load_glossary(lang)
            for key, entry in entries.items():
                missing = required - set(entry.keys())
                assert not missing, (
                    f"Entrée {lang}/{key} manque {missing}"
                )
                for f in required:
                    assert isinstance(entry[f], str)
                    assert entry[f].strip(), f"Entrée {lang}/{key}.{f} vide"

    def test_critical_terms_are_documented(self):
        """Garde-fou : les métriques affichées en colonne du classement
        doivent toutes être documentées."""
        critical = {
            "cer", "cer_diplomatic", "wer", "mer", "wil",
            "ligature_score", "diacritic_score", "gini", "anchor_score",
            "bootstrap_ci", "wilcoxon", "friedman", "nemenyi", "cdd",
            "pareto_front", "hallucination_score",
        }
        fr_keys = set(load_glossary("fr").keys())
        missing = critical - fr_keys
        assert not missing, f"Termes critiques absents du glossaire : {missing}"


# ---------------------------------------------------------------------------
# 3. Structure des entrées
# ---------------------------------------------------------------------------

class TestEntryStructure:
    @pytest.mark.parametrize("lang", ["fr", "en"])
    def test_definitions_have_reasonable_length(self, lang):
        entries = load_glossary(lang)
        for key, entry in entries.items():
            d = entry["definition"]
            # 2-3 phrases attendues — longueur typique 80-400 caractères
            assert 30 <= len(d) <= 1000, (
                f"{lang}/{key}.definition longueur inhabituelle ({len(d)})"
            )

    @pytest.mark.parametrize("lang", ["fr", "en"])
    def test_no_html_tags_in_text(self, lang):
        """Le contenu est rendu dans une `<p>` via textContent côté JS — il
        ne doit pas contenir de HTML qui serait étiqueté littéralement."""
        entries = load_glossary(lang)
        html_re = re.compile(r"<[a-z]+[\s/>]", re.IGNORECASE)
        for key, entry in entries.items():
            for f in ("definition", "measures", "usage", "limits", "reference"):
                assert not html_re.search(entry[f]), (
                    f"{lang}/{key}.{f} contient du HTML : {entry[f][:80]}"
                )


# ---------------------------------------------------------------------------
# 4. Intégration au rapport HTML
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def benchmark_result():
    from picarones import fixtures
    return fixtures.generate_sample_benchmark(n_docs=5)


class TestReportIntegration:
    def test_report_embeds_glossary_json(self, benchmark_result, tmp_path):
        from picarones.reports_v2.html.generator import ReportGenerator
        out = tmp_path / "report.html"
        ReportGenerator(benchmark_result).generate(out)
        html = out.read_text(encoding="utf-8")
        m = re.search(r"const GLOSSARY = (\{.*?\});\s*\n", html)
        assert m, "GLOSSARY const non trouvé"
        glossary = json.loads(m.group(1))
        assert "cer" in glossary
        assert "definition" in glossary["cer"]

    def test_report_contains_side_panels(self, benchmark_result, tmp_path):
        from picarones.reports_v2.html.generator import ReportGenerator
        out = tmp_path / "report.html"
        ReportGenerator(benchmark_result).generate(out)
        html = out.read_text(encoding="utf-8")
        assert 'id="glossary-panel"' in html
        assert 'id="customize-panel"' in html
        assert 'class="side-panel-close"' in html

    def test_report_has_advanced_button_in_nav(self, benchmark_result, tmp_path):
        from picarones.reports_v2.html.generator import ReportGenerator
        out = tmp_path / "report.html"
        ReportGenerator(benchmark_result).generate(out)
        html = out.read_text(encoding="utf-8")
        assert 'btn-customize' in html
        assert 'openCustomize()' in html

    def test_ranking_columns_have_glossary_keys(self, benchmark_result, tmp_path):
        from picarones.reports_v2.html.generator import ReportGenerator
        out = tmp_path / "report.html"
        ReportGenerator(benchmark_result).generate(out)
        html = out.read_text(encoding="utf-8")
        for k in ("cer", "wer", "ligature_score", "anchor_score"):
            assert f'data-glossary-key="{k}"' in html, f"Header pour {k} sans data-glossary-key"

    def test_app_js_has_glossary_and_customize_functions(self, benchmark_result, tmp_path):
        from picarones.reports_v2.html.generator import ReportGenerator
        out = tmp_path / "report.html"
        ReportGenerator(benchmark_result).generate(out)
        html = out.read_text(encoding="utf-8")
        for fn in (
            "function openGlossary",
            "function injectGlossaryButtons",
            "function openCustomize",
            "function applyCompositeScore",
            "function restoreCustomFromURL",
            "function resetCustomization",
        ):
            assert fn in html, f"Fonction {fn} manquante"

    def test_english_glossary_for_en_locale(self, benchmark_result, tmp_path):
        from picarones.reports_v2.html.generator import ReportGenerator
        out = tmp_path / "report_en.html"
        ReportGenerator(benchmark_result, lang="en").generate(out)
        html = out.read_text(encoding="utf-8")
        m = re.search(r"const GLOSSARY = (\{.*?\});\s*\n", html)
        glossary = json.loads(m.group(1))
        # L'entrée CER doit être en anglais
        assert "Character Error Rate" in glossary["cer"]["title"]


# ---------------------------------------------------------------------------
# 5. i18n
# ---------------------------------------------------------------------------

class TestI18nKeysForCustomize:
    def test_required_customize_keys_present(self):
        from picarones.reports_v2.i18n import get_labels
        required = {
            "btn_customize", "customize_title",
            "customize_columns", "customize_filters",
            "customize_weights", "customize_weights_warning",
            "customize_weights_enable", "customize_weights_disable",
            "customize_reset",
            "glossary_definition", "glossary_measures", "glossary_usage",
            "glossary_limits", "glossary_reference",
        }
        for lang in ("fr", "en"):
            labels = get_labels(lang)
            missing = required - set(labels.keys())
            assert not missing, f"i18n {lang} manque {missing}"


# ---------------------------------------------------------------------------
# 6. Garde-fou anti-prescription
# ---------------------------------------------------------------------------

class TestNoPrescriptionGuards:
    """Vérifie que le panneau "Mode avancé" expose bien le warning explicite
    et que les poids de score composite sont à 0 par défaut côté JS."""

    def test_warning_message_is_visible(self, tmp_path):
        from picarones.reports_v2.html.generator import ReportGenerator
        from picarones import fixtures
        bench = fixtures.generate_sample_benchmark(n_docs=3)
        out = tmp_path / "r.html"
        ReportGenerator(bench).generate(out)
        html = out.read_text(encoding="utf-8")
        # FR par défaut : "Il n'existe pas de pondération universellement valide"
        assert "pondération universellement valide" in html or \
               "universally valid weighting" in html

    def test_default_weights_are_empty_in_js_state(self, tmp_path):
        from picarones.reports_v2.html.generator import ReportGenerator
        from picarones import fixtures
        bench = fixtures.generate_sample_benchmark(n_docs=3)
        out = tmp_path / "r.html"
        ReportGenerator(bench).generate(out)
        html = out.read_text(encoding="utf-8")
        # _CUSTOM_STATE initial doit avoir weights: {} et weightsEnabled: false
        assert "weightsEnabled: false" in html
        assert "weights: {}" in html
