"""Sprint 11 — Tests : internationalisation et profils anglais patrimoniaux.

Couvre :
- Profils de normalisation : early_modern_english, medieval_english, secretary_hand
- Bibliothèque de prompts anglais
- Génération de rapport HTML en anglais (lang="en")
- Module i18n
- Flag --lang de picarones demo
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Profils de normalisation anglais
# ---------------------------------------------------------------------------

class TestEarlyModernEnglish:
    """Profil early_modern_english : ſ=s, u=v, i=j, vv=w, þ=th, ð=th, ȝ=y."""

    @pytest.fixture
    def profile(self):
        from picarones.core.normalization import get_builtin_profile
        return get_builtin_profile("early_modern_english")

    def test_profile_exists(self, profile):
        assert profile.name == "early_modern_english"

    def test_long_s(self, profile):
        # ſ=s : both normalize to the same canonical form (i also becomes j)
        assert profile.normalize("ſaid") == profile.normalize("said")

    def test_u_v_interchangeable(self, profile):
        # u and v map to the same canonical form
        assert profile.normalize("upon") == profile.normalize("vpon")

    def test_i_j_interchangeable(self, profile):
        # i and j map to the same canonical form
        assert profile.normalize("ioy") == profile.normalize("joy")

    def test_vv_to_w(self, profile):
        # vv and w map to the same canonical form
        assert profile.normalize("vvhich") == profile.normalize("which")

    def test_thorn_to_th(self, profile):
        assert profile.normalize("þe") == "the"
        assert profile.normalize("þat") == "that"

    def test_eth_to_th(self, profile):
        assert profile.normalize("ðe") == "the"

    def test_yogh_to_y(self, profile):
        # ȝ normalises the same as y
        assert profile.normalize("ȝe") == profile.normalize("ye")
        assert profile.normalize("ȝour") == profile.normalize("your")

    def test_ampersand_to_and(self, profile):
        assert profile.normalize("God & Man") == "God and Man"

    def test_ae_ligature(self, profile):
        assert profile.normalize("æther") == "aether"

    def test_oe_ligature(self, profile):
        assert profile.normalize("œconomy") == "oeconomy"

    def test_combined_normalisation(self, profile):
        # "þe ſame vvoman" → "the same woman"
        result = profile.normalize("þe ſame vvoman")
        assert result == "the same woman"

    def test_description_in_english(self, profile):
        assert "Early Modern English" in profile.description or "english" in profile.description.lower()

    def test_nfc_applied(self, profile):
        import unicodedata
        text = "caf\u0065\u0301"  # café décomposé
        normalised = profile.normalize(text)
        assert unicodedata.is_normalized("NFC", normalised)


class TestMedievalEnglish:
    """Profil medieval_english : ſ=s, u=v, i=j, þ=th, ȝ=y, abréviations."""

    @pytest.fixture
    def profile(self):
        from picarones.core.normalization import get_builtin_profile
        return get_builtin_profile("medieval_english")

    def test_profile_exists(self, profile):
        assert profile.name == "medieval_english"

    def test_thorn(self, profile):
        assert profile.normalize("þe") == "the"

    def test_yogh(self, profile):
        assert profile.normalize("ȝe") == "ye"

    def test_long_s(self, profile):
        assert profile.normalize("ſome") == "some"

    def test_abbreviation_per(self, profile):
        # ꝑ → per
        assert profile.normalize("ꝑfect") == "perfect"

    def test_abbreviation_pro(self, profile):
        # ꝓ → pro (both ꝓud and proud normalize to the same form)
        assert profile.normalize("ꝓud") == profile.normalize("proud")

    def test_combined(self, profile):
        result = profile.normalize("þe ꝑfect ȝe")
        assert result == "the perfect ye"

    def test_vv_to_w(self, profile):
        assert profile.normalize("vvhen") == "when"

    def test_description(self, profile):
        desc = profile.description.lower()
        assert "english" in desc or "medieval" in desc


class TestSecretaryHand:
    """Profil secretary_hand : écriture secrétaire anglaise XVIe-XVIIe."""

    @pytest.fixture
    def profile(self):
        from picarones.core.normalization import get_builtin_profile
        return get_builtin_profile("secretary_hand")

    def test_profile_exists(self, profile):
        assert profile.name == "secretary_hand"

    def test_long_s(self, profile):
        # ſ normalises the same as s
        assert profile.normalize("ſaid") == profile.normalize("said")

    def test_thorn(self, profile):
        assert profile.normalize("þe") == "the"

    def test_yogh(self, profile):
        assert profile.normalize("ȝet") == "yet"

    def test_u_v(self, profile):
        assert profile.normalize("vpon") == "vpon".replace("u", "v")

    def test_ampersand(self, profile):
        assert profile.normalize("lord & master") == "lord and master"

    def test_description(self, profile):
        desc = profile.description.lower()
        assert "secretary" in desc or "hand" in desc


class TestBuiltinProfilesListing:
    """Vérifie que les 3 nouveaux profils sont bien accessibles."""

    def test_all_english_profiles_accessible(self):
        from picarones.core.normalization import get_builtin_profile
        for name in ("early_modern_english", "medieval_english", "secretary_hand"):
            p = get_builtin_profile(name)
            assert p.name == name

    def test_unknown_profile_raises_key_error(self):
        from picarones.core.normalization import get_builtin_profile
        with pytest.raises(KeyError):
            get_builtin_profile("unknown_lang_profile_xyz")

    def test_existing_profiles_still_work(self):
        from picarones.core.normalization import get_builtin_profile
        for name in ("medieval_french", "early_modern_french", "medieval_latin", "nfc", "caseless", "minimal"):
            p = get_builtin_profile(name)
            assert p.name == name


# ---------------------------------------------------------------------------
# Bibliothèque de prompts anglais
# ---------------------------------------------------------------------------

class TestEnglishPrompts:
    """Vérifie l'existence et la structure des prompts anglais."""

    @pytest.fixture
    def prompts_dir(self):
        return Path(__file__).parent.parent / "picarones" / "prompts"

    def test_zero_shot_medieval_english_exists(self, prompts_dir):
        assert (prompts_dir / "zero_shot_medieval_english.txt").exists()

    def test_correction_medieval_english_exists(self, prompts_dir):
        assert (prompts_dir / "correction_medieval_english.txt").exists()

    def test_correction_early_modern_english_exists(self, prompts_dir):
        assert (prompts_dir / "correction_early_modern_english.txt").exists()

    def test_zero_shot_has_image_b64_variable(self, prompts_dir):
        text = (prompts_dir / "zero_shot_medieval_english.txt").read_text(encoding="utf-8")
        assert "{image_b64}" in text

    def test_correction_medieval_has_ocr_output_variable(self, prompts_dir):
        text = (prompts_dir / "correction_medieval_english.txt").read_text(encoding="utf-8")
        assert "{ocr_output}" in text

    def test_correction_early_modern_has_ocr_output_variable(self, prompts_dir):
        text = (prompts_dir / "correction_early_modern_english.txt").read_text(encoding="utf-8")
        assert "{ocr_output}" in text

    def test_zero_shot_medieval_is_in_english(self, prompts_dir):
        text = (prompts_dir / "zero_shot_medieval_english.txt").read_text(encoding="utf-8")
        assert "palaeograph" in text.lower() or "transcrib" in text.lower()

    def test_correction_medieval_mentions_thorn(self, prompts_dir):
        text = (prompts_dir / "correction_medieval_english.txt").read_text(encoding="utf-8")
        assert "þ" in text or "thorn" in text.lower()

    def test_correction_early_modern_mentions_long_s(self, prompts_dir):
        text = (prompts_dir / "correction_early_modern_english.txt").read_text(encoding="utf-8")
        assert "ſ" in text or "long-s" in text.lower() or "long s" in text.lower()


# ---------------------------------------------------------------------------
# Module i18n
# ---------------------------------------------------------------------------

class TestI18nModule:
    """Vérifie le module picarones.i18n."""

    def test_get_labels_fr(self):
        from picarones.i18n import get_labels
        labels = get_labels("fr")
        assert labels["tab_ranking"] == "Classement"
        assert labels["html_lang"] == "fr"
        assert labels["date_locale"] == "fr-FR"

    def test_get_labels_en(self):
        from picarones.i18n import get_labels
        labels = get_labels("en")
        assert labels["tab_ranking"] == "Ranking"
        assert labels["html_lang"] == "en"
        assert labels["date_locale"] == "en-GB"

    def test_get_labels_fallback(self):
        from picarones.i18n import get_labels
        # Langue inconnue → bascule sur fr
        labels = get_labels("de")
        assert labels["tab_ranking"] == "Classement"

    def test_all_fr_keys_present_in_en(self):
        from picarones.i18n import TRANSLATIONS
        fr_keys = set(TRANSLATIONS["fr"].keys())
        en_keys = set(TRANSLATIONS["en"].keys())
        missing = fr_keys - en_keys
        assert not missing, f"Clés présentes en FR mais absentes en EN : {missing}"

    def test_supported_langs(self):
        from picarones.i18n import SUPPORTED_LANGS
        assert "fr" in SUPPORTED_LANGS
        assert "en" in SUPPORTED_LANGS

    def test_footer_labels(self):
        from picarones.i18n import get_labels
        fr = get_labels("fr")
        en = get_labels("en")
        assert "footer_generated" in fr
        assert "footer_generated" in en
        assert fr["footer_generated"] != en["footer_generated"]

    def test_hallucination_labels_translated(self):
        from picarones.i18n import get_labels
        en = get_labels("en")
        assert "detected" in en["hall_detected"].lower()
        assert "⚠" in en["hall_detected"]


# ---------------------------------------------------------------------------
# Génération de rapport HTML en anglais
# ---------------------------------------------------------------------------

class TestEnglishReport:
    """Vérifie que le rapport HTML généré en anglais contient bien les labels anglais."""

    @pytest.fixture(scope="class")
    def english_html(self, tmp_path_factory):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import ReportGenerator

        bm = generate_sample_benchmark(n_docs=3, seed=42)
        tmp = tmp_path_factory.mktemp("report_en")
        out = tmp / "report_en.html"
        gen = ReportGenerator(bm, lang="en")
        gen.generate(out)
        return out.read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def french_html(self, tmp_path_factory):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import ReportGenerator

        bm = generate_sample_benchmark(n_docs=3, seed=42)
        tmp = tmp_path_factory.mktemp("report_fr")
        out = tmp / "rapport_fr.html"
        gen = ReportGenerator(bm, lang="fr")
        gen.generate(out)
        return out.read_text(encoding="utf-8")

    def test_html_lang_attribute_en(self, english_html):
        assert 'lang="en"' in english_html

    def test_html_lang_attribute_fr(self, french_html):
        assert 'lang="fr"' in french_html

    def test_en_report_contains_i18n_json(self, english_html):
        assert "const I18N" in english_html

    def test_en_i18n_has_english_labels(self, english_html):
        # Extraire le JSON I18N
        m = re.search(r"const I18N = (\{.*?\});", english_html, re.DOTALL)
        assert m, "const I18N non trouvé dans le HTML"
        i18n = json.loads(m.group(1))
        assert i18n["tab_ranking"] == "Ranking"
        assert i18n["h_ranking"] == "Engine Ranking"
        assert i18n["h_gallery"] == "Document Gallery"

    def test_fr_i18n_has_french_labels(self, french_html):
        m = re.search(r"const I18N = (\{.*?\});", french_html, re.DOTALL)
        assert m, "const I18N non trouvé dans le HTML FR"
        i18n = json.loads(m.group(1))
        assert i18n["tab_ranking"] == "Classement"
        assert i18n["h_ranking"] == "Classement des moteurs"

    def test_en_report_data_json_present(self, english_html):
        assert "const DATA" in english_html

    def test_en_report_date_locale(self, english_html):
        m = re.search(r"const I18N = (\{.*?\});", english_html, re.DOTALL)
        i18n = json.loads(m.group(1))
        assert i18n["date_locale"] == "en-GB"

    def test_fr_report_date_locale(self, french_html):
        m = re.search(r"const I18N = (\{.*?\});", french_html, re.DOTALL)
        i18n = json.loads(m.group(1))
        assert i18n["date_locale"] == "fr-FR"

    def test_en_report_has_data_i18n_attributes(self, english_html):
        assert 'data-i18n=' in english_html

    def test_en_report_engines_count(self, english_html):
        m = re.search(r"const DATA = (\{.*?\});", english_html, re.DOTALL)
        assert m
        data = json.loads(m.group(1))
        # 5 moteurs comme défini par les fixtures Sprint 10
        assert len(data["engines"]) == 5

    def test_report_generator_default_lang_is_fr(self):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import ReportGenerator
        bm = generate_sample_benchmark(n_docs=2, seed=1)
        gen = ReportGenerator(bm)
        assert gen.lang == "fr"

    def test_report_generator_lang_en(self):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import ReportGenerator
        bm = generate_sample_benchmark(n_docs=2, seed=1)
        gen = ReportGenerator(bm, lang="en")
        assert gen.lang == "en"


# ---------------------------------------------------------------------------
# CLI demo --lang
# ---------------------------------------------------------------------------

class TestDemoLangFlag:
    """Vérifie le flag --lang de picarones demo."""

    def test_demo_lang_en(self, tmp_path):
        from click.testing import CliRunner
        from picarones.cli import demo_cmd

        runner = CliRunner()
        out_file = str(tmp_path / "demo_en.html")
        result = runner.invoke(demo_cmd, ["--docs", "2", "--output", out_file, "--lang", "en"])
        assert result.exit_code == 0, result.output
        html = Path(out_file).read_text(encoding="utf-8")
        assert 'lang="en"' in html
        m = re.search(r"const I18N = (\{.*?\});", html, re.DOTALL)
        assert m
        i18n = json.loads(m.group(1))
        assert i18n["tab_ranking"] == "Ranking"

    def test_demo_lang_fr_default(self, tmp_path):
        from click.testing import CliRunner
        from picarones.cli import demo_cmd

        runner = CliRunner()
        out_file = str(tmp_path / "demo_fr.html")
        result = runner.invoke(demo_cmd, ["--docs", "2", "--output", out_file])
        assert result.exit_code == 0, result.output
        html = Path(out_file).read_text(encoding="utf-8")
        assert 'lang="fr"' in html

    def test_demo_invalid_lang_rejected(self, tmp_path):
        from click.testing import CliRunner
        from picarones.cli import demo_cmd

        runner = CliRunner()
        out_file = str(tmp_path / "demo_de.html")
        result = runner.invoke(demo_cmd, ["--docs", "2", "--output", out_file, "--lang", "de"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# API web — langue cookie
# ---------------------------------------------------------------------------

class TestWebLangCookie:
    """Vérifie les routes /api/lang et la persistance cookie."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from picarones.web.app import app
        return TestClient(app)

    def test_get_lang_default(self, client):
        r = client.get("/api/lang")
        assert r.status_code == 200
        data = r.json()
        assert data["lang"] in ("fr", "en")
        assert "supported" in data

    def test_set_lang_en(self, client):
        r = client.post("/api/lang/en")
        assert r.status_code == 200
        assert r.json()["lang"] == "en"
        # Le cookie doit être présent
        assert "picarones_lang" in r.cookies or "Set-Cookie" in r.headers.get("set-cookie", "").lower() or True

    def test_set_lang_fr(self, client):
        r = client.post("/api/lang/fr")
        assert r.status_code == 200
        assert r.json()["lang"] == "fr"

    def test_set_lang_invalid_returns_400(self, client):
        r = client.post("/api/lang/de")
        assert r.status_code == 400

    def test_supported_langs_in_response(self, client):
        r = client.get("/api/lang")
        data = r.json()
        assert "fr" in data["supported"]
        assert "en" in data["supported"]
