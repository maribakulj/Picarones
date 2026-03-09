"""Tests Sprint 4 : normalisation diplomatique, import IIIF, adaptateurs API OCR."""

from __future__ import annotations

import json
import os
import pytest

from picarones.core.normalization import (
    NormalizationProfile,
    DIPLOMATIC_FR_MEDIEVAL,
    DIPLOMATIC_FR_EARLY_MODERN,
    DIPLOMATIC_LATIN_MEDIEVAL,
    DIPLOMATIC_MINIMAL,
    DEFAULT_DIPLOMATIC_PROFILE,
    _apply_diplomatic_table,
    get_builtin_profile,
)
from picarones.core.metrics import compute_metrics, aggregate_metrics, MetricsResult
from picarones.importers.iiif import (
    IIIFManifestParser,
    IIIFCanvas,
    parse_page_selector,
    _extract_label,
    _best_image_url_v2,
    _best_image_url_v3,
    _guess_extension,
    _slugify,
)


# ===========================================================================
# Tests NormalizationProfile
# ===========================================================================

class TestNormalizationProfile:

    def test_default_nfc_only(self):
        profile = NormalizationProfile(name="test")
        assert profile.nfc is True
        assert profile.caseless is False
        assert profile.diplomatic_table == {}

    def test_normalize_nfc(self):
        profile = NormalizationProfile(name="nfc_only")
        # NFD vs NFC : après NFC, les deux doivent être identiques
        decomposed = "e\u0301"  # e + accent
        assert profile.normalize(decomposed) == "\u00e9"  # é NFC

    def test_normalize_caseless(self):
        profile = NormalizationProfile(name="caseless", caseless=True)
        assert profile.normalize("Bonjour MONDE") == "bonjour monde"

    def test_normalize_diplomatic_table(self):
        profile = NormalizationProfile(
            name="test",
            diplomatic_table={"ſ": "s", "u": "v"}
        )
        # "maiſon": ſ→s gives "maison", no u present → "maison"
        assert profile.normalize("maiſon") == "maison"
        # "uers" (vers ancien): u→v gives "vers"
        assert profile.normalize("uers") == "vers"

    def test_normalize_order_nfc_then_caseless_then_diplomatic(self):
        """L'ordre est : NFC → caseless → table diplomatique."""
        profile = NormalizationProfile(
            name="combined",
            caseless=True,
            diplomatic_table={"ſ": "s"}
        )
        result = profile.normalize("Maiſon")
        assert result == "maison"

    def test_as_dict(self):
        profile = NormalizationProfile(
            name="medieval_french",
            nfc=True,
            caseless=False,
            diplomatic_table={"ſ": "s"},
            description="Test",
        )
        d = profile.as_dict()
        assert d["name"] == "medieval_french"
        assert d["diplomatic_table"] == {"ſ": "s"}
        assert d["caseless"] is False

    def test_from_dict(self):
        data = {
            "name": "custom",
            "caseless": True,
            "diplomatic": {"ſ": "s", "u": "v"},
            "description": "Custom profile",
        }
        profile = NormalizationProfile.from_dict(data)
        assert profile.name == "custom"
        assert profile.caseless is True
        assert profile.diplomatic_table == {"ſ": "s", "u": "v"}

    def test_from_dict_defaults(self):
        profile = NormalizationProfile.from_dict({})
        assert profile.name == "custom"
        assert profile.nfc is True
        assert profile.caseless is False

    def test_from_yaml(self, tmp_path):
        yaml_content = "name: my_profile\ncaseless: false\ndiplomatic:\n  \u017f: s\n  u: v\n"
        yaml_file = tmp_path / "profile.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")
        try:
            profile = NormalizationProfile.from_yaml(yaml_file)
            assert profile.name == "my_profile"
            assert profile.diplomatic_table == {"\u017f": "s", "u": "v"}
        except RuntimeError as e:
            if "pyyaml" in str(e):
                pytest.skip("pyyaml non installé")
            raise


class TestApplyDiplomaticTable:

    def test_simple_substitutions(self):
        table = {"ſ": "s", "u": "v"}
        # "maiſon": ſ→s gives "maison"; no u → "maison"
        assert _apply_diplomatic_table("maiſon", table) == "maison"
        # "uers": u→v gives "vers"
        assert _apply_diplomatic_table("uers", table) == "vers"

    def test_multi_char_key_priority(self):
        """Les clés multi-chars sont appliquées avant les clés simples."""
        table = {"ae": "X", "a": "Y"}
        # "ae" doit être remplacé en "X" et non "Ye"
        result = _apply_diplomatic_table("aeb", table)
        assert result == "Xb"

    def test_ampersand_to_et(self):
        table = {"&": "et"}
        assert _apply_diplomatic_table("noir & blanc", table) == "noir et blanc"

    def test_empty_table(self):
        assert _apply_diplomatic_table("hello", {}) == "hello"

    def test_empty_text(self):
        assert _apply_diplomatic_table("", {"a": "b"}) == ""


class TestGetBuiltinProfile:

    def test_medieval_french(self):
        profile = get_builtin_profile("medieval_french")
        assert profile.name == "medieval_french"
        assert "ſ" in profile.diplomatic_table
        assert profile.diplomatic_table["ſ"] == "s"

    def test_early_modern_french(self):
        profile = get_builtin_profile("early_modern_french")
        assert "ſ" in profile.diplomatic_table

    def test_medieval_latin(self):
        profile = get_builtin_profile("medieval_latin")
        assert "ꝑ" in profile.diplomatic_table

    def test_minimal(self):
        profile = get_builtin_profile("minimal")
        assert "ſ" in profile.diplomatic_table
        assert "u" not in profile.diplomatic_table

    def test_nfc(self):
        profile = get_builtin_profile("nfc")
        assert profile.nfc is True
        assert profile.diplomatic_table == {}

    def test_caseless(self):
        profile = get_builtin_profile("caseless")
        assert profile.caseless is True

    def test_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="inexistant"):
            get_builtin_profile("inexistant")

    def test_default_profile_is_medieval_french(self):
        assert DEFAULT_DIPLOMATIC_PROFILE.name == "medieval_french"


# ===========================================================================
# Tests CER diplomatique dans compute_metrics
# ===========================================================================

class TestDiplomaticCER:

    def test_cer_diplomatic_computed_by_default(self):
        """Le CER diplomatique est calculé par défaut avec le profil médiéval."""
        result = compute_metrics("maiſon", "maison")
        assert result.cer_diplomatic is not None
        assert result.diplomatic_profile_name == "medieval_french"

    def test_cer_diplomatic_lower_than_exact_for_long_s(self):
        """
        Avec ſ→s : le CER diplomatique doit être 0.0 pour "maiſon" vs "maison"
        car après normalisation les deux deviennent "maivon" ou "maison".
        """
        # "maiſon" vs "maison" — différence uniquement sur ſ vs s
        result = compute_metrics("maiſon", "maison")
        # CER brut > 0 (ſ ≠ s, deux bytes UTF-8 vs un)
        assert result.cer > 0.0
        # CER diplomatique = 0 car ſ et s sont équivalents dans le profil médiéval
        assert result.cer_diplomatic == pytest.approx(0.0)

    def test_cer_diplomatic_in_as_dict(self):
        result = compute_metrics("maiſon", "maison")
        d = result.as_dict()
        assert "cer_diplomatic" in d
        assert "diplomatic_profile_name" in d

    def test_cer_diplomatic_with_custom_profile(self):
        from picarones.core.normalization import NormalizationProfile
        profile = NormalizationProfile(
            name="test_profile",
            diplomatic_table={"ſ": "s"}
        )
        result = compute_metrics("maiſon", "maison", normalization_profile=profile)
        assert result.cer_diplomatic == pytest.approx(0.0)
        assert result.diplomatic_profile_name == "test_profile"

    def test_cer_diplomatic_not_in_as_dict_when_none(self):
        """Si le CER diplomatique n'a pas pu être calculé, il n'est pas dans as_dict."""
        result = MetricsResult(
            cer=0.1, cer_nfc=0.1, cer_caseless=0.1,
            wer=0.1, wer_normalized=0.1, mer=0.1, wil=0.1,
            reference_length=10, hypothesis_length=10,
            cer_diplomatic=None, diplomatic_profile_name=None,
        )
        d = result.as_dict()
        assert "cer_diplomatic" not in d

    def test_aggregate_metrics_includes_diplomatic_cer(self):
        """aggregate_metrics doit agréger cer_diplomatic quand disponible."""
        results = [
            MetricsResult(
                cer=0.1, cer_nfc=0.1, cer_caseless=0.1,
                wer=0.1, wer_normalized=0.1, mer=0.1, wil=0.1,
                reference_length=10, hypothesis_length=10,
                cer_diplomatic=0.05, diplomatic_profile_name="medieval_french",
            ),
            MetricsResult(
                cer=0.2, cer_nfc=0.2, cer_caseless=0.2,
                wer=0.2, wer_normalized=0.2, mer=0.2, wil=0.2,
                reference_length=10, hypothesis_length=10,
                cer_diplomatic=0.10, diplomatic_profile_name="medieval_french",
            ),
        ]
        agg = aggregate_metrics(results)
        assert "cer_diplomatic" in agg
        assert agg["cer_diplomatic"]["mean"] == pytest.approx(0.075)
        assert agg["cer_diplomatic"].get("profile") == "medieval_french"


# ===========================================================================
# Tests parse_page_selector
# ===========================================================================

class TestParsePageSelector:

    def test_all(self):
        assert parse_page_selector("all", 10) == list(range(10))

    def test_empty_string(self):
        assert parse_page_selector("", 5) == list(range(5))

    def test_single_page(self):
        assert parse_page_selector("3", 10) == [2]  # 0-based

    def test_range(self):
        assert parse_page_selector("1-5", 10) == [0, 1, 2, 3, 4]

    def test_comma_list(self):
        assert parse_page_selector("1,3,5", 10) == [0, 2, 4]

    def test_combined(self):
        result = parse_page_selector("1-3,5,8-9", 10)
        assert result == [0, 1, 2, 4, 7, 8]

    def test_deduplication(self):
        result = parse_page_selector("1,1,2", 5)
        assert result == [0, 1]

    def test_sorted_output(self):
        result = parse_page_selector("5,1,3", 10)
        assert result == [0, 2, 4]

    def test_page_out_of_range_raises(self):
        with pytest.raises(ValueError):
            parse_page_selector("15", 10)

    def test_range_out_of_bounds_raises(self):
        with pytest.raises(ValueError):
            parse_page_selector("1-15", 10)

    def test_invalid_syntax_raises(self):
        with pytest.raises((ValueError, Exception)):
            parse_page_selector("abc", 10)

    def test_last_page(self):
        assert parse_page_selector("10", 10) == [9]

    def test_first_page(self):
        assert parse_page_selector("1", 10) == [0]


# ===========================================================================
# Tests IIIFManifestParser — IIIF v2
# ===========================================================================

def _make_v2_manifest(num_canvases: int = 3, with_service: bool = False) -> dict:
    """Fabrique un manifeste IIIF v2 minimal de test."""
    canvases = []
    for i in range(num_canvases):
        resource: dict
        if with_service:
            resource = {
                "@type": "dctypes:Image",
                "service": {"@id": f"https://example.com/iiif/img{i+1}"},
            }
        else:
            resource = {
                "@type": "dctypes:Image",
                "@id": f"https://example.com/images/img{i+1}.jpg",
            }
        canvases.append({
            "@id": f"https://example.com/canvas/{i+1}",
            "@type": "sc:Canvas",
            "label": f"f. {i+1}r",
            "width": 2000,
            "height": 3000,
            "images": [
                {
                    "@type": "oa:Annotation",
                    "motivation": "sc:painting",
                    "resource": resource,
                    "on": f"https://example.com/canvas/{i+1}",
                }
            ],
        })
    return {
        "@context": "http://iiif.io/api/presentation/2/context.json",
        "@type": "sc:Manifest",
        "@id": "https://example.com/manifest.json",
        "label": "Manuscript de test",
        "sequences": [
            {
                "@type": "sc:Sequence",
                "canvases": canvases,
            }
        ],
    }


def _make_v3_manifest(num_canvases: int = 3) -> dict:
    """Fabrique un manifeste IIIF v3 minimal de test."""
    items = []
    for i in range(num_canvases):
        items.append({
            "id": f"https://example.com/canvas/{i+1}",
            "type": "Canvas",
            "label": {"fr": [f"Page {i+1}"]},
            "width": 1500,
            "height": 2200,
            "items": [
                {
                    "id": f"https://example.com/canvas/{i+1}/ap",
                    "type": "AnnotationPage",
                    "items": [
                        {
                            "id": f"https://example.com/canvas/{i+1}/ap/a",
                            "type": "Annotation",
                            "motivation": "painting",
                            "body": {
                                "id": f"https://example.com/images/{i+1}/full/max/0/default.jpg",
                                "type": "Image",
                                "format": "image/jpeg",
                            },
                            "target": f"https://example.com/canvas/{i+1}",
                        }
                    ],
                }
            ],
        })
    return {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "id": "https://example.com/manifest.json",
        "type": "Manifest",
        "label": {"fr": ["Manuscrit v3 de test"]},
        "items": items,
    }


class TestIIIFManifestParserV2:

    def test_version_detection(self):
        manifest = _make_v2_manifest()
        parser = IIIFManifestParser(manifest)
        assert parser.version == 2

    def test_canvases_count(self):
        parser = IIIFManifestParser(_make_v2_manifest(5))
        assert len(parser.canvases()) == 5

    def test_canvas_label(self):
        parser = IIIFManifestParser(_make_v2_manifest())
        canvases = parser.canvases()
        assert canvases[0].label == "f. 1r"
        assert canvases[1].label == "f. 2r"

    def test_canvas_image_url_direct(self):
        parser = IIIFManifestParser(_make_v2_manifest())
        canvases = parser.canvases()
        assert canvases[0].image_url == "https://example.com/images/img1.jpg"

    def test_canvas_image_url_via_service(self):
        parser = IIIFManifestParser(_make_v2_manifest(with_service=True))
        canvases = parser.canvases()
        assert "/full/max/0/default.jpg" in canvases[0].image_url

    def test_canvas_dimensions(self):
        parser = IIIFManifestParser(_make_v2_manifest())
        c = parser.canvases()[0]
        assert c.width == 2000
        assert c.height == 3000

    def test_canvas_index(self):
        parser = IIIFManifestParser(_make_v2_manifest(3))
        canvases = parser.canvases()
        for i, c in enumerate(canvases):
            assert c.index == i

    def test_label(self):
        parser = IIIFManifestParser(_make_v2_manifest())
        assert parser.label == "Manuscript de test"

    def test_empty_sequences(self):
        manifest = {
            "@context": "http://iiif.io/api/presentation/2/context.json",
            "@type": "sc:Manifest",
            "label": "Empty",
            "sequences": [],
        }
        parser = IIIFManifestParser(manifest)
        assert parser.canvases() == []


class TestIIIFManifestParserV3:

    def test_version_detection(self):
        manifest = _make_v3_manifest()
        parser = IIIFManifestParser(manifest)
        assert parser.version == 3

    def test_canvases_count(self):
        parser = IIIFManifestParser(_make_v3_manifest(4))
        assert len(parser.canvases()) == 4

    def test_canvas_label_from_language_map(self):
        parser = IIIFManifestParser(_make_v3_manifest())
        canvases = parser.canvases()
        assert "Page 1" in canvases[0].label

    def test_canvas_image_url(self):
        parser = IIIFManifestParser(_make_v3_manifest())
        canvases = parser.canvases()
        assert "default.jpg" in canvases[0].image_url

    def test_manifest_label_language_map(self):
        parser = IIIFManifestParser(_make_v3_manifest())
        assert "v3" in parser.label.lower() or "test" in parser.label.lower()

    def test_type_manifest_triggers_v3(self):
        """Un manifeste avec type == 'Manifest' est détecté comme v3."""
        manifest = {"type": "Manifest", "items": []}
        parser = IIIFManifestParser(manifest)
        assert parser.version == 3


class TestExtractLabel:

    def test_string(self):
        assert _extract_label("Page 1") == "Page 1"

    def test_list(self):
        assert _extract_label(["Page 1", "Page 2"]) == "Page 1"

    def test_dict_fr(self):
        assert _extract_label({"fr": ["Folio 1r"]}) == "Folio 1r"

    def test_dict_en(self):
        assert _extract_label({"en": ["Folio 1r"]}) == "Folio 1r"

    def test_dict_none_key(self):
        assert _extract_label({"none": ["Label"]}) == "Label"

    def test_empty_string(self):
        assert _extract_label("") == ""

    def test_none_value(self):
        result = _extract_label(None)
        assert isinstance(result, str)


class TestBestImageUrlV2:

    def test_direct_id(self):
        resource = {"@id": "https://example.com/img.jpg"}
        url = _best_image_url_v2(resource, {})
        assert url == "https://example.com/img.jpg"

    def test_service_id(self):
        resource = {
            "@id": "https://example.com/info.json",
            "service": {"@id": "https://example.com/iiif/img1"},
        }
        url = _best_image_url_v2(resource, {})
        assert url == "https://example.com/iiif/img1/full/max/0/default.jpg"

    def test_service_list(self):
        resource = {
            "service": [
                {"@id": "https://example.com/iiif/img2"},
            ]
        }
        url = _best_image_url_v2(resource, {})
        assert url == "https://example.com/iiif/img2/full/max/0/default.jpg"


class TestBestImageUrlV3:

    def test_direct_body_image(self):
        canvas = {
            "items": [
                {
                    "type": "AnnotationPage",
                    "items": [
                        {
                            "type": "Annotation",
                            "motivation": "painting",
                            "body": {
                                "id": "https://example.com/img.jpg",
                                "type": "Image",
                            },
                        }
                    ],
                }
            ]
        }
        url = _best_image_url_v3(canvas)
        assert url == "https://example.com/img.jpg"

    def test_body_via_service(self):
        canvas = {
            "items": [
                {
                    "items": [
                        {
                            "body": {
                                "type": "Image",
                                "id": "",
                                "service": [{"id": "https://example.com/iiif/3/img1"}],
                            }
                        }
                    ]
                }
            ]
        }
        url = _best_image_url_v3(canvas)
        assert "/full/max/0/default.jpg" in url

    def test_empty_canvas(self):
        url = _best_image_url_v3({})
        assert url == ""


class TestGuessExtension:

    def test_jpg(self):
        assert _guess_extension("https://example.com/img.jpg") == ".jpg"

    def test_png(self):
        assert _guess_extension("https://example.com/img.png") == ".png"

    def test_tiff(self):
        assert _guess_extension("https://example.com/img.tiff") == ".tiff"

    def test_iiif_default(self):
        # URL IIIF standard contient /default.jpg
        url = "https://example.com/iiif/img/full/max/0/default.jpg"
        assert _guess_extension(url) == ".jpg"

    def test_unknown_defaults_to_jpg(self):
        assert _guess_extension("https://example.com/resource/123") == ".jpg"


class TestSlugify:

    def test_simple(self):
        assert _slugify("Page 1") == "Page_1"

    def test_special_chars_removed(self):
        result = _slugify("f. 1r (recto)")
        assert "/" not in result
        assert "." not in result

    def test_max_length(self):
        long_label = "x" * 100
        assert len(_slugify(long_label)) <= 60

    def test_empty(self):
        assert _slugify("") == ""


# ===========================================================================
# Tests structure des nouveaux moteurs OCR (sans appel réseau)
# ===========================================================================

class TestMistralOCREngine:

    def test_import(self):
        from picarones.engines.mistral_ocr import MistralOCREngine
        assert MistralOCREngine is not None

    def test_name(self):
        from picarones.engines.mistral_ocr import MistralOCREngine
        engine = MistralOCREngine()
        assert engine.name == "mistral_ocr"

    def test_version_default_model(self):
        from picarones.engines.mistral_ocr import MistralOCREngine
        engine = MistralOCREngine()
        # Le modèle par défaut est désormais mistral-ocr-latest (API OCR native)
        assert "mistral-ocr" in engine.version()

    def test_version_custom_model(self):
        from picarones.engines.mistral_ocr import MistralOCREngine
        engine = MistralOCREngine({"model": "pixtral-large-latest"})
        assert engine.version() == "pixtral-large-latest"

    def test_missing_api_key_raises(self, monkeypatch, tmp_path):
        from picarones.engines.mistral_ocr import MistralOCREngine
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        engine = MistralOCREngine()
        # Créer un fichier image factice
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff")  # JPEG header minimal
        with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
            engine._run_ocr(img)

    def test_exported_from_engines(self):
        from picarones.engines import MistralOCREngine
        assert MistralOCREngine is not None


class TestGoogleVisionEngine:

    def test_import(self):
        from picarones.engines.google_vision import GoogleVisionEngine
        assert GoogleVisionEngine is not None

    def test_name(self):
        from picarones.engines.google_vision import GoogleVisionEngine
        engine = GoogleVisionEngine()
        assert engine.name == "google_vision"

    def test_version(self):
        from picarones.engines.google_vision import GoogleVisionEngine
        engine = GoogleVisionEngine()
        assert engine.version() == "v1"

    def test_missing_credentials_raises(self, monkeypatch, tmp_path):
        from picarones.engines.google_vision import GoogleVisionEngine
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        engine = GoogleVisionEngine()
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff")
        with pytest.raises(RuntimeError):
            engine._run_ocr(img)

    def test_exported_from_engines(self):
        from picarones.engines import GoogleVisionEngine
        assert GoogleVisionEngine is not None


class TestAzureDocIntelEngine:

    def test_import(self):
        from picarones.engines.azure_doc_intel import AzureDocIntelEngine
        assert AzureDocIntelEngine is not None

    def test_name(self):
        from picarones.engines.azure_doc_intel import AzureDocIntelEngine
        engine = AzureDocIntelEngine()
        assert engine.name == "azure_doc_intel"

    def test_missing_key_raises(self, monkeypatch, tmp_path):
        from picarones.engines.azure_doc_intel import AzureDocIntelEngine
        monkeypatch.delenv("AZURE_DOC_INTEL_KEY", raising=False)
        monkeypatch.delenv("AZURE_DOC_INTEL_ENDPOINT", raising=False)
        engine = AzureDocIntelEngine()
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff")
        with pytest.raises(RuntimeError):
            engine._run_ocr(img)

    def test_exported_from_engines(self):
        from picarones.engines import AzureDocIntelEngine
        assert AzureDocIntelEngine is not None


# ===========================================================================
# Tests CLI — commande import iiif
# ===========================================================================

class TestCLIImportIIIF:

    def test_import_group_exists(self):
        from picarones.cli import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ["import", "--help"])
        assert result.exit_code == 0

    def test_import_iiif_command_exists(self):
        from picarones.cli import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ["import", "iiif", "--help"])
        assert result.exit_code == 0
        assert "manifest_url" in result.output.lower() or "MANIFEST_URL" in result.output

    def test_import_iiif_options(self):
        from picarones.cli import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ["import", "iiif", "--help"])
        assert "--pages" in result.output
        assert "--output" in result.output

    def test_import_iiif_requires_url(self):
        from picarones.cli import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ["import", "iiif"])
        # Sans URL, doit afficher une erreur
        assert result.exit_code != 0


# ===========================================================================
# Tests fixtures Sprint 4 (CER diplomatique dans la démo)
# ===========================================================================

class TestFixturesDiplomaticCER:

    def test_gt_texts_contain_medieval_graphies(self):
        """Les textes GT de démo doivent contenir des graphies médiévales."""
        from picarones.fixtures import _GT_TEXTS
        all_gt = " ".join(_GT_TEXTS)
        # Les GT doivent contenir au moins ſ, & ou æ/œ
        has_medieval_chars = any(c in all_gt for c in ["ſ", "&", "æ", "œ"])
        assert has_medieval_chars, "Les GT de démo doivent inclure des graphies médiévales pour illustrer le CER diplomatique"

    def test_benchmark_results_have_diplomatic_cer(self):
        """Les résultats du benchmark fictif doivent inclure le CER diplomatique."""
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark()
        for engine_report in bm.engine_reports:
            for doc_result in engine_report.document_results:
                if doc_result.metrics.error is None:
                    # Le CER diplomatique doit être calculé
                    assert doc_result.metrics.cer_diplomatic is not None, (
                        f"CER diplomatique manquant pour {engine_report.engine_name}"
                    )
                    break  # Un seul doc suffit pour vérifier

    def test_diplomatic_cer_lower_for_medieval_graphies(self):
        """Pour un texte avec ſ, le CER diplomatique doit être ≤ CER exact."""
        result = compute_metrics(
            "maiſon & jardin",  # GT avec graphies médiévales
            "maison et jardin",  # OCR avec graphies modernisées
        )
        assert result.cer_diplomatic is not None
        # CER diplomatique doit être inférieur ou égal au CER exact
        assert result.cer_diplomatic <= result.cer


# ===========================================================================
# Tests rapport HTML Sprint 4 (CER diplomatique affiché)
# ===========================================================================

class TestReportDiplomaticCER:

    def test_report_data_has_cer_diplomatic(self):
        """_build_report_data doit inclure cer_diplomatic dans engines_summary."""
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import _build_report_data

        bm = generate_sample_benchmark()
        data = _build_report_data(bm, images_b64={})

        # Chaque entrée engines doit avoir cer_diplomatic (ou None)
        assert "engines" in data
        for engine_data in data["engines"]:
            assert "cer_diplomatic" in engine_data, (
                f"cer_diplomatic manquant dans {engine_data.get('name', '?')}"
            )

    def test_html_contains_cer_diplo_column(self, tmp_path):
        """Le HTML généré doit contenir la colonne CER diplo."""
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import ReportGenerator

        bm = generate_sample_benchmark()
        out = tmp_path / "report_test.html"
        ReportGenerator(bm).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "diplo" in html.lower() or "diplomatique" in html.lower(), (
            "Le rapport HTML doit mentionner le CER diplomatique"
        )

    def test_html_contains_medieval_graphie_indicator(self, tmp_path):
        """Le rapport doit mentionner les graphies médiévales (ſ=s ou u=v)."""
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import ReportGenerator

        bm = generate_sample_benchmark()
        out = tmp_path / "report_test.html"
        ReportGenerator(bm).generate(out)
        html = out.read_text(encoding="utf-8")
        # Le tooltip ou la légende doit mentionner les correspondances diplomatiques
        assert "ſ=s" in html or "u=v" in html or "diplomatique" in html.lower()
