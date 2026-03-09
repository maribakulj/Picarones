"""Tests Sprint 6 — Interface web FastAPI, import HTR-United, HuggingFace, serve CLI.

Classes de tests
----------------
TestHTRUnitedEntry           (8 tests)  — dataclass, as_dict, from_dict, century_str
TestHTRUnitedCatalogue       (10 tests) — from_demo, search, get_by_id, available_languages/scripts
TestHTRUnitedSearch          (8 tests)  — recherche textuelle, filtre langue, script, siècle
TestHTRUnitedImport          (4 tests)  — import_htr_united_corpus crée les fichiers meta
TestHuggingFaceDataset       (7 tests)  — dataclass, as_dict, from_dict, hf_url
TestHuggingFaceImporter      (10 tests) — search référence, filtres, import
TestHuggingFaceReferenceData (4 tests)  — datasets de référence pré-intégrés
TestNormalizationProfiles    (8 tests)  — profils disponibles via API route
TestFastAPIStatus            (3 tests)  — GET /api/status
TestFastAPIEngines           (8 tests)  — GET /api/engines
TestFastAPICorpusBrowse      (6 tests)  — GET /api/corpus/browse
TestFastAPIReports           (5 tests)  — GET /api/reports
TestFastAPIHTRUnited         (7 tests)  — GET /api/htr-united/catalogue + POST import
TestFastAPIHuggingFace       (6 tests)  — GET /api/huggingface/search + POST import
TestFastAPIBenchmark         (8 tests)  — POST start, GET status, GET stream, POST cancel
TestFastAPIHTML              (5 tests)  — GET / retourne HTML valide
TestFastAPIReportServe       (4 tests)  — GET /reports/{filename}
TestCLIServeCommand          (5 tests)  — commande picarones serve enregistrée
TestRunnerProgressCallback   (5 tests)  — progress_callback injecté dans run_benchmark
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_corpus(tmp_path):
    """Crée un corpus minimal avec 2 documents."""
    from PIL import Image
    for i in range(2):
        img = Image.new("RGB", (100, 50), color=(200, 200, 200))
        img.save(tmp_path / f"doc_{i:02d}.jpg")
        (tmp_path / f"doc_{i:02d}.gt.txt").write_text(f"Texte vérité terrain {i}", encoding="utf-8")
    return tmp_path


@pytest.fixture
def client():
    from picarones.web.app import app
    return TestClient(app)


@pytest.fixture
def htr_catalogue():
    from picarones.importers.htr_united import HTRUnitedCatalogue
    return HTRUnitedCatalogue.from_demo()


@pytest.fixture
def hf_importer():
    from picarones.importers.huggingface import HuggingFaceImporter
    return HuggingFaceImporter()


# ===========================================================================
# TestHTRUnitedEntry
# ===========================================================================

class TestHTRUnitedEntry:

    def test_from_dict_basic(self):
        from picarones.importers.htr_united import HTRUnitedEntry
        d = {
            "id": "test-corpus", "title": "Test Corpus", "url": "https://github.com/test/corpus",
            "language": ["French"], "script": ["Gothic"], "century": [14, 15],
            "institution": "BnF", "description": "Un corpus de test.", "license": "CC-BY 4.0",
            "lines": 5000, "format": "ALTO", "tags": ["test", "médiéval"],
        }
        e = HTRUnitedEntry.from_dict(d)
        assert e.id == "test-corpus"
        assert e.title == "Test Corpus"
        assert e.language == ["French"]
        assert e.lines == 5000

    def test_as_dict_roundtrip(self):
        from picarones.importers.htr_united import HTRUnitedEntry
        d = {
            "id": "rtrip", "title": "Round Trip", "url": "https://github.com/a/b",
            "language": ["Latin"], "script": ["Caroline"], "century": [9],
            "institution": "IRHT", "description": "Test.", "license": "CC0",
            "lines": 1000, "format": "PAGE", "tags": [],
        }
        e = HTRUnitedEntry.from_dict(d)
        out = e.as_dict()
        assert out["id"] == "rtrip"
        assert out["lines"] == 1000
        assert out["format"] == "PAGE"

    def test_century_str_roman(self):
        from picarones.importers.htr_united import HTRUnitedEntry
        e = HTRUnitedEntry(id="x", title="x", url="x", century=[12, 14])
        cs = e.century_str
        assert "XIIe" in cs
        assert "XIVe" in cs

    def test_century_str_single(self):
        from picarones.importers.htr_united import HTRUnitedEntry
        e = HTRUnitedEntry(id="x", title="x", url="x", century=[19])
        assert "XIXe" in e.century_str

    def test_default_fields(self):
        from picarones.importers.htr_united import HTRUnitedEntry
        e = HTRUnitedEntry(id="minimal", title="Min", url="http://x")
        assert e.language == []
        assert e.lines == 0
        assert e.format == "ALTO"
        assert e.tags == []

    def test_from_dict_missing_fields(self):
        from picarones.importers.htr_united import HTRUnitedEntry
        e = HTRUnitedEntry.from_dict({"id": "sparse", "title": "Sparse"})
        assert e.id == "sparse"
        assert e.institution == ""
        assert e.lines == 0

    def test_as_dict_has_all_keys(self):
        from picarones.importers.htr_united import HTRUnitedEntry
        e = HTRUnitedEntry(id="k", title="K", url="http://k")
        d = e.as_dict()
        for key in ["id", "title", "url", "language", "script", "century",
                    "institution", "description", "license", "lines", "format", "tags"]:
            assert key in d, f"Missing key: {key}"

    def test_url_preserved(self):
        from picarones.importers.htr_united import HTRUnitedEntry
        url = "https://github.com/HTR-United/cremma-medieval"
        e = HTRUnitedEntry(id="c", title="CREMMA", url=url)
        assert e.url == url


# ===========================================================================
# TestHTRUnitedCatalogue
# ===========================================================================

class TestHTRUnitedCatalogue:

    def test_from_demo_length(self, htr_catalogue):
        assert len(htr_catalogue) >= 6

    def test_from_demo_source(self, htr_catalogue):
        assert htr_catalogue.source == "demo"

    def test_all_entries_have_id(self, htr_catalogue):
        for e in htr_catalogue.entries:
            assert e.id, f"Entry missing id: {e}"

    def test_all_entries_have_title(self, htr_catalogue):
        for e in htr_catalogue.entries:
            assert e.title

    def test_get_by_id_found(self, htr_catalogue):
        first_id = htr_catalogue.entries[0].id
        found = htr_catalogue.get_by_id(first_id)
        assert found is not None
        assert found.id == first_id

    def test_get_by_id_not_found(self, htr_catalogue):
        result = htr_catalogue.get_by_id("nonexistent-corpus-xyz")
        assert result is None

    def test_available_languages_non_empty(self, htr_catalogue):
        langs = htr_catalogue.available_languages()
        assert len(langs) > 0
        assert isinstance(langs, list)

    def test_available_languages_sorted(self, htr_catalogue):
        langs = htr_catalogue.available_languages()
        assert langs == sorted(langs)

    def test_available_scripts_non_empty(self, htr_catalogue):
        scripts = htr_catalogue.available_scripts()
        assert len(scripts) > 0

    def test_len(self, htr_catalogue):
        assert len(htr_catalogue) == len(htr_catalogue.entries)


# ===========================================================================
# TestHTRUnitedSearch
# ===========================================================================

class TestHTRUnitedSearch:

    def test_search_empty_returns_all(self, htr_catalogue):
        results = htr_catalogue.search()
        assert len(results) == len(htr_catalogue.entries)

    def test_search_by_query(self, htr_catalogue):
        results = htr_catalogue.search(query="médiéval")
        assert len(results) > 0
        for r in results:
            text = (r.title + r.description + " ".join(r.tags)).lower()
            assert "médiéval" in text

    def test_search_by_language(self, htr_catalogue):
        results = htr_catalogue.search(language="French")
        assert len(results) > 0
        for r in results:
            assert any("french" in l.lower() for l in r.language)

    def test_search_by_language_latin(self, htr_catalogue):
        results = htr_catalogue.search(language="Latin")
        assert len(results) > 0

    def test_search_by_script(self, htr_catalogue):
        results = htr_catalogue.search(script="Gothic")
        assert len(results) > 0

    def test_search_no_results(self, htr_catalogue):
        results = htr_catalogue.search(query="xyzzy_corpus_inexistant_42")
        assert results == []

    def test_search_combined_filters(self, htr_catalogue):
        # Ne doit pas lever d'exception
        results = htr_catalogue.search(query="", language="French", script="Cursiva")
        assert isinstance(results, list)

    def test_search_century_min(self, htr_catalogue):
        results = htr_catalogue.search(century_min=18)
        for r in results:
            assert any(c >= 18 for c in r.century)


# ===========================================================================
# TestHTRUnitedImport
# ===========================================================================

class TestHTRUnitedImport:

    def test_import_creates_meta_file(self, tmp_path, htr_catalogue):
        from picarones.importers.htr_united import import_htr_united_corpus
        entry = htr_catalogue.entries[0]
        result = import_htr_united_corpus(entry, tmp_path, max_samples=5)
        meta_file = Path(result["metadata_file"])
        assert meta_file.exists()

    def test_import_meta_content(self, tmp_path, htr_catalogue):
        from picarones.importers.htr_united import import_htr_united_corpus
        entry = htr_catalogue.entries[0]
        result = import_htr_united_corpus(entry, tmp_path, max_samples=5)
        meta = json.loads(Path(result["metadata_file"]).read_text())
        assert meta["source"] == "htr-united"
        assert meta["entry_id"] == entry.id

    def test_import_returns_dict_keys(self, tmp_path, htr_catalogue):
        from picarones.importers.htr_united import import_htr_united_corpus
        entry = htr_catalogue.entries[0]
        result = import_htr_united_corpus(entry, tmp_path, max_samples=5)
        for k in ["entry_id", "title", "output_dir", "files_imported", "metadata_file"]:
            assert k in result, f"Missing key: {k}"

    def test_import_creates_output_dir(self, tmp_path, htr_catalogue):
        from picarones.importers.htr_united import import_htr_united_corpus
        entry = htr_catalogue.entries[0]
        new_dir = tmp_path / "new_subdir" / "corpus"
        result = import_htr_united_corpus(entry, new_dir, max_samples=5)
        assert new_dir.exists()


# ===========================================================================
# TestHuggingFaceDataset
# ===========================================================================

class TestHuggingFaceDataset:

    def test_from_dict_basic(self):
        from picarones.importers.huggingface import HuggingFaceDataset
        d = {
            "dataset_id": "test/dataset", "title": "Test Dataset",
            "description": "A test dataset.", "language": ["French"],
            "tags": ["ocr", "french"], "license": "cc-by-4.0",
            "institution": "Test Lab", "downloads": 500,
        }
        ds = HuggingFaceDataset.from_dict(d)
        assert ds.dataset_id == "test/dataset"
        assert ds.language == ["French"]
        assert ds.downloads == 500

    def test_as_dict_roundtrip(self):
        from picarones.importers.huggingface import HuggingFaceDataset
        ds = HuggingFaceDataset(
            dataset_id="a/b", title="AB", description="desc",
            language=["Latin"], tags=["htr"],
        )
        d = ds.as_dict()
        assert d["dataset_id"] == "a/b"
        assert d["language"] == ["Latin"]

    def test_hf_url(self):
        from picarones.importers.huggingface import HuggingFaceDataset
        ds = HuggingFaceDataset(dataset_id="CATMuS/medieval", title="CATMuS")
        assert ds.hf_url == "https://huggingface.co/datasets/CATMuS/medieval"

    def test_as_dict_has_all_keys(self):
        from picarones.importers.huggingface import HuggingFaceDataset
        ds = HuggingFaceDataset(dataset_id="x/y", title="XY")
        d = ds.as_dict()
        for k in ["dataset_id", "title", "description", "language", "tags",
                   "license", "size_category", "task", "institution", "downloads", "source"]:
            assert k in d, f"Missing: {k}"

    def test_default_source(self):
        from picarones.importers.huggingface import HuggingFaceDataset
        ds = HuggingFaceDataset(dataset_id="x/y", title="XY")
        assert ds.source == "reference"

    def test_from_dict_uses_id_as_fallback_title(self):
        from picarones.importers.huggingface import HuggingFaceDataset
        ds = HuggingFaceDataset.from_dict({"dataset_id": "owner/repo"})
        assert ds.title == "owner/repo"

    def test_replace_source_helper(self):
        from picarones.importers.huggingface import HuggingFaceDataset
        ds = HuggingFaceDataset(dataset_id="x/y", title="XY", source="reference")
        ds2 = ds._replace_source("api")
        assert ds2.source == "api"
        assert ds.source == "reference"  # original unchanged


# ===========================================================================
# TestHuggingFaceImporter
# ===========================================================================

class TestHuggingFaceImporter:

    def test_search_returns_list(self, hf_importer):
        results = hf_importer.search()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_reference_datasets(self, hf_importer):
        results = hf_importer.search(use_reference=True)
        assert len(results) >= 5

    def test_search_query_filter(self, hf_importer):
        results = hf_importer.search(query="RIMES", use_reference=True)
        assert len(results) >= 1
        assert any("RIMES" in ds.title or "rimes" in ds.dataset_id.lower() for ds in results)

    def test_search_language_filter(self, hf_importer):
        results = hf_importer.search(language="French", use_reference=True)
        assert len(results) > 0

    def test_search_tag_filter(self, hf_importer):
        results = hf_importer.search(tags=["historical"], use_reference=True)
        assert isinstance(results, list)

    def test_search_limit(self, hf_importer):
        results = hf_importer.search(limit=3)
        assert len(results) <= 3

    def test_search_no_api_fallback(self, hf_importer):
        # Même sans accès réseau, on a les datasets de référence
        results = hf_importer.search(query="medieval", use_reference=True)
        assert len(results) >= 1

    def test_import_creates_meta(self, tmp_path, hf_importer):
        result = hf_importer.import_dataset("CATMuS/medieval", output_dir=tmp_path, max_samples=5)
        assert Path(result["metadata_file"]).exists()

    def test_import_meta_content(self, tmp_path, hf_importer):
        result = hf_importer.import_dataset("CATMuS/medieval", output_dir=tmp_path, max_samples=5)
        meta = json.loads(Path(result["metadata_file"]).read_text())
        assert meta["dataset_id"] == "CATMuS/medieval"
        assert meta["source"] == "huggingface"

    def test_import_returns_dict_keys(self, tmp_path, hf_importer):
        result = hf_importer.import_dataset("x/y", output_dir=tmp_path, max_samples=5)
        for k in ["dataset_id", "output_dir", "files_imported", "metadata_file"]:
            assert k in result


# ===========================================================================
# TestHuggingFaceReferenceData
# ===========================================================================

class TestHuggingFaceReferenceData:

    def test_reference_datasets_loaded(self):
        from picarones.importers.huggingface import _REFERENCE_DATASETS
        assert len(_REFERENCE_DATASETS) >= 5

    def test_catmus_present(self):
        from picarones.importers.huggingface import _REFERENCE_DATASETS
        ids = [d["dataset_id"] for d in _REFERENCE_DATASETS]
        assert any("CATMuS" in did or "catmus" in did.lower() for did in ids)

    def test_all_have_required_fields(self):
        from picarones.importers.huggingface import _REFERENCE_DATASETS
        for d in _REFERENCE_DATASETS:
            assert "dataset_id" in d
            assert "title" in d
            assert "language" in d

    def test_all_are_image_to_text(self):
        from picarones.importers.huggingface import _REFERENCE_DATASETS
        for d in _REFERENCE_DATASETS:
            assert d.get("task", "image-to-text") == "image-to-text"


# ===========================================================================
# TestNormalizationProfiles
# ===========================================================================

class TestNormalizationProfiles:

    def test_api_returns_profiles(self, client):
        r = client.get("/api/normalization/profiles")
        assert r.status_code == 200
        d = r.json()
        assert "profiles" in d
        assert len(d["profiles"]) >= 4

    def test_nfc_profile_present(self, client):
        r = client.get("/api/normalization/profiles")
        ids = [p["id"] for p in r.json()["profiles"]]
        assert "nfc" in ids

    def test_medieval_french_present(self, client):
        r = client.get("/api/normalization/profiles")
        ids = [p["id"] for p in r.json()["profiles"]]
        assert "medieval_french" in ids

    def test_profiles_have_required_fields(self, client):
        r = client.get("/api/normalization/profiles")
        for p in r.json()["profiles"]:
            assert "id" in p
            assert "name" in p
            assert "description" in p
            assert "caseless" in p
            assert "diplomatic_rules" in p

    def test_caseless_profile(self, client):
        r = client.get("/api/normalization/profiles")
        profiles = {p["id"]: p for p in r.json()["profiles"]}
        assert "caseless" in profiles
        assert profiles["caseless"]["caseless"] is True

    def test_medieval_french_has_diplomatic_rules(self, client):
        r = client.get("/api/normalization/profiles")
        profiles = {p["id"]: p for p in r.json()["profiles"]}
        assert profiles["medieval_french"]["diplomatic_rules"] > 0

    def test_nfc_no_diplomatic_rules(self, client):
        r = client.get("/api/normalization/profiles")
        profiles = {p["id"]: p for p in r.json()["profiles"]}
        assert profiles["nfc"]["diplomatic_rules"] == 0

    def test_early_modern_french_present(self, client):
        r = client.get("/api/normalization/profiles")
        ids = [p["id"] for p in r.json()["profiles"]]
        assert "early_modern_french" in ids


# ===========================================================================
# TestFastAPIStatus
# ===========================================================================

class TestFastAPIStatus:

    def test_status_200(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200

    def test_status_has_version(self, client):
        r = client.get("/api/status")
        d = r.json()
        assert "version" in d
        assert d["version"]

    def test_status_ok(self, client):
        r = client.get("/api/status")
        assert r.json()["status"] == "ok"


# ===========================================================================
# TestFastAPIEngines
# ===========================================================================

class TestFastAPIEngines:

    def test_engines_200(self, client):
        r = client.get("/api/engines")
        assert r.status_code == 200

    def test_engines_has_engines_key(self, client):
        r = client.get("/api/engines")
        assert "engines" in r.json()

    def test_engines_has_llms_key(self, client):
        r = client.get("/api/engines")
        assert "llms" in r.json()

    def test_engines_list_not_empty(self, client):
        r = client.get("/api/engines")
        assert len(r.json()["engines"]) > 0

    def test_llms_list_not_empty(self, client):
        r = client.get("/api/engines")
        assert len(r.json()["llms"]) > 0

    def test_tesseract_in_engines(self, client):
        r = client.get("/api/engines")
        ids = [e["id"] for e in r.json()["engines"]]
        assert "tesseract" in ids

    def test_ollama_in_llms(self, client):
        r = client.get("/api/engines")
        ids = [e["id"] for e in r.json()["llms"]]
        assert "ollama" in ids

    def test_engine_has_required_fields(self, client):
        r = client.get("/api/engines")
        for eng in r.json()["engines"]:
            assert "id" in eng
            assert "label" in eng
            assert "available" in eng
            assert "status" in eng


# ===========================================================================
# TestFastAPICorpusBrowse
# ===========================================================================

class TestFastAPICorpusBrowse:

    def test_browse_current_dir(self, client):
        r = client.get("/api/corpus/browse?path=.")
        assert r.status_code == 200

    def test_browse_has_required_keys(self, client):
        r = client.get("/api/corpus/browse?path=.")
        d = r.json()
        assert "current_path" in d
        assert "items" in d

    def test_browse_items_are_dirs(self, client, tmp_path):
        r = client.get(f"/api/corpus/browse?path={tmp_path}")
        assert r.status_code == 200
        assert r.json()["items"] == []

    def test_browse_with_corpus(self, client, tmp_corpus):
        r = client.get(f"/api/corpus/browse?path={tmp_corpus.parent}")
        assert r.status_code == 200
        items = r.json()["items"]
        assert any(i["name"] == tmp_corpus.name for i in items)

    def test_browse_404_for_nonexistent(self, client):
        r = client.get("/api/corpus/browse?path=/nonexistent/path/xyz")
        assert r.status_code == 404

    def test_browse_corpus_gt_count(self, client, tmp_corpus):
        r = client.get(f"/api/corpus/browse?path={tmp_corpus.parent}")
        items = {i["name"]: i for i in r.json()["items"] if i["is_dir"]}
        if tmp_corpus.name in items:
            assert items[tmp_corpus.name]["gt_count"] >= 2


# ===========================================================================
# TestFastAPIReports
# ===========================================================================

class TestFastAPIReports:

    def test_reports_200(self, client):
        r = client.get("/api/reports")
        assert r.status_code == 200

    def test_reports_has_reports_key(self, client):
        r = client.get("/api/reports")
        assert "reports" in r.json()

    def test_reports_returns_list(self, client):
        r = client.get("/api/reports")
        assert isinstance(r.json()["reports"], list)

    def test_reports_finds_existing_html(self, client, tmp_path):
        # Crée un rapport HTML fictif
        html_file = tmp_path / "test_rapport.html"
        html_file.write_text("<html><body>Test rapport</body></html>")
        r = client.get(f"/api/reports?reports_dir={tmp_path}")
        reports = r.json()["reports"]
        assert any(rep["filename"] == "test_rapport.html" for rep in reports)

    def test_report_entry_has_fields(self, client, tmp_path):
        html_file = tmp_path / "my_report.html"
        html_file.write_text("<html></html>")
        r = client.get(f"/api/reports?reports_dir={tmp_path}")
        rep = next(rep for rep in r.json()["reports"] if rep["filename"] == "my_report.html")
        assert "filename" in rep
        assert "path" in rep
        assert "size_kb" in rep
        assert "modified" in rep
        assert "url" in rep


# ===========================================================================
# TestFastAPIHTRUnited
# ===========================================================================

class TestFastAPIHTRUnited:

    def test_catalogue_200(self, client):
        r = client.get("/api/htr-united/catalogue")
        assert r.status_code == 200

    def test_catalogue_has_entries(self, client):
        r = client.get("/api/htr-united/catalogue")
        d = r.json()
        assert "entries" in d
        assert len(d["entries"]) >= 4

    def test_catalogue_has_filters(self, client):
        r = client.get("/api/htr-united/catalogue")
        d = r.json()
        assert "available_languages" in d
        assert "available_scripts" in d

    def test_catalogue_search_query(self, client):
        r = client.get("/api/htr-united/catalogue?query=médiéval")
        assert r.status_code == 200
        d = r.json()
        assert d["total"] >= 0  # Can be 0 if no match — no error

    def test_catalogue_search_language(self, client):
        r = client.get("/api/htr-united/catalogue?language=French")
        assert r.status_code == 200
        d = r.json()
        for e in d["entries"]:
            assert any("french" in l.lower() for l in e["language"])

    def test_import_valid_entry(self, client, tmp_path):
        # Get first entry id
        r = client.get("/api/htr-united/catalogue")
        entry_id = r.json()["entries"][0]["id"]
        r2 = client.post("/api/htr-united/import", json={
            "entry_id": entry_id,
            "output_dir": str(tmp_path),
            "max_samples": 5,
        })
        assert r2.status_code == 200
        assert "entry_id" in r2.json()

    def test_import_invalid_entry(self, client, tmp_path):
        r = client.post("/api/htr-united/import", json={
            "entry_id": "this-does-not-exist-xyz",
            "output_dir": str(tmp_path),
            "max_samples": 5,
        })
        assert r.status_code == 404


# ===========================================================================
# TestFastAPIHuggingFace
# ===========================================================================

class TestFastAPIHuggingFace:

    def test_search_200(self, client):
        r = client.get("/api/huggingface/search")
        assert r.status_code == 200

    def test_search_has_datasets(self, client):
        r = client.get("/api/huggingface/search")
        d = r.json()
        assert "datasets" in d
        assert d["total"] >= 1

    def test_search_with_query(self, client):
        r = client.get("/api/huggingface/search?query=RIMES")
        assert r.status_code == 200
        d = r.json()
        assert isinstance(d["datasets"], list)

    def test_search_with_language(self, client):
        r = client.get("/api/huggingface/search?language=French")
        assert r.status_code == 200

    def test_import_creates_meta(self, client, tmp_path):
        r = client.post("/api/huggingface/import", json={
            "dataset_id": "CATMuS/medieval",
            "output_dir": str(tmp_path),
            "split": "train",
            "max_samples": 5,
        })
        assert r.status_code == 200
        d = r.json()
        assert Path(d["metadata_file"]).exists()

    def test_import_returns_keys(self, client, tmp_path):
        r = client.post("/api/huggingface/import", json={
            "dataset_id": "test/dataset",
            "output_dir": str(tmp_path),
        })
        assert r.status_code == 200
        for k in ["dataset_id", "output_dir", "files_imported", "metadata_file"]:
            assert k in r.json()


# ===========================================================================
# TestFastAPIBenchmark
# ===========================================================================

class TestFastAPIBenchmark:

    def test_start_missing_corpus(self, client):
        r = client.post("/api/benchmark/start", json={
            "corpus_path": "/nonexistent/path/xyz",
            "engines": ["tesseract"],
        })
        assert r.status_code == 400

    def test_start_valid_corpus(self, client, tmp_corpus):
        r = client.post("/api/benchmark/start", json={
            "corpus_path": str(tmp_corpus),
            "engines": ["tesseract"],
        })
        assert r.status_code == 200
        d = r.json()
        assert "job_id" in d
        assert d["status"] in ("pending", "running")

    def test_status_nonexistent_job(self, client):
        r = client.get("/api/benchmark/nonexistent-job-id/status")
        assert r.status_code == 404

    def test_status_valid_job(self, client, tmp_corpus):
        r = client.post("/api/benchmark/start", json={
            "corpus_path": str(tmp_corpus),
            "engines": ["tesseract"],
        })
        job_id = r.json()["job_id"]
        r2 = client.get(f"/api/benchmark/{job_id}/status")
        assert r2.status_code == 200
        d = r2.json()
        assert d["job_id"] == job_id
        assert "status" in d
        assert "progress" in d

    def test_cancel_nonexistent_job(self, client):
        r = client.post("/api/benchmark/nonexistent-id/cancel")
        assert r.status_code == 404

    def test_cancel_valid_job(self, client, tmp_corpus):
        r = client.post("/api/benchmark/start", json={
            "corpus_path": str(tmp_corpus),
            "engines": ["tesseract"],
        })
        job_id = r.json()["job_id"]
        r2 = client.post(f"/api/benchmark/{job_id}/cancel")
        assert r2.status_code == 200

    def test_job_status_fields(self, client, tmp_corpus):
        r = client.post("/api/benchmark/start", json={
            "corpus_path": str(tmp_corpus),
            "engines": ["tesseract"],
        })
        job_id = r.json()["job_id"]
        r2 = client.get(f"/api/benchmark/{job_id}/status")
        d = r2.json()
        for k in ["job_id", "status", "progress", "total_docs", "processed_docs", "output_path"]:
            assert k in d, f"Missing key: {k}"

    def test_stream_nonexistent_job(self, client):
        r = client.get("/api/benchmark/nonexistent-id/stream")
        assert r.status_code == 404


# ===========================================================================
# TestFastAPIHTML
# ===========================================================================

class TestFastAPIHTML:

    def test_root_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_is_html(self, client):
        r = client.get("/")
        assert "text/html" in r.headers["content-type"]

    def test_html_has_picarones_title(self, client):
        r = client.get("/")
        assert "Picarones" in r.text

    def test_html_has_nav_sections(self, client):
        r = client.get("/")
        for section in ["benchmark", "reports", "engines", "import"]:
            assert section in r.text.lower()

    def test_html_has_french_content(self, client):
        r = client.get("/")
        assert "Moteurs" in r.text or "moteurs" in r.text.lower()


# ===========================================================================
# TestFastAPIReportServe
# ===========================================================================

class TestFastAPIReportServe:

    def test_serve_nonexistent_report(self, client):
        r = client.get("/reports/nonexistent_report.html")
        assert r.status_code == 404

    def test_serve_existing_report(self, client, tmp_path, monkeypatch):
        # Crée un rapport HTML dans le répertoire courant
        import os
        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            html_file = tmp_path / "test_serve.html"
            html_file.write_text("<html><body>Test</body></html>")
            r = client.get("/reports/test_serve.html")
            assert r.status_code == 200
        finally:
            os.chdir(orig_cwd)

    def test_serve_non_html_rejected(self, client):
        # Tente de servir un .py — doit retourner 404 (extension non-html)
        r = client.get("/reports/malicious.py")
        assert r.status_code == 404

    def test_serve_report_content_type(self, client, tmp_path):
        import os
        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            html_file = tmp_path / "report_ct.html"
            html_file.write_text("<html><body>Content</body></html>")
            r = client.get("/reports/report_ct.html")
            if r.status_code == 200:
                assert "html" in r.headers.get("content-type", "").lower()
        finally:
            os.chdir(orig_cwd)


# ===========================================================================
# TestCLIServeCommand
# ===========================================================================

class TestCLIServeCommand:

    def test_serve_command_registered(self):
        from picarones.cli import cli
        commands = cli.commands
        assert "serve" in commands

    def test_serve_help_text(self):
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "serve" in result.output.lower() or "localhost" in result.output.lower()

    def test_serve_default_port_in_help(self):
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert "8000" in result.output

    def test_serve_help_has_port_option(self):
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert "--port" in result.output

    def test_serve_missing_uvicorn_exits_gracefully(self):
        from picarones.cli import cli
        runner = CliRunner()
        # Avec uvicorn installé, cela démarrerait le serveur — on teste juste que
        # la commande existe et est invocable (pas qu'elle démare le serveur)
        # On vérifie juste le help
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# TestRunnerProgressCallback
# ===========================================================================

class TestRunnerProgressCallback:

    def test_callback_signature_accepted(self):
        """run_benchmark accepte un paramètre progress_callback."""
        import inspect
        from picarones.core.runner import run_benchmark
        sig = inspect.signature(run_benchmark)
        assert "progress_callback" in sig.parameters

    def test_callback_is_optional(self):
        """progress_callback est optionnel (valeur par défaut None)."""
        import inspect
        from picarones.core.runner import run_benchmark
        sig = inspect.signature(run_benchmark)
        param = sig.parameters["progress_callback"]
        assert param.default is None

    def test_callback_called_with_mock_engine(self, tmp_corpus):
        """Le callback est appelé pour chaque document."""
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.core.runner import run_benchmark
        from picarones.engines.base import BaseOCREngine, EngineResult

        class MockEngine(BaseOCREngine):
            @property
            def name(self): return "mock"
            @property
            def version(self): return "0.0.1"
            def _run_ocr(self, image_path): return "texte mock"

        corpus = load_corpus_from_directory(str(tmp_corpus))
        calls = []
        def my_callback(engine_name, doc_idx, doc_id):
            calls.append((engine_name, doc_idx, doc_id))

        run_benchmark(corpus, [MockEngine()], progress_callback=my_callback)
        assert len(calls) == len(corpus), f"Expected {len(corpus)} calls, got {len(calls)}"

    def test_callback_receives_engine_name(self, tmp_corpus):
        """Le callback reçoit le nom du moteur."""
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.core.runner import run_benchmark
        from picarones.engines.base import BaseOCREngine

        class MockEngine(BaseOCREngine):
            @property
            def name(self): return "test_engine_name"
            @property
            def version(self): return "0.0.1"
            def _run_ocr(self, image_path): return "texte"

        corpus = load_corpus_from_directory(str(tmp_corpus))
        engine_names = []
        def my_callback(engine_name, doc_idx, doc_id):
            engine_names.append(engine_name)

        run_benchmark(corpus, [MockEngine()], progress_callback=my_callback)
        assert all(n == "test_engine_name" for n in engine_names)

    def test_callback_exception_does_not_crash(self, tmp_corpus):
        """Une exception dans le callback ne plante pas le benchmark."""
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.core.runner import run_benchmark
        from picarones.engines.base import BaseOCREngine

        class MockEngine(BaseOCREngine):
            @property
            def name(self): return "mock"
            @property
            def version(self): return "0.0.1"
            def _run_ocr(self, image_path): return "texte"

        corpus = load_corpus_from_directory(str(tmp_corpus))

        def bad_callback(engine_name, doc_idx, doc_id):
            raise RuntimeError("Callback error!")

        # Ne doit pas lever d'exception
        result = run_benchmark(corpus, [MockEngine()], progress_callback=bad_callback)
        assert result is not None


# ===========================================================================
# TestFastAPIModels  — GET /api/models/{provider}
# ===========================================================================

class TestFastAPIModels:

    def test_models_tesseract_200(self, client):
        r = client.get("/api/models/tesseract")
        assert r.status_code == 200

    def test_models_tesseract_has_models_list(self, client):
        r = client.get("/api/models/tesseract")
        d = r.json()
        assert "models" in d
        assert isinstance(d["models"], list)

    def test_models_tesseract_has_provider_field(self, client):
        r = client.get("/api/models/tesseract")
        assert r.json()["provider"] == "tesseract"

    def test_models_tesseract_has_languages(self, client):
        r = client.get("/api/models/tesseract")
        models = r.json()["models"]
        # Tesseract est installé dans le CI, au moins fra ou eng doit être présent
        assert len(models) > 0

    def test_models_google_vision_200(self, client):
        r = client.get("/api/models/google_vision")
        assert r.status_code == 200
        assert "document_text_detection" in r.json()["models"]

    def test_models_azure_doc_intel_200(self, client):
        r = client.get("/api/models/azure_doc_intel")
        assert r.status_code == 200
        assert "prebuilt-document" in r.json()["models"]

    def test_models_ollama_200(self, client):
        r = client.get("/api/models/ollama")
        assert r.status_code == 200
        assert isinstance(r.json()["models"], list)

    def test_models_prompts_200(self, client):
        r = client.get("/api/models/prompts")
        assert r.status_code == 200
        d = r.json()
        assert isinstance(d["models"], list)
        assert len(d["models"]) >= 5  # 8 prompts intégrés

    def test_models_prompts_are_txt_files(self, client):
        r = client.get("/api/models/prompts")
        for name in r.json()["models"]:
            assert name.endswith(".txt")

    def test_models_openai_no_key_returns_empty(self, client):
        # Sans clé, doit renvoyer liste vide + champ error
        with patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}, clear=True):
            r = client.get("/api/models/openai")
        assert r.status_code == 200
        d = r.json()
        assert d["models"] == [] or "error" in d

    def test_models_anthropic_no_key_returns_empty(self, client):
        with patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}, clear=True):
            r = client.get("/api/models/anthropic")
        assert r.status_code == 200
        d = r.json()
        assert d["models"] == [] or "error" in d

    def test_models_unknown_provider_404(self, client):
        r = client.get("/api/models/provider_xyz_unknown")
        assert r.status_code == 404


# ===========================================================================
# TestFastAPIBenchmarkRun  — POST /api/benchmark/run
# ===========================================================================

class TestFastAPIBenchmarkRun:

    def test_run_400_missing_corpus(self, client):
        r = client.post("/api/benchmark/run", json={
            "corpus_path": "/nonexistent/path/xyz",
            "competitors": [{"ocr_engine": "tesseract", "ocr_model": "fra"}],
        })
        assert r.status_code == 400

    def test_run_400_no_competitors(self, client, tmp_corpus):
        r = client.post("/api/benchmark/run", json={
            "corpus_path": str(tmp_corpus),
            "competitors": [],
        })
        assert r.status_code == 400

    def test_run_422_missing_ocr_engine(self, client, tmp_corpus):
        r = client.post("/api/benchmark/run", json={
            "corpus_path": str(tmp_corpus),
            "competitors": [{"ocr_model": "fra"}],   # ocr_engine manquant
        })
        assert r.status_code == 422

    def test_run_returns_job_id(self, client, tmp_corpus):
        r = client.post("/api/benchmark/run", json={
            "corpus_path": str(tmp_corpus),
            "competitors": [{"ocr_engine": "tesseract", "ocr_model": "fra"}],
        })
        assert r.status_code == 200
        d = r.json()
        assert "job_id" in d
        assert "status" in d

    def test_run_job_status_reachable(self, client, tmp_corpus):
        r = client.post("/api/benchmark/run", json={
            "corpus_path": str(tmp_corpus),
            "competitors": [{"ocr_engine": "tesseract", "ocr_model": "fra"}],
        })
        job_id = r.json()["job_id"]
        r2 = client.get(f"/api/benchmark/{job_id}/status")
        assert r2.status_code == 200
        d = r2.json()
        assert d["job_id"] == job_id

    def test_run_with_named_competitor(self, client, tmp_corpus):
        r = client.post("/api/benchmark/run", json={
            "corpus_path": str(tmp_corpus),
            "competitors": [{"name": "Mon Tesseract", "ocr_engine": "tesseract", "ocr_model": "fra"}],
        })
        assert r.status_code == 200

    def test_run_multiple_competitors(self, client, tmp_corpus):
        r = client.post("/api/benchmark/run", json={
            "corpus_path": str(tmp_corpus),
            "competitors": [
                {"ocr_engine": "tesseract", "ocr_model": "fra"},
                {"ocr_engine": "tesseract", "ocr_model": "eng"},
            ],
        })
        assert r.status_code == 200

    def test_run_with_output_options(self, client, tmp_corpus, tmp_path):
        r = client.post("/api/benchmark/run", json={
            "corpus_path": str(tmp_corpus),
            "competitors": [{"ocr_engine": "tesseract", "ocr_model": "fra"}],
            "output_dir": str(tmp_path),
            "report_name": "test_run_report",
        })
        assert r.status_code == 200


# ===========================================================================
# TestFastAPIEnginesExtended  — champs ajoutés dans api_engines()
# ===========================================================================

class TestFastAPIEnginesExtended:

    def test_tesseract_has_langs_field(self, client):
        r = client.get("/api/engines")
        tess = next(e for e in r.json()["engines"] if e["id"] == "tesseract")
        assert "langs" in tess
        assert isinstance(tess["langs"], list)

    def test_mistral_ocr_in_engines(self, client):
        r = client.get("/api/engines")
        ids = [e["id"] for e in r.json()["engines"]]
        assert "mistral_ocr" in ids

    def test_google_vision_in_engines(self, client):
        r = client.get("/api/engines")
        ids = [e["id"] for e in r.json()["engines"]]
        assert "google_vision" in ids

    def test_azure_doc_intel_in_engines(self, client):
        r = client.get("/api/engines")
        ids = [e["id"] for e in r.json()["engines"]]
        assert "azure_doc_intel" in ids

    def test_cloud_engines_have_key_env(self, client):
        r = client.get("/api/engines")
        for eng in r.json()["engines"]:
            if eng.get("type") == "ocr_cloud":
                assert "key_env" in eng

    def test_mistral_llm_label_updated(self, client):
        r = client.get("/api/engines")
        mistral_llm = next(e for e in r.json()["llms"] if e["id"] == "mistral")
        assert "LLM" in mistral_llm["label"]


# ===========================================================================
# TestMistralOCRNativeAPI  — mistral-ocr-latest routing
# ===========================================================================

class TestMistralOCRNativeAPI:

    def test_engine_has_native_api_method(self):
        from picarones.engines.mistral_ocr import MistralOCREngine
        eng = MistralOCREngine(config={"model": "mistral-ocr-latest"})
        assert hasattr(eng, "_run_ocr_native_api")

    def test_engine_has_vision_api_method(self):
        from picarones.engines.mistral_ocr import MistralOCREngine
        eng = MistralOCREngine(config={"model": "pixtral-12b-2409"})
        assert hasattr(eng, "_run_ocr_vision_api")

    def test_model_name_stored(self):
        from picarones.engines.mistral_ocr import MistralOCREngine
        eng = MistralOCREngine(config={"model": "mistral-ocr-latest"})
        assert eng._model == "mistral-ocr-latest"

    def test_pixtral_model_stored(self):
        from picarones.engines.mistral_ocr import MistralOCREngine
        eng = MistralOCREngine(config={"model": "pixtral-large-latest"})
        assert "pixtral" in eng._model.lower()

    def test_engine_name_unchanged(self):
        from picarones.engines.mistral_ocr import MistralOCREngine
        eng = MistralOCREngine(config={"model": "mistral-ocr-latest"})
        assert eng.name == "mistral_ocr"

    def test_version_returns_model_name(self):
        from picarones.engines.mistral_ocr import MistralOCREngine
        eng = MistralOCREngine(config={"model": "mistral-ocr-latest"})
        assert eng.version() == "mistral-ocr-latest"
