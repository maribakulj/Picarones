"""Tests d'unité pour ``picarones.adapters.corpus.htr_united``.

Phase 6 du chantier post-rewrite : extraits du god-file
``tests/web/test_sprint6_web_interface.py`` (1563 LOC) qui mélangeait
des tests d'unité (HTR-United, HuggingFace) avec des tests
d'intégration FastAPI.  Ces 4 classes sont totalement autonomes —
elles testent le module ``adapters/corpus/htr_united.py`` sans
toucher au web.

Couvre :

- ``HTRUnitedEntry`` (dataclass) : ``from_dict`` / ``as_dict`` /
  ``century_str``, défauts, round-trip.
- ``HTRUnitedCatalogue`` : ``from_demo`` (taille, source),
  ``get_by_id``, ``available_languages``, ``available_scripts``.
- Méthode ``search()`` : filtres par query, language, script,
  century_min, combinaisons.
- ``import_htr_united_corpus`` : tests réseau marqués ``network``
  (timeout 30 s sur GitHub raw, exclus du run local par défaut).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures partagées
# ---------------------------------------------------------------------------

@pytest.fixture
def htr_catalogue():
    from picarones.adapters.corpus.htr_united import HTRUnitedCatalogue
    return HTRUnitedCatalogue.from_demo()


# ===========================================================================
# HTRUnitedEntry — dataclass
# ===========================================================================

class TestHTRUnitedEntry:

    def test_from_dict_basic(self):
        from picarones.adapters.corpus.htr_united import HTRUnitedEntry
        d = {
            "id": "test-corpus", "title": "Test Corpus", "url": "https://github.com/test/corpus",
            "language": ["French"], "script": ["Gothic"], "century": [14, 15],
            "institution": "Test Org", "description": "Un corpus de test.", "license": "CC-BY 4.0",
            "lines": 5000, "format": "ALTO", "tags": ["test", "médiéval"],
        }
        e = HTRUnitedEntry.from_dict(d)
        assert e.id == "test-corpus"
        assert e.title == "Test Corpus"
        assert e.language == ["French"]
        assert e.lines == 5000

    def test_as_dict_roundtrip(self):
        from picarones.adapters.corpus.htr_united import HTRUnitedEntry
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
        from picarones.adapters.corpus.htr_united import HTRUnitedEntry
        e = HTRUnitedEntry(id="x", title="x", url="x", century=[12, 14])
        cs = e.century_str
        assert "XIIe" in cs
        assert "XIVe" in cs

    def test_century_str_single(self):
        from picarones.adapters.corpus.htr_united import HTRUnitedEntry
        e = HTRUnitedEntry(id="x", title="x", url="x", century=[19])
        assert "XIXe" in e.century_str

    def test_default_fields(self):
        from picarones.adapters.corpus.htr_united import HTRUnitedEntry
        e = HTRUnitedEntry(id="minimal", title="Min", url="http://x")
        assert e.language == []
        assert e.lines == 0
        assert e.format == "ALTO"
        assert e.tags == []

    def test_from_dict_missing_fields(self):
        from picarones.adapters.corpus.htr_united import HTRUnitedEntry
        e = HTRUnitedEntry.from_dict({"id": "sparse", "title": "Sparse"})
        assert e.id == "sparse"
        assert e.institution == ""
        assert e.lines == 0

    def test_as_dict_has_all_keys(self):
        from picarones.adapters.corpus.htr_united import HTRUnitedEntry
        e = HTRUnitedEntry(id="k", title="K", url="http://k")
        d = e.as_dict()
        for key in ["id", "title", "url", "language", "script", "century",
                    "institution", "description", "license", "lines", "format", "tags"]:
            assert key in d, f"Missing key: {key}"

    def test_url_preserved(self):
        from picarones.adapters.corpus.htr_united import HTRUnitedEntry
        url = "https://github.com/HTR-United/cremma-medieval"
        e = HTRUnitedEntry(id="c", title="CREMMA", url=url)
        assert e.url == url


# ===========================================================================
# HTRUnitedCatalogue — listing
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
# HTRUnitedCatalogue.search — filtres
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
            assert any("french" in lg.lower() for lg in r.language)

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
# import_htr_united_corpus — tests réseau (skippés par défaut)
# ===========================================================================

@pytest.mark.network
class TestHTRUnitedImport:
    """Tests qui hit GitHub via ``urllib.request.urlopen(timeout=30)``.

    Marqués ``network`` (Sprint A5) pour être exclus du run local par
    défaut (sandbox sans accès réseau → 4 timeouts de 30s = bloque la
    suite). La CI réseau-friendly les exécute via ``pytest -m network``.
    """

    def test_import_creates_meta_file(self, tmp_path, htr_catalogue):
        from picarones.adapters.corpus.htr_united import import_htr_united_corpus
        entry = htr_catalogue.entries[0]
        result = import_htr_united_corpus(entry, tmp_path, max_samples=5)
        meta_file = Path(result["metadata_file"])
        assert meta_file.exists()

    def test_import_meta_content(self, tmp_path, htr_catalogue):
        from picarones.adapters.corpus.htr_united import import_htr_united_corpus
        entry = htr_catalogue.entries[0]
        result = import_htr_united_corpus(entry, tmp_path, max_samples=5)
        meta = json.loads(Path(result["metadata_file"]).read_text())
        assert meta["source"] == "htr-united"
        assert meta["entry_id"] == entry.id

    def test_import_returns_dict_keys(self, tmp_path, htr_catalogue):
        from picarones.adapters.corpus.htr_united import import_htr_united_corpus
        entry = htr_catalogue.entries[0]
        result = import_htr_united_corpus(entry, tmp_path, max_samples=5)
        for k in ["entry_id", "title", "output_dir", "files_imported", "metadata_file"]:
            assert k in result, f"Missing key: {k}"

    def test_import_creates_output_dir(self, tmp_path, htr_catalogue):
        from picarones.adapters.corpus.htr_united import import_htr_united_corpus
        entry = htr_catalogue.entries[0]
        new_dir = tmp_path / "new_subdir" / "corpus"
        import_htr_united_corpus(entry, new_dir, max_samples=5)
        assert new_dir.exists()
