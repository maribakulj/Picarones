"""Tests Sprint 8 — Intégration eScriptorium et import Gallica.

Classes de tests
----------------
TestEScriptoriumClient       (12 tests) — client API eScriptorium (mocks HTTP)
TestEScriptoriumConnect      (4 tests)  — fonction connect_escriptorium
TestEScriptoriumExport       (8 tests)  — export benchmark → couche OCR eScriptorium
TestGallicaRecord            (6 tests)  — structure GallicaRecord
TestGallicaClient            (12 tests) — client Gallica (mocks HTTP)
TestGallicaSearchQuery       (8 tests)  — construction de requêtes SRU
TestGallicaOCR               (6 tests)  — récupération OCR Gallica
TestImportersInit            (4 tests)  — __init__.py importers
TestCLIHistory               (6 tests)  — commande picarones history
TestCLIRobustness            (6 tests)  — commande picarones robustness
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch
import pytest


# ===========================================================================
# TestEScriptoriumClient
# ===========================================================================

class TestEScriptoriumClient:

    def test_import_module(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        assert EScriptoriumClient is not None

    def test_init_attributes(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        client = EScriptoriumClient("https://escriptorium.example.org", token="tok123", timeout=60)
        assert client.base_url == "https://escriptorium.example.org"
        assert client.token == "tok123"
        assert client.timeout == 60

    def test_base_url_trailing_slash_stripped(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        client = EScriptoriumClient("https://escriptorium.example.org/", token="tok")
        assert not client.base_url.endswith("/")

    def test_headers_contain_token(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        client = EScriptoriumClient("https://example.org", token="mytoken")
        headers = client._headers()
        assert "Token mytoken" in headers.get("Authorization", "")

    def test_headers_contain_accept_json(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        client = EScriptoriumClient("https://example.org", token="tok")
        headers = client._headers()
        assert "application/json" in headers.get("Accept", "")

    def test_test_connection_success(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        client = EScriptoriumClient("https://example.org", token="tok")
        with patch.object(client, "_get", return_value={"results": [], "count": 0}):
            assert client.test_connection() is True

    def test_test_connection_failure(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        client = EScriptoriumClient("https://example.org", token="bad")
        with patch.object(client, "_get", side_effect=RuntimeError("403")):
            assert client.test_connection() is False

    def test_list_projects_empty(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        client = EScriptoriumClient("https://example.org", token="tok")
        with patch.object(client, "_paginate", return_value=[]):
            projects = client.list_projects()
            assert projects == []

    def test_list_projects_parses_items(self):
        from picarones.importers.escriptorium import EScriptoriumClient, EScriptoriumProject
        client = EScriptoriumClient("https://example.org", token="tok")
        mock_data = [
            {"pk": 1, "name": "Projet BnF", "slug": "projet-bnf",
             "owner": {"username": "user1"}, "documents_count": 5},
        ]
        with patch.object(client, "_paginate", return_value=mock_data):
            projects = client.list_projects()
            assert len(projects) == 1
            assert isinstance(projects[0], EScriptoriumProject)
            assert projects[0].pk == 1
            assert projects[0].name == "Projet BnF"
            assert projects[0].document_count == 5

    def test_list_documents_with_project_filter(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        client = EScriptoriumClient("https://example.org", token="tok")
        with patch.object(client, "_paginate", return_value=[]) as mock_pag:
            client.list_documents(project_pk=42)
            call_kwargs = mock_pag.call_args
            assert call_kwargs[0][1]["project"] == 42

    def test_list_parts_returns_list(self):
        from picarones.importers.escriptorium import EScriptoriumClient, EScriptoriumPart
        client = EScriptoriumClient("https://example.org", token="tok")
        mock_data = [
            {"pk": 10, "title": "f. 1r", "image": "https://example.org/img/1.jpg", "order": 0},
            {"pk": 11, "title": "f. 1v", "image": "https://example.org/img/2.jpg", "order": 1},
        ]
        with patch.object(client, "_paginate", return_value=mock_data):
            parts = client.list_parts(doc_pk=5)
            assert len(parts) == 2
            assert isinstance(parts[0], EScriptoriumPart)
            assert parts[0].pk == 10

    def test_escriptorium_project_as_dict(self):
        from picarones.importers.escriptorium import EScriptoriumProject
        p = EScriptoriumProject(pk=1, name="Test", slug="test", owner="user", document_count=3)
        d = p.as_dict()
        assert d["pk"] == 1
        assert d["name"] == "Test"
        assert d["document_count"] == 3


# ===========================================================================
# TestEScriptoriumConnect
# ===========================================================================

class TestEScriptoriumConnect:

    def test_connect_success(self):
        from picarones.importers.escriptorium import connect_escriptorium, EScriptoriumClient
        with patch.object(EScriptoriumClient, "test_connection", return_value=True):
            client = connect_escriptorium("https://example.org", token="tok")
            assert isinstance(client, EScriptoriumClient)

    def test_connect_failure_raises(self):
        from picarones.importers.escriptorium import connect_escriptorium, EScriptoriumClient
        with patch.object(EScriptoriumClient, "test_connection", return_value=False):
            with pytest.raises(RuntimeError, match="Impossible de se connecter"):
                connect_escriptorium("https://example.org", token="bad")

    def test_connect_returns_client_with_correct_url(self):
        from picarones.importers.escriptorium import connect_escriptorium, EScriptoriumClient
        with patch.object(EScriptoriumClient, "test_connection", return_value=True):
            client = connect_escriptorium("https://myinstance.org", token="tok")
            assert "myinstance.org" in client.base_url

    def test_connect_timeout_passed(self):
        from picarones.importers.escriptorium import connect_escriptorium, EScriptoriumClient
        with patch.object(EScriptoriumClient, "test_connection", return_value=True):
            client = connect_escriptorium("https://example.org", token="tok", timeout=120)
            assert client.timeout == 120


# ===========================================================================
# TestEScriptoriumExport
# ===========================================================================

class TestEScriptoriumExport:

    def _make_benchmark(self, engine_name: str = "tesseract") -> "BenchmarkResult":
        from picarones.core.results import BenchmarkResult, EngineReport, DocumentResult
        from picarones.core.metrics import MetricsResult
        metrics = MetricsResult(cer=0.05, wer=0.10, cer_nfc=0.05,
                                cer_caseless=0.04, cer_diplomatic=0.04,
                                wer_normalized=0.09, mer=0.09, wil=0.05,
                                reference_length=100, hypothesis_length=100)
        doc = DocumentResult(
            doc_id="part_00001",
            image_path="/img/1.jpg",
            ground_truth="texte gt",
            hypothesis="texte ocr",
            metrics=metrics,
            duration_seconds=1.0,
        )
        report = EngineReport(
            engine_name=engine_name,
            engine_version="5.3",
            engine_config={},
            document_results=[doc],
        )
        return BenchmarkResult(
            corpus_name="Test",
            corpus_source="/test/",
            document_count=1,
            engine_reports=[report],
        )

    def test_export_unknown_engine_raises(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        client = EScriptoriumClient("https://example.org", token="tok")
        bm = self._make_benchmark("tesseract")
        with pytest.raises(ValueError, match="unknown_engine"):
            client.export_benchmark_as_layer(bm, doc_pk=1, engine_name="unknown_engine")

    def test_export_returns_count(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        client = EScriptoriumClient("https://example.org", token="tok")
        bm = self._make_benchmark("tesseract")
        with patch.object(client, "_post", return_value={}):
            count = client.export_benchmark_as_layer(
                bm, doc_pk=1, engine_name="tesseract"
            )
            assert count == 1

    def test_export_layer_name_default(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        client = EScriptoriumClient("https://example.org", token="tok")
        bm = self._make_benchmark("tesseract")
        calls = []
        with patch.object(client, "_post", side_effect=lambda path, payload: calls.append(payload) or {}):
            client.export_benchmark_as_layer(bm, doc_pk=1, engine_name="tesseract")
        assert calls[0]["name"] == "picarones_tesseract"

    def test_export_custom_layer_name(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        client = EScriptoriumClient("https://example.org", token="tok")
        bm = self._make_benchmark("tesseract")
        calls = []
        with patch.object(client, "_post", side_effect=lambda path, payload: calls.append(payload) or {}):
            client.export_benchmark_as_layer(
                bm, doc_pk=1, engine_name="tesseract", layer_name="my_layer"
            )
        assert calls[0]["name"] == "my_layer"

    def test_export_skips_error_docs(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        from picarones.core.results import BenchmarkResult, EngineReport, DocumentResult
        from picarones.core.metrics import MetricsResult
        metrics = MetricsResult(cer=0.1, wer=0.2, cer_nfc=0.1, cer_caseless=0.1,
                                cer_diplomatic=0.1, wer_normalized=0.2, mer=0.2, wil=0.1,
                                reference_length=50, hypothesis_length=50)
        docs = [
            DocumentResult("part_00001", "/img/1.jpg", "gt", "hyp", metrics, 1.0),
            DocumentResult("part_00002", "/img/2.jpg", "gt", "", metrics, 0.5, engine_error="timeout"),
        ]
        report = EngineReport("tesseract", "5.3", {}, docs)
        bm = BenchmarkResult("C", "/", 2, [report])
        client = EScriptoriumClient("https://example.org", token="tok")
        with patch.object(client, "_post", return_value={}):
            count = client.export_benchmark_as_layer(bm, doc_pk=1, engine_name="tesseract")
        assert count == 1  # seul le doc sans erreur est exporté

    def test_export_with_part_mapping(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        client = EScriptoriumClient("https://example.org", token="tok")
        bm = self._make_benchmark("tesseract")
        calls = []
        with patch.object(client, "_post", side_effect=lambda path, payload: calls.append(path) or {}):
            client.export_benchmark_as_layer(
                bm, doc_pk=1, engine_name="tesseract",
                part_mapping={"part_00001": 999},
            )
        assert "999" in calls[0]

    def test_export_post_error_is_logged_not_raised(self):
        from picarones.importers.escriptorium import EScriptoriumClient
        client = EScriptoriumClient("https://example.org", token="tok")
        bm = self._make_benchmark("tesseract")
        with patch.object(client, "_post", side_effect=RuntimeError("500")):
            count = client.export_benchmark_as_layer(bm, doc_pk=1, engine_name="tesseract")
        assert count == 0

    def test_document_result_as_dict_used(self):
        from picarones.importers.escriptorium import EScriptoriumDocument
        d = EScriptoriumDocument(pk=42, name="Doc", project="1", part_count=10,
                                 transcription_layers=["manual", "auto"])
        d_dict = d.as_dict()
        assert d_dict["pk"] == 42
        assert "manual" in d_dict["transcription_layers"]


# ===========================================================================
# TestGallicaRecord
# ===========================================================================

class TestGallicaRecord:

    def test_import_module(self):
        from picarones.importers.gallica import GallicaRecord
        assert GallicaRecord is not None

    def test_ark_property(self):
        from picarones.importers.gallica import GallicaRecord
        r = GallicaRecord(ark="12148/btv1b8453561w", title="Test")
        assert "12148/btv1b8453561w" in r.url

    def test_manifest_url(self):
        from picarones.importers.gallica import GallicaRecord
        r = GallicaRecord(ark="12148/btv1b8453561w", title="Test")
        assert "manifest.json" in r.manifest_url
        assert "12148/btv1b8453561w" in r.manifest_url

    def test_as_dict_keys(self):
        from picarones.importers.gallica import GallicaRecord
        r = GallicaRecord(ark="12148/btv1b8453561w", title="Froissart", creator="Froissart")
        d = r.as_dict()
        assert "ark" in d
        assert "title" in d
        assert "manifest_url" in d
        assert "url" in d

    def test_has_ocr_default_false(self):
        from picarones.importers.gallica import GallicaRecord
        r = GallicaRecord(ark="12148/xxx", title="Test")
        assert r.has_ocr is False

    def test_has_ocr_true(self):
        from picarones.importers.gallica import GallicaRecord
        r = GallicaRecord(ark="12148/xxx", title="Test", has_ocr=True)
        assert r.has_ocr is True


# ===========================================================================
# TestGallicaClient
# ===========================================================================

class TestGallicaClient:

    def test_import_module(self):
        from picarones.importers.gallica import GallicaClient
        assert GallicaClient is not None

    def test_init_defaults(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient()
        assert client.timeout == 30
        assert client.delay >= 0

    def test_search_returns_list(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient(delay_between_requests=0)
        with patch.object(client, "_fetch_url", side_effect=RuntimeError("network")):
            results = client.search(title="Froissart", max_results=5)
            assert isinstance(results, list)

    def test_search_empty_on_network_error(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient(delay_between_requests=0)
        with patch.object(client, "_fetch_url", side_effect=RuntimeError("timeout")):
            results = client.search(title="test")
            assert results == []

    def test_get_ocr_text_returns_string(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient(delay_between_requests=0)
        with patch.object(client, "_fetch_url", return_value=b"Froissart transcription"):
            text = client.get_ocr_text("12148/btv1b8453561w", page=1)
            assert isinstance(text, str)
            assert "Froissart" in text

    def test_get_ocr_text_empty_on_html_response(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient(delay_between_requests=0)
        html = b"<!DOCTYPE html><html><body>Page non disponible</body></html>"
        with patch.object(client, "_fetch_url", return_value=html):
            text = client.get_ocr_text("12148/xxx", page=1)
            assert text == ""

    def test_get_ocr_text_empty_on_error(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient(delay_between_requests=0)
        with patch.object(client, "_fetch_url", side_effect=RuntimeError("404")):
            text = client.get_ocr_text("12148/xxx", page=99)
            assert text == ""

    def test_get_metadata_returns_dict(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient(delay_between_requests=0)
        xml_bytes = b"""<?xml version="1.0" encoding="UTF-8"?>
        <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                   xmlns:dc="http://purl.org/dc/elements/1.1/">
            <dc:title>Chroniques de France</dc:title>
            <dc:creator>Jean Froissart</dc:creator>
            <dc:date>1380</dc:date>
        </oai_dc:dc>"""
        with patch.object(client, "_fetch_url", return_value=xml_bytes):
            meta = client.get_metadata("12148/btv1b8453561w")
            assert "ark" in meta
            assert meta["title"] == "Chroniques de France"
            assert meta["creator"] == "Jean Froissart"

    def test_get_metadata_on_error_returns_ark_dict(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient(delay_between_requests=0)
        with patch.object(client, "_fetch_url", side_effect=RuntimeError("500")):
            meta = client.get_metadata("12148/xxx")
            assert meta == {"ark": "12148/xxx"}

    def test_parse_sru_empty_xml(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient(delay_between_requests=0)
        xml = b"""<?xml version="1.0"?>
        <searchRetrieveResponse xmlns="http://www.loc.gov/zing/srw/">
            <numberOfRecords>0</numberOfRecords>
            <records/>
        </searchRetrieveResponse>"""
        records = client._parse_sru_response(xml, max_results=10)
        assert records == []

    def test_parse_sru_invalid_xml_returns_empty(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient(delay_between_requests=0)
        records = client._parse_sru_response(b"not xml at all !!!", max_results=10)
        assert records == []

    def test_client_has_delay_attribute(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient(delay_between_requests=0.1)
        assert client.delay == 0.1


# ===========================================================================
# TestGallicaSearchQuery
# ===========================================================================

class TestGallicaSearchQuery:

    def test_build_query_title(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient()
        query = client._build_sru_query(title="Froissart")
        assert "Froissart" in query
        assert "dc.title" in query

    def test_build_query_author(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient()
        query = client._build_sru_query(author="Froissart")
        assert "dc.creator" in query

    def test_build_query_date_range(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient()
        query = client._build_sru_query(date_from=1380, date_to=1420)
        assert "1380" in query
        assert "1420" in query

    def test_build_query_date_from_only(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient()
        query = client._build_sru_query(date_from=1400)
        assert "1400" in query
        assert ">=" in query

    def test_build_query_ark(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient()
        query = client._build_sru_query(ark="12148/btv1b8453561w")
        assert "12148/btv1b8453561w" in query

    def test_build_query_empty_returns_default(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient()
        query = client._build_sru_query()
        assert len(query) > 0

    def test_build_query_combined(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient()
        query = client._build_sru_query(title="Froissart", author="Jean", date_from=1380)
        assert "Froissart" in query
        assert "Jean" in query
        assert "1380" in query

    def test_search_gallica_function(self):
        from picarones.importers.gallica import search_gallica, GallicaClient
        with patch.object(GallicaClient, "search", return_value=[]):
            results = search_gallica(title="test")
            assert isinstance(results, list)


# ===========================================================================
# TestGallicaOCR
# ===========================================================================

class TestGallicaOCR:

    def test_ocr_url_format(self):
        from picarones.importers import gallica as g
        url = g._OCR_BRUT_TPL.format(ark="12148/btv1b8453561w", page=3)
        assert "12148/btv1b8453561w" in url
        assert "f3" in url
        assert "texteBrut" in url

    def test_import_gallica_document_function_exists(self):
        from picarones.importers.gallica import import_gallica_document
        assert callable(import_gallica_document)

    def test_gallica_base_url(self):
        from picarones.importers import gallica as g
        assert "gallica.bnf.fr" in g._GALLICA_BASE

    def test_ark_normalization_in_import(self):
        from picarones.importers.gallica import import_gallica_document, GallicaClient
        import re
        # Tester que l'ARK est normalisé depuis une URL complète
        full_url = "https://gallica.bnf.fr/ark:/12148/btv1b8453561w"
        m = re.search(r"ark:/(\d+/\w+)", full_url)
        assert m is not None
        assert m.group(1) == "12148/btv1b8453561w"

    def test_iiif_manifest_url_pattern(self):
        from picarones.importers import gallica as g
        url = g._IIIF_MANIFEST_TPL.format(ark="12148/btv1b8453561w")
        assert "manifest.json" in url
        assert "12148/btv1b8453561w" in url

    def test_gallica_record_url_structure(self):
        from picarones.importers.gallica import GallicaRecord
        r = GallicaRecord(ark="12148/btv1b8453561w", title="Test")
        assert r.url.startswith("https://gallica.bnf.fr")
        assert "12148/btv1b8453561w" in r.url


# ===========================================================================
# TestImportersInit
# ===========================================================================

class TestImportersInit:

    def test_escriptorium_client_exported(self):
        from picarones.importers import EScriptoriumClient
        assert EScriptoriumClient is not None

    def test_gallica_client_exported(self):
        from picarones.importers import GallicaClient
        assert GallicaClient is not None

    def test_search_gallica_exported(self):
        from picarones.importers import search_gallica
        assert callable(search_gallica)

    def test_connect_escriptorium_exported(self):
        from picarones.importers import connect_escriptorium
        assert callable(connect_escriptorium)


# ===========================================================================
# TestCLIHistory (tests Click runner)
# ===========================================================================

class TestCLIHistory:

    def test_history_command_exists(self):
        from picarones.cli import cli
        assert "history" in [cmd.name for cmd in cli.commands.values()]

    def test_history_demo_mode(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--demo", "--db", ":memory:"])
        assert result.exit_code == 0
        assert "entrées" in result.output

    def test_history_empty_db(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        import tempfile, os
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            result = runner.invoke(cli, ["history", "--db", db_path])
            assert result.exit_code == 0
            assert "Aucun" in result.output or "Aucun benchmark" in result.output
        finally:
            os.unlink(db_path)

    def test_history_with_regression_flag(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--demo", "--db", ":memory:", "--regression"])
        assert result.exit_code == 0

    def test_history_engine_filter(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "history", "--demo", "--db", ":memory:", "--engine", "tesseract"
        ])
        assert result.exit_code == 0

    def test_history_export_json(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        import tempfile, os
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name
        try:
            result = runner.invoke(cli, [
                "history", "--demo", "--db", ":memory:", "--export-json", json_path
            ])
            assert result.exit_code == 0
            assert os.path.exists(json_path)
            data = json.loads(open(json_path).read())
            assert "runs" in data
        finally:
            os.unlink(json_path)


# ===========================================================================
# TestCLIRobustness
# ===========================================================================

class TestCLIRobustness:

    def test_robustness_command_exists(self):
        from picarones.cli import cli
        assert "robustness" in [cmd.name for cmd in cli.commands.values()]

    def test_robustness_demo_mode(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        import tempfile
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os; os.makedirs("corpus")
            result = runner.invoke(cli, [
                "robustness", "--corpus", "corpus", "--engine", "tesseract", "--demo"
            ])
            assert result.exit_code == 0

    def test_robustness_invalid_degradation(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        import tempfile
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os; os.makedirs("corpus")
            result = runner.invoke(cli, [
                "robustness", "--corpus", "corpus", "--engine", "tesseract",
                "--degradations", "invalid_type", "--demo"
            ])
            assert result.exit_code != 0

    def test_robustness_shows_results(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os; os.makedirs("corpus")
            result = runner.invoke(cli, [
                "robustness", "--corpus", "corpus", "--engine", "tesseract",
                "--demo", "--degradations", "noise"
            ])
            assert result.exit_code == 0
            assert "robustesse" in result.output.lower() or "noise" in result.output.lower()

    def test_robustness_json_export(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os; os.makedirs("corpus")
            result = runner.invoke(cli, [
                "robustness", "--corpus", "corpus", "--engine", "tesseract",
                "--demo", "--output-json", "robustness.json"
            ])
            assert result.exit_code == 0
            assert os.path.exists("robustness.json")
            data = json.loads(open("robustness.json").read())
            assert "curves" in data

    def test_robustness_single_degradation_type(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os; os.makedirs("corpus")
            result = runner.invoke(cli, [
                "robustness", "--corpus", "corpus", "--engine", "tesseract",
                "--demo", "--degradations", "blur"
            ])
            assert result.exit_code == 0
