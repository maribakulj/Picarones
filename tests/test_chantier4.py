"""Tests du chantier 4 (post-Sprint 97) : LLM + Gallica/IIIF + CLI workflows.

Couvre :

- Sous-chantier 4.A : ``normalize_llm_content`` + ``log_http_error``
  factorisés dans :mod:`picarones.llm.base`, propagés aux 4 adapters.
- Sous-chantier 4.B : helpers HTTP factorisés dans
  :mod:`picarones.importers._http`, Gallica et IIIF y délèguent.
- Sous-chantier 4.C : 3 nouvelles sous-commandes CLI ``diagnose``,
  ``economics``, ``edition`` qui mappent un profil de calcul
  (chantier 2) à un workflow.
"""

from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────────────────
# 4.A — LLM base helpers
# ──────────────────────────────────────────────────────────────────────────


class TestNormalizeLlmContent:
    def test_str_passes_through(self):
        from picarones.llm.base import normalize_llm_content

        assert normalize_llm_content("hello") == "hello"
        # Idempotence : retourne l'objet exact pour str
        s = "test"
        assert normalize_llm_content(s) is s

    def test_none_returns_empty(self):
        from picarones.llm.base import normalize_llm_content

        assert normalize_llm_content(None) == ""

    def test_empty_string_passes(self):
        from picarones.llm.base import normalize_llm_content

        assert normalize_llm_content("") == ""

    def test_list_of_chunks_with_text_attr(self):
        """Cas Mistral SDK : list[ContentChunk]. Sprint 15 fix."""
        from picarones.llm.base import normalize_llm_content

        class MockChunk:
            def __init__(self, text):
                self.text = text

        result = normalize_llm_content([MockChunk("hello "), MockChunk("world")])
        assert result == "hello world"

    def test_list_of_dicts_with_text_key(self):
        """Cas Anthropic SDK : list[dict] avec clé 'text'."""
        from picarones.llm.base import normalize_llm_content

        result = normalize_llm_content([{"text": "a"}, {"text": "b"}])
        assert result == "ab"

    def test_list_of_strings(self):
        from picarones.llm.base import normalize_llm_content

        assert normalize_llm_content(["foo", "bar"]) == "foobar"

    def test_mixed_list(self):
        from picarones.llm.base import normalize_llm_content

        class MockChunk:
            def __init__(self, text):
                self.text = text

        result = normalize_llm_content([
            MockChunk("a"), "b", {"text": "c"},
        ])
        assert result == "abc"

    def test_none_in_list_skipped(self):
        from picarones.llm.base import normalize_llm_content

        assert normalize_llm_content([None, "a", None, "b"]) == "ab"

    def test_object_with_text_attribute(self):
        from picarones.llm.base import normalize_llm_content

        class TextHolder:
            text = "hello"
        assert normalize_llm_content(TextHolder()) == "hello"


class TestLogHttpError:
    def test_401_logs_invalid_key(self, caplog):
        from picarones.llm.base import log_http_error

        class FakeExc(Exception):
            status_code = 401

        with caplog.at_level("WARNING"):
            log_http_error("OpenAIAdapter", "gpt-4o", FakeExc("Unauthorized"),
                           env_var="OPENAI_API_KEY")
        assert any("401" in r.message and "OPENAI_API_KEY" in r.message
                   for r in caplog.records)

    def test_429_logs_rate_limit(self, caplog):
        from picarones.llm.base import log_http_error

        class FakeExc(Exception):
            status_code = 429

        with caplog.at_level("WARNING"):
            log_http_error("MistralAdapter", "mistral-large", FakeExc("Too Many"))
        assert any("429" in r.message and "rate" in r.message.lower()
                   for r in caplog.records)

    def test_5xx_logs_server_error(self, caplog):
        from picarones.llm.base import log_http_error

        class FakeExc(Exception):
            status_code = 503

        with caplog.at_level("WARNING"):
            log_http_error("AnthropicAdapter", "claude-sonnet", FakeExc("Service unavailable"))
        assert any("503" in r.message and "serveur" in r.message.lower()
                   for r in caplog.records)

    def test_no_status_code_logs_generic(self, caplog):
        from picarones.llm.base import log_http_error

        with caplog.at_level("WARNING"):
            log_http_error("Foo", "bar", ValueError("random"))
        # Doit produire un warning (générique)
        assert any("Foo" in r.message for r in caplog.records)


class TestLlmAdaptersInheritEnvVar:
    """Le chantier 4 a ajouté ``api_key_env_var`` aux 3 adapters cloud."""

    def test_mistral_declares_env_var(self):
        from picarones.llm.mistral_adapter import MistralAdapter
        assert MistralAdapter.api_key_env_var == "MISTRAL_API_KEY"

    def test_openai_declares_env_var(self):
        from picarones.llm.openai_adapter import OpenAIAdapter
        assert OpenAIAdapter.api_key_env_var == "OPENAI_API_KEY"

    def test_anthropic_declares_env_var(self):
        from picarones.llm.anthropic_adapter import AnthropicAdapter
        assert AnthropicAdapter.api_key_env_var == "ANTHROPIC_API_KEY"

    def test_ollama_no_env_var(self):
        """Ollama est local — pas de clé API."""
        from picarones.llm.ollama_adapter import OllamaAdapter
        assert OllamaAdapter.api_key_env_var is None


# ──────────────────────────────────────────────────────────────────────────
# 4.B — Helpers HTTP factorisés (Gallica → IIIF fusion)
# ──────────────────────────────────────────────────────────────────────────


class TestHttpHelpers:
    def test_validate_http_url_accepts_https(self):
        from picarones.importers._http import validate_http_url
        validate_http_url("https://gallica.bnf.fr/test")  # ne lève pas

    def test_validate_http_url_accepts_http(self):
        from picarones.importers._http import validate_http_url
        validate_http_url("http://localhost:8080/x")

    @pytest.mark.parametrize("scheme", ["file", "ftp", "data", "javascript", "ssh"])
    def test_validate_http_url_rejects_other_schemes(self, scheme):
        from picarones.importers._http import validate_http_url
        with pytest.raises(ValueError, match="non autorisé"):
            validate_http_url(f"{scheme}://example.com/x")


class TestIiifAliasesDelegateToHttp:
    """Les noms ``_validate_url`` et ``_download_url`` exposés depuis
    :mod:`picarones.importers.iiif` doivent rester disponibles
    (rétrocompat des tests Sprint 4) — ils délèguent aux helpers
    factorisés."""

    def test_iiif_validate_url_is_alias(self):
        from picarones.importers import iiif
        from picarones.importers._http import validate_http_url
        assert iiif._validate_url is validate_http_url

    def test_iiif_download_url_is_alias(self):
        from picarones.importers import iiif
        from picarones.importers._http import download_url
        assert iiif._download_url is download_url


class TestGallicaDelegatesToHttp:
    def test_gallica_validate_url_delegates(self):
        from picarones.importers.gallica import GallicaClient
        client = GallicaClient()
        # Doit accepter https
        client._validate_url("https://gallica.bnf.fr/x")
        # Doit rejeter un schéma invalide via le helper factorisé
        with pytest.raises(ValueError, match="non autorisé"):
            client._validate_url("file:///etc/passwd")

    def test_gallica_uses_iiif_for_image_download(self):
        """``GallicaClient.import_document`` délègue à IIIFImporter."""
        # Lecture statique du source — pas d'appel réseau
        from pathlib import Path
        gallica_src = (
            Path(__file__).parent.parent
            / "picarones" / "importers" / "gallica.py"
        ).read_text(encoding="utf-8")
        # Confirme que Gallica importe IIIFImporter
        assert "from picarones.importers.iiif import IIIFImporter" in gallica_src


# ──────────────────────────────────────────────────────────────────────────
# 4.C — Workflows CLI dédiés
# ──────────────────────────────────────────────────────────────────────────


class TestCliWorkflows:
    def test_three_new_commands_registered(self):
        from pathlib import Path

        cli_src = (
            Path(__file__).parent.parent / "picarones" / "cli.py"
        ).read_text(encoding="utf-8")
        # Vérification statique : les 3 commandes existent
        assert '@cli.command("diagnose")' in cli_src
        assert '@cli.command("economics")' in cli_src
        assert '@cli.command("edition")' in cli_src
        assert "def diagnose_cmd(" in cli_src
        assert "def economics_cmd(" in cli_src
        assert "def edition_cmd(" in cli_src

    def test_workflows_map_correct_profile(self):
        from pathlib import Path
        cli_src = (
            Path(__file__).parent.parent / "picarones" / "cli.py"
        ).read_text(encoding="utf-8")
        # Chaque commande doit fixer le bon profil
        # diagnose → diagnostics, economics → economics, edition → philological
        assert 'profile="diagnostics"' in cli_src
        assert 'profile="economics"' in cli_src
        assert 'profile="philological"' in cli_src

    def test_run_workflow_helper_exists(self):
        """Le helper commun ``_run_workflow`` factorise la logique des
        4 commandes (run + diagnose + economics + edition) — un seul
        endroit pour patcher si la logique évolue."""
        import ast
        from pathlib import Path

        cli_src = (
            Path(__file__).parent.parent / "picarones" / "cli.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(cli_src)
        funcs = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        assert "_run_workflow" in funcs

    @pytest.mark.parametrize("cmd_name", ["diagnose", "economics", "edition"])
    def test_command_help_works(self, cmd_name):
        """Les 3 commandes répondent à --help sans crash."""
        try:
            from click.testing import CliRunner

            from picarones.cli import cli as cli_group
        except ImportError:
            pytest.skip("click non installé")

        runner = CliRunner()
        result = runner.invoke(cli_group, [cmd_name, "--help"])
        assert result.exit_code == 0, result.output
        assert "--corpus" in result.output
        assert "--engines" in result.output
