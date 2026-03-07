"""Tests Sprint 9 — Documentation, packaging et intégration finale.

Classes de tests
----------------
TestVersion              (4 tests)  — version cohérente dans tous les fichiers
TestMainModule           (3 tests)  — python -m picarones fonctionne
TestMakefile             (5 tests)  — Makefile syntaxe et cibles
TestDockerfile           (6 tests)  — Dockerfile structure et commandes
TestDockerCompose        (5 tests)  — docker-compose.yml structure
TestCIWorkflow           (6 tests)  — .github/workflows/ci.yml structure
TestPyInstallerSpec      (4 tests)  — picarones.spec structure
TestCLIDemoEndToEnd      (6 tests)  — picarones demo bout en bout
TestReadme               (5 tests)  — README.md complet et bilingue
TestInstallMd            (4 tests)  — INSTALL.md contenu
TestChangelog            (5 tests)  — CHANGELOG.md contenu et structure
TestContributing         (4 tests)  — CONTRIBUTING.md contenu
"""

from __future__ import annotations

import re
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent


# ===========================================================================
# TestVersion
# ===========================================================================

class TestVersion:

    def test_version_in_init(self):
        from picarones import __version__
        assert __version__ == "1.0.0"

    def test_version_in_pyproject(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'version = "1.0.0"' in pyproject

    def test_version_cli(self):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_version_consistent(self):
        """La version dans __init__.py et pyproject.toml doit être identique."""
        from picarones import __version__
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        m = re.search(r'version\s*=\s*"([^"]+)"', pyproject)
        assert m is not None
        pyproject_version = m.group(1)
        assert __version__ == pyproject_version, (
            f"Version incohérente : __init__.py={__version__} vs pyproject.toml={pyproject_version}"
        )


# ===========================================================================
# TestMainModule
# ===========================================================================

class TestMainModule:

    def test_main_module_exists(self):
        main_path = ROOT / "picarones" / "__main__.py"
        assert main_path.exists(), "picarones/__main__.py est manquant"

    def test_main_imports_cli(self):
        content = (ROOT / "picarones" / "__main__.py").read_text(encoding="utf-8")
        assert "from picarones.cli import cli" in content

    def test_main_importable(self):
        import importlib
        mod = importlib.import_module("picarones.__main__")
        assert hasattr(mod, "cli")


# ===========================================================================
# TestMakefile
# ===========================================================================

class TestMakefile:

    @pytest.fixture
    def makefile(self):
        path = ROOT / "Makefile"
        assert path.exists(), "Makefile est manquant"
        return path.read_text(encoding="utf-8")

    def test_makefile_exists(self):
        assert (ROOT / "Makefile").exists()

    def test_has_install_target(self, makefile):
        assert "install:" in makefile

    def test_has_test_target(self, makefile):
        assert "test:" in makefile

    def test_has_demo_target(self, makefile):
        assert "demo:" in makefile

    def test_has_docker_build_target(self, makefile):
        assert "docker-build:" in makefile

    def test_has_help_target(self, makefile):
        assert "help:" in makefile


# ===========================================================================
# TestDockerfile
# ===========================================================================

class TestDockerfile:

    @pytest.fixture
    def dockerfile(self):
        path = ROOT / "Dockerfile"
        assert path.exists(), "Dockerfile est manquant"
        return path.read_text(encoding="utf-8")

    def test_dockerfile_exists(self):
        assert (ROOT / "Dockerfile").exists()

    def test_has_python_base(self, dockerfile):
        assert "python:3.11" in dockerfile

    def test_has_tesseract_install(self, dockerfile):
        assert "tesseract-ocr" in dockerfile

    def test_has_picarones_serve_cmd(self, dockerfile):
        assert "picarones" in dockerfile
        assert "serve" in dockerfile
        assert "0.0.0.0" in dockerfile

    def test_has_workdir(self, dockerfile):
        assert "WORKDIR" in dockerfile

    def test_has_healthcheck(self, dockerfile):
        assert "HEALTHCHECK" in dockerfile


# ===========================================================================
# TestDockerCompose
# ===========================================================================

class TestDockerCompose:

    @pytest.fixture
    def compose(self):
        path = ROOT / "docker-compose.yml"
        assert path.exists(), "docker-compose.yml est manquant"
        return path.read_text(encoding="utf-8")

    def test_compose_exists(self):
        assert (ROOT / "docker-compose.yml").exists()

    def test_has_picarones_service(self, compose):
        assert "picarones:" in compose

    def test_has_ollama_service(self, compose):
        assert "ollama" in compose

    def test_has_port_mapping(self, compose):
        assert "8000" in compose

    def test_has_volume_for_history(self, compose):
        assert "picarones_history" in compose


# ===========================================================================
# TestCIWorkflow
# ===========================================================================

class TestCIWorkflow:

    @pytest.fixture
    def ci(self):
        path = ROOT / ".github" / "workflows" / "ci.yml"
        assert path.exists(), ".github/workflows/ci.yml est manquant"
        return path.read_text(encoding="utf-8")

    def test_ci_exists(self):
        assert (ROOT / ".github" / "workflows" / "ci.yml").exists()

    def test_has_python_311(self, ci):
        assert "3.11" in ci

    def test_has_python_312(self, ci):
        assert "3.12" in ci

    def test_has_linux_macos_windows(self, ci):
        assert "ubuntu-latest" in ci
        assert "macos-latest" in ci
        assert "windows-latest" in ci

    def test_has_pytest_step(self, ci):
        assert "pytest" in ci

    def test_has_demo_job(self, ci):
        assert "demo" in ci


# ===========================================================================
# TestPyInstallerSpec
# ===========================================================================

class TestPyInstallerSpec:

    @pytest.fixture
    def spec(self):
        path = ROOT / "picarones.spec"
        assert path.exists(), "picarones.spec est manquant"
        return path.read_text(encoding="utf-8")

    def test_spec_exists(self):
        assert (ROOT / "picarones.spec").exists()

    def test_spec_has_analysis(self, spec):
        assert "Analysis(" in spec

    def test_spec_has_picarones_cli(self, spec):
        assert "picarones.cli" in spec

    def test_spec_has_exe(self, spec):
        assert "EXE(" in spec


# ===========================================================================
# TestCLIDemoEndToEnd
# ===========================================================================

class TestCLIDemoEndToEnd:

    def test_demo_runs_without_error(self, tmp_path):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "demo", "--docs", "3",
            "--output", str(tmp_path / "test.html"),
        ])
        assert result.exit_code == 0, f"demo a échoué : {result.output}"

    def test_demo_generates_html_file(self, tmp_path):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        output = tmp_path / "rapport.html"
        runner.invoke(cli, ["demo", "--docs", "3", "--output", str(output)])
        assert output.exists()

    def test_demo_html_contains_expected_content(self, tmp_path):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        output = tmp_path / "rapport.html"
        runner.invoke(cli, ["demo", "--docs", "3", "--output", str(output)])
        content = output.read_text(encoding="utf-8")
        assert "Picarones" in content
        assert "CER" in content
        assert len(content) > 50_000, f"Rapport trop petit : {len(content):,} octets"

    def test_demo_with_history_flag(self, tmp_path):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "demo", "--docs", "3",
            "--output", str(tmp_path / "test.html"),
            "--with-history",
        ])
        assert result.exit_code == 0
        assert "CER" in result.output

    def test_demo_with_robustness_flag(self, tmp_path):
        from click.testing import CliRunner
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "demo", "--docs", "3",
            "--output", str(tmp_path / "test.html"),
            "--with-robustness",
        ])
        assert result.exit_code == 0

    def test_demo_with_json_output(self, tmp_path):
        from click.testing import CliRunner
        from picarones.cli import cli
        import json
        runner = CliRunner()
        json_out = tmp_path / "results.json"
        result = runner.invoke(cli, [
            "demo", "--docs", "3",
            "--output", str(tmp_path / "test.html"),
            "--json-output", str(json_out),
        ])
        assert result.exit_code == 0
        assert json_out.exists()
        data = json.loads(json_out.read_text())
        assert "engine_reports" in data


# ===========================================================================
# TestReadme
# ===========================================================================

class TestReadme:

    @pytest.fixture
    def readme(self):
        path = ROOT / "README.md"
        assert path.exists()
        return path.read_text(encoding="utf-8")

    def test_readme_has_french_section(self, readme):
        assert "Fonctionnalités" in readme or "Picarones" in readme

    def test_readme_has_english_section(self, readme):
        assert "English" in readme or "Quick Start" in readme

    def test_readme_has_installation(self, readme):
        assert "Installation" in readme
        assert "pip install" in readme

    def test_readme_has_cli_examples(self, readme):
        assert "picarones demo" in readme
        assert "picarones run" in readme

    def test_readme_has_engines_table(self, readme):
        assert "Tesseract" in readme
        assert "Pero OCR" in readme


# ===========================================================================
# TestInstallMd
# ===========================================================================

class TestInstallMd:

    @pytest.fixture
    def install(self):
        path = ROOT / "INSTALL.md"
        assert path.exists(), "INSTALL.md est manquant"
        return path.read_text(encoding="utf-8")

    def test_has_linux_section(self, install):
        assert "Linux" in install or "Ubuntu" in install

    def test_has_macos_section(self, install):
        assert "macOS" in install

    def test_has_windows_section(self, install):
        assert "Windows" in install

    def test_has_docker_section(self, install):
        assert "Docker" in install


# ===========================================================================
# TestChangelog
# ===========================================================================

class TestChangelog:

    @pytest.fixture
    def changelog(self):
        path = ROOT / "CHANGELOG.md"
        assert path.exists(), "CHANGELOG.md est manquant"
        return path.read_text(encoding="utf-8")

    def test_has_sprint1(self, changelog):
        assert "Sprint 1" in changelog or "0.1.0" in changelog

    def test_has_sprint8(self, changelog):
        assert "Sprint 8" in changelog or "0.8.0" in changelog

    def test_has_sprint9(self, changelog):
        assert "Sprint 9" in changelog or "1.0.0" in changelog

    def test_has_versions(self, changelog):
        # Au moins 2 versions documentées
        versions = re.findall(r"\[[\d.]+\]", changelog)
        assert len(versions) >= 2

    def test_has_date(self, changelog):
        assert "2025" in changelog


# ===========================================================================
# TestContributing
# ===========================================================================

class TestContributing:

    @pytest.fixture
    def contrib(self):
        path = ROOT / "CONTRIBUTING.md"
        assert path.exists(), "CONTRIBUTING.md est manquant"
        return path.read_text(encoding="utf-8")

    def test_has_how_to_add_engine(self, contrib):
        assert "moteur" in contrib.lower() or "engine" in contrib.lower()

    def test_has_tests_section(self, contrib):
        assert "test" in contrib.lower()

    def test_has_pull_request_section(self, contrib):
        assert "pull request" in contrib.lower() or "PR" in contrib

    def test_has_code_style(self, contrib):
        assert "Google" in contrib or "docstring" in contrib.lower() or "style" in contrib.lower()
