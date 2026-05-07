"""Tests Sprint 70 — CLI pipeline + loader YAML.

Couvre :

1. ``_resolve_class`` : dotted path valide, module manquant,
   classe manquante, chemin invalide.
2. ``load_pipeline_spec_from_dict`` : spec valide, ``name``
   manquant, ``steps`` manquants, étape sans ``module``, args
   invalides, classe non BaseModule, ``inputs_from`` valide,
   ``inputs_from`` avec type inconnu.
3. ``load_pipeline_spec_from_yaml`` : fichier introuvable,
   YAML invalide, document complet.
4. ``load_comparison_specs_*`` : champ ``pipelines`` requis, N
   specs construites.
5. CLI ``picarones pipeline run`` : exécution end-to-end avec un
   MockOCR référencé via dotted path, sortie JSON et HTML.
6. CLI ``picarones pipeline compare`` : comparaison avec ranking
   affiché, output HTML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from picarones.domain.artifacts import ArtifactType

from picarones.domain.module_protocol import BaseModule
from picarones.measurements.pipeline_spec_loader import (
    PipelineSpecLoadError,
    _resolve_class,
    load_comparison_specs_from_dict,
    load_pipeline_spec_from_dict,
    load_pipeline_spec_from_yaml,
)


# Module de test au top-level pour pouvoir être référencé par dotted path.
class _CLIMockOCR(BaseModule):
    input_types = (ArtifactType.IMAGE,)
    output_types = (ArtifactType.TEXT,)
    execution_mode: Any = "io"

    def __init__(self, fixed_text: str = "hello world") -> None:
        self._fixed = fixed_text

    @property
    def name(self) -> str:
        return "cli-mock-ocr"

    def process(self, inputs):
        return {ArtifactType.TEXT: self._fixed}


class _NotABaseModule:
    pass


# ──────────────────────────────────────────────────────────────────────────
# 1. _resolve_class
# ──────────────────────────────────────────────────────────────────────────


class TestResolveClass:
    def test_valid_dotted_path(self) -> None:
        cls = _resolve_class(
            "tests.cli.test_sprint70_pipeline_cli._CLIMockOCR",
        )
        assert cls is _CLIMockOCR

    def test_missing_dot(self) -> None:
        with pytest.raises(PipelineSpecLoadError, match="invalide"):
            _resolve_class("invalid_no_dot")

    def test_module_not_found(self) -> None:
        with pytest.raises(PipelineSpecLoadError, match="introuvable"):
            _resolve_class("non_existing_module_xyz.Foo")

    def test_class_not_in_module(self) -> None:
        with pytest.raises(PipelineSpecLoadError, match="introuvable"):
            _resolve_class(
                "tests.cli.test_sprint70_pipeline_cli.DoesNotExist",
            )

    def test_target_is_not_a_class(self) -> None:
        with pytest.raises(PipelineSpecLoadError, match="n'est pas une classe"):
            # sys.path est un attribut du module sys, pas une classe
            _resolve_class("sys.path")


# ──────────────────────────────────────────────────────────────────────────
# 2. load_pipeline_spec_from_dict
# ──────────────────────────────────────────────────────────────────────────


class TestLoadFromDict:
    def test_valid_minimal(self) -> None:
        data = {
            "name": "ocr_only",
            "steps": [
                {
                    "name": "ocr",
                    "module": "tests.cli.test_sprint70_pipeline_cli._CLIMockOCR",
                },
            ],
        }
        spec = load_pipeline_spec_from_dict(data)
        assert spec.name == "ocr_only"
        assert len(spec.steps) == 1
        assert spec.steps[0].name == "ocr"
        assert isinstance(spec.steps[0].module, _CLIMockOCR)

    def test_with_args(self) -> None:
        data = {
            "name": "ocr_with_args",
            "steps": [
                {
                    "name": "ocr",
                    "module": "tests.cli.test_sprint70_pipeline_cli._CLIMockOCR",
                    "args": {"fixed_text": "custom output"},
                },
            ],
        }
        spec = load_pipeline_spec_from_dict(data)
        assert spec.steps[0].module._fixed == "custom output"

    def test_missing_name(self) -> None:
        data = {"steps": [{"name": "x", "module": "foo.bar"}]}
        with pytest.raises(PipelineSpecLoadError, match="``name``"):
            load_pipeline_spec_from_dict(data)

    def test_missing_steps(self) -> None:
        data = {"name": "p"}
        with pytest.raises(PipelineSpecLoadError, match="``steps``"):
            load_pipeline_spec_from_dict(data)

    def test_step_without_module(self) -> None:
        data = {"name": "p", "steps": [{"name": "x"}]}
        with pytest.raises(PipelineSpecLoadError, match="``module``"):
            load_pipeline_spec_from_dict(data)

    def test_step_args_not_dict(self) -> None:
        data = {
            "name": "p",
            "steps": [
                {
                    "name": "x",
                    "module": "tests.cli.test_sprint70_pipeline_cli._CLIMockOCR",
                    "args": "not_a_dict",
                },
            ],
        }
        with pytest.raises(PipelineSpecLoadError, match="``args``"):
            load_pipeline_spec_from_dict(data)

    def test_class_not_basemodule(self) -> None:
        data = {
            "name": "p",
            "steps": [
                {
                    "name": "x",
                    "module": "tests.cli.test_sprint70_pipeline_cli._NotABaseModule",
                },
            ],
        }
        with pytest.raises(PipelineSpecLoadError, match="BaseModule"):
            load_pipeline_spec_from_dict(data)

    def test_invalid_constructor_args(self) -> None:
        data = {
            "name": "p",
            "steps": [
                {
                    "name": "x",
                    "module": "tests.cli.test_sprint70_pipeline_cli._CLIMockOCR",
                    "args": {"unknown_arg": 42},
                },
            ],
        }
        with pytest.raises(PipelineSpecLoadError, match="instancier"):
            load_pipeline_spec_from_dict(data)

    def test_inputs_from_valid(self) -> None:
        data = {
            "name": "p",
            "steps": [
                {
                    "name": "ocr",
                    "module": "tests.cli.test_sprint70_pipeline_cli._CLIMockOCR",
                },
                {
                    "name": "second",
                    "module": "tests.cli.test_sprint70_pipeline_cli._CLIMockOCR",
                    "inputs_from": {"image": "__initial__"},
                },
            ],
        }
        spec = load_pipeline_spec_from_dict(data)
        assert spec.steps[1].inputs_from == {ArtifactType.IMAGE: "__initial__"}

    def test_inputs_from_unknown_type(self) -> None:
        data = {
            "name": "p",
            "steps": [
                {
                    "name": "x",
                    "module": "tests.cli.test_sprint70_pipeline_cli._CLIMockOCR",
                    "inputs_from": {"unknown_type": "ocr"},
                },
            ],
        }
        with pytest.raises(PipelineSpecLoadError, match="type d'artefact"):
            load_pipeline_spec_from_dict(data)


# ──────────────────────────────────────────────────────────────────────────
# 3. load_pipeline_spec_from_yaml
# ──────────────────────────────────────────────────────────────────────────


class TestLoadFromYaml:
    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(PipelineSpecLoadError, match="introuvable"):
            load_pipeline_spec_from_yaml(tmp_path / "nope.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "broken.yaml"
        p.write_text("name: ok\nsteps: [unclosed", encoding="utf-8")
        with pytest.raises(PipelineSpecLoadError, match="YAML invalide"):
            load_pipeline_spec_from_yaml(p)

    def test_valid_yaml_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "spec.yaml"
        p.write_text(
            "name: ocr\n"
            "steps:\n"
            "  - name: ocr\n"
            "    module: tests.cli.test_sprint70_pipeline_cli._CLIMockOCR\n",
            encoding="utf-8",
        )
        spec = load_pipeline_spec_from_yaml(p)
        assert spec.name == "ocr"
        assert len(spec.steps) == 1


# ──────────────────────────────────────────────────────────────────────────
# 4. load_comparison_specs
# ──────────────────────────────────────────────────────────────────────────


class TestLoadComparison:
    def test_valid(self) -> None:
        data = {
            "pipelines": [
                {
                    "name": "a",
                    "steps": [{
                        "name": "ocr",
                        "module": "tests.cli.test_sprint70_pipeline_cli._CLIMockOCR",
                    }],
                },
                {
                    "name": "b",
                    "steps": [{
                        "name": "ocr",
                        "module": "tests.cli.test_sprint70_pipeline_cli._CLIMockOCR",
                    }],
                },
            ],
        }
        specs = load_comparison_specs_from_dict(data)
        assert [s.name for s in specs] == ["a", "b"]

    def test_missing_pipelines(self) -> None:
        with pytest.raises(PipelineSpecLoadError, match="``pipelines``"):
            load_comparison_specs_from_dict({})


# ──────────────────────────────────────────────────────────────────────────
# 5. CLI pipeline run
# ──────────────────────────────────────────────────────────────────────────


def _make_corpus_dir(tmp_path: Path) -> Path:
    """Crée un répertoire de corpus minimal avec 1 doc."""
    img = tmp_path / "doc1.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # header PNG vide
    gt = tmp_path / "doc1.gt.txt"
    gt.write_text("hello world", encoding="utf-8")
    return tmp_path


class TestPipelineRunCLI:
    def test_run_basic(self, tmp_path: Path) -> None:
        from picarones.cli import cli

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        _make_corpus_dir(corpus_dir)

        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(
            "name: ocr_only\n"
            "steps:\n"
            "  - name: ocr\n"
            "    module: tests.cli.test_sprint70_pipeline_cli._CLIMockOCR\n",
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["pipeline", "run", str(spec_path), "--corpus", str(corpus_dir)],
        )
        assert result.exit_code == 0, result.output
        assert "ocr_only" in result.output
        assert "1/1 succès" in result.output or "1 / 1 succès" in result.output

    def test_run_with_outputs(self, tmp_path: Path) -> None:
        from picarones.cli import cli

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        _make_corpus_dir(corpus_dir)

        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(
            "name: ocr_only\n"
            "steps:\n"
            "  - name: ocr\n"
            "    module: tests.cli.test_sprint70_pipeline_cli._CLIMockOCR\n",
            encoding="utf-8",
        )
        json_out = tmp_path / "out.json"
        html_out = tmp_path / "out.html"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "pipeline", "run", str(spec_path),
                "--corpus", str(corpus_dir),
                "--output-json", str(json_out),
                "--output-html", str(html_out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert json_out.exists()
        assert html_out.exists()
        assert "<!doctype html>" in html_out.read_text(encoding="utf-8")
        import json
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        assert payload["pipeline_name"] == "ocr_only"
        assert payload["n_docs"] == 1


# ──────────────────────────────────────────────────────────────────────────
# 6. CLI pipeline compare
# ──────────────────────────────────────────────────────────────────────────


class TestPipelineCompareCLI:
    def test_compare_basic(self, tmp_path: Path) -> None:
        from picarones.cli import cli

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        _make_corpus_dir(corpus_dir)

        specs_path = tmp_path / "specs.yaml"
        specs_path.write_text(
            "pipelines:\n"
            "  - name: a\n"
            "    steps:\n"
            "      - name: ocr\n"
            "        module: tests.cli.test_sprint70_pipeline_cli._CLIMockOCR\n"
            "  - name: b\n"
            "    steps:\n"
            "      - name: ocr\n"
            "        module: tests.cli.test_sprint70_pipeline_cli._CLIMockOCR\n",
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "pipeline", "compare",
                str(specs_path),
                "--corpus", str(corpus_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Comparaison de 2 pipelines" in result.output
        assert "Classement par CER" in result.output

    def test_compare_with_html_and_baseline(self, tmp_path: Path) -> None:
        from picarones.cli import cli

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        _make_corpus_dir(corpus_dir)

        specs_path = tmp_path / "specs.yaml"
        specs_path.write_text(
            "pipelines:\n"
            "  - name: a\n"
            "    steps:\n"
            "      - name: ocr\n"
            "        module: tests.cli.test_sprint70_pipeline_cli._CLIMockOCR\n"
            "  - name: b\n"
            "    steps:\n"
            "      - name: ocr\n"
            "        module: tests.cli.test_sprint70_pipeline_cli._CLIMockOCR\n",
            encoding="utf-8",
        )
        html_out = tmp_path / "comparison.html"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "pipeline", "compare",
                str(specs_path),
                "--corpus", str(corpus_dir),
                "--output-html", str(html_out),
                "--baseline", "a",
            ],
        )
        assert result.exit_code == 0, result.output
        assert html_out.exists()
        content = html_out.read_text(encoding="utf-8")
        assert "<!doctype html>" in content
        # Baseline marquée dans le tableau de gain
        assert "(référence)" in content


# ──────────────────────────────────────────────────────────────────────────
# 7. CLI help discoverable
# ──────────────────────────────────────────────────────────────────────────


class TestCliHelp:
    def test_pipeline_group_listed(self) -> None:
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "pipeline" in result.output

    def test_pipeline_run_help(self) -> None:
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["pipeline", "run", "--help"])
        assert "SPEC_PATH" in result.output
        assert "--corpus" in result.output
        assert "--output-json" in result.output
        assert "--output-html" in result.output

    def test_pipeline_compare_help(self) -> None:
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["pipeline", "compare", "--help"])
        assert "SPECS_PATH" in result.output
        assert "--baseline" in result.output
