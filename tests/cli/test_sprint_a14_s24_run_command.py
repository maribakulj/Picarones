"""Sprint A14-S24 — ``picarones-rewrite run`` (workflow YAML → bench → HTML).

Couverture :

- **RunSpec parsing** : YAML valide → ``RunSpec``, échantillons
  variés (corpus_zip / corpus_dir, multi-pipelines, vues canoniques,
  ``adapter_kwargs``).
- **RunSpec validation** : XOR ``corpus_zip`` / ``corpus_dir``,
  rejet vues non canoniques, rejet pipelines homonymes.
- **Dotted path resolver** : import + récupération de la classe ;
  refus modules absents, classes inexistantes, chemins mal formés.
- **CLI run E2E** : YAML → benchmark complet avec adapter mock
  importé via dotted path → 3 fichiers persistés + HTML généré.
- **Erreurs CLI** : spec invalide → exit 1 avec message ; classe
  introuvable → exit 1.
"""

from __future__ import annotations

import io
import json
import textwrap
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from picarones.app.cli import cli
from picarones.app.services import (
    RunSpec,
    RunSpecLoadError,
    load_run_spec_from_yaml,
    resolve_adapter_class,
)


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89"
    )


def _make_corpus_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr("doc01.png", _png_bytes())
        zf.writestr("doc01.gt.txt", "Hello world")
        zf.writestr("doc02.png", _png_bytes())
        zf.writestr("doc02.gt.txt", "Bonjour monde")
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────
# RunSpec : parsing + validation
# ──────────────────────────────────────────────────────────────────


class TestRunSpecParsing:
    def test_minimal_valid_spec(self) -> None:
        yaml_text = textwrap.dedent("""
            corpus_zip: ./corpus.zip
            pipelines:
              - name: p1
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: tests.fixtures.cli_mock_adapters.MockTextOCR
                    input_types: [image]
                    output_types: [raw_text]
            views: [text_final]
            output_dir: ./out
        """)
        spec = load_run_spec_from_yaml(yaml_text)
        assert isinstance(spec, RunSpec)
        assert spec.corpus_zip == "./corpus.zip"
        assert len(spec.pipelines) == 1
        assert spec.pipelines[0].steps[0].adapter_class.endswith(
            "MockTextOCR",
        )

    def test_corpus_dir_alternative(self) -> None:
        yaml_text = textwrap.dedent("""
            corpus_dir: ./extracted
            pipelines:
              - name: p1
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: x.y.Z
                    input_types: [image]
                    output_types: [raw_text]
            views: [text_final]
            output_dir: ./out
        """)
        spec = load_run_spec_from_yaml(yaml_text)
        assert spec.corpus_dir == "./extracted"
        assert spec.corpus_zip is None

    def test_both_corpus_zip_and_dir_rejected(self) -> None:
        yaml_text = textwrap.dedent("""
            corpus_zip: ./a.zip
            corpus_dir: ./b
            pipelines:
              - name: p1
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: x.y.Z
                    input_types: [image]
                    output_types: [raw_text]
            views: [text_final]
            output_dir: ./out
        """)
        with pytest.raises(RunSpecLoadError, match="exactement l'un"):
            load_run_spec_from_yaml(yaml_text)

    def test_neither_corpus_source_rejected(self) -> None:
        yaml_text = textwrap.dedent("""
            pipelines:
              - name: p1
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: x.y.Z
                    input_types: [image]
                    output_types: [raw_text]
            views: [text_final]
            output_dir: ./out
        """)
        with pytest.raises(RunSpecLoadError, match="exactement l'un"):
            load_run_spec_from_yaml(yaml_text)

    def test_non_canonical_view_rejected(self) -> None:
        yaml_text = textwrap.dedent("""
            corpus_zip: ./c.zip
            pipelines:
              - name: p1
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: x.y.Z
                    input_types: [image]
                    output_types: [raw_text]
            views: [my_custom_view]
            output_dir: ./out
        """)
        with pytest.raises(RunSpecLoadError, match="vue.*inconnue"):
            load_run_spec_from_yaml(yaml_text)

    def test_duplicate_pipeline_names_rejected(self) -> None:
        yaml_text = textwrap.dedent("""
            corpus_zip: ./c.zip
            pipelines:
              - name: same
                initial_inputs: [image]
                steps:
                  - {id: a, adapter_class: x.y.A, input_types: [image], output_types: [raw_text]}
              - name: same
                initial_inputs: [image]
                steps:
                  - {id: b, adapter_class: x.y.B, input_types: [image], output_types: [raw_text]}
            views: [text_final]
            output_dir: ./out
        """)
        with pytest.raises(RunSpecLoadError, match="dupliqu"):
            load_run_spec_from_yaml(yaml_text)

    def test_corrupt_yaml_rejected(self) -> None:
        with pytest.raises(RunSpecLoadError, match="mal form"):
            load_run_spec_from_yaml("not: valid: yaml: [unbalanced")

    def test_empty_yaml_rejected(self) -> None:
        with pytest.raises(RunSpecLoadError, match="vide"):
            load_run_spec_from_yaml("")

    def test_root_not_mapping_rejected(self) -> None:
        with pytest.raises(RunSpecLoadError, match="mapping"):
            load_run_spec_from_yaml("- just a list\n- of strings")

    def test_kwargs_pass_through(self) -> None:
        yaml_text = textwrap.dedent("""
            corpus_zip: ./c.zip
            pipelines:
              - name: p1
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: tests.fixtures.cli_mock_adapters.MockTextOCR
                    adapter_kwargs:
                      copy_gt: false
                    input_types: [image]
                    output_types: [raw_text]
            views: [text_final]
            output_dir: ./out
        """)
        spec = load_run_spec_from_yaml(yaml_text)
        assert spec.pipelines[0].steps[0].adapter_kwargs == {
            "copy_gt": False,
        }


# ──────────────────────────────────────────────────────────────────
# Dotted path resolver
# ──────────────────────────────────────────────────────────────────


class TestResolveAdapterClass:
    def test_resolves_existing_class(self) -> None:
        cls = resolve_adapter_class(
            "tests.fixtures.cli_mock_adapters.MockTextOCR",
        )
        assert cls.__name__ == "MockTextOCR"

    def test_colon_separator_also_works(self) -> None:
        cls = resolve_adapter_class(
            "tests.fixtures.cli_mock_adapters:MockTextOCR",
        )
        assert cls.__name__ == "MockTextOCR"

    def test_unknown_module_raises(self) -> None:
        with pytest.raises(RunSpecLoadError, match="introuvable"):
            resolve_adapter_class("tests.does_not_exist.NopeClass")

    def test_unknown_attribute_raises(self) -> None:
        with pytest.raises(RunSpecLoadError, match="absent"):
            resolve_adapter_class(
                "tests.fixtures.cli_mock_adapters.NoSuchClass",
            )

    def test_attribute_is_not_a_class(self) -> None:
        with pytest.raises(RunSpecLoadError, match="n'est pas une classe"):
            # ``__name__`` est un str — pas une classe.
            resolve_adapter_class(
                "tests.fixtures.cli_mock_adapters.__name__",
            )

    def test_malformed_path_rejected(self) -> None:
        with pytest.raises(RunSpecLoadError, match="invalide"):
            resolve_adapter_class("noseparator")
        with pytest.raises(RunSpecLoadError, match="mal form"):
            resolve_adapter_class(".StartsWithDot")


# ──────────────────────────────────────────────────────────────────
# CLI run : E2E avec adapter mock importé via dotted path
# ──────────────────────────────────────────────────────────────────


class TestCLIRunE2E:
    def test_full_workflow_zip_to_html(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # 1. Préparer un corpus.zip.
        corpus_zip = tmp_path / "corpus.zip"
        corpus_zip.write_bytes(_make_corpus_zip())

        # 2. Préparer une spec YAML.
        spec_path = tmp_path / "run.yaml"
        out_dir = tmp_path / "out"
        report_path = out_dir / "report.html"
        spec_path.write_text(textwrap.dedent(f"""
            corpus_zip: {corpus_zip}
            corpus_name: cli_e2e
            corpus_metadata:
              language: fr
            pipelines:
              - name: tess_only
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: tests.fixtures.cli_mock_adapters.MockTextOCR
                    input_types: [image]
                    output_types: [raw_text]
            views: [text_final, searchability]
            output_dir: {out_dir}
            report_html: {report_path}
            report_lang: fr
            code_version: "1.0.0-cli-e2e"
        """))

        # 3. Invoquer la CLI.
        result = runner.invoke(cli, ["run", "--spec", str(spec_path)])
        assert result.exit_code == 0, result.output
        assert "Corpus chargé" in result.output
        assert "Run persisté" in result.output
        assert "Rapport HTML" in result.output

        # 4. Vérifier les artefacts attendus.
        results_dir = out_dir / "results"
        assert (results_dir / "run_manifest.json").exists()
        assert (results_dir / "pipeline_results.jsonl").exists()
        assert (results_dir / "view_results.jsonl").exists()
        assert report_path.exists()

        # 5. Manifest content.
        manifest = json.loads(
            (results_dir / "run_manifest.json").read_text(),
        )
        assert manifest["corpus_name"] == "cli_e2e"
        assert manifest["n_documents"] == 2
        assert "tess_only" in manifest["pipeline_names"]
        assert manifest["code_version"] == "1.0.0-cli-e2e"
        assert len(manifest["view_specs"]) == 2

        # 6. Rapport HTML est cohérent.
        html = report_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "cli_e2e" in html
        assert "tess_only" in html

        # 7. ViewResults présentes.
        view_lines = [
            json.loads(line)
            for line in (results_dir / "view_results.jsonl").read_text().strip().split("\n")
            if line.strip()
        ]
        # 2 docs × 1 pipeline × 2 vues = 4 ViewResult attendus
        # (text_final et searchability acceptent tous deux RAW_TEXT).
        assert len(view_lines) == 4
        view_names = {v["view_name"] for v in view_lines}
        assert view_names == {"text_final", "searchability"}

        # 8. Métriques valides : MockTextOCR copie la GT → CER 0.
        for vr in view_lines:
            if vr["view_name"] == "text_final":
                assert vr["metric_values"]["cer"] == 0.0

    def test_no_report_flag_skips_html(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip())
        spec_path = tmp_path / "run.yaml"
        out_dir = tmp_path / "out"
        report_path = out_dir / "report.html"
        spec_path.write_text(textwrap.dedent(f"""
            corpus_zip: {corpus_zip}
            pipelines:
              - name: p
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: tests.fixtures.cli_mock_adapters.MockTextOCR
                    input_types: [image]
                    output_types: [raw_text]
            views: [text_final]
            output_dir: {out_dir}
            report_html: {report_path}
        """))
        result = runner.invoke(cli, [
            "run", "--spec", str(spec_path), "--no-report",
        ])
        assert result.exit_code == 0
        assert not report_path.exists()
        assert "Rapport HTML" not in result.output

    def test_corpus_dir_alternative_works(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # Pré-extraire le corpus dans un dir.
        corpus_dir = tmp_path / "extracted"
        corpus_dir.mkdir()
        (corpus_dir / "doc01.png").write_bytes(_png_bytes())
        (corpus_dir / "doc01.gt.txt").write_text("text")
        spec_path = tmp_path / "run.yaml"
        out_dir = tmp_path / "out"
        spec_path.write_text(textwrap.dedent(f"""
            corpus_dir: {corpus_dir}
            corpus_name: dir_corpus
            pipelines:
              - name: p
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: tests.fixtures.cli_mock_adapters.MockTextOCR
                    input_types: [image]
                    output_types: [raw_text]
            views: [text_final]
            output_dir: {out_dir}
        """))
        result = runner.invoke(cli, ["run", "--spec", str(spec_path)])
        assert result.exit_code == 0, result.output
        assert "dir_corpus" in result.output


# ──────────────────────────────────────────────────────────────────
# CLI run : erreurs gérées
# ──────────────────────────────────────────────────────────────────


class TestCLIRunErrors:
    def test_invalid_yaml_returns_exit_1(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        spec_path = tmp_path / "bad.yaml"
        spec_path.write_text("not: valid: yaml: [bad")
        result = runner.invoke(cli, ["run", "--spec", str(spec_path)])
        assert result.exit_code == 1
        assert "spec invalide" in result.output

    def test_missing_view_canonical_rejected(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip())
        spec_path = tmp_path / "r.yaml"
        spec_path.write_text(textwrap.dedent(f"""
            corpus_zip: {corpus_zip}
            pipelines:
              - name: p
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: tests.fixtures.cli_mock_adapters.MockTextOCR
                    input_types: [image]
                    output_types: [raw_text]
            views: [unknown_view]
            output_dir: {tmp_path / "out"}
        """))
        result = runner.invoke(cli, ["run", "--spec", str(spec_path)])
        assert result.exit_code == 1
        assert "vue" in result.output.lower()

    def test_unknown_adapter_class_returns_exit_1(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip())
        spec_path = tmp_path / "r.yaml"
        spec_path.write_text(textwrap.dedent(f"""
            corpus_zip: {corpus_zip}
            pipelines:
              - name: p
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: tests.does_not_exist.Nope
                    input_types: [image]
                    output_types: [raw_text]
            views: [text_final]
            output_dir: {tmp_path / "out"}
        """))
        result = runner.invoke(cli, ["run", "--spec", str(spec_path)])
        assert result.exit_code == 1
        assert "résolution pipeline" in result.output

    def test_missing_spec_file_exit_2(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        result = runner.invoke(cli, [
            "run", "--spec", str(tmp_path / "nonexistent.yaml"),
        ])
        assert result.exit_code == 2

    def test_required_spec_option(
        self, runner: CliRunner,
    ) -> None:
        result = runner.invoke(cli, ["run"])
        assert result.exit_code == 2
        assert "--spec" in result.output


# ──────────────────────────────────────────────────────────────────
# Smoke : groupe CLI inclut bien run
# ──────────────────────────────────────────────────────────────────


class TestS25ProjectionEnabledInCLI:
    """Validation S25 : un pipeline qui produit ALTO_XML est désormais
    correctement évalué par TextView via projection automatique
    ALTO → texte, dans le contexte CLI.

    Avant S25, ce cas retournait ``failed_metrics`` car le projecteur
    ne stockait pas son output et le loader CLI ne savait pas
    récupérer le texte projeté."""

    def test_alto_pipeline_evaluated_via_textview_projection(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # Construire un corpus avec image + GT texte (pour TextView via
        # projection ALTO→texte) et GT ALTO (pour AltoView direct).
        from picarones.formats.alto.types import (
            AltoBBox, AltoDocument, AltoLine, AltoPage, AltoString,
            AltoTextBlock,
        )
        from picarones.formats.alto.writer import write_alto

        def _alto_for(text: str) -> bytes:
            doc = AltoDocument(pages=(AltoPage(blocks=(AltoTextBlock(lines=(AltoLine(strings=tuple(
                AltoString(content=w, bbox=AltoBBox(hpos=0, vpos=0, width=10, height=10))
                for w in text.split()
            )),),),),),),)
            return write_alto(doc)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("doc01.png", _png_bytes())
            zf.writestr("doc01.gt.txt", "Hello world")
            zf.writestr("doc01.gt.alto.xml", _alto_for("Hello world"))
            zf.writestr("doc02.png", _png_bytes())
            zf.writestr("doc02.gt.txt", "Bonjour monde")
            zf.writestr("doc02.gt.alto.xml", _alto_for("Bonjour monde"))
        corpus_zip = tmp_path / "corpus.zip"
        corpus_zip.write_bytes(buf.getvalue())

        spec_path = tmp_path / "run.yaml"
        out_dir = tmp_path / "out"
        spec_path.write_text(textwrap.dedent(f"""
            corpus_zip: {corpus_zip}
            corpus_name: s25_alto_proj
            pipelines:
              - name: pero_like
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: tests.fixtures.cli_mock_adapters.MockAltoOCR
                    input_types: [image]
                    output_types: [alto_xml]
            views: [text_final, alto_documentary]
            output_dir: {out_dir}
            code_version: "1.0.0-s25"
        """))

        result = runner.invoke(cli, ["run", "--spec", str(spec_path)])
        assert result.exit_code == 0, result.output

        # Le pipeline a produit ALTO_XML, donc :
        # - text_final via projection alto_to_text → CER 0.
        # - alto_documentary direct → validity 1.
        results_dir = out_dir / "results"
        view_lines = [
            json.loads(line)
            for line in (results_dir / "view_results.jsonl").read_text().strip().split("\n")
            if line.strip()
        ]
        # 2 docs × (1 text_final via projection + 1 alto_documentary direct) = 4.
        assert len(view_lines) == 4

        # Vérifier que text_final est bien renseignée (pas omise) — la
        # projection a réussi.
        text_results = [v for v in view_lines if v["view_name"] == "text_final"]
        assert len(text_results) == 2
        for vr in text_results:
            # Métriques cer/wer présentes et = 0 (ALTO contient la GT).
            assert vr["metric_values"]["cer"] == 0.0
            # Le projection_report est présent (preuve que la projection
            # ALTO → texte a bien eu lieu).
            assert vr["projection_report"] is not None
            assert vr["projection_report"]["projector_name"] == "alto_to_text"
            # Aucune métrique en échec.
            assert vr["failed_metrics"] == {}

        # AltoView direct (sans projection).
        alto_results = [v for v in view_lines if v["view_name"] == "alto_documentary"]
        assert len(alto_results) == 2
        for vr in alto_results:
            assert vr["projection_report"] is None
            assert vr["failed_metrics"] == {}


class TestGroupIncludesRun:
    def test_help_lists_run_subcommand(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_run_help_documents_options(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--spec" in result.output
        assert "--no-report" in result.output
