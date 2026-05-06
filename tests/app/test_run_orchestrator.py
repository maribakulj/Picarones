"""Tests unitaires de :class:`RunOrchestrator` (couche ``app/services/``).

Le ``RunOrchestrator`` est testé ici **directement** (sans passer par
la CLI Click).  Les tests ``tests/cli/test_sprint_a14_s24_run_command.py``
le testent indirectement via le wrapper Click — c'est complémentaire
mais pas suffisant pour vérifier le contrat du service.

Couverture
----------
- ``execute()`` retourne un :class:`OrchestrationResult` complet
  (run_result, extracted_corpus_dir, persisted_files, report_path).
- ``report_renderer=None`` ne génère aucun rapport, même si
  ``spec.report_html`` est renseigné.
- ``report_renderer=callable`` SANS ``spec.report_html`` ne génère
  rien (l'orchestrateur ne décide pas seul d'un chemin).
- ``report_renderer=callable`` ET ``spec.report_html`` → invocation
  du renderer avec le ``RunResult``, ``output_path`` et ``lang``.
- Le corpus chargé est sandboxé sous l'``output_dir`` du caller.
- Les 3 fichiers persistés sont écrits dans ``output_dir/results/``.
- Une ``CorpusImportError`` (corpus invalide) propage proprement.
- Une ``RunSpecLoadError`` (adapter dotted-path inconnu) propage
  proprement.
- Le helper ``_default_gt_factory`` traite ``CORRECTED_TEXT`` comme
  comparable à la GT ``RAW_TEXT`` (les deux sont du texte plat).
- Le helper ``_default_inputs_factory`` lève quand ``image_uri`` est
  absent.
- Le ``_filesystem_payload_loader`` lit RAW_TEXT/CORRECTED_TEXT/
  ALTO_XML, lève sur type non géré ou URI absent.
- Disambiguation ``_build_pipelines`` : 2 pipelines avec la même
  classe d'adapter mais des kwargs distincts → 2 instances
  distinctes (cas ``PrecomputedTextAdapter`` × ``source_label``).
"""

from __future__ import annotations

import io
import textwrap
import zipfile
from pathlib import Path

import pytest

from picarones.app.results import RunResult
from picarones.app.schemas import load_run_spec_from_yaml
from picarones.app.services import (
    CorpusImportError,
    OrchestrationResult,
    RunOrchestrator,
)
from picarones.app.services.run_orchestrator import (
    _default_gt_factory,
    _default_inputs_factory,
    _filesystem_payload_loader,
    _kwargs_signature,
    _make_context_factory,
)
from picarones.app.schemas.run_spec import RunSpecLoadError
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.documents import DocumentRef, GroundTruthRef


# ──────────────────────────────────────────────────────────────────
# Helpers communs
# ──────────────────────────────────────────────────────────────────


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89"
    )


def _make_corpus_zip(n_docs: int = 2) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        for i in range(1, n_docs + 1):
            doc_id = f"doc{i:02d}"
            zf.writestr(f"{doc_id}.png", _png_bytes())
            zf.writestr(f"{doc_id}.gt.txt", "Bonjour le monde")
            # Source pré-calculée pour PrecomputedTextAdapter.
            zf.writestr(f"{doc_id}.tess.txt", "Bonjour le monde")
    return buf.getvalue()


def _build_spec_yaml(
    *,
    corpus_zip: Path,
    output_dir: Path,
    report_html: str | None = None,
) -> str:
    base = textwrap.dedent(f"""
        corpus_zip: {corpus_zip}
        corpus_name: orchestrator_test
        pipelines:
          - name: tess_only
            initial_inputs: [image]
            steps:
              - id: ocr
                adapter_class: picarones.adapters.ocr.precomputed.PrecomputedTextAdapter
                adapter_kwargs:
                  source_label: tess
                input_types: [image]
                output_types: [raw_text]
        views: [text_final]
        output_dir: {output_dir}
        code_version: "1.0.0-orch-test"
    """)
    if report_html is not None:
        base += f"report_html: {report_html}\n"
    return base


# ──────────────────────────────────────────────────────────────────
# Cycle de vie ``execute()``
# ──────────────────────────────────────────────────────────────────


def _stub_renderer_called(records: list) -> "callable":
    """Crée un renderer qui enregistre ses appels et écrit un fichier
    minimal.  Utile pour vérifier l'invocation sans dépendre de
    ``HtmlReportRenderer``."""

    def _render(result: RunResult, output_path: Path, lang: str) -> Path:
        records.append({"corpus": result.manifest.corpus_name, "lang": lang})
        output_path.write_text(f"stub:{lang}", encoding="utf-8")
        return output_path

    return _render


class TestExecuteHappyPath:
    def test_returns_orchestration_result_complete(
        self, tmp_path: Path,
    ) -> None:
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip(n_docs=2))
        out_dir = tmp_path / "out"
        spec = load_run_spec_from_yaml(
            _build_spec_yaml(corpus_zip=corpus_zip, output_dir=out_dir),
        )

        orchestrator = RunOrchestrator(out_dir)
        result = orchestrator.execute(spec)

        assert isinstance(result, OrchestrationResult)
        assert isinstance(result.run_result, RunResult)
        assert result.run_result.n_documents == 2
        assert result.run_result.manifest.corpus_name == "orchestrator_test"
        # Corpus extrait sous le workspace.  ``.resolve()`` normalise
        # cross-OS (macOS résout ``/var/folders/...`` →
        # ``/private/var/folders/...``).
        assert result.extracted_corpus_dir.exists()
        assert result.extracted_corpus_dir.resolve().is_relative_to(
            out_dir.resolve(),
        )
        # S41 — 4 fichiers persistés (artifacts_index séparé).
        assert set(result.persisted_files) == {
            "manifest", "pipeline_results", "artifacts_index", "view_results",
        }
        for path in result.persisted_files.values():
            assert path.exists()
            assert path.resolve().is_relative_to(out_dir.resolve())
        # Pas de rapport car aucun renderer fourni.
        assert result.report_path is None

    def test_persisted_files_under_results_subdir(
        self, tmp_path: Path,
    ) -> None:
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip())
        out_dir = tmp_path / "out"
        spec = load_run_spec_from_yaml(
            _build_spec_yaml(corpus_zip=corpus_zip, output_dir=out_dir),
        )
        result = RunOrchestrator(out_dir).execute(spec)
        expected_parent = (out_dir / "results").resolve()
        for path in result.persisted_files.values():
            assert path.parent.resolve() == expected_parent


class TestReportRendererInjection:
    def test_no_renderer_skips_report_even_with_spec_path(
        self, tmp_path: Path,
    ) -> None:
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip())
        out_dir = tmp_path / "out"
        report_path = out_dir / "rapport.html"
        spec = load_run_spec_from_yaml(_build_spec_yaml(
            corpus_zip=corpus_zip,
            output_dir=out_dir,
            report_html=str(report_path),
        ))
        result = RunOrchestrator(out_dir).execute(spec, report_renderer=None)
        assert result.report_path is None
        assert not report_path.exists()

    def test_renderer_without_spec_path_skips(
        self, tmp_path: Path,
    ) -> None:
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip())
        out_dir = tmp_path / "out"
        spec = load_run_spec_from_yaml(_build_spec_yaml(
            corpus_zip=corpus_zip,
            output_dir=out_dir,
            report_html=None,
        ))
        records: list[dict] = []
        result = RunOrchestrator(out_dir).execute(
            spec, report_renderer=_stub_renderer_called(records),
        )
        assert result.report_path is None
        assert records == []  # renderer pas invoqué

    def test_renderer_invoked_when_both_present(
        self, tmp_path: Path,
    ) -> None:
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip())
        out_dir = tmp_path / "out"
        report_path = out_dir / "rapport.html"
        spec = load_run_spec_from_yaml(_build_spec_yaml(
            corpus_zip=corpus_zip,
            output_dir=out_dir,
            report_html=str(report_path),
        ))
        records: list[dict] = []
        result = RunOrchestrator(out_dir).execute(
            spec, report_renderer=_stub_renderer_called(records),
        )
        assert result.report_path == report_path
        assert report_path.exists()
        assert report_path.read_text(encoding="utf-8").startswith("stub:")
        assert records == [
            {"corpus": "orchestrator_test", "lang": "fr"},
        ]


# ──────────────────────────────────────────────────────────────────
# Erreurs typées propagées
# ──────────────────────────────────────────────────────────────────


class TestErrorPropagation:
    def test_corpus_dir_inexistant_raises(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        spec = load_run_spec_from_yaml(textwrap.dedent(f"""
            corpus_dir: {tmp_path / "does_not_exist"}
            pipelines:
              - name: p
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: picarones.adapters.ocr.precomputed.PrecomputedTextAdapter
                    adapter_kwargs:
                      source_label: tess
                    input_types: [image]
                    output_types: [raw_text]
            views: [text_final]
            output_dir: {out_dir}
        """))
        with pytest.raises(CorpusImportError, match="n'est pas un répertoire"):
            RunOrchestrator(out_dir).execute(spec)

    def test_unknown_adapter_class_raises(self, tmp_path: Path) -> None:
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip())
        out_dir = tmp_path / "out"
        spec = load_run_spec_from_yaml(textwrap.dedent(f"""
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
            output_dir: {out_dir}
        """))
        with pytest.raises(RunSpecLoadError, match="introuvable"):
            RunOrchestrator(out_dir).execute(spec)


# ──────────────────────────────────────────────────────────────────
# Disambiguation des adapters
# ──────────────────────────────────────────────────────────────────


class TestPipelineDisambiguation:
    def test_same_class_different_kwargs_yields_distinct_instances(
        self, tmp_path: Path,
    ) -> None:
        """Cas BnF : 2 pipelines utilisent ``PrecomputedTextAdapter``
        mais avec ``source_label`` différents → ils doivent recevoir
        des instances distinctes (sinon le 2ème lirait les fichiers
        du 1er)."""
        # Corpus avec 2 sources pré-calculées différentes.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("doc01.png", _png_bytes())
            zf.writestr("doc01.gt.txt", "Bonjour")
            zf.writestr("doc01.tess.txt", "Bonjour")  # source 1
            zf.writestr("doc01.gpt4v.txt", "Bonjur")  # source 2 (1 erreur)
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(buf.getvalue())

        out_dir = tmp_path / "out"
        spec = load_run_spec_from_yaml(textwrap.dedent(f"""
            corpus_zip: {corpus_zip}
            pipelines:
              - name: tess
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: picarones.adapters.ocr.precomputed.PrecomputedTextAdapter
                    adapter_kwargs:
                      source_label: tess
                    input_types: [image]
                    output_types: [raw_text]
              - name: gpt
                initial_inputs: [image]
                steps:
                  - id: ocr
                    adapter_class: picarones.adapters.ocr.precomputed.PrecomputedTextAdapter
                    adapter_kwargs:
                      source_label: gpt4v
                    input_types: [image]
                    output_types: [raw_text]
            views: [text_final]
            output_dir: {out_dir}
        """))
        result = RunOrchestrator(out_dir).execute(spec)
        # 1 doc × 2 pipelines = 2 ViewResult.  Ils doivent avoir des
        # candidate_artifact_id distincts (preuves d'instances distinctes).
        view_results = result.run_result.view_results_for("text_final")
        owners = {
            "tess" if "precomputed_tess" in vr.candidate_artifact_id and "tess:" in vr.candidate_artifact_id
            else "gpt" if "precomputed_gpt4v" in vr.candidate_artifact_id else "?"
            for vr in view_results
        }
        # Au moins 2 owners distincts.
        assert len(owners) >= 2


# ──────────────────────────────────────────────────────────────────
# Helpers privés (importés directement pour couverture explicite)
# ──────────────────────────────────────────────────────────────────


class TestDefaultGtFactory:
    def test_returns_artifact_for_present_gt(self) -> None:
        doc = DocumentRef(
            id="doc01",
            ground_truths=(
                GroundTruthRef(type=ArtifactType.RAW_TEXT, uri="/path/gt.txt"),
            ),
        )
        gt = _default_gt_factory(doc, ArtifactType.RAW_TEXT)
        assert gt is not None
        assert gt.type == ArtifactType.RAW_TEXT
        assert gt.uri == "/path/gt.txt"

    def test_corrected_text_falls_back_to_raw_text_gt(self) -> None:
        """Convention : un candidat CORRECTED_TEXT est comparé contre
        la GT RAW_TEXT (les deux sont du texte plat)."""
        doc = DocumentRef(
            id="doc01",
            ground_truths=(
                GroundTruthRef(type=ArtifactType.RAW_TEXT, uri="/path/gt.txt"),
            ),
        )
        gt = _default_gt_factory(doc, ArtifactType.CORRECTED_TEXT)
        assert gt is not None
        assert gt.type == ArtifactType.RAW_TEXT  # fallback explicite

    def test_returns_none_when_gt_absent(self) -> None:
        doc = DocumentRef(id="doc01", ground_truths=())
        gt = _default_gt_factory(doc, ArtifactType.RAW_TEXT)
        assert gt is None


class TestDefaultInputsFactory:
    def test_returns_image_artifact(self) -> None:
        doc = DocumentRef(id="doc01", image_uri="/path/img.png")
        inputs = _default_inputs_factory(doc)
        assert ArtifactType.IMAGE in inputs
        assert inputs[ArtifactType.IMAGE].uri == "/path/img.png"

    def test_raises_when_image_uri_absent(self) -> None:
        doc = DocumentRef(id="doc01")
        with pytest.raises(CorpusImportError, match="sans ``image_uri``"):
            _default_inputs_factory(doc)


class TestContextFactory:
    def test_factory_propagates_code_version(self) -> None:
        factory = _make_context_factory("1.2.3")
        doc = DocumentRef(id="doc01", image_uri="/x")
        ctx = factory(doc, "my_pipeline")
        assert ctx.document_id == "doc01"
        assert ctx.code_version == "1.2.3"
        assert ctx.pipeline_name == "my_pipeline"


class TestFilesystemPayloadLoader:
    def test_loads_raw_text(self, tmp_path: Path) -> None:
        path = tmp_path / "t.txt"
        path.write_text("Hello", encoding="utf-8")
        art = Artifact(
            id="d:t", document_id="d", type=ArtifactType.RAW_TEXT, uri=str(path),
        )
        assert _filesystem_payload_loader(art) == "Hello"

    def test_loads_corrected_text(self, tmp_path: Path) -> None:
        path = tmp_path / "c.txt"
        path.write_text("Bonjour", encoding="utf-8")
        art = Artifact(
            id="d:c", document_id="d", type=ArtifactType.CORRECTED_TEXT,
            uri=str(path),
        )
        assert _filesystem_payload_loader(art) == "Bonjour"

    def test_loads_alto_xml(self, tmp_path: Path) -> None:
        from picarones.formats.alto.types import (
            AltoBBox, AltoDocument, AltoLine, AltoPage, AltoString,
            AltoTextBlock,
        )
        from picarones.formats.alto.writer import write_alto

        doc = AltoDocument(pages=(AltoPage(blocks=(AltoTextBlock(lines=(AltoLine(strings=(
            AltoString(content="Hi", bbox=AltoBBox(hpos=0, vpos=0, width=10, height=10)),
        ),),),),),),))
        path = tmp_path / "a.xml"
        path.write_bytes(write_alto(doc))
        art = Artifact(
            id="d:a", document_id="d", type=ArtifactType.ALTO_XML, uri=str(path),
        )
        loaded = _filesystem_payload_loader(art)
        assert loaded.pages[0].blocks[0].lines[0].strings[0].content == "Hi"

    def test_raises_on_missing_uri(self) -> None:
        art = Artifact(
            id="d:x", document_id="d", type=ArtifactType.RAW_TEXT,
        )
        with pytest.raises(FileNotFoundError, match="sans URI"):
            _filesystem_payload_loader(art)

    def test_raises_on_unsupported_type(self, tmp_path: Path) -> None:
        path = tmp_path / "x.bin"
        path.write_bytes(b"\x00" * 4)
        art = Artifact(
            id="d:x", document_id="d", type=ArtifactType.IMAGE, uri=str(path),
        )
        with pytest.raises(ValueError, match="non géré"):
            _filesystem_payload_loader(art)


class TestKwargsSignature:
    def test_empty_dict(self) -> None:
        assert _kwargs_signature({}) == ""

    def test_single_kwarg(self) -> None:
        assert _kwargs_signature({"k": "v"}) == "k='v'"

    def test_sorted_stable(self) -> None:
        # Ordre d'insertion ne doit pas changer la signature.
        sig_a = _kwargs_signature({"b": 2, "a": 1})
        sig_b = _kwargs_signature({"a": 1, "b": 2})
        assert sig_a == sig_b

    def test_distinguishes_values(self) -> None:
        assert (
            _kwargs_signature({"k": 1})
            != _kwargs_signature({"k": 2})
        )
