"""Tests du helper ``prepare_preset_args`` (Phase B3-final).

Vérifie la conversion ``(Corpus legacy + instances d'adapters)`` →
``PresetArgs`` prête à passer à ``RunOrchestrator.execute_preset()``.
"""

from __future__ import annotations

from pathlib import Path


from picarones.adapters.ocr.base import BaseOCRAdapter
from picarones.app.services import PresetArgs, prepare_preset_args
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.evaluation.corpus import Corpus, Document


# ──────────────────────────────────────────────────────────────────────
# Mock minimal
# ──────────────────────────────────────────────────────────────────────


class _MockOCR(BaseOCRAdapter):
    def __init__(self, name: str = "mock") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def execute(self, inputs, params, context):
        out_dir = Path(context.workspace_uri)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{context.document_id}.txt"
        out_path.write_text("hello", encoding="utf-8")
        return {ArtifactType.RAW_TEXT: Artifact(
            id=f"{context.document_id}:{self._name}:raw_text",
            document_id=context.document_id,
            type=ArtifactType.RAW_TEXT,
            produced_by_step="ocr",
            uri=str(out_path),
        )}


def _make_corpus(tmp_path: Path, n: int = 1) -> Corpus:
    docs = []
    for i in range(n):
        img = tmp_path / f"doc{i}.png"
        img.write_bytes(b"x")
        docs.append(Document(
            image_path=img,
            ground_truth="hello",
            doc_id=f"doc{i}",
        ))
    return Corpus(name="helper_test", documents=docs)


# ──────────────────────────────────────────────────────────────────────
# Cas nominal — un engine seul
# ──────────────────────────────────────────────────────────────────────


class TestNominal:
    def test_returns_preset_args_with_all_fields_populated(
        self, tmp_path: Path,
    ) -> None:
        corpus = _make_corpus(tmp_path)
        engine = _MockOCR()
        workspace = tmp_path / "ws"
        workspace.mkdir()

        args = prepare_preset_args(
            corpus, [engine], workspace_dir=workspace,
        )

        assert isinstance(args, PresetArgs)
        assert args.corpus_spec.name == "helper_test"
        assert len(args.pipeline_specs) == 1
        # Resolver retourne l'adapter quand on demande son name canonique.
        assert args.adapter_resolver(engine.name) is engine
        assert args.extracted_dir == workspace
        assert args.adapter_kwargs == {}

    def test_default_views_is_text_final(self, tmp_path: Path) -> None:
        corpus = _make_corpus(tmp_path)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        args = prepare_preset_args(
            corpus, [_MockOCR()], workspace_dir=workspace,
        )
        assert args.spec.views == ("text_final",)

    def test_custom_views_propagated(self, tmp_path: Path) -> None:
        corpus = _make_corpus(tmp_path)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        args = prepare_preset_args(
            corpus, [_MockOCR()],
            workspace_dir=workspace,
            views=("text_final", "alto_documentary", "searchability"),
        )
        assert args.spec.views == (
            "text_final", "alto_documentary", "searchability",
        )


# ──────────────────────────────────────────────────────────────────────
# Multi-engines
# ──────────────────────────────────────────────────────────────────────


class TestMultipleEngines:
    def test_two_engines_produce_two_pipeline_specs(
        self, tmp_path: Path,
    ) -> None:
        corpus = _make_corpus(tmp_path)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        a = _MockOCR(name="a")
        b = _MockOCR(name="b")

        args = prepare_preset_args(corpus, [a, b], workspace_dir=workspace)

        assert len(args.pipeline_specs) == 2
        # Resolver est capable de répondre aux 2 noms.
        assert args.adapter_resolver("a") is a
        assert args.adapter_resolver("b") is b


# ──────────────────────────────────────────────────────────────────────
# Conversions hétérogènes (char_exclude frozenset, normalization objet)
# ──────────────────────────────────────────────────────────────────────


class TestConversions:
    def test_char_exclude_frozenset_converted_to_string(
        self, tmp_path: Path,
    ) -> None:
        corpus = _make_corpus(tmp_path)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        args = prepare_preset_args(
            corpus, [_MockOCR()],
            workspace_dir=workspace,
            char_exclude=frozenset({"!", ".", ","}),
        )
        # Le RunSpec attend une string ; le helper convertit.
        assert args.spec.char_exclude is not None
        assert set(args.spec.char_exclude) == {"!", ".", ","}

    def test_normalization_profile_object_converted_to_name(
        self, tmp_path: Path,
    ) -> None:
        from picarones.formats.text.normalization import get_builtin_profile

        corpus = _make_corpus(tmp_path)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        profile = get_builtin_profile("caseless")
        args = prepare_preset_args(
            corpus, [_MockOCR()],
            workspace_dir=workspace,
            normalization_profile=profile,
        )
        assert args.spec.normalization_profile == "caseless"

    def test_normalization_profile_string_passthrough(
        self, tmp_path: Path,
    ) -> None:
        corpus = _make_corpus(tmp_path)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        args = prepare_preset_args(
            corpus, [_MockOCR()],
            workspace_dir=workspace,
            normalization_profile="medieval_french",
        )
        assert args.spec.normalization_profile == "medieval_french"


# ──────────────────────────────────────────────────────────────────────
# Intégration avec execute_preset (cas bout-en-bout)
# ──────────────────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_args_can_be_consumed_by_execute_preset(
        self, tmp_path: Path,
    ) -> None:
        """Pattern complet : prepare → execute_preset → converter."""
        from picarones.app.services import (
            RunOrchestrator,
            run_result_to_benchmark_result,
        )

        corpus = _make_corpus(tmp_path, n=2)
        engine = _MockOCR()
        workspace = tmp_path / "gt"
        out_dir = tmp_path / "run"

        args = prepare_preset_args(
            corpus, [engine],
            workspace_dir=workspace, output_dir=out_dir,
        )
        orch_result = RunOrchestrator(out_dir).execute_preset(
            spec=args.spec,
            corpus_spec=args.corpus_spec,
            extracted_dir=args.extracted_dir,
            pipeline_specs=args.pipeline_specs,
            adapter_resolver=args.adapter_resolver,
            adapter_kwargs=args.adapter_kwargs,
        )
        assert orch_result.run_result.n_documents == 2

        # 3e étape optionnelle : convertir en BenchmarkResult legacy.
        bm = run_result_to_benchmark_result(
            orch_result.run_result,
            corpus=corpus, engines=[engine],
            char_exclude=None, normalization_profile=None,
            profile="standard",
        )
        assert bm.document_count == 2
        assert len(bm.engine_reports) == 1
        assert bm.engine_reports[0].engine_name == "mock"


# ──────────────────────────────────────────────────────────────────────
# Default output_dir
# ──────────────────────────────────────────────────────────────────────


class TestDefaultOutputDir:
    def test_defaults_to_workspace_parent_run(self, tmp_path: Path) -> None:
        """Sans ``output_dir``, le helper utilise
        ``workspace_dir.parent / "run"``."""
        corpus = _make_corpus(tmp_path)
        workspace = tmp_path / "gt"
        workspace.mkdir()
        args = prepare_preset_args(
            corpus, [_MockOCR()], workspace_dir=workspace,
        )
        assert args.spec.output_dir == str(tmp_path / "run")
