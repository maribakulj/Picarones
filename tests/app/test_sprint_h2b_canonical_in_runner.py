"""Sprint H.2.b — ``run_benchmark_via_service`` consomme des
``BaseOCRAdapter`` canoniques.

Vérifie que :

- ``engine_to_pipeline_spec`` produit une ``PipelineSpec`` valide pour
  un ``BaseOCRAdapter`` canonique.
- ``build_adapter_resolver`` enregistre directement le ``BaseOCRAdapter``
  (pas de wrapping intermédiaire).
- Bout-en-bout : un ``PrecomputedTextAdapter`` consommé par
  ``run_benchmark_via_service`` produit un ``BenchmarkResult`` valide.
"""

from __future__ import annotations

from pathlib import Path

from picarones.adapters.ocr import (
    BaseOCRAdapter,
    PrecomputedTextAdapter,
    ocr_adapter_from_name,
)
from picarones.app.services.benchmark_runner import (
    build_adapter_resolver,
    engine_to_pipeline_spec,
)
from picarones.evaluation.corpus import Corpus, Document
from tests._migration_helpers import run_via_orchestrator


# ──────────────────────────────────────────────────────────────────────
# Mock canonique minimal
# ──────────────────────────────────────────────────────────────────────


class _MockCanonicalOCR(BaseOCRAdapter):
    """Adapter canonique trivial — retourne ``"text from mock"`` pour
    n'importe quelle image, sans toucher au filesystem (le contrat
    StepExecutor exige une URI mais on l'écrit dans tmp_path)."""

    def __init__(self, name: str = "mock_canonical") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def execute(self, inputs, params, context):
        from picarones.domain.artifacts import Artifact, ArtifactType

        out_dir = Path(context.workspace_uri)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{context.document_id}_mock.txt"
        out_path.write_text("text from mock", encoding="utf-8")
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:{self._name}:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
                uri=str(out_path),
            ),
        }


def _make_corpus(tmp_path: Path, n: int = 1) -> Corpus:
    docs = []
    for i in range(n):
        img = tmp_path / f"doc{i}.png"
        img.write_bytes(b"x")
        docs.append(Document(
            image_path=img,
            ground_truth="text from mock",
            doc_id=f"doc{i}",
        ))
    return Corpus(name="canonical_test", documents=docs)


# ──────────────────────────────────────────────────────────────────────
# 1. engine_to_pipeline_spec accepte un BaseOCRAdapter
# ──────────────────────────────────────────────────────────────────────


class TestEngineToPipelineSpecCanonical:
    def test_canonical_adapter_produces_mono_step_spec(self) -> None:
        adapter = _MockCanonicalOCR(name="my_ocr")
        spec = engine_to_pipeline_spec(adapter)

        assert spec.name == "ocr_only_my_ocr"
        assert len(spec.steps) == 1
        assert spec.steps[0].adapter_name == "my_ocr"

    def test_canonical_adapter_uses_adapter_input_output_types(self) -> None:
        adapter = _MockCanonicalOCR()
        spec = engine_to_pipeline_spec(adapter)
        from picarones.domain.artifacts import ArtifactType

        # PrecomputedTextAdapter declares IMAGE → RAW_TEXT.
        assert ArtifactType.IMAGE in spec.steps[0].input_types
        assert ArtifactType.RAW_TEXT in spec.steps[0].output_types

    def test_factory_built_adapter_works(self) -> None:
        adapter = ocr_adapter_from_name(
            "precomputed", source_label="bnf",
        )
        assert isinstance(adapter, PrecomputedTextAdapter)
        spec = engine_to_pipeline_spec(adapter)
        assert spec.steps[0].adapter_name == "precomputed_bnf"


# ──────────────────────────────────────────────────────────────────────
# 2. build_adapter_resolver enregistre direct (sans wrapping)
# ──────────────────────────────────────────────────────────────────────


class TestBuildAdapterResolverCanonical:
    def test_canonical_registered_without_wrapping(self) -> None:
        adapter = _MockCanonicalOCR(name="my_ocr")
        resolver = build_adapter_resolver([adapter])

        registered = resolver("my_ocr")
        # L'instance retournée est l'adapter lui-même, pas un wrapper.
        assert registered is adapter


# ──────────────────────────────────────────────────────────────────────
# 3. run_benchmark_via_service bout-en-bout avec adapter canonique
# ──────────────────────────────────────────────────────────────────────


class TestRunBenchmarkWithCanonical:
    def test_canonical_only_run_succeeds(self, tmp_path: Path) -> None:
        corpus = _make_corpus(tmp_path)
        adapter = _MockCanonicalOCR(name="my_ocr")

        bm = run_via_orchestrator(corpus, [adapter])

        assert bm.document_count == 1
        assert len(bm.engine_reports) == 1
        report = bm.engine_reports[0]
        assert report.engine_name == "my_ocr"
        # Hypothèse correctement extraite de l'artefact RAW_TEXT.
        assert report.document_results[0].hypothesis == "text from mock"

    def test_canonical_adapter_no_pipeline_metadata(
        self, tmp_path: Path,
    ) -> None:
        """Un BaseOCRAdapter n'a pas ``is_pipeline`` → pas de
        ``pipeline_metadata`` (cohérent avec un OCR seul legacy)."""
        corpus = _make_corpus(tmp_path)
        adapter = _MockCanonicalOCR()
        bm = run_via_orchestrator(corpus, [adapter])
        assert bm.engine_reports[0].document_results[0].pipeline_metadata == {}

    def test_canonical_adapter_version_unknown(self, tmp_path: Path) -> None:
        """``BaseOCRAdapter`` n'a pas de ``version()`` — tolérance
        ``_safe_engine_version`` retourne ``"unknown"``."""
        corpus = _make_corpus(tmp_path)
        adapter = _MockCanonicalOCR()
        bm = run_via_orchestrator(corpus, [adapter])
        assert bm.engine_reports[0].engine_version == "unknown"

    def test_multiple_canonical_run(self, tmp_path: Path) -> None:
        """Plusieurs adapters canoniques dans la même liste."""
        corpus = _make_corpus(tmp_path)
        a = _MockCanonicalOCR(name="canon_a")
        b = _MockCanonicalOCR(name="canon_b")

        bm = run_via_orchestrator(corpus, [a, b])

        assert len(bm.engine_reports) == 2
        engine_names = {r.engine_name for r in bm.engine_reports}
        assert engine_names == {"canon_a", "canon_b"}

    def test_canonical_with_partial_dir(self, tmp_path: Path) -> None:
        """Le chemin resumable (D.2.b) marche aussi avec des
        adapters canoniques."""
        corpus = _make_corpus(tmp_path, n=2)
        adapter = _MockCanonicalOCR(name="resumable_canon")

        bm = run_via_orchestrator(
            corpus, [adapter], partial_dir=tmp_path / "partials",
        )

        assert bm.document_count == 2
        # Le partial a été supprimé après succès.
        from picarones.app.services.partial_store import (
            _partial_path,
        )
        partial = _partial_path(corpus.name, adapter.name, tmp_path / "partials")
        assert not partial.exists()
