"""Sprint D.1.a (plan v2.0) — helpers de mapping Corpus → CorpusSpec.

Vérifie que ``picarones.app.services._legacy_runner_adapter`` produit
des ``DocumentRef`` et ``CorpusSpec`` cohérents avec les ``Document``
et ``Corpus`` legacy.

Cette première itération du Sprint D pose la fondation des sub-phases
D.1.b-e qui construiront un adapter ``run_benchmark_via_service``
complet.  Les helpers testés ici sont **purs** (pas de réseau, pas de
LLM, juste de la transformation data) et réutilisables.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picarones.adapters.legacy_engines._step_executor import (
    LegacyOCREngineExecutor,
)
from picarones.adapters.legacy_engines.base import BaseOCREngine
from picarones.adapters.llm.base import BaseLLMAdapter
from picarones.app.services._legacy_runner_adapter import (
    build_adapter_resolver,
    corpus_to_corpus_spec,
    document_to_document_ref,
    engine_to_pipeline_spec,
    run_result_to_benchmark_result,
)
from picarones.domain.artifacts import ArtifactType
from picarones.domain.errors import PicaronesError
from picarones.domain.pipeline_spec import INITIAL_STEP_ID
from picarones.evaluation.corpus import (
    AltoGT,
    Corpus,
    Document,
    EntitiesGT,
    PageGT,
    ReadingOrderGT,
    TextGT,
)


# ──────────────────────────────────────────────────────────────────────
# Mocks réutilisés pour D.1.b
# ──────────────────────────────────────────────────────────────────────


class _MockOCR(BaseOCREngine):
    def __init__(self, name: str = "mock_ocr") -> None:
        super().__init__(config={})
        self._name = name

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    def version(self) -> str:
        return "1.0"

    def _run_ocr(self, image_path):
        return "ocr text"


class _MockLLM(BaseLLMAdapter):
    def __init__(self, model: str = "mock-1") -> None:
        super().__init__(model=model, config={})

    @property
    def name(self) -> str:
        return "mock_llm"

    @property
    def default_model(self) -> str:
        return "mock-1"

    def _call(self, prompt, image_b64=None):
        return "corrected"


# ──────────────────────────────────────────────────────────────────────
# document_to_document_ref
# ──────────────────────────────────────────────────────────────────────


class TestDocumentToDocumentRef:
    def test_minimal_document(self, tmp_path: Path) -> None:
        """Un Document avec image + ground_truth se mappe en DocumentRef
        avec image_uri + GroundTruthRef RAW_TEXT."""
        img = tmp_path / "doc1.png"
        img.write_bytes(b"\x89PNG fake")
        doc = Document(
            image_path=img,
            ground_truth="bonjour le monde",
            doc_id="doc1",
        )

        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        assert ref.id == "doc1"
        assert ref.image_uri == str(img)
        assert len(ref.ground_truths) == 1
        gt = ref.ground_truths[0]
        assert gt.type == ArtifactType.RAW_TEXT
        # Le contenu de la GT est écrit dans workspace_dir.
        assert Path(gt.uri).read_text(encoding="utf-8") == "bonjour le monde"

    def test_image_path_none_yields_image_uri_none(self, tmp_path: Path) -> None:
        # Un document sans image (corpus textuel post-OCR).
        # Note : Document() exige un image_path str ou Path — on teste
        # le cas borderline où le caller passe None explicitement via
        # un workaround.  Le mapping doit tomber sur image_uri=None.
        doc = Document(
            image_path=tmp_path / "phantom.png",  # pas créé
            ground_truth="texte sans image",
            doc_id="textonly",
        )
        # On simule "pas d'image" en utilisant un chemin qui n'existe
        # pas — DocumentRef accepte image_uri quelconque (validation
        # filesystem est côté caller).
        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        assert ref.image_uri is not None  # le path est conservé tel quel

    def test_workspace_dir_must_exist(self, tmp_path: Path) -> None:
        doc = Document(
            image_path=tmp_path / "x.png",
            ground_truth="x",
            doc_id="x",
        )
        with pytest.raises(PicaronesError, match="doit exister"):
            document_to_document_ref(
                doc, workspace_dir=tmp_path / "nope_does_not_exist",
            )

    def test_unsafe_doc_id_is_sanitized(self, tmp_path: Path) -> None:
        """Un doc_id avec espaces ou accents est nettoyé pour respecter
        le regex de DocumentRef (alphanum + ._-/)."""
        img = tmp_path / "fancy.png"
        img.write_bytes(b"x")
        doc = Document(
            image_path=img,
            ground_truth="hello",
            doc_id="folio 1.r (été)",
        )
        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        # Espaces et parenthèses → underscores ; le résultat reste
        # informatif et conforme.
        assert " " not in ref.id
        assert "(" not in ref.id
        assert "folio_1.r" in ref.id

    def test_empty_doc_id_falls_back_to_doc(self, tmp_path: Path) -> None:
        img = tmp_path / "noid.png"
        img.write_bytes(b"x")
        # Document avec doc_id vide → __post_init__ remplit avec stem.
        # On force le contournement en assignant après.
        doc = Document(image_path=img, ground_truth="g")
        doc.doc_id = ""  # type: ignore[misc]
        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        assert ref.id == "doc"

    def test_with_alto_payload(self, tmp_path: Path) -> None:
        """Un document avec un payload ALTO produit un GroundTruthRef
        ALTO_XML."""
        img = tmp_path / "altodoc.png"
        img.write_bytes(b"x")
        doc = Document(
            image_path=img,
            ground_truth="alto baseline text",
            doc_id="altodoc",
            ground_truths={
                ArtifactType.RAW_TEXT: TextGT(text="alto baseline text"),
                ArtifactType.ALTO_XML: AltoGT(
                    xml_content="<alto/>",
                ),
            },
        )
        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        types = [gt.type for gt in ref.ground_truths]
        assert ArtifactType.ALTO_XML in types
        alto = next(gt for gt in ref.ground_truths if gt.type == ArtifactType.ALTO_XML)
        # Le contenu XML est écrit dans le workspace.
        assert Path(alto.uri).read_text(encoding="utf-8") == "<alto/>"

    def test_with_entities_payload(self, tmp_path: Path) -> None:
        img = tmp_path / "entdoc.png"
        img.write_bytes(b"x")
        ents = [{"label": "PER", "start": 0, "end": 5, "text": "Marie"}]
        doc = Document(
            image_path=img,
            ground_truth="Marie",
            doc_id="entdoc",
            ground_truths={
                ArtifactType.RAW_TEXT: TextGT(text="Marie"),
                ArtifactType.ENTITIES: EntitiesGT(entities=ents),
            },
        )
        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        ent_ref = next(
            gt for gt in ref.ground_truths if gt.type == ArtifactType.ENTITIES
        )
        # JSON sérialisé avec préservation des accents.
        loaded = json.loads(Path(ent_ref.uri).read_text(encoding="utf-8"))
        assert loaded == ents

    def test_with_existing_source_path_no_copy(self, tmp_path: Path) -> None:
        """Si le payload GT a un ``source_path`` existant, l'URI pointe
        directement dessus (pas de copie redondante)."""
        img = tmp_path / "withsource.png"
        img.write_bytes(b"x")

        # Crée un fichier ALTO source réel.
        existing_alto = tmp_path / "real.alto.xml"
        existing_alto.write_text("<alto>real</alto>", encoding="utf-8")

        doc = Document(
            image_path=img,
            ground_truth="x",
            doc_id="withsource",
            ground_truths={
                ArtifactType.RAW_TEXT: TextGT(text="x"),
                ArtifactType.ALTO_XML: AltoGT(
                    xml_content="<alto>memcontent</alto>",
                    source_path=existing_alto,
                ),
            },
        )
        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        alto = next(
            gt for gt in ref.ground_truths if gt.type == ArtifactType.ALTO_XML
        )
        # L'URI pointe sur le fichier source, pas sur une copie.
        assert alto.uri == str(existing_alto)

    def test_all_five_levels(self, tmp_path: Path) -> None:
        img = tmp_path / "allgt.png"
        img.write_bytes(b"x")
        doc = Document(
            image_path=img,
            ground_truth="t",
            doc_id="allgt",
            ground_truths={
                ArtifactType.RAW_TEXT: TextGT(text="t"),
                ArtifactType.ALTO_XML: AltoGT(xml_content="<alto/>"),
                ArtifactType.PAGE_XML: PageGT(xml_content="<PAGE/>"),
                ArtifactType.ENTITIES: EntitiesGT(entities=[]),
                ArtifactType.READING_ORDER: ReadingOrderGT(
                    region_order=["r1", "r2"],
                ),
            },
        )
        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        types = {gt.type for gt in ref.ground_truths}
        assert types == {
            ArtifactType.RAW_TEXT,
            ArtifactType.ALTO_XML,
            ArtifactType.PAGE_XML,
            ArtifactType.ENTITIES,
            ArtifactType.READING_ORDER,
        }


# ──────────────────────────────────────────────────────────────────────
# corpus_to_corpus_spec
# ──────────────────────────────────────────────────────────────────────


class TestCorpusToCorpusSpec:
    def test_empty_corpus(self, tmp_path: Path) -> None:
        corpus = Corpus(name="empty", documents=[])
        spec = corpus_to_corpus_spec(corpus, workspace_dir=tmp_path)
        assert spec.name == "empty"
        assert spec.documents == ()

    def test_single_document(self, tmp_path: Path) -> None:
        img = tmp_path / "d1.png"
        img.write_bytes(b"x")
        doc = Document(
            image_path=img,
            ground_truth="hello",
            doc_id="d1",
        )
        corpus = Corpus(name="mini", documents=[doc])
        spec = corpus_to_corpus_spec(corpus, workspace_dir=tmp_path)
        assert spec.name == "mini"
        assert len(spec.documents) == 1
        assert spec.documents[0].id == "d1"

    def test_metadata_scalar_values_propagate(self, tmp_path: Path) -> None:
        img = tmp_path / "d.png"
        img.write_bytes(b"x")
        corpus = Corpus(
            name="meta",
            documents=[Document(image_path=img, ground_truth="x", doc_id="d")],
            metadata={
                "language": "fr",
                "n_pages": 42,
                "is_curated": True,
                "complex": {"nested": "ignored"},  # ignoré (pas scalaire)
            },
        )
        spec = corpus_to_corpus_spec(corpus, workspace_dir=tmp_path)
        assert spec.metadata["language"] == "fr"
        assert spec.metadata["n_pages"] == "42"
        assert spec.metadata["is_curated"] == "True"
        assert "complex" not in spec.metadata

    def test_source_path_added_to_metadata(self, tmp_path: Path) -> None:
        img = tmp_path / "d.png"
        img.write_bytes(b"x")
        corpus = Corpus(
            name="src",
            documents=[Document(image_path=img, ground_truth="x", doc_id="d")],
            source_path="/some/dir",
        )
        spec = corpus_to_corpus_spec(corpus, workspace_dir=tmp_path)
        assert spec.metadata["source_path"] == "/some/dir"

    def test_workspace_dir_must_exist(self, tmp_path: Path) -> None:
        corpus = Corpus(name="x", documents=[])
        with pytest.raises(PicaronesError, match="doit exister"):
            corpus_to_corpus_spec(
                corpus, workspace_dir=tmp_path / "nope_dont_exist",
            )

    def test_preserves_document_order(self, tmp_path: Path) -> None:
        """L'ordre des documents est préservé (utile pour la
        reproductibilité)."""
        docs = []
        for i in range(5):
            img = tmp_path / f"doc{i}.png"
            img.write_bytes(b"x")
            docs.append(
                Document(
                    image_path=img,
                    ground_truth=f"text {i}",
                    doc_id=f"doc{i}",
                ),
            )
        corpus = Corpus(name="ordered", documents=docs)
        spec = corpus_to_corpus_spec(corpus, workspace_dir=tmp_path)
        assert [d.id for d in spec.documents] == ["doc0", "doc1", "doc2", "doc3", "doc4"]

    def test_unique_doc_ids_required(self, tmp_path: Path) -> None:
        """``CorpusSpec`` impose des doc_ids uniques.  Si deux Documents
        partagent le même id, la conversion lève."""
        from picarones.domain.errors import CorpusSpecError

        img1 = tmp_path / "a.png"
        img1.write_bytes(b"x")
        img2 = tmp_path / "b.png"
        img2.write_bytes(b"x")
        docs = [
            Document(image_path=img1, ground_truth="a", doc_id="same"),
            Document(image_path=img2, ground_truth="b", doc_id="same"),
        ]
        corpus = Corpus(name="dup", documents=docs)
        with pytest.raises(CorpusSpecError, match="dupliqu"):
            corpus_to_corpus_spec(corpus, workspace_dir=tmp_path)


# ──────────────────────────────────────────────────────────────────────
# engine_to_pipeline_spec
# ──────────────────────────────────────────────────────────────────────


class TestEngineToPipelineSpec:
    def test_ocr_only_produces_single_step_spec(self) -> None:
        ocr = _MockOCR(name="my_ocr")
        spec = engine_to_pipeline_spec(ocr)
        assert len(spec.steps) == 1
        step = spec.steps[0]
        assert step.id == "ocr"
        assert step.kind == "ocr"
        assert step.adapter_name == "my_ocr"
        assert ArtifactType.IMAGE in step.input_types
        assert ArtifactType.RAW_TEXT in step.output_types
        assert step.inputs_from[ArtifactType.IMAGE] == INITIAL_STEP_ID

    def test_ocr_only_initial_inputs_is_image(self) -> None:
        ocr = _MockOCR()
        spec = engine_to_pipeline_spec(ocr)
        assert spec.initial_inputs == (ArtifactType.IMAGE,)

    def test_ocr_only_name_is_safe(self) -> None:
        """Un engine.name avec caractères spéciaux donne quand même un
        spec.name conforme."""
        ocr = _MockOCR(name="weird name (v2)")
        spec = engine_to_pipeline_spec(ocr)
        # Le nom de la spec ne doit contenir que des chars autorisés.
        for ch in spec.name:
            assert ch.isalnum() or ch in "_-"

    def test_ocr_llm_pipeline_text_only(self) -> None:
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode

        ocr = _MockOCR(name="upstream_ocr")
        llm = _MockLLM(model="mock-1")
        pipeline = OCRLLMPipeline(
            ocr_engine=ocr,
            llm_adapter=llm,
            mode=PipelineMode.TEXT_ONLY,
        )
        spec = engine_to_pipeline_spec(pipeline)
        # Spec composée : 2 steps (OCR + LLM).
        assert len(spec.steps) == 2
        assert spec.steps[0].adapter_name == "upstream_ocr"
        assert spec.steps[1].adapter_name == "mock_llm:mock-1"
        # Le step LLM hérite du prompt template via params.
        assert "prompt_template" in spec.steps[1].params

    def test_ocr_llm_pipeline_zero_shot_no_ocr_step(self) -> None:
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode

        llm = _MockLLM(model="vlm-1")
        pipeline = OCRLLMPipeline(
            llm_adapter=llm,
            mode=PipelineMode.ZERO_SHOT,
        )
        spec = engine_to_pipeline_spec(pipeline)
        # Un seul step (VLM).
        assert len(spec.steps) == 1
        assert spec.steps[0].adapter_name == "mock_llm:vlm-1"
        assert ArtifactType.RAW_TEXT in spec.steps[0].output_types


# ──────────────────────────────────────────────────────────────────────
# build_adapter_resolver
# ──────────────────────────────────────────────────────────────────────


class TestBuildAdapterResolver:
    def test_single_ocr_engine_registered(self) -> None:
        ocr = _MockOCR(name="my_ocr")
        resolver = build_adapter_resolver([ocr])
        step = resolver("my_ocr")
        assert isinstance(step, LegacyOCREngineExecutor)

    def test_unknown_name_raises_keyerror(self) -> None:
        ocr = _MockOCR()
        resolver = build_adapter_resolver([ocr])
        with pytest.raises(KeyError, match="adapter inconnu"):
            resolver("unknown_engine")

    def test_multiple_engines_registered(self) -> None:
        ocr_a = _MockOCR(name="engine_a")
        ocr_b = _MockOCR(name="engine_b")
        resolver = build_adapter_resolver([ocr_a, ocr_b])
        step_a = resolver("engine_a")
        step_b = resolver("engine_b")
        assert isinstance(step_a, LegacyOCREngineExecutor)
        assert isinstance(step_b, LegacyOCREngineExecutor)

    def test_collision_on_same_name_raises(self) -> None:
        """Deux engines avec le même name → PicaronesError (le resolver
        ne peut pas distinguer les deux instances)."""
        ocr_a = _MockOCR(name="dup")
        ocr_b = _MockOCR(name="dup")  # même name, instance différente
        with pytest.raises(PicaronesError, match="enregistré"):
            build_adapter_resolver([ocr_a, ocr_b])

    def test_pipeline_registers_subcomponents(self) -> None:
        """Pour un OCRLLMPipeline, le resolver enregistre l'OCR
        sous-jacent (wrappé) et le LLM (qui est déjà StepExecutor),
        pas le pipeline lui-même."""
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode

        ocr = _MockOCR(name="inner_ocr")
        llm = _MockLLM(model="mock-1")
        pipeline = OCRLLMPipeline(
            ocr_engine=ocr,
            llm_adapter=llm,
            mode=PipelineMode.TEXT_ONLY,
        )
        resolver = build_adapter_resolver([pipeline])
        # Les sous-composants sont disponibles…
        assert isinstance(resolver("inner_ocr"), LegacyOCREngineExecutor)
        assert resolver("mock_llm:mock-1") is llm
        # …mais pas le pipeline lui-même par son nom (le resolver
        # référence par adapter_name dans la spec, pas par engine).
        with pytest.raises(KeyError):
            resolver(pipeline.name)

    def test_zero_shot_pipeline_only_registers_llm(self) -> None:
        """En zero_shot, ocr_engine=None → seul le LLM est enregistré."""
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode

        llm = _MockLLM(model="vlm-1")
        pipeline = OCRLLMPipeline(
            llm_adapter=llm,
            mode=PipelineMode.ZERO_SHOT,
        )
        resolver = build_adapter_resolver([pipeline])
        assert resolver("mock_llm:vlm-1") is llm


# ──────────────────────────────────────────────────────────────────────
# Intégration : engine_to_pipeline_spec + build_adapter_resolver
# ──────────────────────────────────────────────────────────────────────


class TestEngineSpecResolverIntegration:
    def test_spec_adapter_names_resolve(self) -> None:
        """Tous les ``adapter_name`` de la spec produite par
        ``engine_to_pipeline_spec`` doivent être résolvables par
        ``build_adapter_resolver([engine])``."""
        ocr = _MockOCR(name="resolved_ocr")
        spec = engine_to_pipeline_spec(ocr)
        resolver = build_adapter_resolver([ocr])
        for step in spec.steps:
            executor = resolver(step.adapter_name)
            assert executor is not None

    def test_pipeline_spec_resolvers_all_steps(self) -> None:
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode

        ocr = _MockOCR(name="upstream")
        llm = _MockLLM(model="mock-1")
        pipeline = OCRLLMPipeline(
            ocr_engine=ocr,
            llm_adapter=llm,
            mode=PipelineMode.TEXT_AND_IMAGE,
        )
        spec = engine_to_pipeline_spec(pipeline)
        resolver = build_adapter_resolver([pipeline])
        # Les 2 steps (OCR + LLM) doivent pouvoir être résolus.
        for step in spec.steps:
            assert resolver(step.adapter_name) is not None


# ──────────────────────────────────────────────────────────────────────
# run_result_to_benchmark_result (D.1.c)
# ──────────────────────────────────────────────────────────────────────


def _make_pipeline_result(
    *,
    pipeline_name: str,
    document_id: str,
    text_final: str,
    workspace_dir: Path,
    artifact_type: ArtifactType = ArtifactType.RAW_TEXT,
    duration: float = 0.5,
    step_error: str | None = None,
):
    """Fabrique un PipelineResult fictif pour les tests."""
    from picarones.domain.artifacts import Artifact
    from picarones.pipeline.types import PipelineResult, StepResult

    out_path = workspace_dir / f"{document_id}_{artifact_type.value}.txt"
    out_path.write_text(text_final, encoding="utf-8")
    artifact = Artifact(
        id=f"{document_id}:step:{artifact_type.value}",
        document_id=document_id,
        type=artifact_type,
        uri=str(out_path),
        produced_by_step="step",
    )
    step = StepResult(
        step_id="step",
        succeeded=step_error is None,
        duration_seconds=duration,
        error=step_error,
        produced_artifacts={artifact_type.value: artifact.id},
    )
    return PipelineResult(
        pipeline_name=pipeline_name,
        document_id=document_id,
        step_results=(step,),
        succeeded=step_error is None,
        duration_seconds=duration,
        artifacts=(artifact,),
    )


def _make_run_result(
    *,
    document_results,
):
    """Fabrique un RunResult fictif autour de RunDocumentResult."""
    from picarones.app.results import RunResult
    from picarones.domain.run_manifest import RunManifest, utcnow

    now = utcnow()
    manifest = RunManifest(
        run_id="test-run",
        corpus_name="test",
        n_documents=len(document_results),
        code_version="test",
        started_at=now,
        completed_at=now,
    )
    return RunResult(manifest=manifest, document_results=tuple(document_results))


class TestRunResultToBenchmarkResult:
    def test_single_engine_single_doc(self, tmp_path: Path) -> None:
        """Un OCR seul, un doc — la conversion produit un BenchmarkResult
        avec 1 EngineReport et 1 DocumentResult."""
        from picarones.app.results import RunDocumentResult

        ocr = _MockOCR(name="my_ocr")
        img = tmp_path / "d1.png"
        img.write_bytes(b"x")
        doc = Document(
            image_path=img,
            ground_truth="bonjour le monde",
            doc_id="d1",
        )
        corpus = Corpus(name="test", documents=[doc])

        # Pipeline a produit RAW_TEXT="bonjour le monde" → CER=0
        pr = _make_pipeline_result(
            pipeline_name="ocr_only_my_ocr",
            document_id="d1",
            text_final="bonjour le monde",
            workspace_dir=tmp_path,
        )
        rdr = RunDocumentResult(
            document_id="d1",
            pipeline_results=(pr,),
        )
        run_result = _make_run_result(document_results=[rdr])

        bm = run_result_to_benchmark_result(
            run_result, corpus=corpus, engines=[ocr],
        )
        assert bm.corpus_name == "test"
        assert bm.document_count == 1
        assert len(bm.engine_reports) == 1
        report = bm.engine_reports[0]
        assert report.engine_name == "my_ocr"
        assert report.engine_version == "1.0"
        assert len(report.document_results) == 1
        dr = report.document_results[0]
        assert dr.doc_id == "d1"
        assert dr.ground_truth == "bonjour le monde"
        assert dr.hypothesis == "bonjour le monde"
        # Texte identique → CER 0 (parfait).
        assert dr.metrics.cer == pytest.approx(0.0)
        # Pas d'erreur engine.
        assert dr.engine_error is None
        # OCR seul → pas d'ocr_intermediate, pas de pipeline_metadata.
        assert dr.ocr_intermediate is None
        assert dr.pipeline_metadata == {}

    def test_corrected_text_takes_precedence_over_raw_text(
        self, tmp_path: Path,
    ) -> None:
        """Un pipeline OCR+LLM produit RAW_TEXT (intermediate) ET
        CORRECTED_TEXT (final).  Le converter prend CORRECTED_TEXT
        comme hypothèse et expose RAW_TEXT en ocr_intermediate."""
        from picarones.app.results import RunDocumentResult
        from picarones.domain.artifacts import Artifact
        from picarones.pipeline.types import PipelineResult, StepResult

        # Construit manuellement un PipelineResult avec 2 artifacts
        # (RAW_TEXT + CORRECTED_TEXT).
        raw_path = tmp_path / "raw.txt"
        raw_path.write_text("texte ocr brut", encoding="utf-8")
        corr_path = tmp_path / "corr.txt"
        corr_path.write_text("texte ocr corrigé", encoding="utf-8")
        raw_art = Artifact(
            id="d1:ocr:raw_text",
            document_id="d1",
            type=ArtifactType.RAW_TEXT,
            uri=str(raw_path),
            produced_by_step="ocr",
        )
        corr_art = Artifact(
            id="d1:llm:corrected_text",
            document_id="d1",
            type=ArtifactType.CORRECTED_TEXT,
            uri=str(corr_path),
            produced_by_step="llm",
        )
        ocr_step = StepResult(
            step_id="ocr",
            succeeded=True,
            duration_seconds=0.3,
            produced_artifacts={ArtifactType.RAW_TEXT.value: "d1:ocr:raw_text"},
        )
        llm_step = StepResult(
            step_id="llm",
            succeeded=True,
            duration_seconds=0.7,
            produced_artifacts={
                ArtifactType.CORRECTED_TEXT.value: "d1:llm:corrected_text",
            },
        )
        pr = PipelineResult(
            pipeline_name="composed",
            document_id="d1",
            step_results=(ocr_step, llm_step),
            succeeded=True,
            duration_seconds=1.0,
            artifacts=(raw_art, corr_art),
        )
        rdr = RunDocumentResult(document_id="d1", pipeline_results=(pr,))
        run_result = _make_run_result(document_results=[rdr])

        # Pipeline OCR+LLM côté legacy
        from picarones.pipelines.base import OCRLLMPipeline, PipelineMode

        ocr = _MockOCR(name="upstream_ocr")
        llm = _MockLLM(model="mock-1")
        pipeline = OCRLLMPipeline(
            ocr_engine=ocr,
            llm_adapter=llm,
            mode=PipelineMode.TEXT_ONLY,
        )
        img = tmp_path / "d1.png"
        img.write_bytes(b"x")
        doc = Document(
            image_path=img,
            ground_truth="texte ocr corrigé",
            doc_id="d1",
        )
        corpus = Corpus(name="test", documents=[doc])

        bm = run_result_to_benchmark_result(
            run_result, corpus=corpus, engines=[pipeline],
        )
        dr = bm.engine_reports[0].document_results[0]
        # CORRECTED_TEXT a précédence comme hypothèse.
        assert dr.hypothesis == "texte ocr corrigé"
        # RAW_TEXT est exposé en ocr_intermediate.
        assert dr.ocr_intermediate == "texte ocr brut"
        # CER 0 (texte identique à la GT).
        assert dr.metrics.cer == pytest.approx(0.0)
        # pipeline_metadata présent (engine est un pipeline).
        assert dr.pipeline_metadata.get("is_pipeline") is True
        assert dr.pipeline_metadata.get("pipeline_mode") == "text_only"
        assert dr.pipeline_metadata.get("llm_model") == "mock-1"
        assert dr.pipeline_metadata.get("llm_provider") == "mock_llm"

    def test_step_error_propagated_to_engine_error(
        self, tmp_path: Path,
    ) -> None:
        from picarones.app.results import RunDocumentResult

        ocr = _MockOCR()
        img = tmp_path / "d1.png"
        img.write_bytes(b"x")
        doc = Document(image_path=img, ground_truth="x", doc_id="d1")
        corpus = Corpus(name="t", documents=[doc])

        pr = _make_pipeline_result(
            pipeline_name="ocr_only",
            document_id="d1",
            text_final="",
            workspace_dir=tmp_path,
            step_error="OCR failed",
        )
        rdr = RunDocumentResult(document_id="d1", pipeline_results=(pr,))
        run_result = _make_run_result(document_results=[rdr])

        bm = run_result_to_benchmark_result(
            run_result, corpus=corpus, engines=[ocr],
        )
        dr = bm.engine_reports[0].document_results[0]
        assert dr.engine_error == "OCR failed"

    def test_engine_report_aggregates_metrics(self, tmp_path: Path) -> None:
        """Les métriques agrégées par engine sont calculées via
        ``aggregate_metrics``."""
        from picarones.app.results import RunDocumentResult

        ocr = _MockOCR(name="agg_ocr")
        # Deux documents, hypothèses parfaites.
        docs = []
        prs = []
        for i in range(2):
            img = tmp_path / f"d{i}.png"
            img.write_bytes(b"x")
            docs.append(Document(image_path=img, ground_truth=f"texte {i}", doc_id=f"d{i}"))
            prs.append(_make_pipeline_result(
                pipeline_name="ocr_only_agg_ocr",
                document_id=f"d{i}",
                text_final=f"texte {i}",
                workspace_dir=tmp_path,
            ))
        corpus = Corpus(name="t", documents=docs)
        rdrs = [
            RunDocumentResult(document_id=f"d{i}", pipeline_results=(prs[i],))
            for i in range(2)
        ]
        run_result = _make_run_result(document_results=rdrs)

        bm = run_result_to_benchmark_result(
            run_result, corpus=corpus, engines=[ocr],
        )
        report = bm.engine_reports[0]
        assert len(report.document_results) == 2
        assert report.aggregated_metrics  # non-vide
        # CER moyen = 0 sur deux documents parfaits — l'agrégateur
        # produit un dict ``{mean, median, min, max, stdev}`` par
        # famille de métrique.
        cer_stats = report.aggregated_metrics.get("cer")
        assert cer_stats is not None
        assert cer_stats["mean"] == pytest.approx(0.0)
        assert report.aggregated_metrics.get("document_count") == 2

    def test_mismatch_documents_raises(self, tmp_path: Path) -> None:
        from picarones.app.results import RunDocumentResult

        img = tmp_path / "d.png"
        img.write_bytes(b"x")
        corpus = Corpus(
            name="t",
            documents=[
                Document(image_path=img, ground_truth="a", doc_id="d1"),
                Document(image_path=img, ground_truth="b", doc_id="d2"),
            ],
        )
        # run_result a 1 doc, corpus en a 2 → incohérent.
        pr = _make_pipeline_result(
            pipeline_name="x", document_id="d1", text_final="x",
            workspace_dir=tmp_path,
        )
        rdr = RunDocumentResult(document_id="d1", pipeline_results=(pr,))
        run_result = _make_run_result(document_results=[rdr])

        ocr = _MockOCR()
        with pytest.raises(PicaronesError, match="Mismatch"):
            run_result_to_benchmark_result(
                run_result, corpus=corpus, engines=[ocr],
            )
