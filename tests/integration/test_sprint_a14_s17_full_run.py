"""Sprint A14-S17 — run complet avec persistance JSONL.

Définition de done : un benchmark produit un dossier ``result/``
lisible humainement où on voit :

- ``run_manifest.json`` — métadonnées (run_id, corpus, pipelines,
  vues, code_version, timestamps).
- ``pipeline_results.jsonl`` — un PipelineResult par ligne avec
  document_id.
- ``view_results.jsonl`` — un ViewResult par ligne avec
  document_id.

Le test exécute :
- 2 pipelines mock (un OCR pur RAW_TEXT, un OCR+ALTO).
- 3 documents synthétiques.
- 2 vues canoniques (TextView + AltoView — SearchView est testée
  séparément en S16).
- Persistance dans tmp_path.
- Vérification des fichiers produits + structure du RunResult.

Setup disque
------------
Le ``AltoToText`` projecteur (S9) lit son XML depuis l'``Artifact.uri``
filesystem.  La fixture écrit donc des fichiers ALTO XML réels sur
disque sous ``tmp_path/alto_files/`` et les stubs OCR pointent leurs
artefacts ALTO vers ces fichiers via leur URI.  Cela reproduit
l'usage production où un moteur écrit son XML dans un workspace
sandboxé (S19).
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.app.services import BenchmarkService
from picarones.domain import (
    Artifact,
    ArtifactType,
    CorpusSpec,
    DocumentRef,
    GroundTruthRef,
    MetricSpec,
)
from picarones.evaluation.metrics.alto_structural import (
    compute_alto_validity,
    compute_line_count_ratio,
    compute_word_box_coverage,
)
from picarones.evaluation.projectors import (
    AltoToText,
    CanonicalToText,
    PageToText,
    ProjectorRegistry,
)
from picarones.evaluation.registry import MetricRegistry
from picarones.evaluation.views import (
    DefaultEvaluationViewExecutor,
    build_alto_view,
    build_text_view,
)
from picarones.formats.alto.types import (
    AltoBBox,
    AltoDocument,
    AltoLine,
    AltoPage,
    AltoString,
    AltoTextBlock,
)
from picarones.formats.alto.writer import write_alto
from picarones.pipeline import (
    CorpusRunner,
    PipelineExecutor,
    PipelineSpec,
    PipelineStep,
    RunContext,
)


# ──────────────────────────────────────────────────────────────────
# Fixtures de données
# ──────────────────────────────────────────────────────────────────


_GT_TEXTS = {
    "doc01": "Bonjour le monde",
    "doc02": "Test multi documents",
    "doc03": "Troisième fixture",
}


def _build_alto(text: str) -> AltoDocument:
    """Produit un AltoDocument 1 page / 1 bloc / 1 ligne avec bbox
    sur chaque mot."""
    return AltoDocument(pages=(AltoPage(blocks=(AltoTextBlock(lines=(AltoLine(strings=tuple(
        AltoString(content=w, bbox=AltoBBox(hpos=0, vpos=0, width=10, height=10))
        for w in text.split()
    )),),),),),),)


# ──────────────────────────────────────────────────────────────────
# Adapters / pipelines mock
# ──────────────────────────────────────────────────────────────────


class _TextOCRStub:
    """OCR mock qui produit RAW_TEXT déterministe."""

    name = "text_ocr"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def execute(self, inputs, params, context):
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:text_ocr:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
            ),
        }


class _AltoOCRStub:
    """OCR mock qui produit ALTO_XML + RAW_TEXT déterministes.

    Les fichiers ALTO sont supposés déjà présents sur disque dans
    ``alto_files_dir`` (écrits par la fixture).  L'artefact ALTO
    pointe sa ``uri`` vers ce fichier — pour reproduire la chaîne
    de production où un moteur ALTO écrit son XML dans un workspace
    et l'expose via URI.
    """

    name = "alto_ocr"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.ALTO_XML, ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def __init__(self, alto_files_dir: Path) -> None:
        self._alto_files_dir = Path(alto_files_dir)

    def execute(self, inputs, params, context):
        alto_path = self._alto_files_dir / f"{context.document_id}.cand.alto.xml"
        return {
            ArtifactType.ALTO_XML: Artifact(
                id=f"{context.document_id}:alto_ocr:alto",
                document_id=context.document_id,
                type=ArtifactType.ALTO_XML,
                produced_by_step="ocr",
                uri=str(alto_path),
            ),
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:alto_ocr:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
            ),
        }


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _stub_cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    common = sum(1 for a, b in zip(reference, hypothesis) if a == b)
    return 1.0 - (common / max(len(reference), len(hypothesis)))


def _stub_wer(reference: str, hypothesis: str) -> float:
    rw = reference.split()
    hw = hypothesis.split()
    if not rw:
        return 0.0 if not hw else 1.0
    common = sum(1 for a, b in zip(rw, hw) if a == b)
    return 1.0 - (common / len(rw))


def _write_alto_files(tmp_path: Path) -> tuple[Path, dict[str, Path], dict[str, Path]]:
    """Écrit GT et candidate ALTO XML sur disque pour chaque doc.

    Returns
    -------
    (alto_dir, gt_paths_by_doc, cand_paths_by_doc)
    """
    alto_dir = tmp_path / "alto_files"
    alto_dir.mkdir(parents=True, exist_ok=True)

    gt_paths: dict[str, Path] = {}
    cand_paths: dict[str, Path] = {}
    for doc_id, text in _GT_TEXTS.items():
        gt_doc = _build_alto(text)
        cand_doc = _build_alto(text)  # Texte parfait → ALTO identique.

        gt_path = alto_dir / f"{doc_id}.gt.alto.xml"
        cand_path = alto_dir / f"{doc_id}.cand.alto.xml"
        gt_path.write_bytes(write_alto(gt_doc))
        cand_path.write_bytes(write_alto(cand_doc))

        gt_paths[doc_id] = gt_path
        cand_paths[doc_id] = cand_path

    return alto_dir, gt_paths, cand_paths


# ──────────────────────────────────────────────────────────────────
# Setup complet (param tmp_path)
# ──────────────────────────────────────────────────────────────────


def _build_service(tmp_path: Path) -> tuple[BenchmarkService, dict[str, Path]]:
    """Construit le BenchmarkService avec fixtures sur disque.

    Returns
    -------
    (service, gt_paths_by_doc)
    """
    alto_dir, gt_paths, _cand_paths = _write_alto_files(tmp_path)

    # Métriques (TextView + AltoView)
    metrics = MetricRegistry()
    for name, fn in (
        ("cer", _stub_cer),
        ("wer", _stub_wer),
        ("mer", _stub_cer),
        ("wil", _stub_wer),
    ):
        metrics.register(
            MetricSpec(
                name=name,
                input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
            ),
            fn,
        )
    for name, fn in (
        ("alto_validity", compute_alto_validity),
        ("alto_line_count_ratio", compute_line_count_ratio),
        ("alto_word_box_coverage", compute_word_box_coverage),
    ):
        metrics.register(
            MetricSpec(
                name=name,
                input_types=(ArtifactType.ALTO_XML, ArtifactType.ALTO_XML),
                higher_is_better=True,
            ),
            fn,
        )

    # Projecteurs
    projectors = ProjectorRegistry()
    projectors.register(AltoToText())
    projectors.register(PageToText())
    projectors.register(CanonicalToText())

    # Loader hybride :
    # - pour les RAW_TEXT directs (id se termine par ":raw_text") on
    #   retourne le texte parfait depuis _GT_TEXTS.
    # - pour les artefacts projetés (id se termine par ":projected_text")
    #   on retourne aussi le texte parfait (la projection a déjà fait
    #   son travail en lisant le XML disque).
    # - pour les ALTO_XML (GT ou candidat), on parse le fichier disque.
    from picarones.formats.alto.parser import parse_alto

    def loader(art: Artifact):
        if art.type == ArtifactType.RAW_TEXT:
            # GT ou candidat texte direct, ou résultat de projection.
            return _GT_TEXTS[art.document_id]
        if art.type == ArtifactType.ALTO_XML:
            if art.uri is None:
                raise KeyError(f"ALTO artefact {art.id} sans URI")
            return parse_alto(Path(art.uri).read_bytes())
        raise KeyError(f"loader ne sait pas charger {art.id} (type {art.type})")

    view_executor = DefaultEvaluationViewExecutor(metrics, projectors, loader)

    # Pipeline executor + corpus runner.
    registry_adapters = {
        "text_ocr": _TextOCRStub(),
        "alto_ocr": _AltoOCRStub(alto_dir),
    }
    pipeline_executor = PipelineExecutor(
        adapter_resolver=lambda n: registry_adapters[n],
    )
    corpus_runner = CorpusRunner(
        pipeline_executor,
        max_in_flight=2,
        timeout_seconds_per_doc=10.0,
        poll_interval_seconds=0.005,
    )

    service = BenchmarkService(
        corpus_runner=corpus_runner,
        view_executor=view_executor,
        code_version="1.0.0-s17-test",
    )
    return service, gt_paths


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────


def _build_corpus_and_specs(gt_paths: dict[str, Path]):
    docs = tuple(
        DocumentRef(
            id=doc_id,
            image_uri=f"/tmp/{doc_id}.png",
            ground_truths=(
                GroundTruthRef(
                    type=ArtifactType.RAW_TEXT,
                    uri=f"/tmp/{doc_id}.gt.txt",
                ),
                GroundTruthRef(
                    type=ArtifactType.ALTO_XML,
                    uri=str(gt_paths[doc_id]),
                ),
            ),
        )
        for doc_id in _GT_TEXTS
    )
    corpus = CorpusSpec(name="s17_fixture", documents=docs)

    text_pipeline = PipelineSpec(
        name="text_only_pipeline",
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(PipelineStep(
            id="ocr", kind="ocr", adapter_name="text_ocr",
            input_types=(ArtifactType.IMAGE,),
            output_types=(ArtifactType.RAW_TEXT,),
        ),),
    )
    alto_pipeline = PipelineSpec(
        name="alto_pipeline",
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(PipelineStep(
            id="ocr", kind="ocr", adapter_name="alto_ocr",
            input_types=(ArtifactType.IMAGE,),
            output_types=(ArtifactType.ALTO_XML, ArtifactType.RAW_TEXT),
        ),),
    )
    views = (build_text_view(), build_alto_view())

    return corpus, [text_pipeline, alto_pipeline], list(views)


def _build_factories(gt_paths: dict[str, Path]):
    def gt_factory(doc, art_type):
        gt_ref = doc.gt_for(art_type)
        if gt_ref is None:
            return None
        return Artifact(
            id=f"{doc.id}:gt:{'raw_text' if art_type == ArtifactType.RAW_TEXT else 'alto'}",
            document_id=doc.id,
            type=art_type,
            uri=gt_ref.uri,
        )

    def inputs_factory(doc):
        return {ArtifactType.IMAGE: Artifact(
            id=f"{doc.id}:image", document_id=doc.id,
            type=ArtifactType.IMAGE, uri=doc.image_uri,
        )}

    def ctx_factory(doc, pipeline_name):
        return RunContext(
            document_id=doc.id,
            code_version="1.0.0-s17-test",
            pipeline_name=pipeline_name,
        )

    return gt_factory, inputs_factory, ctx_factory


class TestFullRun:
    def test_run_produces_pipeline_results_for_each_doc(self, tmp_path: Path) -> None:
        service, gt_paths = _build_service(tmp_path)
        corpus, pipelines, views = _build_corpus_and_specs(gt_paths)
        gt_factory, inputs_factory, ctx_factory = _build_factories(gt_paths)

        result = service.run(
            corpus=corpus,
            pipelines=pipelines,
            views=views,
            ground_truth_factory=gt_factory,
            pipeline_inputs_factory=inputs_factory,
            context_factory=ctx_factory,
        )
        assert result.n_documents == 3
        for doc_result in result.document_results:
            assert len(doc_result.pipeline_results) == 2
            pipeline_names = {pr.pipeline_name for pr in doc_result.pipeline_results}
            assert pipeline_names == {"text_only_pipeline", "alto_pipeline"}

    def test_omission_pattern_textview_includes_both_pipelines(self, tmp_path: Path) -> None:
        """TextView accepte RAW_TEXT et ALTO_XML → les 2 pipelines
        sont éligibles."""
        service, gt_paths = _build_service(tmp_path)
        corpus, pipelines, views = _build_corpus_and_specs(gt_paths)
        gt_factory, inputs_factory, ctx_factory = _build_factories(gt_paths)

        result = service.run(
            corpus=corpus,
            pipelines=pipelines,
            views=views,
            ground_truth_factory=gt_factory,
            pipeline_inputs_factory=inputs_factory,
            context_factory=ctx_factory,
        )

        text_view_results = result.view_results_for("text_final")
        # text_only_pipeline produit RAW_TEXT (1 éligible).
        # alto_pipeline produit RAW_TEXT + ALTO_XML (2 éligibles).
        # Total : 3 docs × (1 + 2) = 9 ViewResult.
        assert len(text_view_results) == 9
        for vr in text_view_results:
            assert vr.view_name == "text_final"

    def test_omission_pattern_altoview_omits_text_only_pipeline(self, tmp_path: Path) -> None:
        """AltoView n'accepte qu'ALTO_XML → text_only_pipeline OMIS."""
        service, gt_paths = _build_service(tmp_path)
        corpus, pipelines, views = _build_corpus_and_specs(gt_paths)
        gt_factory, inputs_factory, ctx_factory = _build_factories(gt_paths)

        result = service.run(
            corpus=corpus,
            pipelines=pipelines,
            views=views,
            ground_truth_factory=gt_factory,
            pipeline_inputs_factory=inputs_factory,
            context_factory=ctx_factory,
        )

        alto_view_results = result.view_results_for("alto_documentary")
        # 3 docs × 1 pipeline (alto_pipeline) × 1 artefact ALTO = 3 results.
        assert len(alto_view_results) == 3
        for vr in alto_view_results:
            assert "alto_ocr" in vr.candidate_artifact_id

    def test_view_results_have_metric_values(self, tmp_path: Path) -> None:
        service, gt_paths = _build_service(tmp_path)
        corpus, pipelines, views = _build_corpus_and_specs(gt_paths)
        gt_factory, inputs_factory, ctx_factory = _build_factories(gt_paths)

        result = service.run(
            corpus=corpus,
            pipelines=pipelines,
            views=views,
            ground_truth_factory=gt_factory,
            pipeline_inputs_factory=inputs_factory,
            context_factory=ctx_factory,
        )
        for vr in result.view_results_for("text_final"):
            # CER stub doit être 0 (texte parfait dans la fixture).
            assert vr.metric_values.get("cer") == 0.0
            assert vr.failed_metrics == {}


class TestPersistence:
    def test_persist_writes_three_files(self, tmp_path: Path) -> None:
        service, gt_paths = _build_service(tmp_path)
        corpus, pipelines, views = _build_corpus_and_specs(gt_paths)
        gt_factory, inputs_factory, ctx_factory = _build_factories(gt_paths)

        result = service.run(
            corpus=corpus,
            pipelines=pipelines,
            views=views,
            ground_truth_factory=gt_factory,
            pipeline_inputs_factory=inputs_factory,
            context_factory=ctx_factory,
        )
        out_dir = tmp_path / "run_output"
        files = service.persist(result, out_dir)
        assert files["manifest"].exists()
        assert files["pipeline_results"].exists()
        assert files["view_results"].exists()

    def test_persisted_manifest_is_valid_json(self, tmp_path: Path) -> None:
        service, gt_paths = _build_service(tmp_path)
        corpus, pipelines, views = _build_corpus_and_specs(gt_paths)
        gt_factory, inputs_factory, ctx_factory = _build_factories(gt_paths)

        result = service.run(
            corpus=corpus,
            pipelines=pipelines,
            views=views,
            ground_truth_factory=gt_factory,
            pipeline_inputs_factory=inputs_factory,
            context_factory=ctx_factory,
        )
        out_dir = tmp_path / "run_output"
        files = service.persist(result, out_dir)
        manifest_data = json.loads(files["manifest"].read_text())
        assert manifest_data["corpus_name"] == "s17_fixture"
        assert manifest_data["n_documents"] == 3
        assert manifest_data["code_version"] == "1.0.0-s17-test"
        assert "text_only_pipeline" in manifest_data["pipeline_names"]
        assert "alto_pipeline" in manifest_data["pipeline_names"]

    def test_persisted_jsonl_is_streamable(self, tmp_path: Path) -> None:
        """Chaque ligne de pipeline_results.jsonl et view_results.jsonl
        est un JSON valide indépendamment (streaming)."""
        service, gt_paths = _build_service(tmp_path)
        corpus, pipelines, views = _build_corpus_and_specs(gt_paths)
        gt_factory, inputs_factory, ctx_factory = _build_factories(gt_paths)

        result = service.run(
            corpus=corpus,
            pipelines=pipelines,
            views=views,
            ground_truth_factory=gt_factory,
            pipeline_inputs_factory=inputs_factory,
            context_factory=ctx_factory,
        )
        files = service.persist(result, tmp_path / "out")

        # pipeline_results.jsonl : 3 docs × 2 pipelines = 6 lignes.
        pipeline_lines = files["pipeline_results"].read_text().strip().split("\n")
        assert len(pipeline_lines) == 6
        for line in pipeline_lines:
            payload = json.loads(line)
            assert "document_id" in payload
            assert "pipeline_name" in payload

        # view_results.jsonl : 9 (TextView) + 3 (AltoView) = 12 lignes.
        view_lines = files["view_results"].read_text().strip().split("\n")
        assert len(view_lines) == 12
        for line in view_lines:
            payload = json.loads(line)
            assert "document_id" in payload
            assert "view_name" in payload
            assert "metric_values" in payload


class TestRunResultHelpers:
    def test_pipeline_results_for_returns_correct_subset(self, tmp_path: Path) -> None:
        service, gt_paths = _build_service(tmp_path)
        corpus, pipelines, views = _build_corpus_and_specs(gt_paths)
        gt_factory, inputs_factory, ctx_factory = _build_factories(gt_paths)
        result = service.run(
            corpus=corpus,
            pipelines=pipelines,
            views=views,
            ground_truth_factory=gt_factory,
            pipeline_inputs_factory=inputs_factory,
            context_factory=ctx_factory,
        )
        # 3 docs × 1 pipeline (filtré sur "text_only_pipeline").
        text_results = result.pipeline_results_for("text_only_pipeline")
        assert len(text_results) == 3
        for pr in text_results:
            assert pr.pipeline_name == "text_only_pipeline"
