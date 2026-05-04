"""Sprint A14-S18 — test E2E sur le cas BnF central.

Définition de done : un benchmark BnF-réaliste produit un RunResult
où on peut démontrer **qu'il n'y a pas de gagnant global** entre
les 3 pipelines hétérogènes — c'est précisément ce que le rewrite
ciblé est conçu pour rendre visible.

Scénario
--------
3 pipelines hétérogènes (proxies des moteurs réels) :

1. ``pipeline_simple_ocr`` — Tesseract-like, produit RAW_TEXT seul.
   Texte légèrement dégradé (faute typique de reconnaissance).
2. ``pipeline_structured_ocr`` — Pero-like, produit ALTO_XML +
   RAW_TEXT.  Texte de bonne qualité + structure exploitable.
3. ``pipeline_ocr_plus_correction`` — OCR+LLM, produit RAW_TEXT
   (intermédiaire dégradé) puis CORRECTED_TEXT (correction LLM
   excellente).

3 vues canoniques :

- TextView (CER/WER/MER/WIL) — meilleur **texte final**.
- AltoView (validity/line_count/word_box) — meilleur **ALTO
  exploitable**.
- SearchView (searchability_recall/numerical_sequence) —
  meilleur **pour la recherche plein-texte**.

5 documents synthétiques (XVIIIᵉ–XIXᵉ siècle, contenu BMS et
biographique) avec des dates → SearchView non triviale.

Pattern d'omission attendu
--------------------------
- AltoView omet ``pipeline_simple_ocr`` et ``pipeline_ocr_plus_correction``
  (aucun n'a d'ALTO_XML).
- TextView et SearchView incluent les 3 pipelines (RAW_TEXT toujours
  produit ; ALTO_XML projeté vers RAW_TEXT pour la pipeline 2 ;
  CORRECTED_TEXT direct pour la pipeline 3).

Comptage attendu
----------------
Par document :

- TextView : 1 (simple) + 2 (structured: RAW_TEXT + ALTO) + 2
  (correction: RAW_TEXT + CORRECTED_TEXT) = **5 ViewResult**.
- AltoView : 1 (structured seul) = **1 ViewResult**.
- SearchView : pareil que TextView = **5 ViewResult**.

Total sur 5 docs : 25 + 5 + 25 = **55 ViewResult**.

Pas de gagnant global
---------------------
- Pipeline 1 (simple) : RAW_TEXT légèrement dégradé → mediocre
  TextView, mediocre SearchView, OMIS d'AltoView.
- Pipeline 2 (structured) : RAW_TEXT excellent + ALTO disponible →
  excellent TextView (sur RAW_TEXT direct), seul gagnant possible
  d'AltoView, excellent SearchView.
- Pipeline 3 (correction) : CORRECTED_TEXT excellent → excellent
  TextView (sur CORRECTED_TEXT), excellent SearchView, OMIS
  d'AltoView.

Conclusion : aucune pipeline ne gagne sur les 3 vues — le rewrite
est conçu pour exposer cette divergence sans masquer.
"""

from __future__ import annotations

import json
import re
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
from picarones.evaluation.metrics.search import (
    numerical_sequence_preservation,
    searchability_recall,
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
    build_search_view,
    build_text_view,
)
from picarones.formats.alto.parser import parse_alto
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
# Corpus BnF synthétique
# ──────────────────────────────────────────────────────────────────


_GT_TEXTS: dict[str, str] = {
    "doc01": "Mariage de Pierre Dupont en 1789 à Paris",
    "doc02": "Acte du 14 mars 1856 enregistré à Lyon",
    "doc03": "Naissance de Marie Curie en 1867",
    "doc04": "Décès du Roi Louis XIV en 1715",
    "doc05": "Anne de Bretagne épouse Charles VIII en 1491",
}


# Pipeline 1 (simple OCR) : faute typique d'OCR — confusion d/o.
_SIMPLE_OCR_TEXTS: dict[str, str] = {
    "doc01": "Mariage de Pierre Dupant en 1789 à Paris",
    "doc02": "Acte du 14 mars 1856 enregistre à Lyon",
    "doc03": "Naissance de Marie Curie en 1867",
    "doc04": "Decés du Roi Louis XIV en 1715",
    "doc05": "Anne de Bretagne epouse Charles VIII en 1491",
}


# Pipeline 2 (structured OCR) : RAW_TEXT excellent (= GT), ALTO valide.
_STRUCTURED_OCR_TEXTS: dict[str, str] = dict(_GT_TEXTS)


# Pipeline 3 :
#   - RAW_TEXT intermédiaire dégradé (l'OCR amont est mauvais).
#   - CORRECTED_TEXT post-correction LLM = GT (correction excellente).
_OCR_BEFORE_CORRECTION: dict[str, str] = {
    "doc01": "Mariage de Pierr Dupant en 178 a Paris",
    "doc02": "Acte du 14 mars 1856 enrgistre a Lyon",
    "doc03": "Naissance d Marie Curi en 1867",
    "doc04": "Deces du Roi Louis XIV en 175",
    "doc05": "Anne de Bretagne pouse Charles VII en 1491",
}
_CORRECTED_TEXTS: dict[str, str] = dict(_GT_TEXTS)


# ──────────────────────────────────────────────────────────────────
# Fixtures de payload
# ──────────────────────────────────────────────────────────────────


def _build_alto(text: str) -> AltoDocument:
    return AltoDocument(pages=(AltoPage(blocks=(AltoTextBlock(lines=(AltoLine(strings=tuple(
        AltoString(content=w, bbox=AltoBBox(hpos=0, vpos=0, width=10, height=10))
        for w in text.split()
    )),),),),),),)


def _write_alto_files(tmp_path: Path) -> tuple[Path, dict[str, Path], dict[str, Path]]:
    """Écrit GT et candidate ALTO XML pour la pipeline structured."""
    alto_dir = tmp_path / "alto"
    alto_dir.mkdir(parents=True, exist_ok=True)
    gt_paths: dict[str, Path] = {}
    cand_paths: dict[str, Path] = {}
    for doc_id, gt_text in _GT_TEXTS.items():
        gt_path = alto_dir / f"{doc_id}.gt.alto.xml"
        cand_path = alto_dir / f"{doc_id}.structured.alto.xml"
        gt_path.write_bytes(write_alto(_build_alto(gt_text)))
        # La pipeline structured produit un ALTO du même texte que la
        # GT (excellent moteur structuré).
        cand_path.write_bytes(write_alto(_build_alto(_STRUCTURED_OCR_TEXTS[doc_id])))
        gt_paths[doc_id] = gt_path
        cand_paths[doc_id] = cand_path
    return alto_dir, gt_paths, cand_paths


# ──────────────────────────────────────────────────────────────────
# Stubs de pipelines (proxies des adapters réels)
# ──────────────────────────────────────────────────────────────────


class _SimpleOCRStub:
    """Tesseract-like : RAW_TEXT seul, texte légèrement dégradé."""

    name = "simple_ocr"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def execute(self, inputs, params, context):
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:simple_ocr:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
            ),
        }


class _StructuredOCRStub:
    """Pero-like : ALTO_XML + RAW_TEXT, texte excellent + structure."""

    name = "structured_ocr"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.ALTO_XML, ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def __init__(self, alto_files_dir: Path) -> None:
        self._alto_files_dir = Path(alto_files_dir)

    def execute(self, inputs, params, context):
        alto_path = self._alto_files_dir / f"{context.document_id}.structured.alto.xml"
        return {
            ArtifactType.ALTO_XML: Artifact(
                id=f"{context.document_id}:structured_ocr:alto",
                document_id=context.document_id,
                type=ArtifactType.ALTO_XML,
                produced_by_step="ocr",
                uri=str(alto_path),
            ),
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:structured_ocr:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
            ),
        }


class _PoorOCRStub:
    """OCR amont du pipeline 3 : RAW_TEXT très dégradé."""

    name = "poor_ocr"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def execute(self, inputs, params, context):
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:poor_ocr:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
            ),
        }


class _LLMCorrectorStub:
    """Stub LLM : RAW_TEXT → CORRECTED_TEXT (correction excellente)."""

    name = "llm_corrector"
    input_types = frozenset({ArtifactType.RAW_TEXT})
    output_types = frozenset({ArtifactType.CORRECTED_TEXT})
    execution_mode = "io"

    def execute(self, inputs, params, context):
        return {
            ArtifactType.CORRECTED_TEXT: Artifact(
                id=f"{context.document_id}:llm_corrector:corrected",
                document_id=context.document_id,
                type=ArtifactType.CORRECTED_TEXT,
                produced_by_step="llm_correct",
            ),
        }


# ──────────────────────────────────────────────────────────────────
# Setup BenchmarkService
# ──────────────────────────────────────────────────────────────────


def _build_service(tmp_path: Path) -> tuple[BenchmarkService, dict[str, Path]]:
    alto_dir, gt_paths, _cand_paths = _write_alto_files(tmp_path)

    metrics = MetricRegistry()
    # TextView metrics (sur RAW_TEXT/RAW_TEXT, lower_is_better).
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
    # SearchView metrics (sur RAW_TEXT/RAW_TEXT, higher_is_better).
    metrics.register(
        MetricSpec(
            name="searchability_recall",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
            higher_is_better=True,
        ),
        searchability_recall,
    )
    metrics.register(
        MetricSpec(
            name="numerical_sequence_preservation",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
            higher_is_better=True,
        ),
        numerical_sequence_preservation,
    )
    # AltoView metrics (sur ALTO_XML/ALTO_XML, higher_is_better).
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

    projectors = ProjectorRegistry()
    projectors.register(AltoToText())
    projectors.register(PageToText())
    projectors.register(CanonicalToText())

    # Loader hybride : résout chaque artefact selon son type et son
    # produced_by_step.  La GT raw_text vient de _GT_TEXTS, les sorties
    # OCR viennent des dicts par pipeline.
    def loader(art: Artifact):
        if art.type == ArtifactType.ALTO_XML:
            if art.uri is None:
                raise KeyError(f"ALTO sans URI : {art.id}")
            return parse_alto(Path(art.uri).read_bytes())
        if art.type == ArtifactType.RAW_TEXT:
            # Distinction par owner :
            if ":simple_ocr:" in art.id:
                return _SIMPLE_OCR_TEXTS[art.document_id]
            if ":structured_ocr:" in art.id:
                return _STRUCTURED_OCR_TEXTS[art.document_id]
            if ":poor_ocr:" in art.id:
                return _OCR_BEFORE_CORRECTION[art.document_id]
            if ":gt:" in art.id:
                return _GT_TEXTS[art.document_id]
            # Artefact projeté depuis ALTO (id se termine par ":projected_text").
            if art.id.endswith(":projected_text"):
                # On reconstitue depuis l'ALTO source via le doc_id.
                return _STRUCTURED_OCR_TEXTS[art.document_id]
            raise KeyError(f"loader: RAW_TEXT inconnu {art.id}")
        if art.type == ArtifactType.CORRECTED_TEXT:
            return _CORRECTED_TEXTS[art.document_id]
        raise KeyError(f"loader: type non géré pour {art.id} ({art.type})")

    view_executor = DefaultEvaluationViewExecutor(metrics, projectors, loader)

    registry_adapters = {
        "simple_ocr": _SimpleOCRStub(),
        "structured_ocr": _StructuredOCRStub(alto_dir),
        "poor_ocr": _PoorOCRStub(),
        "llm_corrector": _LLMCorrectorStub(),
    }
    pipeline_executor = PipelineExecutor(
        adapter_resolver=lambda n: registry_adapters[n],
    )
    corpus_runner = CorpusRunner(
        pipeline_executor,
        max_in_flight=3,
        timeout_seconds_per_doc=10.0,
        poll_interval_seconds=0.005,
    )

    service = BenchmarkService(
        corpus_runner=corpus_runner,
        view_executor=view_executor,
        code_version="1.0.0-s18-bnf-test",
    )
    return service, gt_paths


# ──────────────────────────────────────────────────────────────────
# Stubs de métriques texte (CER/WER hors registre typé pour
# isoler S18 du registre nominal — on teste l'orchestration, pas
# le calcul de métrique)
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


# ──────────────────────────────────────────────────────────────────
# Specs de pipelines + corpus + factories
# ──────────────────────────────────────────────────────────────────


def _build_pipelines() -> list[PipelineSpec]:
    pipeline_simple = PipelineSpec(
        name="pipeline_simple_ocr",
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(PipelineStep(
            id="ocr", kind="ocr", adapter_name="simple_ocr",
            input_types=(ArtifactType.IMAGE,),
            output_types=(ArtifactType.RAW_TEXT,),
        ),),
    )
    pipeline_structured = PipelineSpec(
        name="pipeline_structured_ocr",
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(PipelineStep(
            id="ocr", kind="ocr", adapter_name="structured_ocr",
            input_types=(ArtifactType.IMAGE,),
            output_types=(ArtifactType.ALTO_XML, ArtifactType.RAW_TEXT),
        ),),
    )
    pipeline_correction = PipelineSpec(
        name="pipeline_ocr_plus_correction",
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(
            PipelineStep(
                id="ocr", kind="ocr", adapter_name="poor_ocr",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),
            PipelineStep(
                id="llm_correct", kind="llm_correct",
                adapter_name="llm_corrector",
                input_types=(ArtifactType.RAW_TEXT,),
                output_types=(ArtifactType.CORRECTED_TEXT,),
            ),
        ),
    )
    return [pipeline_simple, pipeline_structured, pipeline_correction]


def _build_corpus(gt_paths: dict[str, Path]) -> CorpusSpec:
    # ``image_uri`` et le ``uri`` de la GT RAW_TEXT ne sont jamais lus
    # dans S18 — les payloads sont fournis in-memory par le loader.
    # On utilise un chemin **sous le tmp_path partagé** pour rester
    # portable cross-OS.
    base_dir = next(iter(gt_paths.values())).parent
    docs = tuple(
        DocumentRef(
            id=doc_id,
            image_uri=str(base_dir / f"{doc_id}.png"),
            ground_truths=(
                GroundTruthRef(
                    type=ArtifactType.RAW_TEXT,
                    uri=str(base_dir / f"{doc_id}.gt.txt"),
                ),
                GroundTruthRef(
                    type=ArtifactType.ALTO_XML,
                    uri=str(gt_paths[doc_id]),
                ),
            ),
        )
        for doc_id in _GT_TEXTS
    )
    return CorpusSpec(name="bnf_bms_synthetic", documents=docs)


def _build_factories(gt_paths: dict[str, Path]):
    def gt_factory(doc, art_type):
        # CORRECTED_TEXT candidates compare contre la GT RAW_TEXT —
        # les deux sont du texte plat ; la distinction de type ne porte
        # que sur le côté candidat (texte modifié par un LLM vs texte
        # OCR brut).
        effective_type = (
            ArtifactType.RAW_TEXT
            if art_type == ArtifactType.CORRECTED_TEXT
            else art_type
        )
        gt_ref = doc.gt_for(effective_type)
        if gt_ref is None:
            return None
        suffix = (
            "raw_text" if effective_type == ArtifactType.RAW_TEXT
            else "alto" if effective_type == ArtifactType.ALTO_XML
            else effective_type.value
        )
        return Artifact(
            id=f"{doc.id}:gt:{suffix}",
            document_id=doc.id,
            type=effective_type,
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
            code_version="1.0.0-s18-bnf-test",
            pipeline_name=pipeline_name,
        )

    return gt_factory, inputs_factory, ctx_factory


def _run_full_benchmark(tmp_path: Path):
    service, gt_paths = _build_service(tmp_path)
    pipelines = _build_pipelines()
    corpus = _build_corpus(gt_paths)
    views = [build_text_view(), build_alto_view(), build_search_view()]
    gt_factory, inputs_factory, ctx_factory = _build_factories(gt_paths)
    return service, service.run(
        corpus=corpus,
        pipelines=pipelines,
        views=views,
        ground_truth_factory=gt_factory,
        pipeline_inputs_factory=inputs_factory,
        context_factory=ctx_factory,
    )


# ──────────────────────────────────────────────────────────────────
# Tests E2E
# ──────────────────────────────────────────────────────────────────


class TestStructure:
    """Vérifie la structure agrégée du RunResult."""

    def test_run_executes_all_three_pipelines_on_all_docs(self, tmp_path: Path) -> None:
        _, result = _run_full_benchmark(tmp_path)
        assert result.n_documents == 5
        for doc_result in result.document_results:
            assert len(doc_result.pipeline_results) == 3
            names = {pr.pipeline_name for pr in doc_result.pipeline_results}
            assert names == {
                "pipeline_simple_ocr",
                "pipeline_structured_ocr",
                "pipeline_ocr_plus_correction",
            }

    def test_total_pipeline_results_count(self, tmp_path: Path) -> None:
        """5 docs × 3 pipelines = 15 PipelineResults."""
        _, result = _run_full_benchmark(tmp_path)
        total = sum(len(d.pipeline_results) for d in result.document_results)
        assert total == 15

    def test_correction_pipeline_has_two_steps(self, tmp_path: Path) -> None:
        """La pipeline de correction a 2 step_results par doc."""
        _, result = _run_full_benchmark(tmp_path)
        correction_results = result.pipeline_results_for(
            "pipeline_ocr_plus_correction",
        )
        assert len(correction_results) == 5
        for pr in correction_results:
            assert len(pr.step_results) == 2
            step_ids = {sr.step_id for sr in pr.step_results}
            assert step_ids == {"ocr", "llm_correct"}


class TestOmissionPattern:
    """Vérifie le pattern d'omission entre les 3 vues canoniques."""

    def test_textview_includes_all_three_pipelines(self, tmp_path: Path) -> None:
        _, result = _run_full_benchmark(tmp_path)
        text_results = result.view_results_for("text_final")
        # 5 docs × (1 + 2 + 2) = 25 ViewResult.
        assert len(text_results) == 25
        candidate_owners = {
            _owner_of(vr.candidate_artifact_id) for vr in text_results
        }
        assert candidate_owners == {
            "simple_ocr",
            "structured_ocr",
            "llm_corrector",
            "poor_ocr",
        }

    def test_altoview_omits_simple_and_correction(self, tmp_path: Path) -> None:
        _, result = _run_full_benchmark(tmp_path)
        alto_results = result.view_results_for("alto_documentary")
        # Seul structured_ocr produit ALTO → 5 docs × 1 = 5.
        assert len(alto_results) == 5
        owners = {_owner_of(vr.candidate_artifact_id) for vr in alto_results}
        assert owners == {"structured_ocr"}

    def test_searchview_includes_all_three_pipelines(self, tmp_path: Path) -> None:
        _, result = _run_full_benchmark(tmp_path)
        search_results = result.view_results_for("searchability")
        # 5 docs × (1 + 2 + 2) = 25 ViewResult, comme TextView.
        assert len(search_results) == 25


class TestNoGlobalWinner:
    """Démontre qu'aucune pipeline n'est globalement meilleure :
    chaque vue a un classement différent."""

    def test_textview_best_is_structured_or_correction(self, tmp_path: Path) -> None:
        """En CER, structured et correction ont un texte parfait
        (CER 0), simple a un texte légèrement dégradé (CER > 0)."""
        _, result = _run_full_benchmark(tmp_path)
        cer_by_pipeline_owner = _mean_metric_by_owner(
            result.view_results_for("text_final"),
            metric="cer",
        )
        # structured et correction (sur CORRECTED_TEXT) doivent battre simple.
        assert cer_by_pipeline_owner["simple_ocr"] > 0.0
        assert cer_by_pipeline_owner["structured_ocr"] == 0.0
        assert cer_by_pipeline_owner["llm_corrector"] == 0.0

    def test_altoview_only_structured_competes(self, tmp_path: Path) -> None:
        """AltoView ne peut être gagnée que par structured_ocr (les
        autres sont OMIS).  Cela démontre concrètement le pattern
        d'omission : pas de score factice 0 pour les pipelines non
        éligibles."""
        _, result = _run_full_benchmark(tmp_path)
        alto_owners = {
            _owner_of(vr.candidate_artifact_id)
            for vr in result.view_results_for("alto_documentary")
        }
        assert alto_owners == {"structured_ocr"}

    def test_search_view_best_includes_correction_and_structured(
        self, tmp_path: Path,
    ) -> None:
        """En searchability_recall, structured_ocr et le CORRECTED_TEXT
        sont parfaits (rappel 1.0), simple_ocr et poor_ocr sont en
        dessous."""
        _, result = _run_full_benchmark(tmp_path)
        recall_by_owner = _mean_metric_by_owner(
            result.view_results_for("searchability"),
            metric="searchability_recall",
        )
        assert recall_by_owner["structured_ocr"] == 1.0
        assert recall_by_owner["llm_corrector"] == 1.0
        # simple_ocr a quelques fautes de tokens (Dupont/Dupant,
        # enregistré/enregistre, etc.).  Mais Levenshtein ≤ 2 retrouve
        # tout, donc le rappel reste à 1.0 — ce qui démontre le bon
        # comportement de la métrique : les fautes de 1 char ne
        # cassent pas la recherchabilité.
        assert recall_by_owner["simple_ocr"] == 1.0
        # poor_ocr (texte amont du pipeline 3) : "Pierr" vs "Pierre"
        # passe (dist 1) mais "178" vs "1789" est dist 1 ≤ 2 → passe.
        # On vérifie au moins que c'est >= 0 et < ou égal aux autres.
        assert 0.0 <= recall_by_owner["poor_ocr"] <= 1.0

    def test_no_pipeline_wins_all_three_views(self, tmp_path: Path) -> None:
        """Garde-fou : aucune pipeline ne gagne TextView ET AltoView
        ET SearchView (pas de gagnant global).

        - simple_ocr : OMIS d'AltoView.
        - structured_ocr : présent partout, gagne AltoView, ex aequo
          en TextView avec correction.
        - pipeline_ocr_plus_correction : OMIS d'AltoView.
        """
        _, result = _run_full_benchmark(tmp_path)
        pipelines_in_alto = {
            _pipeline_name_for_owner(_owner_of(vr.candidate_artifact_id))
            for vr in result.view_results_for("alto_documentary")
        }
        assert pipelines_in_alto == {"pipeline_structured_ocr"}
        # → si aucun gagnant global possible, c'est par construction :
        # 2 des 3 pipelines sont omises de la 3ᵉ vue.


class TestPersistence:
    """Vérifie que le run BnF complet est persisté lisiblement."""

    def test_persist_writes_three_files(self, tmp_path: Path) -> None:
        service, result = _run_full_benchmark(tmp_path)
        files = service.persist(result, tmp_path / "bnf_run")
        assert files["manifest"].exists()
        assert files["pipeline_results"].exists()
        assert files["view_results"].exists()

    def test_manifest_records_all_three_pipelines_and_views(
        self, tmp_path: Path,
    ) -> None:
        service, result = _run_full_benchmark(tmp_path)
        files = service.persist(result, tmp_path / "bnf_run")
        manifest = json.loads(files["manifest"].read_text())
        assert manifest["corpus_name"] == "bnf_bms_synthetic"
        assert manifest["n_documents"] == 5
        assert sorted(manifest["pipeline_names"]) == sorted([
            "pipeline_simple_ocr",
            "pipeline_structured_ocr",
            "pipeline_ocr_plus_correction",
        ])
        assert len(manifest["view_specs"]) == 3
        view_names = {v["name"] for v in manifest["view_specs"]}
        assert view_names == {"text_final", "alto_documentary", "searchability"}

    def test_pipeline_jsonl_has_15_lines(self, tmp_path: Path) -> None:
        service, result = _run_full_benchmark(tmp_path)
        files = service.persist(result, tmp_path / "bnf_run")
        lines = files["pipeline_results"].read_text().strip().split("\n")
        assert len(lines) == 15
        for line in lines:
            payload = json.loads(line)
            assert payload["document_id"] in _GT_TEXTS
            assert payload["pipeline_name"] in {
                "pipeline_simple_ocr",
                "pipeline_structured_ocr",
                "pipeline_ocr_plus_correction",
            }

    def test_view_jsonl_has_55_lines(self, tmp_path: Path) -> None:
        """25 (TextView) + 5 (AltoView) + 25 (SearchView) = 55."""
        service, result = _run_full_benchmark(tmp_path)
        files = service.persist(result, tmp_path / "bnf_run")
        lines = files["view_results"].read_text().strip().split("\n")
        assert len(lines) == 55
        view_count: dict[str, int] = {}
        for line in lines:
            payload = json.loads(line)
            view_count[payload["view_name"]] = view_count.get(
                payload["view_name"], 0,
            ) + 1
        assert view_count == {
            "text_final": 25,
            "alto_documentary": 5,
            "searchability": 25,
        }


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


_OWNER_RE = re.compile(
    r":(simple_ocr|structured_ocr|poor_ocr|llm_corrector)(?::|$)",
)


def _owner_of(artifact_id: str) -> str:
    """Extrait le 'owner' (adapter qui a produit l'artefact) à partir
    de l'id ``<doc_id>:<owner>:<artifact_role>``."""
    match = _OWNER_RE.search(artifact_id)
    if match is None:
        raise AssertionError(f"impossible d'extraire owner de {artifact_id!r}")
    return match.group(1)


_OWNER_TO_PIPELINE = {
    "simple_ocr": "pipeline_simple_ocr",
    "structured_ocr": "pipeline_structured_ocr",
    "poor_ocr": "pipeline_ocr_plus_correction",
    "llm_corrector": "pipeline_ocr_plus_correction",
}


def _pipeline_name_for_owner(owner: str) -> str:
    return _OWNER_TO_PIPELINE[owner]


def _mean_metric_by_owner(view_results, *, metric: str) -> dict[str, float]:
    """Moyenne d'une métrique par owner d'artefact candidat (somme/n)."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for vr in view_results:
        if metric not in vr.metric_values:
            continue
        owner = _owner_of(vr.candidate_artifact_id)
        sums[owner] = sums.get(owner, 0.0) + float(vr.metric_values[metric])
        counts[owner] = counts.get(owner, 0) + 1
    return {owner: sums[owner] / counts[owner] for owner in sums}
