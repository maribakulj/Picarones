"""Régression : Tesseract seul → « Analyse des caractères » vide à
cause d'un ``missing_output`` fantôme.

Symptôme rapporté (mai 2026, branche ``claude/fix-tesseract-benchmark``)
=======================================================================
Pour un benchmark **Tesseract seul**, le rapport HTML affiche dans la
vue « Analyse des caractères » :

- ``Ligatures —`` / ``Diacritiques —`` (scores ``None``) ;
- « Aucune donnée de confusion disponible » ;
- « Aucune donnée taxonomique disponible » ;
- mais ``Qualité image moy. 63.6 %`` est renseignée.

Root cause
==========
``_canonical_adapter_to_spec`` construisait la
``PipelineStep.output_types`` du benchmark mono-step à partir de
``adapter.output_types`` — l'ensemble **maximal** de
``TesseractAdapter`` : ``{RAW_TEXT, CONFIDENCES, ALTO_XML}``.

Or, avec la config par défaut (``expose_alto=False`` ; ``CONFIDENCES``
best-effort), ``execute()`` ne produit que ``RAW_TEXT``.  Le
``PipelineExecutor`` marque alors le step en échec
(``error="missing_output: ['alto_xml']"``) sur **chaque** document →
``engine_error`` positionné → les 6 hooks ``requires_success=True``
(confusion, char_scores, taxonomy, structure, line_metrics,
hallucination) sont sautés → analyse caractères vide.

Seul ``image_quality`` (volontairement **sans** ``requires_success``)
survivait, d'où l'unique « Qualité image moy. » renseignée — exactement
le symptôme observé.

Le fix précédent (commit ``e3066b0``, ``success = engine_error is
None``) ne couvrait QUE le cas « sortie vide **sans** erreur ».  Ici
l'``engine_error`` est bien positionné (un ``missing_output``
*fantôme*), donc ce fix-là ne s'appliquait pas.

Fix verrouillé ici
===================
``BaseOCRAdapter.effective_output_types`` (sous-ensemble *garanti*) +
override ``TesseractAdapter`` (``{RAW_TEXT}`` ; ``ALTO_XML`` ssi
``expose_alto``) + ``_canonical_adapter_to_spec`` qui consomme
``effective_output_types``.
"""

from __future__ import annotations

from pathlib import Path

from picarones.adapters.ocr import BaseOCRAdapter
from picarones.adapters.ocr.tesseract import TesseractAdapter
from picarones.app.services._benchmark_adapter_resolver import (
    engine_to_pipeline_spec,
)
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.evaluation.corpus import Corpus, Document
from tests._migration_helpers import run_via_orchestrator


class _TesseractLikeOCR(TesseractAdapter):
    """Tesseract réel, mais ``execute`` stubé (pas de binaire
    ``tesseract`` en CI).

    Conserve volontairement le vrai ``output_types`` *maximal* et la
    vraie propriété ``effective_output_types`` de ``TesseractAdapter``
    — seul ``execute`` est remplacé pour produire UNIQUEMENT
    ``RAW_TEXT`` (= comportement par défaut : ``expose_alto=False`` +
    extraction confidences best-effort qui ne sort rien).  C'est la
    reproduction fidèle du contrat qui déclenchait le bug.
    """

    def __init__(self, hypothesis: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._hypothesis = hypothesis

    def execute(self, inputs, params, context):  # noqa: ARG002
        out_dir = Path(context.workspace_uri)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{context.document_id}_tess.txt"
        out_path.write_text(self._hypothesis, encoding="utf-8")
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:{self.name}:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
                uri=str(out_path),
            ),
        }


def _make_corpus(tmp_path: Path, gt: str, doc_id: str = "doc0") -> Corpus:
    img = tmp_path / f"{doc_id}.png"
    img.write_bytes(b"x")
    return Corpus(
        name="tesseract_missing_output_test",
        documents=[Document(
            image_path=img, ground_truth=gt, doc_id=doc_id,
        )],
    )


class TestEffectiveOutputTypesContract:
    def test_class_output_types_still_maximal(self) -> None:
        """Le contrat de *capacité* (constante de classe) est
        inchangé : la propriété ``effective_output_types`` est une
        restriction d'*instance*, pas une modification de l'attribut
        de classe (le test historique
        ``test_sprint_a14_s30`` reste vert)."""
        assert TesseractAdapter.output_types == frozenset({
            ArtifactType.RAW_TEXT,
            ArtifactType.CONFIDENCES,
            ArtifactType.ALTO_XML,
        })

    def test_default_instance_guarantees_only_raw_text(self) -> None:
        adapter = TesseractAdapter()
        assert adapter.effective_output_types == frozenset(
            {ArtifactType.RAW_TEXT},
        )

    def test_expose_alto_adds_alto_xml(self) -> None:
        """``expose_alto=True`` est un opt-in assumé (consommé par une
        AltoView) → ``ALTO_XML`` redevient un type requis."""
        adapter = TesseractAdapter(expose_alto=True)
        assert adapter.effective_output_types == frozenset({
            ArtifactType.RAW_TEXT,
            ArtifactType.ALTO_XML,
        })

    def test_expose_confidences_does_not_add_confidences(self) -> None:
        """``CONFIDENCES`` reste hors du set garanti même avec
        ``expose_confidences=True`` : sidecar best-effort sans
        consommateur côté pipeline (la calibration lit
        ``StepResult.token_confidences``, canal distinct)."""
        adapter = TesseractAdapter(expose_confidences=True)
        assert ArtifactType.CONFIDENCES not in (
            adapter.effective_output_types
        )

    def test_base_default_is_full_output_types(self) -> None:
        """Adapter simple (n'over-déclare pas) : le défaut
        ``BaseOCRAdapter.effective_output_types == output_types``
        n'altère rien."""

        class _Simple(BaseOCRAdapter):
            @property
            def name(self) -> str:
                return "simple"

            def execute(self, inputs, params, context):  # noqa: ARG002
                return {}

        assert _Simple().effective_output_types == _Simple.output_types


class TestSpecBuiltFromEffectiveTypes:
    def test_default_tesseract_spec_requires_only_raw_text(self) -> None:
        """Cœur du fix : la ``PipelineStep`` générée n'exige plus
        ``CONFIDENCES`` / ``ALTO_XML`` → plus de ``missing_output``
        fantôme."""
        spec = engine_to_pipeline_spec(TesseractAdapter())
        out = set(spec.steps[0].output_types)
        assert out == {ArtifactType.RAW_TEXT}
        assert ArtifactType.ALTO_XML not in out
        assert ArtifactType.CONFIDENCES not in out

    def test_expose_alto_spec_includes_alto(self) -> None:
        spec = engine_to_pipeline_spec(
            TesseractAdapter(expose_alto=True),
        )
        assert ArtifactType.ALTO_XML in set(
            spec.steps[0].output_types,
        )


class TestTesseractOnlyRunPopulatesCharacterAnalysis:
    """Reproduction bout-en-bout du symptôme utilisateur, vérifiée
    *résolue* (chemin production : RunOrchestrator + converter +
    hooks)."""

    def test_no_phantom_engine_error_and_analysis_present(
        self, tmp_path: Path,
    ) -> None:
        corpus = _make_corpus(tmp_path, gt="abc")
        adapter = _TesseractLikeOCR(hypothesis="abd")  # c → d
        bm = run_via_orchestrator(corpus, [adapter])

        dr = bm.engine_reports[0].document_results[0]
        # 1. Plus d'``engine_error`` fantôme (« missing_output:
        #    ['alto_xml'] »).
        assert dr.engine_error is None, (
            f"engine_error fantôme : {dr.engine_error!r} — la spec "
            "exige encore un artefact best-effort non produit"
        )
        # 2. Les 3 cibles de la vue « Analyse des caractères » sont
        #    peuplées (hooks ``requires_success`` exécutés).
        assert dr.confusion_matrix is not None
        assert dr.char_scores is not None
        assert dr.taxonomy is not None
        # 3. + compagnons du même profil.
        assert dr.structure is not None
        assert dr.line_metrics is not None
        assert dr.hallucination_metrics is not None

    def test_engine_level_aggregates_present(
        self, tmp_path: Path,
    ) -> None:
        """``EngineReport.aggregated_*`` alimentent ``DATA.engines``
        côté rapport HTML — le sélecteur « Moteur : » et la heatmap
        de confusion en dépendent."""
        corpus = _make_corpus(tmp_path, gt="café noir")
        adapter = _TesseractLikeOCR(hypothesis="cafe noir")
        bm = run_via_orchestrator(corpus, [adapter])

        report = bm.engine_reports[0]
        assert report.aggregated_confusion is not None
        assert "matrix" in report.aggregated_confusion
        assert report.aggregated_char_scores is not None
        assert report.diacritic_score is not None
        assert report.aggregated_taxonomy is not None

    def test_empty_tesseract_output_still_analyzed(
        self, tmp_path: Path,
    ) -> None:
        """Combine le fix présent (pas de ``missing_output``) avec le
        fix B3-final (``success = engine_error is None``) : un
        Tesseract qui ne reconnaît RIEN (sortie vide, sans erreur)
        produit quand même la matrice de suppressions — c'est LE cas
        diagnostic d'un outil de benchmark OCR."""
        corpus = _make_corpus(
            tmp_path, gt="Texte de référence œuvre",
        )
        adapter = _TesseractLikeOCR(hypothesis="")
        bm = run_via_orchestrator(corpus, [adapter])

        dr = bm.engine_reports[0].document_results[0]
        assert dr.hypothesis == ""
        assert dr.engine_error is None
        assert dr.metrics.cer == 1.0
        assert dr.confusion_matrix is not None
        assert dr.char_scores is not None
        assert dr.taxonomy is not None
