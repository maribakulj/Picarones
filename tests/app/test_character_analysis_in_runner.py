"""Le converter ``run_result_to_benchmark_result`` doit alimenter la
vue "Analyse des caractères" du rapport HTML.

Le template ``picarones/reports/html/templates/view_characters.html``
+ le bloc ``renderCharView`` de ``_app.js`` consomment trois agrégats
au niveau ``EngineReport`` (``aggregated_confusion``,
``aggregated_char_scores``, ``aggregated_taxonomy``) et trois champs
par document (``confusion_matrix``, ``char_scores``, ``taxonomy``).
Avant le rewrite, la vue retombait sur ses placeholders "Aucune
donnée … disponible" parce que le runner ne calculait plus que CER /
WER.  Ce test verrouille la régression.
"""

from __future__ import annotations

from pathlib import Path

from picarones.adapters.ocr import BaseOCRAdapter
from picarones.evaluation.corpus import Corpus, Document
from tests._migration_helpers import run_via_orchestrator


class _MangleOCR(BaseOCRAdapter):
    """Adapter qui produit une hypothèse fixe, indépendante de l'image.

    Permet de fabriquer des paires (GT, OCR) déterministes pour
    valider les analyses caractères.
    """

    def __init__(self, hypothesis: str, name: str = "mangle") -> None:
        self._hypothesis = hypothesis
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def execute(self, inputs, params, context):
        from picarones.domain.artifacts import Artifact, ArtifactType

        out_dir = Path(context.workspace_uri)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{context.document_id}_mangle.txt"
        out_path.write_text(self._hypothesis, encoding="utf-8")
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:{self._name}:raw_text",
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
        name="char_analysis_test",
        documents=[Document(
            image_path=img, ground_truth=gt, doc_id=doc_id,
        )],
    )


class TestPerDocumentCharacterAnalysis:
    def test_confusion_matrix_populated_for_substitution(
        self, tmp_path: Path,
    ) -> None:
        """Un caractère substitué produit une entrée dans la matrice."""
        corpus = _make_corpus(tmp_path, gt="abc")
        adapter = _MangleOCR(hypothesis="abd")  # c → d
        bm = run_via_orchestrator(corpus, [adapter])

        dr = bm.engine_reports[0].document_results[0]
        assert dr.confusion_matrix is not None
        matrix = dr.confusion_matrix["matrix"]
        # SequenceMatcher peut classer ce remplacement comme delete +
        # insert ou comme une substitution ``c → d``.  On vérifie que
        # le caractère GT ``c`` apparaît avec une cible non-identique.
        assert "c" in matrix
        assert any(ocr_ch != "c" for ocr_ch in matrix["c"])

    def test_char_scores_ligature_perfect_match(
        self, tmp_path: Path,
    ) -> None:
        """Une ligature ﬁ reconnue ﬁ ou ``fi`` compte comme correcte."""
        corpus = _make_corpus(tmp_path, gt="ﬁn")
        adapter = _MangleOCR(hypothesis="fin")  # ﬁ → fi accepté
        bm = run_via_orchestrator(corpus, [adapter])

        dr = bm.engine_reports[0].document_results[0]
        assert dr.char_scores is not None
        lig = dr.char_scores["ligature"]
        assert lig["total_in_gt"] == 1
        assert lig["correctly_recognized"] == 1
        assert lig["score"] == 1.0

    def test_char_scores_diacritic_lost(self, tmp_path: Path) -> None:
        """Un ``é`` remplacé par ``e`` chute le score diacritique."""
        corpus = _make_corpus(tmp_path, gt="été")
        adapter = _MangleOCR(hypothesis="ete")
        bm = run_via_orchestrator(corpus, [adapter])

        dr = bm.engine_reports[0].document_results[0]
        assert dr.char_scores is not None
        diac = dr.char_scores["diacritic"]
        assert diac["total_in_gt"] == 2  # deux ``é``
        assert diac["correctly_recognized"] == 0
        assert diac["score"] == 0.0

    def test_taxonomy_records_diacritic_error(
        self, tmp_path: Path,
    ) -> None:
        """La taxonomie classe ``café → cafe`` comme diacritic_error."""
        corpus = _make_corpus(tmp_path, gt="café noir")
        adapter = _MangleOCR(hypothesis="cafe noir")
        bm = run_via_orchestrator(corpus, [adapter])

        dr = bm.engine_reports[0].document_results[0]
        assert dr.taxonomy is not None
        assert dr.taxonomy["counts"]["diacritic_error"] >= 1

    def test_standard_profile_activates_companion_hooks(
        self, tmp_path: Path,
    ) -> None:
        """Le câblage du système de hooks active toutes les analyses
        document-level enregistrées en profil ``standard`` — pas
        seulement les trois champs de la vue caractères.  Ce test
        documente le contrat : si un nouveau hook est ajouté à
        ``builtin_hooks._STANDARD_PROFILES``, il devient
        automatiquement disponible dans le runner.
        """
        corpus = _make_corpus(tmp_path, gt="café noir")
        adapter = _MangleOCR(hypothesis="cafe noir")
        bm = run_via_orchestrator(corpus, [adapter])

        dr = bm.engine_reports[0].document_results[0]
        # Les trois cibles de la vue "Analyse des caractères".
        assert dr.confusion_matrix is not None
        assert dr.char_scores is not None
        assert dr.taxonomy is not None
        # Et les compagnons enregistrés dans le même profil.
        assert dr.structure is not None
        assert dr.line_metrics is not None
        assert dr.hallucination_metrics is not None


class TestEngineLevelAggregates:
    def test_aggregated_confusion_present(self, tmp_path: Path) -> None:
        corpus = _make_corpus(tmp_path, gt="abc")
        adapter = _MangleOCR(hypothesis="abd")
        bm = run_via_orchestrator(corpus, [adapter])

        report = bm.engine_reports[0]
        assert report.aggregated_confusion is not None
        assert "matrix" in report.aggregated_confusion

    def test_aggregated_char_scores_carries_ligature_score(
        self, tmp_path: Path,
    ) -> None:
        """``EngineReport.ligature_score`` lit
        ``aggregated_char_scores['ligature']['score']``.  Doit être
        renseigné après un bench standard."""
        corpus = _make_corpus(tmp_path, gt="ﬁn")
        adapter = _MangleOCR(hypothesis="ﬁn")
        bm = run_via_orchestrator(corpus, [adapter])

        report = bm.engine_reports[0]
        assert report.aggregated_char_scores is not None
        assert report.ligature_score == 1.0
        assert report.diacritic_score is not None

    def test_aggregated_taxonomy_present(self, tmp_path: Path) -> None:
        corpus = _make_corpus(tmp_path, gt="été")
        adapter = _MangleOCR(hypothesis="ete")
        bm = run_via_orchestrator(corpus, [adapter])

        report = bm.engine_reports[0]
        assert report.aggregated_taxonomy is not None
        assert "counts" in report.aggregated_taxonomy
        assert report.aggregated_taxonomy["total_errors"] >= 1

    def test_engine_report_as_dict_exposes_aggregates(
        self, tmp_path: Path,
    ) -> None:
        """Le sérialiseur JSON doit propager les agrégats — la vue
        HTML les lit depuis le dict (cf.
        ``picarones.reports.html.data.engines``).
        """
        corpus = _make_corpus(tmp_path, gt="café")
        adapter = _MangleOCR(hypothesis="cafe")
        bm = run_via_orchestrator(corpus, [adapter])

        report_dict = bm.engine_reports[0].as_dict()
        assert "aggregated_confusion" in report_dict
        assert "aggregated_char_scores" in report_dict
        assert "aggregated_taxonomy" in report_dict


class TestProfilePropagation:
    def test_minimal_profile_disables_hooks(
        self, tmp_path: Path,
    ) -> None:
        """``profile="minimal"`` court-circuite tous les hooks doc-level
        et corpus-level — preuve que le runner consomme bien le
        registre ``metric_hooks`` (et non un calcul inline).
        """
        corpus = _make_corpus(tmp_path, gt="café")
        adapter = _MangleOCR(hypothesis="cafe")
        bm = run_via_orchestrator(
            corpus, [adapter], profile="minimal",
        )

        dr = bm.engine_reports[0].document_results[0]
        assert dr.confusion_matrix is None
        assert dr.char_scores is None
        assert dr.taxonomy is None

        report = bm.engine_reports[0]
        assert report.aggregated_confusion is None
        assert report.aggregated_char_scores is None
        assert report.aggregated_taxonomy is None


class TestPartialDirPath:
    def test_aggregates_present_with_partial_dir(
        self, tmp_path: Path,
    ) -> None:
        """Le chemin resumable (``partial_dir``) recalcule les agrégats
        après la fusion ``loaded + new`` — pas seulement le chemin
        rapide ``_run_benchmark_unified``.
        """
        corpus = _make_corpus(tmp_path, gt="été")
        adapter = _MangleOCR(hypothesis="ete", name="resume")
        bm = run_via_orchestrator(
            corpus, [adapter], partial_dir=tmp_path / "partials",
        )

        report = bm.engine_reports[0]
        assert report.aggregated_confusion is not None
        assert report.aggregated_char_scores is not None
        assert report.aggregated_taxonomy is not None
