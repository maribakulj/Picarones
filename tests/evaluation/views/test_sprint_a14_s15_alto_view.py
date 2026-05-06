"""Sprint A14-S15 — AltoView (vue canonique 2).

6 cas couvrant la fidélité documentaire ALTO + le pattern
d'omission explicite des pipelines qui ne produisent pas d'ALTO.
"""

from __future__ import annotations

import pytest

from picarones.domain import (
    Artifact,
    ArtifactType,
    MetricSpec,
)
from picarones.evaluation.metrics.alto_structural import (
    compute_alto_validity,
    compute_line_count_ratio,
    compute_word_box_coverage,
)
from picarones.evaluation.projectors import ProjectorRegistry
from picarones.evaluation.registry import MetricRegistry
from picarones.evaluation.views import (
    DEFAULT_ALTO_METRICS,
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


# ──────────────────────────────────────────────────────────────────────
# Fixtures ALTO
# ──────────────────────────────────────────────────────────────────────


def _line(*words: str, with_bbox: bool = True) -> AltoLine:
    strings = tuple(
        AltoString(
            content=w,
            bbox=AltoBBox(hpos=0, vpos=0, width=10, height=10) if with_bbox else None,
        )
        for w in words
    )
    return AltoLine(strings=strings)


def _doc(*lines: AltoLine, n_blocks: int = 1) -> AltoDocument:
    """Construit un AltoDocument avec ``n_blocks`` blocs partageant
    les lignes."""
    if n_blocks == 1:
        return AltoDocument(pages=(AltoPage(
            blocks=(AltoTextBlock(lines=lines),),
        ),),)
    # Distribute lines across blocks (tous identiques pour simplifier)
    chunks = [lines] * n_blocks
    return AltoDocument(pages=(AltoPage(
        blocks=tuple(AltoTextBlock(lines=c) for c in chunks),
    ),),)


def _empty_doc() -> AltoDocument:
    return AltoDocument()


# ──────────────────────────────────────────────────────────────────────
# Métriques individuelles
# ──────────────────────────────────────────────────────────────────────


class TestAltoMetrics:
    def test_validity_full_doc(self) -> None:
        d = _doc(_line("a", "b"))
        assert compute_alto_validity(d, d) == 1.0

    def test_validity_empty_doc(self) -> None:
        assert compute_alto_validity(_doc(_line("a")), _empty_doc()) == 0.0

    def test_line_count_ratio_equal(self) -> None:
        d1 = _doc(_line("a"), _line("b"), _line("c"))
        d2 = _doc(_line("x"), _line("y"), _line("z"))
        assert compute_line_count_ratio(d1, d2) == 1.0

    def test_line_count_ratio_partial(self) -> None:
        d1 = _doc(_line("a"), _line("b"), _line("c"), _line("d"))  # 4
        d2 = _doc(_line("x"), _line("y"))  # 2
        assert compute_line_count_ratio(d1, d2) == 0.5

    def test_line_count_ratio_both_empty(self) -> None:
        assert compute_line_count_ratio(_empty_doc(), _empty_doc()) == 1.0

    def test_word_box_coverage_full(self) -> None:
        d = _doc(_line("a", "b", "c", with_bbox=True))
        assert compute_word_box_coverage(d, d) == 1.0

    def test_word_box_coverage_partial(self) -> None:
        # 2 mots avec bbox, 1 sans
        line = AltoLine(strings=(
            AltoString(content="a", bbox=AltoBBox(hpos=0, vpos=0, width=1, height=1)),
            AltoString(content="b", bbox=AltoBBox(hpos=0, vpos=0, width=1, height=1)),
            AltoString(content="c", bbox=None),
        ))
        d = AltoDocument(pages=(AltoPage(blocks=(AltoTextBlock(lines=(line,),),),),),)
        assert abs(compute_word_box_coverage(d, d) - 2 / 3) < 1e-9

    def test_word_box_coverage_no_bbox(self) -> None:
        d = _doc(_line("a", "b", with_bbox=False))
        assert compute_word_box_coverage(d, d) == 0.0


# ──────────────────────────────────────────────────────────────────────
# AltoView shape
# ──────────────────────────────────────────────────────────────────────


class TestAltoViewShape:
    def test_default_view_accepts_only_alto_xml(self) -> None:
        """Cas 1 — AltoView n'accepte que ALTO_XML."""
        view = build_alto_view()
        assert view.accepts(ArtifactType.ALTO_XML)
        assert not view.accepts(ArtifactType.RAW_TEXT)
        assert not view.accepts(ArtifactType.PAGE_XML)
        assert not view.accepts(ArtifactType.CANONICAL_DOCUMENT)
        assert not view.accepts(ArtifactType.IMAGE)

    def test_default_metrics(self) -> None:
        view = build_alto_view()
        assert view.metric_names == DEFAULT_ALTO_METRICS
        assert "alto_validity" in view.metric_names
        assert "alto_line_count_ratio" in view.metric_names
        assert "alto_word_box_coverage" in view.metric_names

    def test_no_projection(self) -> None:
        view = build_alto_view()
        assert view.projection is None
        # Pas de projection même par type source.
        assert view.projection_for(ArtifactType.ALTO_XML) is None

    def test_warnings_signal_omission_pattern(self) -> None:
        view = build_alto_view()
        warnings_text = " ".join(view.warnings)
        assert "OMIS" in warnings_text or "omis" in warnings_text


# ──────────────────────────────────────────────────────────────────────
# AltoView avec executor
# ──────────────────────────────────────────────────────────────────────


def _build_alto_executor(payloads: dict[str, AltoDocument]) -> DefaultEvaluationViewExecutor:
    metrics = MetricRegistry()
    metrics.register(
        MetricSpec(
            name="alto_validity",
            input_types=(ArtifactType.ALTO_XML, ArtifactType.ALTO_XML),
            higher_is_better=True,
        ),
        compute_alto_validity,
    )
    metrics.register(
        MetricSpec(
            name="alto_line_count_ratio",
            input_types=(ArtifactType.ALTO_XML, ArtifactType.ALTO_XML),
            higher_is_better=True,
        ),
        compute_line_count_ratio,
    )
    metrics.register(
        MetricSpec(
            name="alto_word_box_coverage",
            input_types=(ArtifactType.ALTO_XML, ArtifactType.ALTO_XML),
            higher_is_better=True,
        ),
        compute_word_box_coverage,
    )
    projectors = ProjectorRegistry()  # AltoView n'a pas besoin de projecteur

    def loader(art: Artifact) -> AltoDocument:
        if art.id not in payloads:
            raise KeyError(f"missing payload {art.id}")
        return payloads[art.id]

    return DefaultEvaluationViewExecutor.from_registries(metrics, projectors, loader)


class TestAltoViewWithExecutor:
    def test_perfect_alto_yields_all_ones(self) -> None:
        """Cas 2 — Hypothèse identique à la GT → toutes métriques = 1.0."""
        gt = _doc(_line("a", "b"), _line("c", "d"))
        payloads = {"gt": gt, "cand": gt}
        executor = _build_alto_executor(payloads)
        view = build_alto_view()
        gt_art = Artifact(id="gt", document_id="d", type=ArtifactType.ALTO_XML)
        cand = Artifact(id="cand", document_id="d", type=ArtifactType.ALTO_XML)
        result = executor.evaluate(view, cand, gt_art, pipeline_name="test")
        assert result.metric_values["alto_validity"] == 1.0
        assert result.metric_values["alto_line_count_ratio"] == 1.0
        assert result.metric_values["alto_word_box_coverage"] == 1.0
        assert result.failed_metrics == {}

    def test_partial_quality_alto(self) -> None:
        """Cas 3 — Hypothèse avec moins de lignes → ratio < 1, autres OK."""
        gt = _doc(_line("a"), _line("b"), _line("c"), _line("d"))  # 4 lignes
        cand = _doc(_line("x"), _line("y"))  # 2 lignes
        payloads = {"gt": gt, "cand": cand}
        executor = _build_alto_executor(payloads)
        view = build_alto_view()
        gt_art = Artifact(id="gt", document_id="d", type=ArtifactType.ALTO_XML)
        cand_art = Artifact(id="cand", document_id="d", type=ArtifactType.ALTO_XML)
        result = executor.evaluate(view, cand_art, gt_art, pipeline_name="test")
        assert result.metric_values["alto_validity"] == 1.0  # cohérent
        assert result.metric_values["alto_line_count_ratio"] == 0.5
        assert result.metric_values["alto_word_box_coverage"] == 1.0


# ──────────────────────────────────────────────────────────────────────
# Pattern d'omission : pipelines sans ALTO ne sont PAS dans AltoView
# ──────────────────────────────────────────────────────────────────────


class TestOmissionPattern:
    """Le caller (service applicatif) doit OMETTRE les pipelines qui
    ne produisent pas d'ALTO_XML, plutôt que de leur attribuer un
    score factice à 0.

    Le test démontre le pattern recommandé.
    """

    def test_caller_filters_pipelines_by_view_acceptance(self) -> None:
        """Cas 4 — Pattern : boucler sur (vue, candidats), filtrer
        ceux dont le type n'est pas dans candidate_types."""
        view = build_alto_view()

        # Simulons 3 pipelines avec leurs sorties principales :
        candidates = [
            ("tesseract_text", ArtifactType.RAW_TEXT),       # PAS d'ALTO
            ("ocr_llm_alto", ArtifactType.ALTO_XML),         # ALTO ✓
            ("vlm_alto_reconstructed", ArtifactType.ALTO_XML),  # ALTO ✓
        ]

        # Le caller filtre :
        eligible = [
            (name, art_type)
            for name, art_type in candidates
            if view.accepts(art_type)
        ]

        omitted = [
            (name, art_type)
            for name, art_type in candidates
            if not view.accepts(art_type)
        ]

        assert len(eligible) == 2
        assert ("ocr_llm_alto", ArtifactType.ALTO_XML) in eligible
        assert ("vlm_alto_reconstructed", ArtifactType.ALTO_XML) in eligible

        assert len(omitted) == 1
        assert omitted[0][0] == "tesseract_text"

    def test_executor_raises_value_error_if_caller_doesnt_filter(self) -> None:
        """Cas 5 — Garde-fou : si le caller n'a pas filtré et passe
        un RAW_TEXT à AltoView, ``executor.evaluate`` lève ``ValueError``
        explicite."""
        payloads = {"cand": "this is text", "gt": _doc(_line("a"))}
        executor = _build_alto_executor(payloads)
        view = build_alto_view()
        cand = Artifact(id="cand", document_id="d", type=ArtifactType.RAW_TEXT)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.ALTO_XML)
        with pytest.raises(ValueError, match="n'accepte pas"):
            executor.evaluate(view, cand, gt, pipeline_name="test")


# ──────────────────────────────────────────────────────────────────────
# Cas central BnF : TextView + AltoView complémentaires
# ──────────────────────────────────────────────────────────────────────


class TestBnFDualViewUsage:
    """Démontre que le rapport BnF cible peut présenter TextView ET
    AltoView pour les **mêmes** pipelines, mais avec des sets de
    pipelines différents.

    Pipeline 1 : Tesseract texte brut → présent dans TextView, OMIS d'AltoView.
    Pipeline 2 : OCR+LLM avec ALTO → présent dans les DEUX.
    Pipeline 3 : VLM avec ALTO reconstruit → présent dans les DEUX.

    Le test ne fait PAS l'évaluation complète (la stub mémoire ne
    porte que ce qui est utile).  Il vérifie le **pattern** : pour
    chaque vue, quels pipelines sont éligibles.
    """

    def test_two_views_select_different_pipeline_sets(self) -> None:
        """Cas 6 — Définition de done S15 :
          * Tesseract → omis d'AltoView, présent dans TextView
          * OCR+LLM+ALTO → dans les deux
          * VLM+ALTO → dans les deux
        """
        text_view = build_text_view()
        alto_view = build_alto_view()

        pipelines = [
            ("tesseract", ArtifactType.RAW_TEXT),
            ("ocr_llm_alto", ArtifactType.ALTO_XML),
            ("vlm_alto", ArtifactType.ALTO_XML),
        ]

        text_eligible = {
            n for n, t in pipelines if text_view.accepts(t)
        }
        alto_eligible = {
            n for n, t in pipelines if alto_view.accepts(t)
        }

        # TextView accepte les 3.
        assert text_eligible == {"tesseract", "ocr_llm_alto", "vlm_alto"}

        # AltoView omet Tesseract, garde les 2 ALTO.
        assert alto_eligible == {"ocr_llm_alto", "vlm_alto"}
        assert "tesseract" not in alto_eligible

        # Les pipelines présents dans AltoView sont un SOUS-ENSEMBLE de
        # ceux présents dans TextView (cohérence : si un pipeline
        # produit de l'ALTO, son texte est aussi extractible).
        assert alto_eligible.issubset(text_eligible)
