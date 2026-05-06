"""Sprint A14-S16 — sanity check inter-vues sur le cas BnF central.

Vérifie qu'un même pipeline a une cohérence (et parfois une
divergence intéressante) entre TextView, AltoView et SearchView.

Cas démontrés :
- Pipeline parfait → toutes vues maximisent.
- Pipeline avec erreur sur une année → SearchView baisse fortement,
  TextView baisse légèrement (pattern "perte de données critiques
  invisible au CER global").
- Pipeline sans ALTO → AltoView l'OMET, autres vues l'évaluent.
"""

from __future__ import annotations


from picarones.domain import Artifact, ArtifactType, MetricSpec
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
from picarones.formats.alto.types import (
    AltoBBox,
    AltoDocument,
    AltoLine,
    AltoPage,
    AltoString,
    AltoTextBlock,
)


# ──────────────────────────────────────────────────────────────────
# Stubs métriques texte (cer/wer simplifiés sans jiwer)
# ──────────────────────────────────────────────────────────────────


def _stub_cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    common = sum(1 for a, b in zip(reference, hypothesis) if a == b)
    return 1.0 - (common / max(len(reference), len(hypothesis)))


def _stub_wer(reference: str, hypothesis: str) -> float:
    ref_w = reference.split()
    hyp_w = hypothesis.split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    common = sum(1 for a, b in zip(ref_w, hyp_w) if a == b)
    return 1.0 - (common / len(ref_w))


def _build_unified_executor(payloads: dict) -> DefaultEvaluationViewExecutor:
    """Executor configuré pour TextView + AltoView + SearchView."""
    metrics = MetricRegistry()
    # TextView metrics
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
    # AltoView metrics
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
    # SearchView metrics
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

    projectors = ProjectorRegistry()
    projectors.register(AltoToText())
    projectors.register(PageToText())
    projectors.register(CanonicalToText())

    def loader(art: Artifact):
        if art.id not in payloads:
            raise KeyError(art.id)
        return payloads[art.id]

    return DefaultEvaluationViewExecutor.from_registries(metrics, projectors, loader)


# ──────────────────────────────────────────────────────────────────
# Cas 1 — pipeline parfait
# ──────────────────────────────────────────────────────────────────


class TestPerfectPipelineAcrossViews:
    def test_perfect_text_pipeline_maximizes_text_and_search(self) -> None:
        """Un pipeline qui produit du texte parfait :
        - TextView : CER = 0
        - SearchView : recall = 1.0, year preservation = 1.0
        - AltoView : OMIS (pas d'ALTO produit).
        """
        gt_text = "Bonjour Paris en 1789"
        payloads = {"cand": gt_text, "gt_text": gt_text}
        executor = _build_unified_executor(payloads)

        text_view = build_text_view()
        search_view = build_search_view()
        alto_view = build_alto_view()

        cand = Artifact(id="cand", document_id="d", type=ArtifactType.RAW_TEXT)
        gt = Artifact(id="gt_text", document_id="d", type=ArtifactType.RAW_TEXT)

        text_result = executor.evaluate(text_view, cand, gt)
        search_result = executor.evaluate(search_view, cand, gt)

        assert text_result.metric_values["cer"] == 0.0
        assert search_result.metric_values["searchability_recall"] == 1.0
        assert search_result.metric_values["numerical_sequence_preservation"] == 1.0

        # AltoView OMIS : le caller doit filtrer.
        assert not alto_view.accepts(cand.type)


# ──────────────────────────────────────────────────────────────────
# Cas 2 — divergence TextView ↔ SearchView
# ──────────────────────────────────────────────────────────────────


class TestDivergencePattern:
    def test_year_corruption_invisible_to_cer_visible_to_search(self) -> None:
        """Pattern critique : une corruption d'année (1 caractère
        sur ~50) est invisible côté CER mais catastrophique côté
        recherchabilité numérique.

        C'est précisément ce que le rapport BnF doit rendre
        visible — les deux vues racontent des histoires
        complémentaires.
        """
        gt_text = "Charte signée à Paris le 14 juillet 1789 en présence du roi"
        # Hypothèse : le LLM a "corrigé" 1789 en 1798 (faute grossière).
        # Le reste du texte est identique.
        cand_text = "Charte signée à Paris le 14 juillet 1798 en présence du roi"

        payloads = {"cand": cand_text, "gt": gt_text}
        executor = _build_unified_executor(payloads)

        cand = Artifact(id="cand", document_id="d", type=ArtifactType.RAW_TEXT)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)

        text_result = executor.evaluate(build_text_view(), cand, gt)
        search_result = executor.evaluate(build_search_view(), cand, gt)

        # CER ≈ 0.03 (3 chars sur ~58)
        assert text_result.metric_values["cer"] < 0.1, "CER doit rester faible"
        # WER : 1 mot changé sur 11 → 1/11 ≈ 0.09
        assert text_result.metric_values["wer"] < 0.15

        # Mais SearchView : 1789 (GT) n'est PAS dans hyp_years = [1798]
        # → preservation = 0.0 (catastrophique pour un historien).
        assert search_result.metric_values["numerical_sequence_preservation"] == 0.0
        # Searchability : "1789" GT n'est pas matché à "1798" (distance 2,
        # MAIS la longueur est égale, fuzziness ≤ 2 le matche).
        # On vérifie juste qu'il y a un signal mesurable.
        assert search_result.metric_values["searchability_recall"] >= 0.8


# ──────────────────────────────────────────────────────────────────
# Cas 3 — pipeline ALTO évaluable dans les 3 vues
# ──────────────────────────────────────────────────────────────────


def _build_simple_alto(words: list[str], n_lines: int = 1) -> AltoDocument:
    """Construit un AltoDocument avec ``words`` répartis sur
    ``n_lines`` lignes, chaque mot avec une bbox."""
    chunks = [words[i::n_lines] for i in range(n_lines)]
    lines = tuple(
        AltoLine(strings=tuple(
            AltoString(
                content=w,
                bbox=AltoBBox(hpos=0, vpos=0, width=10, height=10),
            )
            for w in chunk
        ))
        for chunk in chunks
    )
    return AltoDocument(pages=(AltoPage(blocks=(AltoTextBlock(lines=lines),),),),)


class TestAltoPipelineEvaluatedInThreeViews:
    def test_alto_pipeline_has_text_alto_search_results(self, tmp_path) -> None:
        """Un pipeline qui produit ALTO_XML est évaluable dans les
        3 vues : TextView (via projection), AltoView (direct),
        SearchView (via projection).
        """
        from picarones.formats.alto import write_alto

        words_gt = "Charte signée Paris 14 juillet 1789".split()
        words_cand = "Charte signée Paris 14 juillet 1789".split()  # identique

        # n_lines=1 pour préserver l'ordre des mots dans l'extraction
        # (sinon ``alto_document_to_text`` produit des sauts de ligne
        # qui font diverger le CER d'une comparaison ligne unique).
        gt_alto = _build_simple_alto(words_gt, n_lines=1)
        cand_alto = _build_simple_alto(words_cand, n_lines=1)
        cand_alto_path = tmp_path / "cand.alto.xml"
        cand_alto_path.write_bytes(write_alto(cand_alto))

        # Payloads : raw text pour les payloads projetés depuis ALTO,
        # AltoDocument pour la GT et le candidat ALTO direct.
        from picarones.evaluation.projectors import alto_document_to_text
        payloads = {
            "gt_text": " ".join(words_gt),
            "gt_alto": gt_alto,
            "cand": cand_alto,  # AltoDocument pour AltoView
            "cand:projected_text": alto_document_to_text(cand_alto),
        }
        executor = _build_unified_executor(payloads)

        gt_text_art = Artifact(id="gt_text", document_id="d", type=ArtifactType.RAW_TEXT)
        gt_alto_art = Artifact(id="gt_alto", document_id="d", type=ArtifactType.ALTO_XML)
        cand_art = Artifact(
            id="cand", document_id="d",
            type=ArtifactType.ALTO_XML, uri=str(cand_alto_path),
        )

        # TextView : projette ALTO → texte, compare au gt_text.
        text_result = executor.evaluate(build_text_view(), cand_art, gt_text_art)
        assert text_result.metric_values["cer"] == 0.0

        # SearchView : projette ALTO → texte, mesure recall + années.
        search_result = executor.evaluate(build_search_view(), cand_art, gt_text_art)
        assert search_result.metric_values["searchability_recall"] == 1.0

        # AltoView : compare ALTO direct contre ALTO GT.
        alto_result = executor.evaluate(build_alto_view(), cand_art, gt_alto_art)
        assert alto_result.metric_values["alto_validity"] == 1.0
        assert alto_result.metric_values["alto_line_count_ratio"] == 1.0
        assert alto_result.metric_values["alto_word_box_coverage"] == 1.0


# ──────────────────────────────────────────────────────────────────
# Cohérence globale : projection report présent ssi projection appliquée
# ──────────────────────────────────────────────────────────────────


class TestProjectionReportConsistency:
    def test_text_search_views_share_projection_report_pattern(self) -> None:
        """Pour un même candidat ALTO_XML évalué dans TextView et
        SearchView, les deux ViewResult doivent porter un
        projection_report (les deux vues projettent vers texte)."""
        gt_text = "test"
        gt_alto = _build_simple_alto(["test"], n_lines=1)
        from picarones.evaluation.projectors import alto_document_to_text
        from picarones.formats.alto import write_alto

        # Pour ce test on n'a pas besoin du fichier réel — on simule
        # via le payload_loader qui retourne directement le texte
        # extrait pour l'id "cand:projected_text".
        payloads = {
            "gt_text": gt_text,
            "cand:projected_text": alto_document_to_text(gt_alto),
        }
        # Mais le projecteur a besoin d'un URI.  On contourne en
        # créant un fichier temporaire dans pytest fixture.
        # Pour ce test simple on écrit dans /tmp.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".alto.xml", delete=False) as f:
            f.write(write_alto(gt_alto))
            cand_uri = f.name

        executor = _build_unified_executor(payloads)
        cand = Artifact(
            id="cand", document_id="d",
            type=ArtifactType.ALTO_XML, uri=cand_uri,
        )
        gt = Artifact(id="gt_text", document_id="d", type=ArtifactType.RAW_TEXT)

        text_result = executor.evaluate(build_text_view(), cand, gt)
        search_result = executor.evaluate(build_search_view(), cand, gt)

        # Les deux doivent avoir un projection_report (même projecteur).
        assert text_result.projection_report is not None
        assert search_result.projection_report is not None
        assert text_result.projection_report.projector_name == "alto_to_text"
        assert search_result.projection_report.projector_name == "alto_to_text"
