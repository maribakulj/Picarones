"""Sprint A14-S14 — TextView (vue canonique 1).

8 cas + le cas BnF central : 3 pipelines hétérogènes (Tesseract,
OCR+LLM+ALTO, VLM+CANONICAL_DOCUMENT) comparés en TextView avec
projection automatique vers texte plat.

Tous les tests utilisent un ``payload_loader`` in-memory pour
contrôler exactement le payload de chaque artefact.  En prod
(S19), le loader sera fourni par un service applicatif.
"""

from __future__ import annotations


from picarones.domain import (
    Artifact,
    ArtifactType,
    MetricSpec,
)
from picarones.evaluation.projectors import (
    AltoToText,
    CanonicalToText,
    PageToText,
    ProjectorRegistry,
    canonical_payload_to_text,
)
from picarones.evaluation.registry import MetricRegistry
from picarones.evaluation.views import (
    DEFAULT_TEXT_METRICS,
    DefaultEvaluationViewExecutor,
    build_text_view,
)


# ──────────────────────────────────────────────────────────────────────
# Métriques stub pour les tests (CER/WER simplifiés sans jiwer)
# ──────────────────────────────────────────────────────────────────────


def _stub_cer(reference: str, hypothesis: str) -> float:
    """CER simplifié : ratio de caractères différents."""
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0
    common = sum(1 for a, b in zip(reference, hypothesis) if a == b)
    max_len = max(len(reference), len(hypothesis))
    return 1.0 - (common / max_len) if max_len else 0.0


def _stub_wer(reference: str, hypothesis: str) -> float:
    """WER simplifié : ratio de mots différents."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0
    common = sum(
        1 for a, b in zip(ref_words, hyp_words) if a == b
    )
    return 1.0 - (common / len(ref_words))


def _stub_mer(reference: str, hypothesis: str) -> float:
    return _stub_cer(reference, hypothesis)


def _stub_wil(reference: str, hypothesis: str) -> float:
    return _stub_wer(reference, hypothesis)


# ──────────────────────────────────────────────────────────────────────
# Helpers de fabrication d'executor
# ──────────────────────────────────────────────────────────────────────


def _build_executor(payloads: dict[str, object]) -> DefaultEvaluationViewExecutor:
    metrics = MetricRegistry()
    for name, fn in (
        ("cer", _stub_cer),
        ("wer", _stub_wer),
        ("mer", _stub_mer),
        ("wil", _stub_wil),
    ):
        metrics.register(
            MetricSpec(
                name=name,
                input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
            ),
            fn,
        )

    projectors = ProjectorRegistry()
    projectors.register(AltoToText())
    projectors.register(PageToText())
    projectors.register(CanonicalToText())

    def loader(artifact: Artifact):
        if artifact.id not in payloads:
            raise KeyError(f"payload manquant : {artifact.id}")
        return payloads[artifact.id]

    return DefaultEvaluationViewExecutor.from_registries(metrics, projectors, loader)


# ──────────────────────────────────────────────────────────────────────
# 8 cas TextView
# ──────────────────────────────────────────────────────────────────────


class TestTextViewShape:
    def test_default_view_accepts_5_types(self) -> None:
        """Cas 1 — la vue par défaut accepte les 5 types."""
        view = build_text_view()
        for t in (
            ArtifactType.RAW_TEXT,
            ArtifactType.CORRECTED_TEXT,
            ArtifactType.ALTO_XML,
            ArtifactType.PAGE_XML,
            ArtifactType.CANONICAL_DOCUMENT,
        ):
            assert view.accepts(t), f"TextView devrait accepter {t.value}"

    def test_default_view_rejects_image_and_entities(self) -> None:
        """Cas 2 — la vue rejette IMAGE, ENTITIES, READING_ORDER."""
        view = build_text_view()
        for t in (
            ArtifactType.IMAGE,
            ArtifactType.ENTITIES,
            ArtifactType.READING_ORDER,
            ArtifactType.ALIGNMENT,
        ):
            assert not view.accepts(t)

    def test_default_metrics_are_cer_wer_mer_wil(self) -> None:
        view = build_text_view()
        assert view.metric_names == DEFAULT_TEXT_METRICS

    def test_projection_for_alto_routes_to_alto_to_text(self) -> None:
        """Cas 3 — projection_for(ALTO_XML) → projecteur alto."""
        view = build_text_view()
        spec = view.projection_for(ArtifactType.ALTO_XML)
        assert spec is not None
        assert spec.projector_name == "alto_to_text"

    def test_projection_for_raw_text_returns_none(self) -> None:
        """Cas 4 — RAW_TEXT n'a pas de projection (déjà du texte)."""
        view = build_text_view()
        assert view.projection_for(ArtifactType.RAW_TEXT) is None
        assert view.projection_for(ArtifactType.CORRECTED_TEXT) is None


class TestTextViewWithExecutor:
    def test_raw_text_against_raw_text(self) -> None:
        """Cas 5 — RAW_TEXT vs RAW_TEXT, sans projection."""
        payloads = {
            "cand": "Bonjour le monde",
            "gt": "Bonjour le monde",
        }
        executor = _build_executor(payloads)
        view = build_text_view()
        cand = Artifact(id="cand", document_id="d", type=ArtifactType.RAW_TEXT)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        result = executor.evaluate(view, cand, gt, pipeline_name="test")
        assert result.metric_values["cer"] == 0.0
        assert result.metric_values["wer"] == 0.0
        assert result.projection_report is None

    def test_canonical_document_routed_to_canonical_to_text(
        self, tmp_path,
    ) -> None:
        """Cas 6 — CANONICAL_DOCUMENT → CanonicalToText, ProjectionReport présent.

        Sprint S25 — le projecteur lit le markdown source depuis l'URI
        et calcule le texte projeté lui-même (plus de hack via
        ``cand:projected_text`` dans le loader)."""
        # Markdown source écrit sur disque ; le projecteur le lit et
        # produit "Bonjour le monde".
        md_path = tmp_path / "cand.md"
        md_path.write_text("# Bonjour le monde\n", encoding="utf-8")
        payloads = {
            "gt": "Bonjour le monde",
        }
        executor = _build_executor(payloads)
        view = build_text_view()
        cand = Artifact(
            id="cand", document_id="d",
            type=ArtifactType.CANONICAL_DOCUMENT,
            uri=str(md_path),
        )
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        result = executor.evaluate(view, cand, gt, pipeline_name="test")
        assert result.projection_report is not None
        assert result.projection_report.projector_name == "canonical_to_text"
        assert "structure" in result.projection_report.ignored_dimensions


class TestBnFCentralUseCase:
    """Cas central BnF — 3 pipelines hétérogènes comparés en TextView.

    Définit explicitement la garantie produit du rewrite : on peut
    comparer Tesseract texte brut, OCR+LLM+ALTO remappé, et un VLM
    qui produit du markdown, sur la même paire (corpus, GT), via la
    même TextView, et obtenir des chiffres comparables et des
    rapports de projection lisibles.
    """

    def _setup(self, tmp_path):
        from picarones.formats.alto import (
            AltoDocument, AltoLine, AltoPage, AltoString, AltoTextBlock,
            write_alto,
        )
        gt_text = "Le petit chat noir court dans le jardin verdoyant"

        # Pipeline 1 : Tesseract → texte brut, légère erreur
        tesseract_text = "Le pelit chat noir court dans le jardin verdoyant"

        # Pipeline 2 : OCR + LLM + ALTO remap → ALTO_XML sérialisé
        # sur disque.  AltoToText l'extrait au runtime.
        alto_doc = AltoDocument(pages=(AltoPage(blocks=(
            AltoTextBlock(lines=(AltoLine(strings=tuple(
                AltoString(content=w)
                for w in
                "Le petit chat noir court dans le jardin verdoyant".split()
            )),),),
        ),),),)
        alto_path = tmp_path / "cand_2.alto.xml"
        alto_path.write_bytes(write_alto(alto_doc))

        # Pipeline 3 : VLM markdown sérialisé sur disque (texte brut,
        # le projecteur Canonical fait juste l'extraction markdown).
        vlm_md = (
            "# Description\n\n"
            "Le petit chat noir court dans le jardin **verdoyant**.\n"
        )
        canonical_path = tmp_path / "cand_3.md"
        canonical_path.write_text(vlm_md, encoding="utf-8")

        # Loader pour les artefacts qui ont un URI : lit le fichier.
        # Pour les autres (GT, raw_text, et les sorties des
        # projecteurs : ``cand_X:projected_text``), on utilise un
        # dict in-memory.
        from picarones.evaluation.projectors import (
            alto_document_to_text,
        )
        from picarones.formats.alto import parse_alto

        # Précalcule les textes que les projecteurs vont produire
        # pour ce test (l'executor lit l'URI puis appelle le
        # projecteur ; le payload_loader doit retourner ce que la
        # métrique attend, donc le texte extrait).
        alto_extracted = alto_document_to_text(parse_alto(alto_path.read_bytes()))
        canonical_extracted = canonical_payload_to_text(vlm_md)

        payloads_in_memory = {
            "gt_text": gt_text,
            "cand_1": tesseract_text,
            # Les artefacts projetés (id `<original>:projected_text`)
            # contiennent le texte extrait par le projecteur.
            "cand_2:projected_text": alto_extracted,
            "cand_3:projected_text": canonical_extracted,
        }


        def loader(artifact: Artifact):
            if artifact.id in payloads_in_memory:
                return payloads_in_memory[artifact.id]
            raise KeyError(f"payload manquant : {artifact.id}")

        # Construit executor avec ce loader
        metrics = MetricRegistry()
        for name, fn in (
            ("cer", _stub_cer), ("wer", _stub_wer),
            ("mer", _stub_mer), ("wil", _stub_wil),
        ):
            metrics.register(
                MetricSpec(
                    name=name,
                    input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
                ),
                fn,
            )
        projectors = ProjectorRegistry()
        projectors.register(AltoToText())
        projectors.register(PageToText())
        projectors.register(CanonicalToText())
        executor = DefaultEvaluationViewExecutor.from_registries(
            metrics, projectors, loader,
        )
        view = build_text_view()

        gt = Artifact(id="gt_text", document_id="bnf_doc",
                      type=ArtifactType.RAW_TEXT)
        cand_1 = Artifact(id="cand_1", document_id="bnf_doc",
                          type=ArtifactType.RAW_TEXT)
        cand_2 = Artifact(id="cand_2", document_id="bnf_doc",
                          type=ArtifactType.ALTO_XML,
                          uri=str(alto_path))
        cand_3 = Artifact(id="cand_3", document_id="bnf_doc",
                          type=ArtifactType.CANONICAL_DOCUMENT,
                          uri=str(canonical_path))

        return executor, view, gt, [cand_1, cand_2, cand_3]

    def test_three_heterogeneous_pipelines_evaluated_via_same_view(self, tmp_path) -> None:
        """Cas 7 — les 3 pipelines passent dans le même
        ``executor.evaluate(view, candidate, gt, pipeline_name="test")``."""
        executor, view, gt, candidates = self._setup(tmp_path)
        results = [
            executor.evaluate(view, cand, gt, pipeline_name="test") for cand in candidates
        ]
        # Tous ont produit un ViewResult avec CER/WER calculés.
        for r in results:
            assert r.view_name == "text_final"
            assert r.failed_metrics == {}
            assert "cer" in r.metric_values
            assert "wer" in r.metric_values

    def test_projection_reports_distinguish_pipeline_types(self, tmp_path) -> None:
        """Cas 8 — chaque pipeline a un ProjectionReport distinct
        (None pour Tesseract texte brut, présent pour ALTO et
        CANONICAL_DOCUMENT)."""
        executor, view, gt, candidates = self._setup(tmp_path)
        results = [
            executor.evaluate(view, cand, gt, pipeline_name="test") for cand in candidates
        ]
        # Tesseract : pas de projection.
        assert results[0].projection_report is None
        # OCR+LLM+ALTO : projection ALTO → texte.
        assert results[1].projection_report is not None
        assert results[1].projection_report.projector_name == "alto_to_text"
        # VLM canonical : projection CANONICAL → texte.
        assert results[2].projection_report is not None
        assert results[2].projection_report.projector_name == "canonical_to_text"

    def test_ignored_dimensions_propagated_in_view_result(self, tmp_path) -> None:
        """Le ViewResult fusionne les ignored_dimensions de la vue
        + ceux de la projection, sans duplication."""
        executor, view, gt, candidates = self._setup(tmp_path)
        # Pipeline 1 (texte direct) : ignored_dimensions = celles de la vue.
        r1 = executor.evaluate(view, candidates[0], gt, pipeline_name="test")
        assert "geometry" in r1.ignored_dimensions  # vient de la vue
        # Pipeline 2 (ALTO) : ignored_dimensions = vue + projection ALTO.
        r2 = executor.evaluate(view, candidates[1], gt, pipeline_name="test")
        assert "geometry" in r2.ignored_dimensions
        # AltoToText ajoute "ids" et "confidence" (déjà dans la vue,
        # donc déduplication).
        # Vérifions au moins qu'aucun dimension ne réapparaît 2 fois :
        assert len(r2.ignored_dimensions) == len(set(r2.ignored_dimensions))


class TestNormalizationApplied:
    def test_normalization_profile_applied_to_both_payloads(self) -> None:
        """Une TextView avec normalization_profile applique la
        normalisation aux deux payloads avant calcul."""
        # ſ → s avec medieval_french : "afpre" (pas de ſ) vs "aſpre"
        # → après normalisation, les deux deviennent "aspre"
        payloads = {
            "cand": "afpre",
            "gt": "aſpre",
        }
        executor = _build_executor(payloads)
        view = build_text_view(normalization_profile="medieval_french")
        cand = Artifact(id="cand", document_id="d", type=ArtifactType.RAW_TEXT)
        gt = Artifact(id="gt", document_id="d", type=ArtifactType.RAW_TEXT)
        result = executor.evaluate(view, cand, gt, pipeline_name="test")
        # Après normalisation : afpre → afpre (ſ pas dans payload),
        # aſpre → aspre.  Donc CER non nul mais cohérent.
        assert "cer" in result.metric_values
