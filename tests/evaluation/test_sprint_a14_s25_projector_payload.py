"""Sprint A14-S25 — projection sans hack loader.

Le test central qui démontre que le fix du protocole ``Projector``
(retourne ``(Artifact, payload, ProjectionReport)`` au lieu de
``(Artifact, ProjectionReport)``) débloque le workflow CLI :
on peut maintenant exécuter une pipeline qui produit ALTO_XML, la
faire évaluer par TextView (qui projette ALTO → texte), et obtenir
des métriques **sans pré-stocker manuellement le payload projeté
dans le loader**.

C'est précisément le cas BnF central :
- Pipeline 1 : Tesseract → RAW_TEXT (TextView direct).
- Pipeline 2 : Pero OCR → ALTO_XML (TextView via projection
  ALTO→texte).

Les deux pipelines doivent être comparables sur la même TextView.
"""

from __future__ import annotations

from pathlib import Path

from picarones.app.services import RegistryService
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.evaluation_spec import MetricSpec
from picarones.evaluation.registry import MetricRegistry
from picarones.evaluation.views import (
    DefaultEvaluationViewExecutor,
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


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _build_alto(text: str) -> AltoDocument:
    return AltoDocument(pages=(AltoPage(blocks=(AltoTextBlock(lines=(AltoLine(strings=tuple(
        AltoString(content=w, bbox=AltoBBox(hpos=0, vpos=0, width=10, height=10))
        for w in text.split()
    )),),),),),),)


def _stub_cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    common = sum(1 for a, b in zip(reference, hypothesis) if a == b)
    return 1.0 - (common / max(len(reference), len(hypothesis)))


def _strict_loader(art: Artifact):
    """Loader qui REFUSE explicitement les artefacts projetés.

    Si l'executor essaie d'appeler ``loader(art)`` sur un artefact
    dont l'id se termine par ``:projected_text``, on lève — preuve
    que le fix S25 fait que l'executor n'appelle PAS le loader sur
    les artefacts projetés.

    Pour les autres artefacts (RAW_TEXT/ALTO_XML avec URI), on lit
    depuis le filesystem.
    """
    if ":projected_text" in art.id:
        raise AssertionError(
            f"S25 régression : le loader a été appelé sur "
            f"l'artefact projeté {art.id!r} — le fix S25 garantit que "
            "le payload est utilisé directement depuis le retour du "
            "projecteur, sans repasser par le loader."
        )
    if art.type == ArtifactType.RAW_TEXT:
        return Path(art.uri).read_text(encoding="utf-8")
    if art.type == ArtifactType.ALTO_XML:
        from picarones.formats.alto.parser import parse_alto
        return parse_alto(Path(art.uri).read_bytes())
    raise KeyError(f"loader strict : type {art.type} non géré")


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────


class TestProjectionWithoutLoaderHack:
    """Avant S25, l'executor appelait ``loader(projected_artifact)`` —
    obligeant les tests à pré-stocker le payload projeté dans une map.
    Après S25, le projecteur retourne le payload directement et
    l'executor ne sollicite plus le loader pour les artefacts projetés.
    """

    def test_alto_to_text_projection_works_without_loader_hack(
        self, tmp_path: Path,
    ) -> None:
        # Setup : un ALTO sur disque + une GT texte sur disque.
        gt_text = "Bonjour le monde"
        alto_doc = _build_alto(gt_text)
        alto_path = tmp_path / "doc.alto.xml"
        alto_path.write_bytes(write_alto(alto_doc))

        gt_path = tmp_path / "doc.gt.txt"
        gt_path.write_text(gt_text, encoding="utf-8")

        # Bootstrap registries via le service S23.
        registries = RegistryService.bootstrap_defaults()

        # Loader strict qui ASSERTE qu'il n'est pas appelé sur l'artefact
        # projeté.
        executor = DefaultEvaluationViewExecutor.from_registries(
            registries.metrics,
            registries.projectors,
            _strict_loader,
        )

        # Candidat : ALTO_XML.  GT : RAW_TEXT.  Vue : TextView qui
        # projette ALTO → texte.
        cand = Artifact(
            id="d1:pero:alto",
            document_id="d1",
            type=ArtifactType.ALTO_XML,
            uri=str(alto_path),
        )
        gt = Artifact(
            id="d1:gt:raw_text",
            document_id="d1",
            type=ArtifactType.RAW_TEXT,
            uri=str(gt_path),
        )
        view = build_text_view()
        result = executor.evaluate(view, cand, gt, pipeline_name="test")

        # Validation : la projection a bien eu lieu, le payload retourné
        # par le projecteur a été utilisé (le loader strict aurait levé
        # sinon), et le CER est 0 puisque le texte ALTO matche la GT.
        assert result.projection_report is not None
        assert result.projection_report.projector_name == "alto_to_text"
        assert result.failed_metrics == {}, (
            f"Métriques en échec inattendues : {result.failed_metrics}"
        )
        assert result.metric_values["cer"] == 0.0
        assert result.metric_values["wer"] == 0.0

    def test_canonical_to_text_projection_works_without_loader_hack(
        self, tmp_path: Path,
    ) -> None:
        # Setup : markdown sur disque + GT texte.
        md_path = tmp_path / "doc.canonical.md"
        md_path.write_text(
            "# Titre\n\nBonjour le monde\n",
            encoding="utf-8",
        )
        gt_path = tmp_path / "doc.gt.txt"
        gt_path.write_text("Titre Bonjour le monde", encoding="utf-8")

        registries = RegistryService.bootstrap_defaults()
        executor = DefaultEvaluationViewExecutor.from_registries(
            registries.metrics,
            registries.projectors,
            _strict_loader,
        )

        cand = Artifact(
            id="d1:vlm:canonical",
            document_id="d1",
            type=ArtifactType.CANONICAL_DOCUMENT,
            uri=str(md_path),
        )
        gt = Artifact(
            id="d1:gt:raw_text",
            document_id="d1",
            type=ArtifactType.RAW_TEXT,
            uri=str(gt_path),
        )
        view = build_text_view()
        result = executor.evaluate(view, cand, gt, pipeline_name="test")

        assert result.projection_report is not None
        assert result.projection_report.projector_name == "canonical_to_text"
        assert result.failed_metrics == {}, (
            f"Métriques en échec inattendues : {result.failed_metrics}"
        )

    def test_loader_still_called_for_non_projected_candidate(
        self, tmp_path: Path,
    ) -> None:
        """Garde-fou : le loader EST appelé pour les artefacts non
        projetés (RAW_TEXT direct), juste pas pour les projetés.
        Vérifie qu'on n'a pas accidentellement court-circuité
        TOUS les chemins."""
        gt_text = "Identique"
        cand_path = tmp_path / "cand.txt"
        cand_path.write_text(gt_text, encoding="utf-8")
        gt_path = tmp_path / "gt.txt"
        gt_path.write_text(gt_text, encoding="utf-8")

        registries = RegistryService.bootstrap_defaults()
        executor = DefaultEvaluationViewExecutor.from_registries(
            registries.metrics,
            registries.projectors,
            _strict_loader,
        )

        cand = Artifact(
            id="d1:tess:raw_text",
            document_id="d1",
            type=ArtifactType.RAW_TEXT,
            uri=str(cand_path),
        )
        gt = Artifact(
            id="d1:gt:raw_text",
            document_id="d1",
            type=ArtifactType.RAW_TEXT,
            uri=str(gt_path),
        )
        view = build_text_view()
        result = executor.evaluate(view, cand, gt, pipeline_name="test")

        # Pas de projection → loader appelé sur le candidat directement.
        assert result.projection_report is None
        assert result.metric_values["cer"] == 0.0


class TestPayloadFromProjectorIsAuthoritative:
    """Garantit que le payload retourné par le projecteur est utilisé
    tel quel (l'executor ne re-réécrit pas, ne re-charge pas)."""

    def test_alto_projector_payload_drives_metric(
        self, tmp_path: Path,
    ) -> None:
        """Quand le projecteur retourne 'X', le métrique compute sur 'X'
        (pas sur autre chose)."""
        gt_text = "exact"
        alto_path = tmp_path / "alto.xml"
        alto_path.write_bytes(write_alto(_build_alto("exact")))

        gt_path = tmp_path / "gt.txt"
        gt_path.write_text(gt_text, encoding="utf-8")

        # Métrique custom qui retourne 1.0 si reference == hypothesis,
        # 0.0 sinon — preuve que la valeur passée à la métrique est
        # bien le payload du projecteur.
        from picarones.evaluation.projectors import ProjectorRegistry, AltoToText

        captured: dict[str, str] = {}

        def capturing_metric(reference: str, hypothesis: str) -> float:
            captured["reference"] = reference
            captured["hypothesis"] = hypothesis
            return 1.0 if reference == hypothesis else 0.0

        metrics = MetricRegistry()
        metrics.register(
            MetricSpec(
                name="capture",
                input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
                higher_is_better=True,
            ),
            capturing_metric,
        )
        projectors = ProjectorRegistry()
        projectors.register(AltoToText())

        from picarones.domain.evaluation_spec import EvaluationView
        from picarones.domain.projection_spec import ProjectionSpec

        # On ne peut pas utiliser build_text_view car ses metric_names
        # incluent cer/wer/mer/wil non enregistrés ici — on construit
        # une vue minimale qui projette ALTO → texte.
        view = EvaluationView(
            name="test_capture",
            description="capture le payload projeté",
            candidate_types=frozenset({ArtifactType.ALTO_XML}),
            projections_by_source_type={
                ArtifactType.ALTO_XML: ProjectionSpec(
                    source_type=ArtifactType.ALTO_XML,
                    target_type=ArtifactType.RAW_TEXT,
                    projector_name="alto_to_text",
                ),
            },
            metric_names=("capture",),
        )

        executor = DefaultEvaluationViewExecutor.from_registries(
            metrics, projectors, _strict_loader,
        )
        cand = Artifact(
            id="d:alto",
            document_id="d",
            type=ArtifactType.ALTO_XML,
            uri=str(alto_path),
        )
        gt = Artifact(
            id="d:gt",
            document_id="d",
            type=ArtifactType.RAW_TEXT,
            uri=str(gt_path),
        )
        result = executor.evaluate(view, cand, gt, pipeline_name="test")
        assert captured["reference"] == "exact"
        assert captured["hypothesis"] == "exact"
        assert result.metric_values["capture"] == 1.0
