"""Projecteurs PAGE XML — Sprint A14-S9.

Convertit un ``PageDocument`` (ou un artefact ``PAGE_XML``) vers
d'autres types d'artefacts.  Symétrique de ``formats.alto.projector``.
"""

from __future__ import annotations

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.evaluation.projectors.base import ProjectionReport
from picarones.formats.pagexml.parser import PageParseError, parse_pagexml
from picarones.formats.pagexml.types import PageDocument


def page_document_to_text(document: PageDocument) -> str:
    """Extrait le texte plat d'un ``PageDocument``.

    Convention :
    - Ordre ``Page → TextRegion → TextLine``.
    - Saut de ligne entre lignes d'une même région.
    - Saut de ligne supplémentaire entre régions.
    """
    page_blocks: list[str] = []
    for page in document.pages:
        for region in page.text_regions:
            lines = [tl.text for tl in region.text_lines if tl.text]
            if lines:
                page_blocks.append("\n".join(lines))
    return "\n\n".join(page_blocks).strip()


class PageToText:
    """Projecteur ``PAGE_XML → RAW_TEXT``."""

    name = "page_to_text"
    source_type = ArtifactType.PAGE_XML
    target_type = ArtifactType.RAW_TEXT

    def project(
        self,
        artifact: Artifact,
        params: dict[str, str | int | float | bool],
    ) -> tuple[Artifact, ProjectionReport]:
        from picarones.domain.errors import ProjectionError
        if artifact.type != self.source_type:
            raise ProjectionError(
                f"PageToText n'accepte que PAGE_XML, reçu "
                f"{artifact.type.value!r}"
            )
        if artifact.uri is None:
            raise ProjectionError(
                f"PageToText : artifact {artifact.id!r} sans URI."
            )
        from pathlib import Path
        try:
            xml_bytes = Path(artifact.uri).read_bytes()
        except OSError as exc:
            raise ProjectionError(
                f"PageToText : impossible de lire {artifact.uri!r} : {exc}"
            ) from exc

        try:
            doc = parse_pagexml(xml_bytes)
        except PageParseError as exc:
            raise ProjectionError(f"PageToText : {exc}") from exc

        text = page_document_to_text(doc)

        target = Artifact(
            id=f"{artifact.id}:projected_text",
            document_id=artifact.document_id,
            type=self.target_type,
            produced_by_step=artifact.produced_by_step,
        )
        report = ProjectionReport(
            source_artifact_id=artifact.id,
            source_type=self.source_type,
            target_type=self.target_type,
            projector_name=self.name,
            lossy=True,
            ignored_dimensions=(
                "geometry",
                "region_structure",
                "baseline",
                "ids",
            ),
            warnings=(
                "L'extraction texte PAGE ignore les coordonnées et "
                "la structure en régions.  Plusieurs TextEquiv (variantes "
                "d'OCR) sont collapsées au premier Unicode rencontré.",
            ),
        )
        return target, report


__all__ = ["page_document_to_text", "PageToText"]
