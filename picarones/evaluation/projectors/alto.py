"""Projecteurs ALTO — Sprint A14-S9.

Convertit un ``AltoDocument`` (ou un artefact ``ALTO_XML``) vers
d'autres types d'artefacts, en documentant explicitement les
pertes via ``ProjectionReport``.

Implémentations
---------------
- ``AltoToText`` — extraction du texte par ordre de lecture
  ``Page → Block → Line → String``.  Gestion césure
  ``HypPart1``/``HypPart2``.

À venir post-livraison :
- ``AltoToLines`` (extraction lignes).
- ``AltoToWordsWithBoxes`` (mots + coordonnées).
"""

from __future__ import annotations

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.evaluation.projectors.base import ProjectionReport
from picarones.formats.alto.parser import AltoParseError, parse_alto
from picarones.formats.alto.types import AltoDocument, AltoLine


def alto_document_to_text(document: AltoDocument) -> str:
    """Extrait le texte plat d'un ``AltoDocument``.

    Conventions :

    - Ordre de lecture ``Page → Block → Line → String``, dans l'ordre
      d'apparition dans le XML.
    - Espace entre les ``String`` d'une même ligne.
    - Saut de ligne entre les ``TextLine``.
    - Saut de ligne supplémentaire entre les ``TextBlock``.
    - **Césure** :
      - Si un ``HypPart1`` porte ``SUBS_CONTENT`` (mot complet), on
        utilise ce mot complet et on saute le ``HypPart2``
        correspondant (même ligne ou ligne suivante du même bloc).
      - Sinon, on concatène ``HypPart1.content + HypPart2.content``
        et on saute le ``HypPart2``.
      - Le saut de ligne visuel entre les deux est **conservé** (le
        mot reconstruit termine la ligne du ``HypPart1``, la ligne
        du ``HypPart2`` continue avec ses autres mots).
    """
    blocks_text: list[str] = []
    for page in document.pages:
        for block in page.blocks:
            block_text = _extract_block_text(block)
            if block_text:
                blocks_text.append(block_text)
    return "\n\n".join(blocks_text).strip()


def _extract_block_text(block: "AltoTextBlock") -> str:
    """Extrait le texte d'un bloc en gérant la césure cross-ligne.

    L'usage standard ALTO place ``HypPart1`` en fin d'une ligne et
    ``HypPart2`` en début de la ligne suivante du **même** bloc.
    """
    from picarones.formats.alto.types import AltoTextBlock as _ATB
    assert isinstance(block, _ATB)
    lines_text: list[str] = []
    skip_first_if_hyppart2 = False
    for line in block.lines:
        text, ended_with_hyp1 = _extract_line_text(
            line, skip_first_if_hyppart2=skip_first_if_hyppart2,
        )
        lines_text.append(text)
        skip_first_if_hyppart2 = ended_with_hyp1
    return "\n".join(lines_text)


def _extract_line_text(
    line: AltoLine,
    *,
    skip_first_if_hyppart2: bool = False,
) -> tuple[str, bool]:
    """Reconstruit le texte d'une ligne.

    Returns
    -------
    tuple[str, bool]
        ``(texte_ligne, ended_with_hyppart1_resolved)``.  Le second
        indique si la ligne se termine par un ``HypPart1`` dont la
        résolution implique de skipper le premier ``HypPart2`` de la
        ligne suivante.
    """
    parts: list[str] = []
    skip_next = False
    ended_with_hyp1 = False
    strings = list(line.strings)
    for i, s in enumerate(strings):
        is_first = (i == 0)
        if skip_next:
            skip_next = False
            continue
        if is_first and skip_first_if_hyppart2 and s.subs_type == "HypPart2":
            # Cross-ligne : la ligne précédente a résolu le HypPart1.
            continue
        if s.subs_type == "HypPart1":
            is_last = (i == len(strings) - 1)
            if s.subs_content:
                parts.append(s.subs_content)
                if i + 1 < len(strings) and strings[i + 1].subs_type == "HypPart2":
                    skip_next = True
                elif is_last:
                    ended_with_hyp1 = True
                continue
            if i + 1 < len(strings) and strings[i + 1].subs_type == "HypPart2":
                parts.append(s.content + strings[i + 1].content)
                skip_next = True
                continue
            parts.append(s.content)
            if is_last:
                ended_with_hyp1 = True
            continue
        parts.append(s.content)
    return " ".join(p for p in parts if p), ended_with_hyp1


# ──────────────────────────────────────────────────────────────────────
# Projecteur conforme au protocole ``Projector`` (Sprint S5)
# ──────────────────────────────────────────────────────────────────────


class AltoToText:
    """Projecteur ``ALTO_XML → RAW_TEXT``.

    Lit le XML depuis l'``Artifact.uri`` (chemin filesystem) si
    présent, sinon attend que le caller ait pré-stocké le payload
    dans un mécanisme externe (ce projecteur ne télécharge rien
    par lui-même — pas de side-effect réseau).

    Pour S9, on s'attend à ce que ``artifact.uri`` pointe vers un
    fichier local lisible.  Le service applicatif (S19) résoudra
    les autres cas (URI distante, payload inline).
    """

    name = "alto_to_text"
    source_type = ArtifactType.ALTO_XML
    target_type = ArtifactType.RAW_TEXT

    def project(
        self,
        artifact: Artifact,
        params: dict[str, str | int | float | bool],
    ) -> tuple[Artifact, str, ProjectionReport]:
        if artifact.type != self.source_type:
            from picarones.domain.errors import ProjectionError
            raise ProjectionError(
                f"AltoToText n'accepte que ALTO_XML, reçu "
                f"{artifact.type.value!r}"
            )

        # Lecture du XML.  Pour S9, on lit depuis le filesystem.
        xml_bytes = self._read_xml(artifact)

        try:
            doc = parse_alto(xml_bytes)
        except AltoParseError as exc:
            from picarones.domain.errors import ProjectionError
            raise ProjectionError(f"AltoToText : {exc}") from exc

        text = alto_document_to_text(doc)

        # Construction de l'artefact résultat.
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
                "block_structure",
                "reading_order",
                "ids",
                "confidence",
            ),
            warnings=(
                "L'extraction texte ALTO ignore les coordonnées, "
                "la structure en blocs, et les IDs.  La césure "
                "HypPart1/HypPart2 est résolue (mot recombiné).",
            ),
        )
        return target, text, report

    @staticmethod
    def _read_xml(artifact: Artifact) -> bytes:
        from picarones.domain.errors import ProjectionError
        if artifact.uri is None:
            raise ProjectionError(
                f"AltoToText : artifact {artifact.id!r} n'a pas d'URI "
                "et le projecteur ne sait pas résoudre les payloads "
                "inline pour S9."
            )
        from pathlib import Path
        path = Path(artifact.uri)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise ProjectionError(
                f"AltoToText : impossible de lire {path!r} : {exc}"
            ) from exc


__all__ = ["alto_document_to_text", "AltoToText"]
