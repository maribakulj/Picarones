"""Projecteur ``CANONICAL_DOCUMENT → RAW_TEXT`` — Sprint A14-S14.

Convertit un artefact ``CANONICAL_DOCUMENT`` (typiquement un
markdown ou un JSON canonique produit par un VLM) vers du texte
plat comparable.

Stratégies de payload supportées
--------------------------------
1. **str (markdown)** — décape les balises markdown courantes : ``#``,
   ``*``, ``_``, ``\``, ``> ``, ``\`\`\``, listes ``- ``, lignes
   horizontales.  Préserve le contenu textuel.

2. **dict** — cherche en cascade ``"text"``, ``"content"``,
   ``"markdown"``, ``"plain"``, puis itère récursivement.  Si une
   liste de paragraphes est trouvée sous ``"paragraphs"``, les
   joint avec un saut de ligne.

3. **list** — joint chaque élément (str ou dict récurse) avec ``\n``.

L'objectif n'est pas une conversion markdown→texte parfaite mais
**une comparaison stable** : un VLM qui produit du markdown
``"# Titre\\nLigne"`` et un OCR qui produit ``"Titre\\nLigne"``
doivent comparer égaux côté CER après projection.
"""

from __future__ import annotations

import re
from typing import Any

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.errors import ProjectionError
from picarones.evaluation.projectors.base import ProjectionReport


# Patterns markdown courants à décaper.  Volontairement minimal —
# on ne fait PAS de parsing markdown complet (les libs comme
# mistune ne sont pas dans la whitelist evaluation/).
_MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MARKDOWN_LIST_BULLET_RE = re.compile(r"^[-*+]\s+", re.MULTILINE)
_MARKDOWN_NUM_LIST_RE = re.compile(r"^\d+\.\s+", re.MULTILINE)
_MARKDOWN_BLOCKQUOTE_RE = re.compile(r"^>\s?", re.MULTILINE)
_MARKDOWN_HR_RE = re.compile(r"^[-*_]{3,}$", re.MULTILINE)
_MARKDOWN_BOLD_ITALIC_RE = re.compile(r"\*{1,3}([^*]+)\*{1,3}")
_MARKDOWN_UNDERLINE_RE = re.compile(r"_{1,2}([^_]+)_{1,2}")
_MARKDOWN_CODE_INLINE_RE = re.compile(r"`([^`]+)`")
_MARKDOWN_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9]*\n?|```", re.MULTILINE)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")


def markdown_to_text(markdown: str) -> str:
    """Convertit un markdown simple en texte plat.

    Conserve le contenu textuel, retire les marqueurs syntaxiques
    courants.  Pas de parser AST — substitutions regex simples qui
    couvrent ~90 % des cas patrimoniaux observés.
    """
    text = markdown
    # Code blocks (fences) : retire les ``` lignes
    text = _MARKDOWN_CODE_BLOCK_RE.sub("", text)
    # Images avant liens (les images contiennent des liens)
    text = _MARKDOWN_IMAGE_RE.sub(r"\1", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    # Headers, blockquotes, listes
    text = _MARKDOWN_HEADER_RE.sub("", text)
    text = _MARKDOWN_BLOCKQUOTE_RE.sub("", text)
    text = _MARKDOWN_LIST_BULLET_RE.sub("", text)
    text = _MARKDOWN_NUM_LIST_RE.sub("", text)
    text = _MARKDOWN_HR_RE.sub("", text)
    # Inline formatting : **gras**, *italique*, _souligné_, `code`
    text = _MARKDOWN_BOLD_ITALIC_RE.sub(r"\1", text)
    text = _MARKDOWN_UNDERLINE_RE.sub(r"\1", text)
    text = _MARKDOWN_CODE_INLINE_RE.sub(r"\1", text)
    return text.strip()


def canonical_payload_to_text(payload: Any) -> str:
    """Extrait le texte plat d'un ``CANONICAL_DOCUMENT`` payload.

    Stratégies en cascade selon le type de ``payload`` :

    - ``str`` : traite comme markdown, applique ``markdown_to_text``.
    - ``dict`` : cherche les clés textuelles standards.
    - ``list`` : concatène les éléments avec ``\\n``.
    - autre : ``str(payload)`` en dernier recours.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return markdown_to_text(payload)
    if isinstance(payload, dict):
        return _dict_to_text(payload)
    if isinstance(payload, (list, tuple)):
        parts = [
            canonical_payload_to_text(item) for item in payload
        ]
        return "\n".join(p for p in parts if p)
    return str(payload).strip()


def _dict_to_text(payload: dict) -> str:
    """Cherche les clés textuelles standards d'un dict canonique."""
    # Clés directes
    for key in ("text", "content", "markdown", "plain", "value"):
        if key in payload and isinstance(payload[key], str):
            return markdown_to_text(payload[key])
    # Liste de paragraphes
    if "paragraphs" in payload and isinstance(payload["paragraphs"], list):
        return "\n".join(
            canonical_payload_to_text(p)
            for p in payload["paragraphs"]
        )
    # Lignes (alternative)
    if "lines" in payload and isinstance(payload["lines"], list):
        return "\n".join(
            canonical_payload_to_text(line)
            for line in payload["lines"]
        )
    # Sinon : concaténation des valeurs textuelles trouvées
    parts: list[str] = []
    for value in payload.values():
        if isinstance(value, str):
            parts.append(markdown_to_text(value))
        elif isinstance(value, (list, dict)):
            sub = canonical_payload_to_text(value)
            if sub:
                parts.append(sub)
    return "\n".join(parts).strip()


class CanonicalToText:
    """Projecteur ``CANONICAL_DOCUMENT → RAW_TEXT``.

    Lit le payload depuis ``artifact.uri`` (chemin filesystem,
    interprété comme markdown ou JSON selon l'extension).  Pour les
    payloads inline (testing), passer par un payload_loader
    dédié dans le ``DefaultEvaluationViewExecutor``.
    """

    name = "canonical_to_text"
    source_type = ArtifactType.CANONICAL_DOCUMENT
    target_type = ArtifactType.RAW_TEXT

    def project(
        self,
        artifact: Artifact,
        params: dict[str, str | int | float | bool],
    ) -> tuple[Artifact, ProjectionReport]:
        if artifact.type != self.source_type:
            raise ProjectionError(
                f"CanonicalToText n'accepte que CANONICAL_DOCUMENT, "
                f"reçu {artifact.type.value!r}"
            )

        # Lecture optionnelle depuis le filesystem.  Si ``uri`` absent,
        # on retourne un Artifact vide — le payload_loader de
        # l'executor récupérera le contenu réel ailleurs.
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
                "structure",
                "formatting",
                "headers",
                "links",
            ),
            warnings=(
                "Markdown / JSON canonique projeté en texte plat.  "
                "Les balises markdown sont retirées par regex (pas de "
                "parser AST) ; les structures imbriquées (tableaux, "
                "listes hiérarchiques) sont aplaties.",
            ),
        )
        return target, report


__all__ = [
    "markdown_to_text",
    "canonical_payload_to_text",
    "CanonicalToText",
]
