"""Loader de payload filesystem + util signature kwargs.

Audit prod P1.1 — sous-package cohésif (ex-module plat).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.formats.alto.parser import parse_alto


def _kwargs_signature(kwargs: dict[str, Any]) -> str:
    """Signature stable d'un dict de kwargs (ordre tri-stable)."""
    return "|".join(f"{k}={kwargs[k]!r}" for k in sorted(kwargs))


def _filesystem_payload_loader(art: Artifact) -> Any:
    """Loader filesystem : lit RAW_TEXT/CORRECTED_TEXT depuis le
    fichier pointé par l'URI, parse ALTO_XML depuis le fichier pointé.

    Les artefacts projetés (sans URI) ne passent pas par ce loader —
    l'executor utilise directement le payload retourné par le
    projecteur.
    """
    if art.uri is None:
        raise FileNotFoundError(
            f"Loader filesystem : artifact {art.id!r} sans URI ; "
            "un projecteur aurait dû fournir le payload.",
        )
    path = Path(art.uri)
    if art.type == ArtifactType.ALTO_XML:
        return parse_alto(path.read_bytes())
    if art.type in (ArtifactType.RAW_TEXT, ArtifactType.CORRECTED_TEXT):
        return path.read_text(encoding="utf-8")
    raise ValueError(
        f"Loader filesystem : type {art.type.value!r} non géré.",
    )


__all__ = ["_filesystem_payload_loader", "_kwargs_signature"]
