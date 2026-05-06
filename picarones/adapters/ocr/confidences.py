"""Sidecar de confidences OCR.

Les confidences au niveau token sont exposées comme un **artefact
dédié** ``ArtifactType.CONFIDENCES`` (sidecar JSON à côté du fichier
texte), pas stuffé dans le résultat texte de l'adapter.  Ce
découplage permet aux vues de calibration (ECE/MCE, reliability
diagram) de consommer les confidences indépendamment de la
production du texte, et n'oblige pas un adapter qui n'a pas de
confidences à porter un champ vide.

Format JSON canonique
---------------------

::

    {
      "tokens": [
        {"text": "Bonjour", "confidence": 0.95},
        {"text": "le",      "confidence": 0.99},
        ...
      ],
      "extractor": "tesseract",
      "model_version": "5.3.0"  // optionnel
    }

- ``confidence`` ∈ [0, 1] (les adapters convertissent eux-mêmes
  depuis leur format natif — Tesseract retourne 0-100, on divise
  par 100).
- Tokens vides ou conf négatives ignorés à la source (cf.
  ``filter_valid_tokens``).

API publique
------------
- ``filter_valid_tokens(raw)`` : nettoie une liste de dicts brutes.
- ``write_confidences_sidecar(text_path, name, tokens, ...)`` :
  écrit ``<stem>.<name>.confidences.json`` à côté du fichier texte.
- ``ConfidenceToken`` (TypedDict léger) : forme attendue du dict.

Anti-sur-ingénierie
-------------------
- Pas de pydantic — TypedDict + json suffisent ; le caller normalise.
- Pas de schéma JSON publié — la stabilité sera tagguée à la livraison.
- Pas de support pour les confidences niveau ligne / paragraphe :
  on aplatit tout au niveau mot (cohérent avec le legacy Sprint 47).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from picarones.domain.artifacts import Artifact, ArtifactType


class ConfidenceToken(TypedDict):
    """Forme canonique d'un token de confidence."""

    text: str
    confidence: float


def filter_valid_tokens(
    raw: list[dict[str, Any]],
) -> list[ConfidenceToken]:
    """Nettoie une liste brute de tokens (ignore les non-mots).

    Filtre :

    - ``text`` vide ou whitespace-only ;
    - ``confidence`` ``None`` ou négative (Tesseract met -1 pour les
      non-mots) ;
    - ``confidence`` > 1.0 → divisé par 100 si ≤ 100, sinon ignoré.

    Retourne une nouvelle liste, ne modifie pas l'input.
    """
    out: list[ConfidenceToken] = []
    for entry in raw:
        text = str(entry.get("text", "") or "").strip()
        if not text:
            continue
        conf = entry.get("confidence")
        if conf is None:
            continue
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_f < 0:
            continue
        if conf_f > 1.0:
            # Tesseract retourne 0-100 ; on normalise.
            if conf_f <= 100.0:
                conf_f = conf_f / 100.0
            else:
                # > 100 = donnée corrompue, on ignore.
                continue
        out.append({"text": text, "confidence": conf_f})
    return out


def write_confidences_sidecar(
    text_path: Path,
    adapter_name: str,
    tokens: list[ConfidenceToken],
    *,
    document_id: str,
    extractor: str | None = None,
    model_version: str | None = None,
) -> Artifact:
    """Écrit un sidecar JSON ``<stem>.<adapter_name>.confidences.json``
    à côté du fichier texte produit par l'OCR.

    Returns
    -------
    Artifact
        Artifact ``CONFIDENCES`` avec ``uri`` pointant vers le sidecar.
    """
    sidecar_path = (
        text_path.parent
        / f"{text_path.stem}.{adapter_name}.confidences.json"
    )
    payload = {
        "tokens": tokens,
        "extractor": extractor or adapter_name,
        "model_version": model_version,
    }
    sidecar_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return Artifact(
        id=f"{document_id}:{adapter_name}:confidences",
        document_id=document_id,
        type=ArtifactType.CONFIDENCES,
        produced_by_step="ocr",
        uri=str(sidecar_path),
    )


__all__ = [
    "ConfidenceToken",
    "filter_valid_tokens",
    "write_confidences_sidecar",
]
