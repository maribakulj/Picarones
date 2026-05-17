"""Résolution du répertoire d'output pour les adapters (OCR/LLM/VLM).

Helper partagé par tous les adapters qui produisent des fichiers de
sortie.  Il vit au top-level de ``adapters/`` plutôt qu'à l'intérieur
de l'un des sous-packages — il sert les trois familles indistinctement.

Un corpus monté en read-only (NAS partagé, volume Docker RO) ne peut
pas accueillir les sorties à côté des fichiers sources.  Le helper
résout le chemin selon une priorité :

1. ``context.workspace_uri`` si non None → écriture dans
   ``<workspace>/<doc_id>/`` (sandbox par run, write-allowed).
2. Fallback ``input_path.parent`` → comportement par défaut quand
   aucun workspace n'est configuré (peut échouer en read-only).

Anti-sur-ingénierie
-------------------
- Pas de quota disk : le ``WorkspaceManager`` gère ça quand un
  caller institutionnel l'exige.
- Pas de support S3/distant : ``workspace_uri`` est un path
  filesystem dans le contrat actuel.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


def _pipeline_path_segment(context: Any) -> str:
    """Segment de chemin isolant les artefacts PAR pipeline.

    Deux pipelines partageant le même OCR amont et le même modèle
    LLM (mais différant par prompt et/ou mode) produisent le MÊME
    ``(input_stem, adapter_name, suffix)``.  Sans discriminant
    pipeline, leurs artefacts s'écrasent mutuellement dans le
    workspace partagé entre sous-runs (RunOrchestrator exécute un
    sous-run séquentiel par pipeline) ; ``_extract_text_outputs``
    relit ensuite ``art.uri`` APRÈS tous les sous-runs et récupère
    le contenu du DERNIER writer pour tous les pipelines → métriques
    identiques pour des pipelines pourtant distincts.

    ``RunContext.pipeline_name`` (déjà borné à 128 ch., distinct par
    pipeline) fournit le discriminant.  On le sanitise pour en faire
    un segment de path valide, durci par un hash court du nom COMPLET
    afin qu'une éventuelle collision de sanitisation/troncature reste
    impossible.
    """
    pipeline_name = getattr(context, "pipeline_name", None) or ""
    if not pipeline_name:
        return "_nopipeline"
    safe = re.sub(r"[^\w\-]", "_", pipeline_name)[:80].strip("_") or "pl"
    digest = hashlib.sha256(pipeline_name.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{digest}"


def resolve_output_path(
    input_path: Path,
    adapter_name: str,
    suffix: str,
    context: Any,
) -> Path:
    """Résout le chemin de sortie pour un artefact d'adapter.

    Convention de nommage : ``<stem>.<adapter_name>.<suffix>``.

    Si ``context.workspace_uri`` est fourni, le fichier va dans
    ``<workspace>/<document_id>/<pipeline_segment>/`` (créé si
    absent) — l'isolation par pipeline empêche deux pipelines
    distincts partageant OCR+modèle LLM de s'écraser leurs artefacts.
    Sinon, fallback sur ``input_path.parent`` avec le segment
    pipeline intercalé dans le nom (cas CLI / corpus local).

    Parameters
    ----------
    input_path:
        Chemin du fichier d'entrée (image, texte, etc.) — utilisé
        pour récupérer le ``stem``.
    adapter_name:
        Nom de l'adapter, intercalé dans le nom du fichier pour
        permettre la cohabitation de plusieurs sorties.
    suffix:
        Extension finale, ex : ``"txt"``, ``"confidences.json"``,
        ``"corrected.txt"``.  Pas de point initial — la fonction
        l'ajoute.
    context:
        ``RunContext`` avec attributs ``document_id`` et
        ``workspace_uri``.  ``workspace_uri`` peut être ``None``
        (mode CLI direct).

    Returns
    -------
    Path
        Chemin absolu où écrire la sortie.  Le répertoire parent
        est créé si nécessaire.
    """
    workspace_uri = getattr(context, "workspace_uri", None)
    document_id = getattr(context, "document_id", None) or "unknown_doc"
    pl_segment = _pipeline_path_segment(context)

    if workspace_uri:
        out_dir = Path(workspace_uri) / document_id / pl_segment
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{input_path.stem}.{adapter_name}.{suffix}"

    return input_path.parent / (
        f"{input_path.stem}.{pl_segment}.{adapter_name}.{suffix}"
    )


__all__ = ["resolve_output_path"]
