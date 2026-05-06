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

from pathlib import Path
from typing import Any


def resolve_output_path(
    input_path: Path,
    adapter_name: str,
    suffix: str,
    context: Any,
) -> Path:
    """Résout le chemin de sortie pour un artefact d'adapter.

    Convention de nommage : ``<stem>.<adapter_name>.<suffix>``.

    Si ``context.workspace_uri`` est fourni, le fichier va dans
    ``<workspace>/<document_id>/`` (créé si absent).  Sinon, fallback
    sur ``input_path.parent`` (cas typique CLI / corpus local).

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

    if workspace_uri:
        out_dir = Path(workspace_uri) / document_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{input_path.stem}.{adapter_name}.{suffix}"

    return input_path.parent / f"{input_path.stem}.{adapter_name}.{suffix}"


__all__ = ["resolve_output_path"]
