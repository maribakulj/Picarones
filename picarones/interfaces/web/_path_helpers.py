"""Helpers de validation de chemin partagés par les routers FastAPI.

Phase 7.2 audit code-quality (2026-05) — le pattern :

.. code-block:: python

    try:
        resolved = validated_path(
            user_path,
            allowed_roots=compute_workspace_roots(UPLOADS_DIR),
            must_exist=False,
        )
    except PathValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

était dupliqué dans :

- ``routers/importers.py:45`` (``_validated_output_dir``)
- ``routers/history.py:53-60`` (inline)
- ``routers/benchmark.py:104-115`` (2 appels successifs)

Factorisé ici en deux fonctions :

- :func:`validated_user_path` — un chemin, retourne ``str``.
- :func:`validated_user_output_dir` — alias sémantique avec
  ``must_exist=False`` par défaut (le ``output_dir`` peut être créé
  ultérieurement par le worker).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from picarones.app.services.path_security import (
    PathValidationError,
    validated_path,
)
from picarones.interfaces.web.security import compute_workspace_roots
from picarones.interfaces.web.state import UPLOADS_DIR


def validated_user_path(
    user_path: str,
    *,
    must_exist: bool = False,
    must_be_dir: bool = False,
) -> Path:
    """Valide un chemin utilisateur contre les racines workspace.

    Wrapper FastAPI autour de :func:`validated_path` : convertit
    ``PathValidationError`` en ``HTTPException(400)`` pour que le
    client reçoive un 400 lisible plutôt qu'un 500 stacktrace.

    Parameters
    ----------
    user_path:
        Chaîne fournie par le client (corps JSON, query param,
        form field, etc.).
    must_exist:
        Si ``True``, refuse un chemin qui n'existe pas.  Utile pour
        les inputs ; mettre à ``False`` pour les outputs (le worker
        créera le répertoire).
    must_be_dir:
        Si ``True``, refuse un fichier (n'accepte que les dossiers).

    Returns
    -------
    Path
        Chemin canonique résolu (relatif à une racine workspace).

    Raises
    ------
    HTTPException(400)
        Si le chemin n'est pas dans une racine autorisée, contient
        ``..`` ou un symlink hors workspace, etc.
    """
    try:
        return validated_path(
            user_path,
            allowed_roots=compute_workspace_roots(UPLOADS_DIR),
            must_exist=must_exist,
            must_be_dir=must_be_dir,
        )
    except PathValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def validated_user_output_dir(user_path: str) -> str:
    """Variante sémantique pour les ``output_dir`` des importers.

    Équivalente à ``str(validated_user_path(user_path, must_exist=False))``.
    Préserve la signature historique de ``_validated_output_dir``
    qui retournait une ``str`` (les backends consommateurs n'attendent
    pas un ``Path``).
    """
    return str(validated_user_path(user_path, must_exist=False))


__all__ = ["validated_user_path", "validated_user_output_dir"]
