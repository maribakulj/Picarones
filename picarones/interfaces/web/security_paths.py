"""Racines autorisées + ré-export validation de chemins.

Audit prod P1.2 — dégonflage du god-module ``security``.  Regroupe
``compute_browse_roots`` / ``compute_workspace_roots`` et ré-exporte
les helpers de ``app.services.path_security`` (foyer canonique,
partagé CLI + web + jobs).  Réimporté par ``security`` (API
publique préservée).
"""

from __future__ import annotations

import os
from pathlib import Path

from picarones.interfaces.web.security_public_mode import is_public_mode

# Ré-export depuis le foyer définitif ``app.services.path_security``
# (Sprint A14-S19) — pas de duplication, le code vit en un seul endroit
# dans la couche app (accessible CLI + jobs background).
from picarones.app.services.path_security import (
    PathValidationError as PathValidationError,
    safe_report_name as safe_report_name,
    validated_path as validated_path,
    validated_prompt_filename as validated_prompt_filename,
)
from picarones.app.services.path_security import (
    _is_within as _is_within,  # noqa: F401
)


def compute_browse_roots(uploads_dir: Path) -> list[Path]:
    """Retourne la liste de répertoires autorisés pour ``/api/corpus/browse``.

    - Variable d'env ``PICARONES_BROWSE_ROOTS`` (séparateur ``os.pathsep``,
      ``:`` sur Linux/macOS, ``;`` sur Windows) : prioritaire si définie.
    - Sinon, mode public ⇒ uniquement ``uploads_dir``.
    - Sinon, mode dev (défaut) ⇒ cwd + uploads_dir + ``/workspaces``
      (Codespaces) + ``tempdir`` (compatibilité ascendante).
    """
    raw = os.environ.get("PICARONES_BROWSE_ROOTS")
    if raw:
        roots = [Path(p).resolve() for p in raw.split(os.pathsep) if p.strip()]
        return roots

    if is_public_mode():
        return [uploads_dir.resolve()]

    import tempfile
    return [
        Path(".").resolve(),
        uploads_dir.resolve(),
        Path("/workspaces").resolve(),
        Path(tempfile.gettempdir()).resolve(),
    ]


def compute_workspace_roots(uploads_dir: Path) -> list[Path]:
    """Retourne les racines autorisées pour les opérations de benchmark.

    Sprint A14-S1 — A.I.0 P0 : utilisé par les endpoints
    ``/api/benchmark/start`` et ``/api/benchmark/run`` pour valider
    ``corpus_path`` et ``output_dir`` via :func:`validated_path`.

    Sémantique :

    - Si ``PICARONES_WORKSPACE_ROOTS`` est défini, prend précédence
      absolue (admin sait ce qu'il fait).
    - Sinon, en mode public : uniquement ``uploads_dir`` (lecture)
      et ``./rapports`` (écriture des rapports générés).
    - Sinon, mode dev : ``compute_browse_roots`` + ``./rapports`` +
      ``./corpus`` (corpus locaux des développeurs).

    En production institutionnelle, exporter ``PICARONES_WORKSPACE_ROOTS``
    pour épingler explicitement les répertoires autorisés.
    """
    raw = os.environ.get("PICARONES_WORKSPACE_ROOTS")
    if raw:
        return [Path(p).expanduser().resolve() for p in raw.split(os.pathsep) if p.strip()]

    base = compute_browse_roots(uploads_dir)
    extras = [
        Path("./rapports").resolve(),
        Path("./corpus").resolve(),
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for p in base + extras:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


__all__ = [
    "PathValidationError",
    "compute_browse_roots",
    "compute_workspace_roots",
    "safe_report_name",
    "validated_path",
    "validated_prompt_filename",
]
