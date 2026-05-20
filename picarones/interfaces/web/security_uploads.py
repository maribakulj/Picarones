"""Validation des fichiers/images uploadés (extrait de ``security.py``).

dégonflage du god-module ``security``.  Cluster
*stateless* : plafonds d'upload + validation Pillow (buffer ou
fichier).  Réimporté par ``security`` pour préserver l'API
(``from picarones.interfaces.web.security import validate_image_safe``
reste valide).
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_max_upload_mb() -> int:
    raw = os.environ.get("PICARONES_MAX_UPLOAD_MB", "100")
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "[security] PICARONES_MAX_UPLOAD_MB invalide (%r) — défaut 100 Mo.", raw
        )
        return 100


#: Taille de bloc pour le streaming multipart → disque.  Le fichier
#: n'est jamais matérialisé en un seul ``bytes`` en RAM : on lit par
#: blocs bornés et on écrit au fil de l'eau.
UPLOAD_CHUNK_SIZE = 1024 * 1024


def get_max_total_upload_mb() -> int:
    """Plafond dur cumulé d'une requête d'upload (tous fichiers).

    Sans plafond total, N fichiers chacun sous la limite unitaire
    saturent quand même le disque/la RAM.  Défaut 500 Mo — aligné sur
    ``MAX_ZIP_TOTAL_SIZE`` (taille décompressée max d'un corpus ZIP).
    """
    raw = os.environ.get("PICARONES_MAX_TOTAL_UPLOAD_MB", "500")
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "[security] PICARONES_MAX_TOTAL_UPLOAD_MB invalide (%r) — "
            "défaut 500 Mo.", raw,
        )
        return 500


def _verify_image_with_pillow(source: Any, filename: str) -> None:
    """Cœur partagé : ``Image.open(source).verify()`` + mapping erreurs.

    ``source`` peut être un ``io.BytesIO`` (upload bufferisé) OU un
    ``Path`` (Pillow lit alors le fichier en flux, sans charger les
    octets en RAM — c'est tout l'intérêt de la variante fichier).
    Factorisé pour ne pas dupliquer le mapping d'exceptions entre
    :func:`validate_image_safe` et :func:`validate_image_file_safe`.
    """
    try:
        import importlib
        Image = importlib.import_module("PIL.Image")
        UnidentifiedImageError = importlib.import_module("PIL").UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover — Pillow est core
        logger.warning("[security] Pillow indisponible — validation image sautée : %s", exc)
        return

    try:
        with Image.open(source) as im:
            im.verify()
    except UnidentifiedImageError as exc:
        raise ValueError(
            f"Image '{filename}' refusée : format non reconnu par Pillow ({exc})."
        ) from exc
    except Image.DecompressionBombError as exc:
        raise ValueError(
            f"Image '{filename}' refusée : bombe de décompression détectée ({exc})."
        ) from exc
    except Exception as exc:
        # Pillow lève un panel d'exceptions hétérogènes (SyntaxError sur les
        # GIF malformés, OSError sur les TIFF corrompus, ValueError divers).
        raise ValueError(
            f"Image '{filename}' refusée : erreur de décodage Pillow ({type(exc).__name__}: {exc})."
        ) from exc


def validate_image_safe(data: bytes, filename: str = "<upload>") -> None:
    """Vérifie qu'un buffer décode comme une image valide sans bombe.

    Levée de ``ValueError`` (à mapper en HTTP 415/422) si :
      - taille > limite ;
      - Pillow rejette l'image (UnidentifiedImageError, DecompressionBombError) ;
      - le format ouvert ne correspond pas à ce que prétend l'extension.

    On ne bloque pas l'absence de Pillow (il est dépendance core), mais on
    log si l'import échoue pour aider au diagnostic.
    """
    max_mb = get_max_upload_mb()
    if len(data) > max_mb * 1024 * 1024:
        raise ValueError(
            f"Image '{filename}' refusée : taille {len(data) / (1024 * 1024):.1f} Mo > "
            f"limite {max_mb} Mo (PICARONES_MAX_UPLOAD_MB)."
        )
    _verify_image_with_pillow(io.BytesIO(data), filename)


def validate_image_file_safe(path: Path, filename: str | None = None) -> None:
    """Variante *fichier* de :func:`validate_image_safe` (audit P0.5).

    Ne lit PAS le fichier entier en RAM : la taille est lue via
    ``stat()`` et Pillow décode en flux depuis le chemin.  À utiliser
    pour les fichiers déjà sur disque (image extraite d'un ZIP, upload
    déjà streamé) au lieu de ``validate_image_safe(path.read_bytes())``.
    """
    name = filename or path.name
    max_mb = get_max_upload_mb()
    size = path.stat().st_size
    if size > max_mb * 1024 * 1024:
        raise ValueError(
            f"Image '{name}' refusée : taille {size / (1024 * 1024):.1f} Mo > "
            f"limite {max_mb} Mo (PICARONES_MAX_UPLOAD_MB)."
        )
    _verify_image_with_pillow(path, name)


__all__ = [
    "UPLOAD_CHUNK_SIZE",
    "get_max_total_upload_mb",
    "get_max_upload_mb",
    "validate_image_file_safe",
    "validate_image_safe",
]
