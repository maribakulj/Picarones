"""Capture du verrou des dépendances au moment d'un run.

Le ``RunManifest`` documente la promesse *« à code_version + corpus +
specs + dependencies_lock identiques, ré-exécuter doit donner les
mêmes résultats »*.  Ce module fournit la capture canonique du
``dependencies_lock``.

Approche
--------
``importlib.metadata.distributions()`` retourne tous les paquets
installés dans l'environnement Python courant — c'est l'API standard
Python (PEP 566) plutôt que d'invoquer ``pip freeze`` en sous-process.
Chaque ``Distribution`` fournit ``name`` + ``version`` ; on en fait
un dict ordonné par ``name`` minuscule pour le déterminisme du
manifest.

Sprint S8.5 — capture des binaires système
------------------------------------------
La couche Python (``pytesseract``) ne suffit pas pour reproduire un
benchmark scientifique : c'est le binaire Tesseract sous-jacent qui
exécute l'OCR.  ``capture_system_binaries_lock()`` ajoute la version
du binaire Tesseract (et autres binaires critiques) au manifest, en
best-effort (silencieux si Tesseract n'est pas installé).

Anti-sur-ingénierie
-------------------
- Pas de capture des hashes de wheel : si la BnF veut une preuve
  d'intégrité supply-chain, elle utilise un lockfile Poetry/uv en
  amont — on ne refait pas le travail.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from importlib.metadata import distributions

logger = logging.getLogger(__name__)


def capture_dependencies_lock() -> dict[str, str]:
    """Retourne un dict ``{nom_package: version}`` trié par nom.

    Tri lexicographique sur ``name.lower()`` pour produire des
    manifests bit-for-bit identiques à environnement constant
    (l'ordre d'itération de ``distributions()`` n'est pas spécifié).
    """
    lock: dict[str, str] = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        version = dist.version
        if name and version:
            lock[name] = version
    return dict(sorted(lock.items(), key=lambda kv: kv[0].lower()))


def _safe_capture_binary_version(
    binary: str, args: tuple[str, ...] = ("--version",),
) -> str | None:
    """Capture la première ligne de sortie de ``<binary> <args>``.

    Retourne ``None`` si le binaire n'est pas dans ``$PATH`` ou si
    l'invocation échoue.  Ne lève jamais.
    """
    if not shutil.which(binary):
        return None
    try:
        result = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(
            "[deps] capture %s %s échouée : %s", binary, " ".join(args), exc,
        )
        return None
    output = (result.stdout or result.stderr).strip()
    if not output:
        return None
    return output.splitlines()[0]


def capture_system_binaries_lock() -> dict[str, str]:
    """Retourne un dict ``{binary_name: version_string}`` pour les
    binaires système critiques à la reproductibilité scientifique.

    Sprint S8.5 — closes the gap left by the pure-Python
    ``capture_dependencies_lock`` : la version du wheel
    ``pytesseract`` ne dit RIEN sur la version du binaire Tesseract
    qui exécute réellement l'OCR.  Sans capturer cette version,
    deux runs avec le même ``dependencies_lock`` peuvent produire
    des CER différents si la base image système a été mise à jour
    entre temps.

    Best-effort : si un binaire n'est pas installé, sa clé est
    absente du dict (pas ``None``, pas d'exception).
    """
    binaries: dict[str, tuple[str, ...]] = {
        # Tesseract OCR — version + langues installées.
        "tesseract": ("--version",),
    }
    lock: dict[str, str] = {}
    for binary, args in binaries.items():
        version = _safe_capture_binary_version(binary, args)
        if version:
            lock[binary] = version
    return lock


__all__ = ["capture_dependencies_lock", "capture_system_binaries_lock"]
