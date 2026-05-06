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

Anti-sur-ingénierie
-------------------
- Pas de capture des hashes de wheel : si la BnF veut une preuve
  d'intégrité supply-chain, elle utilise un lockfile Poetry/uv en
  amont — on ne refait pas le travail.
- Pas de capture des binaires système (Tesseract version, libcuda,
  fonts) : reporté à un sprint dédié si une ré-exécution échoue
  pour cette raison.  Le hash du wheel ``pytesseract`` capture déjà
  la couche Python.
"""

from __future__ import annotations

from importlib.metadata import distributions


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


__all__ = ["capture_dependencies_lock"]
