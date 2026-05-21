"""Garde-fou : tout lien interne dans ``docs/index.md`` doit pointer
vers un fichier réel.

Pourquoi ce test existe
-----------------------

``docs/index.md`` est l'**index canonique** de la documentation : il
est référencé depuis le README, depuis mkdocs.yml, et c'est la
première porte d'entrée pour un nouveau contributeur.

Avant Phase 1, ce fichier contenait 4 liens cassés (``first-benchmark``,
``writing-a-pipeline-module``, ``developer/narrative-engine``,
``user/...``) qui ont survécu pendant le rewrite parce qu'aucun test
ne validait ses propres liens.  Ce garde-fou élimine la classe
d'erreur : si l'index ment, la CI échoue.

Périmètre
---------

On parse les liens markdown ``[texte](cible)`` et on vérifie que la
``cible`` :

- soit pointe vers un fichier existant (résolution relative à
  ``docs/`` ou à la racine pour les ``../X``) ;
- soit est une URL externe (``http://...``, ``mailto:...``) — non
  vérifiée ici, c'est le rôle de tests externes ;
- soit est une ancre intra-document (``#section``) — non vérifiée.

Les liens vers des dossiers (``case-studies/``, ``audits/``) sont
vérifiés comme l'existence du dossier.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX = REPO_ROOT / "docs" / "index.md"

#: Pattern markdown standard : ``[texte](cible)``.  On capture la
#: cible (groupe 2) qu'on évaluera comme chemin.
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _resolve_link(target: str) -> Path | None:
    """Résout une cible de lien relativement à ``docs/index.md``.

    Retourne ``None`` si :
    - URL externe (``http``, ``mailto``, ``#``) ;
    - cible vide ;
    - chemin qui ne se résout pas.
    """
    target = target.strip()

    # URL externe — pas notre problème ici.
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None

    # Retirer l'ancre éventuelle (``foo.md#section``)
    target = target.split("#", 1)[0]
    if not target:
        return None

    # Les liens dans index.md sont relatifs à ``docs/``.
    # Les liens vers la racine (``../GOVERNANCE.md``) doivent
    # remonter au repo root.
    base = INDEX.parent
    resolved = (base / target).resolve()
    return resolved


def test_index_md_exists() -> None:
    assert INDEX.exists(), (
        f"{INDEX} absent — c'est l'index canonique de la doc, il "
        "ne peut pas manquer."
    )


def test_all_internal_links_in_index_resolve() -> None:
    """Tout lien interne dans ``docs/index.md`` doit pointer vers
    un fichier ou dossier existant."""
    text = INDEX.read_text(encoding="utf-8")
    offenders: list[str] = []
    for match in _LINK_RE.finditer(text):
        target = match.group(2)
        resolved = _resolve_link(target)
        if resolved is None:
            continue  # URL externe / ancre — pas notre périmètre
        if not resolved.exists():
            offenders.append(
                f"  « {match.group(1)} » → {target!r} "
                f"(résolu vers {resolved.relative_to(REPO_ROOT) if resolved.is_relative_to(REPO_ROOT) else resolved})"
            )

    assert not offenders, (
        f"{len(offenders)} lien(s) cassé(s) dans docs/index.md :\n"
        + "\n".join(offenders)
        + "\n\n→ Soit créer le fichier cible, soit corriger le lien."
    )
