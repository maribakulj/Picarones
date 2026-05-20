"""Utilitaires de validation et migration des configs utilisateur.

supprime la friction *« reconfigurer chaque session »* :
le client peut télécharger sa config en JSON et la réimporter plus
tard. Ce module définit le schéma versionné et les règles de filtrage
qui empêchent qu'un payload trop riche n'embarque des secrets ou des
clés serveur.
"""

from __future__ import annotations

from typing import Any

CONFIG_SCHEMA_VERSION = 1
"""Bump quand le format change ; ajouter un upgrade path dans ``upgrade_config``."""

ALLOWED_CONFIG_FIELDS: frozenset[str] = frozenset({
    "schema_version",
    "saved_at",
    "label",
    "corpus_path",
    "engines",
    "normalization_profile",
    "char_exclude",
    "lang",
    "report_lang",
    "output_dir",
    "report_name",
    "competitors",
})
"""Liste blanche des champs autorisés dans une config sauvegardée."""


def filter_config(payload: dict) -> dict:
    """Ne garde que les champs autorisés, dans un ordre stable pour les diffs."""
    out: dict[str, Any] = {}
    for k in sorted(ALLOWED_CONFIG_FIELDS):
        if k in payload:
            out[k] = payload[k]
    return out


def upgrade_config(payload: dict) -> dict:
    """Migre les anciennes configs vers le schéma courant.

    Schéma 1 (Sprint 28) : pas de migration nécessaire — on retourne tel quel.
    Future : ajouter des branches sur ``schema_version`` quand le format évolue.
    """
    return payload


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ALLOWED_CONFIG_FIELDS",
    "filter_config",
    "upgrade_config",
]
