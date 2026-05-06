"""Garde-fou : les clés du store d'artefacts sont filesystem-safe.

L'audit S58/S59 a relevé un crash Windows reproductible :
``OSError: [WinError 87] The parameter is incorrect`` sur
``os.replace(tmp, dst)`` quand ``dst`` contient un ``:``.

Cause : ``:`` est un caractère réservé du filesystem NTFS (Alternate
Data Streams) — un filename comme ``abc:raw_text.json`` est rejeté.
Le bug existait depuis S47 mais n'avait jamais été détecté en CI
parce que les builds Windows passaient en silence (l'écriture
non-atomique ``write_text`` directe ne nettoyait pas le tmp donc
laissait un fichier orphelin sans erreur ; après S59 #9 atomique,
le bug est devenu visible).

Ce test verrouille que tout caractère réservé Windows est rejeté.
"""

from __future__ import annotations

from picarones.domain.artifacts import ArtifactType
from picarones.pipeline.cache_helpers import (
    _KEY_SEPARATOR,
    storage_key_for_output,
)

#: Caractères que NTFS / Windows refusent dans un nom de fichier.
#: Source : https://learn.microsoft.com/windows/win32/fileio/naming-a-file
_WINDOWS_FORBIDDEN = frozenset(r'<>:"/\|?*')


def test_storage_key_separator_filesystem_safe() -> None:
    """Le séparateur de clé composite ne contient aucun caractère
    interdit sur Windows.
    """
    assert not (set(_KEY_SEPARATOR) & _WINDOWS_FORBIDDEN), (
        f"_KEY_SEPARATOR={_KEY_SEPARATOR!r} contient un caractère "
        f"réservé Windows.  Voir _WINDOWS_FORBIDDEN={_WINDOWS_FORBIDDEN!r}."
    )


def test_storage_keys_for_all_artifact_types_filesystem_safe() -> None:
    """Pour chaque ``ArtifactType``, la clé composite produite par
    ``storage_key_for_output`` est filesystem-safe.

    Couvre l'intégralité de l'enum — un nouveau type de la forme
    ``my:type`` (avec ``:`` dans la value) ferait échouer ce test
    et exigerait soit la révision du nom du type soit l'introduction
    d'un encoding dans le store.
    """
    fake_hash = "0" * 64  # SHA-256 hex stub
    for at in ArtifactType:
        key = storage_key_for_output(fake_hash, at)
        offending = set(key) & _WINDOWS_FORBIDDEN
        assert not offending, (
            f"storage_key_for_output(hash, {at!r}) = {key!r} contient "
            f"des caractères interdits sur Windows : {sorted(offending)!r}."
        )
