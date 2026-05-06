"""Garde-fous sur la stratégie de migration de schéma du ``JobStore``.

L'audit S58 a identifié que la table ``schema_version`` était une
coquille vide : aucun dispatcher de migrations, aucun warning si
``existing < SCHEMA_VERSION``, aucun test E2E.  Ces tests verrouillent
le contrat :

1. Si ``SCHEMA_VERSION = N``, alors ``_MIGRATIONS`` doit contenir
   les clés ``0..N-1`` (toute base v0..N-1 doit pouvoir migrer
   ascendamment vers N).
2. Une base à une version intermédiaire est migrée jusqu'à
   ``SCHEMA_VERSION``.
3. Une migration manquante est une erreur dure (pas un warning).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from picarones.adapters.storage.job_store import (
    _MIGRATIONS,
    JobStore,
    JobStoreError,
)


def test_migrations_dispatcher_covers_all_intermediate_versions() -> None:
    """``_MIGRATIONS`` couvre toutes les transitions ``v_n → v_{n+1}``
    pour ``n`` de 1 à ``SCHEMA_VERSION - 1``.

    Si ``SCHEMA_VERSION = 1``, le dispatcher peut être vide (pas
    encore de migrations).  Si ``SCHEMA_VERSION = 3``, le dispatcher
    doit avoir les clés 1 et 2.
    """
    for from_v in range(1, JobStore.SCHEMA_VERSION):
        assert from_v in _MIGRATIONS, (
            f"Migration manquante : v{from_v} → v{from_v + 1}.  "
            f"SCHEMA_VERSION = {JobStore.SCHEMA_VERSION} mais "
            f"``_MIGRATIONS[{from_v}]`` est absent."
        )


def test_fresh_db_writes_current_schema_version(tmp_path: Path) -> None:
    """Une DB neuve persiste ``SCHEMA_VERSION`` en clair."""
    JobStore(tmp_path / "fresh.sqlite")
    with sqlite3.connect(str(tmp_path / "fresh.sqlite")) as conn:
        cur = conn.execute("SELECT version FROM schema_version")
        version = cur.fetchone()[0]
    assert version == JobStore.SCHEMA_VERSION


def test_db_at_current_version_opens_idempotently(tmp_path: Path) -> None:
    """Réouvrir une DB à la même version est un no-op (pas de
    double-INSERT, pas de migration spurieuse).
    """
    db = tmp_path / "idem.sqlite"
    JobStore(db)
    JobStore(db)  # ne doit pas lever
    with sqlite3.connect(str(db)) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM schema_version")
        n = cur.fetchone()[0]
    assert n == 1, "schema_version ne doit avoir qu'une ligne."


def test_db_at_future_version_rejected(tmp_path: Path) -> None:
    """Une DB écrite par un binaire futur est rejetée (downgrade
    non supporté)."""
    db = tmp_path / "future.sqlite"
    JobStore(db)
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "UPDATE schema_version SET version = ?",
            (JobStore.SCHEMA_VERSION + 99,),
        )
        conn.commit()
    with pytest.raises(JobStoreError, match="Downgrade non supporté"):
        JobStore(db)


def test_missing_migration_is_hard_error(tmp_path: Path) -> None:
    """Si ``existing < SCHEMA_VERSION`` mais qu'aucune migration n'est
    enregistrée pour la version intermédiaire, ``JobStoreError``.

    Ce test simule SCHEMA_VERSION = 99 sans entrée dans _MIGRATIONS
    en patchant directement.  Garantie : on ne laisse jamais une base
    dans un état schématiquement incohérent silencieusement.
    """
    db = tmp_path / "stale.sqlite"
    JobStore(db)  # crée v1
    # Patch in-test : prétendons que le code attend v99.
    original = JobStore.SCHEMA_VERSION
    JobStore.SCHEMA_VERSION = 99
    try:
        with pytest.raises(JobStoreError, match="migration manquante"):
            JobStore(db)
    finally:
        JobStore.SCHEMA_VERSION = original


def test_migration_chain_applied(tmp_path: Path) -> None:
    """Si SCHEMA_VERSION saute de N versions, toutes les migrations
    intermédiaires sont appliquées dans l'ordre.

    Simule une migration v1 → v2 fictive enregistrée temporairement.
    """
    db = tmp_path / "chain.sqlite"
    JobStore(db)  # v1

    applied: list[int] = []

    def fake_v1_to_v2(conn: sqlite3.Connection) -> None:
        applied.append(1)

    original_version = JobStore.SCHEMA_VERSION
    JobStore.SCHEMA_VERSION = 2
    _MIGRATIONS[1] = fake_v1_to_v2
    try:
        JobStore(db)  # déclenche v1 → v2
        assert applied == [1]
        with sqlite3.connect(str(db)) as conn:
            cur = conn.execute("SELECT version FROM schema_version")
            assert cur.fetchone()[0] == 2
    finally:
        JobStore.SCHEMA_VERSION = original_version
        _MIGRATIONS.pop(1, None)
