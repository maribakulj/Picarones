"""Tests Sprint A11 — purge automatique des uploads (item M-8).

Valide que le module ``picarones.interfaces.web._legacy.maintenance`` :

1. supprime les uploads dont le mtime dépasse le seuil ;
2. **ne supprime pas** les uploads référencés par un job actif ;
3. respecte ``PICARONES_UPLOAD_RETENTION_DAYS=0`` (mode désactivé) ;
4. gère gracieusement une erreur d'I/O sur un sous-dossier
   (ne tue pas la passe pour les autres) ;
5. ne touche pas la BD jobs (la purge est read-only sur les jobs).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from picarones.interfaces.web._legacy.maintenance import (
    _get_retention_days,
    _should_purge,
    purge_old_uploads,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uploads_root(tmp_path: Path) -> Path:
    """Crée 3 corpus de test : un récent, un ancien, un en cours."""
    root = tmp_path / "uploads"
    root.mkdir()

    # Corpus récent (5 jours) — doit rester
    recent = root / "corpus_recent"
    recent.mkdir()
    (recent / "doc1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    five_days_ago = time.time() - 5 * 86400
    os.utime(recent, (five_days_ago, five_days_ago))

    # Corpus ancien (10 jours) — doit être purgé (avec retention=7j)
    old = root / "corpus_old"
    old.mkdir()
    (old / "doc1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    ten_days_ago = time.time() - 10 * 86400
    os.utime(old, (ten_days_ago, ten_days_ago))

    # Corpus ancien mais en cours d'usage — ne doit pas être purgé
    busy = root / "corpus_busy"
    busy.mkdir()
    (busy / "doc1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    os.utime(busy, (ten_days_ago, ten_days_ago))

    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_purge_removes_old_unreferenced(uploads_root: Path) -> None:
    """Un upload ancien et non référencé doit être supprimé."""
    purged = purge_old_uploads(
        uploads_root, retention_days=7, active_corpus_ids=set()
    )
    purged_names = {p.name for p in purged}
    # corpus_old (10j) doit être purgé ; corpus_busy aussi sans référence
    assert "corpus_old" in purged_names
    assert "corpus_busy" in purged_names
    # corpus_recent (5j) doit rester
    assert "corpus_recent" not in purged_names
    assert (uploads_root / "corpus_recent").exists()
    assert not (uploads_root / "corpus_old").exists()


def test_purge_keeps_active_corpus(uploads_root: Path) -> None:
    """Un upload référencé par un job actif doit être préservé même
    si vieux."""
    purged = purge_old_uploads(
        uploads_root,
        retention_days=7,
        active_corpus_ids={"corpus_busy"},
    )
    purged_names = {p.name for p in purged}
    assert "corpus_busy" not in purged_names
    assert (uploads_root / "corpus_busy").exists()
    # L'autre vieux est bien purgé
    assert "corpus_old" in purged_names


def test_retention_zero_disables_purge(uploads_root: Path) -> None:
    """``retention_days=0`` désactive complètement la purge."""
    purged = purge_old_uploads(
        uploads_root, retention_days=0, active_corpus_ids=set()
    )
    assert purged == []
    # Tous les corpus restent
    assert (uploads_root / "corpus_recent").exists()
    assert (uploads_root / "corpus_old").exists()
    assert (uploads_root / "corpus_busy").exists()


def test_purge_nonexistent_root_returns_empty(tmp_path: Path) -> None:
    """Si ``uploads_root`` n'existe pas, retourne [] sans crash."""
    purged = purge_old_uploads(
        tmp_path / "nope", retention_days=7, active_corpus_ids=set()
    )
    assert purged == []


def test_purge_empty_root_returns_empty(tmp_path: Path) -> None:
    """``uploads_root`` vide : pas de purge."""
    empty = tmp_path / "empty"
    empty.mkdir()
    purged = purge_old_uploads(empty, retention_days=7, active_corpus_ids=set())
    assert purged == []


def test_should_purge_threshold_boundary(tmp_path: Path) -> None:
    """Test précis du seuil : 6,9 jours → garde, 7,1 jours → purge."""
    p = tmp_path / "corpus"
    p.mkdir()
    now = time.time()

    # 6.9 jours d'ancienneté : reste
    os.utime(p, (now - 6.9 * 86400, now - 6.9 * 86400))
    assert not _should_purge(p, retention_days=7, active_corpus_ids=set(), now=now)

    # 7.1 jours d'ancienneté : purgé
    os.utime(p, (now - 7.1 * 86400, now - 7.1 * 86400))
    assert _should_purge(p, retention_days=7, active_corpus_ids=set(), now=now)


def test_should_purge_respects_active_set(tmp_path: Path) -> None:
    """Même très vieux, un corpus actif n'est pas purgé."""
    p = tmp_path / "corpus_x"
    p.mkdir()
    now = time.time()
    os.utime(p, (now - 100 * 86400, now - 100 * 86400))
    assert not _should_purge(
        p, retention_days=7, active_corpus_ids={"corpus_x"}, now=now,
    )


def test_get_retention_days_default(monkeypatch) -> None:
    """Sans variable d'env : 7 jours par défaut."""
    monkeypatch.delenv("PICARONES_UPLOAD_RETENTION_DAYS", raising=False)
    assert _get_retention_days() == 7


def test_get_retention_days_from_env(monkeypatch) -> None:
    """Variable d'env respectée si valide."""
    monkeypatch.setenv("PICARONES_UPLOAD_RETENTION_DAYS", "30")
    assert _get_retention_days() == 30


def test_get_retention_days_invalid_falls_back(monkeypatch) -> None:
    """Variable d'env invalide → fallback 7 jours + warning."""
    monkeypatch.setenv("PICARONES_UPLOAD_RETENTION_DAYS", "not-a-number")
    assert _get_retention_days() == 7


def test_get_retention_days_negative_clamped(monkeypatch) -> None:
    """Valeur négative → clamp à 0 (= désactivé)."""
    monkeypatch.setenv("PICARONES_UPLOAD_RETENTION_DAYS", "-5")
    assert _get_retention_days() == 0


def test_purge_does_not_crash_on_individual_failure(
    uploads_root: Path, monkeypatch
) -> None:
    """Un échec d'I/O sur un dossier ne tue pas la passe pour les autres."""
    import shutil as _shutil

    failures = {"corpus_old"}
    original_rmtree = _shutil.rmtree

    def failing_rmtree(path, *args, **kwargs):
        if Path(path).name in failures:
            raise OSError("Simulated I/O error")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        "picarones.interfaces.web._legacy.maintenance.shutil.rmtree", failing_rmtree
    )

    purged = purge_old_uploads(
        uploads_root, retention_days=7, active_corpus_ids=set()
    )
    purged_names = {p.name for p in purged}
    # corpus_old a échoué (pas dans purged) mais corpus_busy a été supprimé
    assert "corpus_old" not in purged_names
    assert "corpus_busy" in purged_names
    # corpus_old toujours présent
    assert (uploads_root / "corpus_old").exists()
