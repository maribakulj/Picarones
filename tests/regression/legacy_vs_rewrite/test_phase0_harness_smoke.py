"""Smoke tests du harness lui-même.

Phase 0 : avant que la moindre comparaison legacy ↔ rewrite ne soit
faite, il faut prouver que le harness :

1. Génère des corpus de référence reproductibles cross-OS.
2. Sait écrire et relire un golden snapshot.
3. Ses comparateurs sémantiques rejettent les vraies différences et
   acceptent les non-significatives.

Ces tests sont marqués ``regression`` mais ne font pas de
comparaison legacy ↔ rewrite — ils valident l'infrastructure
elle-même.

Aux phases suivantes, des fichiers ``test_phaseN_<module>.py``
viendront s'ajouter à côté de celui-ci pour vérifier chaque
fonctionnalité migrée.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.regression.legacy_vs_rewrite.conftest import (
    assert_floats_equal,
    assert_golden_match,
    assert_json_semantic_equal,
    assert_set_equal,
)


pytestmark = pytest.mark.regression


# ──────────────────────────────────────────────────────────────────
# Corpus
# ──────────────────────────────────────────────────────────────────


def test_small_corpus_has_three_documents(small_corpus_dir: Path) -> None:
    """``small_corpus_dir`` produit 3 paires (image + GT)."""
    pngs = sorted(small_corpus_dir.glob("*.png"))
    gts = sorted(small_corpus_dir.glob("*.gt.txt"))
    assert len(pngs) == 3, f"3 PNG attendus, {len(pngs)} trouvés."
    assert len(gts) == 3, f"3 GT attendues, {len(gts)} trouvées."
    for png in pngs:
        gt = png.with_suffix("").with_suffix(".gt.txt")
        assert gt.exists(), f"GT manquante pour {png.name}."


def test_medium_corpus_has_thirty_documents(medium_corpus_dir: Path) -> None:
    """``medium_corpus_dir`` produit 30 paires."""
    pngs = sorted(medium_corpus_dir.glob("*.png"))
    assert len(pngs) == 30


def test_corpus_generation_is_idempotent(small_corpus_dir: Path) -> None:
    """Re-générer le corpus ne réécrit pas les fichiers existants."""
    pngs_before = {p: p.stat().st_mtime for p in small_corpus_dir.glob("*.png")}
    # Re-déclencher la génération en réimportant la fixture (ici on
    # appelle directement la primitive — le test n'est pas sale, c'est
    # le contrat d'idempotence qui est vérifié).
    from tests.regression.legacy_vs_rewrite.conftest import (
        _generate_synthetic_corpus,
    )
    _generate_synthetic_corpus(
        small_corpus_dir,
        documents=[
            ("doc01", "BENEDICTUS DEUS"),
            ("doc02", "Anno Domini MCMXVII"),
            ("doc03", "Folio 23 recto"),
        ],
    )
    pngs_after = {p: p.stat().st_mtime for p in small_corpus_dir.glob("*.png")}
    for path, mtime_before in pngs_before.items():
        assert pngs_after[path] == mtime_before, (
            f"{path.name} a été ré-écrit alors qu'il existait déjà."
        )


# ──────────────────────────────────────────────────────────────────
# Golden snapshots
# ──────────────────────────────────────────────────────────────────


def test_golden_path_creates_directories(golden_path, tmp_path) -> None:
    """``golden_path('phase', 'corpus', 'file')`` crée le dossier."""
    p = golden_path("phase0", "smoke", "tmp.txt")
    assert p.parent.exists()
    # Cleanup pour ne pas polluer.
    if p.exists():
        p.unlink()


def test_golden_match_writes_on_first_run(
    tmp_path: Path,
    regen_golden: bool,
) -> None:
    """Quand le fichier golden n'existe pas, on l'écrit (premier run)."""
    target = tmp_path / "first.txt"
    assert_golden_match("hello", target, regen=False)  # écrit
    assert target.read_text() == "hello"


def test_golden_match_passes_when_identical(tmp_path: Path) -> None:
    """Quand actual == golden, le test passe silencieusement."""
    target = tmp_path / "id.txt"
    target.write_text("identical content")
    assert_golden_match("identical content", target, regen=False)


def test_golden_match_fails_when_different(tmp_path: Path) -> None:
    """Quand actual != golden, AssertionError."""
    target = tmp_path / "diff.txt"
    target.write_text("expected text")
    with pytest.raises(AssertionError, match="Golden mismatch"):
        assert_golden_match("actual text", target, regen=False)


def test_golden_match_regen_overwrites(tmp_path: Path) -> None:
    """En mode regen, le fichier est ré-écrit même si différent."""
    target = tmp_path / "regen.txt"
    target.write_text("old")
    assert_golden_match("new", target, regen=True)
    assert target.read_text() == "new"


# ──────────────────────────────────────────────────────────────────
# Comparateurs sémantiques
# ──────────────────────────────────────────────────────────────────


def test_assert_floats_equal_within_eps() -> None:
    assert_floats_equal(1.0000000001, 1.0, eps=1e-9)


def test_assert_floats_equal_rejects_outside_eps() -> None:
    with pytest.raises(AssertionError, match="diff="):
        assert_floats_equal(1.001, 1.0, eps=1e-9)


def test_assert_set_equal_accepts_reorder() -> None:
    assert_set_equal([3, 1, 2], [1, 2, 3])


def test_assert_set_equal_rejects_missing() -> None:
    with pytest.raises(AssertionError, match="manquants"):
        assert_set_equal([1, 2], [1, 2, 3])


def test_assert_set_equal_rejects_extra() -> None:
    with pytest.raises(AssertionError, match="en trop"):
        assert_set_equal([1, 2, 3, 4], [1, 2, 3])


def test_assert_json_semantic_ignores_key_order() -> None:
    a = {"b": 2, "a": 1}
    e = {"a": 1, "b": 2}
    assert_json_semantic_equal(a, e)


def test_assert_json_semantic_detects_real_diff() -> None:
    with pytest.raises(AssertionError, match="JSON différents"):
        assert_json_semantic_equal({"a": 1}, {"a": 2})


def test_assert_json_semantic_handles_lists() -> None:
    """Les listes gardent l'ordre — c'est le contrat JSON."""
    with pytest.raises(AssertionError):
        assert_json_semantic_equal([1, 2], [2, 1])
