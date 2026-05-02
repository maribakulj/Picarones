"""Tests Sprint A5 — structure et idempotence du corpus de référence.

Item M-14. Le corpus est généré au runtime via ``_generate.py``. Ce
fichier valide que la génération produit la structure attendue et est
idempotente (deux générations successives produisent les mêmes octets).

L'exécution effective du benchmark Tesseract sur le corpus se fait
dans le workflow CI ``perf_regression.yml`` (cron hebdo) — pas ici,
car ça exigerait que Tesseract soit installé sur la machine de test
(disponible en CI, pas garanti en dev).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path



REFERENCE_DIR = Path(__file__).parent / "reference_corpus"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_corpus_directory_exists() -> None:
    """Le dossier doit exister et contenir le script + le README."""
    assert REFERENCE_DIR.exists() and REFERENCE_DIR.is_dir()
    assert (REFERENCE_DIR / "_generate.py").exists()
    assert (REFERENCE_DIR / "README.md").exists()


def test_each_doc_has_image_and_gt() -> None:
    """Chaque ``doc_<id>.png`` a son ``doc_<id>.gt.txt`` jumeau."""
    pngs = sorted(REFERENCE_DIR.glob("doc_*.png"))
    gts = sorted(REFERENCE_DIR.glob("doc_*.gt.txt"))
    assert len(pngs) >= 5, "Au moins 5 documents de référence attendus"
    assert len(pngs) == len(gts), (
        f"{len(pngs)} PNG mais {len(gts)} GT — alignement cassé."
    )
    for png in pngs:
        gt = png.with_suffix(".gt.txt")
        assert gt.exists(), f"GT manquante pour {png.name}"
        assert gt.stat().st_size > 0, f"GT vide pour {png.name}"


def test_corpus_generation_is_idempotent(tmp_path: Path) -> None:
    """Deux générations successives doivent produire des PNG bit-à-bit
    identiques. Garantit la reproductibilité du baseline CER."""
    # Copie le script dans un tmp_path
    script_target = tmp_path / "_generate.py"
    shutil.copy(REFERENCE_DIR / "_generate.py", script_target)

    import importlib.util
    spec = importlib.util.spec_from_file_location("gen", script_target)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    mod.generate(out1)
    mod.generate(out2)

    pngs1 = sorted(out1.glob("doc_*.png"))
    pngs2 = sorted(out2.glob("doc_*.png"))
    assert [p.name for p in pngs1] == [p.name for p in pngs2]

    for p1, p2 in zip(pngs1, pngs2, strict=True):
        h1 = _file_sha256(p1)
        h2 = _file_sha256(p2)
        assert h1 == h2, (
            f"Génération non-idempotente pour {p1.name} : {h1} vs {h2}. "
            "Vérifier que la police par défaut Pillow est stable."
        )


def test_gt_files_are_utf8(tmp_path: Path) -> None:
    """Les fichiers GT doivent être en UTF-8 valide (pas de BOM, pas de
    caractères de contrôle inutiles)."""
    for gt in REFERENCE_DIR.glob("doc_*.gt.txt"):
        text = gt.read_text(encoding="utf-8")
        assert text.strip(), f"{gt.name} est vide après strip"
        assert "\x00" not in text, f"{gt.name} contient un NUL byte"


def test_no_unexpected_files_in_corpus_dir() -> None:
    """Garde-fou : le dossier ne doit pas accumuler de fichiers parasites
    (ex : `.partial.json` du runner, `.DS_Store` macOS)."""
    allowed = {
        "_generate.py",
        "README.md",
        "test_reference_corpus_structure.py",  # parfois listé via os.scandir si test à proximité
    }
    unexpected = []
    for f in REFERENCE_DIR.iterdir():
        if f.name in allowed:
            continue
        if f.suffix in (".png", ".txt"):
            continue  # documents générés
        if f.name.startswith("__"):
            continue  # __pycache__
        unexpected.append(f.name)
    assert not unexpected, (
        f"Fichiers parasites dans reference_corpus/ : {unexpected}"
    )
