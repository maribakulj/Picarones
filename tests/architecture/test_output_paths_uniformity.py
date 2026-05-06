"""Garde-fou : tous les adapters qui écrivent un output passent par
``resolve_output_path``.

L'audit S58 a relevé que S51 (helper de résolution de chemin pour
respecter ``context.workspace_uri``) n'était appliqué qu'à 1 OCR sur
5 + LLM/VLM.  Les 4 autres OCR (Pero, Mistral, Google Vision, Azure
DI) écrivaient encore directement dans ``image_path.parent``,
plantant en mode read-only mount — exactement le problème que S51
prétendait régler.

Ce test rejette tout ``image_path.parent / f"{stem}.{name}.txt"``
ou variante équivalente dans les modules d'adapter (OCR/LLM/VLM).
La forme canonique unique est ``resolve_output_path(...)``.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Modules à scanner — tous les adapters qui produisent des fichiers
#: de sortie.
ADAPTER_DIRS: tuple[Path, ...] = (
    REPO_ROOT / "picarones" / "adapters" / "ocr",
    REPO_ROOT / "picarones" / "adapters" / "llm",
    REPO_ROOT / "picarones" / "adapters" / "vlm",
)

#: Module canonique qui définit le helper — exempté du test.
HELPER_MODULE: Path = (
    REPO_ROOT / "picarones" / "adapters" / "output_paths.py"
)

#: Modules exemptés avec justification.
#:
#: - ``ocr/precomputed.py`` : adapter qui **lit** un texte pré-calculé
#:   placé manuellement à côté de l'image par l'utilisateur.  Le
#:   ``image_path.parent`` est l'emplacement attendu de l'**input**,
#:   pas une sortie produite par l'adapter.  La sémantique attendue
#:   par les utilisateurs est précisément « cherche à côté de
#:   l'image » — déplacer ça vers le workspace casserait l'usage
#:   documenté.
EXEMPTED: frozenset[Path] = frozenset({
    REPO_ROOT / "picarones" / "adapters" / "ocr" / "precomputed.py",
})

#: Pattern interdit : écriture directe à côté de l'image source.
#: ``image_path.parent / f"…"`` ou ``input_path.parent / f"…"``.
FORBIDDEN_PATTERN: re.Pattern[str] = re.compile(
    r"(?:image_path|input_path|img_path)\s*\.\s*parent\s*/\s*f[\"']",
)


def _adapter_files() -> list[Path]:
    files: list[Path] = []
    for d in ADAPTER_DIRS:
        if d.exists():
            files.extend(
                p for p in d.rglob("*.py")
                if p != HELPER_MODULE and p not in EXEMPTED
            )
    return sorted(files)


def test_adapters_write_via_resolve_output_path() -> None:
    """Aucun adapter ne contourne ``resolve_output_path``."""
    offenders: list[tuple[str, int, str]] = []
    for f in _adapter_files():
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN_PATTERN.search(line):
                rel = f.relative_to(REPO_ROOT).as_posix()
                offenders.append((rel, i, line.strip()))
    if offenders:
        sample = "\n".join(
            f"  {p}:{n} → {s}" for p, n, s in offenders[:10]
        )
        raise AssertionError(
            f"\n{len(offenders)} adapter(s) écrivent à côté de "
            "l'image source au lieu de passer par "
            "``resolve_output_path``.  Cela casse les corpus "
            "montés en read-only.\n\n"
            f"{sample}\n\n"
            "Remplacer par ``resolve_output_path(input_path=...,"
            " adapter_name=self.name, suffix=..., context=context)``."
        )
