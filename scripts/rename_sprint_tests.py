"""Chantier 1 (audit prod) — dé-sprintage des noms de fichiers de test.

Phase 1.0 = livrable revu AVANT tout renommage : ce script porte la
**règle** de dé-sprintage + la **table d'overrides curée** (collisions
arbitrées, fichiers à supprimer plutôt que renommer, refs externes à
patcher en lockstep).  ``--check`` n'écrit RIEN (revue de stratégie).
``--apply <dir>`` exécute le renommage ``git mv`` d'UN dossier +
patche les refs externes connues le concernant (Phase 1.1..1.N).

Principe : la revue porte sur la RÈGLE + la petite table d'overrides
(≈3 cas sur 184), pas sur 184 lignes générées.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"

# Règle : retire le 1er token sprint après ``test_``.
# Formes : sprint<N> | sprint<N>_... | sprint_<lettre><N>[_s<N>] |
#          sprint_<alnum> | s<N>.
_DESPRINT = re.compile(
    r"^test_(?:sprint_[a-z]?[0-9]+(?:_s[0-9]+)?|sprint[0-9]+|"
    r"sprint_[a-z0-9]+|s[0-9]+)_(?P<rest>.+)$",
)


def desprint(name: str) -> str | None:
    m = _DESPRINT.match(name)
    return f"test_{m.group('rest')}" if m else None


def is_sprint_named(name: str) -> bool:
    # ``test_sprint*`` ou ``test_s<digit>*`` ; exclut ``test_s<lettre>``
    # (test_scientific_audit_2026, test_storage_keys…, test_specs…).
    if name.startswith("test_sprint"):
        return True
    return bool(re.match(r"^test_s[0-9]", name))


# ── Overrides curés (Phase 1.0, arbitrés manuellement) ──────────────
# Collision : 2 fichiers du même dossier dé-sprintent vers le même nom.
# tests/adapters/vlm/ : la série A14 est la suite canonique
# post-rewrite → garde le nom court ; l'ancienne S4 est suffixée.
OVERRIDES: dict[str, str] = {
    "tests/adapters/vlm/test_sprint_a14_s45_vlm_adapters.py":
        "test_vlm_adapters.py",
    "tests/adapters/vlm/test_s4_vlm_adapters.py":
        "test_vlm_adapters_coverage.py",
}

# À SUPPRIMER (pas renommer) en Phase 1.final : ce test AUDITE la
# convention de numérotation sprint elle-même — rendu obsolète par le
# garde-fou anti-régression ``test_no_sprint_named_tests.py``.
DELETE_IN_FINAL: list[str] = [
    "tests/docs/test_sprint_numbering.py",
]

# Refs externes à patcher EN LOCKSTEP avec le lot concerné.
#   (chemin source, "ancien" → "nouveau" calculé)
EXTERNAL_REF_FILES = [
    "CLAUDE.md",
    "Makefile",
]
# Docs : nombreuses réfs ``test_s*`` — patchées par lot via grep
# ciblé au moment du renommage du dossier correspondant (cf. --apply).
# Import inter-tests connu (même lot tests/evaluation/metrics) :
INTRA_TEST_IMPORT = (
    "tests/evaluation/metrics/test_sprint23_anti_hallucination.py",
    "tests.evaluation.metrics.test_sprint19_narrative_engine",
    "tests.evaluation.metrics.test_narrative_engine",
)


def build_map() -> dict[str, str]:
    """Retourne {chemin_relatif_ancien: nouveau_basename}."""
    out: dict[str, str] = {}
    for p in sorted(TESTS.rglob("test_*.py")):
        rel = p.relative_to(REPO).as_posix()
        if not is_sprint_named(p.name):
            continue
        if rel in {*DELETE_IN_FINAL}:
            continue  # supprimé en final, pas renommé
        if rel in OVERRIDES:
            out[rel] = OVERRIDES[rel]
            continue
        nw = desprint(p.name)
        if nw is None:
            raise SystemExit(f"NON DÉSPRINTABLE (étendre la règle) : {rel}")
        out[rel] = nw
    return out


def check() -> int:
    m = build_map()
    # Collisions résiduelles (intra-dossier) post-overrides.
    seen: dict[str, str] = {}
    collisions = []
    for old, nw in m.items():
        d = str(Path(old).parent)
        key = f"{d}/{nw}"
        if key in seen:
            collisions.append((seen[key], old, key))
        else:
            seen[key] = old
        # Cible déjà existante non-sprint dans le dossier ?
        tgt = REPO / d / nw
        if tgt.exists() and (REPO / old).name != nw:
            collisions.append(("<existant>", old, key))
    bydir: dict[str, int] = {}
    for old in m:
        bydir[str(Path(old).parent)] = bydir.get(str(Path(old).parent), 0) + 1
    print(f"Fichiers à renommer : {len(m)}")
    print(f"À supprimer en Phase 1.final : {len(DELETE_IN_FINAL)} "
          f"({', '.join(DELETE_IN_FINAL)})")
    print(f"Overrides curés (collisions arbitrées) : {len(OVERRIDES)}")
    for k, v in OVERRIDES.items():
        print(f"  {k}  ->  {v}")
    print(f"Collisions résiduelles : {len(collisions)}")
    for a, b, k in collisions:
        print(f"  !! {a} ⨯ {b} -> {k}")
    print("Répartition par dossier (ordre d'application conseillé "
          "= concentrique) :")
    for d in sorted(bydir):
        print(f"  {d:45s} {bydir[d]:3d}")
    return 1 if collisions else 0


def apply_dir(target_dir: str) -> int:
    """Phase 1.1..1.N — renomme UN dossier (git mv) + patche les refs
    externes le concernant.  Idempotent, vert exigé après."""
    m = {o: n for o, n in build_map().items()
         if str(Path(o).parent) == target_dir.rstrip("/")}
    if not m:
        print(f"Aucun fichier sprint dans {target_dir}")
        return 0
    renamed: list[tuple[str, str]] = []
    for old, nw in m.items():
        new = str(Path(old).parent / nw)
        subprocess.run(["git", "mv", old, new], check=True, cwd=REPO)
        renamed.append((Path(old).name, nw))
        print(f"git mv {old} -> {new}")
    # Patch refs externes (CLAUDE.md, Makefile, docs/) pour ces fichiers.
    for src in EXTERNAL_REF_FILES + _docs_files():
        sp = REPO / src
        if not sp.exists():
            continue
        txt = sp.read_text(encoding="utf-8")
        orig = txt
        for old_name, new_name in renamed:
            txt = txt.replace(old_name, new_name)
        if txt != orig:
            sp.write_text(txt, encoding="utf-8")
            print(f"patché refs : {src}")
    # Import inter-tests connu.
    if target_dir.rstrip("/") == "tests/evaluation/metrics":
        ip = REPO / INTRA_TEST_IMPORT[0]
        if ip.exists():
            t = ip.read_text(encoding="utf-8").replace(
                INTRA_TEST_IMPORT[1], INTRA_TEST_IMPORT[2])
            ip.write_text(t, encoding="utf-8")
            print(f"patché import inter-tests : {INTRA_TEST_IMPORT[0]}")
    return 0


def _docs_files() -> list[str]:
    return [p.relative_to(REPO).as_posix()
            for p in (REPO / "docs").rglob("*.md")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--apply", metavar="DIR",
                    help="renomme un dossier (git mv) + patche refs")
    a = ap.parse_args()
    if a.apply:
        sys.exit(apply_dir(a.apply))
    sys.exit(check())
