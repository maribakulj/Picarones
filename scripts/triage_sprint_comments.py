"""Chantier 2 (audit prod) — triage des commentaires « sprint » du code.

Phase 2.0 = livrable revu AVANT tout retrait : extrait chaque
commentaire/ligne de docstring portant un marqueur de sprint/phase
dans ``picarones/``, propose une catégorie et le remplacement exact,
et **ne modifie RIEN**.  La revue porte sur la TABLE (surtout les A
et les B), pas sur 250 hunks de diff à l'aveugle.

Catégories
----------
A  supprimer la ligne — narration chrono pure, aucune info perdue.
B  retirer le seul préfixe ``Sprint X —/Phase Y —/Audit Z —`` et
   GARDER le reste (contrainte/invariant/raison non évidente).
R  revue humaine obligatoire — ambigu : le filet mécanique a
   interdit A mais le « reste » est vide/trivial, ou contexte
   docstring multi-ligne, ou référence load-bearing possible.

Filet mécanique (rend A *impossible à l'aveugle*) : tout commentaire
contenant un mot-contrainte ou une référence cross-fichier ne peut
PAS être A → forcé B (si reste non vide) ou R.  Pire cas dégradé =
on garde un commentaire supprimable (verbeux mais sûr), jamais
l'inverse.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "picarones"

# Marqueur sprint/phase = un TAG (pas le mot « phase » en prose).
TAG = re.compile(
    r"(Sprint\s+[A-Za-z]?\d[\w.\-]*|Phase\s+[A-Za-z]?\d+[\w.\-]*|"
    r"Audit\s+prod[\w .\-]*|audit\s+code-quality[\w .\-]*|"
    r"Plan\s+A-H|[AB]\d+-final|Sprint\s+\d+)",
)
# Préfixe à retirer en catégorie B : le tag + séparateur (— - : .).
TAG_PREFIX = re.compile(
    r"^\(?\s*(?:" + TAG.pattern + r")\s*\)?\s*[—\-:.]*\s*",
)

# Filet : un commentaire qui matche => JAMAIS catégorie A.
CONSTRAINT = re.compile(
    r"\b(doit|sinon|attention|race|deadlock|s[ée]curit[ée]|invariant|"
    r"[ée]quivalen|ne\s+pas|pourquoi|workaround|bug|contournement|"
    r"FIXME|TODO|sciemment|volontairement|garanti|imp[ée]ratif|"
    r"obligatoire|refuse|l[èe]ve|raises?|hotfix|RGPD|fuite|vecteur|"
    r"piège|foot-?gun|jamais|toujours|sous\s+peine|critique)\b",
    re.I,
)
CROSSREF = re.compile(r"\w+\.py:\d|\bcf\.|\bvoir\b|coh[ée]rent\s+avec|"
                      r"copi[ée]\s+de|pattern\b|cf\b")


def _layer(p: Path) -> str:
    rel = p.relative_to(PKG).parts
    return rel[0] if rel else "<root>"


def _classify(text: str) -> tuple[str, str, bool]:
    """text = contenu du commentaire SANS le ``# `` initial.
    Retourne (catégorie, remplacement_proposé, forced_non_A)."""
    forced = bool(CONSTRAINT.search(text) or CROSSREF.search(text))
    remainder = TAG_PREFIX.sub("", text).strip()
    tag_only = remainder == "" or remainder == text.strip()
    if tag_only:
        # le commentaire EST (essentiellement) le tag → rien à garder
        return ("R" if forced else "A", "", forced)
    # il reste du texte après le tag
    if forced or len(remainder.split()) > 3:
        return ("B", remainder, forced)
    # reste court non contraint : libellé de section (« métriques
    # avancées ») → A, sauf filet
    return ("R" if forced else "A", "", forced)


def scan() -> list[dict]:
    rows: list[dict] = []
    for p in sorted(PKG.rglob("*.py")):
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, ln in enumerate(lines, 1):
            if not TAG.search(ln):
                continue
            stripped = ln.lstrip()
            is_comment = stripped.startswith("#")
            # commentaire de ligne pur, ou trailing ``code  # ...``
            if is_comment:
                text = stripped[1:].strip()
                kind = "comment"
            elif "#" in ln and TAG.search(ln.split("#", 1)[1]):
                text = ln.split("#", 1)[1].strip()
                kind = "trailing"
            else:
                # marqueur dans une docstring / chaîne
                text = stripped
                kind = "docstring"
            cat, repl, forced = _classify(text)
            if kind == "docstring":
                # jamais d'auto-A sur du texte en chaîne : revue
                cat = "R" if cat == "A" else cat
            rows.append({
                "file": p.relative_to(REPO).as_posix(),
                "line": i, "layer": _layer(p), "kind": kind,
                "cat": cat, "forced_non_A": forced,
                "text": text, "repl": repl,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dump", metavar="FILE",
                    help="écrit la table complète (revue)")
    a = ap.parse_args()
    rows = scan()
    by_layer: dict[str, dict[str, int]] = {}
    for r in rows:
        d = by_layer.setdefault(r["layer"], {"A": 0, "B": 0, "R": 0})
        d[r["cat"]] += 1
    tot = {"A": 0, "B": 0, "R": 0}
    for d in by_layer.values():
        for k in tot:
            tot[k] += d[k]
    print(f"Commentaires sprint/phase : {len(rows)}")
    print(f"  A (supprimer)       : {tot['A']}")
    print(f"  B (dé-préfixer)     : {tot['B']}")
    print(f"  R (revue humaine)   : {tot['R']}")
    print(f"  dont forcés non-A   : {sum(r['forced_non_A'] for r in rows)}")
    print("Par couche (ordre concentrique conseillé) :")
    for lyr in sorted(by_layer):
        d = by_layer[lyr]
        print(f"  {lyr:14s} A={d['A']:3d} B={d['B']:3d} R={d['R']:3d}")
    print("\n--- TOUTES les catégories A (scrutiny maximale) ---")
    for r in rows:
        if r["cat"] == "A":
            print(f"  {r['file']}:{r['line']}  «{r['text'][:90]}»")
    if a.dump:
        out = [
            f"{r['cat']}\t{r['file']}:{r['line']}\t{r['kind']}\t"
            f"forced={r['forced_non_A']}\tTEXT={r['text']}\t"
            f"REPL={r['repl']}"
            for r in rows
        ]
        Path(a.dump).write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"\nTable complète écrite : {a.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
