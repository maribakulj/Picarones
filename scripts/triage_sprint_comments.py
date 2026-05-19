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


# Dé-taguage SÛR uniquement : tag = préfixe propre, OU ``(tag)``
# isolé en fin (rien d'autre dans la parenthèse).  Tout tag en
# milieu de phrase / fusionné (``Sprint A9 / M-5``, ``S11 + Phase
# 8``, ``(Sprint A3, item B-3)``) → NON auto-dé-tagable (le retirer
# mutile la phrase) → catégorie R (revue humaine).
_PREFIX = re.compile(
    r"^\(?\s*(?:" + TAG.pattern + r")\s*\)?\s*[—\-:]\s+(?P<rest>\S.*)$",
)
_TRAIL = re.compile(
    r"^(?P<head>\S.*?)\s*(?:\((?:" + TAG.pattern + r")\)|[—\-]\s*(?:"
    + TAG.pattern + r"))\s*[.]?\s*$",
)
_TAG_ALONE = re.compile(
    r"^\(?\s*(?:" + TAG.pattern + r")\s*\)?\s*[.:]?\s*$",
)


def _classify(text: str) -> tuple[str, str, bool]:
    """text = contenu du commentaire SANS le ``# `` initial.
    Retourne (catégorie, remplacement_proposé, forced_non_A).

    Conservateur : on n'auto-modifie QUE les cas dont le résultat
    est syntaxiquement propre et déterministe.  Tout le reste → R.
    """
    forced = bool(CONSTRAINT.search(text) or CROSSREF.search(text))
    t = text.strip()

    # Tag seul (± parenthèse/ponctuation) : suppression sûre.
    if _TAG_ALONE.match(t):
        return ("R" if forced else "A", "", forced)

    # Tag = préfixe propre ``Tag — <reste>`` → garder <reste>.
    m = _PREFIX.match(t)
    if m:
        rest = m.group("rest").strip()
        if rest and not TAG.search(rest):  # pas de 2e tag résiduel
            return ("B", rest, forced)
        return ("R", "", forced)

    # ``<head> (Tag)`` ou ``<head> — Tag`` en fin, rien d'autre dans
    # la parenthèse → garder <head>.
    m = _TRAIL.match(t)
    if m:
        head = m.group("head").strip(" .—-:")
        if head and not TAG.search(head):
            return ("B", head, forced)
        return ("R", "", forced)

    # Tag en milieu / fusionné / parenthèse à contenu mixte :
    # retrait mécanique mutilerait la phrase → revue humaine.
    return ("R", "", forced)


def analyze_line(ln: str) -> dict | None:
    """Source de vérité unique : classe UNE ligne live.
    Retourne ``None`` si pas de tag.  Utilisé par scan() ET apply()
    (re-lecture live → robuste à toute dérive)."""
    if not TAG.search(ln):
        return None
    stripped = ln.lstrip()
    if stripped.startswith("#"):
        text, kind = stripped[1:].strip(), "comment"
    elif "#" in ln and TAG.search(ln.split("#", 1)[1]):
        text, kind = ln.split("#", 1)[1].strip(), "trailing"
    else:
        text, kind = stripped, "docstring"
    cat, repl, forced = _classify(text)
    if kind == "docstring" and cat == "A":
        cat = "R"  # jamais d'auto-suppression de prose-en-chaîne
    return {"kind": kind, "cat": cat, "repl": repl,
            "forced_non_A": forced, "text": text}


def scan() -> list[dict]:
    rows: list[dict] = []
    for p in sorted(PKG.rglob("*.py")):
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, ln in enumerate(lines, 1):
            a = analyze_line(ln)
            if a is None:
                continue
            rows.append({
                "file": p.relative_to(REPO).as_posix(),
                "line": i, "layer": _layer(p), **a,
            })
    return rows


def _tq(s: str) -> int:
    return s.count('"""') + s.count("'''")


def apply_layer(layer: str) -> int:
    """Applique A+B d'UNE couche.  Re-lecture live + recalcul (pas de
    confiance au scan stocké) ; A/B comment+trailing+docstring ;
    invariant : triple-quotes inchangées (docstring) ; garde ultime :
    si le fichier résultant ne parse plus (ast) → NON écrit, abort."""
    import ast
    targets: dict[Path, list[int]] = {}
    for r in scan():
        if r["layer"] == layer and r["cat"] in ("A", "B"):
            targets.setdefault(REPO / r["file"], []).append(r["line"])
    if not targets:
        print(f"Aucun A/B dans la couche {layer}")
        return 0
    n_a = n_b = n_skip = 0
    for fp, line_nos in targets.items():
        lines = fp.read_text(encoding="utf-8").splitlines(keepends=True)
        drop: set[int] = set()
        for ln_no in sorted(set(line_nos), reverse=True):
            raw = lines[ln_no - 1]
            nl = "\n" if raw.endswith("\n") else ""
            ln = raw[:-1] if nl else raw
            a = analyze_line(ln)
            if a is None or a["cat"] not in ("A", "B"):
                n_skip += 1
                continue
            indent = ln[: len(ln) - len(ln.lstrip())]
            if a["cat"] == "A":
                if a["kind"] == "comment":
                    drop.add(ln_no - 1)
                    n_a += 1
                elif a["kind"] == "trailing":
                    code = ln.split("#", 1)[0].rstrip()
                    lines[ln_no - 1] = code + nl
                    n_a += 1
                else:
                    n_skip += 1
                continue
            # cat == B
            repl = a["repl"]
            if not repl or TAG.search(repl):
                n_skip += 1
                continue
            if a["kind"] == "comment":
                new = f"{indent}# {repl}"
            elif a["kind"] == "trailing":
                new = f"{ln.split('#', 1)[0].rstrip()}  # {repl}"
            else:  # docstring : prose-en-chaîne
                new = f"{indent}{repl}"
                if _tq(new) != _tq(ln):  # invariant quotes
                    n_skip += 1
                    continue
            lines[ln_no - 1] = new + nl
            n_b += 1
        new_text = "".join(
            x for k, x in enumerate(lines) if k not in drop
        )
        try:
            ast.parse(new_text)
        except SyntaxError as e:
            print(f"ABORT {fp} : ne parse plus ({e}) — NON écrit")
            return 2
        fp.write_text(new_text, encoding="utf-8")
    print(f"{layer} : A={n_a} B={n_b} skip={n_skip}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dump", metavar="FILE",
                    help="écrit la table complète (revue)")
    ap.add_argument("--apply", metavar="LAYER",
                    help="applique A+B d'une couche (re-lecture live)")
    a = ap.parse_args()
    if a.apply:
        return apply_layer(a.apply)
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
