"""Garde-fou anti-régression — pas de narration sprint *nettoyable*.

Audit prod Chantier 2 : la narration de chantier (« Sprint X — »,
« Phase B2.7 — », « Audit prod P1 — ») a été retirée du code
courant de ``picarones/`` (catégorie A = supprimée, B = préfixe
retiré, WHY conservé).  Reste un résidu R (≈480) : commentaires où
le tag est *fusionné* à une phrase porteuse d'invariant ou en
docstring multi-ligne — retrait mécanique = mutilation, donc
laissés à une revue humaine ultérieure (hors périmètre auto).

Ce test verrouille **deux** invariants :

1. **A == 0 et B == 0** via le classifieur de triage : toute
   narration sprint *proprement dé-tagable* (préfixe propre / tag
   seul) qui réapparaîtrait → échec CI.  L'auteur doit commenter
   par intention, pas par sprint.
2. **Compteur TOTAL ≤ baseline** (ratchet-down absolu) : ferme le
   *seul* trou réel du test #1 — une narration sous forme R
   (tag fusionné mid-phrase, ``# pattern Sprint 78``) passerait #1
   silencieusement.  Le total ne peut désormais que DÉCROÎTRE ;
   tout ajout de narration (même R-style) échoue, et toute
   résorption oblige à resserrer le baseline (pattern doc_paths).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "triage_sprint_comments.py"
)


def _load_triage():
    spec = importlib.util.spec_from_file_location("_triage", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: Compteur total de narrations sprint dans ``picarones/`` à la
#: clôture du Chantier 2 (A=0, B=0, R=483).  Ratchet-down absolu :
#: si on descend en dessous (revue humaine de R, reformulation),
#: BAISSER cette valeur dans le même commit.  Ne JAMAIS augmenter.
BASELINE = 483


def test_no_auto_cleanable_sprint_narrative() -> None:
    triage = _load_triage()
    rows = triage.scan()
    offenders = [
        f"{r['cat']} {r['file']}:{r['line']}  «{r['text'][:80]}»"
        for r in rows
        if r["cat"] in ("A", "B")
    ]
    assert not offenders, (
        "Narration sprint *nettoyable* réapparue (catégorie A/B). "
        "Nommer/commenter par INTENTION, pas par sprint — "
        "l'historique vit dans git/CHANGELOG. Lancer "
        "`python scripts/triage_sprint_comments.py --check` :\n  "
        + "\n  ".join(offenders)
    )


def test_total_sprint_narrative_at_or_below_baseline() -> None:
    """Ratchet-down absolu : toute narration sprint ajoutée (même
    forme R fusionnée) fait dépasser le baseline → échec.  Ferme le
    *seul* trou réel du test #1."""
    triage = _load_triage()
    total = len(triage.scan())
    assert total <= BASELINE, (
        f"Narration sprint en hausse : {total} > baseline {BASELINE}. "
        "Le ratchet est strictement décroissant — toute nouvelle "
        "mention Sprint/Phase/Audit dans picarones/ doit être "
        "reformulée par intention (l'historique vit dans CHANGELOG). "
        "Lancer ``python scripts/triage_sprint_comments.py --check``."
    )


def test_baseline_must_be_tightened_when_progress_made() -> None:
    """Pattern miroir (cf. ``test_doc_paths``) : si le total est
    descendu sous le baseline, c'est qu'une revue R a porté ses
    fruits — BAISSER :data:`BASELINE` dans le même commit pour
    verrouiller le gain."""
    triage = _load_triage()
    total = len(triage.scan())
    assert total >= BASELINE, (
        f"Excellent : {total} narrations < baseline {BASELINE}. "
        f"Resserrer ``BASELINE = {total}`` dans ce fichier pour "
        "verrouiller le progrès."
    )
