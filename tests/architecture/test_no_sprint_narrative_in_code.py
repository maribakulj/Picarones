"""Garde-fou anti-régression — pas de narration sprint *nettoyable*.

Audit prod Chantier 2 : la narration de chantier (« Sprint X — »,
« Phase B2.7 — », « Audit prod P1 — ») a été retirée du code
courant de ``picarones/`` (catégorie A = supprimée, B = préfixe
retiré, WHY conservé).  Reste un résidu R (≈480) : commentaires où
le tag est *fusionné* à une phrase porteuse d'invariant ou en
docstring multi-ligne — retrait mécanique = mutilation, donc
laissés à une revue humaine ultérieure (hors périmètre auto).

Ce test verrouille l'invariant **A == 0 et B == 0** via le même
classifieur que l'outil de triage : tout nouveau commentaire
sprint *proprement dé-tagable* (préfixe propre / tag seul) qui
réapparaîtrait est catégorie A ou B → échec CI.  L'auteur doit
nommer/commenter par intention, pas par épisode de chantier
(l'historique vit dans git + CHANGELOG).  Le résidu R n'est PAS
compté ici (il est légitime tant qu'il porte un invariant) — sa
résorption éventuelle est un chantier de revue dédié, pas un
ratchet automatique.
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
