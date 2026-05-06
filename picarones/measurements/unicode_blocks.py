"""Précision par bloc Unicode — Sprint 55.

Sprint 55 — A.II.3.1 du plan d'évolution 2026 (métriques philologiques).

Pourquoi ce module
------------------
Pour un éditeur d'imprimés anciens ou un médiéviste, la question
n'est pas seulement *« quel CER global ? »* mais *« quels caractères
historiques ce moteur restitue-t-il fidèlement ? »*.  Une phrase de
synthèse actionnable en un coup d'œil :

> *« GPT-4o restitue 95 % du Latin de Base mais seulement 12 % des
> formes de présentation latine (ﬁ, ﬂ, ſ…). »*

Ce module agrège la précision par **bloc Unicode standard** (Latin de
Base, Latin Étendu A/B, Diacritiques combinants, Présentation latine,
etc.).  Le résultat permet directement de choisir un moteur selon le
type de glyphes attendus dans le corpus.

Stratégie de découpage
----------------------
Cohérente avec NER (Sprint 38), Flesch (Sprint 52), Reading order F1
(Sprint 53), Layout F1 (Sprint 54) : couche de calcul pure d'abord.
Le câblage runner et la vue HTML suivent dans des sprints dédiés.

Convention d'alignement
-----------------------
Alignement caractère par caractère via ``difflib.SequenceMatcher`` :

- chaque caractère de la GT est classé dans son bloc Unicode,
- pour chaque position GT couverte par un opcode ``equal`` →
  +1 dans ``correct[bloc]``,
- pour chaque position GT non couverte (replace, delete) → +0,
- les insertions côté hypothèse (caractères absents de la GT) ne
  contribuent à aucun bloc — elles sont visibles uniquement via le
  CER global.

Précision par bloc = ``correct[bloc] / total[bloc]``.

Liste des blocs reconnus
------------------------
Centrée sur les glyphes courants des corpus patrimoniaux européens.
Tout caractère hors de cette table est classé dans ``"Other"``
(garantit une couverture exhaustive : ``sum(total[bloc]) ==
len(GT)``).
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Optional

from picarones.core.metric_registry import register_metric
from picarones.domain.artifacts import ArtifactType

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Table des blocs Unicode reconnus
# ──────────────────────────────────────────────────────────────────────────

# Triplets (nom, code_point_min, code_point_max) — bornes inclusives.
# Centré sur les blocs pertinents pour les corpus patrimoniaux
# européens (manuscrits médiévaux, imprimés anciens, archives).
# Source : https://www.unicode.org/charts/
_UNICODE_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("Basic Latin",                              0x0000, 0x007F),
    ("Latin-1 Supplement",                       0x0080, 0x00FF),
    ("Latin Extended-A",                         0x0100, 0x017F),
    ("Latin Extended-B",                         0x0180, 0x024F),
    ("IPA Extensions",                           0x0250, 0x02AF),
    ("Spacing Modifier Letters",                 0x02B0, 0x02FF),
    ("Combining Diacritical Marks",              0x0300, 0x036F),
    ("Greek and Coptic",                         0x0370, 0x03FF),
    ("Cyrillic",                                 0x0400, 0x04FF),
    ("Hebrew",                                   0x0590, 0x05FF),
    ("Arabic",                                   0x0600, 0x06FF),
    ("General Punctuation",                      0x2000, 0x206F),
    ("Superscripts and Subscripts",              0x2070, 0x209F),
    ("Currency Symbols",                         0x20A0, 0x20CF),
    ("Combining Diacritical Marks Supplement",   0x1DC0, 0x1DFF),
    ("Latin Extended Additional",                0x1E00, 0x1EFF),
    ("Latin Extended-C",                         0x2C60, 0x2C7F),
    ("Latin Extended-D",                         0xA720, 0xA7FF),  # médiéval
    ("Latin Extended-E",                         0xAB30, 0xAB6F),
    ("Alphabetic Presentation Forms",            0xFB00, 0xFB4F),  # ﬁ, ﬂ, ﬀ…
    ("Mathematical Alphanumeric Symbols",        0x1D400, 0x1D7FF),
    ("Medieval Unicode Font Initiative (MUFI)",  0xE000, 0xF8FF),  # PUA
)


def get_block(char: str) -> str:
    """Retourne le nom du bloc Unicode contenant ``char``.

    Pour un caractère hors des blocs listés (ex. CJK, emoji, etc.),
    retourne ``"Other"``.  Pour une chaîne multi-caractères, on
    considère uniquement le premier code-point.
    """
    if not char:
        return "Other"
    cp = ord(char[0])
    for name, lo, hi in _UNICODE_BLOCKS:
        if lo <= cp <= hi:
            return name
    return "Other"


# ──────────────────────────────────────────────────────────────────────────
# Calcul d'accuracy par bloc
# ──────────────────────────────────────────────────────────────────────────


def compute_unicode_block_accuracy(
    reference: Optional[str],
    hypothesis: Optional[str],
) -> dict:
    """Calcule la précision (recall caractère) par bloc Unicode.

    Parameters
    ----------
    reference:
        Texte GT.  Chaque caractère est classé dans son bloc Unicode.
    hypothesis:
        Texte produit par le moteur OCR.

    Returns
    -------
    dict
        ``{
            "per_block": {
                bloc_name: {
                    "correct": int,    # caractères GT correctement restitués
                    "total":   int,    # caractères GT du bloc
                    "accuracy": float, # correct / total ∈ [0, 1]
                },
                ...
            },
            "global_accuracy": float,    # somme(correct) / somme(total)
            "n_chars_reference": int,
        }``

    Cas dégénérés
    -------------
    - GT vide → ``per_block`` vide, ``global_accuracy = 0.0``,
      ``n_chars_reference = 0``.
    - hypothèse vide + GT non-vide → tous les blocs à
      ``accuracy = 0``.
    - GT et hyp identiques → tous les blocs à ``accuracy = 1``.
    """
    ref = reference or ""
    hyp = hypothesis or ""
    n_ref = len(ref)

    if n_ref == 0:
        return {
            "per_block": {},
            "global_accuracy": 0.0,
            "n_chars_reference": 0,
        }

    # 1. Compter le total par bloc
    total: dict[str, int] = {}
    for ch in ref:
        b = get_block(ch)
        total[b] = total.get(b, 0) + 1

    # 2. Aligner par opcodes de SequenceMatcher
    #    Pour chaque opcode ``equal``, les positions ``i1..i2-1`` du GT
    #    sont correctement restituées → +1 par caractère dans son bloc.
    correct: dict[str, int] = {b: 0 for b in total}
    matcher = SequenceMatcher(a=ref, b=hyp, autojunk=False)
    for op, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if op != "equal":
            continue
        for i in range(i1, i2):
            b = get_block(ref[i])
            correct[b] = correct.get(b, 0) + 1

    per_block: dict[str, dict] = {}
    for b in sorted(total):
        n = total[b]
        c = correct.get(b, 0)
        per_block[b] = {
            "correct": c,
            "total": n,
            "accuracy": c / n if n > 0 else 0.0,
        }

    n_correct_total = sum(d["correct"] for d in per_block.values())
    return {
        "per_block": per_block,
        "global_accuracy": n_correct_total / n_ref,
        "n_chars_reference": n_ref,
    }


def unicode_block_global_accuracy(
    reference: Optional[str],
    hypothesis: Optional[str],
) -> float:
    """Raccourci : retourne ``global_accuracy`` (fraction de
    caractères GT correctement restitués)."""
    return compute_unicode_block_accuracy(reference, hypothesis)["global_accuracy"]


# ──────────────────────────────────────────────────────────────────────────
# Enregistrement dans le registre typé (Sprint 34)
# ──────────────────────────────────────────────────────────────────────────


@register_metric(
    name="unicode_block_global_accuracy",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description=(
        "Fraction de caractères GT correctement restitués par "
        "l'OCR (alignement caractère par caractère via difflib). "
        "Pour le détail par bloc Unicode (Latin de Base, Présentation "
        "latine, etc.), utiliser compute_unicode_block_accuracy."
    ),
    higher_is_better=True,
    tags={"text", "unicode", "philology"},
)
def _registered_global_accuracy(reference: str, hypothesis: str) -> float:
    return unicode_block_global_accuracy(reference, hypothesis)


__all__ = [
    "get_block",
    "compute_unicode_block_accuracy",
    "unicode_block_global_accuracy",
]
