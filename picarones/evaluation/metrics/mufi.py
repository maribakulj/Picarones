"""Couverture MUFI — Sprint 57.

Sprint 57 — A.II.3.3 du plan d'évolution 2026 (clôture axe A.II.3
philologique).

Pourquoi ce module
------------------
La **Medieval Unicode Font Initiative** (MUFI v4.0) standardise les
caractères médiévaux que les éditeurs critiques attendent dans une
transcription fidèle : signes d'abréviation, ligatures, lettres
spéciales (ƿ wynn, þ thorn), ponctuation médiévale, marques
diacritiques rares, etc.  Pour les médiévistes, la **couverture
MUFI** d'un moteur OCR/HTR est un critère éditorial central.

Ce module mesure le taux de **caractères MUFI de la GT
correctement restitués** dans l'OCR, après alignement caractère par
caractère (même approche que la précision par bloc Unicode du
Sprint 55).

Détection des caractères MUFI
-----------------------------
La spécification MUFI v4.0 référence ~1300 caractères dans plusieurs
plages Unicode.  Plutôt que d'embarquer la liste exhaustive (qui
évolue), on utilise un **set de plages caractéristiques** suffisant
pour les corpus patrimoniaux européens courants :

- PUA principal (U+E000–U+F8FF) : zone usuelle des glyphes MUFI
  qui n'ont pas (encore) de point de code Unicode standard.
- Latin Extended-D (U+A720–U+A7FF) : abréviations latines
  médiévales (ꝑ, ꝓ, ꝗ, etc.).
- Combining Diacritical Marks Supplement (U+1DC0–U+1DFF) :
  diacritiques médiévaux rares (macron suscript, etc.).
- Alphabetic Presentation Forms (U+FB00–U+FB4F) : ligatures
  (ﬁ, ﬂ, ﬀ).
- Une **liste explicite** de caractères médiévaux dans les blocs
  Latin Extended-A/B/Additional (þ, ð, ƿ, ſ, æ, œ, etc.)

L'utilisateur peut personnaliser via le paramètre ``custom_chars``
de ``compute_mufi_coverage`` pour étendre ou restreindre.

Stratégie de découpage
----------------------
Cohérente avec NER (Sprint 38), Flesch (52), Reading order F1 (53),
Layout F1 (54), Bloc Unicode (55), Abréviations (56) : couche de
calcul pure d'abord.  Le câblage runner et la vue HTML suivent dans
des sprints dédiés.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from rapidfuzz.distance import Levenshtein

from picarones.evaluation.metric_registry import register_metric
from picarones.domain.artifacts import ArtifactType

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Plages Unicode considérées comme MUFI
# ──────────────────────────────────────────────────────────────────────────

# Triplets (nom, lo, hi) inclusifs.  Source : MUFI v4.0 spec
# (https://mufi.info/) + revue manuelle des caractères patrimoniaux
# courants.
_MUFI_RANGES: tuple[tuple[str, int, int], ...] = (
    ("Private Use Area",                          0xE000, 0xF8FF),
    ("Latin Extended-D",                          0xA720, 0xA7FF),
    ("Combining Diacritical Marks Supplement",    0x1DC0, 0x1DFF),
    ("Alphabetic Presentation Forms",             0xFB00, 0xFB4F),
)

# Caractères MUFI explicites hors plages couvertes par les ranges.
# Surtout des glyphes médiévaux standardisés en Unicode mais qui ne
# sont pas dans le PUA ni dans Latin Extended-D : þ, ð, ƿ, ſ, æ, œ,
# ø, ƀ, ƕ, etc.  Liste raisonnée pour les corpus européens médiévaux.
_MUFI_EXPLICIT_CHARS: frozenset[str] = frozenset(
    [
        # Lettres médiévales standard
        "þ", "Þ",  # thorn — vieil anglais, islandais
        "ð", "Ð",  # eth — vieil anglais, islandais
        "ƿ", "Ƿ",  # wynn — vieil anglais
        "ſ",       # s long médiéval (déjà U+017F)
        "æ", "Æ",  # ash
        "œ", "Œ",  # ethel
        "ø", "Ø",  # o barré
        # Lettres rares avec barré (pour préfixes abréviés)
        "ƀ",       # b barré
        "ŧ",       # t barré
        "đ",       # d barré
        "ħ",       # h barré
        # Yogh
        "ȝ", "Ȝ",
        # Autres signes médiévaux courants
        "ꜿ",       # con
        # Note : la liste est volontairement courte ; pour étendre,
        # l'utilisateur peut passer ``custom_chars`` à
        # ``compute_mufi_coverage``.
    ]
)


def is_mufi_char(char: str, custom_chars: Optional[frozenset[str]] = None) -> bool:
    """Retourne ``True`` si ``char`` est considéré comme MUFI.

    Reconnaît :

    - les caractères dans les plages Unicode MUFI (``_MUFI_RANGES``),
    - les caractères de la liste explicite (``_MUFI_EXPLICIT_CHARS``),
    - tout caractère supplémentaire fourni via ``custom_chars``.

    Pour une chaîne multi-caractères, seul le premier code-point
    est considéré.
    """
    if not char:
        return False
    cp = ord(char[0])
    for _name, lo, hi in _MUFI_RANGES:
        if lo <= cp <= hi:
            return True
    if char[0] in _MUFI_EXPLICIT_CHARS:
        return True
    if custom_chars and char[0] in custom_chars:
        return True
    return False


# ──────────────────────────────────────────────────────────────────────────
# Calcul de couverture MUFI
# ──────────────────────────────────────────────────────────────────────────


def compute_mufi_coverage(
    reference: Optional[str],
    hypothesis: Optional[str],
    custom_chars: Optional[Iterable[str]] = None,
) -> dict:
    """Calcule la couverture MUFI : taux de caractères MUFI de la GT
    correctement restitués dans l'hypothèse.

    Parameters
    ----------
    reference:
        Texte GT.
    hypothesis:
        Texte produit par l'OCR.
    custom_chars:
        Itérable optionnel de caractères supplémentaires à considérer
        comme MUFI (utile pour les éditeurs ayant une convention
        propre).  Chaque entrée doit être un caractère unique.

    Returns
    -------
    dict
        ``{
            "n_mufi_chars_reference": int,    # caractères MUFI dans la GT
            "n_mufi_chars_preserved": int,    # MUFI restitués correctement
            "coverage": float,                 # ∈ [0, 1] ou 0 si N=0
            "per_char": {char: {"total", "preserved", "coverage"}},
            "missed_chars": list[str],         # caractères MUFI ratés
        }``

    Cas dégénérés
    -------------
    - GT vide ou sans caractère MUFI → ``coverage = 0`` (convention :
      pas de récompense gratuite).
    - Hyp vide + MUFI dans GT → ``coverage = 0``.
    - GT et hyp identiques avec MUFI → ``coverage = 1``.
    """
    ref = reference or ""
    hyp = hypothesis or ""
    extra: Optional[frozenset[str]] = (
        frozenset(c for c in custom_chars if c) if custom_chars else None
    )

    # 1. Identifier les positions MUFI dans la GT
    mufi_positions = [i for i, ch in enumerate(ref) if is_mufi_char(ch, extra)]
    n_total = len(mufi_positions)

    if n_total == 0:
        return {
            "n_mufi_chars_reference": 0,
            "n_mufi_chars_preserved": 0,
            "coverage": 0.0,
            "per_char": {},
            "missed_chars": [],
        }

    # 2. Aligner via la distance de Levenshtein (audit F4/F14 :
    #    alignement minimal cohérent avec le CER, plus difflib).
    correct_positions: set[int] = set()
    for op in Levenshtein.opcodes(ref, hyp):
        if op.tag == "equal":
            correct_positions.update(range(op.src_start, op.src_end))

    # 3. Compter par caractère
    per_char_total: dict[str, int] = {}
    per_char_preserved: dict[str, int] = {}
    missed: list[str] = []
    for i in mufi_positions:
        ch = ref[i]
        per_char_total[ch] = per_char_total.get(ch, 0) + 1
        if i in correct_positions:
            per_char_preserved[ch] = per_char_preserved.get(ch, 0) + 1
        else:
            missed.append(ch)

    n_preserved = sum(per_char_preserved.values())
    per_char = {
        ch: {
            "total": per_char_total[ch],
            "preserved": per_char_preserved.get(ch, 0),
            "coverage": (
                per_char_preserved.get(ch, 0) / per_char_total[ch]
                if per_char_total[ch] > 0
                else 0.0
            ),
        }
        for ch in sorted(per_char_total)
    }

    return {
        "n_mufi_chars_reference": n_total,
        "n_mufi_chars_preserved": n_preserved,
        "coverage": n_preserved / n_total,
        "per_char": per_char,
        "missed_chars": missed,
    }


def mufi_coverage(
    reference: Optional[str], hypothesis: Optional[str],
) -> float:
    """Raccourci : retourne la couverture MUFI globale ∈ [0, 1]."""
    return compute_mufi_coverage(reference, hypothesis)["coverage"]


# ──────────────────────────────────────────────────────────────────────────
# Enregistrement dans le registre typé (Sprint 34)
# ──────────────────────────────────────────────────────────────────────────


@register_metric(
    name="mufi_coverage",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description=(
        "Taux de caractères MUFI (Medieval Unicode Font Initiative) "
        "de la GT correctement restitués dans l'OCR. Critère "
        "éditorial central pour les médiévistes."
    ),
    higher_is_better=True,
    tags={"text", "mufi", "philology", "medieval"},
)
def _registered_mufi_coverage(reference: str, hypothesis: str) -> float:
    return mufi_coverage(reference, hypothesis)


__all__ = [
    "is_mufi_char",
    "compute_mufi_coverage",
    "mufi_coverage",
]
