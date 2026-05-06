"""Marqueurs typographiques de l'imprimé ancien (XVIᵉ-XVIIIᵉ).

Sprint 58 — Étape 3 / extension philologique du plan d'évolution
2026.

Pourquoi ce module
------------------
Les Sprints 56 (abréviations Capelli) et 57 (couverture MUFI) sont
orientés **médiéval scribal**.  Mais Picarones doit aussi servir
les éditeurs d'**imprimés anciens** (XVIᵉ-XVIIIᵉ siècles), pour
qui les marqueurs caractéristiques ne sont pas scribaux mais
**typographiques** : ligatures composées (ﬁ, ﬂ, ﬀ, ﬃ, ﬄ, ﬅ),
s long (ſ), i sans point (ı), esperluette (&), tildes nasaux
indiquant une abréviation (ã = an/am, õ = on/om).

Distinction avec MUFI/abbreviations
------------------------------------
- ``mufi.py`` (Sprint 57) : caractères médiévaux scribaux
  (Capelli + lettres þ ð ƿ + PUA MUFI).
- ``abbreviations.py`` (Sprint 56) : signes d'abréviation latins
  scribaux médiévaux (ꝑ ꝓ ⁊ + tildes scribaux).
- ``early_modern_typography.py`` (ce module) : marqueurs
  **typographiques** de la composition imprimée ancienne.

Les ligatures ﬁ et ﬂ sont communes aux deux univers (médiéval et
imprimé ancien) ; le choix du module à utiliser dépend du **corpus**
et de l'angle d'analyse éditoriale, pas du caractère pris isolément.

Catégorisation
--------------
Les marqueurs sont classés en cinq catégories pour permettre un
breakdown éditorial :

1. ``ligatures`` : ﬁ ﬂ ﬀ ﬃ ﬄ ﬅ
2. ``long_s`` : ſ
3. ``dotless_i`` : ı
4. ``ampersand`` : & (esperluette typographique)
5. ``nasal_tildes`` : ã õ ũ ñ ē ī (abréviation par tilde nasal)

``compute_early_modern_metrics`` retourne le taux de préservation
par catégorie + global.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Optional

from picarones.evaluation.metric_registry import register_metric
from picarones.domain.artifacts import ArtifactType

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Marqueurs typographiques imprimé ancien
# ──────────────────────────────────────────────────────────────────────────

# Ligatures typographiques héritées de l'incunable (XVᵉ) et toujours
# courantes jusqu'au XVIIIᵉ avant la normalisation typographique.
LIGATURES: frozenset[str] = frozenset({
    "ﬀ",  # U+FB00 ff
    "ﬁ",  # U+FB01 fi
    "ﬂ",  # U+FB02 fl
    "ﬃ",  # U+FB03 ffi
    "ﬄ",  # U+FB04 ffl
    "ﬅ",  # U+FB05 long s + t
    "ﬆ",  # U+FB06 st
})

# S long : Latin Extended-A.  Caractéristique de la typographie
# antérieure à 1800.
LONG_S: frozenset[str] = frozenset({"ſ"})  # U+017F

# i sans point : utilisé en typographie ancienne, parfois confondu
# avec un l ou un 1 par les OCR modernes.
DOTLESS_I: frozenset[str] = frozenset({"ı"})  # U+0131

# Esperluette typographique : "&" remplace fréquemment "et" dans
# les imprimés ; sa préservation discrimine un OCR diplomatique
# d'un OCR modernisant.
AMPERSAND: frozenset[str] = frozenset({"&"})

# Tildes nasaux : pré-composés (ñ ã ẽ ĩ õ ũ) ou séquences
# lettre + U+0303 combinant.  En imprimé ancien, ã = an/am abrégé,
# õ = on/om, etc.  Distinction avec les tildes scribaux médiévaux
# (Sprint 56) : ici on cible les **pré-composés** ou séquences sur
# des voyelles (le scribal médiéval cible plutôt p̃ q̃).
NASAL_TILDE_PRECOMPOSED: frozenset[str] = frozenset({
    "ã", "Ã",  # U+00E3 / U+00C3
    "ñ", "Ñ",  # U+00F1 / U+00D1
    "õ", "Õ",  # U+00F5 / U+00D5
    "ũ", "Ũ",  # U+0169 / U+0168
    "ẽ", "Ẽ",  # U+1EBD / U+1EBC
    "ĩ", "Ĩ",  # U+0129 / U+0128
})

# Voyelles susceptibles de porter un tilde combinant pour former
# un tilde nasal (couvre les écritures NFD non pré-composées).
_NASAL_TILDE_VOWELS: frozenset[str] = frozenset(
    "aeiouAEIOU"
)
_COMBINING_TILDE = "̃"


# Catégorisation : nom → set de caractères pré-composés ou séquences.
_CATEGORIES: dict[str, frozenset[str]] = {
    "ligatures": LIGATURES,
    "long_s": LONG_S,
    "dotless_i": DOTLESS_I,
    "ampersand": AMPERSAND,
    "nasal_tildes": NASAL_TILDE_PRECOMPOSED,
}


# ──────────────────────────────────────────────────────────────────────────
# Détection des marqueurs dans la GT
# ──────────────────────────────────────────────────────────────────────────


def _detect_markers(text: str) -> list[tuple[int, str, str]]:
    """Retourne les positions des marqueurs typographiques dans
    ``text``.

    Forme de sortie : ``[(index, marker, category), ...]`` dans
    l'ordre d'apparition.  Pour les tildes nasaux non
    pré-composés, on détecte les séquences ``voyelle + U+0303`` et
    on retourne l'index de la voyelle.
    """
    if not text:
        return []
    found: list[tuple[int, str, str]] = []
    i = 0
    while i < len(text):
        ch = text[i]
        # Cas 1 : marqueur pré-composé dans une catégorie
        category = _category_of_char(ch)
        if category is not None:
            found.append((i, ch, category))
            i += 1
            continue
        # Cas 2 : voyelle + tilde combinant → nasal_tildes
        if (
            ch in _NASAL_TILDE_VOWELS
            and i + 1 < len(text)
            and text[i + 1] == _COMBINING_TILDE
        ):
            seq = ch + _COMBINING_TILDE
            found.append((i, seq, "nasal_tildes"))
            i += 2
            continue
        i += 1
    return found


def _category_of_char(ch: str) -> Optional[str]:
    """Retourne la catégorie d'un caractère typographique ou
    ``None`` s'il n'est pas reconnu."""
    for cat, chars in _CATEGORIES.items():
        if ch in chars:
            return cat
    return None


# ──────────────────────────────────────────────────────────────────────────
# Calcul de la préservation par catégorie
# ──────────────────────────────────────────────────────────────────────────


def compute_early_modern_metrics(
    reference: Optional[str],
    hypothesis: Optional[str],
) -> dict:
    """Mesure la préservation des marqueurs typographiques de
    l'imprimé ancien dans l'OCR.

    Stratégie d'alignement
    ----------------------
    Pour chaque marqueur identifié dans la GT à la position ``i``,
    on vérifie si l'OCR l'a préservé en utilisant l'alignement
    caractère par caractère via ``difflib.SequenceMatcher`` (même
    méthode que les Sprints 55/57) :

    - Marqueur **mono-caractère** (ﬁ, ſ, ı, &, ã…) : la position
      ``i`` est-elle dans un opcode ``equal`` ?
    - Marqueur **bi-caractère** (voyelle + U+0303) : les positions
      ``i`` et ``i+1`` sont-elles toutes deux dans un opcode
      ``equal`` ?

    Returns
    -------
    dict
        ``{
            "n_markers_reference":  int,
            "n_markers_preserved":  int,
            "global_preservation":  float,    # ∈ [0, 1]
            "per_category": {
                category: {"total", "preserved", "preservation"}
            },
            "missed_markers": [{"index", "marker", "category"}, ...],
        }``

    Cas dégénérés : GT vide ou sans marqueur → tous compteurs à 0,
    ``global_preservation = 0``.
    """
    ref = reference or ""
    hyp = hypothesis or ""

    # Forme NFD pour reconnaître les tildes nasaux décomposés (ã =
    # 'a' + U+0303) côté GT — on conserve toutefois la forme passée
    # pour les indices rapportés dans missed_markers.
    markers = _detect_markers(ref)
    n_total = len(markers)

    if n_total == 0:
        return {
            "n_markers_reference": 0,
            "n_markers_preserved": 0,
            "global_preservation": 0.0,
            "per_category": {},
            "missed_markers": [],
        }

    # Aligner GT/hyp et récupérer le set des positions GT couvertes
    # par un opcode "equal".
    matcher = SequenceMatcher(a=ref, b=hyp, autojunk=False)
    correct_positions: set[int] = set()
    for op, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if op == "equal":
            correct_positions.update(range(i1, i2))

    per_cat_total: dict[str, int] = {}
    per_cat_preserved: dict[str, int] = {}
    n_preserved = 0
    missed: list[dict] = []

    for index, marker, category in markers:
        per_cat_total[category] = per_cat_total.get(category, 0) + 1
        # Marqueur préservé si toutes ses positions GT sont dans
        # un opcode "equal".
        marker_len = len(marker)
        positions_ok = all(
            (index + k) in correct_positions for k in range(marker_len)
        )
        if positions_ok:
            per_cat_preserved[category] = (
                per_cat_preserved.get(category, 0) + 1
            )
            n_preserved += 1
        else:
            missed.append({
                "index": index,
                "marker": marker,
                "category": category,
            })

    per_category = {
        cat: {
            "total": per_cat_total[cat],
            "preserved": per_cat_preserved.get(cat, 0),
            "preservation": (
                per_cat_preserved.get(cat, 0) / per_cat_total[cat]
                if per_cat_total[cat] > 0
                else 0.0
            ),
        }
        for cat in sorted(per_cat_total)
    }

    return {
        "n_markers_reference": n_total,
        "n_markers_preserved": n_preserved,
        "global_preservation": n_preserved / n_total,
        "per_category": per_category,
        "missed_markers": missed,
    }


def early_modern_preservation(
    reference: Optional[str], hypothesis: Optional[str],
) -> float:
    """Raccourci : taux global de préservation des marqueurs
    typographiques de l'imprimé ancien."""
    return compute_early_modern_metrics(
        reference, hypothesis,
    )["global_preservation"]


# ──────────────────────────────────────────────────────────────────────────
# Helpers exposés
# ──────────────────────────────────────────────────────────────────────────


def detect_markers(text: Optional[str]) -> list[tuple[int, str, str]]:
    """Wrapper public sur ``_detect_markers`` (acceptant ``None``)."""
    return _detect_markers(text or "")


def get_category(char: str) -> Optional[str]:
    """Retourne la catégorie typographique d'un caractère
    (``ligatures``, ``long_s``, ``dotless_i``, ``ampersand``,
    ``nasal_tildes``) ou ``None``.

    Pour un tilde combinant suivi d'une voyelle, l'utilisateur doit
    utiliser ``detect_markers`` qui gère les séquences.
    """
    return _category_of_char(char[0]) if char else None


# ──────────────────────────────────────────────────────────────────────────
# Enregistrement dans le registre typé (Sprint 34)
# ──────────────────────────────────────────────────────────────────────────


@register_metric(
    name="early_modern_preservation",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description=(
        "Taux de préservation des marqueurs typographiques de "
        "l'imprimé ancien (XVIᵉ-XVIIIᵉ) : ligatures ﬁ ﬂ ﬀ, s long ſ, "
        "i sans point ı, esperluette &, tildes nasaux ã õ. Critère "
        "éditorial pour les éditions diplomatiques d'imprimés anciens."
    ),
    higher_is_better=True,
    tags={"text", "typography", "early_modern", "philology"},
)
def _registered_early_modern(reference: str, hypothesis: str) -> float:
    return early_modern_preservation(reference, hypothesis)


__all__ = [
    "LIGATURES",
    "LONG_S",
    "DOTLESS_I",
    "AMPERSAND",
    "NASAL_TILDE_PRECOMPOSED",
    "detect_markers",
    "get_category",
    "compute_early_modern_metrics",
    "early_modern_preservation",
]
