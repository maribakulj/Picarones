"""Scores de reconnaissance des ligatures et des diacritiques.

Ces métriques sont spécifiques aux documents patrimoniaux (manuscrits, imprimés
anciens) où ligatures et diacritiques jouent un rôle paléographique essentiel.

Ligatures
---------
Caractères encodés comme une séquence unique dans Unicode mais représentant
deux ou plusieurs glyphes fusionnés : ﬁ (fi), ﬂ (fl), œ, æ, etc.

Pour chaque ligature présente dans le GT, on vérifie si l'OCR a produit
soit le caractère Unicode équivalent, soit la séquence décomposée équivalente.

Diacritiques
-----------
Accents, cédilles, trémas et autres signes diacritiques. Pour chaque caractère
accentué dans le GT, on vérifie si l'OCR a conservé le diacritique ou l'a
remplacé par la lettre de base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import unicodedata


# ---------------------------------------------------------------------------
# Tables de ligatures (char ligature → séquences équivalentes acceptées)
# ---------------------------------------------------------------------------

#: Table principale des ligatures et leurs équivalents acceptés.
#: Clé = caractère ligature Unicode ; valeur = liste de séquences équivalentes.
LIGATURE_TABLE: dict[str, list[str]] = {
    # Ligatures typographiques latines (Unicode Letterlike Symbols / Alphabetic Presentation Forms)
    "\uFB00": ["ff"],           # ﬀ ff
    "\uFB01": ["fi"],           # ﬁ fi
    "\uFB02": ["fl"],           # ﬂ fl
    "\uFB03": ["ffi"],          # ﬃ ffi
    "\uFB04": ["ffl"],          # ﬄ ffl
    "\uFB05": ["st", "\u017Ft"], # ﬅ st / ſt
    "\uFB06": ["st"],           # ﬆ st (variante)
    # Ligatures latines patrimoniales (Unicode Latin Extended Additional)
    "\u0153": ["oe"],           # œ oe
    "\u00E6": ["ae"],           # æ ae
    "\u0152": ["OE"],           # Œ OE
    "\u00C6": ["AE"],           # Æ AE
    # Abréviations latines / médiévales
    "\uA751": ["per", "p\u0332"],  # ꝑ per / p̲
    "\uA753": ["pro"],          # ꝓ pro
    "\uA757": ["que"],          # ꝗ que
    # Ligatures germaniques
    "\u00DF": ["ss"],           # ß ss
    "\u1E9E": ["SS"],           # ẞ SS
}

# Ensemble de toutes les ligatures pour recherche rapide
_ALL_LIGATURES: frozenset[str] = frozenset(LIGATURE_TABLE)

# Mapping inverse : séquence → ligature
_SEQ_TO_LIGATURE: dict[str, str] = {}
for _lig, _seqs in LIGATURE_TABLE.items():
    for _seq in _seqs:
        _SEQ_TO_LIGATURE[_seq] = _lig


# ---------------------------------------------------------------------------
# Table des caractères diacritiques
# ---------------------------------------------------------------------------

def _build_diacritic_map() -> dict[str, str]:
    """Construit automatiquement la table diacritique depuis l'Unicode."""
    table: dict[str, str] = {}
    for codepoint in range(0x00C0, 0x0250):  # Latin Étendu A + B
        ch = chr(codepoint)
        nfd = unicodedata.normalize("NFD", ch)
        if len(nfd) > 1:  # le caractère est décomposable
            base = nfd[0]  # lettre de base
            if base.isalpha() and base != ch:
                table[ch] = base
    # Compléments manuels
    table.update({
        "\u0107": "c",  # ć
        "\u0119": "e",  # ę
        "\u0142": "l",  # ł
        "\u0144": "n",  # ń
        "\u015B": "s",  # ś
        "\u017A": "z",  # ź
        "\u017C": "z",  # ż
    })
    return table


DIACRITIC_MAP: dict[str, str] = _build_diacritic_map()
_ALL_DIACRITICS: frozenset[str] = frozenset(DIACRITIC_MAP)

# Ligatures qui NE sont PAS des diacritiques (pour éviter les doublons)
_LIGATURE_SET: frozenset[str] = frozenset(LIGATURE_TABLE)


# ---------------------------------------------------------------------------
# Résultats structurés
# ---------------------------------------------------------------------------

@dataclass
class LigatureScore:
    """Score de reconnaissance des ligatures pour une paire (GT, OCR)."""

    total_in_gt: int = 0
    """Nombre de ligatures présentes dans le GT."""
    correctly_recognized: int = 0
    """Nombre de ligatures correctement transcrites (unicode ou équivalent)."""
    score: float = 0.0
    """Taux de reconnaissance = correctly_recognized / total_in_gt. 1.0 si total=0."""
    per_ligature: dict[str, dict] = field(default_factory=dict)
    """Détail par ligature : {'ﬁ': {'gt_count': 5, 'ocr_correct': 3, 'score': 0.6}}"""

    def as_dict(self) -> dict:
        return {
            "total_in_gt": self.total_in_gt,
            "correctly_recognized": self.correctly_recognized,
            "score": round(self.score, 4),
            "per_ligature": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in self.per_ligature.items()
            },
        }


@dataclass
class DiacriticScore:
    """Score de conservation des diacritiques pour une paire (GT, OCR)."""

    total_in_gt: int = 0
    """Nombre de caractères accentués dans le GT."""
    correctly_recognized: int = 0
    """Nombre de diacritiques correctement conservés."""
    score: float = 0.0
    """Taux de conservation = correctly_recognized / total_in_gt. 1.0 si total=0."""
    per_diacritic: dict[str, dict] = field(default_factory=dict)
    """Détail par caractère diacritique."""

    def as_dict(self) -> dict:
        return {
            "total_in_gt": self.total_in_gt,
            "correctly_recognized": self.correctly_recognized,
            "score": round(self.score, 4),
            "per_diacritic": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in self.per_diacritic.items()
            },
        }


# ---------------------------------------------------------------------------
# Calcul des scores
# ---------------------------------------------------------------------------

def compute_ligature_score(ground_truth: str, hypothesis: str) -> LigatureScore:
    """Calcule le score de reconnaissance des ligatures.

    Pour chaque ligature dans le GT, on vérifie si l'OCR a produit :
    - Exactement le même caractère ligature Unicode (ex. ﬁ → ﬁ)
    - Ou la séquence de lettres équivalente (ex. ﬁ → fi)

    Les deux sont considérés comme corrects — ce qui correspond à la pratique
    éditoriale patrimoniaux (certains éditeurs développent les ligatures).

    Parameters
    ----------
    ground_truth:
        Texte de référence.
    hypothesis:
        Texte produit par l'OCR.

    Returns
    -------
    LigatureScore
    """
    if not ground_truth:
        return LigatureScore(score=1.0)

    # Construire un index de position dans l'hypothèse pour recherche rapide
    hyp_norm = unicodedata.normalize("NFC", hypothesis)
    gt_norm = unicodedata.normalize("NFC", ground_truth)

    per_lig: dict[str, dict] = {}
    total = 0
    correct = 0

    # Trouver toutes les ligatures dans le GT
    i = 0
    while i < len(gt_norm):
        ch = gt_norm[i]
        if ch in _ALL_LIGATURES:
            total += 1
            equivalents = [ch] + LIGATURE_TABLE[ch]  # unicode direct ou séquences équivalentes

            # Vérifier si la position correspondante dans l'OCR contient l'équivalent
            is_correct = _check_char_at_context(gt_norm, hyp_norm, i, ch, equivalents)
            if is_correct:
                correct += 1

            if ch not in per_lig:
                per_lig[ch] = {"gt_count": 0, "ocr_correct": 0, "score": 0.0}
            per_lig[ch]["gt_count"] += 1
            if is_correct:
                per_lig[ch]["ocr_correct"] += 1
        i += 1

    # Calculer les scores individuels
    for lig_data in per_lig.values():
        lig_data["score"] = (
            lig_data["ocr_correct"] / lig_data["gt_count"]
            if lig_data["gt_count"] > 0
            else 1.0
        )

    score = correct / total if total > 0 else 1.0
    return LigatureScore(
        total_in_gt=total,
        correctly_recognized=correct,
        score=score,
        per_ligature=per_lig,
    )


def compute_diacritic_score(ground_truth: str, hypothesis: str) -> DiacriticScore:
    """Calcule le score de conservation des diacritiques.

    Pour chaque caractère accentué dans le GT, on vérifie si l'OCR a produit
    le même caractère (conservation) ou a substitué la lettre de base (perte).
    On accepte aussi les formes NFD équivalentes.

    Parameters
    ----------
    ground_truth:
        Texte de référence.
    hypothesis:
        Texte produit par l'OCR.

    Returns
    -------
    DiacriticScore
    """
    if not ground_truth:
        return DiacriticScore(score=1.0)

    gt_norm = unicodedata.normalize("NFC", ground_truth)
    hyp_norm = unicodedata.normalize("NFC", hypothesis)

    per_diac: dict[str, dict] = {}
    total = 0
    correct = 0

    # Utiliser difflib pour l'alignement
    import difflib
    matcher = difflib.SequenceMatcher(None, gt_norm, hyp_norm, autojunk=False)
    gt_to_hyp: dict[int, Optional[int]] = {}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                gt_to_hyp[i1 + k] = j1 + k
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                gt_to_hyp[i1 + k] = j1 + k
        else:
            # delete ou replace de longueurs différentes
            for k in range(i1, i2):
                gt_to_hyp[k] = None

    for i, ch in enumerate(gt_norm):
        if ch in _ALL_DIACRITICS and ch not in _LIGATURE_SET:
            total += 1
            hyp_pos = gt_to_hyp.get(i)
            is_correct = False
            if hyp_pos is not None and hyp_pos < len(hyp_norm):
                hyp_ch = hyp_norm[hyp_pos]
                is_correct = (hyp_ch == ch)
            if is_correct:
                correct += 1

            if ch not in per_diac:
                per_diac[ch] = {"gt_count": 0, "ocr_correct": 0, "score": 0.0}
            per_diac[ch]["gt_count"] += 1
            if is_correct:
                per_diac[ch]["ocr_correct"] += 1

    for diac_data in per_diac.values():
        diac_data["score"] = (
            diac_data["ocr_correct"] / diac_data["gt_count"]
            if diac_data["gt_count"] > 0
            else 1.0
        )

    score = correct / total if total > 0 else 1.0
    return DiacriticScore(
        total_in_gt=total,
        correctly_recognized=correct,
        score=score,
        per_diacritic=per_diac,
    )


def _check_char_at_context(
    gt: str,
    hyp: str,
    gt_pos: int,
    gt_char: str,
    equivalents: list[str],
) -> bool:
    """Vérifie si la position correspondante dans l'hypothèse contient un équivalent."""
    # Approche simple : chercher si l'hypothèse contient le caractère ou son équivalent
    # dans une fenêtre autour de la position estimée
    for equiv in equivalents:
        if equiv in hyp:
            return True
    return False


def aggregate_ligature_scores(scores: list[LigatureScore]) -> dict:
    """Agrège les scores de ligatures sur un corpus."""
    total_gt = sum(s.total_in_gt for s in scores)
    total_correct = sum(s.correctly_recognized for s in scores)
    score = total_correct / total_gt if total_gt > 0 else 1.0

    # Agrégation par ligature
    per_lig: dict[str, dict] = {}
    for s in scores:
        for lig, data in s.per_ligature.items():
            if lig not in per_lig:
                per_lig[lig] = {"gt_count": 0, "ocr_correct": 0}
            per_lig[lig]["gt_count"] += data["gt_count"]
            per_lig[lig]["ocr_correct"] += data["ocr_correct"]
    for lig_data in per_lig.values():
        lig_data["score"] = (
            lig_data["ocr_correct"] / lig_data["gt_count"]
            if lig_data["gt_count"] > 0 else 1.0
        )

    return {
        "score": round(score, 4),
        "total_in_gt": total_gt,
        "correctly_recognized": total_correct,
        "per_ligature": per_lig,
    }


def aggregate_diacritic_scores(scores: list[DiacriticScore]) -> dict:
    """Agrège les scores diacritiques sur un corpus."""
    total_gt = sum(s.total_in_gt for s in scores)
    total_correct = sum(s.correctly_recognized for s in scores)
    score = total_correct / total_gt if total_gt > 0 else 1.0
    return {
        "score": round(score, 4),
        "total_in_gt": total_gt,
        "correctly_recognized": total_correct,
    }
