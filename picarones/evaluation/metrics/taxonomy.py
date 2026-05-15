"""Taxonomie des erreurs OCR — classification automatique (classes 1 à 9).

Chaque erreur identifiée par l'alignement GT↔OCR est catégorisée selon
la taxonomie Picarones :

| Classe | Nom               | Description                                        |
|--------|-------------------|----------------------------------------------------|
| 1      | visual_confusion  | Confusion morphologique (rn/m, l/1, O/0, u/n…)    |
| 2      | diacritic_error   | Diacritique absent, incorrect ou ajouté            |
| 3      | case_error        | Erreur de casse uniquement (A/a)                   |
| 4      | ligature_error    | Ligature non résolue ou mal résolue               |
| 5      | abbreviation_error| Abréviation médiévale non développée               |
| 6      | hapax             | Mot introuvable dans tout lexique                  |
| 7      | segmentation_error| Fusion ou fragmentation de tokens (mots/lignes)    |
| 8      | oov_character     | Caractère hors-vocabulaire du moteur               |
| 9      | lacuna            | Texte présent dans le GT absent de l'OCR           |
| 10     | over_normalization| Sur-normalisation LLM (voir evaluation/metrics/)   |

Note : la classe 10 est calculée par picarones/evaluation/metrics/over_normalization.py.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein


# ---------------------------------------------------------------------------
# Tables de référence pour la classification
# ---------------------------------------------------------------------------

#: Confusions visuelles bien connues en OCR (caractères morphologiquement proches)
VISUAL_CONFUSIONS: dict[frozenset, str] = {}
_VISUAL_PAIRS: list[tuple[str, str]] = [
    # Minuscules
    ("r", "n"), ("rn", "m"), ("l", "1"), ("l", "i"), ("l", "|"),
    ("O", "0"), ("O", "o"), ("u", "n"), ("n", "u"), ("v", "u"),
    ("c", "e"), ("e", "c"), ("a", "o"), ("o", "a"),
    ("f", "ſ"), ("ſ", "f"), ("f", "t"),
    ("h", "li"), ("h", "lı"),
    ("m", "rn"), ("m", "in"),
    ("d", "cl"), ("d", "a"),
    ("q", "g"), ("p", "q"),
    # Majuscules ↔ minuscules homographes (classe 1, pas classe 3)
    ("I", "l"), ("I", "1"),
    # Chiffres
    ("1", "I"), ("1", "l"), ("0", "O"),
    # Ponctuation
    (".", ","), (",", "."),
]
for _a, _b in _VISUAL_PAIRS:
    VISUAL_CONFUSIONS[frozenset({_a, _b})] = f"{_a}/{_b}"

#: Couples de ligatures pour la détection des erreurs de ligatures
from picarones.evaluation.metrics.char_scores import LIGATURE_TABLE, DIACRITIC_MAP  # noqa: E402

# Caractères hors-ASCII présumés hors-vocabulaire (alphabet non latin de base)
_LATIN_BASIC = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    " \t\n.,;:!?-_'\"«»()[]{}/@#%&*+=/\\|<>~^")


# ---------------------------------------------------------------------------
# Résultat structuré
# ---------------------------------------------------------------------------

@dataclass
class TaxonomyResult:
    """Résultat de la classification taxonomique des erreurs pour un document."""

    counts: dict[str, int] = field(default_factory=dict)
    """Nombre d'erreurs par classe. Clés : 'visual_confusion', 'diacritic_error'…"""

    examples: dict[str, list[dict]] = field(default_factory=dict)
    """Exemples d'erreurs par classe (max 5 par classe).
    Format : [{'gt': 'chaîne', 'ocr': 'chaîne', 'position': int}]
    """

    total_errors: int = 0
    """Nombre total d'erreurs classifiées."""

    @property
    def class_distribution(self) -> dict[str, float]:
        """Distribution relative (0–1) par classe."""
        if not self.total_errors:
            return {}
        return {
            cls: round(cnt / self.total_errors, 4)
            for cls, cnt in self.counts.items()
        }

    def as_dict(self) -> dict:
        return {
            "counts": self.counts,
            "total_errors": self.total_errors,
            "class_distribution": self.class_distribution,
            "examples": {
                cls: exs[:3] for cls, exs in self.examples.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaxonomyResult":
        return cls(
            counts=data.get("counts", {}),
            examples=data.get("examples", {}),
            total_errors=data.get("total_errors", 0),
        )


# Noms des classes en ordre
ERROR_CLASSES = [
    "visual_confusion",
    "diacritic_error",
    "case_error",
    "ligature_error",
    "abbreviation_error",
    "hapax",
    "segmentation_error",
    "oov_character",
    "lacuna",
]


# ---------------------------------------------------------------------------
# Classification principale
# ---------------------------------------------------------------------------

def classify_errors(
    ground_truth: str,
    hypothesis: str,
    max_examples: int = 5,
) -> TaxonomyResult:
    """Classifie automatiquement les erreurs OCR dans une paire GT/OCR.

    L'alignement utilise la distance de **Levenshtein** au niveau mot
    (rapidfuzz, cohérent avec le WER) — audit scientifique F14.

    Auparavant l'alignement passait par ``difflib.SequenceMatcher``
    (Ratcliff–Obershelp, non minimal) et, dans un bloc ``replace`` de
    longueurs inégales, ne comptait QUE l'écart de longueur comme
    ``segmentation_error`` en **abandonnant la classification de toutes
    les substitutions** de ce bloc ; les mots insérés non hors-vocab
    n'étaient pas comptés du tout.  La distribution des 9 classes était
    donc systématiquement biaisée (segmentation sur-comptée, autres
    classes sous-comptées, ``total_errors`` sous-estimé).

    Désormais : chaque opération de l'alignement minimal est classée
    (``total_errors`` ≈ distance d'édition mot).  Une région de
    différences dont la concaténation sans espaces GT == hypothèse
    (casefold) est une vraie re-segmentation (classe 7) ; sinon les
    mots sont appariés (substitutions classées), le surplus GT →
    lacune (9), le surplus hypothèse → hors-vocab (8) ou hapax (6).

    Parameters
    ----------
    ground_truth:
        Texte de référence (vérité terrain).
    hypothesis:
        Texte produit par l'OCR.
    max_examples:
        Nombre maximal d'exemples conservés par classe.

    Returns
    -------
    TaxonomyResult
    """
    counts: dict[str, int] = {cls: 0 for cls in ERROR_CLASSES}
    examples: dict[str, list[dict]] = {cls: [] for cls in ERROR_CLASSES}
    total = 0

    if not ground_truth and not hypothesis:
        return TaxonomyResult(counts=counts, examples=examples, total_errors=0)

    # -----------------------------------------------------------------------
    # Niveau mot : détecter segmentation (classe 7) et lacunes (classe 9)
    # -----------------------------------------------------------------------
    gt_words = ground_truth.split()
    hyp_words = hypothesis.split()

    # Fusionner les opérations non-``equal`` consécutives en régions de
    # différence, pour pouvoir détecter une re-segmentation (espaces
    # déplacés) au niveau de la région plutôt que d'un seul opcode.
    ops = list(Levenshtein.opcodes(gt_words, hyp_words))
    regions: list[tuple[int, int, int, int]] = []
    for op in ops:
        if op.tag == "equal":
            continue
        i1, i2, j1, j2 = op.src_start, op.src_end, op.dest_start, op.dest_end
        if regions and regions[-1][1] == i1 and regions[-1][3] == j1:
            p = regions[-1]
            regions[-1] = (p[0], i2, p[2], j2)
        else:
            regions.append((i1, i2, j1, j2))

    for i1, i2, j1, j2 in regions:
        gt_seg = gt_words[i1:i2]
        hyp_seg = hyp_words[j1:j2]

        # Re-segmentation : le **nombre de tokens change** ET le contenu
        # concaténé est identique (« le dit » ↔ « ledit », « par ce
        # que » ↔ « parce que »).  La condition de changement de nombre
        # de tokens est essentielle : sans elle, une simple erreur de
        # casse sur n mots (« Bonjour Monde » ↔ « bonjour monde », même
        # nombre de tokens, concaténation égale en casefold) serait à
        # tort classée segmentation au lieu de case_error.
        gt_join = "".join(gt_seg).casefold()
        hyp_join = "".join(hyp_seg).casefold()
        if (
            gt_seg and hyp_seg
            and len(gt_seg) != len(hyp_seg)
            and gt_join == hyp_join
        ):
            n_seg = max(1, abs(len(gt_seg) - len(hyp_seg)))
            counts["segmentation_error"] += n_seg
            total += n_seg
            if len(examples["segmentation_error"]) < max_examples:
                examples["segmentation_error"].append({
                    "gt": " ".join(gt_seg),
                    "ocr": " ".join(hyp_seg),
                    "position": i1,
                })
            continue

        # Sinon : apparier mot-à-mot le préfixe commun ; classer chaque
        # substitution ; le surplus GT = lacune, le surplus hyp = OOV/hapax.
        paired = min(len(gt_seg), len(hyp_seg))
        for k in range(paired):
            gt_w, hyp_w = gt_seg[k], hyp_seg[k]
            if gt_w != hyp_w:
                _classify_word_error(gt_w, hyp_w, counts, examples, max_examples)
                total += 1
        for w in gt_seg[paired:]:
            counts["lacuna"] += 1
            total += 1
            if len(examples["lacuna"]) < max_examples:
                examples["lacuna"].append({"gt": w, "ocr": "", "position": i1})
        for w in hyp_seg[paired:]:
            cls = "oov_character" if _is_oov_word(w) else "hapax"
            counts[cls] += 1
            total += 1
            if len(examples[cls]) < max_examples:
                examples[cls].append({"gt": "", "ocr": w, "position": j1})

    return TaxonomyResult(
        counts=counts,
        examples=examples,
        total_errors=total,
    )


def _classify_word_error(
    gt_word: str,
    hyp_word: str,
    counts: dict[str, int],
    examples: dict[str, list[dict]],
    max_examples: int,
) -> None:
    """Classifie l'erreur entre deux mots non-identiques."""
    # Classe 3 : erreur de casse seule
    if gt_word.casefold() == hyp_word.casefold() and gt_word != hyp_word:
        counts["case_error"] += 1
        if len(examples["case_error"]) < max_examples:
            examples["case_error"].append({"gt": gt_word, "ocr": hyp_word})
        return

    # Classe 4 : erreur de ligature
    gt_norm = unicodedata.normalize("NFC", gt_word)
    hyp_norm = unicodedata.normalize("NFC", hyp_word)
    if _is_ligature_error(gt_norm, hyp_norm):
        counts["ligature_error"] += 1
        if len(examples["ligature_error"]) < max_examples:
            examples["ligature_error"].append({"gt": gt_word, "ocr": hyp_word})
        return

    # Classe 5 : erreur d'abréviation (présence de ꝑ, ꝓ, ꝗ dans le GT)
    if _is_abbreviation_error(gt_norm, hyp_norm):
        counts["abbreviation_error"] += 1
        if len(examples["abbreviation_error"]) < max_examples:
            examples["abbreviation_error"].append({"gt": gt_word, "ocr": hyp_word})
        return

    # Classe 2 : erreur diacritique
    if _is_diacritic_error(gt_norm, hyp_norm):
        counts["diacritic_error"] += 1
        if len(examples["diacritic_error"]) < max_examples:
            examples["diacritic_error"].append({"gt": gt_word, "ocr": hyp_word})
        return

    # Classe 1 : confusion visuelle (comparaison char par char)
    if _is_visual_confusion(gt_norm, hyp_norm):
        counts["visual_confusion"] += 1
        if len(examples["visual_confusion"]) < max_examples:
            examples["visual_confusion"].append({"gt": gt_word, "ocr": hyp_word})
        return

    # Classe 8 : caractère hors-vocabulaire
    if _is_oov_word(hyp_word):
        counts["oov_character"] += 1
        if len(examples["oov_character"]) < max_examples:
            examples["oov_character"].append({"gt": gt_word, "ocr": hyp_word})
        return

    # Classe 6 : hapax (erreur résiduelle non classifiable)
    counts["hapax"] += 1
    if len(examples["hapax"]) < max_examples:
        examples["hapax"].append({"gt": gt_word, "ocr": hyp_word})


def _is_ligature_error(gt: str, hyp: str) -> bool:
    """Vrai si la différence implique une ligature Unicode."""
    # GT contient une ligature que l'OCR a décomposée, ou vice versa
    for lig, seqs in LIGATURE_TABLE.items():
        if lig in gt:
            for seq in seqs:
                if seq in hyp and lig not in hyp:
                    return True
        for seq in seqs:
            if seq in gt and lig in hyp:
                return True
    return False


def _is_abbreviation_error(gt: str, hyp: str) -> bool:
    """Vrai si le GT contient un caractère d'abréviation médiévale."""
    abbreviation_chars = "\uA751\uA753\uA757"  # ꝑ ꝓ ꝗ
    return any(c in gt for c in abbreviation_chars)


def _is_diacritic_error(gt: str, hyp: str) -> bool:
    """Vrai si la différence est principalement due à des diacritiques."""
    # Comparer les formes sans diacritiques
    def strip_diacritics(text: str) -> str:
        nfd = unicodedata.normalize("NFD", text)
        return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

    gt_stripped = strip_diacritics(gt)
    hyp_stripped = strip_diacritics(hyp)
    # Si les mots sont identiques sans diacritiques → erreur diacritique
    if gt_stripped.casefold() == hyp_stripped.casefold() and gt != hyp:
        return True
    # Si le GT contient des diacritiques que l'OCR a perdus et que les textes
    # sans diacritiques sont identiques (même longueur requise)
    gt_has_diac = any(c in DIACRITIC_MAP for c in gt)
    return gt_has_diac and len(gt) == len(hyp) and gt_stripped.casefold() == hyp_stripped.casefold()


def _is_visual_confusion(gt: str, hyp: str) -> bool:
    """Vrai si la différence implique des confusions visuelles connues."""
    if abs(len(gt) - len(hyp)) > 2:
        return False
    # Vérifier les paires de confusions connues
    for pair in VISUAL_CONFUSIONS:
        chars = list(pair)
        if len(chars) == 2:
            a, b = chars
            if a in gt and b in hyp and a not in hyp:
                return True
            if b in gt and a in hyp and b not in hyp:
                return True
    return False


def _is_oov_word(word: str) -> bool:
    """Vrai si le mot contient des caractères hors de l'alphabet latin de base."""
    return any(c not in _LATIN_BASIC and not c.isalpha() for c in word)


# ---------------------------------------------------------------------------
# Agrégation
# ---------------------------------------------------------------------------

def aggregate_taxonomy(results: list[TaxonomyResult]) -> dict:
    """Agrège les résultats taxonomiques sur un corpus."""
    combined: dict[str, int] = {cls: 0 for cls in ERROR_CLASSES}
    total = 0
    for r in results:
        for cls, cnt in r.counts.items():
            combined[cls] = combined.get(cls, 0) + cnt
        total += r.total_errors

    distribution = {
        cls: round(cnt / total, 4) if total > 0 else 0.0
        for cls, cnt in combined.items()
    }
    return {
        "counts": combined,
        "total_errors": total,
        "class_distribution": distribution,
    }
