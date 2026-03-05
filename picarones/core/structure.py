"""Analyse structurelle des résultats OCR.

Mesures
-------
- **Taux de fusion de lignes** : l'OCR produit moins de lignes que le GT
  (plusieurs lignes GT fusionnées en une seule).
- **Taux de fragmentation** : l'OCR produit plus de lignes que le GT
  (une ligne GT découpée en plusieurs).
- **Score d'ordre de lecture** : corrélation entre l'ordre des mots GT et OCR,
  approximé par la longueur de la sous-séquence commune la plus longue (LCS).
- **Taux de conservation des paragraphes** : respect des sauts de paragraphe.

Ces métriques sont calculées indépendamment du contenu textuel — elles mesurent
la fidélité de la mise en page, pas la qualité des caractères.

Note : sans bounding boxes disponibles, l'analyse se base uniquement sur les
sauts de ligne présents dans les textes GT et OCR.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Optional


@dataclass
class StructureResult:
    """Résultat de l'analyse structurelle pour un document."""

    gt_line_count: int = 0
    """Nombre de lignes dans le GT."""
    ocr_line_count: int = 0
    """Nombre de lignes dans l'OCR."""

    line_fusion_count: int = 0
    """Nombre de fusions de lignes (GT lignes absorbées)."""
    line_fragmentation_count: int = 0
    """Nombre de fragmentations (GT lignes splittées)."""

    reading_order_score: float = 1.0
    """Score d'ordre de lecture [0, 1]. 1 = ordre parfait."""

    paragraph_conservation_score: float = 1.0
    """Score de conservation des paragraphes [0, 1]."""

    @property
    def line_fusion_rate(self) -> float:
        """Taux de fusion = fusions / lignes GT."""
        return self.line_fusion_count / self.gt_line_count if self.gt_line_count > 0 else 0.0

    @property
    def line_fragmentation_rate(self) -> float:
        """Taux de fragmentation = fragmentations / lignes GT."""
        return self.line_fragmentation_count / self.gt_line_count if self.gt_line_count > 0 else 0.0

    @property
    def line_accuracy(self) -> float:
        """Exactitude du nombre de lignes : 1 - |delta| / max(gt, ocr)."""
        if self.gt_line_count == 0 and self.ocr_line_count == 0:
            return 1.0
        max_lines = max(self.gt_line_count, self.ocr_line_count)
        delta = abs(self.gt_line_count - self.ocr_line_count)
        return max(0.0, 1.0 - delta / max_lines)

    def as_dict(self) -> dict:
        return {
            "gt_line_count": self.gt_line_count,
            "ocr_line_count": self.ocr_line_count,
            "line_fusion_count": self.line_fusion_count,
            "line_fragmentation_count": self.line_fragmentation_count,
            "line_fusion_rate": round(self.line_fusion_rate, 4),
            "line_fragmentation_rate": round(self.line_fragmentation_rate, 4),
            "line_accuracy": round(self.line_accuracy, 4),
            "reading_order_score": round(self.reading_order_score, 4),
            "paragraph_conservation_score": round(self.paragraph_conservation_score, 4),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructureResult":
        return cls(
            gt_line_count=data.get("gt_line_count", 0),
            ocr_line_count=data.get("ocr_line_count", 0),
            line_fusion_count=data.get("line_fusion_count", 0),
            line_fragmentation_count=data.get("line_fragmentation_count", 0),
            reading_order_score=data.get("reading_order_score", 1.0),
            paragraph_conservation_score=data.get("paragraph_conservation_score", 1.0),
        )


def analyze_structure(ground_truth: str, hypothesis: str) -> StructureResult:
    """Analyse la structure d'un document OCR comparée au GT.

    Parameters
    ----------
    ground_truth:
        Texte de référence (vérité terrain), avec sauts de ligne.
    hypothesis:
        Texte produit par l'OCR, avec sauts de ligne.

    Returns
    -------
    StructureResult
    """
    gt_lines = [l for l in ground_truth.splitlines() if l.strip()]
    ocr_lines = [l for l in hypothesis.splitlines() if l.strip()]

    n_gt = len(gt_lines)
    n_ocr = len(ocr_lines)

    # Fusions et fragmentations
    fusion_count, frag_count = _count_line_changes(gt_lines, ocr_lines)

    # Score d'ordre de lecture via LCS sur les mots
    reading_order = _reading_order_score(ground_truth, hypothesis)

    # Score de conservation des paragraphes (sauts de ligne vides = paragraphes)
    para_score = _paragraph_conservation_score(ground_truth, hypothesis)

    return StructureResult(
        gt_line_count=n_gt,
        ocr_line_count=n_ocr,
        line_fusion_count=fusion_count,
        line_fragmentation_count=frag_count,
        reading_order_score=reading_order,
        paragraph_conservation_score=para_score,
    )


def _count_line_changes(gt_lines: list[str], ocr_lines: list[str]) -> tuple[int, int]:
    """Compte les fusions et fragmentations de lignes via SequenceMatcher."""
    if not gt_lines or not ocr_lines:
        return 0, 0

    fusion_count = 0
    frag_count = 0

    # Aligner les lignes par contenu
    matcher = difflib.SequenceMatcher(
        None,
        [l.strip()[:30] for l in gt_lines],  # fingerprint court pour la comparaison
        [l.strip()[:30] for l in ocr_lines],
        autojunk=False,
    )

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            gt_len = i2 - i1
            ocr_len = j2 - j1
            if ocr_len < gt_len:
                # Moins de lignes OCR → fusions
                fusion_count += gt_len - ocr_len
            elif ocr_len > gt_len:
                # Plus de lignes OCR → fragmentations
                frag_count += ocr_len - gt_len
        elif tag == "delete":
            # Lignes GT supprimées dans l'OCR → lacunes (pas fusion/frag)
            pass
        elif tag == "insert":
            # Lignes insérées par l'OCR
            frag_count += j2 - j1

    return fusion_count, frag_count


def _reading_order_score(ground_truth: str, hypothesis: str) -> float:
    """Score d'ordre de lecture [0, 1] basé sur la LCS des mots.

    On calcule la longueur de la sous-séquence commune la plus longue (LCS)
    entre les listes de mots GT et OCR. Un score de 1 signifie que tous les
    mots communs apparaissent dans le même ordre.
    """
    gt_words = ground_truth.split()
    hyp_words = hypothesis.split()

    if not gt_words or not hyp_words:
        return 1.0

    # Utiliser SequenceMatcher pour approximer la LCS
    matcher = difflib.SequenceMatcher(None, gt_words, hyp_words, autojunk=False)
    # Ratio est 2 * nb_correspondances / (len_gt + len_ocr)
    # C'est un proxy raisonnable de l'ordre de lecture
    ratio = matcher.ratio()
    return round(ratio, 4)


def _paragraph_conservation_score(ground_truth: str, hypothesis: str) -> float:
    """Score de conservation des paragraphes [0, 1].

    Compte les sauts de paragraphe (lignes vides) dans le GT et mesure
    le taux de conservation dans l'OCR.
    """
    # Un saut de paragraphe = deux sauts de ligne consécutifs
    gt_paras = [p for p in ground_truth.split("\n\n") if p.strip()]
    ocr_paras = [p for p in hypothesis.split("\n\n") if p.strip()]

    n_gt_paras = len(gt_paras)
    if n_gt_paras <= 1:
        return 1.0  # pas de paragraphe distinct → score parfait

    n_ocr_paras = len(ocr_paras)
    delta = abs(n_gt_paras - n_ocr_paras)
    score = max(0.0, 1.0 - delta / n_gt_paras)
    return round(score, 4)


def aggregate_structure(results: list[StructureResult]) -> dict:
    """Agrège les résultats structurels sur un corpus."""
    if not results:
        return {}

    import statistics

    def _mean(values: list[float]) -> float:
        return round(statistics.mean(values), 4) if values else 0.0

    fusion_rates = [r.line_fusion_rate for r in results]
    frag_rates = [r.line_fragmentation_rate for r in results]
    reading_scores = [r.reading_order_score for r in results]
    para_scores = [r.paragraph_conservation_score for r in results]
    line_accuracies = [r.line_accuracy for r in results]

    return {
        "mean_line_fusion_rate": _mean(fusion_rates),
        "mean_line_fragmentation_rate": _mean(frag_rates),
        "mean_reading_order_score": _mean(reading_scores),
        "mean_paragraph_conservation": _mean(para_scores),
        "mean_line_accuracy": _mean(line_accuracies),
        "document_count": len(results),
    }
