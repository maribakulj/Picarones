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

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein


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
    gt_lines = [ln for ln in ground_truth.splitlines() if ln.strip()]
    ocr_lines = [ln for ln in hypothesis.splitlines() if ln.strip()]

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
    """Compte fusions/fragmentations de lignes via l'alignement minimal.

    Audit F4 — l'alignement utilise la distance de Levenshtein
    (rapidfuzz, cohérent avec le CER) au lieu de
    ``difflib.SequenceMatcher`` (Ratcliff–Obershelp).  On fusionne les
    opérations non-``equal`` consécutives en régions ; dans chaque
    région, un déficit de lignes OCR compte comme fusion, un excédent
    comme fragmentation (les blocs ``replace`` 1-pour-1 sont des
    substitutions de contenu, pas des changements de segmentation).
    """
    if not gt_lines or not ocr_lines:
        return 0, 0

    gt_fp = [ln.strip()[:30] for ln in gt_lines]
    ocr_fp = [ln.strip()[:30] for ln in ocr_lines]

    fusion_count = 0
    frag_count = 0
    regions: list[tuple[int, int, int, int]] = []
    for op in Levenshtein.opcodes(gt_fp, ocr_fp):
        if op.tag == "equal":
            continue
        i1, i2, j1, j2 = op.src_start, op.src_end, op.dest_start, op.dest_end
        if regions and regions[-1][1] == i1 and regions[-1][3] == j1:
            p = regions[-1]
            regions[-1] = (p[0], i2, p[2], j2)
        else:
            regions.append((i1, i2, j1, j2))

    for i1, i2, j1, j2 in regions:
        gt_len = i2 - i1
        ocr_len = j2 - j1
        if ocr_len < gt_len:
            fusion_count += gt_len - ocr_len
        elif ocr_len > gt_len:
            frag_count += ocr_len - gt_len

    return fusion_count, frag_count


def _reading_order_score(ground_truth: str, hypothesis: str) -> float:
    """Score d'ordre de lecture [0, 1] basé sur la LCS des mots.

    Audit F4 — calcule la **vraie** plus longue sous-séquence commune
    (``rapidfuzz.distance.LCSseq``, ordre-sensible), normalisée en
    Sørensen–Dice ``2·LCS / (|GT| + |hyp|)``.  Auparavant le code
    renvoyait ``difflib.SequenceMatcher.ratio()`` — une similarité de
    Ratcliff–Obershelp (blocs communs), **pas** une LCS, en
    contradiction avec la docstring.  Un score de 1 ⇔ tous les mots
    communs apparaissent dans le même ordre.
    """
    from rapidfuzz.distance import LCSseq

    gt_words = ground_truth.split()
    hyp_words = hypothesis.split()

    if not gt_words or not hyp_words:
        return 1.0

    lcs = LCSseq.similarity(gt_words, hyp_words)
    denom = len(gt_words) + len(hyp_words)
    return round(2.0 * lcs / denom, 4) if denom else 1.0


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
