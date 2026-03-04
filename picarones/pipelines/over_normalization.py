"""Détection de la sur-normalisation LLM — Classe 10 de la taxonomie des erreurs.

La sur-normalisation désigne le cas où le LLM « corrige » à tort des passages
déjà bien transcrits par l'OCR, en particulier :
- modernisation de graphies médiévales légitimes (nostre → notre, faict → fait)
- normalisation de variantes orthographiques historiques authentiques
- modification de noms propres ou de termes rares sans erreur OCR initiale

Mesure :
    score = nombre de mots (OCR correct → LLM modifié) / nombre de mots OCR corrects

Un score élevé indique que le prompt doit être affiné pour mieux préserver
la graphie originale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OverNormalizationResult:
    """Résultat de la détection de sur-normalisation pour un document."""

    total_correct_ocr_words: int
    over_normalized_count: int
    over_normalized_passages: list[dict] = field(default_factory=list)
    # Chaque entrée : {"gt": str, "ocr": str, "llm": str}

    @property
    def score(self) -> float:
        """Score de sur-normalisation entre 0 (aucune dégradation) et 1 (tout dégradé)."""
        if self.total_correct_ocr_words == 0:
            return 0.0
        return round(self.over_normalized_count / self.total_correct_ocr_words, 4)

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "total_correct_ocr_words": self.total_correct_ocr_words,
            "over_normalized_count": self.over_normalized_count,
            "over_normalized_passages": self.over_normalized_passages[:20],
        }


def detect_over_normalization(
    ground_truth: str,
    ocr_text: str,
    llm_text: str,
    *,
    max_examples: int = 20,
) -> OverNormalizationResult:
    """Détecte la sur-normalisation LLM au niveau des mots.

    Algorithme (alignement positionnel simple, adapté aux textes courts) :
    Pour chaque position i dans min(len(GT), len(OCR), len(LLM)) :
      - Si ocr[i] == gt[i]  → le mot était correct dans l'OCR
      - Si llm[i] != gt[i]  → le LLM a dégradé ce mot correct → sur-normalisation

    Parameters
    ----------
    ground_truth:
        Transcription de référence.
    ocr_text:
        Sortie brute du moteur OCR (avant correction LLM).
    llm_text:
        Sortie après correction par le LLM.
    max_examples:
        Nombre maximal d'exemples de sur-normalisation conservés.

    Returns
    -------
    OverNormalizationResult
    """
    gt_words = ground_truth.split()
    ocr_words = ocr_text.split()
    llm_words = llm_text.split()

    n = min(len(gt_words), len(ocr_words), len(llm_words))

    correct_ocr = 0
    over_norm = 0
    passages: list[dict] = []

    for i in range(n):
        gt_w = gt_words[i]
        ocr_w = ocr_words[i]
        llm_w = llm_words[i]

        if ocr_w == gt_w:
            correct_ocr += 1
            if llm_w != gt_w and len(passages) < max_examples:
                over_norm += 1
                passages.append({"gt": gt_w, "ocr": ocr_w, "llm": llm_w})
            elif llm_w != gt_w:
                over_norm += 1

    return OverNormalizationResult(
        total_correct_ocr_words=correct_ocr,
        over_normalized_count=over_norm,
        over_normalized_passages=passages,
    )


def aggregate_over_normalization(results: list[Optional[OverNormalizationResult]]) -> dict:
    """Agrège les résultats de sur-normalisation sur un ensemble de documents."""
    valid = [r for r in results if r is not None]
    if not valid:
        return {"score": None, "total_correct_ocr_words": 0, "over_normalized_count": 0}

    total_correct = sum(r.total_correct_ocr_words for r in valid)
    total_over = sum(r.over_normalized_count for r in valid)
    score = round(total_over / total_correct, 4) if total_correct > 0 else 0.0

    return {
        "score": score,
        "total_correct_ocr_words": total_correct,
        "over_normalized_count": total_over,
        "document_count": len(valid),
    }
