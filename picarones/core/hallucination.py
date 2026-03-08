"""Détection des hallucinations VLM/LLM — Sprint 10.

Métriques calculées
-------------------
- Taux d'insertion net    : mots/caractères ajoutés absents du GT, distinct du WIL existant
- Ratio de longueur       : len(hyp) / len(gt) — ratio > 1.2 → hallucination potentielle
- Score d'ancrage         : proportion des n-grammes (trigrammes) de la sortie présents dans le GT
- Blocs hallucinés        : segments continus de la sortie sans correspondance GT au-delà d'un seuil
- Badge hallucination     : True si ancrage faible ou ratio de longueur anormal
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers texte
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Découpe en mots (minuscules, sans ponctuation)."""
    return re.findall(r"[^\s]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Génère les n-grammes d'une liste de tokens."""
    if len(tokens) < n:
        return [tuple(tokens)] if tokens else []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# Blocs hallucinés (segments continus sans ancrage)
# ---------------------------------------------------------------------------

@dataclass
class HallucinatedBlock:
    """Segment continu de la sortie sans correspondance dans le GT."""
    start_token: int
    end_token: int
    text: str
    length: int  # nombre de tokens

    def as_dict(self) -> dict:
        return {
            "start_token": self.start_token,
            "end_token": self.end_token,
            "text": self.text,
            "length": self.length,
        }


def _detect_hallucinated_blocks(
    hyp_tokens: list[str],
    gt_token_set: set[str],
    tolerance: int = 3,
    min_block_length: int = 4,
) -> list[HallucinatedBlock]:
    """Détecte les blocs de tokens hypothèse sans correspondance dans le GT.

    Un bloc est un segment contigu de tokens hypothèse dont aucun n'est présent
    dans le vocabulaire GT. Une tolérance de ``tolerance`` tokens connus interrompus
    est acceptée avant de clore un bloc.

    Parameters
    ----------
    hyp_tokens:
        Tokens de la sortie OCR/VLM.
    gt_token_set:
        Ensemble des tokens du GT (pour recherche O(1)).
    tolerance:
        Nombre de tokens connus consécutifs interrompant un bloc avant de le clore.
    min_block_length:
        Longueur minimale (tokens) pour qu'un bloc soit signalé.

    Returns
    -------
    list[HallucinatedBlock]
    """
    blocks: list[HallucinatedBlock] = []
    if not hyp_tokens:
        return blocks

    in_block = False
    block_start = 0
    consecutive_known = 0

    for i, tok in enumerate(hyp_tokens):
        is_unknown = tok not in gt_token_set
        if is_unknown:
            if not in_block:
                in_block = True
                block_start = i
                consecutive_known = 0
            else:
                consecutive_known = 0
        else:
            if in_block:
                consecutive_known += 1
                if consecutive_known >= tolerance:
                    # Clore le bloc
                    end = i - consecutive_known
                    length = end - block_start + 1
                    if length >= min_block_length:
                        text = " ".join(hyp_tokens[block_start:end + 1])
                        blocks.append(HallucinatedBlock(
                            start_token=block_start,
                            end_token=end,
                            text=text,
                            length=length,
                        ))
                    in_block = False
                    consecutive_known = 0

    # Bloc non terminé
    if in_block:
        end = len(hyp_tokens) - 1
        length = end - block_start + 1
        if length >= min_block_length:
            text = " ".join(hyp_tokens[block_start:end + 1])
            blocks.append(HallucinatedBlock(
                start_token=block_start,
                end_token=end,
                text=text,
                length=length,
            ))

    return blocks


# ---------------------------------------------------------------------------
# Résultat structuré
# ---------------------------------------------------------------------------

@dataclass
class HallucinationMetrics:
    """Métriques de détection des hallucinations pour une paire (GT, hypothèse)."""

    net_insertion_rate: float
    """Taux d'insertion nette : tokens hypothèse absents du GT / total tokens hypothèse."""

    length_ratio: float
    """Ratio de longueur : len(hyp) / len(gt) en caractères. > 1.2 = signal d'hallucination."""

    anchor_score: float
    """Score d'ancrage : proportion des trigrammes hypothèse présents dans les trigrammes GT.
    Score élevé → l'hypothèse s'ancre bien dans le GT. Score faible → hallucinations probables."""

    hallucinated_blocks: list[HallucinatedBlock]
    """Segments continus de la sortie sans correspondance GT (au-dessus du seuil de tolérance)."""

    is_hallucinating: bool
    """True si anchor_score < anchor_threshold OU length_ratio > length_ratio_threshold."""

    # Détails supplémentaires
    gt_word_count: int = 0
    hyp_word_count: int = 0
    net_inserted_words: int = 0
    anchor_threshold_used: float = 0.5
    length_ratio_threshold_used: float = 1.2
    ngram_size_used: int = 3

    def as_dict(self) -> dict:
        return {
            "net_insertion_rate": round(self.net_insertion_rate, 6),
            "length_ratio": round(self.length_ratio, 6),
            "anchor_score": round(self.anchor_score, 6),
            "hallucinated_blocks": [b.as_dict() for b in self.hallucinated_blocks],
            "is_hallucinating": self.is_hallucinating,
            "gt_word_count": self.gt_word_count,
            "hyp_word_count": self.hyp_word_count,
            "net_inserted_words": self.net_inserted_words,
            "anchor_threshold_used": self.anchor_threshold_used,
            "length_ratio_threshold_used": self.length_ratio_threshold_used,
            "ngram_size_used": self.ngram_size_used,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HallucinationMetrics":
        blocks = [
            HallucinatedBlock(**b) for b in d.get("hallucinated_blocks", [])
        ]
        return cls(
            net_insertion_rate=d.get("net_insertion_rate", 0.0),
            length_ratio=d.get("length_ratio", 1.0),
            anchor_score=d.get("anchor_score", 1.0),
            hallucinated_blocks=blocks,
            is_hallucinating=d.get("is_hallucinating", False),
            gt_word_count=d.get("gt_word_count", 0),
            hyp_word_count=d.get("hyp_word_count", 0),
            net_inserted_words=d.get("net_inserted_words", 0),
            anchor_threshold_used=d.get("anchor_threshold_used", 0.5),
            length_ratio_threshold_used=d.get("length_ratio_threshold_used", 1.2),
            ngram_size_used=d.get("ngram_size_used", 3),
        )


# ---------------------------------------------------------------------------
# Calcul principal
# ---------------------------------------------------------------------------

def compute_hallucination_metrics(
    reference: str,
    hypothesis: str,
    n: int = 3,
    length_ratio_threshold: float = 1.2,
    anchor_threshold: float = 0.5,
    block_tolerance: int = 3,
    min_block_length: int = 4,
) -> HallucinationMetrics:
    """Calcule les métriques de détection des hallucinations VLM/LLM.

    Parameters
    ----------
    reference:
        Texte de vérité terrain (GT).
    hypothesis:
        Texte produit par le modèle.
    n:
        Taille des n-grammes pour le score d'ancrage (défaut : trigrammes).
    length_ratio_threshold:
        Seuil de ratio de longueur au-dessus duquel on signale une hallucination potentielle.
    anchor_threshold:
        Seuil de score d'ancrage en dessous duquel on signale une hallucination potentielle.
    block_tolerance:
        Nombre de tokens connus consécutifs acceptés dans un bloc halluciné.
    min_block_length:
        Longueur minimale (tokens) pour signaler un bloc halluciné.

    Returns
    -------
    HallucinationMetrics
    """
    gt_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    gt_len_chars = len(reference.strip())
    hyp_len_chars = len(hypothesis.strip())

    # ── Ratio de longueur ────────────────────────────────────────────────
    if gt_len_chars == 0:
        length_ratio = 1.0 if hyp_len_chars == 0 else float("inf")
    else:
        length_ratio = hyp_len_chars / gt_len_chars

    # ── Taux d'insertion nette ───────────────────────────────────────────
    gt_token_set = set(gt_tokens)
    hyp_token_count = len(hyp_tokens)

    if hyp_token_count == 0:
        net_insertion_rate = 0.0
        net_inserted_words = 0
    else:
        net_inserted = [t for t in hyp_tokens if t not in gt_token_set]
        net_inserted_words = len(net_inserted)
        net_insertion_rate = net_inserted_words / hyp_token_count

    # ── Score d'ancrage (n-grammes) ──────────────────────────────────────
    gt_ngrams = set(_ngrams(gt_tokens, n))
    hyp_ngrams = _ngrams(hyp_tokens, n)

    if not hyp_ngrams:
        # Pas de n-grammes dans l'hypothèse → ancrage parfait (hypothèse vide ou trop courte)
        anchor_score = 1.0 if not gt_ngrams else 0.0
    elif not gt_ngrams:
        anchor_score = 0.0
    else:
        anchored = sum(1 for ng in hyp_ngrams if ng in gt_ngrams)
        anchor_score = anchored / len(hyp_ngrams)

    # ── Blocs hallucinés ─────────────────────────────────────────────────
    blocks = _detect_hallucinated_blocks(
        hyp_tokens=hyp_tokens,
        gt_token_set=gt_token_set,
        tolerance=block_tolerance,
        min_block_length=min_block_length,
    )

    # ── Badge hallucination ──────────────────────────────────────────────
    is_hallucinating = (
        anchor_score < anchor_threshold
        or (length_ratio > length_ratio_threshold and length_ratio != float("inf"))
    )

    return HallucinationMetrics(
        net_insertion_rate=net_insertion_rate,
        length_ratio=min(length_ratio, 9.99),  # plafonner pour la sérialisation
        anchor_score=anchor_score,
        hallucinated_blocks=blocks,
        is_hallucinating=is_hallucinating,
        gt_word_count=len(gt_tokens),
        hyp_word_count=hyp_token_count,
        net_inserted_words=net_inserted_words,
        anchor_threshold_used=anchor_threshold,
        length_ratio_threshold_used=length_ratio_threshold,
        ngram_size_used=n,
    )


# ---------------------------------------------------------------------------
# Agrégation sur un corpus
# ---------------------------------------------------------------------------

def aggregate_hallucination_metrics(results: list[HallucinationMetrics]) -> dict:
    """Agrège les métriques d'hallucination sur un corpus.

    Returns
    -------
    dict
        Statistiques agrégées : anchor_score moyen, taux de documents hallucinés…
    """
    if not results:
        return {}

    n = len(results)
    anchor_values = [r.anchor_score for r in results]
    ratio_values = [r.length_ratio for r in results]
    insertion_values = [r.net_insertion_rate for r in results]
    hallucinating_count = sum(1 for r in results if r.is_hallucinating)

    return {
        "anchor_score_mean": round(sum(anchor_values) / n, 6),
        "anchor_score_min": round(min(anchor_values), 6),
        "length_ratio_mean": round(sum(ratio_values) / n, 6),
        "net_insertion_rate_mean": round(sum(insertion_values) / n, 6),
        "hallucinating_doc_count": hallucinating_count,
        "hallucinating_doc_rate": round(hallucinating_count / n, 6),
        "document_count": n,
    }
