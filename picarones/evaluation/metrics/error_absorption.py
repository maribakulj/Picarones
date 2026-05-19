"""Métrique d'absorption d'erreur — Sprint 94 (B.3).

B.3 du plan d'évolution 2026.

Pourquoi ce module
------------------
Quand un module de post-correction LLM aplatit les différences
entre OCR amont, ce n'est pas qu'il « améliore » tous les
moteurs — c'est qu'il introduit ses propres biais qui dominent
ceux de l'OCR.  Mesurer la dégradation par étape ne suffit
pas : il faut **séparer** les deux flux.

À chaque jonction où un module transforme un artefact, on
mesure :

- **Taux de correction** : parmi les erreurs présentes en
  entrée du module, combien sont corrigées en sortie ?
- **Taux d'introduction** : parmi les erreurs présentes en
  sortie, combien sont **nouvelles** (absentes en entrée) ?

C'est la généralisation du score de sur-normalisation
(chantier A.I.7) à toute jonction.  La formule s'applique
uniformément à OCR→LLM, OCR→reconstructor, VLM→ALTO_mapper —
toute jonction qui transforme un artefact en un autre du même
type.

Méthode (token-level)
---------------------
On split en tokens whitespace ``reference``, ``before``,
``after``.  On compare en **multiset** (un token GT consommé
au plus une fois) :

- ``errors_before`` = tokens GT non retrouvés dans ``before``
- ``errors_after``  = tokens GT non retrouvés dans ``after``
- ``corrected``     = ``errors_before \\ errors_after``
  (présents avant, absents après → corrigés)
- ``introduced``    = ``errors_after \\ errors_before``
  (absents avant, présents après → introduits)

Garde-fou : le module ne classe pas les erreurs (visuelles,
abréviations, etc.) — c'est une métrique d'**absorption de
volume**, pas de qualité éditoriale.  L'intersection sémantique
avec ``taxonomy`` (Sprint 5) est documentée dans le glossaire.

Sortie
------
``compute_error_absorption(reference, before, after)`` retourne :

.. code-block:: text

    {
        "n_gt_tokens": int,
        "n_errors_before": int,
        "n_errors_after": int,
        "n_corrected": int,
        "n_introduced": int,
        "n_kept_wrong": int,
        "correction_rate": float | None,    # n_corrected / n_errors_before
        "introduction_rate": float | None,  # n_introduced / n_errors_after
        "net_improvement": int,             # n_corrected - n_introduced
        "corrected_tokens": list[str],
        "introduced_tokens": list[str],
    }

``aggregate_error_absorption(per_doc_results)`` somme les
compteurs corpus-wide et recalcule les taux *micro*.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def _split_words(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return text.split()


def _missing_tokens(
    reference: list[str], hypothesis: list[str],
) -> Counter:
    """Tokens GT manquants en hypothèse au sens multiset.

    Un token GT compte plusieurs fois s'il apparaît plusieurs
    fois ; chaque occurrence en hypothèse en absorbe au plus
    une.  Retourne un Counter ``{token: nb_occurrences_manquees}``.
    """
    ref_count = Counter(reference)
    hyp_count = Counter(hypothesis)
    missing: Counter = Counter()
    for token, n_ref in ref_count.items():
        n_hyp = hyp_count.get(token, 0)
        if n_hyp < n_ref:
            missing[token] = n_ref - n_hyp
    return missing


def compute_error_absorption(
    reference: Optional[str],
    before: Optional[str],
    after: Optional[str],
    *,
    case_sensitive: bool = False,
) -> Optional[dict]:
    """Mesure l'absorption d'erreur entre ``before`` et ``after``.

    Parameters
    ----------
    reference:
        GT (vérité terrain).
    before:
        Sortie de l'étape précédente (typiquement OCR amont).
    after:
        Sortie de l'étape courante (typiquement post-correction LLM).
    case_sensitive:
        Si False (défaut), match case-insensitive — la sortie
        ``corrected_tokens``/``introduced_tokens`` reste en casse
        GT originale.

    Returns
    -------
    dict | None
        ``None`` si la GT est vide ou ne contient aucun token.
    """
    ref_tokens = _split_words(reference)
    if not ref_tokens:
        return None
    before_tokens = _split_words(before)
    after_tokens = _split_words(after)

    if case_sensitive:
        ref_match = list(ref_tokens)
        before_match = list(before_tokens)
        after_match = list(after_tokens)
    else:
        ref_match = [t.lower() for t in ref_tokens]
        before_match = [t.lower() for t in before_tokens]
        after_match = [t.lower() for t in after_tokens]

    # Map case-insensitive token → liste de casses GT originales
    ref_orig_by_match: dict[str, list[str]] = {}
    for orig, m in zip(ref_tokens, ref_match):
        ref_orig_by_match.setdefault(m, []).append(orig)

    missing_before = _missing_tokens(ref_match, before_match)
    missing_after = _missing_tokens(ref_match, after_match)

    n_errors_before = sum(missing_before.values())
    n_errors_after = sum(missing_after.values())

    # Calcul corrigé / introduit en multiset
    corrected_counter: Counter = Counter()
    introduced_counter: Counter = Counter()
    kept_wrong_counter: Counter = Counter()
    all_tokens = set(missing_before) | set(missing_after)
    for tok in all_tokens:
        nb = missing_before.get(tok, 0)
        na = missing_after.get(tok, 0)
        if nb > na:
            corrected_counter[tok] = nb - na
            kept_wrong_counter[tok] = na
        elif na > nb:
            introduced_counter[tok] = na - nb
            kept_wrong_counter[tok] = nb
        else:
            kept_wrong_counter[tok] = nb

    n_corrected = sum(corrected_counter.values())
    n_introduced = sum(introduced_counter.values())
    n_kept_wrong = sum(kept_wrong_counter.values())

    correction_rate = (
        n_corrected / n_errors_before
        if n_errors_before > 0 else None
    )
    introduction_rate = (
        n_introduced / n_errors_after
        if n_errors_after > 0 else None
    )

    def _expand(counter: Counter) -> list[str]:
        out: list[str] = []
        for tok, count in counter.items():
            origs = ref_orig_by_match.get(tok, [tok])
            # Ne renvoie que la casse représentative GT
            display = origs[0] if origs else tok
            out.extend([display] * count)
        return out

    return {
        "n_gt_tokens": len(ref_tokens),
        "n_errors_before": n_errors_before,
        "n_errors_after": n_errors_after,
        "n_corrected": n_corrected,
        "n_introduced": n_introduced,
        "n_kept_wrong": n_kept_wrong,
        "correction_rate": correction_rate,
        "introduction_rate": introduction_rate,
        "net_improvement": n_corrected - n_introduced,
        "corrected_tokens": _expand(corrected_counter),
        "introduced_tokens": _expand(introduced_counter),
    }


def aggregate_error_absorption(
    per_doc: Iterable[Optional[dict]],
    *,
    sample_tokens: int = 50,
) -> Optional[dict]:
    """Agrège les compteurs corpus-wide et recalcule les taux
    *micro*.

    Parameters
    ----------
    per_doc:
        Itérable de sorties de ``compute_error_absorption`` (ou
        ``None`` pour les docs sans GT).
    sample_tokens:
        Nombre maximal de tokens corrigés/introduits gardés dans
        l'échantillon (cap pour ne pas exploser le JSON).

    Returns
    -------
    dict | None
        ``None`` si aucune entry valide.
    """
    docs = [d for d in per_doc if d]
    if not docs:
        return None
    n_gt = sum(int(d.get("n_gt_tokens") or 0) for d in docs)
    n_errors_before = sum(int(d.get("n_errors_before") or 0) for d in docs)
    n_errors_after = sum(int(d.get("n_errors_after") or 0) for d in docs)
    n_corrected = sum(int(d.get("n_corrected") or 0) for d in docs)
    n_introduced = sum(int(d.get("n_introduced") or 0) for d in docs)
    n_kept_wrong = sum(int(d.get("n_kept_wrong") or 0) for d in docs)
    correction_rate = (
        n_corrected / n_errors_before if n_errors_before > 0 else None
    )
    introduction_rate = (
        n_introduced / n_errors_after if n_errors_after > 0 else None
    )
    corrected_sample: list[str] = []
    introduced_sample: list[str] = []
    for d in docs:
        corrected_sample.extend(d.get("corrected_tokens") or [])
        introduced_sample.extend(d.get("introduced_tokens") or [])
        if (
            len(corrected_sample) >= sample_tokens
            and len(introduced_sample) >= sample_tokens
        ):
            break
    return {
        "n_docs": len(docs),
        "n_gt_tokens": n_gt,
        "n_errors_before": n_errors_before,
        "n_errors_after": n_errors_after,
        "n_corrected": n_corrected,
        "n_introduced": n_introduced,
        "n_kept_wrong": n_kept_wrong,
        "correction_rate": correction_rate,
        "introduction_rate": introduction_rate,
        "net_improvement": n_corrected - n_introduced,
        "corrected_tokens_sample": corrected_sample[:sample_tokens],
        "introduced_tokens_sample": introduced_sample[:sample_tokens],
    }


__all__ = [
    "compute_error_absorption",
    "aggregate_error_absorption",
]
