"""Recherchabilité fuzzy + séquences numériques — Sprint A14-S16.

Fonctions de calcul **pures** (sans ``@register_metric`` legacy)
utilisées par ``SearchView``.  Réimplémente la logique des modules
historiques ``picarones.measurements.searchability`` (Sprint 84)
et ``picarones.measurements.numerical_sequences`` (Sprint 85),
sans la dépendance vers le singleton global ``core.metric_registry``.

Les modules legacy seront supprimés au S20 quand le
``MetricRegistry`` instancié explicitement (S5) deviendra le seul
registre.  En attendant, ce module fournit la version "couche
evaluation" propre.

Métriques livrées
-----------------
- ``searchability_recall(reference, hypothesis, max_distance=2)`` —
  proportion de tokens GT retrouvés dans l'hypothèse à distance
  de Levenshtein ≤ ``max_distance``.  Proxy direct de la qualité
  pour la recherche plein-texte (Elastic / Solr / Gallica).

- ``numerical_sequence_preservation(reference, hypothesis)`` —
  fraction des séquences numériques de la GT préservées
  strictement dans l'hypothèse.  Volontairement minimaliste pour
  S16 : détecte uniquement les **années 4 chiffres** (proxy
  réaliste pour les corpus patrimoniaux datés).  Le cas complet
  (numéraux romains, foliations, monnaies, années régnales) reste
  dans le legacy et sera réintégré au S20 avec le registre.

Toutes les métriques ∈ [0, 1] avec ``higher_is_better=True``.
"""

from __future__ import annotations

import re


# ──────────────────────────────────────────────────────────────────
# Levenshtein — DP O(|a|·|b|), mémoire O(min(|a|, |b|))
# ──────────────────────────────────────────────────────────────────


def levenshtein_distance(a: str, b: str) -> int:
    """Distance de Levenshtein (substitution = insertion = suppression = 1).

    Implémentation identique à ``picarones.measurements.searchability``
    (Sprint 84) mais sans le décorateur ``@register_metric``.
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                current[j - 1] + 1,        # insertion
                previous[j] + 1,           # suppression
                previous[j - 1] + cost,    # substitution
            )
        previous = current
    return previous[-1]


# ──────────────────────────────────────────────────────────────────
# Searchability fuzzy
# ──────────────────────────────────────────────────────────────────


def _split_words(text: str | None) -> list[str]:
    if not text:
        return []
    return text.split()


def searchability_recall(
    reference: str,
    hypothesis: str,
    *,
    max_distance: int = 2,
    case_sensitive: bool = False,
) -> float:
    """Rappel fuzzy : fraction des tokens GT retrouvés à distance
    de Levenshtein ≤ ``max_distance``.

    Multi-set : un token hypothèse ne peut servir qu'une fois pour
    être compté comme "match" (alignement bipartite simple).

    Returns
    -------
    float
        ``n_retrouves / n_gt`` ∈ [0, 1].  ``0.0`` si la GT est
        vide (convention identique au legacy Sprint 84).
    """
    if max_distance < 0:
        raise ValueError(f"max_distance doit être ≥ 0, reçu {max_distance}")
    gt_tokens = _split_words(reference)
    hyp_tokens = _split_words(hypothesis)
    n_gt = len(gt_tokens)
    if n_gt == 0:
        return 0.0
    if case_sensitive:
        gt_for_match = list(gt_tokens)
        hyp_for_match = list(hyp_tokens)
    else:
        gt_for_match = [t.lower() for t in gt_tokens]
        hyp_for_match = [t.lower() for t in hyp_tokens]

    hyp_used = [False] * len(hyp_for_match)
    n_match = 0
    for gt_match in gt_for_match:
        best_idx = -1
        best_dist = max_distance + 1
        for hi, used in enumerate(hyp_used):
            if used:
                continue
            hyp_match = hyp_for_match[hi]
            if abs(len(hyp_match) - len(gt_match)) > max_distance:
                continue
            d = levenshtein_distance(gt_match, hyp_match)
            if d < best_dist:
                best_dist = d
                best_idx = hi
                if d == 0:
                    break
        if best_idx >= 0 and best_dist <= max_distance:
            hyp_used[best_idx] = True
            n_match += 1
    return n_match / n_gt


# ──────────────────────────────────────────────────────────────────
# Séquences numériques (S16 minimal : années 4 chiffres)
# ──────────────────────────────────────────────────────────────────


_YEAR_4DIGITS_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
"""Capture les années entre 1000 et 2029 (proxy réaliste pour les
corpus patrimoniaux : chartes médiévales, registres modernes,
coupures de presse XIX-XXIᵉ siècle)."""


def _extract_years(text: str | None) -> list[str]:
    if not text:
        return []
    return _YEAR_4DIGITS_RE.findall(text)


def numerical_sequence_preservation(
    reference: str,
    hypothesis: str,
) -> float:
    """Fraction des années 4 chiffres de la GT préservées strictement
    dans l'hypothèse.

    Returns
    -------
    float
        ``n_preserved / n_gt_years`` ∈ [0, 1].  ``0.0`` si la GT
        ne contient aucune année.

    Note méthodologique
    -------------------
    Volontairement minimaliste pour S16 : seules les années 4
    chiffres sont détectées.  Le pattern complet (numéraux romains,
    foliations ``f. 12r``, monnaies, années régnales ``an III``)
    reste dans ``picarones.measurements.numerical_sequences``
    (Sprint 85) et sera réintégré dans la couche evaluation au S20.

    Multi-set : si la GT contient ``"1789"`` deux fois et
    l'hypothèse une fois, seul un est compté préservé.
    """
    gt_years = _extract_years(reference)
    if not gt_years:
        return 0.0
    hyp_years = _extract_years(hypothesis)
    # Multi-set match.
    hyp_pool = list(hyp_years)
    n_preserved = 0
    for y in gt_years:
        if y in hyp_pool:
            hyp_pool.remove(y)
            n_preserved += 1
    return n_preserved / len(gt_years)


__all__ = [
    "levenshtein_distance",
    "searchability_recall",
    "numerical_sequence_preservation",
]
