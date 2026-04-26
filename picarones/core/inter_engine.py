"""Métriques inter-moteurs (Sprint 35 — Étape 2 du plan d'évolution).

Deux familles de mesures qui répondent à des questions différentes mais
liées :

1. **Divergence taxonomique** (`kl_divergence`, `jensen_shannon_divergence`,
   `taxonomy_divergence_matrix`) — *à quel point les moteurs font-ils des
   erreurs de natures différentes ?*  Une divergence élevée signale des
   moteurs spécialisés sur des classes d'erreurs distinctes (visual vs
   abréviation vs casse) et donc des candidats pour un voting ensemble.

2. **Complémentarité** (`oracle_token_recall`, `complementarity_gap`,
   `pairwise_disagreement_rate`) — *quel CER serait atteignable si on
   combinait les moteurs ?*  La borne inférieure du CER atteignable par
   un voting majoritaire token-level est ``1 - oracle_token_recall``.
   Si elle est très inférieure au CER du meilleur moteur seul, l'effort
   d'un pipeline d'ensemble se justifie.  Sinon non.

Convention de typage
--------------------
Toutes les fonctions sont enregistrables dans le registre Sprint 34 si
on les wrappe par un adaptateur ``(input_types=(TEXT, TEXT))``.  Pour
limiter le bruit, on ne les enregistre **pas** automatiquement : ce sont
des métriques d'agrégation (multi-moteurs ou multi-documents) qui ne
correspondent pas au modèle « une jonction = une métrique » du runner.
Elles sont consommées par les détecteurs narratifs et le rapport HTML.

Note sur l'oracle
-----------------
La métrique ``oracle_token_recall`` retournée ici utilise un alignement
bag-of-words pondéré par multiplicité.  Ce n'est **pas** une vraie
borne atteignable par voting majoritaire séquentiel — c'est une borne
supérieure (proxy optimiste).  La vraie borne demanderait un
alignement séquentiel des hypothèses, ce qui est plus coûteux.  Pour
le diagnostic « ensemble vaut-il le coup ? », le proxy suffit
largement ; on documente clairement la limite dans le glossaire et le
rapport.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Divergence taxonomique (KL / Jensen-Shannon)
# ──────────────────────────────────────────────────────────────────────────


def _smoothed_distribution(
    distribution: dict[str, float],
    keys: list[str],
    epsilon: float = 1e-12,
) -> list[float]:
    """Aligne une distribution sur l'ordre de ``keys`` et lisse les zéros.

    Le lissage évite ``log(0)`` dans la KL.  ``epsilon`` est volontairement
    minuscule pour ne pas modifier le résultat de manière sensible.
    """
    smoothed = [max(distribution.get(k, 0.0), epsilon) for k in keys]
    total = sum(smoothed)
    return [v / total for v in smoothed]


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL-divergence ``D(P||Q)`` en bits, sur l'union des clés.

    Les distributions n'ont pas besoin de partager exactement les mêmes
    clés ; les clés manquantes sont lissées à ``epsilon`` puis
    renormalisées.

    Returns
    -------
    float
        ``D(P||Q) ≥ 0``.  Vaut 0 si et seulement si P == Q.  N'est pas
        symétrique : ``kl(p, q) != kl(q, p)`` en général.
    """
    keys = sorted(set(p.keys()) | set(q.keys()))
    if not keys:
        return 0.0
    p_vec = _smoothed_distribution(p, keys)
    q_vec = _smoothed_distribution(q, keys)
    return sum(pi * math.log2(pi / qi) for pi, qi in zip(p_vec, q_vec))


def jensen_shannon_divergence(
    p: dict[str, float],
    q: dict[str, float],
) -> float:
    """JS-divergence symétrique en bits, bornée dans ``[0, 1]``.

    ``JS(P, Q) = ½ D(P||M) + ½ D(Q||M)`` avec ``M = (P + Q) / 2``.
    Symétrique et bornée — préférable à la KL pour construire une
    matrice triangulaire de divergences entre moteurs.
    """
    keys = sorted(set(p.keys()) | set(q.keys()))
    if not keys:
        return 0.0
    p_vec = _smoothed_distribution(p, keys)
    q_vec = _smoothed_distribution(q, keys)
    m_vec = [(pi + qi) / 2.0 for pi, qi in zip(p_vec, q_vec)]

    def _kl(a: list[float], b: list[float]) -> float:
        return sum(ai * math.log2(ai / bi) for ai, bi in zip(a, b) if ai > 0)

    js = 0.5 * _kl(p_vec, m_vec) + 0.5 * _kl(q_vec, m_vec)
    # Borne théorique : JS ∈ [0, 1] en bits.  Clamp pour absorber les
    # erreurs d'arrondi flottant.
    return max(0.0, min(1.0, js))


def taxonomy_divergence_matrix(
    distributions: dict[str, dict[str, float]],
    metric: str = "js",
) -> dict[str, dict[str, float]]:
    """Construit la matrice de divergence triangulaire entre moteurs.

    Parameters
    ----------
    distributions:
        ``{engine_name: {error_class: probability}}``.  Chaque
        distribution doit sommer à environ 1 (pas de validation stricte
        — les distributions taxonomiques de Picarones sont déjà
        normalisées par ``aggregate_taxonomy``).
    metric:
        ``"js"`` (défaut, symétrique) ou ``"kl"`` (asymétrique).

    Returns
    -------
    dict[str, dict[str, float]]
        Matrice ``{engine_a: {engine_b: divergence}}`` symétrique pour
        ``js``, asymétrique pour ``kl``.  La diagonale vaut 0.
    """
    if metric not in ("js", "kl"):
        raise ValueError(f"metric doit être 'js' ou 'kl' — reçu {metric!r}")
    fn = jensen_shannon_divergence if metric == "js" else kl_divergence

    engines = sorted(distributions.keys())
    matrix: dict[str, dict[str, float]] = {a: {} for a in engines}
    for a in engines:
        for b in engines:
            if a == b:
                matrix[a][b] = 0.0
            elif metric == "js" and b in matrix and a in matrix[b]:
                # Symétrique : recopie pour éviter de recalculer
                matrix[a][b] = matrix[b][a]
            else:
                matrix[a][b] = fn(distributions[a], distributions[b])
    return matrix


# ──────────────────────────────────────────────────────────────────────────
# Complémentarité (oracle token recall)
# ──────────────────────────────────────────────────────────────────────────


def _word_multiset(text: str) -> Counter[str]:
    """Décomposition en multiset de tokens (séparateur whitespace)."""
    return Counter(tok for tok in text.split() if tok)


def oracle_token_recall(
    reference: str,
    hypotheses: dict[str, str],
) -> float:
    """Borne supérieure (proxy bag-of-words) du token-recall atteignable
    par un voting majoritaire entre tous les moteurs fournis.

    Pour chaque token de la référence (avec sa multiplicité), on
    considère qu'il est "préservé" par l'ensemble si au moins un moteur
    en produit une occurrence non encore comptée.  Le score est le ratio
    d'occurrences GT préservées sur le total.

    Parameters
    ----------
    reference:
        Texte GT.
    hypotheses:
        ``{engine_name: hypothesis_text}``.

    Returns
    -------
    float
        Ratio dans ``[0, 1]``.  ``1.0`` = chaque token GT est présent
        dans au moins une hypothèse à hauteur de sa multiplicité.

    Note
    ----
    Cette borne est **optimiste** (supérieure à la vraie borne par
    voting séquentiel) car elle ignore l'ordre d'apparition.  Pour le
    diagnostic « un voting vaut-il l'effort ? » le proxy suffit ; pour
    une vraie borne il faudrait un alignement séquentiel.
    """
    ref_counter = _word_multiset(reference)
    if not ref_counter or not hypotheses:
        return 1.0 if not ref_counter else 0.0

    hyp_counters = [_word_multiset(h) for h in hypotheses.values()]
    total_ref = sum(ref_counter.values())
    preserved = 0
    for token, gt_count in ref_counter.items():
        # Pour chaque moteur, le nombre d'occurrences disponibles, plafonné
        # à la multiplicité GT.  L'oracle prend le max sur les moteurs.
        best = max((min(gt_count, hc.get(token, 0)) for hc in hyp_counters), default=0)
        preserved += best
    return preserved / total_ref


def complementarity_gap(
    reference: str,
    hypotheses: dict[str, str],
) -> dict[str, float]:
    """Compare l'oracle au meilleur moteur seul.

    Returns
    -------
    dict
        ``{
            "oracle_recall": float,        # bag-of-words recall de l'oracle
            "best_single_recall": float,   # meilleur recall token d'un moteur seul
            "best_engine": str,            # nom du moteur correspondant
            "absolute_gap": float,         # oracle - best_single (toujours ≥ 0)
            "relative_gap": float,         # absolute_gap / (1 - best_single + ε)
                                           # = fraction des erreurs encore évitables
                                           # par un ensemble
        }``
    """
    ref_counter = _word_multiset(reference)
    total = sum(ref_counter.values())
    if not total:
        return {
            "oracle_recall": 1.0,
            "best_single_recall": 1.0,
            "best_engine": "",
            "absolute_gap": 0.0,
            "relative_gap": 0.0,
        }

    def _single_recall(hyp_text: str) -> float:
        hc = _word_multiset(hyp_text)
        preserved = sum(min(gt, hc.get(tok, 0)) for tok, gt in ref_counter.items())
        return preserved / total

    if not hypotheses:
        return {
            "oracle_recall": 0.0,
            "best_single_recall": 0.0,
            "best_engine": "",
            "absolute_gap": 0.0,
            "relative_gap": 0.0,
        }

    per_engine = {name: _single_recall(h) for name, h in hypotheses.items()}
    best_engine, best_recall = max(per_engine.items(), key=lambda kv: kv[1])
    oracle = oracle_token_recall(reference, hypotheses)

    absolute_gap = max(0.0, oracle - best_recall)
    # relative_gap : fraction des erreurs du meilleur moteur que l'ensemble
    # serait théoriquement capable de récupérer (∈ [0, 1])
    headroom = max(1.0 - best_recall, 1e-12)
    relative_gap = min(1.0, absolute_gap / headroom)

    return {
        "oracle_recall": oracle,
        "best_single_recall": best_recall,
        "best_engine": best_engine,
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
    }


def pairwise_disagreement_rate(
    reference: str,
    hyp_a: str,
    hyp_b: str,
) -> float:
    """Fraction de tokens GT pour lesquels A et B sont en désaccord.

    Un désaccord = (l'un préserve le token, l'autre non) OU
    (les deux le ratent mais avec des substitutions différentes — non
    capturé ici, on reste sur la version simple présence/absence).

    Returns
    -------
    float
        Ratio dans ``[0, 1]``.  ``0`` = A et B font les mêmes choix
        (pas de gain d'ensemble).  ``1`` = A et B sont toujours en
        désaccord (gain d'ensemble maximal).
    """
    ref_counter = _word_multiset(reference)
    if not ref_counter:
        return 0.0
    a = _word_multiset(hyp_a)
    b = _word_multiset(hyp_b)
    total = sum(ref_counter.values())
    disagree = 0
    for tok, gt_count in ref_counter.items():
        a_pres = min(gt_count, a.get(tok, 0))
        b_pres = min(gt_count, b.get(tok, 0))
        # Compte les positions où A et B donnent une réponse différente
        disagree += abs(a_pres - b_pres)
    return disagree / total


__all__ = [
    "kl_divergence",
    "jensen_shannon_divergence",
    "taxonomy_divergence_matrix",
    "oracle_token_recall",
    "complementarity_gap",
    "pairwise_disagreement_rate",
]
