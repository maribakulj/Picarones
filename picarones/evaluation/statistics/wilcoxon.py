"""Test de Wilcoxon signé-rangé + tests pairwise (Sprint 7).

Test non-paramétrique pour comparer 2 séries appariées (mêmes
documents, deux moteurs différents). Utilise scipy si disponible
(méthode exacte n ≤ 25), sinon approximation normale native (n ≥ 10)
ou table critique simplifiée pour très petits n.
"""

from __future__ import annotations

import math

# Import optionnel de scipy — utilisé pour le test de Wilcoxon si disponible
# (méthode exacte pour n ≤ 25, approximation normale pour n > 25).
# En son absence, l'implémentation native (approximation normale pour n ≥ 10)
# est utilisée automatiquement.
try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon  # type: ignore[import-untyped]
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def wilcoxon_test(
    a: list[float],
    b: list[float],
    zero_method: str = "wilcox",
) -> dict:
    """Test de Wilcoxon signé-rangé entre deux séries de CER appariées.

    Retourne un dict avec :
      - statistic     : W = min(W⁺, W⁻)
      - p_value       : p-value bilatérale
      - significant   : bool (p < 0.05)
      - interpretation : phrase lisible
      - n_pairs       : nombre de paires utilisées (après retrait des zéros)
      - W_plus        : somme des rangs des différences positives
      - W_minus       : somme des rangs des différences négatives

    Hypothèses et limites
    ---------------------
    * Les observations sont appariées (même corpus, deux moteurs différents).
    * Le test est non-paramétrique : aucune hypothèse de normalité des CER.
    * ``zero_method="wilcox"`` (défaut) : les paires sans différence (aᵢ = bᵢ)
      sont simplement exclues.  Les autres méthodes (``"pratt"``, ``"zsplit"``)
      nécessitent scipy.
    * **Approximation normale** (implémentation native, n ≥ 10) :
      L'approximation est raisonnable pour n ≥ 10 et converge vers la
      distribution exacte.  Pour n < 10, une table critique simplifiée est
      utilisée (p ∈ {0.04, 0.20}) — résultat **conservateur**.
    * **scipy** (si installé) : ``scipy.stats.wilcoxon`` est utilisé à la place
      de l'approximation native.  scipy utilise la méthode exacte pour n ≤ 25
      et l'approximation normale pour n > 25, ce qui est plus précis.
    * **Validité** : le test suppose la symétrie de la distribution des
      différences.  Avec de très petits n (< 5), les résultats sont peu fiables
      quelle que soit la méthode.

    Parameters
    ----------
    a, b : séries de CER (même longueur, même ordre de documents)
    zero_method : gestion des paires nulles (défaut : ``"wilcox"``)
    """
    if len(a) != len(b):
        raise ValueError("Les deux listes doivent avoir la même longueur")

    diffs = [x - y for x, y in zip(a, b)]

    # Retirer les zéros (méthode "wilcox")
    if zero_method == "wilcox":
        diffs = [d for d in diffs if d != 0.0]

    n = len(diffs)
    if n == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "interpretation": "Aucune différence entre les deux concurrents.",
            "n_pairs": 0,
        }

    # Rangs des valeurs absolues
    abs_diffs = [abs(d) for d in diffs]
    indexed = sorted(enumerate(abs_diffs), key=lambda x: x[1])

    # Gestion des ex-aequo : rang moyen
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs_diffs[indexed[j][0]] == abs_diffs[indexed[i][0]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # rang moyen (1-based)
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j

    W_plus  = sum(ranks[k] for k in range(n) if diffs[k] > 0)
    W_minus = sum(ranks[k] for k in range(n) if diffs[k] < 0)
    W = min(W_plus, W_minus)

    # Calcul de la p-value : scipy si disponible, sinon approximation native
    if _SCIPY_AVAILABLE:
        try:
            scipy_res = _scipy_wilcoxon(diffs, zero_method=zero_method)
            p_value = float(scipy_res.pvalue)
        except Exception:  # noqa: BLE001 — fallback gracieux
            # Repli sur l'implémentation native en cas d'erreur scipy
            p_value = _native_p_value(n, W)
    else:
        p_value = _native_p_value(n, W)

    significant = p_value < 0.05

    if significant:
        better = "premier" if W_plus < W_minus else "second"
        interpretation = (
            f"Différence statistiquement significative (p = {p_value:.4f} < 0.05). "
            f"Le {better} concurrent obtient de meilleurs scores."
        )
    else:
        interpretation = (
            f"Différence non significative (p = {p_value:.4f} ≥ 0.05). "
            "On ne peut pas conclure que l'un surpasse l'autre."
        )

    return {
        "statistic": round(W, 4),
        "p_value": round(p_value, 6),
        "significant": significant,
        "interpretation": interpretation,
        "n_pairs": n,
        "W_plus": round(W_plus, 4),
        "W_minus": round(W_minus, 4),
    }


def _normal_sf(z: float) -> float:
    """Survival function de la loi normale standard (1 - CDF).

    Approximation Abramowitz & Stegun 26.2.17. Utilisée par cette
    famille pour Wilcoxon ET par friedman_nemenyi pour le fallback
    Wilson-Hilferty quand scipy n'est pas disponible.
    """
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
           + t * (-1.821255978 + t * 1.330274429))))
    phi_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    p = phi_z * poly
    return p if z >= 0 else 1.0 - p


# Table des valeurs critiques de W pour α=0.05 bilatéral (test exact, source : tables de Wilcoxon)
_W_CRITICAL = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 2, 8: 3, 9: 5}


def _wilcoxon_exact_p(n: int, w: float) -> float:
    """P-value approximée pour petits n (< 10) via table critique simplifiée.

    Note : résultat **conservateur** — seules deux valeurs sont retournées :
    0.04 (significatif à 5 %) ou 0.20 (non significatif).
    Préférer scipy pour des p-values exactes.
    """
    critical = _W_CRITICAL.get(n, 0)
    if w <= critical:
        return 0.04  # significatif à 5 %
    return 0.20      # non significatif (approximation conservative)


def _native_p_value(n: int, W: float) -> float:
    """Calcule la p-value via l'approximation normale (n ≥ 10) ou la table exacte (n < 10)."""
    if n >= 10:
        mu = n * (n + 1) / 4.0
        sigma2 = n * (n + 1) * (2 * n + 1) / 24.0
        if sigma2 <= 0:
            return 1.0
        z = abs((W + 0.5) - mu) / math.sqrt(sigma2)  # correction de continuité
        return 2.0 * _normal_sf(z)  # test bilatéral
    return _wilcoxon_exact_p(n, W)


def compute_pairwise_stats(
    engine_cer_map: dict[str, list[float]],
) -> list[dict]:
    """Calcule les tests de Wilcoxon entre toutes les paires de concurrents.

    Parameters
    ----------
    engine_cer_map : dict {engine_name → [cer_doc1, cer_doc2, ...]}

    Returns
    -------
    Liste de dicts, un par paire :
      - engine_a, engine_b, statistic, p_value, significant, interpretation
    """
    names = list(engine_cer_map.keys())
    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a_vals = engine_cer_map[a_name]
            b_vals = engine_cer_map[b_name]
            # Aligner les longueurs
            min_len = min(len(a_vals), len(b_vals))
            if min_len < 2:
                continue
            res = wilcoxon_test(a_vals[:min_len], b_vals[:min_len])
            results.append({
                "engine_a": a_name,
                "engine_b": b_name,
                **res,
            })
    return results


__all__ = [
    # Symboles publics : signature stable, consommés directement par les
    # tests via le ré-export de ``picarones.measurements.statistics``.
    "compute_pairwise_stats",
    "wilcoxon_test",
    # Symboles privés ré-exportés (consommés par certains tests) :
    # ``_SCIPY_AVAILABLE`` est utilisé pour skip les tests scipy quand
    # la dépendance n'est pas installée. ``_normal_sf`` est par ailleurs
    # importée par :mod:`friedman_nemenyi` comme utilité math pure.
    "_SCIPY_AVAILABLE",
    "_normal_sf",
]
