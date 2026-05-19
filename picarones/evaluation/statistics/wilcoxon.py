"""Test de Wilcoxon signé-rangé + tests pairwise

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

    # ``diffs_raw`` conserve les zéros : on le transmet **tel quel** à
    # scipy (qui applique ``zero_method`` lui-même).  Audit F9 : éviter
    # le double retrait des zéros (ici puis dans scipy) qui faussait
    # ``n`` et la p-value.  L'implémentation native travaille sur
    # ``diffs`` (zéros retirés pour la méthode "wilcox").
    diffs_raw = [x - y for x, y in zip(a, b)]

    if zero_method == "wilcox":
        diffs = [d for d in diffs_raw if d != 0.0]
    else:
        # "pratt"/"zsplit" : non gérés par l'implémentation native ;
        # scipy (s'il est là) les applique.  En repli natif, on retombe
        # sur "wilcox" en le signalant dans l'interprétation.
        diffs = [d for d in diffs_raw if d != 0.0]

    n = len(diffs)
    if n == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "interpretation": "Aucune différence entre les deux concurrents.",
            "n_pairs": 0,
            "W_plus": 0.0,
            "W_minus": 0.0,
            "method": "exact",
            "has_ties": False,
        }

    # Rangs des valeurs absolues
    abs_diffs = [abs(d) for d in diffs]
    indexed = sorted(enumerate(abs_diffs), key=lambda x: x[1])

    # Gestion des ex-aequo : rang moyen.  On mémorise la taille des
    # groupes d'ex-aequo : un groupe de taille > 1 invalide la
    # distribution exacte (rangs non distincts) → bascule vers
    # l'approximation normale avec correction d'ex-aequo.
    ranks = [0.0] * n
    tie_sizes: list[int] = []
    i = 0
    while i < n:
        j = i
        while j < n and abs_diffs[indexed[j][0]] == abs_diffs[indexed[i][0]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # rang moyen (1-based)
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        tie_sizes.append(j - i)
        i = j
    has_ties = any(t > 1 for t in tie_sizes)

    W_plus  = sum(ranks[k] for k in range(n) if diffs[k] > 0)
    W_minus = sum(ranks[k] for k in range(n) if diffs[k] < 0)
    W = min(W_plus, W_minus)

    # Calcul de la p-value bilatérale.
    #
    # 1. scipy si disponible : méthode exacte (n ≤ 25) ou approximation
    #    normale (n > 25), appelée sur ``diffs_raw`` (zéros inclus) avec
    #    ``zero_method`` — scipy gère le retrait lui-même (audit F9 : plus
    #    de double retrait).
    # 2. Sinon, implémentation native **exacte** : distribution nulle de
    #    W⁺ énumérée par programmation dynamique sur les 2ⁿ assignations
    #    de signes (valable sans ex-aequo, n ≤ 25 — au-delà l'énumération
    #    est inutile, l'approximation normale converge).  Avec ex-aequo
    #    ou n > 25 : approximation normale avec correction d'ex-aequo et
    #    de continuité.  Plus aucune p-value fabriquée (audit F2 : la
    #    table {0.04, 0.20} retournait des faux positifs pour n ≤ 5, où
    #    la significativité bilatérale à 5 % est mathématiquement
    #    impossible).
    method_used = "exact"
    if _SCIPY_AVAILABLE:
        try:
            scipy_res = _scipy_wilcoxon(diffs_raw, zero_method=zero_method)
            p_value = float(scipy_res.pvalue)
            method_used = "scipy"
        except Exception:  # noqa: BLE001 — fallback gracieux
            p_value, method_used = _native_p_value(n, W_plus, W_minus, tie_sizes)
    else:
        p_value, method_used = _native_p_value(n, W_plus, W_minus, tie_sizes)

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
        # Transparence méthodologique (audit F2/F9) : quelle méthode a
        # produit la p-value, et présence d'ex-aequo (qui force
        # l'approximation normale en l'absence de scipy).
        "method": method_used,
        "has_ties": has_ties,
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


def _exact_signed_rank_two_sided_p(
    n: int, w_plus: float, w_minus: float,
) -> float:
    """P-value bilatérale **exacte** du test des rangs signés (sans ex-aequo).

    Sous H0, chacune des 2ⁿ assignations de signes aux rangs 1..n est
    équiprobable.  La distribution de W⁺ (somme des rangs portant un
    signe +) est le nombre de sous-ensembles de ``{1,…,n}`` de somme
    ``s`` divisé par 2ⁿ — fonction génératrice ``∏(1 + xʳ)``, calculée
    par programmation dynamique (knapsack).  La p-value bilatérale vaut
    ``2·P(W⁺ ≤ T)`` avec ``T = min(W⁺, W⁻)``, bornée à 1.0.  Identique
    au mode exact de ``scipy.stats.wilcoxon``.

    Pour n ≤ 5 la p-value minimale possible est 2/2ⁿ ≥ 0.0625 : le test
    ne peut donc jamais être significatif à 5 % bilatéral — ce que
    l'ancienne table ``{0.04, 0.20}`` violait (faux positifs, audit F2).
    """
    total = n * (n + 1) // 2
    counts = [0] * (total + 1)
    counts[0] = 1
    for r in range(1, n + 1):
        for s in range(total, r - 1, -1):
            counts[s] += counts[s - r]
    t = int(min(w_plus, w_minus))
    tail = sum(counts[: t + 1])
    return min(1.0, 2.0 * tail / float(1 << n))


def _native_p_value(
    n: int,
    w_plus: float,
    w_minus: float,
    tie_sizes: list[int],
) -> tuple[float, str]:
    """P-value bilatérale native + nom de la méthode employée.

    - **Sans ex-aequo et n ≤ 25** : distribution exacte (DP ci-dessus).
    - **Sinon** (ex-aequo, ou n > 25) : approximation normale avec
      correction d'ex-aequo sur la variance et correction de continuité
      standard ``(|W − μ| − ½)/σ`` bornée à 0 (audit F9 : l'ancienne
      forme ``|(W+½) − μ|`` était légèrement anti-conservatrice quand
      W ≈ μ).

    Plus aucune p-value fabriquée (audit F2).
    """
    if n == 0:
        return 1.0, "exact"
    has_ties = any(t > 1 for t in tie_sizes)
    if not has_ties and n <= 25:
        return _exact_signed_rank_two_sided_p(n, w_plus, w_minus), "exact"

    mu = n * (n + 1) / 4.0
    # σ² avec correction d'ex-aequo (Wilcoxon signé-rangé) :
    #   σ² = [n(n+1)(2n+1) − ½·Σ(tⱼ³ − tⱼ)] / 24
    tie_term = sum(t ** 3 - t for t in tie_sizes)
    sigma2 = (n * (n + 1) * (2 * n + 1) - 0.5 * tie_term) / 24.0
    if sigma2 <= 0:
        return 1.0, "normal_approx"
    W = min(w_plus, w_minus)
    z = (abs(W - mu) - 0.5) / math.sqrt(sigma2)
    if z < 0.0:
        z = 0.0
    return min(1.0, 2.0 * _normal_sf(z)), "normal_approx"


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
    # tests via le ré-export de ``picarones.evaluation.statistics``.
    "compute_pairwise_stats",
    "wilcoxon_test",
    # Symboles privés ré-exportés (consommés par certains tests) :
    # ``_SCIPY_AVAILABLE`` est utilisé pour skip les tests scipy quand
    # la dépendance n'est pas installée. ``_normal_sf`` est par ailleurs
    # importée par :mod:`friedman_nemenyi` comme utilité math pure.
    "_SCIPY_AVAILABLE",
    "_normal_sf",
]
