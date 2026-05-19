"""Intervalle de confiance par bootstrap

Méthode de rééchantillonnage non-paramétrique. Pas d'hypothèse de
distribution normale — adapté aux distributions asymétriques de CER
typiques des corpus patrimoniaux.

Méthode : **bootstrap percentile** (Efron).  C'est volontairement le
percentile et non le BCa : déterministe (graine fixe), simple à
auditer, suffisant pour un IC indicatif d'aide à la décision.  Pour
une distribution fortement asymétrique le BCa serait plus précis aux
queues — limite documentée, pas un bug.  Audit scientifique F8 :
l'indice de quantile est désormais la statistique d'ordre correcte
``round(q·(n_iter−1))`` (l'ancien ``int(q·n_iter)`` décalait la borne
haute d'environ un rang).
"""

from __future__ import annotations

import random


def bootstrap_ci(
    values: list[float],
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Intervalle de confiance par bootstrap.

    Parameters
    ----------
    values : liste des valeurs (ex. CER par document)
    n_iter : nombre d'itérations bootstrap (défaut 1000)
    ci     : niveau de confiance (défaut 0.95 → 95 %)
    seed   : graine RNG pour reproductibilité

    Returns
    -------
    (lower, upper) — les bornes de l'IC à ``ci`` %
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_iter):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - ci) / 2.0
    # Statistique d'ordre : quantile q sur un tableau trié 0-indexé de
    # taille n_iter → indice round(q·(n_iter−1)), borné (audit F8).
    def _q_index(q: float) -> int:
        idx = round(q * (n_iter - 1))
        return max(0, min(n_iter - 1, idx))

    return (means[_q_index(alpha)], means[_q_index(1.0 - alpha)])


__all__ = ["bootstrap_ci"]
