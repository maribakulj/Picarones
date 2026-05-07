"""Intervalle de confiance par bootstrap (Sprint 7).

Méthode de rééchantillonnage non-paramétrique. Pas d'hypothèse de
distribution normale — adapté aux distributions asymétriques de CER
typiques des corpus patrimoniaux.
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
    lo_idx = max(0, int(alpha * n_iter))
    hi_idx = min(n_iter - 1, int((1.0 - alpha) * n_iter))
    return (means[lo_idx], means[hi_idx])


__all__ = ["bootstrap_ci"]
