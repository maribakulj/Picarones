"""Coût marginal par erreur évitée — Sprint 91 (A.II.6 chantier 2).

Sprint 91 — A.II.6 chantier 2 du plan d'évolution 2026.

Pourquoi ce module
------------------
La vue Pareto (Sprint 20) trace CER vs coût mais n'arbitre pas
quel surcoût est *raisonnable* pour quelle réduction d'erreur.
Une institution avec un budget contraint a besoin d'une
réponse opérationnelle :

    *« Passer de Tesseract à Mistral OCR coûte 0,83 € par
    erreur évitée — décider selon votre budget par millier
    d'erreurs corrigées. »*

Formule
-------
Pour deux moteurs A et B où B fait **moins** d'erreurs que A
(donc B est plus précis) :

.. code::

    coût_marginal = (coût_B − coût_A) / (errors_A − errors_B)

- Si ``cost_B > cost_A`` et ``errors_B < errors_A`` :
  ``cost_per_avoided_error > 0`` (cas standard, B coûte plus
  pour moins d'erreurs).
- Si ``cost_B ≤ cost_A`` et ``errors_B < errors_A`` :
  ``cost_per_avoided_error ≤ 0`` (cas idéal, B est strictement
  meilleur).
- Si ``errors_B ≥ errors_A`` : non comparable dans ce sens
  (B n'évite pas d'erreur), retourne ``None``.

Sortie
------
``compute_marginal_cost(cost_a, errors_a, cost_b, errors_b)``
retourne ``{cost_per_avoided_error, n_errors_avoided,
cost_delta, dominated}`` ou ``None`` si non comparable.

``compute_marginal_cost_matrix(per_engine)`` retourne, pour
chaque paire ordonnée ``(A → B)`` où B est plus précis, le
coût marginal correspondant.  Trié par coût marginal croissant
(meilleur ratio en tête).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_marginal_cost(
    cost_a: float,
    errors_a: float,
    cost_b: float,
    errors_b: float,
) -> Optional[dict]:
    """Coût marginal du passage A → B (B plus précis).

    Retourne ``None`` si :
    - ``errors_b >= errors_a`` (B n'évite pas d'erreur) ;
    - les valeurs ne sont pas finies.
    """
    try:
        ca = float(cost_a)
        cb = float(cost_b)
        ea = float(errors_a)
        eb = float(errors_b)
    except (TypeError, ValueError):
        return None
    if ea <= eb:
        # B ne fait pas mieux que A → pas de gain à mesurer.
        return None
    n_avoided = ea - eb
    cost_delta = cb - ca
    cost_per_avoided = cost_delta / n_avoided
    dominated = cost_delta <= 0  # B aussi cher ou moins → cas idéal
    return {
        "cost_per_avoided_error": cost_per_avoided,
        "n_errors_avoided": n_avoided,
        "cost_delta": cost_delta,
        "dominated": dominated,
    }


def compute_marginal_cost_matrix(
    per_engine: dict[str, dict],
) -> Optional[dict]:
    """Pour chaque paire A → B où B fait moins d'erreurs, calcule
    le coût marginal.

    Parameters
    ----------
    per_engine:
        Map ``{engine_name: {"cost": float, "errors": float}}``.

    Returns
    -------
    dict | None
        ``{
            "pairs": list[
                {"engine_a", "engine_b", "cost_per_avoided_error",
                 "n_errors_avoided", "cost_delta", "dominated"}
            ],  # triée par cost_per_avoided_error croissant
        }``
        ou ``None`` si moins de 2 moteurs.
    """
    if not per_engine or len(per_engine) < 2:
        return None
    engines = sorted(per_engine.keys())
    pairs: list[dict] = []
    for a in engines:
        for b in engines:
            if a == b:
                continue
            data_a = per_engine[a]
            data_b = per_engine[b]
            try:
                ca = float(data_a.get("cost"))
                ea = float(data_a.get("errors"))
                cb = float(data_b.get("cost"))
                eb = float(data_b.get("errors"))
            except (TypeError, ValueError):
                continue
            result = compute_marginal_cost(ca, ea, cb, eb)
            if result is None:
                continue
            entry = {"engine_a": a, "engine_b": b}
            entry.update(result)
            pairs.append(entry)
    if not pairs:
        return None
    pairs.sort(key=lambda p: p["cost_per_avoided_error"])
    return {"pairs": pairs}


__all__ = [
    "compute_marginal_cost",
    "compute_marginal_cost_matrix",
]
