"""Frontière de Pareto multi-objectifs (Sprint 19).

Algorithme générique sur N objectifs (CER, coût, vitesse, CO₂…).
Renvoie les noms des points non-dominés.
"""

from __future__ import annotations

from typing import Optional


def compute_pareto_front(
    points: list[dict],
    objectives: tuple[str, ...] = ("cer", "cost"),
    name_key: str = "engine",
    minimize: Optional[tuple[bool, ...]] = None,
) -> list[str]:
    """Calcule la frontière de Pareto sur ``len(objectives)`` dimensions.

    Un point ``p`` est Pareto-dominant si aucun autre point n'a, pour TOUS
    les objectifs, une valeur au moins aussi bonne ET au moins une valeur
    strictement meilleure.

    Parameters
    ----------
    points:
        Liste de dicts. Chaque dict doit contenir ``name_key`` et toutes les
        clés de ``objectives``. Les points dont une valeur d'objectif est
        ``None`` sont ignorés (pas de comparaison possible).
    objectives:
        Clés des objectifs à minimiser/maximiser.
    name_key:
        Clé identifiant le point (par défaut ``"engine"``).
    minimize:
        Pour chaque objectif, ``True`` = minimiser (ex. CER, coût),
        ``False`` = maximiser (ex. ancrage). Doit avoir la même longueur
        que ``objectives``.

    Returns
    -------
    Liste des ``name`` des points sur le front Pareto, ordre stable depuis
    ``points``.
    """
    if minimize is None:
        minimize = tuple(True for _ in objectives)
    if len(minimize) != len(objectives):
        raise ValueError("`minimize` doit avoir la même longueur que `objectives`")

    valid = []
    for p in points:
        try:
            vals = tuple(float(p[k]) for k in objectives)
        except (KeyError, TypeError, ValueError):
            continue
        valid.append((p[name_key], vals))

    front: list[str] = []
    for name_a, vals_a in valid:
        dominated = False
        for name_b, vals_b in valid:
            if name_a == name_b:
                continue
            # B domine A si B est ≥ aussi bon partout ET strictement meilleur quelque part
            better_or_equal_everywhere = True
            strictly_better_somewhere = False
            for va, vb, mini in zip(vals_a, vals_b, minimize):
                if mini:
                    if vb > va:
                        better_or_equal_everywhere = False
                        break
                    if vb < va:
                        strictly_better_somewhere = True
                else:  # maximiser
                    if vb < va:
                        better_or_equal_everywhere = False
                        break
                    if vb > va:
                        strictly_better_somewhere = True
            if better_or_equal_everywhere and strictly_better_somewhere:
                dominated = True
                break
        if not dominated:
            front.append(name_a)
    return front


__all__ = ["compute_pareto_front"]
