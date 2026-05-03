"""Courbes de distribution de la performance (Sprint 7).

- :func:`compute_reliability_curve` — pour les X % docs les plus
  faciles, quel est le CER moyen ? Révèle si un moteur a un long
  tail catastrophique.
- :func:`compute_venn_data` — cardinalités pour un diagramme de
  Venn 2 ou 3 moteurs sur les ensembles d'erreurs commises.
"""

from __future__ import annotations


def compute_reliability_curve(
    cer_values: list[float],
    steps: int = 20,
) -> list[dict]:
    """Pour les X% documents les plus faciles, quel est le CER moyen ?

    Returns
    -------
    Liste de {pct_docs: float, mean_cer: float}
    """
    if not cer_values:
        return []
    sorted_cer = sorted(cer_values)
    n = len(sorted_cer)
    points = []
    for step in range(1, steps + 1):
        pct = step / steps
        cutoff = max(1, int(pct * n))
        subset = sorted_cer[:cutoff]
        mean_cer = sum(subset) / len(subset)
        points.append({"pct_docs": round(pct * 100, 1), "mean_cer": round(mean_cer, 6)})
    return points


def compute_venn_data(
    engine_error_sets: dict[str, set[str]],
) -> dict:
    """Calcule les cardinalités pour un diagramme de Venn entre 2 ou 3 concurrents.

    Parameters
    ----------
    engine_error_sets : {engine_name → set of doc_id:error_token_pair strings}

    Returns
    -------
    Pour 2 concurrents :
      {only_a, only_b, both, label_a, label_b}
    Pour 3 concurrents :
      {only_a, only_b, only_c, ab, ac, bc, abc, label_a, label_b, label_c}
    """
    names = list(engine_error_sets.keys())[:3]  # max 3 pour Venn lisible
    if len(names) < 2:
        return {}

    sets = {n: engine_error_sets[n] for n in names}

    if len(names) == 2:
        a, b = names
        sa, sb = sets[a], sets[b]
        return {
            "type": "venn2",
            "label_a": a,
            "label_b": b,
            "only_a": len(sa - sb),
            "only_b": len(sb - sa),
            "both": len(sa & sb),
        }
    else:
        a, b, c = names
        sa, sb, sc = sets[a], sets[b], sets[c]
        return {
            "type": "venn3",
            "label_a": a,
            "label_b": b,
            "label_c": c,
            "only_a": len(sa - sb - sc),
            "only_b": len(sb - sa - sc),
            "only_c": len(sc - sa - sb),
            "ab": len((sa & sb) - sc),
            "ac": len((sa & sc) - sb),
            "bc": len((sb & sc) - sa),
            "abc": len(sa & sb & sc),
        }


__all__ = ["compute_reliability_curve", "compute_venn_data"]
