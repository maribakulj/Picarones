"""Rendu HTML server-side de la vue stratifiée par script_type (Sprint 46).

Suite directe du Sprint 45 (couche backend). Affiche le classement
moteur par strate sous forme de tableaux pliables (HTML ``<details>``,
pas de JavaScript).

- ``build_stratified_ranking_html`` — un ``<details>`` par strate avec
  tableau ``moteur, médiane, moyenne, docs``. Cellule médiane colorée
  par gradient vert (faible CER) → rouge (CER élevé).

Principe : cohérent avec ``inter_engine_render``, ``ner_render`` et
``calibration_render`` — server-side, déterministe, pas de JS.
Masquage adaptatif : la fonction retourne ``""`` si aucune strate
n'est disponible (``available_strata`` vide).

Anti-injection : tous les noms de moteurs et de strates sont passés
à ``html.escape``.
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.reports_v2._helpers.render_helpers import color_traffic_light


def _format_cer(cer: Optional[float]) -> str:
    if cer is None:
        return "—"
    return f"{cer * 100:.2f} %"


def build_stratified_ranking_html(
    stratified_ranking: Optional[dict],
    available_strata: Optional[list],
    homogeneity: Optional[dict] = None,
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit la section HTML stratifiée.

    Parameters
    ----------
    stratified_ranking:
        ``{stratum: [ranking_entry, …]}`` produit par
        ``BenchmarkResult.stratified_ranking()``.
    available_strata:
        Liste triée des strates (``BenchmarkResult.available_strata()``).
    homogeneity:
        Dict produit par ``BenchmarkResult.corpus_homogeneity()`` si
        disponible — sert à afficher l'écart inter-strate du leader
        en tête de section.
    labels:
        i18n.  Fallback FR si manquantes.

    Returns
    -------
    str
        HTML ``<div>...</div>`` ou ``""`` si stratification absente.
    """
    if not stratified_ranking or not available_strata:
        return ""

    labels = labels or {}
    caption = labels.get(
        "stratification_caption",
        "Classement par strate (script_type)",
    )
    description = labels.get(
        "stratification_description",
        "Le tableau global classe sur l'ensemble du corpus. Quand le "
        "corpus est hétérogène, certains moteurs dominent sur un type "
        "de document et perdent sur un autre — la vue stratifiée le "
        "révèle.",
    )
    engine_label = labels.get("col_engine", "Moteur")
    median_label = labels.get("stratification_median_label", "Médiane CER")
    mean_label = labels.get("stratification_mean_label", "Moyenne CER")
    docs_label = labels.get("stratification_docs_label", "Documents")
    no_data = labels.get("stratification_no_data_label", "—")
    n_docs_in_stratum_label = labels.get(
        "stratification_n_docs_label", "documents",
    )

    parts: list[str] = []
    parts.append('<div class="stratified-ranking" style="margin-top:1.2rem">')
    parts.append(
        f'<h3 style="margin:0 0 .3rem 0">{_e(caption)}</h3>'
    )
    parts.append(
        f'<div style="font-size:.78rem;color:var(--text-muted);'
        f'margin-bottom:.6rem">{_e(description)}</div>'
    )

    # Bandeau d'hétérogénéité si disponible
    if homogeneity and homogeneity.get("max_inter_strata_gap") is not None:
        gap = float(homogeneity["max_inter_strata_gap"])
        leader = str(homogeneity.get("leader") or "")
        min_strat, max_strat = homogeneity.get(
            "leader_max_gap_strata", ["", ""]
        )
        gap_template = labels.get(
            "stratification_gap_summary",
            "Écart inter-strate du leader {leader} : {gap_pct} points "
            "de CER médian (entre « {min_stratum} » et « {max_stratum} »).",
        )
        gap_text = gap_template.format(
            leader=leader,
            gap_pct=f"{gap * 100:.1f}",
            min_stratum=min_strat,
            max_stratum=max_strat,
        )
        # gap_text contient déjà des données utilisateur — on n'échappe pas
        # le template lui-même (i18n connue), mais on n'injecte pas non plus
        # de markup. _e() est appliqué aux variables via format() côté template.
        parts.append(
            f'<div style="font-size:.82rem;background:#fff8e1;'
            f'border-left:3px solid #f9a825;padding:.4rem .6rem;'
            f'margin-bottom:.6rem">⚠ {_e(gap_text)}</div>'
        )

    # Une ``<details>`` par strate (premier ouvert pour donner le contexte)
    for i, stratum in enumerate(available_strata):
        entries = stratified_ranking.get(stratum) or []
        n_docs_total = max((int(e.get("documents") or 0) for e in entries), default=0)
        open_attr = " open" if i == 0 else ""
        parts.append(
            f'<details class="stratum-block"{open_attr} '
            f'style="margin-bottom:.4rem;border:1px solid var(--border);'
            f'border-radius:6px;padding:.4rem .6rem">'
        )
        parts.append(
            f'<summary style="cursor:pointer;font-weight:600">'
            f'{_e(stratum)} '
            f'<span style="font-weight:400;color:var(--text-muted);'
            f'font-size:.85rem">({n_docs_total} {_e(n_docs_in_stratum_label)})</span>'
            f'</summary>'
        )
        parts.append(
            '<table style="border-collapse:collapse;font-size:.85rem;'
            'margin-top:.4rem;width:100%">'
        )
        parts.append("<thead><tr>")
        for hdr in (engine_label, median_label, mean_label, docs_label):
            parts.append(
                f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:left;'
                f'border-bottom:1px solid var(--border);font-weight:600">'
                f'{_e(hdr)}</th>'
            )
        parts.append("</tr></thead><tbody>")
        for entry in entries:
            engine = str(entry.get("engine", ""))
            median = entry.get("median_cer")
            mean = entry.get("mean_cer")
            n_docs = int(entry.get("documents") or 0)
            bg = color_traffic_light(float(median), low_is_good=True, scale_max=0.30) if median is not None else "#f4f4f4"
            parts.append("<tr>")
            parts.append(
                f'<td style="padding:.3rem .5rem;font-weight:600">'
                f'{_e(engine)}</td>'
            )
            parts.append(
                f'<td style="padding:.3rem .5rem;background:{bg};'
                f'font-variant-numeric:tabular-nums">'
                f'{_e(_format_cer(median)) if median is not None else _e(no_data)}'
                f'</td>'
            )
            parts.append(
                f'<td style="padding:.3rem .5rem;'
                f'font-variant-numeric:tabular-nums">'
                f'{_e(_format_cer(mean)) if mean is not None else _e(no_data)}'
                f'</td>'
            )
            parts.append(
                f'<td style="padding:.3rem .5rem;'
                f'font-variant-numeric:tabular-nums">{n_docs}</td>'
            )
            parts.append("</tr>")
        parts.append("</tbody></table>")
        parts.append("</details>")

    parts.append("</div>")
    return "".join(parts)


__all__ = [
    "build_stratified_ranking_html",
]
