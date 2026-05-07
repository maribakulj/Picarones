"""Rendu HTML server-side de la section NER (Sprint 41).

Suite directe des Sprints 38-40 : la couche de calcul, le backend
extracteur et le câblage runner sont en place ; ce module produit les
blocs HTML qui remontent ces données dans le rapport.

- ``build_ner_summary_html`` — encart factuel par moteur : F1 global,
  precision/recall, total entités, hallucinations, missed.
- ``build_ner_per_category_html`` — table heatmap moteur × catégorie,
  cellules colorées par F1 (rouge → vert).

Principe — cohérent avec ``inter_engine_render`` (Sprint 37) : rendu
server-side, pas de JavaScript, déterministe.  Si aucun moteur n'a de
``aggregated_ner``, les fonctions retournent une chaîne vide — la vue
est silencieusement omise (rapport adaptatif).

Anti-injection : tous les noms de moteurs et catégories sont passés à
``html.escape`` avant insertion.
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.reports_v2._helpers.render_helpers import color_traffic_light


def _engines_with_ner(engines_summary: list[dict]) -> list[dict]:
    """Filtre les moteurs qui ont une analyse NER agrégée."""
    return [e for e in engines_summary if e.get("aggregated_ner")]


def build_ner_summary_html(
    engines_summary: list[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit l'encart résumé NER : F1 global par moteur + totaux.

    Parameters
    ----------
    engines_summary:
        Liste de dicts moteur (au moins ``name`` et ``aggregated_ner``).
    labels:
        Dict d'étiquettes i18n.

    Returns
    -------
    str
        HTML ``<div>...</div>`` ou ``""`` si aucun moteur n'a de NER.
    """
    relevant = _engines_with_ner(engines_summary)
    if not relevant:
        return ""

    labels = labels or {}
    caption = labels.get("ner_summary_caption", "Précision sur entités nommées")
    engine_label = labels.get("ner_engine_label", "Moteur")
    f1_label = labels.get("ner_f1_label", "F1 global")
    p_label = labels.get("ner_precision_label", "Précision")
    r_label = labels.get("ner_recall_label", "Rappel")
    docs_label = labels.get("ner_doc_count_label", "Docs évalués")
    halluc_label = labels.get("ner_hallucinated_label", "Hallucinations")
    missed_label = labels.get("ner_missed_label", "Entités manquées")

    parts: list[str] = []
    parts.append('<div class="ner-summary">')
    parts.append(
        f'<div class="ner-summary-caption" style="font-weight:600;'
        f'margin-bottom:.4rem">{_e(caption)}</div>'
    )
    parts.append(
        '<table class="ner-summary-table" '
        'style="border-collapse:collapse;font-size:.85rem;width:100%">'
    )
    parts.append("<thead><tr>")
    for hdr in (engine_label, f1_label, p_label, r_label,
                docs_label, halluc_label, missed_label):
        parts.append(
            f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:left;'
            f'border-bottom:1px solid var(--border);font-weight:600">'
            f'{_e(hdr)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    for engine in relevant:
        agg = engine["aggregated_ner"]
        global_stats = agg.get("global", {}) or {}
        f1 = float(global_stats.get("f1") or 0.0)
        precision = float(global_stats.get("precision") or 0.0)
        recall = float(global_stats.get("recall") or 0.0)
        doc_count = int(agg.get("doc_count") or 0)
        hallucinated = int(agg.get("hallucinated_total") or 0)
        missed = int(agg.get("missed_total") or 0)
        bg = color_traffic_light(f1)
        parts.append("<tr>")
        parts.append(
            f'<td style="padding:.3rem .5rem;font-weight:600">'
            f'{_e(engine.get("name", ""))}</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;background:{bg};'
            f'font-variant-numeric:tabular-nums">{f1 * 100:.1f} %</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;font-variant-numeric:tabular-nums">'
            f'{precision * 100:.1f} %</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;font-variant-numeric:tabular-nums">'
            f'{recall * 100:.1f} %</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;font-variant-numeric:tabular-nums">'
            f'{doc_count}</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;font-variant-numeric:tabular-nums">'
            f'{hallucinated}</td>'
        )
        parts.append(
            f'<td style="padding:.3rem .5rem;font-variant-numeric:tabular-nums">'
            f'{missed}</td>'
        )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def build_ner_per_category_html(
    engines_summary: list[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit la heatmap NER moteur × catégorie d'entité.

    Lignes = moteurs, colonnes = catégories (PER, LOC, ORG, DATE,
    MISC…).  Cellules colorées par F1 (rouge → vert).  La cellule
    affiche le F1 en pourcentage.  Cellules vides quand la catégorie
    n'a pas été observée pour le moteur.

    Returns
    -------
    str
        HTML ``<div>...</div>`` ou ``""`` si pas de données.
    """
    relevant = _engines_with_ner(engines_summary)
    if not relevant:
        return ""

    # Catégories : union sur tous les moteurs, ordre alphabétique
    all_categories: set[str] = set()
    for engine in relevant:
        per_cat = (engine["aggregated_ner"] or {}).get("per_category") or {}
        all_categories.update(per_cat.keys())
    if not all_categories:
        return ""
    categories = sorted(all_categories)

    labels = labels or {}
    caption = labels.get(
        "ner_per_category_caption",
        "F1 par catégorie d'entité (heatmap)",
    )
    engine_label = labels.get("ner_engine_label", "Moteur")
    no_data = labels.get("ner_no_data_label", "—")

    parts: list[str] = []
    parts.append('<div class="ner-per-category">')
    parts.append(
        f'<div class="ner-per-category-caption" '
        f'style="font-weight:600;margin-bottom:.4rem">{_e(caption)}</div>'
    )
    parts.append(
        '<table class="ner-per-category-table" '
        'style="border-collapse:collapse;font-size:.8rem">'
    )
    parts.append("<thead><tr>")
    parts.append(
        f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:left;'
        f'border-bottom:1px solid var(--border)">{_e(engine_label)}</th>'
    )
    for cat in categories:
        parts.append(
            f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:center;'
            f'border-bottom:1px solid var(--border)">{_e(cat)}</th>'
        )
    parts.append("</tr></thead><tbody>")
    for engine in relevant:
        per_cat = (engine["aggregated_ner"] or {}).get("per_category") or {}
        parts.append("<tr>")
        parts.append(
            f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:right;'
            f'border-right:1px solid var(--border);font-weight:600">'
            f'{_e(engine.get("name", ""))}</th>'
        )
        for cat in categories:
            stats = per_cat.get(cat)
            if not stats or int(stats.get("support", 0)) == 0:
                parts.append(
                    f'<td style="padding:.3rem .5rem;text-align:center;'
                    f'background:#f4f4f4;color:var(--text-muted);'
                    f'font-style:italic">{_e(no_data)}</td>'
                )
            else:
                f1 = float(stats.get("f1") or 0.0)
                support = int(stats.get("support", 0))
                bg = color_traffic_light(f1)
                parts.append(
                    f'<td style="padding:.3rem .5rem;text-align:center;'
                    f'background:{bg};color:#222;'
                    f'font-variant-numeric:tabular-nums" '
                    f'title="support={support}">'
                    f'{f1 * 100:.1f} %</td>'
                )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


__all__ = [
    "build_ner_summary_html",
    "build_ner_per_category_html",
]
