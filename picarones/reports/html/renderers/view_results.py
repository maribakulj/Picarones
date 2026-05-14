"""Phase B6 — rendu HTML des ``BenchmarkResult.view_results``.

Présente les ViewResult produits par le ``RunOrchestrator`` (vues
canoniques ``text_final``, ``alto_documentary``, ``searchability``)
sous forme de sections HTML par vue.

Pour chaque vue présente dans ``benchmark.view_results`` :

- Tableau ``engine × moyenne_par_metric`` des métriques de la vue.
- Liste explicite des **pipelines omis** (qui ne produisent pas
  d'artefact éligible — typiquement un OCR sans ALTO_XML omis de
  ``alto_documentary``).
- Note méthodologique en tête (rappel : ALTO != texte plat).

Adaptive
--------
Le renderer retourne ``""`` si ``benchmark.view_results`` est vide
(cas legacy ``run_benchmark_via_service`` sans RunOrchestrator).
"""

from __future__ import annotations

import html
from statistics import mean


# Vues canoniques : libellés FR/EN par défaut + warnings courts.
_VIEW_DESCRIPTORS: dict[str, dict[str, dict[str, str]]] = {
    "text_final": {
        "fr": {
            "title": "Vue textuelle (TextView)",
            "note": (
                "Compare les sorties textuelles finales (RAW_TEXT, "
                "CORRECTED_TEXT) ou projetées (ALTO/PAGE/markdown → "
                "texte plat).  Les pipelines structurés sont projetés "
                "vers du texte avant comparaison ; leur structure "
                "spatiale est ignorée ici."
            ),
        },
        "en": {
            "title": "Text view (TextView)",
            "note": (
                "Compares final text outputs (RAW_TEXT, CORRECTED_TEXT) "
                "or projected ones (ALTO/PAGE/markdown → flat text).  "
                "Structured pipelines are projected to flat text before "
                "comparison ; their spatial structure is ignored here."
            ),
        },
    },
    "alto_documentary": {
        "fr": {
            "title": "Vue documentaire ALTO (AltoView)",
            "note": (
                "Mesure la fidélité STRUCTURELLE et TEXTUELLE de l'ALTO "
                "produit (validité, lignes, bbox, CER/WER sur le texte "
                "extrait).  Les pipelines qui ne produisent pas d'ALTO "
                "sont OMIS de cette vue (pas de score factice)."
            ),
        },
        "en": {
            "title": "ALTO documentary view (AltoView)",
            "note": (
                "Measures STRUCTURAL and TEXTUAL fidelity of the "
                "produced ALTO (validity, lines, bbox, CER/WER on "
                "extracted text).  Pipelines that don't produce ALTO "
                "are OMITTED from this view (no fake score)."
            ),
        },
    },
    "searchability": {
        "fr": {
            "title": "Recherchabilité (SearchView)",
            "note": (
                "Mesure le rappel fuzzy (Levenshtein ≤ 2) et la "
                "préservation des séquences numériques (années, "
                "cotes).  Pertinent pour estimer l'impact d'un pipeline "
                "sur les moteurs de recherche plein texte (Elastic, Solr)."
            ),
        },
        "en": {
            "title": "Searchability (SearchView)",
            "note": (
                "Measures fuzzy recall (Levenshtein ≤ 2) and "
                "numerical-sequence preservation (years, call numbers).  "
                "Relevant to estimate a pipeline's impact on full-text "
                "search engines (Elastic, Solr)."
            ),
        },
    },
}


def _section_descriptor(view_name: str, lang: str) -> dict[str, str]:
    """Retourne ``{title, note}`` pour une vue.  Fallback générique
    pour les vues custom non connues."""
    descriptor = _VIEW_DESCRIPTORS.get(view_name, {}).get(lang)
    if descriptor is not None:
        return descriptor
    if lang == "en":
        return {
            "title": f"View: {view_name}",
            "note": "Custom view — see runner spec for details.",
        }
    return {
        "title": f"Vue : {view_name}",
        "note": "Vue custom — voir la spec du runner pour détails.",
    }


def _format_metric(value: float) -> str:
    """Formate une métrique en pourcentage si ∈ [0, 1], sinon
    notation courte à 4 décimales."""
    if 0.0 <= value <= 1.0:
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"


def _aggregate_view_metrics(
    engine_buckets: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Pour chaque engine, calcule la moyenne de chaque métrique sur
    tous les documents.  Retourne ``{engine: {metric: mean}}``.

    Une métrique absente d'un doc spécifique est ignorée pour la
    moyenne (tolérance aux ``failed_metrics`` de la vue).
    """
    out: dict[str, dict[str, float]] = {}
    for engine, doc_buckets in engine_buckets.items():
        per_metric: dict[str, list[float]] = {}
        for doc_metrics in doc_buckets.values():
            for metric, value in doc_metrics.items():
                if isinstance(value, (int, float)):
                    per_metric.setdefault(metric, []).append(float(value))
        out[engine] = {
            metric: mean(values)
            for metric, values in per_metric.items()
            if values
        }
    return out


def build_view_results_html(
    view_results: dict[str, dict[str, dict[str, dict[str, float]]]] | None,
    all_engine_names: list[str],
    *,
    lang: str = "fr",
) -> str:
    """Construit le HTML des sections par vue.

    Parameters
    ----------
    view_results:
        Map ``{view_name: {engine_name: {doc_id: {metric: value}}}}``
        produite par ``run_result_to_benchmark_result`` (Phase B6).
        ``None`` ou vide → retour ``""`` (rapport legacy intact).
    all_engine_names:
        Liste de tous les noms d'engines du benchmark, pour identifier
        les **pipelines omis** par chaque vue.
    lang:
        ``"fr"`` (défaut) ou ``"en"`` pour les libellés.

    Returns
    -------
    HTML string prêt à être splatté dans ``base.html.j2`` via le
    placeholder ``{{ view_results_html | safe }}``.
    """
    if not view_results:
        return ""

    sections: list[str] = []
    all_engines_set = set(all_engine_names)
    omitted_label = "Pipelines omis" if lang == "fr" else "Omitted pipelines"
    no_omission_label = (
        "Tous les pipelines éligibles."
        if lang == "fr"
        else "All pipelines eligible."
    )
    metric_label = "Métrique" if lang == "fr" else "Metric"

    for view_name in sorted(view_results.keys()):
        descriptor = _section_descriptor(view_name, lang)
        engine_buckets = view_results[view_name]
        aggregated = _aggregate_view_metrics(engine_buckets)
        eligible_engines = sorted(aggregated.keys())
        omitted = sorted(all_engines_set - set(eligible_engines))

        # Set de toutes les métriques observées (union sur engines).
        all_metrics: set[str] = set()
        for engine_metrics in aggregated.values():
            all_metrics.update(engine_metrics.keys())
        sorted_metrics = sorted(all_metrics)

        # En-tête + note méthodologique.
        section_lines: list[str] = [
            '<div class="chart-card view-results-section"'
            ' style="grid-column:1/-1">',
            f'<h3>{html.escape(descriptor["title"])}</h3>',
            f'<p class="view-note"><em>{html.escape(descriptor["note"])}'
            '</em></p>',
        ]

        if not eligible_engines:
            section_lines.append(
                '<p class="view-no-data">'
                + html.escape(
                    "Aucun pipeline éligible pour cette vue."
                    if lang == "fr"
                    else "No pipeline eligible for this view.",
                )
                + "</p>",
            )
        else:
            # Tableau engine × metric.
            section_lines.append('<table class="view-results-table">')
            section_lines.append("<thead><tr>")
            section_lines.append(
                f'<th>{html.escape(metric_label)}</th>',
            )
            for engine in eligible_engines:
                section_lines.append(
                    f'<th>{html.escape(engine)}</th>',
                )
            section_lines.append("</tr></thead>")
            section_lines.append("<tbody>")
            for metric in sorted_metrics:
                section_lines.append("<tr>")
                section_lines.append(
                    f'<td><code>{html.escape(metric)}</code></td>',
                )
                for engine in eligible_engines:
                    value = aggregated.get(engine, {}).get(metric)
                    cell = (
                        _format_metric(value)
                        if value is not None
                        else "—"
                    )
                    section_lines.append(f"<td>{html.escape(cell)}</td>")
                section_lines.append("</tr>")
            section_lines.append("</tbody></table>")

        # Pipelines omis (toujours affiché — explicite > silencieux).
        section_lines.append(
            f'<p class="view-omitted"><strong>{html.escape(omitted_label)} :'
            "</strong> ",
        )
        if omitted:
            section_lines.append(
                ", ".join(f"<code>{html.escape(e)}</code>" for e in omitted),
            )
        else:
            section_lines.append(
                f"<em>{html.escape(no_omission_label)}</em>",
            )
        section_lines.append("</p>")

        section_lines.append("</div>")
        sections.append("\n".join(section_lines))

    return "\n".join(sections)


__all__ = ["build_view_results_html"]
