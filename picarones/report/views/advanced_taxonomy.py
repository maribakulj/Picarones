"""Vue taxonomique avancée — chantier 3 post-Sprint 97.

Regroupe les renderers orientés *édition critique* qui examinent la
structure des erreurs OCR au-delà du CER global :

- :func:`picarones.report.taxonomy_comparison_render.build_taxonomy_comparison_html`
  — diagramme miroir A vs B des proportions d'erreurs par classe
  + tableau de récupérabilité éditoriale.
- :func:`picarones.report.taxonomy_cooccurrence_render.build_taxonomy_cooccurrence_html`
  — heatmap Jaccard des co-occurrences de classes au niveau document
  (opt-in : nécessite ``per_doc_classes``).
- :func:`picarones.report.taxonomy_intra_doc_render.build_taxonomy_intra_doc_html`
  — heatmap classe × position intra-document (opt-in : nécessite des
  paires gt+hyp non compactées).
- :func:`picarones.report.lexical_modernization_render.build_lexical_modernization_html`
  — top-N des tokens GT modernisés par le moteur (opt-in :
  nécessite la sortie de ``compute_lexical_modernization``).

Sources de données automatiques
-------------------------------
- *Comparaison* : utilise ``aggregated_taxonomy.class_distribution``
  (ou ``counts``) du leader CER vs le runner-up. Disponible dès qu'au
  moins 2 moteurs ont une taxonomie agrégée.

Sources de données opt-in (via ``opts``)
----------------------------------------
- ``opts["cooccurrence"]``      : sortie de
  :func:`picarones.core.taxonomy_cooccurrence.compute_taxonomy_cooccurrence`.
- ``opts["intra_doc"]``         : sortie de
  :func:`picarones.core.taxonomy_intra_doc.compute_taxonomy_position_heatmap`.
- ``opts["lexical_modernization"]``  : sortie de
  :func:`picarones.core.lexical_modernization.compute_lexical_modernization`
  agrégée corpus-wide.

Ces calculs ne sont pas faits automatiquement par le runner standard
(coût et données nécessaires non triviaux après ``compact()``) ;
l'utilisateur peut les pré-calculer dans son workflow et les
fournir via :func:`build_advanced_taxonomy_view_html`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _select_two_engines_for_comparison(
    engines_summary: list[dict],
) -> Optional[tuple[dict, dict]]:
    """Choisit deux moteurs à comparer dans le diagramme miroir.

    Stratégie : leader CER (plus bas) vs runner-up (deuxième). Si
    moins de 2 moteurs ont une ``aggregated_taxonomy`` non vide,
    retourne ``None``.
    """
    candidates = [
        e for e in engines_summary
        if isinstance(e.get("aggregated_taxonomy"), dict)
        and (
            e["aggregated_taxonomy"].get("class_distribution")
            or e["aggregated_taxonomy"].get("counts")
        )
    ]
    if len(candidates) < 2:
        return None
    # Tri par CER croissant (leader = meilleur). Les moteurs sans CER
    # vont en queue (clé None considérée comme inf).
    candidates.sort(
        key=lambda e: e.get("cer") if e.get("cer") is not None else float("inf"),
    )
    return candidates[0], candidates[1]


def _extract_class_counts(engine_entry: dict) -> dict[str, float]:
    """Extrait le dict ``{class_name: count}`` d'une entrée moteur.

    Supporte les deux formats observés en production :

    - Sprint 5 historique : ``aggregated_taxonomy["class_distribution"]``
    - Variante : ``aggregated_taxonomy["counts"]``
    """
    tax = engine_entry.get("aggregated_taxonomy") or {}
    counts = tax.get("class_distribution") or tax.get("counts") or {}
    if not isinstance(counts, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in counts.items():
        if isinstance(v, (int, float)) and v >= 0:
            out[str(k)] = float(v)
    return out


def build_advanced_taxonomy_view_html(
    report_data: dict,
    labels: Optional[dict[str, str]] = None,
    *,
    cooccurrence: Optional[dict] = None,
    intra_doc: Optional[dict] = None,
    lexical_modernization: Optional[dict] = None,
) -> str:
    """Compose la vue taxonomique avancée du rapport.

    Parameters
    ----------
    report_data:
        Dict produit par :func:`generator._build_report_data`.
    labels:
        Dict i18n complet.
    cooccurrence:
        Sortie pré-calculée de
        :func:`picarones.core.taxonomy_cooccurrence.compute_taxonomy_cooccurrence`.
        Optionnel — la sous-section est masquée si non fourni.
    intra_doc:
        Sortie pré-calculée de
        :func:`picarones.core.taxonomy_intra_doc.compute_taxonomy_position_heatmap`.
        Optionnel.
    lexical_modernization:
        Sortie pré-calculée de
        :func:`picarones.core.lexical_modernization.aggregate_lexical_modernization`.
        Optionnel.

    Returns
    -------
    str
        HTML de la vue (entête + sous-sections collapsibles) ou
        ``""`` si aucune sous-section n'a de contenu.
    """
    labels = labels or {}
    blocks: list[tuple[str, str]] = []

    # Sous-section 1 : comparaison des deux leaders
    try:
        engines_summary = report_data.get("engines") or []
        pair = _select_two_engines_for_comparison(engines_summary)
        if pair is not None:
            from picarones.core.taxonomy_comparison import compare_taxonomies
            from picarones.report.taxonomy_comparison_render import (
                build_taxonomy_comparison_html,
            )
            engine_a, engine_b = pair
            data = compare_taxonomies(
                engine_a.get("name", "engine_a"),
                _extract_class_counts(engine_a),
                engine_b.get("name", "engine_b"),
                _extract_class_counts(engine_b),
            )
            html = build_taxonomy_comparison_html(data, labels=labels)
            if html:
                blocks.append((
                    labels.get(
                        "advtax_comparison_title",
                        "Comparaison taxonomique (leader vs runner-up)",
                    ),
                    html,
                ))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[advanced_taxonomy_view.comparison] dégradé : %s", exc,
        )

    # Sous-section 2 : co-occurrence (opt-in)
    if cooccurrence:
        try:
            from picarones.report.taxonomy_cooccurrence_render import (
                build_taxonomy_cooccurrence_html,
            )
            html = build_taxonomy_cooccurrence_html(cooccurrence, labels=labels)
            if html:
                blocks.append((
                    labels.get(
                        "advtax_cooccurrence_title",
                        "Co-occurrence de classes d'erreurs",
                    ),
                    html,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[advanced_taxonomy_view.cooccurrence] dégradé : %s", exc,
            )

    # Sous-section 3 : intra-document (opt-in)
    if intra_doc:
        try:
            from picarones.report.taxonomy_intra_doc_render import (
                build_taxonomy_intra_doc_html,
            )
            html = build_taxonomy_intra_doc_html(intra_doc, labels=labels)
            if html:
                blocks.append((
                    labels.get(
                        "advtax_intra_doc_title",
                        "Distribution intra-document des classes",
                    ),
                    html,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[advanced_taxonomy_view.intra_doc] dégradé : %s", exc,
            )

    # Sous-section 4 : modernisation lexicale (opt-in)
    if lexical_modernization:
        try:
            from picarones.report.lexical_modernization_render import (
                build_lexical_modernization_html,
            )
            html = build_lexical_modernization_html(
                lexical_modernization, labels=labels,
            )
            if html:
                blocks.append((
                    labels.get(
                        "advtax_lexmod_title",
                        "Modernisation lexicale (top tokens)",
                    ),
                    html,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[advanced_taxonomy_view.lexmod] dégradé : %s", exc,
            )

    if not blocks:
        return ""

    # Réutilise le shell partagé de la vue economics
    from picarones.report.views.economics import _render_view_shell

    return _render_view_shell(
        view_title=labels.get(
            "advtax_view_title", "Taxonomie avancée des erreurs",
        ),
        view_note=labels.get(
            "advtax_view_note",
            "Vue centrée sur l'édition critique : composition des "
            "erreurs au-delà du CER global, pour décider quel moteur "
            "produit des erreurs récupérables vs irrécupérables.",
        ),
        blocks=blocks,
    )


__all__ = ["build_advanced_taxonomy_view_html"]
