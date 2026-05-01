"""Vue diagnostique du rapport — chantier 3 post-Sprint 97.

Regroupe les renderers orientés *« comprendre POURQUOI on a ces
résultats »* :

- :func:`picarones.report.levers_render.build_levers_section_html`
  — leviers d'amélioration éditoriale (factuels, pas prescriptifs).
- :func:`picarones.report.worst_lines_render.build_worst_lines_table_html`
  — top-N des lignes du corpus avec le pire CER (toutes moteurs
  confondus, opt-in : nécessite ``benchmark`` non compacté).
- :func:`picarones.report.image_predictive_render.build_image_predictive_html`
  — complexité paléographique + homogénéité du corpus (opt-in :
  nécessite la liste des image_qualities individuelles).
- :func:`picarones.report.baseline_render.build_corpus_difficulty_baseline_html`
  — encart « ce corpus est-il habituel ? » (opt-in : nécessite
  l'historique SQLite).
- :func:`picarones.report.longitudinal_render.build_longitudinal_html`
  — évolution longitudinale par moteur (opt-in : idem historique).
- :func:`picarones.report.multirun_stability_render.build_multirun_stability_html`
  — stabilité multi-runs (opt-in : nécessite N runs).

Sources de données automatiques
-------------------------------
- *Leviers* : :func:`picarones.measurements.levers.detect_levers` est appelée
  sur ``report_data``. Couvre :
  ``dominant_recoverable_class``, ``pareto_concentration``,
  ``complementarity_observation``, ``lexical_modernization_observation``,
  ``robustness_projection_observation``.

Sources de données opt-in (via ``opts``)
----------------------------------------
- ``opts["benchmark"]``        : ``BenchmarkResult`` non compacté (worst lines).
- ``opts["image_qualities"]``  : liste de dicts image_quality par doc.
- ``opts["baseline_data"]``    : sortie de
  :func:`picarones.measurements.baseline_comparison.compute_corpus_difficulty_percentile`.
- ``opts["longitudinal"]``     : map ``{engine: longitudinal_data}``.
- ``opts["stability"]``        : sortie de
  :func:`picarones.measurements.reliability.compute_multirun_stability`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def build_diagnostics_view_html(
    report_data: dict,
    labels: Optional[dict[str, str]] = None,
    *,
    benchmark: Optional[Any] = None,
    image_qualities: Optional[list[dict]] = None,
    baseline_data: Optional[dict] = None,
    longitudinal: Optional[dict] = None,
    stability: Optional[list[dict]] = None,
    history_values: Optional[list[float]] = None,
) -> str:
    """Compose la vue diagnostique du rapport.

    Parameters
    ----------
    report_data:
        Dict produit par :func:`generator._build_report_data`.
    labels:
        Dict i18n complet.
    benchmark:
        ``BenchmarkResult`` non compacté pour la sous-section worst
        lines (qui re-split les hypothèses par doc et engine).
        Si ``None`` ou si les ``DocumentResult`` ont été compactés,
        la sous-section est masquée.
    image_qualities:
        Liste de dicts ``{contrast, noise_level, blur_score, …}``
        par document, pré-calculée par le runner (ex. extraction
        depuis les ``EngineReport.document_results`` avant compact).
    baseline_data:
        Sortie de
        :func:`picarones.measurements.baseline_comparison.compute_corpus_difficulty_percentile`.
        Active l'encart « ce corpus est-il habituel ? ».
    longitudinal:
        Sortie de
        :func:`picarones.measurements.longitudinal.compute_corpus_longitudinal`.
        Active la table d'évolution.
    stability:
        Liste enrichie de ``{engine_name, ...stability_data}`` par
        moteur, sortie de
        :func:`picarones.measurements.reliability.compute_multirun_stability`.
        Active la table de stabilité multi-runs.
    history_values:
        Valeurs historiques de difficulté du corpus, utilisées pour
        rendre le boxplot dans l'encart baseline.

    Returns
    -------
    str
        HTML de la vue ou ``""`` si aucune sous-section n'a de
        contenu.
    """
    labels = labels or {}
    blocks: list[tuple[str, str]] = []

    # Sous-section 1 : leviers (calculés automatiquement)
    try:
        from picarones.measurements.levers import detect_levers
        from picarones.report.levers_render import build_levers_section_html
        levers = detect_levers(report_data)
        html = build_levers_section_html(levers, labels=labels)
        if html:
            blocks.append((
                labels.get(
                    "diag_levers_title", "Leviers d'amélioration",
                ),
                html,
            ))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[diagnostics_view.levers] dégradé : %s", exc)

    # Sous-section 2 : encart baseline (opt-in via historique)
    if baseline_data:
        try:
            from picarones.report.baseline_render import (
                build_corpus_difficulty_baseline_html,
            )
            html = build_corpus_difficulty_baseline_html(
                baseline_data,
                history_values or [],
                labels=labels,
            )
            if html:
                blocks.append((
                    labels.get(
                        "diag_baseline_title",
                        "Comparaison historique du corpus",
                    ),
                    html,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[diagnostics_view.baseline] dégradé : %s", exc)

    # Sous-section 3 : profil d'image du corpus (opt-in)
    if image_qualities:
        try:
            from picarones.measurements.image_predictive import (
                aggregate_corpus_predictive,
            )
            from picarones.report.image_predictive_render import (
                build_image_predictive_html,
            )
            aggregated = aggregate_corpus_predictive(image_qualities)
            html = build_image_predictive_html(aggregated, labels=labels)
            if html:
                blocks.append((
                    labels.get(
                        "diag_image_predictive_title",
                        "Profil d'image du corpus",
                    ),
                    html,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[diagnostics_view.image_predictive] dégradé : %s", exc,
            )

    # Sous-section 4 : évolution longitudinale (opt-in)
    if longitudinal:
        try:
            from picarones.report.longitudinal_render import (
                build_longitudinal_html,
            )
            html = build_longitudinal_html(longitudinal, labels=labels)
            if html:
                blocks.append((
                    labels.get(
                        "diag_longitudinal_title",
                        "Évolution longitudinale par moteur",
                    ),
                    html,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[diagnostics_view.longitudinal] dégradé : %s", exc,
            )

    # Sous-section 5 : stabilité multi-runs (opt-in)
    if stability:
        try:
            from picarones.report.multirun_stability_render import (
                build_multirun_stability_html,
            )
            html = build_multirun_stability_html(stability, labels=labels)
            if html:
                blocks.append((
                    labels.get(
                        "diag_stability_title",
                        "Stabilité multi-runs",
                    ),
                    html,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[diagnostics_view.stability] dégradé : %s", exc,
            )

    # Sous-section 6 : worst lines (opt-in via benchmark non compacté)
    if benchmark is not None:
        try:
            from picarones.measurements.worst_lines import extract_worst_lines
            from picarones.report.worst_lines_render import (
                build_worst_lines_table_html,
            )
            entries = extract_worst_lines(benchmark, top_n=20)
            html = build_worst_lines_table_html(entries, labels=labels)
            if html:
                blocks.append((
                    labels.get(
                        "diag_worst_lines_title",
                        "Lignes les pires (top 20, tous moteurs)",
                    ),
                    html,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[diagnostics_view.worst_lines] dégradé : %s", exc,
            )

    if not blocks:
        return ""

    from picarones.report.views.economics import _render_view_shell

    return _render_view_shell(
        view_title=labels.get(
            "diag_view_title", "Diagnostic approfondi",
        ),
        view_note=labels.get(
            "diag_view_note",
            "Vue d'aide à l'interprétation : leviers d'amélioration "
            "factuels (jamais prescriptifs), profil d'image du corpus, "
            "comparaison à l'historique de l'institution, et lignes "
            "les pires pour inspection ciblée.",
        ),
        blocks=blocks,
    )


__all__ = ["build_diagnostics_view_html"]
