"""Vue pipeline composée — chantier 3 post-Sprint 97.

Regroupe les renderers spécifiques aux benchmarks de **pipelines
composées** (axe B du plan d'évolution 2026, Sprints 63-68, 94-97) :

- :func:`picarones.report.pipeline_render.build_pipeline_summary_html`
  — résumé corpus-wide (taux de succès, durée, métriques aux jonctions).
- :func:`picarones.report.pipeline_render.build_pipeline_steps_table_html`
  — tableau par étape (Sprint 67).
- :func:`picarones.report.pipeline_dag_render.build_pipeline_dag_html`
  — visualisation SVG du DAG avec couleur des arêtes selon la métrique.
- :func:`picarones.report.error_absorption_render.build_error_absorption_html`
  — corrections vs introductions à chaque jonction (Sprint 94).
- :func:`picarones.report.incremental_comparison_render.build_incremental_comparison_html`
  — effet isolé d'un slot (LLM, reconstructeur, etc.) en contrôlant
  les autres (Sprint 96).
- :func:`picarones.report.module_audit_render.build_module_audit_html`
  — audit de conformité des modules contribués (Sprint 97).

Cette vue ne s'applique pas au rapport standard (mono-moteur OCR
classique). Elle est appelée explicitement par le workflow
``picarones pipeline run`` (CLI Sprint 70) et par tout outil
extérieur qui consomme un ``PipelineBenchmarkResult``.

Sources de données
------------------
Toutes les sous-sections consomment des structures opt-in passées
en ``opts``. Aucune n'est calculée à partir de ``report_data`` —
c'est par construction (un rapport classique n'a pas de DAG).

- ``opts["pipeline_benchmark"]`` : ``PipelineBenchmarkResult`` (Sprint 64).
- ``opts["dag_nodes"]`` / ``opts["dag_labels"]`` / ``opts["dag_edges"]``
  / ``opts["dag_thresholds"]`` / ``opts["dag_higher_is_better"]`` :
  arguments directs de :func:`build_pipeline_dag_html`.
- ``opts["junctions"]`` : liste de jonctions avec leurs paires
  ``before/after`` pour :func:`build_error_absorption_html`.
- ``opts["incremental_runs"]`` + ``opts["incremental_varying_slot"]`` :
  arguments de :func:`build_incremental_comparison_html`.
- ``opts["module_audits"]`` : liste de ``(manifest, audit_result)``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def build_pipeline_view_html(
    report_data: Optional[dict] = None,
    labels: Optional[dict[str, str]] = None,
    *,
    pipeline_benchmark: Optional[Any] = None,
    dag_nodes: Optional[list] = None,
    dag_labels: Optional[dict[str, str]] = None,
    dag_edges: Optional[list] = None,
    dag_thresholds: Optional[tuple[float, float]] = None,
    dag_higher_is_better: bool = False,
    junctions: Optional[list[dict]] = None,
    incremental_runs: Optional[list] = None,
    incremental_varying_slot: Optional[str] = None,
    incremental_higher_is_better: bool = False,
    module_audits: Optional[list[tuple]] = None,
) -> str:
    """Compose la vue pipeline.

    Parameters
    ----------
    report_data:
        Inutilisé pour cette vue (la pipeline composée a sa propre
        structure de données via ``PipelineBenchmarkResult``).
        Présent dans la signature pour homogénéité avec les autres
        vues du chantier 3.
    labels:
        Dict i18n complet.
    pipeline_benchmark:
        ``PipelineBenchmarkResult`` (Sprint 64) — active les sections
        ``summary`` et ``steps_table`` du :mod:`pipeline_render`.
    dag_nodes, dag_labels, dag_edges, dag_thresholds, dag_higher_is_better:
        Arguments de :func:`build_pipeline_dag_html` (Sprint 95).
    junctions:
        Liste de dicts ``{junction_name, before, after, ...}`` pour
        :func:`build_error_absorption_html` (Sprint 94).
    incremental_runs, incremental_varying_slot, incremental_higher_is_better:
        Arguments de :func:`build_incremental_comparison_html`
        (Sprint 96).
    module_audits:
        Liste de tuples ``(ModuleManifest, AuditResult)`` pour
        :func:`build_module_audit_html` (Sprint 97).

    Returns
    -------
    str
        HTML de la vue ou ``""`` si aucune sous-section opt-in
        n'est fournie.
    """
    labels = labels or {}
    blocks: list[tuple[str, str]] = []

    # Sous-section 1 : résumé + steps table
    if pipeline_benchmark is not None:
        try:
            from picarones.reports.html.renderers.pipeline import (
                build_pipeline_steps_table_html,
                build_pipeline_summary_html,
            )
            summary = build_pipeline_summary_html(pipeline_benchmark)
            steps = build_pipeline_steps_table_html(pipeline_benchmark)
            combined = "\n".join(filter(None, [summary, steps]))
            if combined:
                blocks.append((
                    labels.get(
                        "pipeline_summary_title",
                        "Résumé de la pipeline",
                    ),
                    combined,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[pipeline_view.summary] dégradé : %s", exc,
            )

    # Sous-section 2 : DAG visualization
    if dag_nodes:
        try:
            from picarones.reports.html.renderers.pipeline_dag import (
                build_pipeline_dag_html,
            )
            html = build_pipeline_dag_html(
                nodes=dag_nodes,
                labels=dag_labels or {},
                edges=dag_edges,
                thresholds=dag_thresholds or (0.05, 0.15),
                higher_is_better=dag_higher_is_better,
            )
            if html:
                blocks.append((
                    labels.get(
                        "pipeline_dag_title",
                        "Visualisation du DAG",
                    ),
                    html,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[pipeline_view.dag] dégradé : %s", exc)

    # Sous-section 3 : absorption d'erreur par jonction
    if junctions:
        try:
            from picarones.reports.html.renderers.error_absorption import (
                build_error_absorption_html,
            )
            html = build_error_absorption_html(junctions, labels=labels)
            if html:
                blocks.append((
                    labels.get(
                        "pipeline_absorption_title",
                        "Absorption d'erreur par jonction",
                    ),
                    html,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[pipeline_view.error_absorption] dégradé : %s", exc,
            )

    # Sous-section 4 : comparaison incrémentale (effet d'un slot)
    if incremental_runs and incremental_varying_slot:
        try:
            from picarones.evaluation.metrics.incremental_comparison import (
                compare_isolated_effect,
            )
            from picarones.reports.html.renderers.incremental_comparison import (
                build_incremental_comparison_html,
            )
            comparison = compare_isolated_effect(
                incremental_runs,
                incremental_varying_slot,
                higher_is_better=incremental_higher_is_better,
            )
            html = build_incremental_comparison_html(
                comparison,
                varying_slot=incremental_varying_slot,
                labels=labels,
            )
            if html:
                blocks.append((
                    labels.get(
                        "pipeline_incremental_title",
                        "Comparaison incrémentale",
                    ),
                    html,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[pipeline_view.incremental] dégradé : %s", exc,
            )

    # Sous-section 5 : audit des modules contribués
    if module_audits:
        try:
            from picarones.reports.html.renderers.module_audit import (
                build_module_audit_html,
            )
            html = build_module_audit_html(module_audits, labels=labels)
            if html:
                blocks.append((
                    labels.get(
                        "pipeline_audit_title",
                        "Audit des modules contribués",
                    ),
                    html,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[pipeline_view.audit] dégradé : %s", exc)

    if not blocks:
        return ""

    from picarones.reports.html.views.economics import _render_view_shell

    return _render_view_shell(
        view_title=labels.get(
            "pipeline_view_title", "Banc d'essai de pipeline composée",
        ),
        view_note=labels.get(
            "pipeline_view_note",
            "Vue spécifique aux pipelines composées (axe B) : "
            "métriques aux jonctions, absorption d'erreur, comparaison "
            "incrémentale par slot, audit des modules contribués.",
        ),
        blocks=blocks,
    )


__all__ = ["build_pipeline_view_html"]
