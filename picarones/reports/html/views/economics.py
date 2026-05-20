"""Vue économique du rapport — chantier 3 post

Regroupe les renderers orientés *décision budget* :

- :func:`picarones.reports.html.renderers.throughput.build_throughput_html`
  — pages/h **utilisable** (raw - correction humaine), formule
  HTR-United (5 s/erreur).

Renderers prévus mais nécessitant des données opt-in (cost projection
par volume, coût marginal par erreur évitée) restent non câblés ici :
ils s'activeront quand l'utilisateur fournira ``opts["target_pages"]``
et ``opts["pricing"]`` au constructeur, ou via un workflow CLI dédié
``picarones economics``.

Adaptive masking
----------------
La vue retourne ``""`` quand aucune sous-section n'a de signal
exploitable.  Elle ne s'affiche donc dans le rapport que si au moins
un moteur a un throughput estimable (somme des durées non nulle).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _estimate_engine_throughput_inputs(
    engine_reports: list,
) -> list[dict]:
    """Construit les entrées attendues par
    :func:`picarones.evaluation.metrics.throughput.aggregate_effective_throughput`
    à partir des ``EngineReport`` du benchmark.

    Pour chaque moteur :

    - ``n_pages``         : nombre de documents traités sans erreur OCR.
    - ``duration_seconds``: somme des ``duration_seconds`` des docs réussis.
    - ``n_errors``        : approximation au niveau **mot** ≈
      ``wer × total_words_gt``.  C'est un proxy : on n'a pas l'alignement
      exact, on multiplie le WER moyen par le nombre total de mots dans
      la GT (toutes longueurs confondues).  Cette approximation est
      cohérente avec la définition du WER.

    Le moteur est exclu si ``n_pages == 0`` ou si toutes les durations
    sont nulles (cas d'un cache).
    """
    out: list[dict] = []
    for report in engine_reports:
        successful = [
            dr for dr in report.document_results
            if getattr(dr, "engine_error", None) is None
        ]
        if not successful:
            continue
        total_duration = sum(
            float(getattr(dr, "duration_seconds", 0.0)) for dr in successful
        )
        if total_duration <= 0:
            # Bench depuis cache — pas de mesure de vitesse exploitable
            continue
        # Estimation du nombre de mots GT total (somme des longueurs
        # référence).  ``MetricsResult.reference_length`` est en
        # caractères ; on convertit grossièrement en mots par
        # heuristique 5 caractères/mot pour l'agrégation.
        total_words_gt = 0
        weighted_wer = 0.0
        for dr in successful:
            ref_chars = getattr(dr.metrics, "reference_length", 0) or 0
            ref_words = max(1, int(ref_chars / 5)) if ref_chars else 0
            wer = getattr(dr.metrics, "wer", 0.0) or 0.0
            total_words_gt += ref_words
            weighted_wer += wer * ref_words
        if total_words_gt == 0:
            n_errors = 0
        else:
            mean_wer = weighted_wer / total_words_gt
            n_errors = int(round(mean_wer * total_words_gt))
        out.append({
            "engine_name": report.engine_name,
            "n_pages": len(successful),
            "duration_seconds": total_duration,
            "n_errors": max(0, n_errors),
        })
    return out


def build_economics_view_html(
    report_data: dict,
    labels: Optional[dict[str, str]] = None,
    *,
    engine_reports: Optional[list] = None,
    time_per_error_seconds: float = 5.0,
    extra_html_blocks: Optional[list[str]] = None,
) -> str:
    """Compose la vue économique du rapport.

    Parameters
    ----------
    report_data:
        Dict produit par :func:`generator._build_report_data`.
        Les sous-renderers reçoivent ``labels`` directement ; cette
        fonction n'extrait que les éléments qu'elle peut composer
        à partir de ``report_data``.
    labels:
        Dict i18n complet du rapport.
    engine_reports:
        Liste des ``EngineReport`` du benchmark.  Indispensable pour
        calculer le throughput effectif (besoin des durations
        document par document, non exposées dans ``report_data``).
        Si ``None``, la sous-section throughput est sautée.
    time_per_error_seconds:
        Constante de correction humaine pour le throughput effectif
        (défaut HTR-United : 5 s par erreur mot).
    extra_html_blocks:
        Blocs HTML déjà rendus à inclure tels quels (par exemple
        cost projection par volume, fourni par un workflow CLI dédié).
        Permet d'étendre la vue sans modifier ce module.

    Returns
    -------
    str
        HTML complet de la vue (entête + sous-sections collapsibles)
        ou ``""`` si aucune sous-section ne produit de contenu.
    """
    labels = labels or {}
    blocks: list[tuple[str, str]] = []

    # Sous-section 1 : throughput effectif
    if engine_reports:
        try:
            from picarones.evaluation.metrics.throughput import (
                aggregate_effective_throughput,
            )
            from picarones.reports.html.renderers.throughput import (
                build_throughput_html,
            )
            inputs = _estimate_engine_throughput_inputs(engine_reports)
            aggregated = aggregate_effective_throughput(
                inputs, time_per_error_seconds=time_per_error_seconds,
            )
            html = build_throughput_html(aggregated, labels=labels)
            if html:
                blocks.append((
                    labels.get("economics_throughput_title", "Throughput effectif"),
                    html,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[economics_view.throughput] dégradé : %s", exc,
            )

    # Sous-section 2 : blocs externes (cost projection, marginal cost…)
    if extra_html_blocks:
        for i, html in enumerate(extra_html_blocks):
            if not html:
                continue
            blocks.append((
                labels.get(
                    f"economics_extra_{i}_title",
                    labels.get("economics_extra_title", "Coût projeté"),
                ),
                html,
            ))

    if not blocks:
        return ""

    return _render_view_shell(
        view_title=labels.get("economics_view_title", "Coût et performance"),
        view_note=labels.get(
            "economics_view_note",
            "Vue centrée sur la décision budget : pages traitables par "
            "heure réellement utilisable (en intégrant la correction "
            "humaine post-OCR), et projection de coût par volume cible.",
        ),
        blocks=blocks,
    )


def _render_view_shell(
    *,
    view_title: str,
    view_note: str,
    blocks: list[tuple[str, str]],
) -> str:
    """Compose un shell ``<details>`` collapsible par bloc, premier ouvert.

    Convention de rendu partagée par les 5 vues du chantier 3 :
    chaque sous-section est un ``<details>`` natif (collapsible
    sans JS), avec son sous-titre dans le ``<summary>``.  Le premier
    est ouvert par défaut, les autres fermés (réduit le scroll
    initial).
    """
    from html import escape as _e
    parts: list[str] = []
    parts.append(
        f'<h3 style="margin-top:1.5em">{_e(view_title)}</h3>'
    )
    if view_note:
        parts.append(
            f'<p style="font-size:.82rem;color:var(--text-muted);'
            f'margin:.2em 0 1em">{_e(view_note)}</p>'
        )
    for i, (title, html) in enumerate(blocks):
        open_attr = " open" if i == 0 else ""
        parts.append(
            f'<details{open_attr} style="margin-bottom:1em">'
            f'<summary style="cursor:pointer;font-weight:600;'
            f'padding:.4em 0">{_e(title)}</summary>'
            f'<div style="margin-top:.5em">{html}</div>'
            f'</details>'
        )
    return "\n".join(parts)


__all__ = ["build_economics_view_html"]
