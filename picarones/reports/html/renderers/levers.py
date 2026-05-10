"""Rendu HTML de la section « Leviers d'amélioration » — Sprint 82.

A.I.9 du plan d'évolution 2026.

Suite directe ``picarones/core/levers.py``.  Pattern identique aux
autres rendus (Sprints 41/43/62/67/72/74/75/76/77/80) : **server-
side**, pas de JavaScript, anti-injection systématique.

Vue
---
Une section composée de **cards** : une par levier, triée par
importance décroissante.  Chaque card affiche :

- une *étiquette* (libellé i18n du type de levier) ;
- une *phrase factuelle* qui réutilise les chiffres du
  ``payload`` (anti-hallucination : aucun chiffre n'est calculé
  dans le rendu) ;
- éventuellement un **détail compact** (top-N tokens, top-3
  classes, etc.) ;
- une *note* d'importance : HIGH / MEDIUM / LOW.

Aucune classification automatique « bon » / « mauvais » et aucune
recommandation : la phrase est purement descriptive.
"""

from __future__ import annotations

import logging
from html import escape as _e
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def _lever_label(lever_type: str, labels: dict[str, str]) -> str:
    return labels.get(f"levers_label_{lever_type}", lever_type)


def _format_dominant_recoverable(payload: dict, labels: dict[str, str]) -> str:
    engine = _e(str(payload.get("engine", "?")))
    pct = payload.get("share_recoverable_pct")
    n_recov = payload.get("n_recoverable")
    n_total = payload.get("n_total_errors")
    template = labels.get(
        "levers_dominant_recoverable_phrase",
        "{pct}% des erreurs de {engine} ({n_recov}/{n_total}) sont "
        "classifiées récupérables (case_error, ligature_error, "
        "abbreviation_error).",
    )
    sentence = template.format(
        engine=engine,
        pct=pct,
        n_recov=n_recov,
        n_total=n_total,
    )
    top_classes = payload.get("top_classes") or []
    if top_classes:
        breakdown = ", ".join(
            f"{_e(str(c.get('class', '?')))} ({c.get('count', 0)})"
            for c in top_classes
        )
        detail_label = labels.get("levers_top_classes", "Principales :")
        sentence += (
            f' <span style="opacity:.8">— {_e(detail_label)} '
            f'{breakdown}</span>'
        )
    return sentence


def _format_pareto_concentration(payload: dict, labels: dict[str, str]) -> str:
    engine = _e(str(payload.get("engine", "?")))
    n_top = payload.get("n_docs_top")
    n_total = payload.get("n_docs")
    top_pct = payload.get("top_share_pct")
    cer_pct = payload.get("cer_share_pct")
    template = labels.get(
        "levers_pareto_phrase",
        "Sur {engine}, {n_top} documents ({top_pct}% du corpus) "
        "concentrent {cer_pct}% du CER cumulé "
        "(sur {n_total} documents au total).",
    )
    return template.format(
        engine=engine,
        n_top=n_top,
        n_total=n_total,
        top_pct=top_pct,
        cer_pct=cer_pct,
    )


def _format_complementarity(payload: dict, labels: dict[str, str]) -> str:
    abs_pct = payload.get("absolute_gap_pct")
    rel_pct = payload.get("relative_gap_pct")
    best_engine = payload.get("best_engine")
    if best_engine:
        template = labels.get(
            "levers_complementarity_phrase_with_engine",
            "L'oracle bag-of-words atteint un rappel supérieur de "
            "{abs_pct} points (+{rel_pct}% relatif) à celui du meilleur "
            "moteur seul ({best_engine}).",
        )
        return template.format(
            abs_pct=abs_pct,
            rel_pct=rel_pct,
            best_engine=_e(str(best_engine)),
        )
    template = labels.get(
        "levers_complementarity_phrase",
        "L'oracle bag-of-words atteint un rappel supérieur de "
        "{abs_pct} points (+{rel_pct}% relatif) à celui du meilleur "
        "moteur seul.",
    )
    return template.format(abs_pct=abs_pct, rel_pct=rel_pct)


def _format_lexical_modernization(payload: dict, labels: dict[str, str]) -> str:
    engine = _e(str(payload.get("engine", "?")))
    top_tokens = payload.get("top_tokens") or []
    if not top_tokens:
        return ""
    items = ", ".join(
        f"{_e(str(t.get('gt_token', '?')))} "
        f"({t.get('rate_modernized_pct', 0)}%, "
        f"n={t.get('n_total', 0)})"
        for t in top_tokens
    )
    template = labels.get(
        "levers_lexical_phrase",
        "Top tokens GT systématiquement modernisés par {engine} : {items}.",
    )
    return template.format(engine=engine, items=items)


def _format_robustness_projection(payload: dict, labels: dict[str, str]) -> str:
    engine = _e(str(payload.get("engine", "?")))
    deficit_pct = payload.get("total_expected_deficit_pct")
    n_types = payload.get("n_degradation_types", 0)
    worst_type = payload.get("worst_degradation_type")
    worst_pct = payload.get("worst_degradation_deficit_pct")
    if worst_type and worst_pct is not None:
        template = labels.get(
            "levers_robustness_phrase_with_worst",
            "Déficit projeté de {engine} sur le corpus réel : "
            "{deficit_pct} points de CER cumulés sur {n_types} "
            "dégradations — pire dégradation : {worst_type} "
            "({worst_pct} points).",
        )
        return template.format(
            engine=engine,
            deficit_pct=deficit_pct,
            n_types=n_types,
            worst_type=_e(str(worst_type)),
            worst_pct=worst_pct,
        )
    template = labels.get(
        "levers_robustness_phrase",
        "Déficit projeté de {engine} sur le corpus réel : "
        "{deficit_pct} points de CER cumulés sur {n_types} dégradations.",
    )
    return template.format(
        engine=engine, deficit_pct=deficit_pct, n_types=n_types,
    )


_FORMATTERS = {
    "dominant_recoverable_class": _format_dominant_recoverable,
    "pareto_concentration": _format_pareto_concentration,
    "complementarity_observation": _format_complementarity,
    "lexical_modernization_observation": _format_lexical_modernization,
    "robustness_projection_observation": _format_robustness_projection,
}


def _importance_label(importance: int, labels: dict[str, str]) -> str:
    if importance >= 70:
        return labels.get("levers_importance_high", "Important")
    if importance >= 40:
        return labels.get("levers_importance_medium", "À noter")
    return labels.get("levers_importance_low", "Mineur")


def _importance_color(importance: int) -> str:
    if importance >= 70:
        return "#c2410c"  # orange profond
    if importance >= 40:
        return "#0369a1"  # bleu
    return "#6b7280"      # gris


def build_levers_section_html(
    levers: Iterable,
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit la section HTML des leviers.

    Parameters
    ----------
    levers:
        Itérable de ``Lever`` (ou de dicts avec ``type``,
        ``importance``, ``payload``).
    labels:
        Dict i18n. Clés attendues sous le préfixe ``levers_``.

    Returns
    -------
    str
        Section HTML, ou ``""`` si aucun levier exploitable.
    """
    labels = labels or {}
    cards: list[str] = []
    for lever in levers:
        # Accepter Lever ou dict
        if hasattr(lever, "as_dict"):
            data = lever.as_dict()
        elif isinstance(lever, dict):
            data = lever
        else:
            continue
        lv_type = data.get("type")
        importance = int(data.get("importance") or 0)
        payload = data.get("payload") or {}
        if not lv_type:
            continue
        formatter = _FORMATTERS.get(lv_type)
        if formatter is None:
            continue
        try:
            sentence = formatter(payload, labels)
        except Exception as exc:  # noqa: BLE001 — un formatter cassé ne doit pas casser la section
            logger.warning(
                "[levers_render] formatter %r a échoué sur payload=%r : %s — "
                "ce levier sera omis du rapport",
                lv_type, payload, exc,
            )
            continue
        if not sentence:
            continue
        type_label = _lever_label(lv_type, labels)
        imp_label = _importance_label(importance, labels)
        imp_color = _importance_color(importance)
        cards.append(
            '<div class="lever-card" style="border:1px solid #e5e7eb;'
            'border-left:4px solid ' + imp_color + ';'
            'border-radius:.4rem;padding:.7rem .9rem;'
            'margin:.5rem 0;background:#fafafa">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;margin-bottom:.3rem;font-size:.8rem">'
            f'<span style="font-weight:600;text-transform:uppercase;'
            f'letter-spacing:.5px;color:#374151">'
            f'{_e(type_label)}</span>'
            f'<span style="color:{imp_color};font-weight:600">'
            f'{_e(imp_label)}</span>'
            f'</div>'
            f'<div style="font-size:.95rem;line-height:1.45">'
            f'{sentence}</div>'
            '</div>'
        )

    if not cards:
        return ""

    title = labels.get("levers_title", "Leviers d'amélioration")
    note = labels.get(
        "levers_note",
        "Observations factuelles synthétisées depuis les modules "
        "d'analyse. Aucune recommandation imposée — c'est au "
        "chercheur de juger ce qui est exploitable selon son "
        "workflow.",
    )

    parts = [
        '<section class="levers-section" style="margin:1.5rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.6rem">'
        f'{_e(note)}</div>',
    ]
    parts.extend(cards)
    parts.append('</section>')
    return "".join(parts)


__all__ = [
    "build_levers_section_html",
]
