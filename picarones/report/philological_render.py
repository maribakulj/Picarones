"""Rendu HTML server-side du profil philologique (Sprint 62).

Suite directe Sprint 61 (câblage backend) — produit les blocs HTML
qui exposent les six modules philologiques (Sprints 55-60) dans le
rapport :

- ``unicode_blocks``    (Sprint 55) — précision par bloc Unicode
- ``abbreviations``     (Sprint 56) — score strict + expansion par
                                       abréviation médiévale Capelli
- ``mufi``              (Sprint 57) — couverture MUFI globale + par
                                       caractère
- ``early_modern``      (Sprint 58) — préservation des marqueurs
                                       typographiques imprimé ancien
- ``modern_archives``   (Sprint 59) — strict + expansion par
                                       catégorie d'archive moderne
- ``roman_numerals``    (Sprint 60) — breakdown 5 statuts de
                                       restitution

Principe identique aux Sprints 41 (NER) et 43 (calibration) :

- Rendu **server-side**, pas de JavaScript, déterministe.
- Section adaptive : si aucun moteur n'a de signal pour un module
  donné, la sous-section est silencieusement omise.
- Si **aucun module** n'a de signal sur l'ensemble des moteurs,
  ``build_philological_profile_html`` retourne une chaîne vide et
  le bloc complet n'apparaît pas dans la vue analyses.
- **Aucune classification automatique** : on affiche les chiffres
  bruts par catégorie/bloc/statut, le chercheur juge lui-même la
  convention adoptée.
- Anti-injection : tous les noms de moteurs, catégories, statuts,
  caractères passent par ``html.escape`` avant insertion.
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional

from picarones.reports_v2._helpers.render_helpers import color_traffic_light


# ──────────────────────────────────────────────────────────────────────────
# Helpers de coloration
# ──────────────────────────────────────────────────────────────────────────


def _engines_with_module(
    engines_summary: list[dict], module: str,
) -> list[dict]:
    """Filtre les moteurs ayant des données pour le module donné."""
    out: list[dict] = []
    for eng in engines_summary:
        agg = eng.get("aggregated_philological") or {}
        if module in agg and agg[module]:
            out.append(eng)
    return out


def _score_cell(score: Optional[float], extra: str = "") -> str:
    """Rend une cellule colorée.  ``None`` → cellule grise « — »."""
    if score is None:
        return (
            '<td style="padding:.3rem .5rem;text-align:center;'
            'background:#f0f0f0;color:#999">—</td>'
        )
    color = color_traffic_light(score)
    text = f"{score * 100:.1f}%"
    if extra:
        text += f" <span style=\"opacity:.6;font-size:.85em\">({_e(extra)})</span>"
    return (
        f'<td style="padding:.3rem .5rem;text-align:center;'
        f'background:{color}">{text}</td>'
    )


def _table_header(
    columns: list[str], engine_label: str,
) -> str:
    """Construit l'entête d'un tableau moteur × colonnes."""
    parts = [
        '<thead><tr>',
        f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:left;'
        f'border-bottom:1px solid var(--border);font-weight:600">'
        f'{_e(engine_label)}</th>',
    ]
    for col in columns:
        parts.append(
            f'<th scope=\"col\" style="padding:.3rem .5rem;text-align:center;'
            f'border-bottom:1px solid var(--border);font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append('</tr></thead>')
    return "".join(parts)


def _engine_label_cell(name: str) -> str:
    return (
        f'<td style="padding:.3rem .5rem;font-weight:500;'
        f'border-bottom:1px solid var(--border-light)">{_e(name)}</td>'
    )


def _section_open(title: str, note: str = "") -> str:
    parts = [
        '<div class="philological-section" '
        'style="margin:1rem 0;padding:.75rem;'
        'background:var(--bg-secondary);border-radius:6px">',
        f'<div style="font-weight:600;margin-bottom:.4rem">{_e(title)}</div>',
    ]
    if note:
        parts.append(
            f'<div style="font-size:.8rem;opacity:.75;margin-bottom:.5rem">'
            f'{_e(note)}</div>'
        )
    return "".join(parts)


def _section_close() -> str:
    return "</div>"


def _table_open() -> str:
    return (
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.85rem">'
    )


def _table_close() -> str:
    return "</table>"


# ──────────────────────────────────────────────────────────────────────────
# Sprint 55 — Précision par bloc Unicode
# ──────────────────────────────────────────────────────────────────────────


def build_unicode_blocks_section(
    engines_summary: list[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    relevant = _engines_with_module(engines_summary, "unicode_blocks")
    if not relevant:
        return ""
    labels = labels or {}
    title = labels.get(
        "philo_unicode_blocks_title", "Précision par bloc Unicode",
    )
    note = labels.get(
        "philo_unicode_blocks_note",
        "Pourcentage de caractères correctement restitués par bloc "
        "Unicode rencontré dans la GT (hors Basic Latin).",
    )
    engine_label = labels.get("philo_engine_label", "Moteur")
    global_label = labels.get("philo_global_label", "Global")

    # Collecte tous les blocs présents (hors Basic Latin déjà filtré
    # par adaptive masking, mais on défilte ici si Basic Latin
    # apparaît malgré tout chez certains moteurs).
    all_blocks: set[str] = set()
    for eng in relevant:
        per_block = eng["aggregated_philological"]["unicode_blocks"].get(
            "per_block", {},
        )
        for block in per_block:
            if block != "Basic Latin":
                all_blocks.add(block)
    blocks = sorted(all_blocks)
    if not blocks:
        return ""

    parts = [_section_open(title, note), _table_open()]
    parts.append(_table_header([global_label] + blocks, engine_label))
    parts.append("<tbody>")
    for eng in relevant:
        agg = eng["aggregated_philological"]["unicode_blocks"]
        global_acc = agg.get("global_accuracy", 0.0)
        n_chars = agg.get("n_chars_total", 0)
        parts.append("<tr>")
        parts.append(_engine_label_cell(eng["name"]))
        parts.append(_score_cell(global_acc, extra=f"n={n_chars}"))
        per_block = agg.get("per_block", {})
        for block in blocks:
            stats = per_block.get(block)
            if stats and stats.get("total", 0) > 0:
                parts.append(_score_cell(
                    stats["accuracy"], extra=f"n={stats['total']}",
                ))
            else:
                parts.append(_score_cell(None))
        parts.append("</tr>")
    parts.append("</tbody>")
    parts.append(_table_close())
    parts.append(_section_close())
    return "".join(parts)


# (sections suivantes définies plus loin)


# ──────────────────────────────────────────────────────────────────────────
# Sprint 56 — Abréviations Capelli médiévales
# ──────────────────────────────────────────────────────────────────────────


def build_abbreviations_section(
    engines_summary: list[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    relevant = _engines_with_module(engines_summary, "abbreviations")
    if not relevant:
        return ""
    labels = labels or {}
    title = labels.get(
        "philo_abbreviations_title",
        "Abréviations médiévales (Capelli)",
    )
    note = labels.get(
        "philo_abbreviations_note",
        "Strict = forme abrégée (ꝑ, ꝓ, ⁊…) préservée telle quelle ; "
        "Expansion = abrégée OU forme développée (per, pro, et…) "
        "présente. Le ratio strict/expansion par moteur indique la "
        "convention adoptée (diplomatique / modernisante).",
    )
    engine_label = labels.get("philo_engine_label", "Moteur")
    strict_label = labels.get("philo_strict_label", "Strict")
    expansion_label = labels.get("philo_expansion_label", "Expansion")
    n_label = labels.get("philo_n_total_label", "n total")

    parts = [_section_open(title, note), _table_open()]
    parts.append(_table_header(
        [strict_label, expansion_label, n_label], engine_label,
    ))
    parts.append("<tbody>")
    for eng in relevant:
        agg = eng["aggregated_philological"]["abbreviations"]
        parts.append("<tr>")
        parts.append(_engine_label_cell(eng["name"]))
        parts.append(_score_cell(agg.get("global_strict_score", 0.0)))
        parts.append(_score_cell(agg.get("global_expansion_score", 0.0)))
        parts.append(
            f'<td style="padding:.3rem .5rem;text-align:center">'
            f'{agg.get("n_abbreviations_in_reference", 0)}</td>'
        )
        parts.append("</tr>")
    parts.append("</tbody>")
    parts.append(_table_close())
    parts.append(_section_close())
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Sprint 57 — Couverture MUFI
# ──────────────────────────────────────────────────────────────────────────


def build_mufi_section(
    engines_summary: list[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    relevant = _engines_with_module(engines_summary, "mufi")
    if not relevant:
        return ""
    labels = labels or {}
    title = labels.get(
        "philo_mufi_title",
        "Couverture MUFI (Medieval Unicode Font Initiative)",
    )
    note = labels.get(
        "philo_mufi_note",
        "Taux de caractères MUFI de la GT (þ, ð, ƿ, ſ, æ, lettres "
        "PUA…) correctement restitués dans l'OCR. Critère éditorial "
        "central pour les médiévistes.",
    )
    engine_label = labels.get("philo_engine_label", "Moteur")
    coverage_label = labels.get("philo_mufi_coverage_label", "Couverture")
    n_label = labels.get("philo_n_total_label", "n total")

    parts = [_section_open(title, note), _table_open()]
    parts.append(_table_header(
        [coverage_label, n_label], engine_label,
    ))
    parts.append("<tbody>")
    for eng in relevant:
        agg = eng["aggregated_philological"]["mufi"]
        parts.append("<tr>")
        parts.append(_engine_label_cell(eng["name"]))
        parts.append(_score_cell(agg.get("coverage", 0.0)))
        parts.append(
            f'<td style="padding:.3rem .5rem;text-align:center">'
            f'{agg.get("n_mufi_chars_reference", 0)}</td>'
        )
        parts.append("</tr>")
    parts.append("</tbody>")
    parts.append(_table_close())
    parts.append(_section_close())
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Sprint 58 — Marqueurs typographiques imprimé ancien (heatmap)
# ──────────────────────────────────────────────────────────────────────────


def build_early_modern_section(
    engines_summary: list[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    relevant = _engines_with_module(engines_summary, "early_modern")
    if not relevant:
        return ""
    labels = labels or {}
    title = labels.get(
        "philo_early_modern_title",
        "Marqueurs typographiques imprimé ancien (XVIᵉ-XVIIIᵉ)",
    )
    note = labels.get(
        "philo_early_modern_note",
        "Préservation des ligatures (ﬁ ﬂ ﬀ), s long (ſ), i sans "
        "point (ı), esperluette (&) et tildes nasaux (ã õ ñ). "
        "Une ligne par moteur, une colonne par catégorie.",
    )
    engine_label = labels.get("philo_engine_label", "Moteur")
    global_label = labels.get("philo_global_label", "Global")

    all_cats: set[str] = set()
    for eng in relevant:
        all_cats.update(
            eng["aggregated_philological"]["early_modern"]
            .get("per_category", {}).keys(),
        )
    cats = sorted(all_cats)
    if not cats:
        return ""

    parts = [_section_open(title, note), _table_open()]
    parts.append(_table_header([global_label] + cats, engine_label))
    parts.append("<tbody>")
    for eng in relevant:
        agg = eng["aggregated_philological"]["early_modern"]
        n_total = agg.get("n_markers_reference", 0)
        parts.append("<tr>")
        parts.append(_engine_label_cell(eng["name"]))
        parts.append(_score_cell(
            agg.get("global_preservation", 0.0), extra=f"n={n_total}",
        ))
        per_cat = agg.get("per_category", {})
        for cat in cats:
            stats = per_cat.get(cat)
            if stats and stats.get("total", 0) > 0:
                parts.append(_score_cell(
                    stats["preservation"], extra=f"n={stats['total']}",
                ))
            else:
                parts.append(_score_cell(None))
        parts.append("</tr>")
    parts.append("</tbody>")
    parts.append(_table_close())
    parts.append(_section_close())
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Sprint 59 — Archives modernes : strict + expansion par catégorie
# ──────────────────────────────────────────────────────────────────────────


def build_modern_archives_section(
    engines_summary: list[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    relevant = _engines_with_module(engines_summary, "modern_archives")
    if not relevant:
        return ""
    labels = labels or {}
    title = labels.get(
        "philo_modern_archives_title",
        "Abréviations des archives modernes (XIXᵉ-XXᵉ)",
    )
    note = labels.get(
        "philo_modern_archives_note",
        "Strict = abrégé préservé (Mme, S.A.R., bd, vol., …) ; "
        "Expansion = abrégé OU forme développée. Affiché par "
        "catégorie : civilité, ordinaux, monnaie, administratif, "
        "état civil, ponctuation typo, latin, biblio, adresse.",
    )
    engine_label = labels.get("philo_engine_label", "Moteur")
    global_label = labels.get("philo_global_label", "Global")
    strict_label = labels.get("philo_strict_label", "Strict")
    expansion_label = labels.get("philo_expansion_label", "Expansion")

    all_cats: set[str] = set()
    for eng in relevant:
        all_cats.update(
            eng["aggregated_philological"]["modern_archives"]
            .get("per_category", {}).keys(),
        )
    cats = sorted(all_cats)

    parts = [_section_open(title, note)]
    parts.append(
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.85rem">'
    )
    parts.append("<thead><tr>")
    parts.append(
        f'<th scope=\"col\" rowspan="2" style="padding:.3rem .5rem;text-align:left;'
        f'border-bottom:1px solid var(--border);font-weight:600">'
        f'{_e(engine_label)}</th>'
    )
    parts.append(
        f'<th scope=\"col\" colspan="2" style="padding:.3rem .5rem;text-align:center;'
        f'border-bottom:1px solid var(--border);font-weight:600">'
        f'{_e(global_label)}</th>'
    )
    for cat in cats:
        parts.append(
            f'<th scope=\"col\" colspan="2" style="padding:.3rem .5rem;text-align:center;'
            f'border-bottom:1px solid var(--border);font-weight:600">'
            f'{_e(cat)}</th>'
        )
    parts.append("</tr><tr>")
    for _ in range(1 + len(cats)):
        parts.append(
            f'<th scope=\"col\" style="padding:.2rem .4rem;text-align:center;'
            f'font-size:.75rem;font-weight:500;opacity:.7">'
            f'{_e(strict_label)}</th>'
        )
        parts.append(
            f'<th scope=\"col\" style="padding:.2rem .4rem;text-align:center;'
            f'font-size:.75rem;font-weight:500;opacity:.7">'
            f'{_e(expansion_label)}</th>'
        )
    parts.append("</tr></thead>")
    parts.append("<tbody>")
    for eng in relevant:
        agg = eng["aggregated_philological"]["modern_archives"]
        parts.append("<tr>")
        parts.append(_engine_label_cell(eng["name"]))
        parts.append(_score_cell(agg.get("global_strict_score", 0.0)))
        parts.append(_score_cell(agg.get("global_expansion_score", 0.0)))
        per_cat = agg.get("per_category", {})
        for cat in cats:
            stats = per_cat.get(cat)
            if stats and stats.get("n_total", 0) > 0:
                parts.append(_score_cell(
                    stats["strict_score"],
                    extra=f"n={stats['n_total']}",
                ))
                parts.append(_score_cell(stats["expansion_score"]))
            else:
                parts.append(_score_cell(None))
                parts.append(_score_cell(None))
        parts.append("</tr>")
    parts.append("</tbody>")
    parts.append(_table_close())
    parts.append(_section_close())
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Sprint 60 — Numéraux romains : breakdown 5 statuts
# ──────────────────────────────────────────────────────────────────────────


def build_roman_numerals_section(
    engines_summary: list[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    relevant = _engines_with_module(engines_summary, "roman_numerals")
    if not relevant:
        return ""
    labels = labels or {}
    title = labels.get(
        "philo_roman_numerals_title",
        "Numéraux romains : restitution par statut",
    )
    note = labels.get(
        "philo_roman_numerals_note",
        "Pour chaque numéral romain de la GT, statut de restitution : "
        "strict (forme exacte), case_changed (casse modifiée), "
        "j_dropped (j médiéval normalisé), converted_to_arabic, lost. "
        "Le breakdown indique la convention : majoritaire strict → "
        "diplomatique ; majoritaire arabic → modernisation profonde.",
    )
    engine_label = labels.get("philo_engine_label", "Moteur")
    n_label = labels.get("philo_n_total_label", "n total")

    statuses = (
        "strict_preserved", "case_changed", "j_dropped",
        "converted_to_arabic", "lost",
    )
    status_labels = {
        s: labels.get(f"philo_roman_status_{s}", s) for s in statuses
    }

    parts = [_section_open(title, note), _table_open()]
    parts.append(_table_header(
        [n_label] + [status_labels[s] for s in statuses],
        engine_label,
    ))
    parts.append("<tbody>")
    for eng in relevant:
        agg = eng["aggregated_philological"]["roman_numerals"]
        n_total = agg.get("n_numerals_reference", 0)
        per_status = agg.get("per_status", {})
        parts.append("<tr>")
        parts.append(_engine_label_cell(eng["name"]))
        parts.append(
            f'<td style="padding:.3rem .5rem;text-align:center">'
            f'{n_total}</td>'
        )
        for status in statuses:
            count = per_status.get(status, 0)
            if n_total > 0:
                ratio = count / n_total
                # Pour « lost » on inverse la couleur (un haut taux
                # de perte est mauvais).  Pour les autres on garde
                # la sémantique « plus c'est haut, plus l'OCR a
                # adopté ce statut ».
                color = (
                    color_traffic_light(1.0 - ratio) if status == "lost"
                    else color_traffic_light(ratio)
                )
                parts.append(
                    f'<td style="padding:.3rem .5rem;text-align:center;'
                    f'background:{color}">{count} '
                    f'<span style="opacity:.6;font-size:.85em">'
                    f'({ratio * 100:.0f}%)</span></td>'
                )
            else:
                parts.append(_score_cell(None))
        parts.append("</tr>")
    parts.append("</tbody>")
    parts.append(_table_close())
    parts.append(_section_close())
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Agrégateur principal
# ──────────────────────────────────────────────────────────────────────────


def build_philological_profile_html(
    engines_summary: list[dict],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Assemble les six sections en un bloc unique.

    Retourne ``""`` si aucune section n'a de contenu (c.-à-d.
    aucun moteur n'a de signal philologique sur le corpus).
    """
    sections = [
        build_unicode_blocks_section(engines_summary, labels),
        build_abbreviations_section(engines_summary, labels),
        build_mufi_section(engines_summary, labels),
        build_early_modern_section(engines_summary, labels),
        build_modern_archives_section(engines_summary, labels),
        build_roman_numerals_section(engines_summary, labels),
    ]
    non_empty = [s for s in sections if s]
    if not non_empty:
        return ""
    labels = labels or {}
    main_title = labels.get(
        "philo_profile_title", "Profil philologique",
    )
    main_note = labels.get(
        "philo_profile_note",
        "Données brutes par catégorie de marqueur philologique. "
        "L'outil ne classifie pas la convention adoptée par chaque "
        "moteur — c'est au chercheur de lire les chiffres et de "
        "conclure selon ses critères éditoriaux.",
    )
    parts = [
        '<div class="philological-profile">',
        f'<h3 style="margin-top:0">{_e(main_title)}</h3>',
        f'<p style="font-size:.85rem;opacity:.8;margin-bottom:.5rem">'
        f'{_e(main_note)}</p>',
    ]
    parts.extend(non_empty)
    parts.append("</div>")
    return "".join(parts)


__all__ = [
    "build_philological_profile_html",
    "build_unicode_blocks_section",
    "build_abbreviations_section",
    "build_mufi_section",
    "build_early_modern_section",
    "build_modern_archives_section",
    "build_roman_numerals_section",
]
