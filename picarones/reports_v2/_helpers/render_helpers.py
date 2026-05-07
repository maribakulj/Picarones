"""Helpers de rendu mutualisés.

Phase 5 — module relocalisé depuis
``picarones.report.render_helpers`` vers
``picarones.reports_v2._helpers.render_helpers``.  Le chemin legacy
reste disponible via un shim avec ``DeprecationWarning`` ;
suppression prévue en 2.0.

Centralise les fonctions de coloration et le builder de grille SVG qui
étaient auparavant dupliqués dans chaque ``*_render.py``. Avant cette
consolidation, le projet comptait 25 versions différentes de
``_color_for_*`` (toutes des dégradés rouge/jaune/vert ou blanc/couleur
légèrement différentes) et 2 versions de ``_build_heatmap_svg``
(matrice de classes × positions). Le test
``tests/architecture/test_render_helpers.py`` mesure cette duplication
et bloque sa réapparition.

API
---
- :func:`color_traffic_light` — gradient rouge → jaune → vert. Couvre
  la majorité des cellules du rapport (CER, F1, recall, ECE, deficit,
  drag, CV, etc.). Argument ``low_is_good`` pour inverser la sémantique.
- :func:`color_single_gradient` — gradient blanc → couleur intense.
  Utilisé pour les heatmaps Jaccard, densité, lexical modernization.
- :func:`color_diverging` — gradient signé (négatif → neutre → positif).
  Utilisé pour les deltas Flesch, amélioration nette, sur/sous-norm.
- :func:`text_color_for_bg` — noir ou blanc selon la luminosité du fond.
- :func:`build_grid_svg` — builder de heatmap SVG paramétré.

Conventions de bornes
---------------------
Trois conventions de paramétrage cohabitent (par dessein, pas par
maladresse) :

- :func:`color_traffic_light` accepte ``scale_min`` + ``scale_max``
  parce que les cellules concernées (CER, ECE, deficit) peuvent
  démarrer à une borne basse non nulle (rang 1 = vert, ou
  ``scale_min=0.30`` pour démarrer le dégradé à partir d'un seuil).
- :func:`color_single_gradient` accepte ``max_value`` parce que ces
  cellules (Jaccard, densité) sont toujours bornées en bas par 0 —
  pas besoin de ``scale_min``.
- :func:`color_diverging` accepte ``max_abs`` parce que ces cellules
  (deltas signés) sont symétriques autour de 0 — la borne est la
  même des deux côtés.

Le choix des couleurs reflète la sémantique métier :

- **Traffic-light** rouge/jaune/vert : convention historique
  largement comprise pour vision trichromate normale. **Compromis
  d'accessibilité accepté** : la confusion rouge/vert affecte ~8 %
  des hommes (deutéranopie/protanopie). Une migration vers la
  palette Okabe-Ito de :mod:`picarones.report.colors` est tracée
  comme dette dans un sprint dédié.
- **Diverging** bleu/vert/orange par défaut : vert au centre =
  neutre, extrémités opposées sémantiquement, et ces 3 teintes
  restent distinguables en daltonisme deutéranope. Choix retenu
  parce que les cellules diverging sont moins nombreuses et
  qu'on a pu repartir de zéro en les écrivant.

Palette
-------
Les bornes RGB des dégradés traffic-light sont la moyenne des palettes
ad hoc qui peuplaient les 25 helpers d'origine. Cohérence visuelle
unifiée tout en restant proche du rendu antérieur (≤ 10 unités RGB
d'écart sur la majorité des bornes), pour ne pas casser les tests
d'intégration HTML existants.
"""

from __future__ import annotations

from html import escape as _e
from typing import Callable, Optional


# ──────────────────────────────────────────────────────────────────
# Palettes — bornes RGB partagées par tous les dégradés.
#
# Choix éditorial : on conserve l'esprit « rouge → jaune → vert » des
# helpers historiques plutôt que la palette daltonien-friendly
# Okabe-Ito de ``colors.py`` (utilisée pour les badges principaux).
# Migrer les cellules de tableau vers Okabe-Ito serait un sprint
# d'accessibilité dédié, hors scope de la consolidation.
# ──────────────────────────────────────────────────────────────────
GRADIENT_RED_RGB: tuple[int, int, int] = (220, 100, 100)
GRADIENT_YELLOW_RGB: tuple[int, int, int] = (240, 220, 130)
GRADIENT_GREEN_RGB: tuple[int, int, int] = (130, 200, 130)

#: Couleurs cibles pour les single-gradients fréquents.
GRADIENT_TARGET_BLUE: tuple[int, int, int] = (30, 58, 138)      # Jaccard, specialization
GRADIENT_TARGET_ORANGE: tuple[int, int, int] = (194, 65, 12)    # densité, lexical mod.
GRADIENT_TARGET_RED: tuple[int, int, int] = (200, 60, 60)       # divergence inter-engine

#: Couleurs cibles pour les diverging gradients.
DIVERGING_NEGATIVE_RGB: tuple[int, int, int] = (95, 145, 215)   # bleu (under-norm)
DIVERGING_NEUTRAL_RGB: tuple[int, int, int] = (130, 200, 130)   # vert (centre, OK)
DIVERGING_POSITIVE_RGB: tuple[int, int, int] = (220, 130, 60)   # orange (over-norm)


# ──────────────────────────────────────────────────────────────────
# Helpers internes
# ──────────────────────────────────────────────────────────────────
def _interp(a: int, b: int, t: float) -> int:
    """Interpolation linéaire bornée à un canal RGB ∈ [0, 255]."""
    return max(0, min(255, int(a + (b - a) * t)))


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


# ──────────────────────────────────────────────────────────────────
# API publique : couleurs
# ──────────────────────────────────────────────────────────────────
def color_traffic_light(
    value: float,
    *,
    low_is_good: bool = False,
    scale_max: float = 1.0,
    scale_min: float = 0.0,
) -> str:
    """Gradient rouge → jaune → vert proportionnel à ``value``.

    Paramètres
    ----------
    value : float
        Valeur à colorer.
    low_is_good : bool, default ``False``
        Si ``True``, ``value = scale_min`` → vert et ``value = scale_max``
        → rouge (sémantique « plus c'est bas, mieux c'est » : ECE,
        deficit, drag, CV, taux d'introduction d'erreurs…).
        Si ``False`` (défaut), c'est l'inverse (sémantique « plus c'est
        haut, mieux c'est » : F1, recall, taux de correction…).
    scale_max : float, default ``1.0``
        Borne haute de l'échelle. Au-delà, la couleur sature.
    scale_min : float, default ``0.0``
        Borne basse de l'échelle.

    Retour
    ------
    str
        Couleur hex au format ``#rrggbb``.
    """
    span = scale_max - scale_min
    if span <= 0:
        f = 0.5
    else:
        f = (value - scale_min) / span
    f = max(0.0, min(1.0, f))
    if low_is_good:
        f = 1.0 - f
    if f <= 0.5:
        t = f / 0.5
        r = _interp(GRADIENT_RED_RGB[0], GRADIENT_YELLOW_RGB[0], t)
        g = _interp(GRADIENT_RED_RGB[1], GRADIENT_YELLOW_RGB[1], t)
        b = _interp(GRADIENT_RED_RGB[2], GRADIENT_YELLOW_RGB[2], t)
    else:
        t = (f - 0.5) / 0.5
        r = _interp(GRADIENT_YELLOW_RGB[0], GRADIENT_GREEN_RGB[0], t)
        g = _interp(GRADIENT_YELLOW_RGB[1], GRADIENT_GREEN_RGB[1], t)
        b = _interp(GRADIENT_YELLOW_RGB[2], GRADIENT_GREEN_RGB[2], t)
    return _rgb_to_hex(r, g, b)


def color_single_gradient(
    value: float,
    *,
    end_rgb: tuple[int, int, int],
    max_value: float = 1.0,
    start_rgb: tuple[int, int, int] = (255, 255, 255),
) -> str:
    """Gradient simple ``start_rgb`` → ``end_rgb`` proportionnel à ``value/max_value``.

    Utilisé pour les heatmaps qui n'ont pas de sémantique « bon/mauvais »
    mais juste une intensité (Jaccard, densité d'occurrence, taux de
    modernisation lexicale).
    """
    if max_value <= 0:
        f = 0.0
    else:
        f = max(0.0, min(1.0, value / max_value))
    r = _interp(start_rgb[0], end_rgb[0], f)
    g = _interp(start_rgb[1], end_rgb[1], f)
    b = _interp(start_rgb[2], end_rgb[2], f)
    return _rgb_to_hex(r, g, b)


def color_diverging(
    value: float,
    *,
    max_abs: float = 1.0,
    negative_rgb: tuple[int, int, int] = DIVERGING_NEGATIVE_RGB,
    neutral_rgb: tuple[int, int, int] = DIVERGING_NEUTRAL_RGB,
    positive_rgb: tuple[int, int, int] = DIVERGING_POSITIVE_RGB,
) -> str:
    """Gradient signé : ``value < 0`` → ``negative_rgb`` (par défaut bleu),
    ``value ≈ 0`` → ``neutral_rgb`` (par défaut vert),
    ``value > 0`` → ``positive_rgb`` (par défaut orange).

    Saturation à ``|value| = max_abs``.
    """
    if max_abs <= 0:
        return _rgb_to_hex(*neutral_rgb)
    f = max(-1.0, min(1.0, value / max_abs))
    if f >= 0:
        r = _interp(neutral_rgb[0], positive_rgb[0], f)
        g = _interp(neutral_rgb[1], positive_rgb[1], f)
        b = _interp(neutral_rgb[2], positive_rgb[2], f)
    else:
        t = -f
        r = _interp(neutral_rgb[0], negative_rgb[0], t)
        g = _interp(neutral_rgb[1], negative_rgb[1], t)
        b = _interp(neutral_rgb[2], negative_rgb[2], t)
    return _rgb_to_hex(r, g, b)


def text_color_for_bg(intensity: float, *, threshold: float = 0.55) -> str:
    """Retourne ``"#fff"`` sur fond foncé, ``"#222"`` sur fond clair.

    ``intensity`` ∈ [0, 1] : 0 = fond clair, 1 = fond très foncé.
    Pour les heatmaps single-gradient, c'est typiquement la même valeur
    que celle passée à :func:`color_single_gradient`.
    """
    return "#fff" if intensity > threshold else "#222"


# ──────────────────────────────────────────────────────────────────
# API publique : barème CER par paliers (badges du rapport)
# ──────────────────────────────────────────────────────────────────
#
# Les badges de qualité du rapport (galerie, tableau de classement)
# n'utilisent pas un dégradé continu mais un barème discret à 4
# paliers calibrés sur les seuils éditoriaux usuels :
#
#   < 5 %  : vert    (qualité publication directe)
#   < 15 % : jaune   (relecture humaine légère)
#   < 30 % : orange  (relecture humaine systématique)
#   ≥ 30 % : rouge   (catastrophique, à reprendre)
#
# Les couleurs sont importées de :mod:`picarones.report.colors`
# (palette Okabe-Ito daltonien-friendly active par défaut).


def cer_step_color(cer: float) -> str:
    """Couleur de texte CSS pour un score CER, par paliers.

    Voir le barème dans le bloc de documentation ci-dessus.
    """
    from picarones.reports_v2._helpers.colors import (
        COLOR_GREEN,
        COLOR_ORANGE,
        COLOR_RED,
        COLOR_YELLOW,
    )
    if cer < 0.05:
        return COLOR_GREEN
    if cer < 0.15:
        return COLOR_YELLOW
    if cer < 0.30:
        return COLOR_ORANGE
    return COLOR_RED


def cer_step_bg(cer: float) -> str:
    """Couleur de fond CSS associée à :func:`cer_step_color`."""
    from picarones.reports_v2._helpers.colors import (
        BG_GREEN,
        BG_ORANGE,
        BG_RED,
        BG_YELLOW,
    )
    if cer < 0.05:
        return BG_GREEN
    if cer < 0.15:
        return BG_YELLOW
    if cer < 0.30:
        return BG_ORANGE
    return BG_RED


# ──────────────────────────────────────────────────────────────────
# API publique : grille SVG
# ──────────────────────────────────────────────────────────────────
def build_grid_svg(
    *,
    n_rows: int,
    n_cols: int,
    row_label_fn: Callable[[int], str],
    col_label_fn: Callable[[int], str],
    cell_color_fn: Callable[[int, int], str],
    cell_text_fn: Callable[[int, int], Optional[str]] = lambda r, c: None,
    cell_text_color_fn: Callable[[int, int], str] = lambda r, c: "#222",
    cell_w: int = 36,
    cell_h: int = 36,
    label_left: int = 130,
    label_top: int = 80,
    rotate_col_labels: bool = False,
    aria_label: str = "Heatmap",
    x_axis_title: Optional[str] = None,
) -> str:
    """Construit une heatmap SVG paramétrable.

    Architecture commune des deux `_build_heatmap_svg` historiques
    (taxonomy_cooccurrence et taxonomy_intra_doc), mutualisée ici.

    Paramètres
    ----------
    n_rows, n_cols : int
        Dimensions de la grille.
    row_label_fn, col_label_fn : Callable[[int], str]
        Étiquettes des lignes (gauche) et colonnes (haut).
    cell_color_fn : Callable[[int, int], str]
        Retourne la couleur hex de fond pour la cellule (row, col).
    cell_text_fn : Callable[[int, int], Optional[str]]
        Texte à afficher dans la cellule, ou ``None`` pour ne rien afficher.
    cell_text_color_fn : Callable[[int, int], str]
        Couleur du texte de la cellule (typiquement obtenue via
        :func:`text_color_for_bg`).
    cell_w, cell_h : int
        Dimensions de chaque cellule en pixels.
    label_left, label_top : int
        Marges réservées aux étiquettes.
    rotate_col_labels : bool
        Si ``True``, les étiquettes de colonnes sont rotées de -45°
        (utile quand elles sont longues).
    aria_label : str
        Étiquette d'accessibilité du SVG.
    x_axis_title : Optional[str]
        Titre optionnel de l'axe horizontal, affiché en bas du SVG.

    Retour
    ------
    str
        SVG complet, ou ``""`` si la grille est vide.
    """
    if n_rows == 0 or n_cols == 0:
        return ""

    extra_bottom = 30 if x_axis_title else 10
    width = label_left + n_cols * cell_w + 10
    height = label_top + n_rows * cell_h + extra_bottom

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{_e(aria_label)}">',
    ]

    # Étiquettes de colonnes
    for j in range(n_cols):
        cx = label_left + j * cell_w + cell_w // 2
        cy = label_top - 6
        label = _e(col_label_fn(j))
        if rotate_col_labels:
            parts.append(
                f'<text x="{cx}" y="{cy}" '
                f'transform="rotate(-45 {cx} {cy})" '
                f'font-size="11" fill="#333" text-anchor="start">'
                f'{label}</text>'
            )
        else:
            parts.append(
                f'<text x="{cx}" y="{cy}" '
                f'font-size="10" fill="#666" text-anchor="middle">'
                f'{label}</text>'
            )

    # Cellules + étiquettes de lignes
    for i in range(n_rows):
        rx = label_left - 6
        ry = label_top + i * cell_h + cell_h // 2 + 4
        parts.append(
            f'<text x="{rx}" y="{ry}" '
            f'font-size="11" fill="#333" text-anchor="end">'
            f'{_e(row_label_fn(i))}</text>'
        )
        for j in range(n_cols):
            x = label_left + j * cell_w
            y = label_top + i * cell_h
            color = cell_color_fn(i, j)
            parts.append(
                f'<rect x="{x}" y="{y}" '
                f'width="{cell_w}" height="{cell_h}" '
                f'fill="{color}" stroke="#ddd" stroke-width="0.5"/>'
            )
            text = cell_text_fn(i, j)
            if text is not None:
                text_color = cell_text_color_fn(i, j)
                parts.append(
                    f'<text x="{x + cell_w // 2}" '
                    f'y="{y + cell_h // 2 + 4}" '
                    f'font-size="10" fill="{text_color}" '
                    f'text-anchor="middle">'
                    f'{_e(text)}</text>'
                )

    if x_axis_title:
        cx_axis = label_left + (n_cols * cell_w) // 2
        cy_axis = height - 6
        parts.append(
            f'<text x="{cx_axis}" y="{cy_axis}" '
            f'font-size="11" fill="#666" text-anchor="middle" '
            f'font-style="italic">'
            f'{_e(x_axis_title)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


__all__ = [
    "GRADIENT_RED_RGB",
    "GRADIENT_YELLOW_RGB",
    "GRADIENT_GREEN_RGB",
    "GRADIENT_TARGET_BLUE",
    "GRADIENT_TARGET_ORANGE",
    "GRADIENT_TARGET_RED",
    "DIVERGING_NEGATIVE_RGB",
    "DIVERGING_NEUTRAL_RGB",
    "DIVERGING_POSITIVE_RGB",
    "cer_step_color",
    "cer_step_bg",
    "color_traffic_light",
    "color_single_gradient",
    "color_diverging",
    "text_color_for_bg",
    "build_grid_svg",
]
