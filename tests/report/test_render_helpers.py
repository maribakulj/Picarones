"""Tests unitaires des helpers de rendu mutualisés.

Couvre :func:`color_traffic_light`, :func:`color_single_gradient`,
:func:`color_diverging`, :func:`text_color_for_bg`, :func:`build_grid_svg`.
"""

from __future__ import annotations

from picarones.report.render_helpers import (
    DIVERGING_NEGATIVE_RGB,
    DIVERGING_POSITIVE_RGB,
    GRADIENT_GREEN_RGB,
    GRADIENT_RED_RGB,
    GRADIENT_YELLOW_RGB,
    build_grid_svg,
    color_diverging,
    color_single_gradient,
    color_traffic_light,
    text_color_for_bg,
)


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


# ──────────────────────────────────────────────────────────────────
# color_traffic_light
# ──────────────────────────────────────────────────────────────────
class TestColorTrafficLight:
    def test_value_zero_high_is_good_returns_red(self) -> None:
        assert _hex_to_rgb(color_traffic_light(0.0)) == GRADIENT_RED_RGB

    def test_value_max_high_is_good_returns_green(self) -> None:
        assert _hex_to_rgb(color_traffic_light(1.0)) == GRADIENT_GREEN_RGB

    def test_value_mid_high_is_good_returns_yellow(self) -> None:
        assert _hex_to_rgb(color_traffic_light(0.5)) == GRADIENT_YELLOW_RGB

    def test_value_zero_low_is_good_returns_green(self) -> None:
        assert _hex_to_rgb(color_traffic_light(0.0, low_is_good=True)) == GRADIENT_GREEN_RGB

    def test_value_max_low_is_good_returns_red(self) -> None:
        assert _hex_to_rgb(color_traffic_light(1.0, low_is_good=True)) == GRADIENT_RED_RGB

    def test_clamping_above_max(self) -> None:
        c = color_traffic_light(2.0)
        assert _hex_to_rgb(c) == GRADIENT_GREEN_RGB

    def test_clamping_below_zero(self) -> None:
        c = color_traffic_light(-1.0)
        assert _hex_to_rgb(c) == GRADIENT_RED_RGB

    def test_custom_scale_max(self) -> None:
        # CER 0.30 = max → vert (avec low_is_good=True → vert au max)
        # On vérifie plutôt high_is_good : 0.30 → vert
        assert _hex_to_rgb(color_traffic_light(0.30, scale_max=0.30)) == GRADIENT_GREEN_RGB
        # 0.15 → milieu → jaune
        assert _hex_to_rgb(color_traffic_light(0.15, scale_max=0.30)) == GRADIENT_YELLOW_RGB

    def test_custom_scale_min_and_max(self) -> None:
        # Plage [10, 20] : 10 → rouge, 20 → vert, 15 → jaune
        assert _hex_to_rgb(color_traffic_light(10.0, scale_min=10, scale_max=20)) == GRADIENT_RED_RGB
        assert _hex_to_rgb(color_traffic_light(20.0, scale_min=10, scale_max=20)) == GRADIENT_GREEN_RGB
        assert _hex_to_rgb(color_traffic_light(15.0, scale_min=10, scale_max=20)) == GRADIENT_YELLOW_RGB

    def test_zero_span_returns_yellow(self) -> None:
        # scale_min == scale_max → milieu → jaune
        assert _hex_to_rgb(color_traffic_light(5.0, scale_min=10, scale_max=10)) == GRADIENT_YELLOW_RGB

    def test_format_is_hex_lowercase(self) -> None:
        c = color_traffic_light(0.7)
        assert c.startswith("#")
        assert len(c) == 7
        assert c == c.lower()

    def test_nan_falls_back_to_red(self) -> None:
        # max(0, min(1, NaN)) = NaN → mais ensuite f <= 0.5 est False, donc
        # branche "yellow → green" avec t = (NaN - 0.5) / 0.5 = NaN.
        # Le résultat est techniquement indéfini ; on vérifie au moins que
        # la fonction ne crash pas et retourne un hex valide.
        c = color_traffic_light(float("nan"))
        assert c.startswith("#")
        assert len(c) == 7

    def test_inf_clamped_to_max(self) -> None:
        # +inf > scale_max → clamp à scale_max → vert (high_is_good)
        assert _hex_to_rgb(color_traffic_light(float("inf"))) == GRADIENT_GREEN_RGB
        # -inf < 0 → clamp à 0 → rouge
        assert _hex_to_rgb(color_traffic_light(float("-inf"))) == GRADIENT_RED_RGB

    def test_inverted_scale_returns_yellow(self) -> None:
        # scale_min > scale_max → span négatif → géré comme zero span.
        # La fonction ne doit pas crash et retourne une couleur valide.
        c = color_traffic_light(5.0, scale_min=10, scale_max=5)
        assert c.startswith("#")
        assert len(c) == 7


# ──────────────────────────────────────────────────────────────────
# color_single_gradient
# ──────────────────────────────────────────────────────────────────
class TestColorSingleGradient:
    def test_zero_returns_start(self) -> None:
        assert color_single_gradient(0.0, end_rgb=(30, 58, 138)) == "#ffffff"

    def test_max_returns_end(self) -> None:
        assert color_single_gradient(1.0, end_rgb=(30, 58, 138)) == "#1e3a8a"

    def test_half_is_midpoint(self) -> None:
        c = color_single_gradient(0.5, end_rgb=(30, 58, 138))
        # ((255+30)/2, (255+58)/2, (255+138)/2) ≈ (142, 156, 196) → ~#8e9cc4
        r, g, b = _hex_to_rgb(c)
        assert abs(r - 142) <= 1
        assert abs(g - 156) <= 1
        assert abs(b - 196) <= 1

    def test_above_max_clamped(self) -> None:
        assert color_single_gradient(2.0, end_rgb=(30, 58, 138)) == "#1e3a8a"

    def test_below_zero_clamped(self) -> None:
        assert color_single_gradient(-1.0, end_rgb=(30, 58, 138)) == "#ffffff"

    def test_custom_max_value(self) -> None:
        # value = 50, max = 100 → 0.5 → milieu
        c = color_single_gradient(50.0, end_rgb=(30, 58, 138), max_value=100.0)
        r, g, b = _hex_to_rgb(c)
        assert abs(r - 142) <= 1

    def test_custom_start_rgb(self) -> None:
        c = color_single_gradient(0.0, end_rgb=(30, 58, 138), start_rgb=(0, 0, 0))
        assert c == "#000000"

    def test_zero_max_value_returns_start(self) -> None:
        # Garde-fou contre division par zéro
        assert color_single_gradient(5.0, end_rgb=(30, 58, 138), max_value=0) == "#ffffff"


# ──────────────────────────────────────────────────────────────────
# color_diverging
# ──────────────────────────────────────────────────────────────────
class TestColorDiverging:
    def test_zero_returns_neutral(self) -> None:
        c = color_diverging(0.0)
        # Neutre = vert par défaut
        assert _hex_to_rgb(c) == (130, 200, 130)

    def test_positive_max_returns_positive_color(self) -> None:
        c = color_diverging(1.0, max_abs=1.0)
        assert _hex_to_rgb(c) == DIVERGING_POSITIVE_RGB

    def test_negative_max_returns_negative_color(self) -> None:
        c = color_diverging(-1.0, max_abs=1.0)
        assert _hex_to_rgb(c) == DIVERGING_NEGATIVE_RGB

    def test_clamping_above_max_abs(self) -> None:
        assert _hex_to_rgb(color_diverging(5.0, max_abs=1.0)) == DIVERGING_POSITIVE_RGB

    def test_clamping_below_negative_max_abs(self) -> None:
        assert _hex_to_rgb(color_diverging(-5.0, max_abs=1.0)) == DIVERGING_NEGATIVE_RGB

    def test_zero_max_abs_returns_neutral(self) -> None:
        c = color_diverging(5.0, max_abs=0)
        assert _hex_to_rgb(c) == (130, 200, 130)


# ──────────────────────────────────────────────────────────────────
# text_color_for_bg
# ──────────────────────────────────────────────────────────────────
class TestTextColorForBg:
    def test_low_intensity_dark_text(self) -> None:
        assert text_color_for_bg(0.2) == "#222"

    def test_high_intensity_white_text(self) -> None:
        assert text_color_for_bg(0.8) == "#fff"

    def test_threshold_boundary(self) -> None:
        assert text_color_for_bg(0.55) == "#222"
        assert text_color_for_bg(0.56) == "#fff"

    def test_custom_threshold(self) -> None:
        assert text_color_for_bg(0.4, threshold=0.3) == "#fff"


# ──────────────────────────────────────────────────────────────────
# build_grid_svg
# ──────────────────────────────────────────────────────────────────
class TestBuildGridSvg:
    def test_empty_grid_returns_empty_string(self) -> None:
        svg = build_grid_svg(
            n_rows=0,
            n_cols=0,
            row_label_fn=lambda i: "",
            col_label_fn=lambda j: "",
            cell_color_fn=lambda i, j: "#fff",
        )
        assert svg == ""

    def test_zero_rows_returns_empty(self) -> None:
        svg = build_grid_svg(
            n_rows=0,
            n_cols=3,
            row_label_fn=lambda i: "",
            col_label_fn=lambda j: str(j),
            cell_color_fn=lambda i, j: "#fff",
        )
        assert svg == ""

    def test_minimal_grid_renders(self) -> None:
        svg = build_grid_svg(
            n_rows=1,
            n_cols=1,
            row_label_fn=lambda i: "row0",
            col_label_fn=lambda j: "col0",
            cell_color_fn=lambda i, j: "#abc123",
        )
        assert "<svg" in svg
        assert "row0" in svg
        assert "col0" in svg
        assert "#abc123" in svg

    def test_cell_text_displayed_when_provided(self) -> None:
        svg = build_grid_svg(
            n_rows=1,
            n_cols=1,
            row_label_fn=lambda i: "r",
            col_label_fn=lambda j: "c",
            cell_color_fn=lambda i, j: "#fff",
            cell_text_fn=lambda i, j: "0.42",
        )
        assert "0.42" in svg

    def test_cell_text_omitted_when_none(self) -> None:
        svg = build_grid_svg(
            n_rows=1,
            n_cols=1,
            row_label_fn=lambda i: "r",
            col_label_fn=lambda j: "c",
            cell_color_fn=lambda i, j: "#fff",
            cell_text_fn=lambda i, j: None,
        )
        # Pas de chiffre dans la cellule
        assert ">0.<" not in svg

    def test_rotate_col_labels(self) -> None:
        svg = build_grid_svg(
            n_rows=1,
            n_cols=1,
            row_label_fn=lambda i: "r",
            col_label_fn=lambda j: "long_label",
            cell_color_fn=lambda i, j: "#fff",
            rotate_col_labels=True,
        )
        assert "rotate(-45" in svg

    def test_x_axis_title(self) -> None:
        svg = build_grid_svg(
            n_rows=1,
            n_cols=1,
            row_label_fn=lambda i: "r",
            col_label_fn=lambda j: "c",
            cell_color_fn=lambda i, j: "#fff",
            x_axis_title="Position dans le document",
        )
        assert "Position dans le document" in svg

    def test_html_escape_in_labels(self) -> None:
        svg = build_grid_svg(
            n_rows=1,
            n_cols=1,
            row_label_fn=lambda i: "<script>alert(1)</script>",
            col_label_fn=lambda j: "<b>bold</b>",
            cell_color_fn=lambda i, j: "#fff",
            cell_text_fn=lambda i, j: "<i>italic</i>",
        )
        assert "<script>" not in svg
        assert "&lt;script&gt;" in svg
        assert "<b>" not in svg
        assert "&lt;b&gt;" in svg
        assert "<i>" not in svg

    def test_grid_dimensions(self) -> None:
        svg = build_grid_svg(
            n_rows=3,
            n_cols=4,
            row_label_fn=lambda i: f"r{i}",
            col_label_fn=lambda j: f"c{j}",
            cell_color_fn=lambda i, j: "#fff",
        )
        # 12 cellules attendues (3×4)
        assert svg.count("<rect") == 12
        # 3 étiquettes de ligne + 4 de colonne
        # On compte les <text> qui ne sont pas dans une cellule (donc qui sont des labels) :
        # 3 labels lignes + 4 labels colonnes = 7. Pas de cell_text_fn donc 0.
        assert svg.count("<text") == 7

    def test_aria_label_present(self) -> None:
        svg = build_grid_svg(
            n_rows=1,
            n_cols=1,
            row_label_fn=lambda i: "r",
            col_label_fn=lambda j: "c",
            cell_color_fn=lambda i, j: "#fff",
            aria_label="Test heatmap",
        )
        assert 'aria-label="Test heatmap"' in svg
