"""Tests Sprint 95 — B.4 : visualisation DAG d'un pipeline composé.

Couvre :

1. ``build_pipeline_dag_html`` :
   - vide / None → ``""``
   - 1 nœud → SVG sans arête
   - 2 nœuds + 1 arête
   - 3 nœuds chaînés
   - arêtes auto-déduites si non fournies
   - couleur selon seuil de la métrique
   - mode higher_is_better
2. Anti-injection sur nom de nœud, type d'artefact, nom de
   métrique.
3. Affichage de la valeur de métrique formatée.
4. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.report.pipeline_dag_render import build_pipeline_dag_html


def _load_labels(lang: str) -> dict:
    p = (
        Path(__file__).parent.parent.parent
        / "picarones" / "reports_v2" / "i18n" / f"{lang}.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────
# 1. build_pipeline_dag_html
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def test_empty_returns_empty(self) -> None:
        assert build_pipeline_dag_html(None) == ""
        assert build_pipeline_dag_html([]) == ""

    def test_single_node_renders_svg_no_edge(self) -> None:
        nodes = [{"name": "tess", "output_types": ["TEXT"]}]
        html = build_pipeline_dag_html(nodes, _load_labels("fr"))
        assert "<svg" in html
        assert "tess" in html
        # Pas de flèche tracée (pas d'arête)
        assert "marker-end" not in html

    def test_two_nodes_one_edge(self) -> None:
        nodes = [
            {"name": "ocr", "output_types": ["TEXT"]},
            {"name": "llm", "input_types": ["TEXT"]},
        ]
        edges = [{"from": "ocr", "to": "llm",
                  "artifact_type": "TEXT",
                  "metric_name": "cer",
                  "metric_value": 0.04}]
        html = build_pipeline_dag_html(
            nodes, _load_labels("fr"), edges=edges,
        )
        # Nœuds présents
        assert "ocr" in html
        assert "llm" in html
        # Étiquettes d'arête
        assert "TEXT" in html
        assert "cer" in html
        assert "4.0%" in html
        # Flèche présente
        assert "marker-end" in html

    def test_three_nodes_chain(self) -> None:
        nodes = [
            {"name": "a"}, {"name": "b"}, {"name": "c"},
        ]
        edges = [
            {"from": "a", "to": "b", "metric_value": 0.05},
            {"from": "b", "to": "c", "metric_value": 0.10},
        ]
        html = build_pipeline_dag_html(nodes, edges=edges)
        # Deux flèches
        assert html.count("marker-end") == 2

    def test_auto_edges_when_missing(self) -> None:
        # Pas d'arêtes fournies → auto-déduit séquentielles
        nodes = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        html = build_pipeline_dag_html(nodes)
        assert html.count("marker-end") == 2

    def test_colour_green_for_low_cer(self) -> None:
        # Sprint A7 (m-5) : palette Okabe-Ito (daltonien-friendly).
        # Le test valide la sémantique « ≤ 0.05 → bon » sans coder en
        # dur le hex (qui peut évoluer avec la palette).  Comparaison
        # via ``COLOR_GREEN`` du module canonique.
        from picarones.report.colors import COLOR_GREEN

        nodes = [{"name": "a"}, {"name": "b"}]
        edges = [{"from": "a", "to": "b",
                  "metric_value": 0.02}]  # ≤ 0.05 → bon
        html = build_pipeline_dag_html(nodes, edges=edges)
        assert COLOR_GREEN in html

    def test_colour_yellow(self) -> None:
        from picarones.report.colors import COLOR_YELLOW

        nodes = [{"name": "a"}, {"name": "b"}]
        edges = [{"from": "a", "to": "b", "metric_value": 0.10}]
        html = build_pipeline_dag_html(nodes, edges=edges)
        assert COLOR_YELLOW in html

    def test_colour_red_for_high_cer(self) -> None:
        from picarones.report.colors import COLOR_RED

        nodes = [{"name": "a"}, {"name": "b"}]
        edges = [{"from": "a", "to": "b", "metric_value": 0.30}]
        html = build_pipeline_dag_html(nodes, edges=edges)
        assert COLOR_RED in html

    def test_higher_is_better_inverts(self) -> None:
        # F1 = 0.95 = bonne qualité (haut)
        from picarones.report.colors import COLOR_GREEN

        nodes = [{"name": "a"}, {"name": "b"}]
        edges = [{"from": "a", "to": "b", "metric_value": 0.96}]
        html = build_pipeline_dag_html(
            nodes, edges=edges, higher_is_better=True,
        )
        assert COLOR_GREEN in html

    def test_unknown_node_in_edge_skipped(self) -> None:
        nodes = [{"name": "a"}, {"name": "b"}]
        edges = [
            {"from": "a", "to": "b", "metric_value": 0.05},
            {"from": "ghost", "to": "b", "metric_value": 0.01},
        ]
        html = build_pipeline_dag_html(nodes, edges=edges)
        # Une seule flèche valide
        assert html.count("marker-end") == 1

    def test_handles_missing_metric_value(self) -> None:
        nodes = [{"name": "a"}, {"name": "b"}]
        edges = [{"from": "a", "to": "b",
                  "artifact_type": "TEXT",
                  "metric_name": "cer"}]  # pas de valeur
        html = build_pipeline_dag_html(nodes, edges=edges)
        assert "—" in html or "cer" in html


# ──────────────────────────────────────────────────────────────────────────
# 2. Anti-injection
# ──────────────────────────────────────────────────────────────────────────


class TestAntiInjection:
    def test_node_name(self) -> None:
        nodes = [{"name": "<script>alert(1)</script>"}]
        html = build_pipeline_dag_html(nodes, _load_labels("fr"))
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_artifact_type(self) -> None:
        nodes = [{"name": "a"}, {"name": "b"}]
        edges = [{"from": "a", "to": "b",
                  "artifact_type": "<img/>",
                  "metric_value": 0.05}]
        html = build_pipeline_dag_html(nodes, edges=edges)
        assert "<img/>" not in html
        assert "&lt;img" in html

    def test_metric_name(self) -> None:
        nodes = [{"name": "a"}, {"name": "b"}]
        edges = [{"from": "a", "to": "b",
                  "metric_name": "<script>x",
                  "metric_value": 0.05}]
        html = build_pipeline_dag_html(nodes, edges=edges)
        assert "<script>x" not in html
        assert "&lt;script&gt;" in html

    def test_input_output_types(self) -> None:
        nodes = [{"name": "a", "input_types": ["<svg/>"],
                  "output_types": ["<x>"]}]
        html = build_pipeline_dag_html(nodes, _load_labels("fr"))
        assert "<svg/>" not in html
        assert "&lt;svg" in html


# ──────────────────────────────────────────────────────────────────────────
# 3. Rendu en anglais
# ──────────────────────────────────────────────────────────────────────────


class TestI18nRendering:
    def test_english(self) -> None:
        nodes = [{"name": "a"}]
        html = build_pipeline_dag_html(nodes, _load_labels("en"))
        assert "Inspection tool" in html or "source of truth" in html


# ──────────────────────────────────────────────────────────────────────────
# 4. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


_KEYS = {
    "dag_title", "dag_note", "dag_legend",
    "dag_legend_green", "dag_legend_yellow", "dag_legend_red",
}


class TestI18nCompleteness:
    def test_fr(self) -> None:
        d = _load_labels("fr")
        assert not _KEYS - d.keys()

    def test_en(self) -> None:
        d = _load_labels("en")
        assert not _KEYS - d.keys()
