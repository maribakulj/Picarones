"""Tests Sprint 72 — A.I.1 chantier 1 : vue « Worst lines globale ».

Couvre :

1. ``extract_worst_lines`` :
   - Top-N respecté, tri par CER décroissant
   - Filtre par moteur
   - Filtre par strate (``script_type``)
   - Lignes avec CER == 0 ignorées
   - DocumentResult sans ``line_metrics`` ignoré
   - Index de ligne hors borne → texte vide mais entrée incluse
     si au moins l'un des deux côtés a du texte
   - top_n=0 → liste vide
2. ``WorstLineEntry`` : rang attribué après tri (1-based).
3. ``build_worst_lines_table_html`` :
   - Tableau rendu avec colonnes attendues
   - Chaîne vide si entries vide
   - Colonne strate omise si aucune entry n'a script_type
   - Cellule CER colorée
   - Diff GT/hyp rendu (rouge barré + vert)
4. Anti-injection : nom moteur, doc_id, ligne GT/hyp avec
   ``<script>`` correctement échappés.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from picarones.evaluation.metrics.worst_lines import WorstLineEntry, extract_worst_lines
from picarones.reports_v2.html.renderers.worst_lines import build_worst_lines_table_html


# ──────────────────────────────────────────────────────────────────────────
# Mocks pour BenchmarkResult / EngineReport / DocumentResult
# ──────────────────────────────────────────────────────────────────────────
# On évite les vrais dataclasses du runner (lourds, dépendances) pour
# garder les tests focalisés sur la logique d'extraction.


@dataclass
class _DocResult:
    doc_id: str
    ground_truth: str
    hypothesis: str
    line_metrics: dict[str, Any] | None = None


@dataclass
class _EngineReport:
    engine_name: str
    document_results: list[_DocResult] = field(default_factory=list)


@dataclass
class _Benchmark:
    engine_reports: list[_EngineReport] = field(default_factory=list)
    doc_strata: dict[str, str] | None = None


def _make_benchmark() -> _Benchmark:
    """Construit un benchmark de test : 2 moteurs × 3 docs."""
    bench = _Benchmark(doc_strata={"d0": "imprime", "d1": "manuscrit", "d2": "manuscrit"})
    for engine_name, cer_offsets in (("tess", 0.0), ("pero", 0.1)):
        docs = []
        for doc_id, gt, hyp, cer_lines in (
            ("d0", "ligne0\nligne1\nligne2", "ligne0\nlignE1\nligne2",
             [0.0, 0.2, 0.0]),
            ("d1", "abc\ndef\nghi", "abc\nXXX\nghi",
             [0.0, 1.0, 0.0]),
            ("d2", "alpha\nbeta\ngamma", "alpha\nbeta\nXXXXX",
             [0.0, 0.0, 0.7]),
        ):
            docs.append(_DocResult(
                doc_id=doc_id,
                ground_truth=gt,
                hypothesis=hyp,
                line_metrics={
                    "cer_per_line": [c + cer_offsets for c in cer_lines],
                },
            ))
        bench.engine_reports.append(
            _EngineReport(engine_name=engine_name, document_results=docs),
        )
    return bench


# ──────────────────────────────────────────────────────────────────────────
# 1. extract_worst_lines
# ──────────────────────────────────────────────────────────────────────────


class TestExtractBasic:
    def test_top_n_respected(self) -> None:
        bench = _make_benchmark()
        out = extract_worst_lines(bench, top_n=3)
        assert len(out) == 3

    def test_sorted_by_cer_desc(self) -> None:
        bench = _make_benchmark()
        out = extract_worst_lines(bench, top_n=20)
        cers = [e.cer for e in out]
        assert cers == sorted(cers, reverse=True)

    def test_rank_is_1_based(self) -> None:
        bench = _make_benchmark()
        out = extract_worst_lines(bench, top_n=5)
        ranks = [e.rank for e in out]
        assert ranks == list(range(1, len(out) + 1))

    def test_top_n_zero_returns_empty(self) -> None:
        bench = _make_benchmark()
        assert extract_worst_lines(bench, top_n=0) == []

    def test_lines_with_zero_cer_ignored(self) -> None:
        bench = _make_benchmark()
        out = extract_worst_lines(bench, top_n=100)
        for entry in out:
            assert entry.cer > 0.0


class TestFilters:
    def test_engine_filter(self) -> None:
        bench = _make_benchmark()
        out = extract_worst_lines(bench, top_n=20, engine_filter="pero")
        assert all(e.engine_name == "pero" for e in out)
        assert len(out) > 0

    def test_engine_filter_unknown_engine(self) -> None:
        bench = _make_benchmark()
        out = extract_worst_lines(
            bench, top_n=20, engine_filter="non_existing",
        )
        assert out == []

    def test_strata_filter(self) -> None:
        bench = _make_benchmark()
        out = extract_worst_lines(
            bench, top_n=20, script_type_filter="manuscrit",
        )
        assert all(e.script_type == "manuscrit" for e in out)
        assert len(out) > 0

    def test_strata_filter_unknown_strata(self) -> None:
        bench = _make_benchmark()
        out = extract_worst_lines(
            bench, top_n=20, script_type_filter="non_existing",
        )
        assert out == []


class TestEdgeCases:
    def test_no_line_metrics(self) -> None:
        bench = _Benchmark(engine_reports=[
            _EngineReport(engine_name="x", document_results=[
                _DocResult(doc_id="d", ground_truth="x", hypothesis="x",
                           line_metrics=None),
            ]),
        ])
        assert extract_worst_lines(bench) == []

    def test_empty_engine_reports(self) -> None:
        bench = _Benchmark()
        assert extract_worst_lines(bench) == []

    def test_no_doc_strata_attribute(self) -> None:
        # benchmark sans attribut doc_strata → pas de filtre strata
        # mais l'extraction fonctionne
        bench = _Benchmark(engine_reports=[
            _EngineReport(engine_name="x", document_results=[
                _DocResult(
                    doc_id="d", ground_truth="abc", hypothesis="aXc",
                    line_metrics={"cer_per_line": [0.5]},
                ),
            ]),
        ])
        out = extract_worst_lines(bench, top_n=5)
        assert len(out) == 1
        assert out[0].script_type is None

    def test_hyp_shorter_than_gt(self) -> None:
        # Hyp a moins de lignes que GT — ligne en trop dans GT
        # est récupérée avec hyp_line=""
        bench = _Benchmark(engine_reports=[
            _EngineReport(engine_name="x", document_results=[
                _DocResult(
                    doc_id="d", ground_truth="abc\ndef\nghi",
                    hypothesis="abc",  # 1 ligne seulement
                    line_metrics={"cer_per_line": [0.0, 1.0, 1.0]},
                ),
            ]),
        ])
        out = extract_worst_lines(bench, top_n=5)
        assert len(out) == 2  # lignes 1 et 2 avec CER = 1.0
        for entry in out:
            assert entry.hyp_line == ""


# ──────────────────────────────────────────────────────────────────────────
# 2. build_worst_lines_table_html
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def _sample_entries(self) -> list[WorstLineEntry]:
        return [
            WorstLineEntry(
                rank=1, cer=0.95, engine_name="tess", doc_id="d1",
                line_index=2, gt_line="bonjour le monde",
                hyp_line="bnjour 1e mnde", script_type="imprime",
            ),
            WorstLineEntry(
                rank=2, cer=0.42, engine_name="pero", doc_id="d3",
                line_index=0, gt_line="hello world",
                hyp_line="hello wOrld", script_type="manuscrit",
            ),
        ]

    def test_renders_table(self) -> None:
        html = build_worst_lines_table_html(self._sample_entries())
        assert "<table" in html
        assert "tess" in html
        assert "pero" in html
        assert "d1" in html
        assert "d3" in html

    def test_empty_returns_empty(self) -> None:
        assert build_worst_lines_table_html([]) == ""

    def test_columns_present(self) -> None:
        html = build_worst_lines_table_html(self._sample_entries())
        for col in ("Rang", "CER", "Moteur", "Document", "Ligne"):
            assert col in html

    def test_strata_column_when_present(self) -> None:
        html = build_worst_lines_table_html(self._sample_entries())
        assert "Strate" in html
        assert "imprime" in html
        assert "manuscrit" in html

    def test_strata_column_omitted_when_absent(self) -> None:
        entries = [
            WorstLineEntry(
                rank=1, cer=0.5, engine_name="t", doc_id="d", line_index=0,
                gt_line="abc", hyp_line="aXc", script_type=None,
            ),
        ]
        html = build_worst_lines_table_html(entries)
        assert "Strate" not in html

    def test_cer_cell_colored(self) -> None:
        html = build_worst_lines_table_html(self._sample_entries())
        assert "background:#" in html

    def test_diff_rendered(self) -> None:
        html = build_worst_lines_table_html(self._sample_entries())
        # Diff inline : couleurs rouge clair pour suppressions, vert pour insertions
        assert "#fdd" in html
        assert "#dfd" in html

    def test_cer_displayed_as_percent(self) -> None:
        html = build_worst_lines_table_html(self._sample_entries())
        assert "95.0%" in html
        assert "42.0%" in html


# ──────────────────────────────────────────────────────────────────────────
# 3. Anti-injection
# ──────────────────────────────────────────────────────────────────────────


class TestAntiInjection:
    def test_engine_name_escaped(self) -> None:
        entries = [
            WorstLineEntry(
                rank=1, cer=0.5, engine_name="<script>alert(1)</script>",
                doc_id="d", line_index=0,
                gt_line="abc", hyp_line="aXc",
            ),
        ]
        html = build_worst_lines_table_html(entries)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_doc_id_escaped(self) -> None:
        entries = [
            WorstLineEntry(
                rank=1, cer=0.5, engine_name="t",
                doc_id="<img src=x>", line_index=0,
                gt_line="abc", hyp_line="aXc",
            ),
        ]
        html = build_worst_lines_table_html(entries)
        assert "<img src=x>" not in html
        assert "&lt;img" in html

    def test_gt_line_escaped(self) -> None:
        entries = [
            WorstLineEntry(
                rank=1, cer=0.5, engine_name="t", doc_id="d", line_index=0,
                gt_line="<b>HACK</b>", hyp_line="bonjour",
            ),
        ]
        html = build_worst_lines_table_html(entries)
        # La balise brute ne doit pas être présente.  Le diff
        # caractère-par-caractère peut splitter ``<b>`` en chunks
        # séparés mais chaque chunk est échappé.
        assert "<b>HACK</b>" not in html
        assert "&lt;" in html
        assert "&gt;" in html

    def test_label_via_i18n_escaped(self) -> None:
        entries = [
            WorstLineEntry(
                rank=1, cer=0.5, engine_name="t", doc_id="d", line_index=0,
                gt_line="abc", hyp_line="aXc",
            ),
        ]
        labels = {"worst_lines_title": "<b>X</b>"}
        html = build_worst_lines_table_html(entries, labels=labels)
        assert "<b>X</b>" not in html
        assert "&lt;b&gt;" in html
