"""Tests Sprint 75 — A.I.4 chantier 1 : co-occurrence taxonomique.

Couvre :

1. ``compute_taxonomy_cooccurrence`` :
   - Matrice symétrique
   - Diagonale = 1.0 pour classes présentes
   - Classes toujours ensemble → Jaccard = 1
   - Classes jamais ensemble → Jaccard = 0
   - Cas dégénéré : per_doc_classes vide → None
   - ``min_doc_count`` filtre les classes anecdotiques
   - ``top_pairs`` triées par Jaccard descendant
2. Rendu HTML :
   - SVG bien formé
   - Table top_pairs présente
   - Cellules colorées
   - ``""`` si ``data is None``
3. Anti-injection : noms de classes contenant ``<script>``.
4. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picarones.evaluation.metrics.taxonomy_cooccurrence import (
    compute_taxonomy_cooccurrence,
)
from picarones.reports.html.renderers.taxonomy_cooccurrence import (
    build_taxonomy_cooccurrence_html,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. Couche de calcul
# ──────────────────────────────────────────────────────────────────────────


class TestCompute:
    def test_classes_always_together_jaccard_one(self) -> None:
        # 5 docs, A et B toujours ensemble
        per_doc = [
            {"A", "B"}, {"A", "B"}, {"A", "B"},
            {"A", "B"}, {"A", "B"},
        ]
        result = compute_taxonomy_cooccurrence(per_doc)
        assert result is not None
        assert result["cooccurrence_matrix"]["A"]["B"] == 1.0

    def test_classes_never_together_jaccard_zero(self) -> None:
        # A et B mutuellement exclusifs
        per_doc = [
            {"A"}, {"A"}, {"B"}, {"B"}, {"B"},
        ]
        result = compute_taxonomy_cooccurrence(per_doc)
        assert result is not None
        assert result["cooccurrence_matrix"]["A"]["B"] == 0.0

    def test_diagonal_is_one(self) -> None:
        per_doc = [{"A"}, {"B"}, {"A", "B"}]
        result = compute_taxonomy_cooccurrence(per_doc)
        assert result["cooccurrence_matrix"]["A"]["A"] == 1.0
        assert result["cooccurrence_matrix"]["B"]["B"] == 1.0

    def test_symmetric(self) -> None:
        per_doc = [{"A", "B"}, {"A"}, {"B"}, {"A", "B"}]
        result = compute_taxonomy_cooccurrence(per_doc)
        assert result["cooccurrence_matrix"]["A"]["B"] == \
               result["cooccurrence_matrix"]["B"]["A"]

    def test_partial_overlap(self) -> None:
        # 4 docs : A∪B = 4, A∩B = 2 → Jaccard = 0.5
        per_doc = [{"A", "B"}, {"A", "B"}, {"A"}, {"B"}]
        result = compute_taxonomy_cooccurrence(per_doc)
        assert result["cooccurrence_matrix"]["A"]["B"] == pytest.approx(0.5)

    def test_empty_corpus_returns_none(self) -> None:
        assert compute_taxonomy_cooccurrence([]) is None
        assert compute_taxonomy_cooccurrence([set(), set()]) is None

    def test_min_doc_count_filter(self) -> None:
        # A apparaît 5 fois, B apparaît 1 fois (anecdotique)
        per_doc = [{"A", "B"}, {"A"}, {"A"}, {"A"}, {"A"}]
        result = compute_taxonomy_cooccurrence(per_doc, min_doc_count=2)
        assert result is not None
        assert "A" in result["classes"]
        assert "B" not in result["classes"]

    def test_top_pairs_sorted(self) -> None:
        per_doc = [
            {"A", "B", "C"},  # 3 ensemble
            {"A", "B"},       # AB
            {"A", "B"},       # AB
            {"C"},            # C seul
        ]
        result = compute_taxonomy_cooccurrence(per_doc, top_n_pairs=10)
        assert result is not None
        top = result["top_pairs"]
        # Triées par Jaccard décroissant
        for i in range(len(top) - 1):
            assert top[i][2] >= top[i + 1][2]

    def test_top_pairs_count_limit(self) -> None:
        per_doc = [{"A", "B", "C", "D"}]
        result = compute_taxonomy_cooccurrence(per_doc, top_n_pairs=2)
        assert len(result["top_pairs"]) == 2

    def test_doc_count_correct(self) -> None:
        per_doc = [{"A"}, {"A", "B"}, {"B"}]
        result = compute_taxonomy_cooccurrence(per_doc)
        assert result["doc_count"]["A"] == 2
        assert result["doc_count"]["B"] == 2
        assert result["n_documents"] == 3

    def test_none_doc_skipped(self) -> None:
        per_doc = [{"A"}, None, {"B"}]
        result = compute_taxonomy_cooccurrence(per_doc)
        assert result is not None
        assert result["n_documents"] == 2


# ──────────────────────────────────────────────────────────────────────────
# 2. Rendu HTML
# ──────────────────────────────────────────────────────────────────────────


class TestRender:
    def test_returns_empty_when_data_none(self) -> None:
        assert build_taxonomy_cooccurrence_html(None) == ""

    def test_returns_empty_when_classes_empty(self) -> None:
        data = {"classes": [], "cooccurrence_matrix": {},
                "top_pairs": [], "n_documents": 0}
        assert build_taxonomy_cooccurrence_html(data) == ""

    def test_renders_svg(self) -> None:
        per_doc = [{"A", "B"}, {"A"}, {"B"}]
        data = compute_taxonomy_cooccurrence(per_doc)
        html = build_taxonomy_cooccurrence_html(data)
        assert "<svg" in html
        assert "</svg>" in html
        assert "Co-occurrence" in html

    def test_renders_top_pairs_table(self) -> None:
        per_doc = [{"A", "B"}, {"A", "B"}, {"A"}]
        data = compute_taxonomy_cooccurrence(per_doc)
        html = build_taxonomy_cooccurrence_html(data)
        # Table avec en-têtes Paire + Jaccard
        assert "Paire" in html
        assert "Jaccard" in html
        # Au moins une cellule de valeur Jaccard
        assert "0." in html or "1." in html

    def test_jaccard_values_displayed(self) -> None:
        per_doc = [{"A", "B"}] * 5  # toujours ensemble → 1.0
        data = compute_taxonomy_cooccurrence(per_doc)
        html = build_taxonomy_cooccurrence_html(data)
        assert "1.00" in html

    def test_class_labels_present(self) -> None:
        per_doc = [{"ligature_error", "abbreviation_error"}, {"ligature_error"}]
        data = compute_taxonomy_cooccurrence(per_doc)
        html = build_taxonomy_cooccurrence_html(data)
        assert "ligature_error" in html
        assert "abbreviation_error" in html

    def test_n_docs_displayed(self) -> None:
        per_doc = [{"A", "B"}, {"A"}, {"B"}]
        data = compute_taxonomy_cooccurrence(per_doc)
        html = build_taxonomy_cooccurrence_html(data)
        assert "3" in html  # n_documents = 3


# ──────────────────────────────────────────────────────────────────────────
# 3. Anti-injection
# ──────────────────────────────────────────────────────────────────────────


class TestAntiInjection:
    def test_class_name_with_script_escaped(self) -> None:
        per_doc = [{"<script>", "B"}, {"<script>"}]
        data = compute_taxonomy_cooccurrence(per_doc)
        html = build_taxonomy_cooccurrence_html(data)
        assert "<script>" not in html.replace(
            "<script>alert", "@@@",  # ne devrait pas être présent de toute façon
        )
        assert "&lt;script&gt;" in html

    def test_label_via_i18n_escaped(self) -> None:
        per_doc = [{"A", "B"}]
        data = compute_taxonomy_cooccurrence(per_doc)
        labels = {"taxocooc_title": "<b>Hack</b>"}
        html = build_taxonomy_cooccurrence_html(data, labels=labels)
        assert "<b>Hack</b>" not in html
        assert "&lt;b&gt;Hack&lt;/b&gt;" in html


# ──────────────────────────────────────────────────────────────────────────
# 4. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


class TestI18nCompleteness:
    def _load(self, lang: str) -> dict:
        path = (
            Path(__file__).parent.parent.parent
            / "picarones" / "reports" / "i18n" / f"{lang}.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_all_keys_fr(self) -> None:
        d = self._load("fr")
        for key in (
            "taxocooc_title", "taxocooc_note", "taxocooc_n_docs",
            "taxocooc_pair_label", "taxocooc_jaccard_label",
        ):
            assert key in d, f"manque clé FR : {key}"

    def test_all_keys_en(self) -> None:
        d_fr = self._load("fr")
        d_en = self._load("en")
        for key in d_fr:
            if key.startswith("taxocooc_"):
                assert key in d_en, f"manque clé EN : {key}"
