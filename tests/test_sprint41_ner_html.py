"""Tests Sprint 41 — section NER dans le rapport HTML.

Couvre :

1. ``build_ner_summary_html`` rend le tableau résumé (F1/P/R + totaux).
2. ``build_ner_per_category_html`` rend la heatmap moteur × catégorie.
3. **Masquage adaptatif** : les deux retournent ``""`` si aucun moteur
   n'a de ``aggregated_ner``, ou si le sous-champ ``per_category`` est
   absent (pour le second).
4. **Anti-injection** : un nom de moteur ou une catégorie contenant des
   balises HTML est échappé.
5. **Intégration template** : le rapport HTML inclut la section quand
   au moins un moteur a un ``aggregated_ner``, et l'omet sinon.
6. **i18n** : les clés FR/EN existent et sont utilisées.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picarones.fixtures import generate_sample_benchmark
from picarones.report.generator import ReportGenerator
from picarones.report.ner_render import (
    build_ner_per_category_html,
    build_ner_summary_html,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _engine_with_ner(name: str = "tess") -> dict:
    return {
        "name": name,
        "aggregated_ner": {
            "global": {
                "precision": 0.85, "recall": 0.78, "f1": 0.81, "support": 50,
            },
            "per_category": {
                "PER":  {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 30},
                "LOC":  {"precision": 0.7, "recall": 0.6, "f1": 0.65, "support": 15},
                "DATE": {"precision": 1.0, "recall": 0.9, "f1": 0.95, "support": 5},
            },
            "doc_count": 50,
            "hallucinated_total": 8,
            "missed_total": 11,
        },
    }


def _engine_without_ner(name: str = "no_ner") -> dict:
    return {"name": name, "aggregated_ner": None}


# ──────────────────────────────────────────────────────────────────────────
# 1. build_ner_summary_html
# ──────────────────────────────────────────────────────────────────────────


class TestNerSummaryHtml:
    def test_renders_table_with_engine_row(self) -> None:
        html = build_ner_summary_html([_engine_with_ner("tess")])
        assert "ner-summary" in html
        assert "tess" in html
        assert "81.0 %" in html  # F1
        assert "85.0 %" in html  # Precision
        assert "78.0 %" in html  # Recall
        assert "50" in html      # doc_count
        assert "8" in html       # hallucinations
        assert "11" in html      # missed

    def test_multiple_engines(self) -> None:
        engines = [_engine_with_ner("a"), _engine_with_ner("b")]
        html = build_ner_summary_html(engines)
        assert "<tr>" in html
        # Deux moteurs → trois <tr> minimum (header + 2 lignes)
        assert html.count("<tr>") >= 3


# ──────────────────────────────────────────────────────────────────────────
# 2. build_ner_per_category_html
# ──────────────────────────────────────────────────────────────────────────


class TestNerPerCategoryHtml:
    def test_renders_heatmap_with_categories(self) -> None:
        html = build_ner_per_category_html([_engine_with_ner("tess")])
        assert "ner-per-category" in html
        # Trois catégories en en-têtes
        for cat in ("PER", "LOC", "DATE"):
            assert cat in html
        # Le F1 est rendu en pourcentage
        assert "85.0 %" in html  # PER
        assert "65.0 %" in html  # LOC
        assert "95.0 %" in html  # DATE

    def test_categories_union_across_engines(self) -> None:
        e1 = _engine_with_ner("a")
        e2 = {
            "name": "b",
            "aggregated_ner": {
                "global": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 5},
                "per_category": {
                    "ORG": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 5},
                },
                "doc_count": 5, "hallucinated_total": 0, "missed_total": 0,
            },
        }
        html = build_ner_per_category_html([e1, e2])
        for cat in ("PER", "LOC", "DATE", "ORG"):
            assert cat in html

    def test_no_data_marker_for_missing_category(self) -> None:
        e1 = _engine_with_ner("a")
        # b n'a que ORG, donc ses cellules PER/LOC/DATE doivent afficher "—"
        e2 = {
            "name": "b",
            "aggregated_ner": {
                "global": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 5},
                "per_category": {
                    "ORG": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 5},
                },
                "doc_count": 5, "hallucinated_total": 0, "missed_total": 0,
            },
        }
        html = build_ner_per_category_html([e1, e2])
        # "—" apparaît dans les cellules vides
        assert "—" in html


# ──────────────────────────────────────────────────────────────────────────
# 3. Masquage adaptatif
# ──────────────────────────────────────────────────────────────────────────


class TestAdaptiveMasking:
    def test_summary_empty_when_no_engine_has_ner(self) -> None:
        assert build_ner_summary_html([]) == ""
        assert build_ner_summary_html([_engine_without_ner()]) == ""

    def test_per_category_empty_when_no_engine_has_ner(self) -> None:
        assert build_ner_per_category_html([]) == ""
        assert build_ner_per_category_html([_engine_without_ner()]) == ""

    def test_per_category_empty_when_no_categories(self) -> None:
        engine = {
            "name": "x",
            "aggregated_ner": {
                "global": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "per_category": {},   # vide
                "doc_count": 0,
                "hallucinated_total": 0, "missed_total": 0,
            },
        }
        assert build_ner_per_category_html([engine]) == ""


# ──────────────────────────────────────────────────────────────────────────
# 4. Anti-injection
# ──────────────────────────────────────────────────────────────────────────


class TestAntiInjection:
    def test_engine_name_with_html_chars_escaped(self) -> None:
        engine = _engine_with_ner("<script>alert(1)</script>")
        html = build_ner_summary_html([engine])
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_category_label_with_html_chars_escaped(self) -> None:
        engine = {
            "name": "x",
            "aggregated_ner": {
                "global": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
                "per_category": {
                    "<img src=x>": {
                        "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1,
                    },
                },
                "doc_count": 1, "hallucinated_total": 0, "missed_total": 0,
            },
        }
        html = build_ner_per_category_html([engine])
        assert "<img" not in html or "&lt;img" in html


# ──────────────────────────────────────────────────────────────────────────
# 5. Intégration ReportGenerator
# ──────────────────────────────────────────────────────────────────────────


class TestReportIntegration:
    def test_section_absent_when_no_ner(self, tmp_path: Path) -> None:
        bench = generate_sample_benchmark()
        # Aucun engine_reports[i].aggregated_ner par défaut
        for r in bench.engine_reports:
            assert r.aggregated_ner is None

        out = tmp_path / "report.html"
        ReportGenerator(bench).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "ner-summary" not in html
        assert "ner-per-category" not in html

    def test_section_present_when_at_least_one_engine_has_ner(
        self, tmp_path: Path,
    ) -> None:
        bench = generate_sample_benchmark()
        bench.engine_reports[0].aggregated_ner = {
            "global": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 30},
            "per_category": {
                "PER": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 30},
            },
            "doc_count": 30, "hallucinated_total": 5, "missed_total": 4,
        }
        out = tmp_path / "report.html"
        ReportGenerator(bench).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "ner-summary" in html
        assert "ner-per-category" in html
        assert "87.0 %" in html

    def test_french_locale_uses_french_labels(self, tmp_path: Path) -> None:
        bench = generate_sample_benchmark()
        bench.engine_reports[0].aggregated_ner = {
            "global": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 30},
            "per_category": {
                "PER": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 30},
            },
            "doc_count": 30, "hallucinated_total": 5, "missed_total": 4,
        }
        out = tmp_path / "report_fr.html"
        ReportGenerator(bench, lang="fr").generate(out)
        html = out.read_text(encoding="utf-8")
        # Labels français
        assert "Entités manquées" in html or "Hallucinations" in html
        assert "F1 global" in html

    def test_english_locale_uses_english_labels(self, tmp_path: Path) -> None:
        bench = generate_sample_benchmark()
        bench.engine_reports[0].aggregated_ner = {
            "global": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 30},
            "per_category": {
                "PER": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 30},
            },
            "doc_count": 30, "hallucinated_total": 5, "missed_total": 4,
        }
        out = tmp_path / "report_en.html"
        ReportGenerator(bench, lang="en").generate(out)
        html = out.read_text(encoding="utf-8")
        assert "Missed entities" in html or "Hallucinations" in html
        assert "Global F1" in html


# ──────────────────────────────────────────────────────────────────────────
# 6. i18n FR/EN — clés présentes et non vides
# ──────────────────────────────────────────────────────────────────────────


REQUIRED_KEYS = (
    "h_ner",
    "ner_note",
    "ner_summary_caption",
    "ner_per_category_caption",
    "ner_engine_label",
    "ner_f1_label",
    "ner_precision_label",
    "ner_recall_label",
    "ner_doc_count_label",
    "ner_hallucinated_label",
    "ner_missed_label",
    "ner_no_data_label",
)


class TestI18NCompleteness:
    @pytest.mark.parametrize("lang", ["fr", "en"])
    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_key_present(self, lang: str, key: str) -> None:
        path = (
            Path(__file__).parent.parent
            / "picarones" / "report" / "i18n" / f"{lang}.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert key in data, f"Clé {key!r} manquante dans {lang}.json"
        assert data[key].strip(), f"Clé {key!r} vide dans {lang}.json"
