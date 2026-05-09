"""Tests Sprint 37 — section inter-moteurs dans le rapport HTML.

Couvre :

1. ``build_divergence_matrix_html`` rend une table HTML colorée avec une
   ligne par moteur, masque la diagonale, et affiche la paire la plus
   divergente quand disponible.
2. ``build_oracle_gap_html`` rend l'encart factuel (best engine, oracle,
   gap absolu/relatif, nombre de docs).
3. **Masquage adaptatif** : les deux fonctions retournent ``""`` si
   ``inter_engine_analysis`` est ``None``, vide, ou s'il n'y a pas
   assez de données.
4. **Intégration template** : le rapport HTML inclut la section quand
   ``inter_engine_analysis`` est peuplé, et l'omet sinon (principe de
   masquage automatique du rapport adaptatif).
5. **i18n** : les clés FR/EN existent et sont utilisées (texte localisé).
6. **Anti-injection** : un nom de moteur contenant des caractères HTML
   est échappé.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picarones.evaluation.synthetic import generate_sample_benchmark
from picarones.reports_v2.html.generator import ReportGenerator
from picarones.reports_v2.html.renderers.inter_engine import (
    build_divergence_matrix_html,
    build_oracle_gap_html,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def realistic_iea() -> dict:
    """Analyse inter-moteurs réaliste avec 3 moteurs."""
    return {
        "engines": ["tess", "pero", "mistral"],
        "complementarity": {
            "oracle_recall": 0.95,
            "best_single_recall": 0.70,
            "best_engine": "pero",
            "absolute_gap": 0.25,
            "relative_gap": 0.83,
            "doc_count": 47,
            "per_engine_recall": {"pero": 0.7, "tess": 0.5, "mistral": 0.65},
            "per_doc": [],
        },
        "taxonomy_divergence": {
            "metric": "js",
            "matrix": {
                "tess":    {"tess": 0.0, "pero": 0.42, "mistral": 0.18},
                "pero":    {"tess": 0.42, "pero": 0.0, "mistral": 0.31},
                "mistral": {"tess": 0.18, "pero": 0.31, "mistral": 0.0},
            },
            "max_pair": ["tess", "pero", 0.42],
        },
    }


# ──────────────────────────────────────────────────────────────────────────
# 1. build_divergence_matrix_html
# ──────────────────────────────────────────────────────────────────────────


class TestDivergenceMatrixHTML:
    def test_renders_full_matrix(self, realistic_iea: dict) -> None:
        html = build_divergence_matrix_html(realistic_iea)
        assert "divergence-matrix" in html
        # Trois moteurs en têtes de colonne
        for name in ("tess", "pero", "mistral"):
            assert name in html
        # La diagonale est étiquetée explicitement (pas une valeur)
        assert "(identité)" in html or "(identity)" in html

    def test_off_diagonal_values_present(self, realistic_iea: dict) -> None:
        html = build_divergence_matrix_html(realistic_iea)
        # Les trois valeurs hors-diagonale doivent apparaître (à l'arrondi près)
        assert "0.420" in html  # tess ↔ pero
        assert "0.310" in html  # pero ↔ mistral
        assert "0.180" in html  # tess ↔ mistral

    def test_max_pair_labelled(self, realistic_iea: dict) -> None:
        html = build_divergence_matrix_html(realistic_iea)
        # Doit annoncer la paire la plus divergente (tess ↔ pero)
        assert "<strong>tess</strong>" in html
        assert "<strong>pero</strong>" in html
        # Et la valeur de divergence dans cette annonce
        assert "0.420" in html

    def test_empty_when_no_analysis(self) -> None:
        assert build_divergence_matrix_html(None) == ""
        assert build_divergence_matrix_html({}) == ""
        assert build_divergence_matrix_html({"taxonomy_divergence": None}) == ""

    def test_empty_when_single_engine(self) -> None:
        # Une matrice à un seul moteur n'a pas de sens
        iea = {
            "taxonomy_divergence": {
                "metric": "js",
                "matrix": {"only": {"only": 0.0}},
                "max_pair": None,
            }
        }
        assert build_divergence_matrix_html(iea) == ""

    def test_uses_i18n_labels(self, realistic_iea: dict) -> None:
        labels = {
            "divergence_caption": "CUSTOM_CAPTION",
            "divergence_metric_label": "CUSTOM_METRIC",
            "divergence_max_pair_label": "CUSTOM_MAX_PAIR",
            "divergence_diagonal_label": "CUSTOM_DIAG",
        }
        html = build_divergence_matrix_html(realistic_iea, labels=labels)
        assert "CUSTOM_CAPTION" in html
        assert "CUSTOM_METRIC" in html
        assert "CUSTOM_MAX_PAIR" in html
        assert "CUSTOM_DIAG" in html


# ──────────────────────────────────────────────────────────────────────────
# 2. build_oracle_gap_html
# ──────────────────────────────────────────────────────────────────────────


class TestOracleGapHTML:
    def test_shows_all_key_numbers(self, realistic_iea: dict) -> None:
        html = build_oracle_gap_html(realistic_iea)
        assert "inter-engine-oracle" in html
        assert "pero" in html              # best_engine
        assert "70.0 %" in html            # best_single_recall
        assert "95.0 %" in html            # oracle_recall
        assert "+25.0 pts" in html         # absolute_gap_pct
        assert "83.0 %" in html            # relative_gap_pct
        assert ">47<" in html              # doc_count

    def test_empty_when_no_complementarity(self) -> None:
        assert build_oracle_gap_html(None) == ""
        assert build_oracle_gap_html({}) == ""
        assert build_oracle_gap_html({"complementarity": None}) == ""

    def test_uses_i18n_labels(self, realistic_iea: dict) -> None:
        labels = {
            "oracle_caption": "CUSTOM_CAP",
            "oracle_best_engine": "CUSTOM_BEST",
            "oracle_explanation": "CUSTOM_EXPL",
        }
        html = build_oracle_gap_html(realistic_iea, labels=labels)
        assert "CUSTOM_CAP" in html
        assert "CUSTOM_BEST" in html
        assert "CUSTOM_EXPL" in html


# ──────────────────────────────────────────────────────────────────────────
# 3. Anti-injection HTML
# ──────────────────────────────────────────────────────────────────────────


class TestAntiInjection:
    def test_engine_name_with_html_chars_is_escaped(self) -> None:
        iea = {
            "taxonomy_divergence": {
                "metric": "js",
                "matrix": {
                    "<script>": {"<script>": 0.0, "safe": 0.4},
                    "safe":     {"<script>": 0.4, "safe": 0.0},
                },
                "max_pair": ["<script>", "safe", 0.4],
            }
        }
        html = build_divergence_matrix_html(iea)
        # Le tag <script> ne doit jamais apparaître brut dans la sortie
        assert "<script>" not in html
        # Et on doit voir la version échappée
        assert "&lt;script&gt;" in html


# ──────────────────────────────────────────────────────────────────────────
# 4. Intégration ReportGenerator
# ──────────────────────────────────────────────────────────────────────────


class TestReportIntegration:
    def test_section_absent_when_no_inter_engine(self, tmp_path: Path) -> None:
        bench = generate_sample_benchmark()
        # La fixture ne calcule pas inter_engine_analysis → None
        assert bench.inter_engine_analysis is None

        out = tmp_path / "report.html"
        ReportGenerator(bench).generate(out)
        html = out.read_text(encoding="utf-8")

        assert "divergence-matrix" not in html
        assert "inter-engine-oracle" not in html

    def test_section_present_when_populated(
        self, tmp_path: Path, realistic_iea: dict
    ) -> None:
        bench = generate_sample_benchmark()
        bench.inter_engine_analysis = realistic_iea

        out = tmp_path / "report.html"
        ReportGenerator(bench).generate(out)
        html = out.read_text(encoding="utf-8")

        assert "divergence-matrix" in html
        assert "inter-engine-oracle" in html
        assert "0.420" in html  # divergence tess↔pero
        assert "95.0 %" in html  # oracle recall

    def test_french_locale_uses_french_labels(
        self, tmp_path: Path, realistic_iea: dict
    ) -> None:
        bench = generate_sample_benchmark()
        bench.inter_engine_analysis = realistic_iea

        out = tmp_path / "report_fr.html"
        ReportGenerator(bench, lang="fr").generate(out)
        html = out.read_text(encoding="utf-8")
        assert "Divergence taxonomique" in html
        assert "Oracle" in html

    def test_english_locale_uses_english_labels(
        self, tmp_path: Path, realistic_iea: dict
    ) -> None:
        bench = generate_sample_benchmark()
        bench.inter_engine_analysis = realistic_iea

        out = tmp_path / "report_en.html"
        ReportGenerator(bench, lang="en").generate(out)
        html = out.read_text(encoding="utf-8")
        assert "Taxonomic divergence" in html
        assert "Oracle" in html


# ──────────────────────────────────────────────────────────────────────────
# 5. i18n FR/EN — clés présentes dans les deux fichiers
# ──────────────────────────────────────────────────────────────────────────


REQUIRED_KEYS = (
    "h_inter_engine",
    "inter_engine_note",
    "divergence_caption",
    "divergence_metric_label",
    "divergence_max_pair_label",
    "divergence_diagonal_label",
    "oracle_caption",
    "oracle_best_engine",
    "oracle_best_recall",
    "oracle_recall",
    "oracle_gap",
    "oracle_doc_count",
    "oracle_recoverable",
    "oracle_explanation",
)


class TestI18NCompleteness:
    @pytest.mark.parametrize("lang", ["fr", "en"])
    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_key_present(self, lang: str, key: str) -> None:
        path = (
            Path(__file__).parent.parent.parent
            / "picarones"
            / "reports_v2"
            / "i18n"
            / f"{lang}.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert key in data, f"Clé {key!r} manquante dans {lang}.json"
        assert data[key].strip(), f"Clé {key!r} vide dans {lang}.json"
