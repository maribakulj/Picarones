"""Tests pour le sprint 14 — Filtrage interactif des documents dans le rapport HTML.

Vérifie :
- Présence du slider CER dans le rapport HTML généré
- Présence du bouton reset d'exclusions galerie
- Les fonctions JS _buildCSVRows / exportCSV (deux CSV)
- La logique de filtrage robuste (CER seuil, ancrage, ratio, exclusions manuelles)
- Les nouvelles métriques robustes : médiane, p90, p95
- La liste des documents exclus avec raisons
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — construire un faux rapport HTML
# ---------------------------------------------------------------------------

def _make_fake_benchmark():
    """Retourne un BenchmarkResult minimal pour tester le générateur."""
    from picarones.core.results import BenchmarkResult, EngineReport, DocumentResult
    from picarones.core.metrics import MetricsResult

    def _metrics(cer, wer=0.2):
        return MetricsResult(
            cer=cer, cer_nfc=cer, cer_caseless=cer,
            wer=wer, wer_normalized=wer,
            mer=wer * 0.9, wil=wer * 0.8,
            reference_length=100, hypothesis_length=100,
        )

    doc_results = [
        DocumentResult(
            doc_id=f"doc{i}",
            image_path="",
            ground_truth="test ground truth",
            hypothesis="test hypothesis",
            metrics=_metrics(0.05 * i),
            duration_seconds=1.0,
        )
        for i in range(1, 5)
    ]
    engine_report = EngineReport(
        engine_name="TestEngine",
        engine_version="1.0",
        engine_config={},
        document_results=doc_results,
    )
    return BenchmarkResult(
        corpus_name="TestCorpus",
        corpus_source=None,
        document_count=4,
        engine_reports=[engine_report],
        run_date="2026-01-01",
        picarones_version="1.0.0",
    )


def _generate_html(bm=None) -> str:
    """Génère le HTML complet du rapport pour un BenchmarkResult minimal."""
    from picarones.report.generator import ReportGenerator
    import tempfile, os
    if bm is None:
        bm = _make_fake_benchmark()
    gen = ReportGenerator(bm)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name
    try:
        gen.generate(path)
        return Path(path).read_text(encoding="utf-8")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 1. Structure HTML — nouveaux éléments
# ---------------------------------------------------------------------------

class TestRobustCardHTML:
    """Vérifie la présence des nouveaux contrôles dans le rapport HTML."""

    @pytest.fixture(scope="class")
    def html(self):
        return _generate_html()

    def test_cer_slider_present(self, html):
        """Le slider CER seuil doit être présent."""
        assert 'id="robust-cer"' in html

    def test_cer_slider_range(self, html):
        """Le slider CER doit aller de 0 à 100."""
        assert 'min="0" max="100"' in html or 'min=\\"0\\" max=\\"100\\"' in html or re.search(r'id="robust-cer"[^>]*min="0"[^>]*max="100"', html)

    def test_anchor_slider_present(self, html):
        """Le slider ancrage doit être présent."""
        assert 'id="robust-anchor"' in html

    def test_ratio_slider_present(self, html):
        """Le slider ratio longueur doit être présent."""
        assert 'id="robust-ratio"' in html

    def test_robust_summary_present(self, html):
        """L'élément de résumé d'exclusion doit être présent."""
        assert 'id="robust-summary"' in html

    def test_robust_excluded_docs_present(self, html):
        """L'élément de liste des docs exclus doit être présent."""
        assert 'id="robust-excluded-docs"' in html

    def test_robust_table_wrap_present(self, html):
        """Le tableau comparatif doit avoir un conteneur."""
        assert 'id="robust-table-wrap"' in html


class TestGalleryHTML:
    """Vérifie les nouveaux éléments dans la vue galerie."""

    @pytest.fixture(scope="class")
    def html(self):
        return _generate_html()

    def test_reset_button_present(self, html):
        """Le bouton reset des exclusions doit être présent dans la galerie."""
        assert 'resetGalleryExclusions' in html

    def test_gallery_exclusion_info_present(self, html):
        """L'info d'exclusions manuelles doit être présente."""
        assert 'id="gallery-exclusion-info"' in html

    def test_gallery_reset_btn_present(self, html):
        """L'id gallery-reset-btn doit être présent."""
        assert 'id="gallery-reset-btn"' in html


# ---------------------------------------------------------------------------
# 2. JavaScript — fonctions de filtrage
# ---------------------------------------------------------------------------

class TestRobustMetricsJS:
    """Vérifie que les fonctions JS attendues sont présentes dans le HTML."""

    @pytest.fixture(scope="class")
    def html(self):
        return _generate_html()

    def test_render_robust_metrics_fn(self, html):
        """renderRobustMetrics doit être défini."""
        assert 'function renderRobustMetrics' in html

    def test_robust_stat_fn(self, html):
        """_robustStat (calcul médiane/p90/p95) doit être défini."""
        assert '_robustStat' in html

    def test_delta_cell_fn(self, html):
        """_deltaCell (affichage delta coloré) doit être défini."""
        assert '_deltaCell' in html

    def test_manual_exclusions_set(self, html):
        """_manualExclusions doit être déclaré."""
        assert '_manualExclusions' in html

    def test_toggle_gallery_exclusion_fn(self, html):
        """toggleGalleryExclusion doit être défini."""
        assert 'function toggleGalleryExclusion' in html

    def test_reset_gallery_exclusions_fn(self, html):
        """resetGalleryExclusions doit être défini."""
        assert 'function resetGalleryExclusions' in html

    def test_cer_threshold_used_in_js(self, html):
        """robust-cer doit être lu dans renderRobustMetrics."""
        # La fonction doit lire l'élément robust-cer
        assert "robust-cer" in html


# ---------------------------------------------------------------------------
# 3. JavaScript — export CSV deux feuilles
# ---------------------------------------------------------------------------

class TestExportCSVJS:
    """Vérifie que l'export CSV génère deux fichiers."""

    @pytest.fixture(scope="class")
    def html(self):
        return _generate_html()

    def test_build_csv_rows_fn(self, html):
        """_buildCSVRows doit être défini comme helper."""
        assert 'function _buildCSVRows' in html

    def test_download_csv_fn(self, html):
        """_downloadCSV doit être défini comme helper."""
        assert 'function _downloadCSV' in html

    def test_export_csv_fn_present(self, html):
        """exportCSV doit être défini."""
        assert 'function exportCSV' in html

    def test_robust_csv_filename(self, html):
        """Le nom du fichier robuste doit contenir '_robust'."""
        assert '_robust' in html

    def test_two_downloads_in_export(self, html):
        """exportCSV doit appeler _downloadCSV deux fois."""
        # On compte les appels à _downloadCSV dans la fonction exportCSV
        # (au moins deux occurrences dans le code)
        count = html.count('_downloadCSV(')
        assert count >= 2, f"Attendu ≥2 appels à _downloadCSV, trouvé {count}"


# ---------------------------------------------------------------------------
# 4. Nouvelles métriques robustes — logique Python dans DATA embarquée
# ---------------------------------------------------------------------------

class TestRobustMetricsData:
    """Vérifie que les données nécessaires sont bien présentes dans DATA."""

    @pytest.fixture(scope="class")
    def html(self):
        return _generate_html()

    def test_data_object_present(self, html):
        """Le rapport doit embarquer un objet DATA."""
        assert 'const DATA' in html or 'var DATA' in html

    def test_documents_in_data(self, html):
        """DATA.documents doit être présent."""
        assert '"documents"' in html

    def test_engine_results_in_data(self, html):
        """engine_results doit être présent par document."""
        assert '"engine_results"' in html

    def test_hallucination_metrics_key(self, html):
        """hallucination_metrics doit être référencé dans le JS."""
        assert 'hallucination_metrics' in html

    def test_line_metrics_key(self, html):
        """line_metrics doit être référencé dans le JS (pour gini)."""
        assert 'line_metrics' in html


# ---------------------------------------------------------------------------
# 5. Galerie — checkbox par carte
# ---------------------------------------------------------------------------

class TestGalleryCheckboxJS:
    """Vérifie la logique de checkbox dans renderGallery."""

    @pytest.fixture(scope="class")
    def html(self):
        return _generate_html()

    def test_checkbox_in_gallery_render(self, html):
        """renderGallery doit générer un input checkbox par carte."""
        assert 'type="checkbox"' in html or "type=\\'checkbox\\'" in html or 'type=\\\"checkbox\\\"' in html

    def test_toggle_on_change(self, html):
        """toggleGalleryExclusion doit être appelé à l'onchange du checkbox."""
        assert 'toggleGalleryExclusion' in html

    def test_excluded_card_style(self, html):
        """Les cartes exclues doivent avoir un style dégradé."""
        assert 'opacity' in html and 'isExcluded' in html

    def test_gallery_card_position_relative(self, html):
        """gallery-card doit avoir position:relative pour le checkbox overlay."""
        assert 'position: relative' in html or 'position:relative' in html


# ---------------------------------------------------------------------------
# 6. CSS — btn-secondary classe pour le bouton reset
# ---------------------------------------------------------------------------

class TestCSSClasses:
    """Vérifie la présence des classes CSS nécessaires."""

    @pytest.fixture(scope="class")
    def html(self):
        return _generate_html()

    def test_improved_class_exists(self, html):
        """La classe 'improved' (delta vert) doit exister dans le CSS."""
        assert '.improved' in html

    def test_worsened_class_exists(self, html):
        """La classe 'worsened' (delta rouge) doit exister dans le CSS."""
        assert '.worsened' in html

    def test_robust_controls_class(self, html):
        """robust-controls doit être un sélecteur CSS."""
        assert '.robust-controls' in html
