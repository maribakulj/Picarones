"""Sprint A14-S21 — ``ReportService`` (rendu HTML depuis ``RunResult``).

Couverture :

- Rendu basique : header (corpus, run_id, code_version, timestamps),
  vue d'ensemble pipelines (succès/échecs/durée), une section par
  vue avec table pipeline × métriques.
- **Pattern d'omission visible** : un pipeline qui ne produit pas
  d'artefact éligible affiche ``OMIS`` (pas un ``0`` factice).
- Anti-injection : ``corpus_name`` / ``view.name`` /
  ``pipeline_name`` contenant ``<script>`` sont échappés.
- Persistance round-trip : ``BenchmarkService.persist`` → 3 fichiers
  → ``ReportService.render_from_dir`` → HTML équivalent au rendu
  in-memory.
- Bilingue : labels FR vs EN distincts.
- Cas dégénérés : RunResult vide, vue sans aucun ViewResult.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from picarones.reports_v2.html import HtmlReportRenderer as ReportService
from picarones.domain.evaluation_spec import EvaluationView
from picarones.domain.artifacts import ArtifactType
from picarones.domain.run_manifest import RunManifest
from picarones.app.results import RunResult
from picarones.evaluation.views.base import ViewResult


# ──────────────────────────────────────────────────────────────────
# Helpers de fabrication de RunResult synthétique
# ──────────────────────────────────────────────────────────────────


def _empty_view(
    *,
    name: str = "text_final",
    description: str = "Vue texte final",
    candidate_types: frozenset[ArtifactType] | None = None,
    metric_names: tuple[str, ...] = ("cer", "wer"),
    warnings: tuple[str, ...] = (),
    ignored_dimensions: tuple[str, ...] = (),
) -> EvaluationView:
    return EvaluationView(
        name=name,
        description=description,
        candidate_types=(
            candidate_types if candidate_types is not None
            else frozenset({ArtifactType.RAW_TEXT})
        ),
        projection=None,
        projections_by_source_type={},
        metric_names=metric_names,
        warnings=warnings,
        ignored_dimensions=ignored_dimensions,
    )


def _manifest(
    *,
    corpus_name: str = "test_corpus",
    pipeline_names: tuple[str, ...] = ("pA", "pB"),
    views: tuple[EvaluationView, ...] = (),
    run_id: str = "test_run_001",
    code_version: str = "1.0.0-s21",
    n_documents: int = 2,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        corpus_name=corpus_name,
        n_documents=n_documents,
        pipeline_names=pipeline_names,
        view_specs=views,
        code_version=code_version,
        started_at=datetime(2026, 5, 4, 10, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 5, 4, 10, 0, 1, tzinfo=timezone.utc),
        dependencies_lock={},
        metadata={},
    )


# ──────────────────────────────────────────────────────────────────
# Fixture : run BnF S18 — pour tests d'intégration end-to-end
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def bnf_run_result(tmp_path: Path) -> RunResult:
    """Réutilise le scénario E2E S18 pour un RunResult réaliste."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_sprint_a14_s18_bnf_e2e import _run_full_benchmark
    _, result = _run_full_benchmark(tmp_path)
    return result


# ──────────────────────────────────────────────────────────────────
# Rendu basique
# ──────────────────────────────────────────────────────────────────


class TestBasicRendering:
    def test_render_returns_complete_html_document(self) -> None:
        view = _empty_view()
        manifest = _manifest(views=(view,))
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService().render(result)
        assert html.startswith("<!DOCTYPE html>")
        assert html.rstrip().endswith("</html>")
        assert '<meta charset="utf-8">' in html
        assert "<style>" in html

    def test_header_contains_manifest_fields(self) -> None:
        view = _empty_view()
        manifest = _manifest(
            corpus_name="bnf_xviiie",
            run_id="bnf_xviiie_20260504T100001Z",
            code_version="2.1.0",
            views=(view,),
        )
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService().render(result)
        assert "bnf_xviiie" in html
        assert "bnf_xviiie_20260504T100001Z" in html
        assert "2.1.0" in html
        # Timestamp ISO.
        assert "2026-05-04T10:00:00" in html

    def test_pipelines_overview_lists_all_manifest_pipelines(self) -> None:
        view = _empty_view()
        manifest = _manifest(
            pipeline_names=("alpha", "beta", "gamma"),
            views=(view,),
        )
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService().render(result)
        # Les 3 pipelines apparaissent même sans aucun PipelineResult.
        for name in ("alpha", "beta", "gamma"):
            assert name in html

    def test_one_section_per_view(self) -> None:
        v1 = _empty_view(name="text_final")
        v2 = _empty_view(name="alto_documentary")
        v3 = _empty_view(name="searchability")
        manifest = _manifest(views=(v1, v2, v3))
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService().render(result)
        assert 'id="view-text_final"' in html
        assert 'id="view-alto_documentary"' in html
        assert 'id="view-searchability"' in html


# ──────────────────────────────────────────────────────────────────
# Pattern d'omission visible
# ──────────────────────────────────────────────────────────────────


class TestOmissionVisibility:
    def test_pipeline_with_no_view_results_is_marked_omitted(
        self, bnf_run_result: RunResult,
    ) -> None:
        """Sur le scénario BnF S18, AltoView omet ``pipeline_simple_ocr``
        et ``pipeline_ocr_plus_correction``."""
        html = ReportService().render(bnf_run_result)
        # Trouver la section AltoView et vérifier les omissions.
        alto_section = _extract_section(html, "alto_documentary")
        # Les 2 pipelines omises doivent apparaître avec OMIS, le 3ème
        # avec des valeurs numériques.
        assert "pipeline_simple_ocr" in alto_section
        assert "pipeline_ocr_plus_correction" in alto_section
        # Au moins 2 cellules OMIS dans la section AltoView.
        assert alto_section.count("OMIS") >= 2

    def test_omitted_cell_explains_why(
        self, bnf_run_result: RunResult,
    ) -> None:
        html = ReportService().render(bnf_run_result)
        # Le tooltip explique l'omission (FR par défaut).  ``html.escape``
        # transforme les apostrophes en &#x27; — on cherche les
        # versions échappées.
        assert "ne produisant pas d&#x27;artefact" in html
        assert "Pas de score factice" in html

    def test_no_omitted_marker_on_view_where_all_eligible(
        self, bnf_run_result: RunResult,
    ) -> None:
        """TextView accepte tous les pipelines BnF → pas de OMIS."""
        html = ReportService().render(bnf_run_result)
        text_section = _extract_section(html, "text_final")
        assert "OMIS" not in text_section


# ──────────────────────────────────────────────────────────────────
# Anti-injection HTML
# ──────────────────────────────────────────────────────────────────


class TestAntiInjection:
    def test_corpus_name_with_html_is_escaped(self) -> None:
        view = _empty_view()
        manifest = _manifest(
            corpus_name="<script>alert(1)</script>",
            views=(view,),
        )
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService().render(result)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

    def test_pipeline_name_with_html_is_escaped(self) -> None:
        view = _empty_view()
        manifest = _manifest(
            pipeline_names=("<img src=x onerror=alert(1)>",),
            views=(view,),
        )
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService().render(result)
        assert "<img src=x" not in html
        assert "&lt;img src=x" in html

    def test_view_name_and_description_are_escaped(self) -> None:
        view = _empty_view(
            name="evil_name",
            description='</style><script>x</script>',
        )
        manifest = _manifest(views=(view,))
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService().render(result)
        assert "</style><script>" not in html
        assert "&lt;/style&gt;&lt;script&gt;" in html

    def test_view_warning_is_escaped(self) -> None:
        view = _empty_view(warnings=("<b>injected</b>",))
        manifest = _manifest(views=(view,))
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService().render(result)
        assert "<b>injected</b>" not in html
        assert "&lt;b&gt;injected&lt;/b&gt;" in html


# ──────────────────────────────────────────────────────────────────
# Persistance round-trip
# ──────────────────────────────────────────────────────────────────


class TestPersistenceRoundTrip:
    def test_render_from_dir_matches_render(
        self, bnf_run_result: RunResult, tmp_path: Path,
    ) -> None:
        """Persister puis re-render produit le MÊME HTML que le render
        in-memory : preuve byte-à-byte que la persistance est lossless
        pour les besoins du rapport."""
        from picarones.app.services import BenchmarkService
        # On a besoin d'un BenchmarkService pour appeler persist —
        # mais on peut court-circuiter en utilisant le helper interne.
        out_dir = tmp_path / "persisted"
        # Construire un BenchmarkService bidon juste pour persist :
        # ses deux dépendances ne sont pas appelées par persist().
        from picarones.evaluation.registry import MetricRegistry
        from picarones.evaluation.projectors import ProjectorRegistry
        from picarones.evaluation.views import DefaultEvaluationViewExecutor
        from picarones.pipeline import CorpusRunner, PipelineExecutor
        loader = lambda art: ""  # noqa: E731 — non appelé par persist
        view_executor = DefaultEvaluationViewExecutor.from_registries(
            MetricRegistry(), ProjectorRegistry(), loader,
        )
        runner = CorpusRunner(
            PipelineExecutor(adapter_resolver=lambda n: None),
            max_in_flight=1,
            timeout_seconds_per_doc=1.0,
            poll_interval_seconds=0.001,
        )
        bench = BenchmarkService(
            corpus_runner=runner,
            view_executor=view_executor,
            code_version="1.0.0-s18-bnf-test",
        )
        bench.persist(bnf_run_result, out_dir)

        svc = ReportService()
        html_in_memory = svc.render(bnf_run_result)
        html_from_disk = svc.render_from_dir(out_dir)
        assert html_from_disk == html_in_memory

    def test_load_run_result_roundtrip_preserves_structure(
        self, bnf_run_result: RunResult, tmp_path: Path,
    ) -> None:
        from picarones.app.services import BenchmarkService
        from picarones.evaluation.registry import MetricRegistry
        from picarones.evaluation.projectors import ProjectorRegistry
        from picarones.evaluation.views import DefaultEvaluationViewExecutor
        from picarones.pipeline import CorpusRunner, PipelineExecutor
        loader = lambda art: ""  # noqa: E731
        view_executor = DefaultEvaluationViewExecutor.from_registries(
            MetricRegistry(), ProjectorRegistry(), loader,
        )
        runner = CorpusRunner(
            PipelineExecutor(adapter_resolver=lambda n: None),
            max_in_flight=1,
            timeout_seconds_per_doc=1.0,
            poll_interval_seconds=0.001,
        )
        bench = BenchmarkService(
            corpus_runner=runner,
            view_executor=view_executor,
            code_version="1.0.0-s18-bnf-test",
        )
        out_dir = tmp_path / "persisted2"
        bench.persist(bnf_run_result, out_dir)
        loaded = ReportService.load_run_result(out_dir)
        assert loaded.manifest.corpus_name == bnf_run_result.manifest.corpus_name
        assert loaded.n_documents == bnf_run_result.n_documents
        # Comptes de view_results identiques par vue.
        for view in bnf_run_result.manifest.view_specs:
            assert (
                len(loaded.view_results_for(view.name))
                == len(bnf_run_result.view_results_for(view.name))
            )

    def test_load_run_result_raises_on_missing_files(
        self, tmp_path: Path,
    ) -> None:
        empty_dir = tmp_path / "nothing"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="run_manifest.json"):
            ReportService.load_run_result(empty_dir)


# ──────────────────────────────────────────────────────────────────
# Bilingue FR / EN
# ──────────────────────────────────────────────────────────────────


class TestI18N:
    def test_french_labels_by_default(self) -> None:
        view = _empty_view()
        manifest = _manifest(views=(view,))
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService().render(result)
        assert 'lang="fr"' in html
        assert "Pipelines exécutées" in html
        assert "Avertissements" in html or "Démarré" in html

    def test_english_labels(self) -> None:
        view = _empty_view()
        manifest = _manifest(views=(view,))
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService(lang="en").render(result)
        assert 'lang="en"' in html
        assert "Pipelines executed" in html

    def test_unknown_lang_falls_back_to_french(self) -> None:
        view = _empty_view()
        manifest = _manifest(views=(view,))
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService(lang="xx").render(result)
        assert 'lang="fr"' in html


# ──────────────────────────────────────────────────────────────────
# Cas dégénérés
# ──────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_run_result_renders_without_crashing(self) -> None:
        manifest = _manifest(views=(), pipeline_names=(), n_documents=0)
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService().render(result)
        assert "<!DOCTYPE html>" in html

    def test_view_with_no_view_results_shows_empty_message(self) -> None:
        view = _empty_view(name="lonely_view")
        manifest = _manifest(views=(view,), pipeline_names=())
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService().render(result)
        section = _extract_section(html, "lonely_view")
        # Soit le message "Aucun pipeline" est rendu, soit le tableau
        # est vide (aucune ligne).  Les deux comportements sont OK
        # pour S21.
        assert (
            "Aucun pipeline" in section
            or "<tbody>\n\n</tbody>" in section
        )

    def test_view_displays_warnings_block(self) -> None:
        view = _empty_view(warnings=("Attention : projection lossy.",))
        manifest = _manifest(views=(view,))
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService().render(result)
        assert "Attention : projection lossy." in html
        assert 'class="warnings"' in html

    def test_view_displays_ignored_dimensions(self) -> None:
        view = _empty_view(
            ignored_dimensions=("geometry", "block_structure"),
        )
        manifest = _manifest(views=(view,))
        result = RunResult(manifest=manifest, document_results=())
        html = ReportService().render(result)
        assert "geometry, block_structure" in html


# ──────────────────────────────────────────────────────────────────
# Smoke : rendu complet du scénario BnF S18
# ──────────────────────────────────────────────────────────────────


class TestSmokeBnFScenario:
    def test_bnf_report_contains_all_3_pipelines_and_3_views(
        self, bnf_run_result: RunResult,
    ) -> None:
        html = ReportService().render(bnf_run_result)
        # Pipelines.
        for name in (
            "pipeline_simple_ocr",
            "pipeline_structured_ocr",
            "pipeline_ocr_plus_correction",
        ):
            assert name in html
        # Vues.
        for name in (
            "text_final",
            "alto_documentary",
            "searchability",
        ):
            assert f'id="view-{name}"' in html

    def test_bnf_metric_values_appear(
        self, bnf_run_result: RunResult,
    ) -> None:
        html = ReportService().render(bnf_run_result)
        # Au moins une métrique numérique dans la section TextView
        # (CER 0.0000 pour structured_ocr).
        text_section = _extract_section(html, "text_final")
        # Format ".4f" → quelque chose comme "0.0000" ou "0.0250".
        assert re.search(r"[01]\.\d{4}", text_section), (
            "aucune valeur numérique 4-digit trouvée dans TextView"
        )


# ──────────────────────────────────────────────────────────────────
# Helpers de tests
# ──────────────────────────────────────────────────────────────────


def _extract_section(html: str, view_name: str) -> str:
    """Extrait le HTML de la section ``<section id="view-{view_name}">``
    jusqu'au ``</section>`` correspondant."""
    marker = f'id="view-{view_name}"'
    start = html.find(marker)
    assert start != -1, f"section {view_name!r} introuvable dans le HTML"
    # On remonte au début de <section.
    section_start = html.rfind("<section", 0, start)
    section_end = html.find("</section>", start) + len("</section>")
    return html[section_start:section_end]


# Helper pour calmer pyflakes : ViewResult importé pour signaler
# l'intention de signature des helpers internes du service.
_ = ViewResult
