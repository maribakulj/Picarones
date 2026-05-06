"""Sprint A14-S38 — UI Jinja2 + i18n + static.

Tests de la page d'accueil HTML, des templates Jinja2, du loader
i18n FR/EN, et du mount des fichiers statiques.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from picarones.adapters.storage import JobStore
from picarones.app.services import (
    RegistryService,
    WorkspaceManager,
)
from picarones.interfaces.web import WebAppState, create_app
from picarones.interfaces.web.i18n import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    all_keys,
    translate,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_state(tmp_path: Path, with_jobs: bool = False) -> WebAppState:
    workspace = WorkspaceManager(
        base_dir=tmp_path,
        session_id="s38_test",
    )
    registry = RegistryService.bootstrap_defaults()
    job_store = JobStore(tmp_path / "jobs.db") if with_jobs else None
    return WebAppState(
        workspace=workspace,
        registry=registry,
        corpus=MagicMock(),
        benchmark=MagicMock(),
        orchestrator=MagicMock(),
        job_store=job_store,
        version="1.0.0-s38-test",
    )


# ──────────────────────────────────────────────────────────────────────
# i18n loader
# ──────────────────────────────────────────────────────────────────────


class TestI18nLoader:
    def test_translates_known_key_fr(self) -> None:
        result = translate("nav_home", "fr")
        assert result == "Accueil"

    def test_translates_known_key_en(self) -> None:
        result = translate("nav_home", "en")
        assert result == "Home"

    def test_unknown_language_falls_back_to_default(self) -> None:
        result = translate("nav_home", "klingon")
        # Fallback FR → "Accueil".
        assert result == "Accueil"

    def test_unknown_key_returns_key_itself(self) -> None:
        result = translate("missing_key_xyz", "fr")
        assert result == "missing_key_xyz"

    def test_default_language_constant(self) -> None:
        assert DEFAULT_LANGUAGE == "fr"

    def test_supported_languages_includes_fr_en(self) -> None:
        assert "fr" in SUPPORTED_LANGUAGES
        assert "en" in SUPPORTED_LANGUAGES


class TestI18nCompleteness:
    """Garde-fou : les deux langues doivent partager les mêmes clés."""

    def test_fr_and_en_have_same_keys(self) -> None:
        fr = set(all_keys("fr"))
        en = set(all_keys("en"))
        assert fr == en, (
            f"Asymétrie de clés : "
            f"FR-only = {fr - en}, EN-only = {en - fr}"
        )

    def test_critical_keys_present(self) -> None:
        critical = [
            "app_title", "nav_home", "nav_runs", "nav_jobs",
            "home_intro", "home_no_runs", "home_no_jobs",
            "footer_version",
        ]
        for key in critical:
            assert key in all_keys("fr")
            assert key in all_keys("en")


# ──────────────────────────────────────────────────────────────────────
# Page d'accueil HTML
# ──────────────────────────────────────────────────────────────────────


class TestHomePage:
    def test_home_returns_html(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        # Doctype HTML.
        assert response.text.lstrip().lower().startswith("<!doctype html>")

    def test_home_includes_app_title(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/")
        assert "Picarones" in response.text

    def test_home_in_french_by_default(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/")
        assert "Accueil" in response.text  # nav_home FR
        assert 'lang="fr"' in response.text

    def test_home_in_english_with_lang_param(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/?lang=en")
        assert "Home" in response.text  # nav_home EN
        assert 'lang="en"' in response.text

    def test_home_unknown_lang_falls_back_to_french(
        self, tmp_path: Path,
    ) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/?lang=klingon")
        # Fallback silencieux : on revient à FR (lang=fr dans <html>).
        assert 'lang="fr"' in response.text

    def test_home_shows_metric_count(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/")
        # Le compteur de métriques doit être affiché (texte + nombre).
        assert "métriques enregistrées" in response.text
        assert "projecteurs enregistrés" in response.text

    def test_home_workspace_root_displayed(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/")
        # Le chemin du workspace doit apparaître dans le HTML.
        assert "s38_test" in response.text

    def test_home_empty_state_runs(self, tmp_path: Path) -> None:
        """Sans run persisté, le message 'Aucun run' doit apparaître."""
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/")
        assert "Aucun run persisté" in response.text

    def test_home_lists_runs(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        runs_dir = Path(state.workspace.root) / "runs"
        runs_dir.mkdir()
        run_dir = runs_dir / "run_001"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text(
            json.dumps({
                "run_id": "run_001",
                "corpus_name": "demo",
                "n_documents": 5,
                "pipeline_names": ["tess"],
                "started_at": "2026-05-06T10:00:00Z",
            }),
            encoding="utf-8",
        )
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/")
        assert "run_001" in response.text
        assert "demo" in response.text
        assert "tess" in response.text

    def test_home_empty_state_jobs_without_store(
        self, tmp_path: Path,
    ) -> None:
        """Sans job_store, on affiche 'Aucun job'."""
        state = _make_state(tmp_path, with_jobs=False)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/")
        assert "Aucun job" in response.text

    def test_home_lists_jobs(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path, with_jobs=True)
        state.job_store.create("job_a", total_docs=10)
        state.job_store.mark_running("job_a")
        state.job_store.update_progress("job_a", 0.4)

        app = create_app(state)
        client = TestClient(app)
        response = client.get("/")
        assert "job_a" in response.text
        assert "running" in response.text
        assert "40%" in response.text


# ──────────────────────────────────────────────────────────────────────
# Static
# ──────────────────────────────────────────────────────────────────────


class TestStaticMount:
    def test_main_css_served(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/static/main.css")
        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]
        # Marqueur du CSS.
        assert "color-bg" in response.text

    def test_unknown_static_returns_404(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        app = create_app(state)
        client = TestClient(app)
        response = client.get("/static/nonexistent.css")
        assert response.status_code == 404
