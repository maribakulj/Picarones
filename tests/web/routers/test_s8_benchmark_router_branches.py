"""Sprint S8.7 — couverture des branches non-SSE du benchmark router.

Cible : lignes 100, 163, 170, 223 de
``picarones/interfaces/web/routers/benchmark.py``

- 100 : ``/api/benchmark/start`` retourne 429 quand le sémaphore
  des jobs concurrents est plein ;
- 163 : ``validated_prompt_filename`` est appelé pour chaque
  ``PipelineConfig.prompt_file`` non-vide → un nom de prompt
  invalide doit être rejeté en 400 (vecteur d'exfiltration LLM) ;
- 170 : ``/api/benchmark/run`` retourne 429 quand le sémaphore
  est plein ;
- 223 : ``/api/benchmark/{id}/cancel`` retourne idempotent quand
  le job est déjà ``complete`` ou ``error``.

Le SSE event generator (lignes 286-316) n'est pas couvert ici —
il exige des fixtures async + une simulation de cycle de vie de
job non triviale (tests dédiés ``test_sprint26_*``).
"""

from __future__ import annotations

import threading

import pytest


def _make_app(monkeypatch, tmp_path):
    """App avec ``UPLOADS_DIR`` et workspace_roots qui pointent vers
    ``tmp_path`` pour faire passer la validation des chemins.
    """
    from fastapi import FastAPI

    from picarones.interfaces.web.routers import benchmark as benchmark_router
    from picarones.interfaces.web.routers import corpus as corpus_router

    monkeypatch.setattr(corpus_router, "UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(benchmark_router, "UPLOADS_DIR", tmp_path)

    app = FastAPI()
    app.include_router(benchmark_router.router)
    return app


# ──────────────────────────────────────────────────────────────────────
# 429 — sémaphore de jobs concurrents épuisé
# ──────────────────────────────────────────────────────────────────────


class TestSemaphoreFull429:
    def test_start_returns_429_when_semaphore_exhausted(
        self, monkeypatch, tmp_path,
    ) -> None:
        """``/api/benchmark/start`` doit retourner 429 (pas planter)
        quand ``JOBS_SEMAPHORE.acquire(blocking=False)`` retourne
        False — le worker ops a bien un signal d'epuisement."""
        from fastapi.testclient import TestClient

        from picarones.interfaces.web import state as web_state

        # Crée le corpus et le rapports/ exigés par la validation.
        corpus = tmp_path / "corpus_dir"
        corpus.mkdir()
        rapports = tmp_path / "rapports"
        rapports.mkdir()

        # Sémaphore capacité 0 — jamais acquérable.
        monkeypatch.setattr(
            web_state, "JOBS_SEMAPHORE", threading.Semaphore(0),
        )

        app = _make_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            r = client.post(
                "/api/benchmark/start",
                json={
                    "corpus_path": str(corpus),
                    "engines": ["tesseract"],
                    "output_dir": str(rapports),
                    "lang": "fra",
                },
            )
            assert r.status_code == 429, r.text
            assert (
                "concurrents" in r.text.lower()
                or "max" in r.text.lower()
            )

    def test_run_returns_429_when_semaphore_exhausted(
        self, monkeypatch, tmp_path,
    ) -> None:
        from fastapi.testclient import TestClient

        from picarones.interfaces.web import state as web_state

        corpus = tmp_path / "corpus_dir"
        corpus.mkdir()
        rapports = tmp_path / "rapports"
        rapports.mkdir()

        monkeypatch.setattr(
            web_state, "JOBS_SEMAPHORE", threading.Semaphore(0),
        )

        app = _make_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            r = client.post(
                "/api/benchmark/run",
                json={
                    "corpus_path": str(corpus),
                    "competitors": [
                        {
                            "name": "t",
                            "ocr_engine": "tesseract",
                            "ocr_model": "fra",
                            "llm_provider": "",
                        },
                    ],
                    "output_dir": str(rapports),
                },
            )
            assert r.status_code == 429, r.text


# ──────────────────────────────────────────────────────────────────────
# Validation des prompts (sécurité exfiltration LLM)
# ──────────────────────────────────────────────────────────────────────


class TestPromptFileValidation:
    def test_prompt_file_traversal_returns_400(
        self, monkeypatch, tmp_path,
    ) -> None:
        """Un ``prompt_file`` qui tente de pointer hors de la
        bibliothèque embarquée (``../../etc/passwd``) doit être
        rejeté en 400 — branche ``validated_prompt_filename``
        levée et capturée comme ``PathValidationError``."""
        from fastapi.testclient import TestClient

        corpus = tmp_path / "corpus_dir"
        corpus.mkdir()
        rapports = tmp_path / "rapports"
        rapports.mkdir()

        app = _make_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            r = client.post(
                "/api/benchmark/run",
                json={
                    "corpus_path": str(corpus),
                    "competitors": [
                        {
                            "name": "t",
                            "ocr_engine": "tesseract",
                            "ocr_model": "fra",
                            "llm_provider": "mistral",
                            "llm_model": "ministral-3b-latest",
                            "prompt_file": "../../../etc/passwd",
                        },
                    ],
                    "output_dir": str(rapports),
                },
            )
            assert r.status_code == 400, r.text


# ──────────────────────────────────────────────────────────────────────
# /cancel idempotent sur jobs déjà terminés
# ──────────────────────────────────────────────────────────────────────


class TestCancelIdempotent:
    @pytest.mark.parametrize("terminal_status", ["complete", "error"])
    def test_cancel_already_finished_job_is_noop(
        self, monkeypatch, tmp_path, terminal_status: str,
    ) -> None:
        """``/cancel`` sur un job ``complete`` ou ``error`` doit
        retourner 200 + message ``déjà terminé`` (pas 4xx) — un
        client qui retry ne doit pas voir une erreur."""
        import uuid

        from fastapi.testclient import TestClient

        from picarones.interfaces.web import state as web_state

        # ``job_id`` unique par paramètre — sinon
        # ``JOB_STORE.create_job`` viole la contrainte UNIQUE entre
        # les deux invocations du paramétrage.
        job_id = f"test_job_finished_{terminal_status}_{uuid.uuid4().hex[:8]}"
        job = web_state.BenchmarkJob(
            job_id=job_id, _store=web_state.JOB_STORE,
        )
        web_state.JOB_STORE.create_job(job_id)
        job.set_status(terminal_status)
        web_state.register_job(job)

        app = _make_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            r = client.post(f"/api/benchmark/{job_id}/cancel")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["status"] == terminal_status
            assert "terminé" in body["message"]

    def test_cancel_unknown_job_returns_404(
        self, monkeypatch, tmp_path,
    ) -> None:
        from fastapi.testclient import TestClient

        app = _make_app(monkeypatch, tmp_path)
        with TestClient(app) as client:
            r = client.post(
                "/api/benchmark/never_existed_xyz/cancel",
            )
            assert r.status_code == 404
