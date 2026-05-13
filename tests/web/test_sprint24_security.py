"""Tests Sprint 24 — durcissement sécurité institutionnelle.

Le Sprint 24 ajoute quatre garde-fous orthogonaux à l'interface web :

1. **Mode public** (`PICARONES_PUBLIC_MODE=1`) — désactive les moteurs
   OCR cloud et les pipelines LLM dont les clefs API sont mutualisées
   côté serveur.
2. **Browse roots restreints** via `PICARONES_BROWSE_ROOTS` ou défaut
   adapté au mode (public = uploads seulement, dev = comportement
   historique).
3. **Validation d'image uploadée** (Pillow.verify, limite de taille,
   rejet des bombes de décompression).
4. **Rate limit + plafond de jobs concurrents** par IP.

Plus la **CSP** appliquée par middleware sur toutes les réponses HTTP.

Ces tests couvrent chaque garde-fou en unitaire (le module
``picarones.interfaces.web.security``) puis vérifient l'intégration côté FastAPI
en montant un ``TestClient``.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from picarones.interfaces.web import security as sec


# ---------------------------------------------------------------------------
# 1. Mode public — détection
# ---------------------------------------------------------------------------

class TestIsPublicMode:
    def test_default_is_dev_mode(self, monkeypatch):
        monkeypatch.delenv("PICARONES_PUBLIC_MODE", raising=False)
        assert sec.is_public_mode() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", " 1 "])
    def test_truthy_values_enable_public_mode(self, monkeypatch, value):
        monkeypatch.setenv("PICARONES_PUBLIC_MODE", value)
        assert sec.is_public_mode() is True

    @pytest.mark.parametrize("value", ["0", "", "false", "no", "off"])
    def test_falsy_values_keep_dev_mode(self, monkeypatch, value):
        monkeypatch.setenv("PICARONES_PUBLIC_MODE", value)
        assert sec.is_public_mode() is False


# ---------------------------------------------------------------------------
# 2. Engines autorisés / bloqués
# ---------------------------------------------------------------------------

class TestAssertEnginesAllowed:
    def test_dev_mode_allows_cloud_engines(self, monkeypatch):
        monkeypatch.delenv("PICARONES_PUBLIC_MODE", raising=False)
        sec.assert_engines_allowed(["mistral_ocr", "google_vision"])  # ne lève pas

    def test_public_mode_blocks_cloud_ocr(self, monkeypatch):
        monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
        with pytest.raises(PermissionError, match="cloud"):
            sec.assert_engines_allowed(["mistral_ocr"])

    def test_public_mode_allows_local_engines(self, monkeypatch):
        monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
        sec.assert_engines_allowed(["tesseract", "pero_ocr"])  # ne lève pas

    def test_public_mode_blocks_llm_provider(self, monkeypatch):
        monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
        for provider in ("openai", "anthropic", "mistral"):
            with pytest.raises(PermissionError, match="OCR\\+LLM"):
                sec.assert_llm_provider_allowed(provider)

    def test_empty_provider_is_noop(self, monkeypatch):
        monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
        sec.assert_llm_provider_allowed("")  # ne lève pas


# ---------------------------------------------------------------------------
# 3. Browse roots
# ---------------------------------------------------------------------------

class TestComputeBrowseRoots:
    def test_env_var_overrides_default(self, monkeypatch, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        import os as _os
        monkeypatch.setenv("PICARONES_BROWSE_ROOTS", str(a) + _os.pathsep + str(b))
        roots = sec.compute_browse_roots(tmp_path)
        assert a.resolve() in roots
        assert b.resolve() in roots

    def test_public_mode_default_restricts_to_uploads(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
        monkeypatch.delenv("PICARONES_BROWSE_ROOTS", raising=False)
        roots = sec.compute_browse_roots(tmp_path)
        assert roots == [tmp_path.resolve()]

    def test_dev_mode_default_is_legacy(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PICARONES_PUBLIC_MODE", raising=False)
        monkeypatch.delenv("PICARONES_BROWSE_ROOTS", raising=False)
        roots = sec.compute_browse_roots(tmp_path)
        # cwd + uploads + /workspaces + tempdir : 4 entrées
        assert len(roots) >= 2
        assert tmp_path.resolve() in roots


# ---------------------------------------------------------------------------
# 4. Validation d'image
# ---------------------------------------------------------------------------

def _png_bytes(width: int = 10, height: int = 10) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestValidateImageSafe:
    def test_valid_png_passes(self):
        sec.validate_image_safe(_png_bytes(), filename="ok.png")

    def test_size_limit_rejects_large_buffer(self, monkeypatch):
        monkeypatch.setenv("PICARONES_MAX_UPLOAD_MB", "1")
        big = b"\x00" * (2 * 1024 * 1024)  # 2 Mo
        with pytest.raises(ValueError, match="taille"):
            sec.validate_image_safe(big, filename="big.png")

    def test_garbage_bytes_rejected(self):
        with pytest.raises(ValueError):
            sec.validate_image_safe(b"this is not an image", filename="bad.png")

    def test_php_pretending_to_be_png_rejected(self):
        php = b"<?php phpinfo(); ?>" * 100
        with pytest.raises(ValueError):
            sec.validate_image_safe(php, filename="evil.png")


# ---------------------------------------------------------------------------
# 5. Rate limiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_zero_quota_disables_limit(self):
        rl = sec.RateLimiter(max_per_hour=0)
        for _ in range(50):
            rl.check("1.2.3.4")  # ne lève jamais

    def test_quota_enforced(self):
        rl = sec.RateLimiter(max_per_hour=3)
        rl.check("1.2.3.4")
        rl.check("1.2.3.4")
        rl.check("1.2.3.4")
        with pytest.raises(PermissionError, match="Quota"):
            rl.check("1.2.3.4")

    def test_quota_per_ip(self):
        rl = sec.RateLimiter(max_per_hour=2)
        rl.check("1.1.1.1")
        rl.check("1.1.1.1")
        # Une autre IP n'est pas affectée
        rl.check("2.2.2.2")
        rl.check("2.2.2.2")

    def test_reset_clears_buckets(self):
        rl = sec.RateLimiter(max_per_hour=1)
        rl.check("1.2.3.4")
        rl.reset()
        rl.check("1.2.3.4")  # ne lève plus


# ---------------------------------------------------------------------------
# 6. Helpers env vars
# ---------------------------------------------------------------------------

class TestEnvVarHelpers:
    def test_get_max_upload_mb_default(self, monkeypatch):
        monkeypatch.delenv("PICARONES_MAX_UPLOAD_MB", raising=False)
        assert sec.get_max_upload_mb() == 100

    def test_get_max_upload_mb_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("PICARONES_MAX_UPLOAD_MB", "not-a-number")
        assert sec.get_max_upload_mb() == 100

    def test_get_max_concurrent_jobs_clamped(self, monkeypatch):
        monkeypatch.setenv("PICARONES_MAX_CONCURRENT_JOBS", "0")
        assert sec.get_max_concurrent_jobs() == 1  # min 1

    def test_rate_limit_zero_in_dev(self, monkeypatch):
        monkeypatch.delenv("PICARONES_PUBLIC_MODE", raising=False)
        assert sec.get_rate_limit_per_hour() == 0

    def test_rate_limit_default_in_public(self, monkeypatch):
        monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
        monkeypatch.delenv("PICARONES_RATE_LIMIT_PER_HOUR", raising=False)
        assert sec.get_rate_limit_per_hour() == 5


# ---------------------------------------------------------------------------
# 7. CSP middleware (intégration FastAPI)
# ---------------------------------------------------------------------------

class TestCSPHeaders:
    @pytest.fixture
    def client(self):
        from picarones.interfaces.web.app import app
        return TestClient(app)

    def test_csp_header_present(self, client, monkeypatch):
        # Mode local strict — pas de SPACE_ID dans l'env.
        monkeypatch.delenv("SPACE_ID", raising=False)
        r = client.get("/api/status")
        assert r.status_code == 200
        assert "Content-Security-Policy" in r.headers
        csp = r.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp

    def test_security_headers_present(self, client, monkeypatch):
        monkeypatch.delenv("SPACE_ID", raising=False)
        r = client.get("/api/status")
        assert r.headers.get("X-Content-Type-Options") == "nosniff"
        assert r.headers.get("X-Frame-Options") == "DENY"
        assert r.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_csp_allows_huggingface_iframe_when_on_space(self, client, monkeypatch):
        """Sur HF Space (``SPACE_ID`` défini), la CSP doit autoriser l'embed.

        Garde-fou contre la régression historique « page blanche sur HF » :
        ``frame-ancestors 'none'`` masquait la SPA dans l'iframe parente du
        Hub HuggingFace.
        """
        monkeypatch.setenv("SPACE_ID", "Ma-Ri-Ba-Ku/Picarones")
        r = client.get("/api/status")
        csp = r.headers["Content-Security-Policy"]
        assert "frame-ancestors" in csp
        assert "'none'" not in csp.split("frame-ancestors")[1].split(";")[0]
        assert "huggingface.co" in csp
        assert "*.hf.space" in csp

    def test_x_frame_options_omitted_on_huggingface_space(self, client, monkeypatch):
        """Sur HF Space, ``X-Frame-Options: DENY`` doit être absent.

        Ce header a priorité absolue sur ``frame-ancestors`` dans les anciens
        navigateurs (et reste un fallback moderne) ; le laisser à ``DENY``
        bloque l'iframe parente même avec la CSP permissive.
        """
        monkeypatch.setenv("SPACE_ID", "Ma-Ri-Ba-Ku/Picarones")
        r = client.get("/api/status")
        assert "X-Frame-Options" not in r.headers
        # Les autres headers de durcissement restent intacts.
        assert r.headers.get("X-Content-Type-Options") == "nosniff"
        assert r.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_csp_override_via_env_takes_precedence(self, client, monkeypatch):
        """``PICARONES_CSP`` reste un override absolu pour l'admin."""
        monkeypatch.setenv("PICARONES_CSP", "default-src 'self'; frame-ancestors 'self'")
        monkeypatch.setenv("SPACE_ID", "Ma-Ri-Ba-Ku/Picarones")  # ignoré
        r = client.get("/api/status")
        csp = r.headers["Content-Security-Policy"]
        assert csp == "default-src 'self'; frame-ancestors 'self'"


# ---------------------------------------------------------------------------
# 8. Public mode bloque les benchmarks LLM (intégration FastAPI)
# ---------------------------------------------------------------------------

class TestPublicModeBlocksLLMBenchmark:
    """Vérifie que ``/api/benchmark/run`` refuse en 403 quand un compétiteur
    référence un ``llm_provider`` mutualisé en mode public.

    On contourne le rate limiter en réinjectant un quota nul (mode dev) après
    avoir enclenché le mode public — l'objectif du test est de vérifier le
    refus 403, pas le 429.
    """

    @pytest.fixture
    def client_public(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
        # Désactive le rate limit dans cette session
        from picarones.interfaces.web import app as web_app
        from picarones.interfaces.web import state as web_state
        web_state.RATE_LIMITER.reset()
        web_state.RATE_LIMITER.max_per_hour = 0  # type: ignore[attr-defined]
        # Crée un faux corpus pour passer le contrôle d'existence
        corpus = tmp_path / "corp"
        corpus.mkdir()
        return TestClient(web_app.app), str(corpus)

    def test_run_blocks_openai_competitor(self, client_public):
        client, corpus_path = client_public
        body = {
            "corpus_path": corpus_path,
            "competitors": [
                {
                    "name": "test",
                    "engine_name": "tesseract",
                    "llm_provider": "openai",
                    "llm_model": "gpt-4o",
                    "pipeline_mode": "text_only",
                },
            ],
        }
        r = client.post("/api/benchmark/run", json=body)
        assert r.status_code == 403, r.text
        assert "public" in r.json()["detail"].lower()

    def test_run_blocks_cloud_ocr(self, client_public):
        client, corpus_path = client_public
        body = {
            "corpus_path": corpus_path,
            "competitors": [
                {
                    "engine_name": "mistral_ocr",
                    "llm_provider": "",
                },
            ],
        }
        r = client.post("/api/benchmark/run", json=body)
        assert r.status_code == 403, r.text

    def test_start_blocks_cloud_ocr_engine(self, client_public):
        client, corpus_path = client_public
        body = {
            "corpus_path": corpus_path,
            "engines": ["google_vision"],
        }
        r = client.post("/api/benchmark/start", json=body)
        assert r.status_code == 403, r.text

    def test_start_allows_local_tesseract(self, client_public, monkeypatch):
        # Sous mode public, un benchmark Tesseract local doit passer le
        # garde-fou (le fait qu'il échoue ensuite faute de Tesseract est
        # hors-périmètre — on vérifie juste que ce n'est pas un 403).
        client, corpus_path = client_public
        body = {
            "corpus_path": corpus_path,
            "engines": ["tesseract"],
        }
        r = client.post("/api/benchmark/start", json=body)
        assert r.status_code != 403, r.text


# ---------------------------------------------------------------------------
# 9. _is_path_allowed honore les browse roots
# ---------------------------------------------------------------------------

class TestPathAllowed:
    def test_path_outside_roots_is_blocked(self, monkeypatch, tmp_path):
        # On force le calcul des roots à uploads_dir uniquement
        monkeypatch.setenv("PICARONES_BROWSE_ROOTS", str(tmp_path))
        from picarones.interfaces.web.routers import corpus as web_corpus
        web_corpus._BROWSE_ROOTS = sec.compute_browse_roots(tmp_path)

        outside = Path("/etc").resolve()
        assert web_corpus._is_path_allowed(outside) is False

        inside = tmp_path / "sub"
        inside.mkdir()
        assert web_corpus._is_path_allowed(inside.resolve()) is True
