"""Sprint S8.7 — couverture des helpers env-var fallback et
défense Pillow de ``picarones/interfaces/web/security.py``.

Cible (avant) : 92.18% patch coverage avec 15 lignes manquantes
sur des chemins testables sans mock lourd :

- ``compute_workspace_roots`` avec ``PICARONES_WORKSPACE_ROOTS`` set ;
- ``get_max_upload_mb`` / ``get_max_concurrent_jobs`` /
  ``get_rate_limit_per_hour`` sur valeur invalide → fallback log ;
- ``validate_image_safe`` sur ``DecompressionBombError`` (vraie
  image bomb simulée via abaissement temporaire de
  ``MAX_IMAGE_PIXELS``) ;
- ``_get_csrf_secret`` génère un secret runtime quand
  ``PICARONES_CSRF_SECRET`` absent ;
- ``RateLimiter.check`` purge les hits hors fenêtre.

Tous les tests sont des assertions de comportement réel — pas
de simple « ça ne plante pas ».
"""

from __future__ import annotations

import io
import os
import time

import pytest


# ──────────────────────────────────────────────────────────────────────
# Env var fallbacks — doivent retourner le default sur valeur invalide
# ──────────────────────────────────────────────────────────────────────


class TestEnvVarFallbacks:
    def test_max_upload_mb_invalid_returns_default(
        self, monkeypatch, caplog,
    ) -> None:
        from picarones.interfaces.web.security import get_max_upload_mb

        monkeypatch.setenv("PICARONES_MAX_UPLOAD_MB", "not-a-number")
        with caplog.at_level("WARNING"):
            value = get_max_upload_mb()
        assert value == 100, "default value not returned on invalid env"
        assert any(
            "PICARONES_MAX_UPLOAD_MB" in rec.message for rec in caplog.records
        ), "warning log not emitted on invalid env"

    def test_max_upload_mb_valid_overrides_default(
        self, monkeypatch,
    ) -> None:
        from picarones.interfaces.web.security import get_max_upload_mb

        monkeypatch.setenv("PICARONES_MAX_UPLOAD_MB", "250")
        assert get_max_upload_mb() == 250

    def test_max_upload_mb_clamped_to_one(self, monkeypatch) -> None:
        """Valeur ≤ 0 → clampée à 1 (pas un upload de 0 Mo accepté)."""
        from picarones.interfaces.web.security import get_max_upload_mb

        monkeypatch.setenv("PICARONES_MAX_UPLOAD_MB", "0")
        assert get_max_upload_mb() == 1

    def test_max_concurrent_jobs_invalid_returns_default(
        self, monkeypatch, caplog,
    ) -> None:
        from picarones.interfaces.web.security import get_max_concurrent_jobs

        monkeypatch.setenv("PICARONES_MAX_CONCURRENT_JOBS", "abc")
        with caplog.at_level("WARNING"):
            value = get_max_concurrent_jobs()
        assert value == 2
        assert any(
            "PICARONES_MAX_CONCURRENT_JOBS" in rec.message
            for rec in caplog.records
        )

    def test_rate_limit_invalid_in_public_mode_returns_default(
        self, monkeypatch,
    ) -> None:
        from picarones.interfaces.web.security import get_rate_limit_per_hour

        monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
        monkeypatch.setenv("PICARONES_RATE_LIMIT_PER_HOUR", "not-int")
        assert get_rate_limit_per_hour() == 5

    def test_rate_limit_dev_mode_returns_zero(self, monkeypatch) -> None:
        """Hors mode public, pas de rate limit (0 = illimité)."""
        from picarones.interfaces.web.security import get_rate_limit_per_hour

        monkeypatch.delenv("PICARONES_PUBLIC_MODE", raising=False)
        assert get_rate_limit_per_hour() == 0


# ──────────────────────────────────────────────────────────────────────
# compute_workspace_roots avec env var explicite
# ──────────────────────────────────────────────────────────────────────


class TestComputeWorkspaceRoots:
    def test_env_var_overrides_defaults(self, monkeypatch, tmp_path) -> None:
        from picarones.interfaces.web.security import compute_workspace_roots

        d1 = tmp_path / "ws1"
        d2 = tmp_path / "ws2"
        d1.mkdir()
        d2.mkdir()
        monkeypatch.setenv(
            "PICARONES_WORKSPACE_ROOTS", f"{d1}{os.pathsep}{d2}",
        )
        roots = compute_workspace_roots(tmp_path / "uploads")
        # Les deux paths explicites doivent être présents et résolus.
        resolved = [r.resolve() for r in roots]
        assert d1.resolve() in resolved
        assert d2.resolve() in resolved

    def test_no_env_var_uses_defaults(self, monkeypatch, tmp_path) -> None:
        from picarones.interfaces.web.security import compute_workspace_roots

        monkeypatch.delenv("PICARONES_WORKSPACE_ROOTS", raising=False)
        uploads = tmp_path / "uploads"
        uploads.mkdir()
        roots = compute_workspace_roots(uploads)
        # Au moins ``uploads`` ou un parent doit être inclus.
        resolved = [r.resolve() for r in roots]
        assert any(
            uploads.resolve() == r or uploads.resolve().is_relative_to(r)
            for r in resolved
        )


# ──────────────────────────────────────────────────────────────────────
# validate_image_safe — branche DecompressionBombError
# ──────────────────────────────────────────────────────────────────────


def _tiny_png_bytes() -> bytes:
    """Produit un PNG 4×4 minimal (assez pour déclencher la bomb
    si ``MAX_IMAGE_PIXELS`` est abaissé à 1)."""
    from PIL import Image

    img = Image.new("RGB", (4, 4), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestValidateImageSafe:
    def test_decompression_bomb_rejected(self, monkeypatch) -> None:
        """Simule une bomb en abaissant ``MAX_IMAGE_PIXELS`` sous la
        taille de l'image — Pillow lève alors
        ``DecompressionBombError`` que le helper doit transformer
        en ``ValueError`` propre."""
        from PIL import Image
        from picarones.interfaces.web.security import validate_image_safe

        data = _tiny_png_bytes()
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 2)
        with pytest.raises(ValueError, match="bombe|décompression"):
            validate_image_safe(data, filename="bomb.png")

    def test_size_limit_enforced(self, monkeypatch) -> None:
        """Buffer trop gros → rejet sans tenter Pillow."""
        from picarones.interfaces.web.security import validate_image_safe

        monkeypatch.setenv("PICARONES_MAX_UPLOAD_MB", "1")
        data = b"\x00" * (2 * 1024 * 1024)  # 2 MB > 1 MB limit
        with pytest.raises(ValueError, match="taille"):
            validate_image_safe(data, filename="big.bin")

    def test_valid_image_passes(self) -> None:
        """Contrôle positif : image valide → aucune exception."""
        from picarones.interfaces.web.security import validate_image_safe

        validate_image_safe(_tiny_png_bytes(), filename="ok.png")  # no raise

    def test_corrupt_bytes_rejected(self) -> None:
        """Données non-image → ``ValueError`` (UnidentifiedImage ou
        autre)."""
        from picarones.interfaces.web.security import validate_image_safe

        with pytest.raises(ValueError):
            validate_image_safe(b"not-an-image-at-all", filename="nope.png")


# ──────────────────────────────────────────────────────────────────────
# _get_csrf_secret — fallback runtime
# ──────────────────────────────────────────────────────────────────────


class TestCSRFSecretRuntime:
    def test_env_var_used_when_set(self, monkeypatch) -> None:
        import picarones.interfaces.web.security as sec

        monkeypatch.setenv("PICARONES_CSRF_SECRET", "fixed-secret")
        # Reset le runtime secret pour s'assurer qu'on prend bien l'env.
        monkeypatch.setattr(sec, "_csrf_secret_runtime", None)
        secret = sec._get_csrf_secret()
        assert secret == b"fixed-secret"

    def test_runtime_generated_when_env_absent(
        self, monkeypatch, caplog,
    ) -> None:
        import picarones.interfaces.web.security as sec

        monkeypatch.delenv("PICARONES_CSRF_SECRET", raising=False)
        monkeypatch.setattr(sec, "_csrf_secret_runtime", None)
        with caplog.at_level("WARNING"):
            secret1 = sec._get_csrf_secret()
        assert isinstance(secret1, bytes)
        assert len(secret1) == 32, "secrets.token_bytes(32) attendu"
        # Warning émis pour signaler la config manquante.
        assert any(
            "PICARONES_CSRF_SECRET" in rec.message for rec in caplog.records
        )
        # Appel suivant → même secret (persistant durant la vie du process).
        secret2 = sec._get_csrf_secret()
        assert secret1 == secret2


# ──────────────────────────────────────────────────────────────────────
# RateLimiter.check — pruning de la fenêtre
# ──────────────────────────────────────────────────────────────────────


class TestRateLimiterPruning:
    def test_prunes_expired_hits(self) -> None:
        """Un hit > 1h → purgé du bucket à l'appel suivant.  Couvre
        la branche ``while bucket and bucket[0] < cutoff: popleft()``."""
        from collections import deque

        from picarones.interfaces.web.security import RateLimiter

        rl = RateLimiter(max_per_hour=2)
        # Pose un hit ancien (> 3600s) directement dans le bucket
        # interne pour simuler le passage du temps sans sleep.
        rl._buckets["1.2.3.4"] = deque([time.monotonic() - 7200.0])

        rl.check("1.2.3.4")  # ne doit pas lever
        # Le hit ancien est purgé, seul le nouveau reste.
        assert len(rl._buckets["1.2.3.4"]) == 1, (
            "le hit ancien aurait dû être purgé"
        )

    def test_quota_exceeded_raises(self) -> None:
        from picarones.interfaces.web.security import RateLimiter

        rl = RateLimiter(max_per_hour=2)
        rl.check("5.6.7.8")
        rl.check("5.6.7.8")
        with pytest.raises(PermissionError, match="Quota"):
            rl.check("5.6.7.8")

    def test_disabled_when_max_zero(self) -> None:
        """``max_per_hour=0`` → désactivé, jamais de PermissionError."""
        from picarones.interfaces.web.security import RateLimiter

        rl = RateLimiter(max_per_hour=0)
        for _ in range(100):
            rl.check("9.9.9.9")  # no raise


class TestEntityExtractorAllowlist:
    """Garde-fou P0 — le champ ``entity_extractor`` du payload web
    déclenche un ``importlib.import_module`` + appel du symbole résolu.
    C'est un gadget d'exécution : il doit être fail-closed sur instance
    partagée et explicitement allowlisté en institutionnel."""

    def test_empty_is_noop(self, monkeypatch) -> None:
        from picarones.interfaces.web.security import (
            assert_entity_extractor_allowed,
        )

        monkeypatch.delenv("PICARONES_PUBLIC_MODE", raising=False)
        monkeypatch.delenv(
            "PICARONES_ENTITY_EXTRACTOR_ALLOWLIST", raising=False,
        )
        assert_entity_extractor_allowed("")  # no raise
        assert_entity_extractor_allowed("   ")  # no raise

    def test_public_mode_without_allowlist_rejects(
        self, monkeypatch,
    ) -> None:
        from picarones.interfaces.web.security import (
            assert_entity_extractor_allowed,
        )

        monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
        monkeypatch.delenv(
            "PICARONES_ENTITY_EXTRACTOR_ALLOWLIST", raising=False,
        )
        with pytest.raises(PermissionError, match="entity_extractor"):
            assert_entity_extractor_allowed("os:getcwd")

    def test_non_public_without_allowlist_tolerated(
        self, monkeypatch,
    ) -> None:
        """Opérateur local de confiance — cohérent avec le modèle
        ``compute_browse_roots`` (cwd autorisé hors mode public)."""
        from picarones.interfaces.web.security import (
            assert_entity_extractor_allowed,
        )

        monkeypatch.delenv("PICARONES_PUBLIC_MODE", raising=False)
        monkeypatch.delenv(
            "PICARONES_ENTITY_EXTRACTOR_ALLOWLIST", raising=False,
        )
        assert_entity_extractor_allowed("mypkg.ner:Extractor")  # no raise

    def test_allowlist_match_allowed_all_modes(
        self, monkeypatch,
    ) -> None:
        from picarones.interfaces.web.security import (
            assert_entity_extractor_allowed,
        )

        monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
        monkeypatch.setenv(
            "PICARONES_ENTITY_EXTRACTOR_ALLOWLIST",
            "mypkg.ner:Extractor, other.mod:Fn",
        )
        assert_entity_extractor_allowed("mypkg.ner:Extractor")  # no raise
        assert_entity_extractor_allowed("other.mod:Fn")  # no raise

    def test_allowlist_set_rejects_unlisted_even_non_public(
        self, monkeypatch,
    ) -> None:
        """Allowlist définie ⇒ s'applique dans tous les modes : un
        dotted path hors liste est refusé même hors mode public."""
        from picarones.interfaces.web.security import (
            assert_entity_extractor_allowed,
        )

        monkeypatch.delenv("PICARONES_PUBLIC_MODE", raising=False)
        monkeypatch.setenv(
            "PICARONES_ENTITY_EXTRACTOR_ALLOWLIST", "mypkg.ner:Extractor",
        )
        with pytest.raises(PermissionError, match="hors allowlist"):
            assert_entity_extractor_allowed("os:system")


class TestSecureCookies:
    """``secure_cookies`` — fin du ``secure=False`` codé en dur."""

    def _clear(self, mp):
        for v in (
            "PICARONES_SECURE_COOKIES", "PICARONES_PUBLIC_MODE", "SPACE_ID",
        ):
            mp.delenv(v, raising=False)

    def test_explicit_env_wins(self, monkeypatch) -> None:
        from picarones.interfaces.web.security import secure_cookies

        self._clear(monkeypatch)
        monkeypatch.setenv("PICARONES_SECURE_COOKIES", "1")
        assert secure_cookies() is True
        monkeypatch.setenv("PICARONES_SECURE_COOKIES", "0")
        assert secure_cookies() is False

    def test_local_dev_default_false(self, monkeypatch) -> None:
        from picarones.interfaces.web.security import secure_cookies

        self._clear(monkeypatch)
        assert secure_cookies() is False

    def test_hf_space_default_true(self, monkeypatch) -> None:
        """HF Space = HTTPS certain ⇒ Secure par défaut."""
        from picarones.interfaces.web.security import secure_cookies

        self._clear(monkeypatch)
        monkeypatch.setenv("SPACE_ID", "user/space")
        assert secure_cookies() is True

    def test_public_mode_alone_does_not_imply_secure(
        self, monkeypatch,
    ) -> None:
        """Axe découplé : « mutualisé » n'implique pas « HTTPS ».
        Le compose local est public_mode=1 mais http://127.0.0.1 —
        un cookie Secure y serait silencieusement ignoré."""
        from picarones.interfaces.web.security import secure_cookies

        self._clear(monkeypatch)
        monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
        assert secure_cookies() is False


class TestDeploymentCoherence:
    """``check_deployment_coherence`` — refuse une combinaison
    exposée + CSRF requis + cookies en clair (fail-fast démarrage)."""

    def _clear(self, mp):
        for v in (
            "PICARONES_SECURE_COOKIES", "PICARONES_PUBLIC_MODE",
            "PICARONES_CSRF_REQUIRED", "SPACE_ID",
        ):
            mp.delenv(v, raising=False)

    def test_local_csrf_without_secure_is_tolerated(
        self, monkeypatch,
    ) -> None:
        from picarones.interfaces.web.security import (
            check_deployment_coherence,
        )

        self._clear(monkeypatch)
        monkeypatch.setenv("PICARONES_CSRF_REQUIRED", "1")
        check_deployment_coherence()  # non exposé → pas de raise

    def test_exposed_csrf_without_secure_cookies_blocks_startup(
        self, monkeypatch,
    ) -> None:
        from picarones.interfaces.web.security import (
            check_deployment_coherence,
        )

        self._clear(monkeypatch)
        monkeypatch.setenv("PICARONES_CSRF_REQUIRED", "1")
        monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
        monkeypatch.setenv("PICARONES_SECURE_COOKIES", "0")
        with pytest.raises(RuntimeError, match="incohérente"):
            check_deployment_coherence()

    def test_exposed_csrf_with_secure_cookies_ok(
        self, monkeypatch,
    ) -> None:
        from picarones.interfaces.web.security import (
            check_deployment_coherence,
        )

        self._clear(monkeypatch)
        monkeypatch.setenv("PICARONES_CSRF_REQUIRED", "1")
        monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
        monkeypatch.setenv("PICARONES_SECURE_COOKIES", "1")
        check_deployment_coherence()  # cohérent → pas de raise
