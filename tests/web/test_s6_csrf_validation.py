"""Sprint S6.9 — refus de démarrage si CSRF_REQUIRED sans secret stable.

En mode institutionnel, ``PICARONES_CSRF_REQUIRED=1`` doit être
accompagné de ``PICARONES_CSRF_SECRET`` (généré par l'opérateur,
stable entre redémarrages).  Sans secret stable, les tokens CSRF
sont régénérés à chaque restart du process — tous les tokens
émis avant deviennent invalides → tous les utilisateurs voient
des 403 jusqu'à ce qu'ils rechargent leur page.

Ce comportement est silencieux dans la version pré-S6.9 (juste un
log warning).  Pour l'institutionnel, on veut un échec **dur** au
démarrage.
"""

from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────────────
# 1. Mode public (sans CSRF_REQUIRED) : pas de validation
# ──────────────────────────────────────────────────────────────────────


class TestPublicModeNoValidation:
    def test_no_secret_required_in_public_mode(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from picarones.interfaces.web.security import validate_csrf_config

        monkeypatch.delenv("PICARONES_CSRF_REQUIRED", raising=False)
        monkeypatch.delenv("PICARONES_CSRF_SECRET", raising=False)

        # Ne lève pas — mode public, pas de CSRF.
        validate_csrf_config()


# ──────────────────────────────────────────────────────────────────────
# 2. Mode institutionnel : refus si secret manquant
# ──────────────────────────────────────────────────────────────────────


class TestInstitutionalModeRequiresSecret:
    def test_missing_secret_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from picarones.interfaces.web.security import (
            CSRFConfigError, validate_csrf_config,
        )

        monkeypatch.setenv("PICARONES_CSRF_REQUIRED", "1")
        monkeypatch.delenv("PICARONES_CSRF_SECRET", raising=False)

        with pytest.raises(CSRFConfigError, match="PICARONES_CSRF_SECRET"):
            validate_csrf_config()

    def test_empty_secret_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from picarones.interfaces.web.security import (
            CSRFConfigError, validate_csrf_config,
        )

        monkeypatch.setenv("PICARONES_CSRF_REQUIRED", "1")
        monkeypatch.setenv("PICARONES_CSRF_SECRET", "")

        with pytest.raises(CSRFConfigError):
            validate_csrf_config()

    def test_whitespace_only_secret_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from picarones.interfaces.web.security import (
            CSRFConfigError, validate_csrf_config,
        )

        monkeypatch.setenv("PICARONES_CSRF_REQUIRED", "1")
        monkeypatch.setenv("PICARONES_CSRF_SECRET", "   ")

        with pytest.raises(CSRFConfigError):
            validate_csrf_config()

    def test_valid_secret_passes(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from picarones.interfaces.web.security import validate_csrf_config

        monkeypatch.setenv("PICARONES_CSRF_REQUIRED", "1")
        monkeypatch.setenv(
            "PICARONES_CSRF_SECRET",
            # 64 hex chars = 32 bytes (équivalent ``openssl rand -hex 32``)
            "a" * 64,
        )

        validate_csrf_config()  # ne lève pas


# ──────────────────────────────────────────────────────────────────────
# 3. Refus des secrets trivialement faibles
# ──────────────────────────────────────────────────────────────────────


class TestWeakSecretRejected:
    @pytest.mark.parametrize(
        "weak", ["changeme", "secret", "password", "test", "dev",
                 "CHANGEME", "Secret", "ChangeMe"],
    )
    def test_weak_secret_raises(
        self, monkeypatch: pytest.MonkeyPatch, weak: str,
    ) -> None:
        from picarones.interfaces.web.security import (
            CSRFConfigError, validate_csrf_config,
        )

        monkeypatch.setenv("PICARONES_CSRF_REQUIRED", "1")
        monkeypatch.setenv("PICARONES_CSRF_SECRET", weak)

        with pytest.raises(CSRFConfigError, match="trivialement faible"):
            validate_csrf_config()


# ──────────────────────────────────────────────────────────────────────
# 4. Validation appelée au lifespan FastAPI
# ──────────────────────────────────────────────────────────────────────


class TestLifespanCallsValidation:
    """Le startup hook FastAPI doit appeler ``validate_csrf_config``.
    Sans ce câblage, on peut activer ``PICARONES_CSRF_REQUIRED`` sans
    secret et l'app démarre quand même."""

    def test_app_refuses_to_start_without_secret(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fastapi.testclient import TestClient

        monkeypatch.setenv("PICARONES_CSRF_REQUIRED", "1")
        monkeypatch.delenv("PICARONES_CSRF_SECRET", raising=False)

        # Re-import pour s'assurer qu'on a une instance fraîche.
        # Le lifespan est appelé à l'entrée du context manager du
        # TestClient.
        from picarones.interfaces.web.app import app
        from picarones.interfaces.web.security import CSRFConfigError

        with pytest.raises(CSRFConfigError):
            with TestClient(app):
                pass  # Le startup hook lève avant d'arriver ici.
