"""Sprint S8.7 — couverture finale des petits fichiers pour
faire passer Codecov patch coverage > 95%.

Cibles :
- ``_workflows._validate_cer_threshold`` (CLI callback validation).
- ``module_policy._is_module_subclass`` AttributeError fallback +
  introspection inputs/outputs failure.
- ``history`` router : ``query`` et ``detect_regression`` qui
  lèvent → warning + continue dégradé.
- ``security.validate_image_safe`` branche ``except Exception``
  générique (lignes 239-242) sur erreur Pillow hétérogène.
"""

from __future__ import annotations

import click
import pytest


# ──────────────────────────────────────────────────────────────────────
# CLI _validate_cer_threshold — validation callback
# ──────────────────────────────────────────────────────────────────────


class TestValidateCERThresholdCallback:
    """``--fail-if-cer-above`` doit être en fraction ∈ [0, 1].
    Avant le fix, l'ancienne sémantique acceptait des pourcentages
    (15.0 = 15 %) ; on échoue maintenant bruyamment sur les
    valeurs > 1 pour empêcher la mauvaise interprétation."""

    def _run_callback(self, value):
        from picarones.interfaces.cli._workflows import (
            _validate_cer_threshold,
        )

        return _validate_cer_threshold(ctx=None, param=None, value=value)

    def test_none_passes_through(self) -> None:
        assert self._run_callback(None) is None

    def test_valid_fraction_returned(self) -> None:
        assert self._run_callback(0.15) == 0.15

    def test_zero_accepted(self) -> None:
        assert self._run_callback(0.0) == 0.0

    def test_one_accepted_at_boundary(self) -> None:
        assert self._run_callback(1.0) == 1.0

    def test_negative_value_rejected(self) -> None:
        with pytest.raises(click.BadParameter, match=">= 0|≥ 0"):
            self._run_callback(-0.1)

    def test_legacy_percent_value_rejected(self) -> None:
        """Valeur > 1 (ancienne sémantique pourcentage) doit lever
        avec un message qui explique la migration."""
        with pytest.raises(click.BadParameter) as exc_info:
            self._run_callback(15.0)
        msg = str(exc_info.value)
        assert "fraction" in msg
        assert "15.0" in msg or "0.15" in msg


# ──────────────────────────────────────────────────────────────────────
# module_policy — defensive paths
# ──────────────────────────────────────────────────────────────────────


class TestModulePolicyDefensive:
    def test_is_base_module_no_mro_returns_false(self) -> None:
        """Couvre lignes 220-221 — un objet sans ``__mro__``
        accessible doit retourner False sans planter."""
        from picarones.evaluation.metrics.module_policy import (
            _is_base_module,
        )

        class TrulyBroken:
            @property
            def __mro__(self):  # type: ignore[override]
                raise AttributeError("simulated absent __mro__")

        # ``_is_base_module`` accède à ``cls.__mro__`` directement —
        # on doit lui passer une instance dont l'accès lève.
        result = _is_base_module(TrulyBroken())
        assert result is False

    def test_audit_module_introspection_failure_falls_back(
        self, caplog,
    ) -> None:
        """Couvre lignes 284-292 — si l'accès à
        ``output_types`` lève (manifest custom property qui plante),
        ``audit_module`` retombe sur listes vides + log debug."""
        from picarones.evaluation.metrics.module_policy import (
            ModuleManifest, audit_module,
        )

        class BadManifestModule:
            input_types = "this-should-be-iterable-but-isnt-an-iterable-of-types"

            @property
            def output_types(self):
                # ``getattr(cls, "output_types", None)`` côté audit
                # accède au descriptor → property.__get__ avec cls=None
                # ne lève pas, mais l'itération ``for t in attr_out``
                # plus tard plantera (str pas itérable de types).
                raise RuntimeError("manifest cassé simulé")

        manifest = ModuleManifest(
            name="bad", version="1.0", author="t", license="MIT",
            description="test bad",
            input_types=[], output_types=[],
        )

        with caplog.at_level("DEBUG"):
            result = audit_module(BadManifestModule, manifest)
        # ``audit_module`` retourne un ``AuditResult`` même avec un
        # manifest cassé — c'est tout l'intérêt de la défense.
        assert result is not None


# ──────────────────────────────────────────────────────────────────────
# history router — dégradation gracieuse
# ──────────────────────────────────────────────────────────────────────


class TestHistoryRouterDegraded:
    def _app(self):
        from fastapi import FastAPI

        from picarones.interfaces.web.routers import history as h

        app = FastAPI()
        app.include_router(h.router)
        return app

    def test_query_failure_returns_empty_targets_with_warning(
        self, monkeypatch, caplog,
    ) -> None:
        """Quand ``BenchmarkHistory.query`` lève (DB corrompue,
        schéma migré), on log un warning et on retourne une liste
        vide de régressions plutôt que de planter en 500.  Couvre
        lignes 52-56."""
        from fastapi.testclient import TestClient

        from picarones.evaluation.metrics import history as eval_history

        # Mock BenchmarkHistory.query pour lever.
        def raising_query(*args, **kwargs):
            raise RuntimeError("DB schema mismatch simulé")

        monkeypatch.setattr(
            eval_history.BenchmarkHistory, "query", raising_query,
        )

        app = self._app()
        with caplog.at_level("WARNING"):
            with TestClient(app) as client:
                r = client.get("/api/history/regressions")
        assert r.status_code == 200, r.text
        # Sans moteur explicite + query qui plante → liste vide.
        assert r.json()["regressions"] == []
        # Warning émis.
        assert any(
            "énumération" in rec.message.lower() or "moteurs" in rec.message.lower()
            for rec in caplog.records
        )

    def test_detect_regression_failure_continues_to_next_engine(
        self, monkeypatch, caplog,
    ) -> None:
        """Quand ``detect_regression`` lève pour un moteur, on log
        un warning et on continue avec les suivants.  Couvre
        lignes 62-66."""
        from fastapi.testclient import TestClient

        from picarones.evaluation.metrics import history as eval_history

        def raising_detect(self, *, engine, threshold):
            raise RuntimeError(f"detect_regression KO pour {engine}")

        monkeypatch.setattr(
            eval_history.BenchmarkHistory,
            "detect_regression",
            raising_detect,
        )

        app = self._app()
        with caplog.at_level("WARNING"):
            with TestClient(app) as client:
                r = client.get(
                    "/api/history/regressions",
                    params={"engine": "tesseract"},
                )
        assert r.status_code == 200, r.text
        # detect a planté → pas de résultat dans la liste.
        assert r.json()["regressions"] == []
        assert any(
            "detect_regression" in rec.message or "tesseract" in rec.message
            for rec in caplog.records
        )


# ──────────────────────────────────────────────────────────────────────
# security.validate_image_safe — branche Exception générique
# ──────────────────────────────────────────────────────────────────────


class TestValidateImageGenericException:
    """Pillow lève un panel d'exceptions hétérogènes (SyntaxError
    sur GIF malformé, OSError sur TIFF corrompu, AttributeError
    interne, etc.) — toutes doivent être transformées en
    ``ValueError`` propre via la branche ``except Exception``.
    Couvre lignes 239-242."""

    def test_generic_pillow_failure_wrapped_in_value_error(
        self, monkeypatch,
    ) -> None:
        from PIL import Image

        from picarones.interfaces.web.security import validate_image_safe

        # Mock Image.open pour retourner un objet dont ``verify()``
        # lève une OSError (typique TIFF corrompu).
        class FakeImg:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def verify(self):
                raise OSError("simulated corrupt TIFF")

        monkeypatch.setattr(Image, "open", lambda *args, **kwargs: FakeImg())

        with pytest.raises(ValueError, match="OSError|erreur de décodage"):
            validate_image_safe(b"any-bytes", filename="corrupt.tiff")
