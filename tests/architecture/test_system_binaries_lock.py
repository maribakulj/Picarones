"""Sprint S8.5 — capture des binaires système dans le RunManifest.

Ferme le trou de reproductibilité laissé par
``capture_dependencies_lock`` qui ne couvre que les paquets Python.
La version du binaire Tesseract (qui exécute réellement l'OCR) n'est
pas dans le wheel ``pytesseract`` et doit être capturée séparément.

Sans cette capture, deux runs avec le même ``dependencies_lock``
peuvent produire des CER différents si la version Tesseract change
entre temps (ex : Debian point-release).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# 1. capture_system_binaries_lock — best-effort
# ──────────────────────────────────────────────────────────────────────


class TestCaptureSystemBinariesLock:
    def test_returns_dict(self) -> None:
        from picarones.app.services.dependencies import (
            capture_system_binaries_lock,
        )
        lock = capture_system_binaries_lock()
        assert isinstance(lock, dict)

    def test_includes_tesseract_when_installed(self) -> None:
        """Si ``tesseract`` est dans ``$PATH``, sa version doit être
        capturée."""
        from picarones.app.services.dependencies import (
            capture_system_binaries_lock,
        )
        import shutil

        if not shutil.which("tesseract"):
            pytest.skip("tesseract non installé sur ce système")

        lock = capture_system_binaries_lock()
        assert "tesseract" in lock
        assert "tesseract" in lock["tesseract"].lower()  # ex : "tesseract 5.3.0"

    def test_missing_binary_silently_omitted(self) -> None:
        """Si un binaire n'est pas dans ``$PATH``, sa clé est absente
        du dict (pas ``None``, pas d'exception)."""
        from picarones.app.services.dependencies import (
            _safe_capture_binary_version,
        )
        result = _safe_capture_binary_version("definitely_not_a_real_binary_xyz")
        assert result is None

    def test_safe_capture_handles_subprocess_error(self) -> None:
        """``subprocess.run`` qui timeout ou crash → ``None``, pas
        de propagation."""
        from picarones.app.services.dependencies import (
            _safe_capture_binary_version,
        )

        # Mock pour simuler un binaire qui timeout.
        import subprocess
        with patch("shutil.which", return_value="/fake/path"):
            with patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired("fake", 5),
            ):
                result = _safe_capture_binary_version("fake")
                assert result is None


# ──────────────────────────────────────────────────────────────────────
# 2. RunManifest accepte system_binaries_lock
# ──────────────────────────────────────────────────────────────────────


class TestRunManifestField:
    def test_default_empty_dict(self) -> None:
        """Manifests pré-S8.5 sans le champ doivent rester
        désérialisables."""
        from datetime import datetime, timezone

        from picarones.domain.run_manifest import RunManifest

        m = RunManifest(
            run_id="r",
            corpus_name="c",
            n_documents=0,
            code_version="1.0.0",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        assert m.system_binaries_lock == {}

    def test_field_persisted_in_serialization(self) -> None:
        """Le champ apparaît dans le dump JSON pour les ingesters
        externes (BnF audit)."""
        from datetime import datetime, timezone

        from picarones.domain.run_manifest import RunManifest

        m = RunManifest(
            run_id="r",
            corpus_name="c",
            n_documents=0,
            code_version="1.0.0",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            system_binaries_lock={"tesseract": "tesseract 5.3.0"},
        )
        dumped = m.model_dump()
        assert "system_binaries_lock" in dumped
        assert dumped["system_binaries_lock"]["tesseract"] == "tesseract 5.3.0"
