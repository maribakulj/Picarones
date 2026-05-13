"""Phase 4.2 audit code-quality — ``POST /api/benchmark/start`` est
définitivement retiré en v2.0.

Le retrait est une **rupture API documentée** dans CHANGELOG : les
clients qui pointaient encore sur l'endpoint legacy reçoivent 404
(au lieu d'un 200 silencieux puis un warning loggé côté serveur).

Les helpers internes associés sont également supprimés :

- ``picarones.interfaces.web.benchmark_utils._legacy_request_to_run_request``
- ``picarones.interfaces.web.benchmark_utils.run_benchmark_thread``
  (v1 ; v2 ``run_benchmark_thread_v2`` est l'unique worker)
- ``picarones.interfaces.web.models.BenchmarkRequest``

Garantit que le nettoyage de la suite ne ressuscite pas accidentellement
le contrat legacy via un re-export.
"""

from __future__ import annotations

import pytest


def _build_client(tmp_path):
    """Construit un client FastAPI pour tester les endpoints."""
    from fastapi.testclient import TestClient

    from picarones.interfaces.web.app import app

    return TestClient(app)


class TestLegacyEndpointRemoved:
    def test_post_benchmark_start_returns_404(self, tmp_path) -> None:
        """``POST /api/benchmark/start`` doit retourner 404 (route inexistante).

        Régression : si quelqu'un ré-ajoute l'endpoint en pensant
        rétablir la rétrocompat, ce test plante.
        """
        client = _build_client(tmp_path)
        resp = client.post(
            "/api/benchmark/start",
            json={"corpus_path": str(tmp_path), "engines": ["tesseract"]},
        )
        assert resp.status_code == 404, (
            f"Endpoint legacy ressuscité — status={resp.status_code}, "
            f"body={resp.text[:200]!r}.  Phase 4.2 audit code-quality "
            f"avait acté la rupture v2.0."
        )

    def test_benchmark_run_still_works(self, tmp_path) -> None:
        """L'endpoint canonique ``POST /api/benchmark/run`` reste opérationnel.

        Vérifie que la rupture v2.0 n'a pas cassé l'endpoint vivant —
        on accepte une 400 (corpus inexistant) ou 200 (job lancé) ;
        le seul échec inadmissible est un 404 (endpoint manquant).
        """
        client = _build_client(tmp_path)
        resp = client.post(
            "/api/benchmark/run",
            json={
                "corpus_path": "/nonexistent/path/xyz",
                "competitors": [
                    {"name": "tesseract", "engine_name": "tesseract"},
                ],
            },
        )
        assert resp.status_code != 404, (
            f"Endpoint /api/benchmark/run absent — body={resp.text[:200]!r}"
        )


class TestLegacyHelpersRemoved:
    def test_run_benchmark_thread_v1_not_importable(self) -> None:
        """L'ancien worker ``run_benchmark_thread`` (v1) ne doit plus
        être importable depuis ``benchmark_utils``.  Seul ``run_benchmark_thread_v2``
        subsiste."""
        from picarones.interfaces.web import benchmark_utils

        assert not hasattr(benchmark_utils, "run_benchmark_thread"), (
            "``run_benchmark_thread`` (v1) a été ressuscité — Phase 4.2 "
            "audit code-quality avait acté la suppression au profit "
            "exclusif de ``run_benchmark_thread_v2``."
        )
        assert hasattr(benchmark_utils, "run_benchmark_thread_v2"), (
            "``run_benchmark_thread_v2`` manquant — régression majeure."
        )

    def test_legacy_request_converter_removed(self) -> None:
        """``_legacy_request_to_run_request`` était le pont entre
        ``BenchmarkRequest`` legacy et ``BenchmarkRunRequest`` v2.
        Plus utile depuis le retrait de ``BenchmarkRequest``."""
        from picarones.interfaces.web import benchmark_utils

        assert not hasattr(benchmark_utils, "_legacy_request_to_run_request")

    def test_benchmark_request_model_removed(self) -> None:
        """``BenchmarkRequest`` (modèle Pydantic v1) ne doit plus
        être importable — les clients utilisent ``BenchmarkRunRequest``."""
        from picarones.interfaces.web import models

        assert not hasattr(models, "BenchmarkRequest")
        assert "BenchmarkRequest" not in getattr(models, "__all__", [])
        # Le modèle canonique reste exposé.
        assert hasattr(models, "BenchmarkRunRequest")


@pytest.mark.parametrize(
    "symbol",
    [
        "run_benchmark_thread",
        "_legacy_request_to_run_request",
    ],
)
def test_symbol_not_in_codebase(symbol: str) -> None:
    """Garde-fou complémentaire : aucune occurrence du symbole legacy
    dans ``picarones/`` (autre que dans les commentaires/CHANGELOG).

    Si un futur PR redéfinit ``run_benchmark_thread`` sous un autre
    chemin, ce test attire l'attention du reviewer.
    """
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    # Pattern strict : ``def {symbol}(`` ou ``{symbol} =`` au début
    # de ligne.  Évite de matcher ``run_benchmark_thread_v2`` quand
    # on cherche ``run_benchmark_thread``.
    out = subprocess.run(
        [
            "grep", "-rEn",
            rf"^(def {symbol}\(|{symbol} =)",
            "picarones/",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 1, (
        f"Symbole legacy ``{symbol}`` réapparu dans picarones/ :\n{out.stdout}"
    )
