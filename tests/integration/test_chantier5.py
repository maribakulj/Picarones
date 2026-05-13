"""Tests du chantier 5 (post-Sprint 97) — découpage des monolithes.

Couvre :

- 5.A : :mod:`picarones.reports.narrative.detectors` est désormais un
  package thématique de 6 sous-modules (1229 lignes → 6 fichiers).
  Tous les imports historiques restent accessibles.
- 5.B : :mod:`picarones.cli` est désormais un package avec 6
  sous-modules + ``__init__.py`` (1519 lignes → 7 fichiers).
  Le groupe ``cli`` reste exporté pour l'entry-point ``pyproject.toml``.
"""

from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────────────────
# 5.A — narrative/detectors décomposé en 6 familles
# ──────────────────────────────────────────────────────────────────────────


class TestDetectorsPackage:
    def test_detectors_is_now_a_package(self):
        """``detectors.py`` est devenu ``detectors/`` (package)."""
        from picarones.reports.narrative import detectors
        # Un package a __path__, un module simple ne l'a pas
        assert hasattr(detectors, "__path__"), (
            "detectors devrait être un package depuis le chantier 5"
        )

    @pytest.mark.parametrize("name", [
        "detect_global_leader_cer",
        "detect_statistical_tie",
        "detect_significant_gap",
        "detect_pareto_alternative",
        "detect_stratum_winner",
        "detect_stratum_collapse",
        "detect_error_profile_outlier",
        "detect_llm_hallucination_flag",
        "detect_robustness_fragile",
        "detect_cost_outlier",
        "detect_speed_winner",
        "detect_confidence_warning",
        "detect_median_mean_gap_warning",
        "detect_stratification_recommended",
        "detect_engine_off_baseline",
        "detect_engine_unstable",
        "detect_regression_in_history",
        "detect_ensemble_opportunity",
        # Sprint A3 — détecteur d'incidents d'importer en mode dégradé.
        "detect_importer_fallback",
        # Sprint A8 — détecteur de pricing périmé (item m-14).
        "detect_pricing_staleness",
    ])
    def test_all_20_detectors_importable_from_root(self, name):
        """Rétrocompat : les 20 détecteurs s'importent depuis le package
        (18 historiques + Sprint A3 + Sprint A8)."""
        from picarones.reports.narrative import detectors
        assert hasattr(detectors, name), f"{name} disparu après chantier 5"
        assert callable(getattr(detectors, name))

    def test_DETECTORS_BY_TYPE_still_exposed(self):
        from picarones.reports.narrative.detectors import DETECTORS_BY_TYPE
        assert isinstance(DETECTORS_BY_TYPE, dict)
        # Sprint A3 → 19 (IMPORTER_FALLBACK_TRIGGERED).
        # Sprint A8 → 20 (PRICING_STALENESS_WARNING).
        assert len(DETECTORS_BY_TYPE) == 20, (
            f"DETECTORS_BY_TYPE doit contenir 20 entrées, en a {len(DETECTORS_BY_TYPE)}"
        )

    def test_register_default_detectors_still_callable(self):
        from picarones.reports.narrative.detectors import register_default_detectors
        assert callable(register_default_detectors)

    @pytest.mark.parametrize("submodule, detector_count", [
        ("ranking", 5),
        # Sprint A8 — pareto passe de 2 à 3 (ajout detect_pricing_staleness).
        ("pareto", 3),
        ("stratum", 3),
        ("quality", 4),
        # Sprint A3 — history passe de 3 à 4 (ajout detect_importer_fallback).
        ("history", 4),
        ("ensemble", 1),
    ])
    def test_submodules_have_expected_detector_count(self, submodule, detector_count):
        """Chaque sous-module thématique a le bon nombre de détecteurs."""
        import importlib

        mod = importlib.import_module(
            f"picarones.reports.narrative.detectors.{submodule}"
        )
        detectors_in_sub = [
            n for n in dir(mod)
            if n.startswith("detect_") and callable(getattr(mod, n))
        ]
        assert len(detectors_in_sub) == detector_count, (
            f"{submodule} : {len(detectors_in_sub)} détecteurs trouvés, "
            f"{detector_count} attendus — {detectors_in_sub}"
        )

    def test_identity_through_submodule_and_root(self):
        """Le détecteur exposé depuis __init__.py et depuis son sous-module
        est la même fonction (pas de redéfinition)."""
        from picarones.reports.narrative.detectors import detect_global_leader_cer
        from picarones.reports.narrative.detectors.ranking import (
            detect_global_leader_cer as via_submodule,
        )
        assert detect_global_leader_cer is via_submodule

    def test_detector_smoke_via_root(self):
        """Smoke test : un détecteur fonctionne via l'import root."""
        from picarones.reports.narrative.detectors import detect_global_leader_cer
        result = detect_global_leader_cer({
            "ranking": [
                {"engine": "tess", "mean_cer": 0.05},
                {"engine": "pero", "mean_cer": 0.07},
            ],
        })
        assert len(result) == 1
        assert result[0].payload["engine"] == "tess"

    def test_helpers_are_in_dedicated_module(self):
        """Les helpers internes (_engines_summary, etc.) vivent dans
        ``_helpers.py`` (pattern modulaire propre)."""
        from picarones.reports.narrative.detectors import _helpers
        assert hasattr(_helpers, "_engines_summary")
        assert hasattr(_helpers, "_engine_by_name")
        assert hasattr(_helpers, "_n_docs")


# ──────────────────────────────────────────────────────────────────────────
# 5.B — cli.py décomposé en package
# ──────────────────────────────────────────────────────────────────────────


class TestCliPackage:
    # Phase 4.4 audit code-quality (2026-05) — les 5 ``try/except
    # ImportError → pytest.skip("click non installé")`` étaient des
    # zombies vacuement vrais : ``click`` est dep obligatoire
    # (``pyproject.toml`` ``click>=8.1.0,<9.0``).  Skip = jamais
    # exécuté.  Remplacés par des imports directs qui plantent
    # franchement si l'environnement est cassé.

    def test_cli_is_now_a_package(self):
        import picarones.interfaces.cli as cli_pkg

        assert hasattr(cli_pkg, "__path__"), (
            "picarones.cli devrait être un package depuis le chantier 5"
        )

    def test_cli_group_still_exported(self):
        """L'entry-point ``picarones.cli:cli`` (pyproject.toml) doit
        rester valide après le chantier 5."""
        from picarones.interfaces.cli import cli

        assert cli is not None

    def test_helpers_still_exported(self):
        """``_setup_logging`` et ``_engine_from_name`` restent accessibles
        depuis ``picarones.cli`` (les sous-modules les utilisent)."""
        import picarones.interfaces.cli as cli_pkg

        assert callable(cli_pkg._setup_logging)
        assert callable(cli_pkg._engine_from_name)

    @pytest.mark.parametrize("submodule", [
        "_workflows",
        "_imports",
        "_serve",
        "_history",
        "_robustness",
    ])
    def test_submodule_loaded(self, submodule):
        import picarones.interfaces.cli as cli_pkg

        assert hasattr(cli_pkg, submodule), (
            f"{submodule} non chargé en cascade — les commandes de cette "
            "famille ne seraient pas enregistrées"
        )

    @pytest.mark.parametrize("cmd_name", [
        "run", "diagnose", "economics", "edition", "compare",
        "metrics", "engines", "info", "report", "demo",
        "serve", "history", "robustness", "import",
    ])
    def test_all_15_commands_registered(self, cmd_name):
        """Les commandes/groupes historiques doivent être enregistrés
        sur le groupe ``cli`` après l'import en cascade.

        Phase 7.D : la commande ``pipeline`` (groupe ``run``/``compare``)
        a été retirée — elle exposait le runner legacy ``PipelineRunner``
        désormais supprimé.  Le compteur historique passe de 15 à 14.
        """
        from picarones.interfaces.cli import cli
        assert hasattr(cli, "commands"), (
            "le groupe cli devrait avoir un attribut commands (Click Group)"
        )
        assert cmd_name in cli.commands, (
            f"commande '{cmd_name}' manquante après le chantier 5 — "
            f"commandes présentes : {sorted(cli.commands.keys())}"
        )
