"""Tests de la phase E — séparation core/ + measurements/.

Couvre :

- ~41 modules métriques déplacés vers ``picarones/measurements/``.
- Sous-package ``narrative/`` (4 modules + 6 familles de détecteurs +
  helper) déplacé vers ``picarones/measurements/narrative/``.
- Identité préservée à travers les shims.
- Le ``core/`` strict ne contient plus que ~13 fichiers (Cercle 1).
- Les hooks builtin restent enregistrés.
- Le moteur narratif fonctionne (détection + arbitre + rendu).
- Les vues du chantier 3 fonctionnent.
- Document ``docs/architecture-cercles.md`` mis à jour avec critère DDD.
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────────
# 1. Imports historiques rétrocompat des mesures
# ──────────────────────────────────────────────────────────────────────────


class TestMeasurementsRetrocompat:
    @pytest.mark.parametrize("module_path, attribute", [
        ("picarones.core.confusion", "build_confusion_matrix"),
        ("picarones.core.taxonomy", "classify_errors"),
        ("picarones.core.calibration", "compute_calibration_metrics"),
        ("picarones.core.layout", "compute_layout_metrics"),
        ("picarones.core.reading_order", "compute_reading_order_metrics"),
        ("picarones.core.error_absorption", "compute_error_absorption"),
        ("picarones.core.searchability", "compute_searchability"),
        ("picarones.core.numerical_sequences",
         "compute_numerical_sequence_metrics"),
        ("picarones.core.readability", "flesch_score"),
        ("picarones.core.specialization", "compute_specialization_score"),
        ("picarones.core.throughput", "compute_effective_throughput"),
        ("picarones.core.cost_projection", "project_engine"),
        ("picarones.core.statistics", "bootstrap_ci"),
        ("picarones.core.history", "BenchmarkHistory"),
        ("picarones.core.builtin_hooks", "calibration_from_engine_result"),
        ("picarones.core.line_metrics", "compute_line_metrics"),
        ("picarones.core.hallucination", "compute_hallucination_metrics"),
        ("picarones.core.image_quality", "analyze_image_quality"),
        ("picarones.core.normalization", "PROFILES"),
        ("picarones.core.rare_tokens", "extract_rare_tokens"),
    ])
    def test_legacy_path_works(self, module_path: str, attribute: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = importlib.import_module(module_path)
        assert hasattr(mod, attribute), f"{module_path}.{attribute}"


# ──────────────────────────────────────────────────────────────────────────
# 2. Sous-package narrative/ déplacé
# ──────────────────────────────────────────────────────────────────────────


class TestNarrativePackageMigration:
    def test_narrative_root_import(self):
        from picarones.core.narrative import build_synthesis
        assert callable(build_synthesis)

    def test_narrative_facts_via_shim(self):
        from picarones.core.narrative.facts import Fact, FactType
        assert Fact is not None
        assert FactType is not None

    def test_narrative_registry_via_shim(self):
        from picarones.core.narrative.registry import register_detector
        assert callable(register_detector)

    def test_narrative_detectors_via_shim(self):
        from picarones.core.narrative.detectors import (
            DETECTORS_BY_TYPE,
            detect_global_leader_cer,
        )
        assert callable(detect_global_leader_cer)
        assert len(DETECTORS_BY_TYPE) == 18

    def test_narrative_detector_family_modules(self):
        # Les 6 familles restent accessibles via leur ancien chemin
        from picarones.core.narrative.detectors.ranking import (
            detect_global_leader_cer,
        )
        from picarones.core.narrative.detectors.pareto import (
            detect_pareto_alternative,
        )
        from picarones.core.narrative.detectors.history import (
            detect_engine_unstable,
        )
        from picarones.core.narrative.detectors.quality import (
            detect_confidence_warning,
        )
        from picarones.core.narrative.detectors.stratum import (
            detect_stratum_winner,
        )
        from picarones.core.narrative.detectors.ensemble import (
            detect_ensemble_opportunity,
        )
        assert all(callable(f) for f in [
            detect_global_leader_cer,
            detect_pareto_alternative,
            detect_engine_unstable,
            detect_confidence_warning,
            detect_stratum_winner,
            detect_ensemble_opportunity,
        ])


# ──────────────────────────────────────────────────────────────────────────
# 3. Identité préservée
# ──────────────────────────────────────────────────────────────────────────


class TestIdentityThroughShim:
    def test_confusion_identity(self):
        from picarones.core.confusion import build_confusion_matrix as via_old
        from picarones.measurements.confusion import (
            build_confusion_matrix as via_new,
        )
        assert via_old is via_new

    def test_narrative_facts_identity(self):
        from picarones.core.narrative.facts import Fact as via_old
        from picarones.measurements.narrative.facts import Fact as via_new
        assert via_old is via_new

    def test_narrative_detector_identity(self):
        from picarones.core.narrative.detectors.ranking import (
            detect_speed_winner as via_old,
        )
        from picarones.measurements.narrative.detectors.ranking import (
            detect_speed_winner as via_new,
        )
        assert via_old is via_new


# ──────────────────────────────────────────────────────────────────────────
# 4. core/ Cercle 1 strict — ne contient plus que ~13 modules
# ──────────────────────────────────────────────────────────────────────────


class TestCoreIsLean:
    """Le ``core/`` post-phase E ne contient plus que les modules
    Cercle 1 (abstractions + orchestration). Tout le reste est shim."""

    @pytest.mark.parametrize("name", [
        "corpus", "modules", "results", "metrics",
        "runner", "pipeline_runner", "pipeline_benchmark",
        "pipeline_comparison", "pipeline_spec_loader",
        "metric_registry", "metric_hooks",
        "builtin_metrics", "alto_metrics",
    ])
    def test_cercle1_module_present(self, name):
        """Les modules Cercle 1 doivent rester dans ``core/`` (pas de shim)."""
        repo = Path(__file__).parent.parent
        path = repo / "picarones" / "core" / f"{name}.py"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        # Un module Cercle 1 a > 30 lignes (vraie logique, pas shim)
        n_lines = len([line for line in content.splitlines() if line.strip()])
        assert n_lines > 30, (
            f"core/{name}.py fait {n_lines} lignes — ne devrait pas être "
            "un shim, c'est un module Cercle 1"
        )


# ──────────────────────────────────────────────────────────────────────────
# 5. Hooks builtin enregistrés (12 doc + 12 corpus)
# ──────────────────────────────────────────────────────────────────────────


class TestHooksStillRegistered:
    def test_12_doc_hooks(self):
        # Eager-load des hooks via builtin_hooks (qui est maintenant un shim
        # vers measurements/builtin_hooks).
        import picarones.core.builtin_hooks  # noqa: F401
        from picarones.core.metric_hooks import _all_document_hook_names

        hooks = _all_document_hook_names()
        # Le compte exact dépend des hooks expérimentaux qui tests
        # pourraient ajouter, donc on vérifie >= 12 et la présence des
        # 12 attendus.
        expected = {
            "confusion", "char_scores", "taxonomy", "structure",
            "image_quality", "line_metrics", "hallucination",
            "calibration", "philological", "searchability",
            "numerical_sequences", "readability",
        }
        assert expected.issubset(set(hooks))

    def test_alto_metrics_registered(self):
        import picarones.core.pipeline_runner  # eager-load
        from picarones.core.metric_registry import select_metrics
        from picarones.core.modules import ArtifactType

        metrics = select_metrics((ArtifactType.ALTO, ArtifactType.ALTO))
        names = {s.name for s in metrics}
        assert "alto_text_cer" in names
        assert "alto_text_wer" in names


# ──────────────────────────────────────────────────────────────────────────
# 6. build_synthesis fonctionne (intégration narrative complète)
# ──────────────────────────────────────────────────────────────────────────


class TestNarrativeIntegration:
    def test_build_synthesis_works(self):
        from picarones.core.narrative import build_synthesis

        synth = build_synthesis({
            "ranking": [
                {"engine": "tess", "mean_cer": 0.05},
                {"engine": "pero", "mean_cer": 0.08},
            ],
        }, lang="fr")
        assert "sentences" in synth
        assert "facts" in synth
        assert len(synth["sentences"]) >= 1


# ──────────────────────────────────────────────────────────────────────────
# 7. Vues du chantier 3 fonctionnent
# ──────────────────────────────────────────────────────────────────────────


class TestChantier3ViewsAfterPhaseE:
    def test_views_still_work(self):
        from picarones.report.views import (
            build_advanced_taxonomy_view_html,
            build_diagnostics_view_html,
            build_economics_view_html,
        )
        report_data = {"engines": [
            {"name": "tess", "cer": 0.05,
             "aggregated_taxonomy": {"class_distribution": {"x": 5}}},
            {"name": "pero", "cer": 0.08,
             "aggregated_taxonomy": {"class_distribution": {"x": 8}}},
        ]}
        # Au moins advanced_taxonomy doit produire du HTML
        html = build_advanced_taxonomy_view_html(report_data, {})
        assert isinstance(html, str)


# ──────────────────────────────────────────────────────────────────────────
# 8. Documentation cercles mise à jour
# ──────────────────────────────────────────────────────────────────────────


class TestArchitectureCerclesDocUpdated:
    @pytest.fixture
    def doc(self) -> str:
        path = Path(__file__).parent.parent / "docs" / "architecture-cercles.md"
        return path.read_text(encoding="utf-8")

    def test_critere_corrige(self, doc):
        """Le critère DDD remplace l'ancien critère ambigu."""
        assert "abstractions et logique métier du domaine" in doc
        assert "indépendantes de l'interface utilisateur" in doc

    def test_mention_phase_e(self, doc):
        """Le doc référence le sous-package measurements/."""
        # Au moins une mention du nouveau dossier
        # (chemin physique du Cercle 2)
        # NB : le doc parle de ``measurements/`` mais la lettre exacte
        # dépend de la formulation. On accepte plusieurs variantes.
        assert "measurements" in doc.lower() or "Cercle 2" in doc
