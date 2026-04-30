"""Tests de la phase A — refonte en 3 cercles (post-chantier 6).

Couvre :

- 4 modules `core/` déplacés vers `extras/academic/` ou
  `extras/governance/` avec shims rétrocompat.
- 4 renderers `report/` déplacés vers `extras/render/` avec shims.
- Identité préservée : ``shim.X is new_location.X`` (pas de duplication
  ni de redéfinition).
- Hygiène anti-verdict : 5 phrases reformulées dans les templates
  narratifs et l'i18n du rapport.
- Document `docs/architecture-cercles.md` présent et complet.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────────
# 1. Modules déplacés vers extras/ — rétrocompat des imports historiques
# ──────────────────────────────────────────────────────────────────────────


class TestRetrocompatHistoricalImports:
    """Les imports `from picarones.core.X` doivent continuer à fonctionner
    après le déplacement vers `picarones.extras.*`."""

    @pytest.mark.parametrize("module_path, attribute", [
        ("picarones.core.taxonomy_intra_doc", "compute_taxonomy_position_heatmap"),
        ("picarones.core.taxonomy_cooccurrence", "compute_taxonomy_cooccurrence"),
        ("picarones.core.image_predictive", "compute_paleographic_complexity"),
        ("picarones.core.image_predictive", "compute_corpus_homogeneity"),
        ("picarones.core.image_predictive", "aggregate_corpus_predictive"),
        ("picarones.core.module_policy", "ModuleManifest"),
        ("picarones.core.module_policy", "validate_manifest"),
        ("picarones.core.module_policy", "audit_module"),
    ])
    def test_core_alias_still_works(self, module_path: str, attribute: str):
        import importlib
        mod = importlib.import_module(module_path)
        assert hasattr(mod, attribute), (
            f"{module_path}.{attribute} a disparu après la phase A — "
            "le shim rétrocompat est cassé"
        )

    @pytest.mark.parametrize("module_path, attribute", [
        ("picarones.report.taxonomy_intra_doc_render", "build_taxonomy_intra_doc_html"),
        ("picarones.report.taxonomy_cooccurrence_render", "build_taxonomy_cooccurrence_html"),
        ("picarones.report.image_predictive_render", "build_image_predictive_html"),
        ("picarones.report.module_audit_render", "build_module_audit_html"),
    ])
    def test_report_alias_still_works(self, module_path: str, attribute: str):
        import importlib
        mod = importlib.import_module(module_path)
        assert hasattr(mod, attribute)


# ──────────────────────────────────────────────────────────────────────────
# 2. Modules accessibles via leur nouveau chemin extras/
# ──────────────────────────────────────────────────────────────────────────


class TestNewExtrasImports:
    @pytest.mark.parametrize("new_path, attribute", [
        ("picarones.extras.academic.taxonomy_intra_doc", "compute_taxonomy_position_heatmap"),
        ("picarones.extras.academic.taxonomy_cooccurrence", "compute_taxonomy_cooccurrence"),
        ("picarones.extras.academic.image_predictive", "aggregate_corpus_predictive"),
        ("picarones.extras.governance.module_policy", "ModuleManifest"),
        ("picarones.extras.render.taxonomy_intra_doc_render", "build_taxonomy_intra_doc_html"),
        ("picarones.extras.render.taxonomy_cooccurrence_render", "build_taxonomy_cooccurrence_html"),
        ("picarones.extras.render.image_predictive_render", "build_image_predictive_html"),
        ("picarones.extras.render.module_audit_render", "build_module_audit_html"),
    ])
    def test_extras_path_works(self, new_path: str, attribute: str):
        import importlib
        mod = importlib.import_module(new_path)
        assert hasattr(mod, attribute)


# ──────────────────────────────────────────────────────────────────────────
# 3. Identité préservée — pas de redéfinition par le shim
# ──────────────────────────────────────────────────────────────────────────


class TestIdentityThroughShim:
    """Le shim doit réexporter la fonction du nouveau chemin, pas la
    redéfinir. Sinon une métrique serait calculée différemment selon
    le chemin d'import."""

    def test_taxonomy_intra_doc_identity(self):
        from picarones.core.taxonomy_intra_doc import (
            compute_taxonomy_position_heatmap as via_old,
        )
        from picarones.extras.academic.taxonomy_intra_doc import (
            compute_taxonomy_position_heatmap as via_new,
        )
        assert via_old is via_new

    def test_image_predictive_identity(self):
        from picarones.core.image_predictive import (
            aggregate_corpus_predictive as via_old,
        )
        from picarones.extras.academic.image_predictive import (
            aggregate_corpus_predictive as via_new,
        )
        assert via_old is via_new

    def test_module_policy_identity(self):
        from picarones.core.module_policy import ModuleManifest as via_old
        from picarones.extras.governance.module_policy import (
            ModuleManifest as via_new,
        )
        assert via_old is via_new

    def test_renderer_identity(self):
        from picarones.report.taxonomy_intra_doc_render import (
            build_taxonomy_intra_doc_html as via_old,
        )
        from picarones.extras.render.taxonomy_intra_doc_render import (
            build_taxonomy_intra_doc_html as via_new,
        )
        assert via_old is via_new


# ──────────────────────────────────────────────────────────────────────────
# 4. Vues du chantier 3 — toujours fonctionnelles
# ──────────────────────────────────────────────────────────────────────────


class TestChantier3ViewsStillWork:
    """Les 5 vues du chantier 3 importent (sous-section opt-in) les
    modules déplacés. Vérifier qu'elles tournent encore après la
    migration."""

    def test_views_import(self):
        from picarones.report.views import (
            build_advanced_taxonomy_view_html,
            build_diagnostics_view_html,
            build_economics_view_html,
            build_pipeline_view_html,
            build_robustness_view_html,
        )
        assert callable(build_advanced_taxonomy_view_html)
        assert callable(build_diagnostics_view_html)
        assert callable(build_economics_view_html)
        assert callable(build_pipeline_view_html)
        assert callable(build_robustness_view_html)

    def test_advanced_taxonomy_with_intra_doc_data(self):
        """La vue advanced_taxonomy accepte des données opt-in
        ``intra_doc`` dont le calcul vient désormais de
        ``picarones.extras.academic``."""
        from picarones.extras.academic.taxonomy_intra_doc import (
            compute_taxonomy_position_heatmap,
        )
        from picarones.report.views import build_advanced_taxonomy_view_html

        # Calcul d'une heatmap minimaliste
        result = compute_taxonomy_position_heatmap(
            "abc def ghi", "abx def ghi", n_bins=3,
        )
        # La vue doit pouvoir composer sans crasher quand on lui passe
        # ces données opt-in
        report_data = {"engines": [
            {"name": "tess", "cer": 0.05,
             "aggregated_taxonomy": {"class_distribution": {"x": 5}}},
            {"name": "pero", "cer": 0.08,
             "aggregated_taxonomy": {"class_distribution": {"x": 8}}},
        ]}
        html = build_advanced_taxonomy_view_html(
            report_data, {}, intra_doc=result,
        )
        # Pas de crash + au moins du contenu (comparison + intra_doc)
        assert isinstance(html, str)


# ──────────────────────────────────────────────────────────────────────────
# 5. Hygiène anti-verdict — phrases reformulées
# ──────────────────────────────────────────────────────────────────────────


class TestAntiVerdictHygiene:
    """Les 5 phrases identifiées comme prescriptives ont été reformulées
    factuellement. Tests anti-régression."""

    @pytest.fixture
    def fr_templates(self) -> str:
        path = (Path(__file__).parent.parent
                / "picarones" / "core" / "narrative" / "templates" / "fr.yaml")
        return path.read_text(encoding="utf-8")

    @pytest.fixture
    def en_templates(self) -> str:
        path = (Path(__file__).parent.parent
                / "picarones" / "core" / "narrative" / "templates" / "en.yaml")
        return path.read_text(encoding="utf-8")

    @pytest.fixture
    def fr_i18n(self) -> str:
        path = (Path(__file__).parent.parent
                / "picarones" / "report" / "i18n" / "fr.json")
        return path.read_text(encoding="utf-8")

    @pytest.fixture
    def en_i18n(self) -> str:
        path = (Path(__file__).parent.parent
                / "picarones" / "report" / "i18n" / "en.json")
        return path.read_text(encoding="utf-8")

    def test_stratum_winner_no_dominate(self, fr_templates, en_templates):
        """`stratum_winner` ne dit plus « domine nettement » /
        « clearly dominates ». Phrasage factuel attendu."""
        assert "domine\n  nettement" not in fr_templates
        assert "domine nettement" not in fr_templates
        assert "clearly\n  dominates" not in en_templates
        assert "clearly dominates" not in en_templates
        # Confirmation présence du nouveau phrasage factuel
        assert "le CER le plus bas" in fr_templates
        assert "the lowest CER" in en_templates

    def test_confidence_warning_no_fragile(self, fr_templates, en_templates):
        """`confidence_warning` ne dit plus « fragile » mais
        « incertitude statistique élevée »."""
        assert "Classement fragile" not in fr_templates
        assert "Ranking is fragile" not in en_templates
        assert "Incertitude statistique" in fr_templates
        assert "High statistical uncertainty" in en_templates

    def test_gini_no_ideal(self, fr_i18n, en_i18n):
        """`gini_cer_ideal` et `gini_cer_note` n'utilisent plus
        « idéal » / « ideal » mais « lecture » / « reading »."""
        assert "\"gini_cer_ideal\": \"— idéal" not in fr_i18n
        assert "\"gini_cer_ideal\": \"— ideal" not in en_i18n
        # Confirmer le nouveau phrasage
        assert "lecture : bas-gauche" in fr_i18n
        assert "reading: bottom-left" in en_i18n

    def test_taxocomp_no_preferable(self, fr_i18n, en_i18n):
        """`taxocomp_note` ne dit plus « préférable » / « preferable »."""
        assert "préférable pour une édition critique" not in fr_i18n
        assert "preferable for a critical edition" not in en_i18n
        # Phrasage factuel
        assert "tend à produire des erreurs plus facilement" in fr_i18n
        assert "tends to produce errors more easily" in en_i18n


# ──────────────────────────────────────────────────────────────────────────
# 6. Document docs/architecture-cercles.md présent et complet
# ──────────────────────────────────────────────────────────────────────────


class TestArchitectureCerclesDoc:
    @pytest.fixture
    def doc(self) -> str:
        path = (Path(__file__).parent.parent / "docs" / "architecture-cercles.md")
        return path.read_text(encoding="utf-8")

    def test_doc_exists(self, doc):
        assert len(doc) > 1000

    def test_doc_describes_three_circles(self, doc):
        assert "Cercle 1" in doc
        assert "Cercle 2" in doc
        assert "Cercle 3" in doc
        assert "Noyau invariant" in doc or "noyau invariant" in doc
        assert "Plugins" in doc or "plugins" in doc

    def test_doc_assigns_specific_modules(self, doc):
        """Le document doit lister explicitement les modules de chaque cercle."""
        # Cercle 1 — quelques noms
        for name in ["corpus.py", "modules.py", "runner.py",
                     "metric_registry.py", "alto_metrics.py"]:
            assert name in doc, f"{name} doit être listé dans le doc"
        # Cercle 3 — modules déplacés en phase A
        for name in ["taxonomy_intra_doc", "image_predictive",
                     "module_policy"]:
            assert name in doc, f"{name} doit être listé dans le doc"

    def test_doc_mentions_extras_path(self, doc):
        """Le doc explique que les Cercle 3 vivent dans `extras/`."""
        assert "extras/academic" in doc
        assert "extras/governance" in doc
        assert "extras/render" in doc


# ──────────────────────────────────────────────────────────────────────────
# 7. Modules originaux ne contiennent plus de logique métier
# ──────────────────────────────────────────────────────────────────────────


class TestOriginalsAreShims:
    """Vérifie que les fichiers laissés à l'ancien emplacement sont
    bien des shims minces, pas des copies de la logique."""

    @pytest.mark.parametrize("path", [
        "picarones/core/taxonomy_intra_doc.py",
        "picarones/core/taxonomy_cooccurrence.py",
        "picarones/core/image_predictive.py",
        "picarones/core/module_policy.py",
        "picarones/report/taxonomy_intra_doc_render.py",
        "picarones/report/taxonomy_cooccurrence_render.py",
        "picarones/report/image_predictive_render.py",
        "picarones/report/module_audit_render.py",
    ])
    def test_is_thin_shim(self, path):
        repo_root = Path(__file__).parent.parent
        content = (repo_root / path).read_text(encoding="utf-8")
        # Un shim < 30 lignes (juste docstring + 2 imports + __all__)
        n_lines = len([line for line in content.splitlines() if line.strip()])
        assert n_lines < 30, (
            f"{path} fait {n_lines} lignes — devrait être un shim mince "
            "(import + réexport, pas de logique métier)"
        )
        # Doit contenir l'indication du déplacement
        assert "déplacé" in content or "extras" in content
