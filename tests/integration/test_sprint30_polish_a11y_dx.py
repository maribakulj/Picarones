"""Tests Sprint 30 — polish, accessibilité et DX.

Sprint 30 livre quatre durcissements transverses :

1. ``picarones/i18n.py`` : chargement thread-safe via verrou explicite,
   ``lru_cache`` sur ``get_labels``, ``reload_translations()`` exposé.
2. ``BaseOCREngine._safe_version()`` log la stacktrace en DEBUG au
   lieu de swallow vers ``"unknown"`` silencieusement.
3. Badges CER WCAG : icône unicode + pattern de bordure + ``aria-label``
   contextuel — la couleur n'est plus la seule info visuelle.
4. Pre-commit hooks : ``.pre-commit-config.yaml`` + section
   ``CONTRIBUTING.md``.
5. ``CHANGELOG.md`` rattrapé Sprints 10-30.
6. ``SPECS.md`` annexé d'un addendum couvrant Sprints 16-30.
"""

from __future__ import annotations

import logging
from pathlib import Path


ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# 1. i18n thread-safe + lru_cache
# ---------------------------------------------------------------------------

class TestI18nCache:
    def test_get_labels_returns_dict(self):
        from picarones.reports_v2.i18n import get_labels
        labels = get_labels("fr")
        assert isinstance(labels, dict)
        assert len(labels) > 5

    def test_get_labels_unknown_falls_back_to_fr(self):
        from picarones.reports_v2.i18n import get_labels
        fr = get_labels("fr")
        unknown = get_labels("xx-pas-existante")
        # Le fallback doit être le contenu fr
        assert unknown == fr

    def test_get_labels_cached(self):
        from picarones.reports_v2 import i18n
        i18n.reload_translations()
        # Premier appel — peuple le cache
        i18n.get_labels("fr")
        # Inspection : le cache lru a un hit
        # (lru_cache expose cache_info() sur la fonction wrappée)
        info_before = i18n._get_labels_cached.cache_info()
        i18n.get_labels("fr")
        info_after = i18n._get_labels_cached.cache_info()
        assert info_after.hits > info_before.hits

    def test_reload_translations_clears_cache(self):
        from picarones.reports_v2 import i18n
        i18n.get_labels("fr")
        info_before = i18n._get_labels_cached.cache_info()
        assert info_before.currsize >= 1
        i18n.reload_translations()
        info_after = i18n._get_labels_cached.cache_info()
        assert info_after.currsize == 0


# ---------------------------------------------------------------------------
# 2. _safe_version log la stacktrace en DEBUG
# ---------------------------------------------------------------------------

class TestSafeVersionLogsDebug:
    def test_exception_in_version_does_not_propagate(self):
        from picarones.adapters.legacy_engines.base import BaseOCREngine

        class BrokenEngine(BaseOCREngine):
            @property
            def name(self) -> str:
                return "broken"

            def version(self) -> str:
                raise RuntimeError("désolé je suis cassé")

            def _run_ocr(self, image_path) -> str:
                return ""

        eng = BrokenEngine()
        # Ne doit pas lever
        v = eng._safe_version()
        assert v == "unknown"

    def test_stacktrace_emitted_at_debug_level(self, caplog):
        from picarones.adapters.legacy_engines.base import BaseOCREngine

        class BrokenEngine(BaseOCREngine):
            @property
            def name(self) -> str:
                return "broken"

            def version(self) -> str:
                raise RuntimeError("oops")

            def _run_ocr(self, image_path) -> str:
                return ""

        eng = BrokenEngine()
        with caplog.at_level(logging.DEBUG, logger="picarones.adapters.legacy_engines.base"):
            eng._safe_version()
        # Le log debug doit mentionner la classe + l'exception
        assert any(
            "BrokenEngine" in r.message and "oops" in r.message
            for r in caplog.records
        ), f"Records: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# 3. Badges CER WCAG (rapport HTML)
# ---------------------------------------------------------------------------

class TestBadgesAccessibility:
    def test_app_js_exposes_tier_helpers(self):
        path = ROOT / "picarones" / "reports_v2" / "html" / "templates" / "_app.js"
        src = path.read_text(encoding="utf-8")
        for fn in ("cerTier", "cerTierIcon", "cerTierLabel"):
            assert f"function {fn}" in src, (
                f"_app.js doit exposer ``function {fn}`` (Sprint 30 a11y)"
            )

    def test_styles_define_tier_patterns(self):
        path = ROOT / "picarones" / "reports_v2" / "html" / "templates" / "_styles.css"
        src = path.read_text(encoding="utf-8")
        for tier in ("excellent", "acceptable", "mediocre", "critical"):
            assert f'data-cer-tier="{tier}"' in src, (
                f"_styles.css doit définir un pattern pour le tier {tier!r}"
            )
        # Au moins quatre styles de bordure différents
        assert "border: 1.5px solid"  in src
        assert "border: 1.5px dashed" in src
        assert "border: 1.5px dotted" in src
        assert "border: 1.5px double" in src

    def test_main_badge_carries_data_attr_and_aria(self):
        path = ROOT / "picarones" / "reports_v2" / "html" / "templates" / "_app.js"
        src = path.read_text(encoding="utf-8")
        assert "setAttribute('data-cer-tier'" in src
        assert "setAttribute('aria-label'" in src


# ---------------------------------------------------------------------------
# 4. Pre-commit + CONTRIBUTING
# ---------------------------------------------------------------------------

class TestPreCommitInfra:
    def test_pre_commit_config_exists(self):
        path = ROOT / ".pre-commit-config.yaml"
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        # Doit référencer ruff (alignement avec le job CI ``lint``)
        assert "ruff" in text.lower()

    def test_pre_commit_yaml_is_well_formed(self):
        import yaml
        path = ROOT / ".pre-commit-config.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "repos" in data
        assert isinstance(data["repos"], list)
        assert any(
            "ruff" in (repo.get("repo") or "").lower()
            for repo in data["repos"]
        )

    def test_contributing_documents_pre_commit(self):
        path = ROOT / "CONTRIBUTING.md"
        text = path.read_text(encoding="utf-8")
        assert "pre-commit" in text.lower()
        assert "pre-commit install" in text


# ---------------------------------------------------------------------------
# 5. Documentation rattrapée
# ---------------------------------------------------------------------------

class TestChangelogAndSpecsUpdated:
    def test_changelog_mentions_recent_sprints(self):
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        # Backport Sprints 10-22 et 23-30 doivent être mentionnés
        for sprint in ("Sprint 11", "Sprint 17", "Sprint 19", "Sprint 22",
                       "Sprint 24", "Sprint 27", "Sprint 30"):
            assert sprint in text, (
                f"CHANGELOG.md doit mentionner {sprint} (Sprint 30 backport)"
            )

    def test_specs_addendum_present(self):
        text = (ROOT / "SPECS.md").read_text(encoding="utf-8")
        assert "Addendum" in text
        # Au moins quatre des nouvelles fonctionnalités annexées
        for keyword in ("narrative", "Pareto", "glossaire", "snapshots"):
            assert keyword in text.lower() or keyword in text, (
                f"SPECS.md addendum doit couvrir {keyword!r}"
            )


# ---------------------------------------------------------------------------
# 6. Intégration : un rapport généré porte les attributs WCAG dans son JS
# ---------------------------------------------------------------------------

class TestGeneratedReportCarriesA11y:
    def test_generated_html_embeds_tier_helpers(self, tmp_path):
        from picarones.evaluation import synthetic as fixtures
        from picarones.reports_v2.html.generator import ReportGenerator

        b = fixtures.generate_sample_benchmark(n_docs=4)
        out = tmp_path / "rapport.html"
        ReportGenerator(b, lang="fr").generate(out)
        html = out.read_text(encoding="utf-8")
        # Les fonctions JS doivent figurer dans le bundle inline
        assert "cerTier" in html
        assert "cerTierIcon" in html
        # Les règles CSS pour les patterns aussi
        assert 'data-cer-tier="excellent"' in html
