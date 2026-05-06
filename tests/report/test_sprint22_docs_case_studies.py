"""Tests Sprint 22 — études de cas, documentation, polish.

Sprint 7 du plan rapport. Vérifie :
  1. La structure du dossier `docs/` (case-studies, user, developer).
  2. Les études de cas amorces sont bien étiquetées "Cas d'école".
  3. Les guides développeur existent et sont non vides.
  4. Le rapport HTML pointe vers les études de cas.
  5. Tests d'intégration end-to-end avec une variété de configurations.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
DOCS = ROOT / "docs"


# ---------------------------------------------------------------------------
# 1. Structure des docs
# ---------------------------------------------------------------------------

class TestDocsStructure:
    def test_docs_directory_exists(self):
        assert DOCS.is_dir(), "docs/ doit exister à la racine"

    def test_subdirectories_present(self):
        # S60 — restructuration Diataxis : ``user/`` éclaté en
        # ``tutorials/`` (apprendre) + ``how-to/`` (résoudre).  Les
        # studies de cas et le dossier developer/ restent.
        for sub in ("case-studies", "tutorials", "developer"):
            assert (DOCS / sub).is_dir(), f"docs/{sub}/ manquant"

    def test_case_studies_index_exists(self):
        assert (DOCS / "case-studies" / "README.md").is_file()

    def test_user_guide_exists(self):
        assert (DOCS / "tutorials" / "reading-a-report.md").is_file()

    def test_developer_index_exists(self):
        assert (DOCS / "developer" / "index.md").is_file()


# ---------------------------------------------------------------------------
# 2. Études de cas
# ---------------------------------------------------------------------------

class TestCaseStudies:
    def test_at_least_two_amorce_studies(self):
        files = sorted((DOCS / "case-studies").glob("*.md"))
        # README.md + au moins 2 études
        non_readme = [f for f in files if f.name != "README.md"]
        assert len(non_readme) >= 2

    def test_each_case_study_is_labeled_as_amorce(self):
        """Garde-fou crucial : pas de fausses études prétendant être réelles."""
        for f in (DOCS / "case-studies").glob("*.md"):
            if f.name == "README.md":
                continue
            content = f.read_text(encoding="utf-8")
            assert "Cas d'école" in content, (
                f"{f.name} doit être explicitement étiquetée 'Cas d'école' "
                "(les études fictives ne doivent pas être présentées comme réelles)"
            )

    def test_each_case_study_has_required_sections(self):
        required_headers = ["Contexte", "Question", "Verdict", "Limites", "Reproductibilité"]
        for f in (DOCS / "case-studies").glob("*.md"):
            if f.name == "README.md":
                continue
            content = f.read_text(encoding="utf-8")
            for h in required_headers:
                assert re.search(rf"^##\s+{h}\b", content, re.MULTILINE), (
                    f"{f.name} manque la section '{h}'"
                )


# ---------------------------------------------------------------------------
# 3. Documentation développeur
# ---------------------------------------------------------------------------

class TestDeveloperDocs:
    # S60 — restructuration Diataxis : ``narrative-engine.md`` a
    # migré sous ``docs/explanation/``.
    @pytest.mark.parametrize("rel_path", [
        "developer/index.md",
        "explanation/narrative-engine.md",
        "developer/extending-glossary.md",
        "developer/extending-i18n.md",
    ])
    def test_dev_doc_exists_and_non_empty(self, rel_path):
        f = DOCS / rel_path
        assert f.is_file(), f"docs/{rel_path} manquant"
        content = f.read_text(encoding="utf-8")
        assert len(content) > 500, f"docs/{rel_path} suspectement court"

    def test_narrative_doc_explains_anti_hallucination(self):
        content = (DOCS / "explanation" / "narrative-engine.md").read_text(encoding="utf-8")
        assert "anti-hallucination" in content.lower() or \
               "traçable" in content.lower(), \
               "Le guide narratif doit expliciter l'invariant anti-hallucination"


# ---------------------------------------------------------------------------
# 4. Intégration au rapport
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def benchmark_result():
    from picarones import fixtures
    return fixtures.generate_sample_benchmark(n_docs=5)


class TestReportIntegration:
    def test_report_links_to_case_studies(self, benchmark_result, tmp_path):
        from picarones.report.generator import ReportGenerator
        out = tmp_path / "report.html"
        ReportGenerator(benchmark_result).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "case-studies" in html

    def test_report_polish_no_consecutive_empty_lines_in_views(self, benchmark_result, tmp_path):
        """Garde-fou cosmétique léger — éviter les blocs vides excessifs
        introduits par les includes Jinja2."""
        from picarones.report.generator import ReportGenerator
        out = tmp_path / "report.html"
        ReportGenerator(benchmark_result).generate(out)
        html = out.read_text(encoding="utf-8")
        # Plus de 6 lignes vides consécutives = problème de mise en page
        assert "\n" * 8 not in html


# ---------------------------------------------------------------------------
# 5. Tests d'intégration end-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Génération sur plusieurs profils de corpus pour valider la robustesse
    de la chaîne après tous les sprints de la phase 0."""

    def test_small_corpus_renders(self, tmp_path):
        from picarones import fixtures
        from picarones.report.generator import ReportGenerator
        bench = fixtures.generate_sample_benchmark(n_docs=2)
        out = tmp_path / "small.html"
        ReportGenerator(bench).generate(out)
        assert out.stat().st_size > 50_000  # Chart.js inline minimum

    def test_large_corpus_renders(self, tmp_path):
        from picarones import fixtures
        from picarones.report.generator import ReportGenerator
        bench = fixtures.generate_sample_benchmark(n_docs=20)
        out = tmp_path / "large.html"
        ReportGenerator(bench).generate(out)
        assert out.stat().st_size > 50_000

    def test_english_locale_full_render(self, tmp_path):
        from picarones import fixtures
        from picarones.report.generator import ReportGenerator
        bench = fixtures.generate_sample_benchmark(n_docs=5)
        out = tmp_path / "en.html"
        ReportGenerator(bench, lang="en").generate(out)
        html = out.read_text(encoding="utf-8")
        # Tous les composants Sprint 16-21 doivent être présents
        for marker in [
            'class="synth-card"',           # Sprint 19 narrative
            'class="cdd-card"',             # Sprint 18 CDD
            'class="chart-card pareto-card"', # Sprint 20 Pareto
            'id="glossary-panel"',          # Sprint 21 glossaire
            'id="customize-panel"',         # Sprint 21 personnalisation
            'btn-customize',
            '<html lang="en">',
        ]:
            assert marker in html, f"Marqueur '{marker}' absent en EN"
