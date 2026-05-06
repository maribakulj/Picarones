"""Tests Sprint 69 — guide utilisateur « Écrire un module pipeline ».

Tests anti-régression sur la documentation utilisateur axe B.

On ne valide pas le contenu en profondeur (la rédaction est
subjective) mais on garantit que les **sections-clés et les
exemples-clés** restent présents dans le doc, pour qu'une refonte
de l'API n'oublie pas de mettre à jour le guide.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# S60 — restructuration Diataxis : ``user/`` éclaté en
# ``tutorials/`` (apprendre).
DOC_PATH = (
    Path(__file__).parent.parent.parent / "docs" / "tutorials"
    / "writing-a-pipeline-module.md"
)


@pytest.fixture(scope="module")
def doc() -> str:
    assert DOC_PATH.exists(), f"doc absente : {DOC_PATH}"
    return DOC_PATH.read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────
# 1. Présence des sections principales
# ──────────────────────────────────────────────────────────────────────────


class TestSections:
    def test_has_tldr(self, doc: str) -> None:
        assert "TL;DR" in doc

    def test_has_contract_section(self, doc: str) -> None:
        assert "contrat `BaseModule`" in doc.lower() or \
               "contrat ``basemodule``" in doc.lower() or \
               "Le contrat `BaseModule`" in doc

    def test_has_examples_section(self, doc: str) -> None:
        assert "Exemples pédagogiques" in doc

    def test_has_orchestration_section(self, doc: str) -> None:
        assert "Orchestrer une pipeline" in doc

    def test_has_html_report_section(self, doc: str) -> None:
        assert "rapport html" in doc.lower()

    def test_has_anti_patterns_section(self, doc: str) -> None:
        assert "Anti-patterns" in doc

    def test_has_best_practices_section(self, doc: str) -> None:
        assert "Bonnes pratiques" in doc


# ──────────────────────────────────────────────────────────────────────────
# 2. Couverture des concepts API
# ──────────────────────────────────────────────────────────────────────────


class TestApiConcepts:
    @pytest.mark.parametrize(
        "concept",
        [
            "BaseModule",
            "ArtifactType",
            "input_types",
            "output_types",
            "execution_mode",
            "process",
            "PipelineSpec",
            "PipelineStep",
            "PipelineRunner",
            "run_pipeline_benchmark",
            "compare_pipelines",
            "inputs_from",
            "build_pipeline_report_html",
            "build_pipeline_comparison_report_html",
            "RankingSpec",
        ],
    )
    def test_concept_mentioned(self, doc: str, concept: str) -> None:
        assert concept in doc, f"concept manquant : {concept}"


# ──────────────────────────────────────────────────────────────────────────
# 3. Philosophie « banc d'essai pas atelier »
# ──────────────────────────────────────────────────────────────────────────


class TestPhilosophy:
    def test_banc_d_essai_mentioned(self, doc: str) -> None:
        # Doit apparaître dans l'intro et dans les anti-patterns
        assert "banc d'essai" in doc.lower()

    def test_no_business_module_warning(self, doc: str) -> None:
        # Le doc doit clairement dire que Picarones n'a PAS de module métier.
        # On normalise : retire les marqueurs markdown (** _ > #) et les
        # espaces multiples (le markdown peut wrapper sur plusieurs
        # lignes, et les blockquotes commencent par `>` à chaque ligne).
        cleaned = doc.lower()
        for token in ("**", "_", ">", "#"):
            cleaned = cleaned.replace(token, " ")
        normalized = " ".join(cleaned.split())
        assert "aucun module métier" in normalized or \
               "pas de module métier" in normalized

    def test_examples_marked_pedagogic(self, doc: str) -> None:
        # Les exemples doivent être explicitement étiquetés
        # « pédagogique » / « mock » pour éviter le copier-coller
        # production
        assert "pédagogique" in doc.lower() or "mock" in doc.lower()


# ──────────────────────────────────────────────────────────────────────────
# 4. Référence des sprints
# ──────────────────────────────────────────────────────────────────────────


class TestSprintReferences:
    @pytest.mark.parametrize(
        "sprint_id", ["63", "64", "65", "66", "67", "68"],
    )
    def test_axis_b_sprint_referenced(
        self, doc: str, sprint_id: str,
    ) -> None:
        assert f"Sprint {sprint_id}" in doc, (
            f"Sprint {sprint_id} non référencé dans le guide"
        )

    def test_phase_0_sprints_referenced(self, doc: str) -> None:
        # Sprints 32-34 sont les fondations dont l'axe B dépend
        assert "Sprint 32" in doc or "Sprint 33" in doc \
               or "Sprint 34" in doc


# ──────────────────────────────────────────────────────────────────────────
# 5. Code snippets exécutables
# ──────────────────────────────────────────────────────────────────────────


class TestCodeSnippets:
    def test_has_python_code_blocks(self, doc: str) -> None:
        # Au moins quelques blocs ```python pour les exemples
        assert doc.count("```python") >= 5

    def test_imports_correct_modules(self, doc: str) -> None:
        # Les imports doivent pointer vers les vrais modules
        # picarones.core.* et picarones.report.*
        assert "from picarones.core.modules import" in doc
        assert "from picarones.core.pipeline import" in doc
        assert "from picarones.measurements.pipeline_benchmark import" in doc
        assert "from picarones.measurements.pipeline_comparison import" in doc
        assert "from picarones.report.pipeline_render import" in doc
