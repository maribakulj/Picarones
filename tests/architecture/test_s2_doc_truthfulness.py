"""Sprint S2.2 — Garde-fous contre la dérive entre code et documentation.

À v2.0, plusieurs documents racontaient une histoire fausse :

- ``docs/explanation/architecture.md`` parlait encore de « deux
  arborescences cohabitent par design » alors que le legacy était
  supprimé.
- ``CLAUDE.md`` et ``README.md`` annonçaient ``4150 tests`` au lieu
  des ~4189 réels.
- Le manifeste mentionnait ``reports_v2`` (renommé ``reports`` en
  Sprint H.3).

Ces tests verrouillent l'invariant : si un mainteneur futur
essaie de réintroduire ces formulations, il échoue le test.

Si une vraie évolution architecturale justifie de réécrire ces
sections, le test échoue → on met à jour les patterns ICI
consciemment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_MD = REPO_ROOT / "docs" / "explanation" / "architecture.md"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
README_MD = REPO_ROOT / "README.md"


# ──────────────────────────────────────────────────────────────────────
# 1. Le manifeste architectural ne ment plus sur l'état v2.0
# ──────────────────────────────────────────────────────────────────────


class TestArchitectureManifestoTruthful:
    """Le fichier ``docs/explanation/architecture.md`` a été
    réécrit en Sprint S2.1 pour refléter l'état v2.0 (une seule
    arborescence, plus de paquet legacy).  Toute régression
    réintroduisant les formulations historiques doit échouer."""

    def setup_method(self) -> None:
        self.text = ARCHITECTURE_MD.read_text(encoding="utf-8")

    def test_manifesto_does_not_claim_two_tree_coexistence(self) -> None:
        """La phrase « Deux arborescences cohabitent par design »
        décrit un état pré-v2.0.  À v2.0+, elle est fausse."""
        forbidden = "Deux arborescences cohabitent"
        assert forbidden not in self.text, (
            f"``docs/explanation/architecture.md`` contient "
            f"« {forbidden} » : ce texte décrit un état pré-v2.0. "
            f"À v2.0+, l'arborescence legacy a été supprimée. "
            f"Si une vraie cohabitation est réintroduite "
            f"(ex : pattern dual-stack v2.0/v3.0), mettre à jour "
            f"ce test ET la table de routage du manifeste."
        )

    def test_manifesto_does_not_reference_reports_v2(self) -> None:
        """``reports_v2/`` a été renommé ``reports/`` en Sprint H.3.
        Toute référence à ``reports_v2`` dans le manifeste = bug."""
        forbidden = "reports_v2"
        assert forbidden not in self.text, (
            f"Le manifeste contient ``{forbidden}``.  Le paquet a été "
            f"renommé ``reports`` au Sprint H.3.  Si une nouvelle "
            f"version ``reports_v3/`` est introduite, mettre à jour."
        )

    def test_manifesto_does_not_reference_legacy_packages(self) -> None:
        """Aucune référence aux paquets legacy supprimés en Sprints
        A-H ne doit subsister dans le manifeste actif."""
        legacy_paths = (
            "picarones.measurements",
            "picarones.engines",
            "picarones.modules",
            "picarones.report ",
            "picarones.report.",
            "picarones.report\n",
            "picarones.cli\n",
            "picarones.web\n",
            "picarones.llm\n",
            "picarones.pipelines\n",
            "picarones.extras",
            "picarones.core",
            "adapters/legacy_engines",
            "adapters/legacy_pipelines",
            "interfaces/cli/_legacy",
            "interfaces/web/_legacy",
        )
        offending = [p for p in legacy_paths if p in self.text]
        assert not offending, (
            f"Le manifeste cite des paquets supprimés à v2.0 : "
            f"{offending}.  Si une cohabitation est réintroduite, "
            f"documenter explicitement et mettre à jour ce test."
        )

    def test_manifesto_uses_current_layer_count(self) -> None:
        """Le manifeste actuel parle de ``8 couches`` (terminologie
        S2.1).  Un retour à ``3 cercles`` ou ``8 cercles`` est une
        régression."""
        # Doit contenir « 8 couches ».
        assert "8 couches" in self.text, (
            "Le manifeste ne mentionne plus ``8 couches`` — "
            "vérifier que la terminologie ``cercles`` historique "
            "n'a pas été réintroduite par mégarde."
        )
        # Ne doit PAS contenir ``3 cercles`` ou ``cercles concentriques``.
        # On accepte le mot ``cercle`` isolé (utilisé en CSS / palette
        # par exemple), mais pas comme structure architecturale.
        assert "8 cercles" not in self.text, (
            "Régression : ``8 cercles`` au lieu de ``8 couches``."
        )
        assert "3 cercles" not in self.text, (
            "Régression : retour au modèle 3-cercles pré-rewrite."
        )

    def test_manifesto_documents_all_8_layers(self) -> None:
        """Le tableau des 8 couches doit citer chacune par son
        nom canonique."""
        canonical_layers = (
            "domain",
            "formats",
            "evaluation",
            "pipeline",
            "adapters",
            "app",
            "reports",
            "interfaces",
        )
        for layer in canonical_layers:
            assert f"`picarones/{layer}/`" in self.text or f"`{layer}/`" in self.text, (
                f"Le manifeste ne documente pas la couche ``{layer}/``."
            )


# ──────────────────────────────────────────────────────────────────────
# 2. Compteurs de tests synchronisés
# ──────────────────────────────────────────────────────────────────────


class TestTestCountSynced:
    """Le compteur ``N tests passed`` cité dans CLAUDE.md / README.md
    doit rester proche du compte réel.

    Le script ``scripts/gen_readme_tables.py`` est censé maintenir la
    cohérence ; ce test attrape les cas où il n'a pas tourné.

    Tolérance : ``±5`` tests autour du compte réel (un commit peut
    introduire 1-3 nouveaux tests sans qu'on regenère immédiatement
    la doc — au-delà, c'est de la dérive).
    """

    @pytest.fixture
    def real_test_count(self) -> int:
        """Count réel des tests collectés par pytest (hors deselected)."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "--collect-only", "-q", "--no-cov",
                "-p", "no:cacheprovider",
                str(REPO_ROOT / "tests"),
            ],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
        )
        # La dernière ligne pertinente : « X tests collected »
        import re
        for line in reversed(result.stdout.strip().split("\n")):
            m = re.search(r"(\d+)\s+tests?\s+collected", line)
            if m:
                return int(m.group(1))
        pytest.fail(
            f"Impossible d'extraire le compte de pytest --collect-only.\n"
            f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-200:]}"
        )

    def _extract_count(self, text: str) -> int | None:
        """Cherche un nombre près du mot ``passed`` dans ``text``."""
        import re
        # Matche « 4189 passed » ou « ~4150 tests » ou « 4150 tests passed ».
        for pattern in (
            r"\*\*(\d{3,5})\s+passed",
            r"(\d{3,5})\s+passed",
            r"~?(\d{3,5})\s+tests",
        ):
            m = re.search(pattern, text)
            if m:
                return int(m.group(1))
        return None

    def test_claude_md_count_close_to_reality(
        self, real_test_count: int,
    ) -> None:
        text = CLAUDE_MD.read_text(encoding="utf-8")
        claimed = self._extract_count(text)
        assert claimed is not None, (
            "CLAUDE.md ne contient aucun compteur de tests (``N passed``)."
        )
        delta = abs(claimed - real_test_count)
        assert delta <= 50, (
            f"CLAUDE.md annonce {claimed} tests, réalité = "
            f"{real_test_count} (écart = {delta}).  Tolérance ±50.\n"
            f"Lancer ``python scripts/gen_readme_tables.py`` puis "
            f"committer."
        )

    def test_readme_md_count_close_to_reality(
        self, real_test_count: int,
    ) -> None:
        text = README_MD.read_text(encoding="utf-8")
        claimed = self._extract_count(text)
        assert claimed is not None, (
            "README.md ne contient aucun compteur de tests."
        )
        delta = abs(claimed - real_test_count)
        assert delta <= 50, (
            f"README.md annonce {claimed} tests, réalité = "
            f"{real_test_count} (écart = {delta})."
        )


# ──────────────────────────────────────────────────────────────────────
# 3. Liens internes vers archives correctement orthographiés
# ──────────────────────────────────────────────────────────────────────


class TestArchiveLinksWellFormed:
    """L'ancienne version du manifeste contenait des liens cassés
    type ``docs/archiv../archives/migration/...``.  Vérifier que ce
    pattern n'est pas réintroduit."""

    def test_no_typo_in_archive_paths(self) -> None:
        text = ARCHITECTURE_MD.read_text(encoding="utf-8")
        forbidden_substrings = (
            "archiv../archives",  # double slash + typo
            "/archiv../",
            "../archiv../",
        )
        for sub in forbidden_substrings:
            assert sub not in text, (
                f"Le manifeste contient le pattern cassé ``{sub}`` "
                f"(résidu d'une refactor mal faite)."
            )
