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
# 2. Compteurs de tests — pas de chiffre exact en prose
# ──────────────────────────────────────────────────────────────────────
#
# Historique : ce module comparait ``N tests passed`` cité dans
# CLAUDE.md / README.md au compte réel via
# ``subprocess.run([..., "pytest", "--collect-only", ...])``.  Trois
# problèmes : (a) pytest-dans-pytest avec ``--cov`` deadlocke sur
# ``.coverage`` ; (b) le compteur réel dérive de ±1 entre OS selon
# les binaires optionnels installés ; (c) un test qui rate à cause
# d'un compteur en prose est purement narratif.
#
# Stratégie actuelle : la prose dit ``5000+ tests`` (sans nombre
# exact), le chiffre canonique vit dans le badge CI.  Ces tests
# verrouillent l'absence de réintroduction d'un compteur exact.

import re


class TestTestCountInProseRemainsApproximate:
    """README et CLAUDE.md ne doivent plus citer de compteur de tests
    exact.  La formulation canonique est ``N+ tests`` / ``N+ passed``
    (avec le ``+`` qui marque l'approximation)."""

    _FORBIDDEN = re.compile(
        r"(?<!\+)\b(\d{4,5})\s+(?:tests|passed)\b",
        re.IGNORECASE,
    )

    def test_readme_uses_approximate_formulation(self) -> None:
        text = README_MD.read_text(encoding="utf-8")
        offenders = self._FORBIDDEN.findall(text)
        assert not offenders, (
            f"README.md cite des compteurs exacts : {offenders}. "
            "Utiliser ``N+ tests`` (ex. ``5000+ tests``)."
        )

    def test_claude_md_uses_approximate_formulation(self) -> None:
        text = CLAUDE_MD.read_text(encoding="utf-8")
        offenders = self._FORBIDDEN.findall(text)
        assert not offenders, (
            f"CLAUDE.md cite des compteurs exacts : {offenders}. "
            "Utiliser ``N+ tests`` (ex. ``5000+ tests``)."
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
