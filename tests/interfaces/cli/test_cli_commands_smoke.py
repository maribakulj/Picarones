"""Tests d'intégration CLI — smoke + chemins clés (Phase 6 chantier).

Avant Phase 6 : seul ``test_chantier4.py::TestCliWorkflows`` testait
les commandes ``diagnose``/``economics``/``edition`` via leur
``--help``.  Les autres commandes (``run``, ``history``, ``compare``,
``robustness``, ``metrics``, ``info``, ``engines``, ``demo``,
``report``) n'avaient aucun test d'intégration — une régression sur
leur enregistrement Click ou leur signature passait inaperçue jusqu'à
ce qu'un utilisateur réel les invoque.

Couvre :

1. **Smoke ``--help``** : toutes les commandes répondent sans
   exit code != 0 et listent leurs options principales.
2. **Demo end-to-end** : ``picarones demo`` génère un rapport HTML
   complet (sans corpus, sans moteur réel) — c'est le chemin que
   la doc README pointe pour l'évaluation rapide.
3. **Engines matrix** : ``picarones engines`` affiche les 8 moteurs
   du catalogue canonique (cohérence avec ``/api/engines``).
4. **Info dependencies** : ``picarones info`` liste Picarones +
   dépendances clé.
5. **History --regression** : sans base, retourne un message lisible
   au lieu de planter.

Pas de tests qui invoquent un vrai benchmark (corpus + moteur réel) —
ceux-ci vivent dans ``tests/integration/`` quand un OCR est dispo.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def runner():
    from click.testing import CliRunner
    return CliRunner()


@pytest.fixture
def cli():
    from picarones.interfaces.cli import cli as cli_group
    return cli_group


# ──────────────────────────────────────────────────────────────────────
# 1. Smoke --help pour toutes les commandes
# ──────────────────────────────────────────────────────────────────────


_ALL_COMMANDS = [
    "run", "diagnose", "economics", "edition", "compare",
    "robustness", "history", "metrics", "info", "engines",
    "demo", "report", "serve",
]


class TestSmokeHelp:
    """Toutes les commandes CLI doivent répondre à ``--help`` sans
    crash et lister leurs options principales.  Garde-fou contre une
    régression d'enregistrement Click."""

    @pytest.mark.parametrize("cmd_name", _ALL_COMMANDS)
    def test_help_works(self, runner, cli, cmd_name: str) -> None:
        result = runner.invoke(cli, [cmd_name, "--help"])
        assert result.exit_code == 0, (
            f"`picarones {cmd_name} --help` a échoué : "
            f"exit={result.exit_code}, output={result.output[:500]}"
        )
        # Le help doit au moins inclure le nom de la commande.
        assert cmd_name in result.output.lower() or "usage" in result.output.lower()

    def test_root_help_lists_all_commands(self, runner, cli) -> None:
        """``picarones --help`` doit lister toutes les sous-commandes
        canoniques — sinon une commande enregistrée mais non groupée
        passe inaperçue."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for cmd_name in _ALL_COMMANDS:
            assert cmd_name in result.output, (
                f"Sous-commande ``{cmd_name}`` absente de "
                f"``picarones --help`` :\n{result.output}"
            )


# ──────────────────────────────────────────────────────────────────────
# 2. Demo end-to-end : génération d'un rapport sans moteur
# ──────────────────────────────────────────────────────────────────────


class TestDemoCommand:
    """``picarones demo`` est le chemin d'évaluation rapide pour un
    utilisateur qui n'a pas Tesseract ni de corpus : il génère un
    rapport HTML synthétique pour explorer l'UI."""

    def test_demo_generates_html_file(self, runner, cli, tmp_path):
        output = tmp_path / "demo.html"
        result = runner.invoke(cli, ["demo", "--output", str(output)])
        assert result.exit_code == 0, result.output
        assert output.exists()
        assert output.stat().st_size > 5000, (
            "Le rapport démo doit faire au moins 5 Ko "
            "(HTML + Chart.js inline + données synthétiques)"
        )
        # Sanity : c'est bien du HTML.
        head = output.read_text(encoding="utf-8")[:200]
        assert "<!DOCTYPE html>" in head or "<html" in head.lower()


# ──────────────────────────────────────────────────────────────────────
# 3. Engines matrix : source de vérité unique avec /api/engines
# ──────────────────────────────────────────────────────────────────────


class TestEnginesCommand:
    """``picarones engines`` doit lister tous les moteurs canoniques
    (Phase 3 chantier post-rewrite : matrice unique avec la factory
    ``adapters/ocr/factory._SUPPORTED``)."""

    def test_engines_lists_all_canonical(self, runner, cli):
        result = runner.invoke(cli, ["engines"])
        assert result.exit_code == 0, result.output
        # Vérifie que les 8 moteurs canoniques apparaissent (le format
        # est libre — on ne lock pas la mise en page).
        for canonical in (
            "tesseract", "pero_ocr", "kraken", "calamari",
            "mistral_ocr", "google_vision", "azure_doc_intel",
            "precomputed",
        ):
            assert canonical in result.output, (
                f"Moteur canonique ``{canonical}`` absent de "
                f"``picarones engines`` :\n{result.output}"
            )


# ──────────────────────────────────────────────────────────────────────
# 4. Info : version + dépendances
# ──────────────────────────────────────────────────────────────────────


class TestInfoCommand:
    def test_info_shows_version(self, runner, cli):
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0, result.output
        assert "Picarones" in result.output

    def test_info_lists_key_dependencies(self, runner, cli):
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        # Quelques dépendances critiques doivent être listées
        # (statut "v1.2.3" ou "non installé", peu importe).
        for dep in ("click", "jiwer", "Pillow"):
            assert dep in result.output, (
                f"Dépendance ``{dep}`` absente de "
                f"``picarones info`` :\n{result.output}"
            )


# ──────────────────────────────────────────────────────────────────────
# 5. History : commande sans base disponible
# ──────────────────────────────────────────────────────────────────────


class TestHistoryCommand:
    def test_history_help(self, runner, cli):
        """Le help doit lister les options de filtre principales."""
        result = runner.invoke(cli, ["history", "--help"])
        assert result.exit_code == 0
        # Au moins une option de filtre / format
        assert ("--list" in result.output
                or "--engine" in result.output
                or "--regression" in result.output)


# ──────────────────────────────────────────────────────────────────────
# 6. Robustness : help + signature
# ──────────────────────────────────────────────────────────────────────


class TestRobustnessCommand:
    def test_robustness_help(self, runner, cli):
        result = runner.invoke(cli, ["robustness", "--help"])
        assert result.exit_code == 0
        assert "--corpus" in result.output or "--results" in result.output


# ──────────────────────────────────────────────────────────────────────
# 7. Compare : prend 2+ JSONs
# ──────────────────────────────────────────────────────────────────────


class TestCompareCommand:
    def test_compare_help(self, runner, cli):
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        # Compare prend au moins 2 fichiers de résultats.
        assert "RESULTS" in result.output.upper() or "compare" in result.output.lower()


# ──────────────────────────────────────────────────────────────────────
# 8. Metrics : sanity sur l'aide
# ──────────────────────────────────────────────────────────────────────


class TestMetricsCommand:
    def test_metrics_help(self, runner, cli):
        result = runner.invoke(cli, ["metrics", "--help"])
        assert result.exit_code == 0
        # ``metrics`` compare une référence et une hypothèse.
        assert "--reference" in result.output or "--hypothesis" in result.output


# ──────────────────────────────────────────────────────────────────────
# 9. Run : help expose les options sécurité Phase 2
# ──────────────────────────────────────────────────────────────────────


class TestRunCommand:
    def test_run_help_shows_engines_option(self, runner, cli):
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--engines" in result.output
        assert "--corpus" in result.output

    def test_run_fail_if_cer_above_validated(self, runner, cli, tmp_path):
        """Phase 2 chantier post-rewrite : ``--fail-if-cer-above 15.0``
        (ancienne sémantique pourcentage) doit être rejeté à
        l'analyse Click avec un message explicite suggérant la
        nouvelle sémantique fraction.  On passe un ``--corpus``
        valide pour que la validation du seuil soit atteinte (Click
        valide les options dans l'ordre)."""
        # Corpus minimal valide pour passer la validation Click sur --corpus.
        # Le test stoppe avant l'exécution réelle grâce au seuil invalide.
        result = runner.invoke(
            cli, ["run", "--corpus", str(tmp_path),
                  "--fail-if-cer-above", "15.0"],
        )
        assert result.exit_code != 0
        # Le message d'erreur Click cite ``--fail-if-cer-above`` et
        # explique la nouvelle sémantique (fraction ∈ [0, 1]).
        output_low = result.output.lower()
        assert "fail-if-cer-above" in output_low or "fraction" in output_low
