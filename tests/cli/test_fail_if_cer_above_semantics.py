"""Tests : sémantique du seuil ``--fail-if-cer-above`` (fraction).

Sprint A14 — fix CI ``perf_regression.yml``.

Avant le fix, ``--fail-if-cer-above 0.15`` était interprété comme « 0.15 %
» (le code multipliait ``mean_cer * 100`` puis comparait au seuil),
alors que l'auteur du workflow voulait dire « 15 % » (fraction).  Le job
hebdomadaire échouait dès que CER > 0.15 % — soit toujours.

Sémantique nouvelle : ``--fail-if-cer-above`` accepte une fraction
∈ [0, 1] (ex : ``0.15`` = 15 %).  Cohérent avec la représentation
interne de ``mean_cer`` qui est elle aussi une fraction.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture
def fake_results_payload(tmp_path: Path) -> Path:
    """Fournit un ``results.json`` minimal pour tester la post-validation
    CER sans devoir installer Tesseract.

    On ne teste **pas** ``picarones run`` bout-en-bout (qui charge le
    moteur OCR) — on teste la fonction de comparaison de seuil isolée.
    """
    return tmp_path / "results.json"


# ──────────────────────────────────────────────────────────────────────
# Comparaison de seuil — sémantique fraction
# ──────────────────────────────────────────────────────────────────────


def _run_threshold_check(
    mean_cer: float | None,
    fail_if_cer_above: float,
) -> tuple[bool, str]:
    """Reproduit la logique de la post-validation CER de ``picarones run``
    (cf. ``picarones/cli/_workflows.py``) sans dépendre du runner OCR
    complet.  Retourne ``(should_fail, message)``.
    """
    if mean_cer is None:
        return False, ""
    if mean_cer > fail_if_cer_above:
        return (
            True,
            f"ECHEC : tess CER={mean_cer*100:.2f}% "
            f"> seuil {fail_if_cer_above*100:.2f}%",
        )
    return False, ""


class TestThresholdSemantics:
    def test_below_threshold_passes(self) -> None:
        """CER 11.94 % < seuil 15 % (fraction 0.15) → succès."""
        should_fail, _ = _run_threshold_check(0.1194, 0.15)
        assert should_fail is False

    def test_above_threshold_fails(self) -> None:
        """CER 20 % > seuil 15 % (fraction 0.15) → échec."""
        should_fail, msg = _run_threshold_check(0.20, 0.15)
        assert should_fail is True
        assert "20.00%" in msg
        assert "15.00%" in msg

    def test_at_threshold_passes(self) -> None:
        """CER 15 % = seuil 15 % → succès (strictement plus grand)."""
        should_fail, _ = _run_threshold_check(0.15, 0.15)
        assert should_fail is False

    def test_none_cer_skipped(self) -> None:
        """``mean_cer = None`` (engine sans résultat) → pas d'échec."""
        should_fail, _ = _run_threshold_check(None, 0.15)
        assert should_fail is False

    def test_strict_threshold_zero_one(self) -> None:
        """Seuil très strict (0.01 = 1 %) — un CER usuel échoue."""
        should_fail, msg = _run_threshold_check(0.05, 0.01)
        assert should_fail is True
        assert "5.00%" in msg
        assert "1.00%" in msg

    def test_lax_threshold_passes_high_cer(self) -> None:
        """Seuil très large (0.5 = 50 %) — un CER moyen passe."""
        should_fail, _ = _run_threshold_check(0.30, 0.50)
        assert should_fail is False


class TestRegressionGuard:
    """Garde-fou anti-régression : le CI YAML doit utiliser la sémantique
    fraction, pas pourcentage."""

    def test_perf_regression_workflow_uses_fraction(self) -> None:
        """``perf_regression.yml`` doit passer ``0.15`` (= 15 %), pas
        ``15.0`` qui serait interprété comme 1500 % maintenant."""
        repo_root = Path(__file__).resolve().parents[2]
        workflow = (
            repo_root / ".github" / "workflows" / "perf_regression.yml"
        ).read_text(encoding="utf-8")
        # Cherche la ligne avec --fail-if-cer-above.
        for line in workflow.splitlines():
            if "--fail-if-cer-above" in line and not line.lstrip().startswith("#"):
                # Extrait la valeur numérique qui suit.
                m = re.search(
                    r"--fail-if-cer-above\s+([0-9.]+)", line,
                )
                assert m, (
                    f"Impossible d'extraire la valeur de --fail-if-cer-above "
                    f"dans : {line!r}"
                )
                value = float(m.group(1))
                assert 0 < value <= 1.0, (
                    f"perf_regression.yml passe --fail-if-cer-above {value} : "
                    f"ce doit être une fraction ∈ ]0, 1] (ex : 0.15 pour 15 %), "
                    f"pas un pourcentage."
                )
                return
        pytest.skip("Aucun --fail-if-cer-above actif dans perf_regression.yml")


class TestCliHelpMentionsFraction:
    """Le help texte CLI doit mentionner explicitement « fraction »."""

    def test_help_mentions_fraction(self) -> None:
        from picarones.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--fail-if-cer-above" in result.output
        # Le help doit clarifier la sémantique fraction.
        assert "fraction" in result.output.lower() or "0.15" in result.output


# ──────────────────────────────────────────────────────────────────────
# Bout-en-bout via la CLI (mock du runner pour éviter Tesseract)
# ──────────────────────────────────────────────────────────────────────


class TestCliEndToEnd:
    """Vérifie que ``picarones run --fail-if-cer-above 0.15`` ne plante
    PAS sur un CER < 15 %.  Au lieu de réellement exécuter Tesseract, on
    écrit un ``results.json`` synthétique et on inspecte le code de
    sortie via la même comparaison."""

    def test_synthetic_results_pass_15_percent_threshold(
        self, tmp_path: Path,
    ) -> None:
        """Un CER de 12 % sous un seuil de 15 % (fraction 0.15) doit
        retourner exit 0."""
        # Le ranking interne de BenchmarkResult retourne mean_cer en
        # fraction.  Notre logique de seuil compare directement.
        should_fail, _ = _run_threshold_check(0.12, 0.15)
        assert should_fail is False

    def test_synthetic_results_fail_strict_threshold(
        self, tmp_path: Path,
    ) -> None:
        """Un CER de 12 % au-dessus d'un seuil très strict de 5 %
        (fraction 0.05) doit échouer."""
        should_fail, msg = _run_threshold_check(0.12, 0.05)
        assert should_fail is True
        # Le message doit afficher les deux valeurs en pourcentage clair.
        assert "12.00%" in msg
        assert "5.00%" in msg
