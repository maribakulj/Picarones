"""Tests de la phase B — extras/historical/ (philologique vers Cercle 3).

Couvre :

- 8 modules philologiques (Cercle 3) déplacés vers `extras/historical/`.
- 2 renderers correspondants déplacés vers `extras/render/`.
- Identité préservée à travers les shims (test ``is``).
- Intégration : `philological_runner` orchestre toujours les 6 modules
  même après déplacement.
- Dépendance Cercle 2 → Cercle 3 (`numerical_sequences` →
  `roman_numerals`) continue de fonctionner via shim.
- pyproject.toml déclare `[historical]` comme extra documentaire.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────────
# 1. Modules historiques accessibles via shims (rétrocompat)
# ──────────────────────────────────────────────────────────────────────────


class TestPhilologicalRetrocompat:
    @pytest.mark.parametrize("module_path, attribute", [
        ("picarones.core.unicode_blocks", "compute_unicode_block_accuracy"),
        ("picarones.core.abbreviations", "compute_abbreviation_metrics"),
        ("picarones.core.mufi", "compute_mufi_coverage"),
        ("picarones.core.early_modern_typography", "compute_early_modern_metrics"),
        ("picarones.core.modern_archives", "compute_modern_archives_metrics"),
        ("picarones.core.roman_numerals", "compute_roman_numeral_metrics"),
        ("picarones.core.lexical_modernization", "compute_lexical_modernization"),
        ("picarones.core.philological_runner", "compute_philological_metrics"),
        ("picarones.core.philological_runner", "aggregate_philological_metrics"),
    ])
    def test_core_alias_still_works(self, module_path: str, attribute: str):
        import importlib
        mod = importlib.import_module(module_path)
        assert hasattr(mod, attribute), (
            f"{module_path}.{attribute} a disparu après la phase B"
        )

    @pytest.mark.parametrize("module_path, attribute", [
        ("picarones.report.philological_render", "build_philological_profile_html"),
        ("picarones.report.lexical_modernization_render",
         "build_lexical_modernization_html"),
    ])
    def test_render_alias_still_works(self, module_path: str, attribute: str):
        import importlib
        mod = importlib.import_module(module_path)
        assert hasattr(mod, attribute)


# ──────────────────────────────────────────────────────────────────────────
# 2. Modules accessibles via leur nouveau chemin extras/historical/
# ──────────────────────────────────────────────────────────────────────────


class TestNewHistoricalImports:
    @pytest.mark.parametrize("new_path, attribute", [
        ("picarones.extras.historical.unicode_blocks",
         "compute_unicode_block_accuracy"),
        ("picarones.extras.historical.abbreviations",
         "compute_abbreviation_metrics"),
        ("picarones.extras.historical.mufi", "compute_mufi_coverage"),
        ("picarones.extras.historical.early_modern_typography",
         "compute_early_modern_metrics"),
        ("picarones.extras.historical.modern_archives",
         "compute_modern_archives_metrics"),
        ("picarones.extras.historical.roman_numerals",
         "compute_roman_numeral_metrics"),
        ("picarones.extras.historical.lexical_modernization",
         "compute_lexical_modernization"),
        ("picarones.extras.historical.philological_runner",
         "compute_philological_metrics"),
        ("picarones.extras.render.philological_render",
         "build_philological_profile_html"),
        ("picarones.extras.render.lexical_modernization_render",
         "build_lexical_modernization_html"),
    ])
    def test_extras_path_works(self, new_path: str, attribute: str):
        import importlib
        mod = importlib.import_module(new_path)
        assert hasattr(mod, attribute)


# ──────────────────────────────────────────────────────────────────────────
# 3. Identité préservée (shim et nouveau chemin = même fonction)
# ──────────────────────────────────────────────────────────────────────────


class TestIdentityThroughShim:
    def test_unicode_blocks_identity(self):
        from picarones.core.unicode_blocks import (
            compute_unicode_block_accuracy as via_old,
        )
        from picarones.extras.historical.unicode_blocks import (
            compute_unicode_block_accuracy as via_new,
        )
        assert via_old is via_new

    def test_philological_runner_identity(self):
        from picarones.core.philological_runner import (
            compute_philological_metrics as via_old,
        )
        from picarones.extras.historical.philological_runner import (
            compute_philological_metrics as via_new,
        )
        assert via_old is via_new

    def test_renderer_identity(self):
        from picarones.report.philological_render import (
            build_philological_profile_html as via_old,
        )
        from picarones.extras.render.philological_render import (
            build_philological_profile_html as via_new,
        )
        assert via_old is via_new


# ──────────────────────────────────────────────────────────────────────────
# 4. Intégration : philological_runner orchestre toujours les 6 modules
# ──────────────────────────────────────────────────────────────────────────


class TestPhilologicalRunnerIntegration:
    """Le runner philologique appelle les 6 modules
    philologiques. Vérifie que cette chaîne fonctionne après le
    déplacement (les imports internes traversent les shims)."""

    def test_runner_returns_dict_or_none(self):
        from picarones.core.philological_runner import (
            compute_philological_metrics,
        )
        # Texte sans signal philologique → None par adaptive masking
        result = compute_philological_metrics(
            "Bonjour le monde", "Bonjour le monde",
        )
        # None acceptable (texte ASCII pur sans aucun marqueur)
        # OU dict vide (signal nul partout)
        assert result is None or isinstance(result, dict)

    def test_runner_with_medieval_text(self):
        """Texte médiéval avec abréviations + numéraux romains : on
        s'attend à au moins un module qui détecte du signal."""
        from picarones.core.philological_runner import (
            compute_philological_metrics,
        )
        # ⁊ = symbole d'abréviation Capelli ; XIV = numéral romain ; ſ = long s
        ref = "⁊ par leſ XIV. fontoyers"
        hyp = "et par les XIV. fontoyers"
        result = compute_philological_metrics(ref, hyp)
        # Au moins un module doit avoir détecté du signal
        # (abbreviations OU early_modern OU roman_numerals)
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) >= 1


# ──────────────────────────────────────────────────────────────────────────
# 5. Dépendance Cercle 2 → Cercle 3 fonctionne via shim
# ──────────────────────────────────────────────────────────────────────────


class TestCercle2DependsOnCercle3ViaShim:
    """``picarones.core.numerical_sequences`` (Cercle 2,
    measurements/) importe ``roman_numerals`` (Cercle 3, extras/).
    Cette dépendance traverse le shim — elle continue à fonctionner."""

    def test_numerical_sequences_uses_roman_numerals(self):
        from picarones.core.numerical_sequences import (
            compute_numerical_sequence_metrics,
        )
        # Texte avec numéral romain
        result = compute_numerical_sequence_metrics(
            "Le roi Louis XIV régna jusqu'en 1715",
            "Le roi Louis XIV régna jusqu'en 1715",
        )
        # Le score strict global doit refléter au moins la détection
        # du romain et de la date
        assert isinstance(result, dict)
        assert result.get("global_strict_score") is not None
        assert result.get("global_strict_score") >= 0.5


# ──────────────────────────────────────────────────────────────────────────
# 6. pyproject.toml déclare l'extra [historical]
# ──────────────────────────────────────────────────────────────────────────


class TestPyprojectExtra:
    def test_historical_extra_declared(self):
        path = Path(__file__).parent.parent / "pyproject.toml"
        content = path.read_text(encoding="utf-8")
        # L'extra [historical] doit être déclaré, même vide
        assert "historical = []" in content or 'historical = [' in content
        # Documentation de l'intention présente
        assert "extras/historical" in content
        assert "Cercle 3" in content


# ──────────────────────────────────────────────────────────────────────────
# 7. Hooks builtin enregistrés conditionnels (philological + lexical)
# ──────────────────────────────────────────────────────────────────────────


class TestBuiltinHooksStillRegisterPhilological:
    """Les hooks ``philological`` et ``lexical_modernization``
    s'enregistrent au chargement de :mod:`picarones.core.builtin_hooks`
    via les imports qui traversent les shims (``from
    picarones.core.philological_runner import ...``)."""

    def test_philological_hook_registered(self):
        # L'import déclenche l'enregistrement
        import picarones.core.builtin_hooks  # noqa: F401
        from picarones.core.metric_hooks import _all_document_hook_names

        assert "philological" in _all_document_hook_names()


# ──────────────────────────────────────────────────────────────────────────
# 8. Modules originaux sont des shims minces
# ──────────────────────────────────────────────────────────────────────────


class TestOriginalsAreShims:
    @pytest.mark.parametrize("path", [
        "picarones/core/unicode_blocks.py",
        "picarones/core/abbreviations.py",
        "picarones/core/mufi.py",
        "picarones/core/early_modern_typography.py",
        "picarones/core/modern_archives.py",
        "picarones/core/roman_numerals.py",
        "picarones/core/lexical_modernization.py",
        "picarones/core/philological_runner.py",
        "picarones/report/philological_render.py",
        "picarones/report/lexical_modernization_render.py",
    ])
    def test_is_thin_shim(self, path):
        repo_root = Path(__file__).parent.parent
        content = (repo_root / path).read_text(encoding="utf-8")
        n_lines = len([line for line in content.splitlines() if line.strip()])
        assert n_lines < 30, (
            f"{path} fait {n_lines} lignes — devrait être un shim mince"
        )
        assert "déplacé" in content or "extras" in content
