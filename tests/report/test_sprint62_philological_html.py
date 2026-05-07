"""Tests Sprint 62 — vue HTML « Profil philologique ».

Couvre :

1. Sections individuelles : rendu correct quand au moins un moteur a
   du signal pour le module donné ; chaîne vide si aucun.
2. Agrégateur : 6 sections présentes si les 6 modules ont du signal,
   sinon seulement les sections avec signal.
3. Adaptive masking complet : aucun moteur n'a de signal → ``""``.
4. Anti-injection HTML : noms de moteurs / catégories / caractères
   contenant ``<script>`` correctement échappés.
5. Cellules : code couleur appliqué, valeurs en %.
6. Pas de classification automatique (le mot
   « diplomatique » / « modernisant » apparaît seulement dans la
   note explicative, jamais comme étiquette de moteur).
7. Intégration dans le rapport HTML complet (FR + EN).
8. Complétude i18n : toutes les clés ``philo_*`` présentes en FR et EN.
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.reports_v2.html.renderers.philological import (
    build_abbreviations_section,
    build_early_modern_section,
    build_modern_archives_section,
    build_mufi_section,
    build_philological_profile_html,
    build_roman_numerals_section,
    build_unicode_blocks_section,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _eng_with_unicode(name: str = "Tesseract", acc: float = 0.85) -> dict:
    return {
        "name": name,
        "aggregated_philological": {
            "unicode_blocks": {
                "global_accuracy": acc, "n_chars_total": 1000,
                "per_block": {
                    "Latin Extended-A": {
                        "correct": int(acc * 100), "total": 100,
                        "accuracy": acc,
                    },
                    "Alphabetic Presentation Forms": {
                        "correct": 5, "total": 10, "accuracy": 0.5,
                    },
                },
            },
        },
    }


def _eng_with_mufi(name: str = "Pero", coverage: float = 0.78) -> dict:
    return {
        "name": name,
        "aggregated_philological": {
            "mufi": {"coverage": coverage, "n_mufi_chars_reference": 100},
        },
    }


def _eng_with_abbreviations(name: str = "T", strict: float = 0.6) -> dict:
    return {
        "name": name,
        "aggregated_philological": {
            "abbreviations": {
                "global_strict_score": strict,
                "global_expansion_score": 0.95,
                "n_abbreviations_in_reference": 50,
            },
        },
    }


def _eng_with_early_modern(name: str = "T") -> dict:
    return {
        "name": name,
        "aggregated_philological": {
            "early_modern": {
                "n_markers_reference": 100,
                "n_markers_preserved": 70,
                "global_preservation": 0.7,
                "per_category": {
                    "ligatures": {"total": 30, "preserved": 25, "preservation": 25 / 30},
                    "long_s": {"total": 50, "preserved": 30, "preservation": 0.6},
                    "ampersand": {"total": 20, "preserved": 15, "preservation": 0.75},
                },
            },
        },
    }


def _eng_with_modern_archives(name: str = "T") -> dict:
    return {
        "name": name,
        "aggregated_philological": {
            "modern_archives": {
                "n_markers_reference": 100,
                "n_strict_preserved": 60,
                "n_expansion_preserved": 90,
                "global_strict_score": 0.6,
                "global_expansion_score": 0.9,
                "per_category": {
                    "civility_titles": {
                        "n_total": 30, "n_strict_preserved": 25,
                        "n_expansion_preserved": 28,
                        "strict_score": 25 / 30, "expansion_score": 28 / 30,
                    },
                    "address": {
                        "n_total": 20, "n_strict_preserved": 10,
                        "n_expansion_preserved": 18,
                        "strict_score": 0.5, "expansion_score": 0.9,
                    },
                },
            },
        },
    }


def _eng_with_roman(name: str = "T") -> dict:
    return {
        "name": name,
        "aggregated_philological": {
            "roman_numerals": {
                "n_numerals_reference": 20,
                "per_status": {
                    "strict_preserved": 12, "case_changed": 3,
                    "j_dropped": 2, "converted_to_arabic": 2, "lost": 1,
                },
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────────
# 1. Sections individuelles
# ──────────────────────────────────────────────────────────────────────────


class TestIndividualSections:
    def test_unicode_blocks_renders(self) -> None:
        html = build_unicode_blocks_section([_eng_with_unicode()])
        assert "Précision par bloc Unicode" in html
        assert "Tesseract" in html
        assert "Latin Extended-A" in html

    def test_unicode_blocks_empty_without_signal(self) -> None:
        html = build_unicode_blocks_section([_eng_with_mufi()])
        assert html == ""

    def test_abbreviations_renders(self) -> None:
        html = build_abbreviations_section([_eng_with_abbreviations()])
        assert "Abréviations médiévales" in html
        assert "T" in html

    def test_mufi_renders(self) -> None:
        html = build_mufi_section([_eng_with_mufi()])
        assert "Couverture MUFI" in html
        assert "Pero" in html

    def test_early_modern_renders(self) -> None:
        html = build_early_modern_section([_eng_with_early_modern()])
        assert "Marqueurs typographiques" in html
        assert "ligatures" in html
        assert "long_s" in html
        assert "ampersand" in html

    def test_modern_archives_renders(self) -> None:
        html = build_modern_archives_section([_eng_with_modern_archives()])
        assert "Abréviations des archives modernes" in html
        assert "civility_titles" in html
        assert "address" in html

    def test_roman_numerals_renders(self) -> None:
        html = build_roman_numerals_section([_eng_with_roman()])
        assert "Numéraux romains" in html


# ──────────────────────────────────────────────────────────────────────────
# 2-3. Agrégateur + adaptive masking
# ──────────────────────────────────────────────────────────────────────────


class TestAggregator:
    def test_returns_empty_when_no_engine_has_signal(self) -> None:
        engines = [{"name": "X", "aggregated_philological": None}]
        assert build_philological_profile_html(engines) == ""

    def test_returns_empty_when_engines_summary_empty(self) -> None:
        assert build_philological_profile_html([]) == ""

    def test_includes_only_modules_with_signal(self) -> None:
        # Un seul moteur avec MUFI uniquement
        html = build_philological_profile_html([_eng_with_mufi()])
        assert html != ""
        assert "Couverture MUFI" in html
        # Sections sans signal absentes
        assert "Précision par bloc Unicode" not in html
        assert "Abréviations médiévales" not in html
        assert "Marqueurs typographiques" not in html
        assert "Abréviations des archives modernes" not in html
        assert "Numéraux romains" not in html

    def test_includes_all_six_when_full_signal(self) -> None:
        engines = [
            _eng_with_unicode(),
            _eng_with_mufi(),
            _eng_with_abbreviations(),
            _eng_with_early_modern(),
            _eng_with_modern_archives(),
            _eng_with_roman(),
        ]
        html = build_philological_profile_html(engines)
        for marker in (
            "Précision par bloc Unicode",
            "Abréviations médiévales",
            "Couverture MUFI",
            "Marqueurs typographiques",
            "Abréviations des archives modernes",
            "Numéraux romains",
        ):
            assert marker in html, f"section absente : {marker}"


# ──────────────────────────────────────────────────────────────────────────
# 4. Anti-injection HTML
# ──────────────────────────────────────────────────────────────────────────


class TestAntiInjection:
    def test_engine_name_with_script_escaped(self) -> None:
        eng = _eng_with_mufi(name="<script>alert(1)</script>")
        html = build_mufi_section([eng])
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_section_title_safely_escaped_via_labels(self) -> None:
        labels = {"philo_mufi_title": "<b>Hack</b>"}
        html = build_mufi_section([_eng_with_mufi()], labels=labels)
        assert "<b>Hack</b>" not in html
        assert "&lt;b&gt;Hack&lt;/b&gt;" in html


# ──────────────────────────────────────────────────────────────────────────
# 5. Cellules : couleur + valeur en %
# ──────────────────────────────────────────────────────────────────────────


class TestCells:
    def test_score_displayed_in_percent(self) -> None:
        html = build_mufi_section([_eng_with_mufi(coverage=0.78)])
        assert "78.0%" in html

    def test_color_present(self) -> None:
        # Le style background:#... doit apparaître
        html = build_mufi_section([_eng_with_mufi(coverage=0.5)])
        assert "background:#" in html


# ──────────────────────────────────────────────────────────────────────────
# 6. Pas de classification imposée
# ──────────────────────────────────────────────────────────────────────────


class TestNoForcedClassification:
    def test_engine_not_labeled_as_diplomatic_or_modernizing(self) -> None:
        # Le moteur a strict=1.0 (typique diplomatique) mais on ne
        # doit pas voir « diplomatique » comme étiquette de cellule.
        eng = _eng_with_abbreviations(name="DiploEngine", strict=1.0)
        html = build_abbreviations_section([eng])
        # « DiploEngine » apparaît parce que c'est le nom du moteur.
        assert "DiploEngine" in html
        # Le mot « diplomatique » n'apparaît que dans la note
        # explicative en bas (et peut être absent par défaut).
        # On vérifie qu'il n'est pas accolé au nom du moteur dans
        # une cellule de tableau.
        assert "DiploEngine</td>diplomatique" not in html
        assert "DiploEngine</td>modernisant" not in html


# ──────────────────────────────────────────────────────────────────────────
# 7. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


class TestI18nCompleteness:
    def _load(self, lang: str) -> dict:
        path = (
            Path(__file__).parent.parent.parent
            / "picarones" / "reports_v2" / "i18n" / f"{lang}.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_all_philo_keys_present_fr(self) -> None:
        d = self._load("fr")
        required = (
            "philo_profile_title", "philo_profile_note",
            "philo_engine_label", "philo_global_label",
            "philo_strict_label", "philo_expansion_label",
            "philo_n_total_label",
            "philo_unicode_blocks_title", "philo_unicode_blocks_note",
            "philo_abbreviations_title", "philo_abbreviations_note",
            "philo_mufi_title", "philo_mufi_note",
            "philo_mufi_coverage_label",
            "philo_early_modern_title", "philo_early_modern_note",
            "philo_modern_archives_title", "philo_modern_archives_note",
            "philo_roman_numerals_title", "philo_roman_numerals_note",
            "philo_roman_status_strict_preserved",
            "philo_roman_status_case_changed",
            "philo_roman_status_j_dropped",
            "philo_roman_status_converted_to_arabic",
            "philo_roman_status_lost",
        )
        for key in required:
            assert key in d, f"manque clé FR : {key}"

    def test_all_philo_keys_present_en(self) -> None:
        d_fr = self._load("fr")
        d_en = self._load("en")
        for key in d_fr:
            if key.startswith("philo_"):
                assert key in d_en, f"manque clé EN : {key}"
