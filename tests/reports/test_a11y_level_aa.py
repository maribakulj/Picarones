"""Tests Sprint A7 — accessibilité WCAG niveau AA.

Items m-1, m-2, m-5, m-6, M-9 de l'audit institutional-readiness-2026-05.

Valide le **niveau AA** :
- WCAG 1.4.3 (Contrast Minimum) — palette Okabe-Ito par défaut
- WCAG 1.4.5 / 1.4.10 — pas de chaîne de fallback FR hardcodée dans
  le JS (i18n complet pour les messages d'absence de données)
- WCAG 1.4.11 — toggle palette daltonien-friendly disponible et
  persistant via URL
- WCAG 3.1.2 — locale BCP-47 par langue dans i18n
- M-9 — ACCESSIBILITY.md publié et linké
"""

from __future__ import annotations

import json
import re

import pytest

from picarones.evaluation.synthetic import generate_sample_benchmark
from picarones.reports.html.generator import ReportGenerator


@pytest.fixture(scope="module")
def demo_html(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("a11y_aa") / "report.html"
    bench = generate_sample_benchmark(n_docs=4)
    ReportGenerator(bench, lang="fr").generate(out)
    return out.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def demo_html_en(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("a11y_aa_en") / "report_en.html"
    bench = generate_sample_benchmark(n_docs=4)
    ReportGenerator(bench, lang="en").generate(out)
    return out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# m-1 + m-2 : i18n des messages d'absence de données
# ---------------------------------------------------------------------------


def test_no_anchor_data_uses_i18n(demo_html: str) -> None:
    """``_app.js:1092`` doit utiliser ``I18N.no_anchor_data`` et non
    une chaîne FR hardcodée."""
    # On cherche le pattern d'utilisation dans le JS embarqué.
    assert "I18N.no_anchor_data" in demo_html, (
        "Le fallback du chart d'ancrage doit utiliser I18N.no_anchor_data."
    )


def test_no_gini_uses_i18n(demo_html: str) -> None:
    """``_app.js:1054`` doit utiliser ``I18N.no_gini``."""
    assert "I18N.no_gini" in demo_html


def test_no_anchor_keys_in_both_languages(
    demo_html: str, demo_html_en: str
) -> None:
    """Les clés ``no_anchor_data`` et ``no_gini`` existent dans les
    deux dictionnaires i18n embarqués."""
    for html in (demo_html, demo_html_en):
        assert re.search(r'"no_anchor_data"\s*:\s*"', html)
        assert re.search(r'"no_gini"\s*:\s*"', html)


# ---------------------------------------------------------------------------
# m-5 : palette daltonien-friendly par défaut + toggle
# ---------------------------------------------------------------------------


def test_default_palette_is_okabe_ito(demo_html: str) -> None:
    """Les hex Okabe-Ito doivent apparaître dans le HTML rendu (au
    moins une fois — ``_cer_color`` les injecte sur les badges)."""
    okabe_hex = ["#0072B2", "#F0E442", "#E69F00", "#D55E00"]
    found = [h for h in okabe_hex if h in demo_html]
    assert len(found) >= 2, (
        f"Au moins 2 couleurs Okabe-Ito attendues, trouvé : {found}"
    )


def test_classic_palette_still_available_via_module() -> None:
    """``CLASSIC_*`` reste exportable pour rétrocompat."""
    from picarones.reports._helpers.colors import (
        CLASSIC_GREEN,
        CLASSIC_ORANGE,
        CLASSIC_RED,
        CLASSIC_YELLOW,
    )

    assert CLASSIC_GREEN == "#16a34a"
    assert CLASSIC_YELLOW == "#ca8a04"
    assert CLASSIC_ORANGE == "#ea580c"
    assert CLASSIC_RED == "#dc2626"


def test_palette_toggle_present_in_advanced_panel(demo_html: str) -> None:
    """Le panneau Avancé doit contenir la case à cocher de bascule
    palette."""
    assert 'id="palette-toggle-cb"' in demo_html
    assert "togglePalette" in demo_html


def test_palette_classic_class_styled(demo_html: str) -> None:
    """Le CSS doit définir le style ``body.palette-classic``
    (override de palette)."""
    assert "body.palette-classic" in demo_html


def test_palette_url_persistence(demo_html: str) -> None:
    """La fonction ``_initPaletteFromURL`` doit lire ``?palette=classic``
    au démarrage."""
    assert "_initPaletteFromURL" in demo_html
    assert "palette=classic" in demo_html or "'palette'" in demo_html


# ---------------------------------------------------------------------------
# m-6 : locale + toLocaleString
# ---------------------------------------------------------------------------


def test_locale_in_i18n_fr(demo_html: str) -> None:
    """L'i18n FR doit déclarer ``locale: "fr-FR"``."""
    assert re.search(r'"locale"\s*:\s*"fr-FR"', demo_html)


def test_locale_in_i18n_en(demo_html_en: str) -> None:
    """L'i18n EN doit déclarer ``locale: "en-GB"``."""
    assert re.search(r'"locale"\s*:\s*"en-GB"', demo_html_en)


def test_fmtnum_helper_present(demo_html: str) -> None:
    """Le helper ``fmtNum`` qui utilise ``toLocaleString(I18N.locale)``
    est défini dans le JS."""
    assert "function fmtNum" in demo_html
    assert "toLocaleString" in demo_html


# ---------------------------------------------------------------------------
# M-9 : ACCESSIBILITY.md
# ---------------------------------------------------------------------------


def test_accessibility_md_exists() -> None:
    """``docs/operations/accessibility.md`` doit exister (Phase 1 D5 :
    déplacé de ACCESSIBILITY.md (racine) vers la section operations
    pour assainir la racine)."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    a11y_md = repo_root / "docs" / "operations" / "accessibility.md"
    assert a11y_md.exists(), (
        "docs/operations/accessibility.md absent — pré-requis M-9 "
        "non satisfait."
    )


def test_accessibility_md_mentions_wcag_aa() -> None:
    """Le fichier doit déclarer l'engagement WCAG 2.1 AA + RGAA 4.1."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "docs" / "operations" / "accessibility.md").read_text(encoding="utf-8")
    assert "WCAG 2.1" in text
    assert "AA" in text
    assert "RGAA" in text


def test_accessibility_md_lists_remediation_items() -> None:
    """Le fichier doit lister les dérogations et l'audit externe planifié."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "docs" / "operations" / "accessibility.md").read_text(encoding="utf-8")
    # Doit mentionner l'audit externe (Sprint A15)
    assert "A15" in text or "audit externe" in text.lower()
    # Doit mentionner Okabe-Ito (palette validée)
    assert "Okabe-Ito" in text


# ---------------------------------------------------------------------------
# Cohérence i18n FR/EN
# ---------------------------------------------------------------------------


def test_i18n_fr_en_have_same_keys() -> None:
    """Les fichiers ``fr.json`` et ``en.json`` doivent avoir
    *exactement* le même set de clés (pas de dérive)."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    fr_keys = set(
        json.loads((repo_root / "picarones/reports/i18n/fr.json").read_text(encoding="utf-8")).keys()
    )
    en_keys = set(
        json.loads((repo_root / "picarones/reports/i18n/en.json").read_text(encoding="utf-8")).keys()
    )
    only_fr = fr_keys - en_keys
    only_en = en_keys - fr_keys
    assert not only_fr and not only_en, (
        f"Divergence i18n :\n  FR seulement: {sorted(only_fr)}\n"
        f"  EN seulement: {sorted(only_en)}"
    )
