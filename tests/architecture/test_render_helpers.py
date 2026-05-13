"""Garde-fou contre la prolifération des helpers de rendu.

Les renderers HTML dans ``picarones/report/`` ont accumulé des helpers
locaux dupliqués (couleur, heatmap SVG, etc.) qui devraient vivre dans
un unique ``picarones/report/render_helpers.py``.

État après le sprint de consolidation : tous les ``_color_for_*`` et
``_build_heatmap_svg`` locaux ont été déplacés dans
``picarones/report/render_helpers.py`` qui expose
:func:`color_traffic_light`, :func:`color_single_gradient`,
:func:`color_diverging`, :func:`text_color_for_bg` et
:func:`build_grid_svg`.

Snapshot v1.0.0 (2026-05-02, post-consolidation) : **0 helper local
dupliqué**.

Test ratchet : ce nombre ne peut que descendre. Si un nouveau helper
``_color_for_*`` ou ``_build_heatmap_svg`` apparaît dans un renderer,
le test échoue. La résolution est de paramétrer un des helpers de
:mod:`picarones.reports._helpers.render_helpers` plutôt que de réintroduire
une fonction locale.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "picarones" / "report"

#: Snapshot v1.0.0 post-consolidation. Doit rester à 0.
HELPER_BASELINE = 0

#: Le module mutualisé est exempté (c'est *là* qu'on veut les voir).
HELPERS_MODULE_NAME = "render_helpers.py"

#: Fichiers à ignorer (pas des renderers).
IGNORED_FILES: frozenset[str] = frozenset({"__init__.py", HELPERS_MODULE_NAME})

#: Patterns capturant les helpers à mutualiser.
#:
#: On vise spécifiquement la duplication observée : coloration et
#: builders SVG génériques. Les helpers vraiment locaux (extraction
#: depuis une structure de données spécifique au domaine, formatage
#: dépendant de la métrique) ne sont *pas* visés.
HELPER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^def\s+_color_for\w*\s*\("),
    re.compile(r"^def\s+_color\s*\("),
    re.compile(r"^def\s+_build_heatmap\w*\s*\("),
)


def _scan_helpers() -> list[tuple[str, int, str]]:
    """Retourne la liste des (chemin_relatif, ligne, signature)."""
    found: list[tuple[str, int, str]] = []
    for path in sorted(REPORT_DIR.rglob("*.py")):
        if path.name in IGNORED_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line_num, line in enumerate(text.splitlines(), 1):
            for pattern in HELPER_PATTERNS:
                if pattern.match(line):
                    rel = path.relative_to(REPO_ROOT).as_posix()
                    found.append((rel, line_num, line.strip()))
                    break
    return found


def test_render_helpers_below_baseline() -> None:
    """Le nombre de helpers locaux ne peut que descendre.

    Quand on consolide un helper vers ``render_helpers.py``, abaisser
    aussi :data:`HELPER_BASELINE` dans le même commit pour verrouiller
    le gain.
    """
    helpers = _scan_helpers()
    count = len(helpers)
    locations = "\n".join(
        f"  {rel}:{line} — {sig}" for rel, line, sig in helpers
    )
    assert count <= HELPER_BASELINE, (
        f"\n{count} helpers locaux trouvés (baseline {HELPER_BASELINE}).\n"
        f"Régression : un nouveau helper a été ajouté.\n\n"
        f"Localisations :\n{locations}\n\n"
        "Soit déplace ce helper dans picarones/report/render_helpers.py "
        "et importe-le, soit relève HELPER_BASELINE consciemment dans "
        "tests/architecture/test_render_helpers.py."
    )


def test_baseline_must_be_tightened_when_progress_made() -> None:
    """Si le compte est sous le baseline, abaisse :data:`HELPER_BASELINE`.

    Force à verrouiller chaque consolidation : sans cette étape, le
    progrès n'est pas figé et une régression future passerait inaperçue
    sous le seuil obsolète.
    """
    count = len(_scan_helpers())
    assert count >= HELPER_BASELINE, (
        f"\nExcellent : {count} helpers vs baseline {HELPER_BASELINE}.\n\n"
        f"Mets à jour HELPER_BASELINE = {count} dans "
        "tests/architecture/test_render_helpers.py pour verrouiller le gain."
    )
