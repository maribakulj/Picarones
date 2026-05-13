"""Phase 2.2 du plan d'audit — chaque module ``picarones.X.Y`` cité
dans ``docs/reference/api-stable.md`` comme rubrique de niveau 3
(``### `picarones....```) doit être réellement importable.

Le document revendique en son préambule :

    > Garantie principale : **existence** — aucun nom listé ne
    > disparaît entre ``1.x.0`` et ``2.0.0`` sans procédure de
    > dépréciation.

L'audit code-quality de mai 2026 a trouvé que 4 modules cités
n'existaient plus :

- ``picarones.pipeline.legacy_runner``
- ``picarones.pipeline.legacy_pipeline_benchmark``
- ``picarones.pipeline.legacy_pipeline_comparison``
- ``picarones.evaluation.metrics.pipeline_spec_loader``

Tous supprimés au passage v2.0 (retrait du legacy).  Ce test
empêche le drift de se reproduire.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

API_STABLE_MD = Path(__file__).resolve().parents[2] / "docs" / "reference" / "api-stable.md"

# Capture les rubriques de niveau 3 ``### `picarones.X.Y```.
# Le backtick fermant est obligatoire (évite de matcher du prose
# qui mentionne ``picarones.X`` sans intention de "rubrique API").
_RUBRIC_RE = re.compile(r"^###\s+`(picarones\.[\w\.]+)`", re.MULTILINE)


def _extract_modules() -> list[str]:
    if not API_STABLE_MD.exists():
        return []
    return _RUBRIC_RE.findall(API_STABLE_MD.read_text(encoding="utf-8"))


def test_every_api_stable_rubric_is_importable() -> None:
    """Chaque module cité comme ``### `picarones.X.Y``` doit s'importer
    sans erreur — garantie d'existence pour les consommateurs externes
    qui se fient à la documentation pour cibler une API stable.
    """
    modules = _extract_modules()
    assert modules, (
        f"{API_STABLE_MD} ne contient aucune rubrique ``### `picarones.X``` — "
        f"le test ne peut pas vérifier le contrat d'existence."
    )

    missing: list[tuple[str, str]] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            missing.append((name, str(exc)))

    assert not missing, (
        "api-stable.md référence des modules qui n'existent plus :\n"
        + "\n".join(f"  - {name} : {err}" for name, err in missing)
        + "\n\nSoit recréer le module, soit retirer la rubrique de "
        "``docs/reference/api-stable.md`` (et documenter la rupture dans "
        "CHANGELOG.md).  La garantie d'existence du document interdit "
        "le drift silencieux."
    )
