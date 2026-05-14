"""Helpers de migration B4 — re-export depuis le module de production.

Phase B3 résiduel (mai 2026) : ``run_via_orchestrator`` a été
promu de ``tests/_migration_helpers.py`` vers
``picarones.app.services.legacy_runner_compat`` pour pouvoir être
consommé aussi par les call sites CLI/Web (qui ne peuvent pas
importer depuis ``tests/``).

Ce module reste comme alias pour préserver les imports des tests
catégorie A migrés en Phase B4.  Sera retiré en Phase B8.
"""

from __future__ import annotations

from picarones.app.services.legacy_runner_compat import run_via_orchestrator

__all__ = ["run_via_orchestrator"]
