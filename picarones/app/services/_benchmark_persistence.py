"""Persistance JSON d'un ``BenchmarkResult`` — module extrait du
god-module ``benchmark_runner.py`` lors de la Phase 6 de l'audit
code-quality (2026-05).

Avant la Phase 6, ``benchmark_runner.py`` portait directement la
fonction ``_persist_benchmark_result_json``.  L'extraction permet :

- de tester la sérialisation isolément
- de réutiliser depuis d'autres workers sans importer le runner
- de réduire le god-module
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


def persist_benchmark_result_json(
    benchmark_result: Any, output_path: Path,
) -> None:
    """Sérialise un ``BenchmarkResult`` en JSON.

    Utilise ``dataclasses.asdict`` pour aplatir la structure
    récursivement.  Le format est compatible avec
    ``BenchmarkResult.from_json_object`` (Phase 2.2 du chantier
    post-rewrite) qui restaure round-trip toutes les analyses
    avancées (taxonomy, NER, calibration, etc.).

    Parameters
    ----------
    benchmark_result:
        Dataclass ``BenchmarkResult`` (couche 3).
    output_path:
        Chemin de sortie ; les répertoires parents sont créés au
        besoin.

    Notes
    -----
    Le ``default=str`` du ``json.dumps`` sert de filet pour les
    types non-natifs (``datetime``, ``Path``, ``Decimal``) — un
    payload typique ne devrait pas en contenir, mais c'est plus
    résilient.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(benchmark_result)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


__all__ = ["persist_benchmark_result_json"]
