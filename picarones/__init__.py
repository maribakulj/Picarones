"""Picarones — Plateforme de benchmark OCR/HTR pour documents patrimoniaux.

Licence Apache 2.0.

API publique du Cercle 1 (abstractions stables) ré-exportée ici pour
permettre :

>>> from picarones import Corpus, Document, BaseModule, ArtifactType
>>> from picarones import BenchmarkResult, EngineReport, DocumentResult

Pour les implémentations (calcul de métriques, runner, adapters OCR…),
utiliser les sous-packages explicites :

>>> from picarones.measurements.runner import run_benchmark
>>> from picarones.measurements.metrics import compute_metrics
>>> from picarones.engines.tesseract import TesseractEngine

Voir ``docs/architecture.md`` pour la cartographie complète des
3 cercles, et ``docs/api-stable.md`` pour le contrat de stabilité.
"""

from __future__ import annotations

# Version (lecture dynamique depuis le package metadata après ``pip install -e .``)
try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("picarones")
except Exception:  # noqa: BLE001
    __version__ = "1.0.0"

__author__ = "Picarones contributors"


# ──────────────────────────────────────────────────────────────────────────
# API publique — Cercle 1 uniquement
# ──────────────────────────────────────────────────────────────────────────

from picarones.core.corpus import (
    Corpus,
    Document,
    GTLevel,
    TextGT,
    AltoGT,
    PageGT,
    EntitiesGT,
    ReadingOrderGT,
    load_corpus_from_directory,
)
from picarones.core.modules import ArtifactType, BaseModule
from picarones.core.results import (
    BenchmarkResult,
    DocumentResult,
    EngineReport,
)
from picarones.core.metrics import MetricsResult, aggregate_metrics
from picarones.core.facts import (
    DetectorRegistry,
    Fact,
    FactImportance,
    FactType,
)
from picarones.core.pipeline import (
    PipelineResult,
    PipelineRunner,
    PipelineSpec,
    PipelineStep,
    StepResult,
)
from picarones.core.metric_registry import (
    MetricSpec,
    compute_at_junction,
    register_metric,
    select_metrics,
)

__all__ = [
    "__version__",
    "__author__",
    # Corpus
    "Corpus",
    "Document",
    "GTLevel",
    "TextGT",
    "AltoGT",
    "PageGT",
    "EntitiesGT",
    "ReadingOrderGT",
    "load_corpus_from_directory",
    # Modules génériques (BaseModule)
    "ArtifactType",
    "BaseModule",
    # Résultats
    "BenchmarkResult",
    "DocumentResult",
    "EngineReport",
    "MetricsResult",
    "aggregate_metrics",
    # Moteur narratif (modèle de données)
    "DetectorRegistry",
    "Fact",
    "FactImportance",
    "FactType",
    # Pipelines composées (axe B)
    "PipelineResult",
    "PipelineRunner",
    "PipelineSpec",
    "PipelineStep",
    "StepResult",
    # Registre de métriques typées
    "MetricSpec",
    "compute_at_junction",
    "register_metric",
    "select_metrics",
]
