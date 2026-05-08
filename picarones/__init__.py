"""Picarones — Plateforme de benchmark OCR/HTR pour documents patrimoniaux.

Licence Apache 2.0.

API publique du Cercle 1 (abstractions stables) ré-exportée ici pour
permettre :

>>> from picarones import Corpus, Document, BaseModule, ArtifactType
>>> from picarones import BenchmarkResult, EngineReport, DocumentResult

Pour les implémentations (calcul de métriques, runner, adapters OCR…),
utiliser les sous-packages explicites :

>>> from picarones.app.services._legacy_runner_adapter import run_benchmark_via_service
>>> from picarones.evaluation.metrics.text_metrics import compute_metrics
>>> from picarones.adapters.legacy_engines.tesseract import TesseractEngine

Voir ``docs/explanation/architecture.md`` pour la cartographie complète des
3 cercles, et ``docs/reference/api-stable.md`` pour le contrat de stabilité.
"""

from __future__ import annotations

# Version (Sprint A9 / M-5) — résolue dans cet ordre :
#   1. ``picarones._version`` injecté au build par setuptools_scm
#      (présent dans le wheel installé) ;
#   2. fallback ``importlib.metadata.version("picarones")`` pour les
#      installations editable où ``_version.py`` peut être stale ;
#   3. fallback final ``"1.0.0"`` si aucune source n'est disponible
#      (ex : tarball sans .git ni metadata).
try:
    from picarones._version import __version__  # type: ignore[import-not-found]
except ImportError:
    try:
        from importlib.metadata import version as _get_version
        __version__ = _get_version("picarones")
    except Exception:  # noqa: BLE001
        __version__ = "1.0.0"

__author__ = "Picarones contributors"


# ──────────────────────────────────────────────────────────────────────────
# API publique — Cercle 1 uniquement
# ──────────────────────────────────────────────────────────────────────────

from picarones.evaluation.corpus import (
    Corpus,
    Document,
    TextGT,
    AltoGT,
    PageGT,
    EntitiesGT,
    ReadingOrderGT,
    load_corpus_from_directory,
)
from picarones.domain.artifacts import ArtifactType
from picarones.domain.module_protocol import BaseModule
from picarones.evaluation.benchmark_result import (
    BenchmarkResult,
    DocumentResult,
    EngineReport,
)
from picarones.evaluation.metric_result import MetricsResult, aggregate_metrics
from picarones.domain.facts import (
    DetectorRegistry,
    Fact,
    FactImportance,
    FactType,
)
from picarones.evaluation.metric_registry import (
    MetricSpec,
    compute_at_junction,
    register_metric,
    select_metrics,
)

# Sprint A3 — trigger d'enregistrement du registre typé (Sprint 34).
# L'import de ``picarones.measurements`` provoque l'exécution des
# décorateurs ``@register_metric`` sur ``cer``, ``wer``, ``mer``,
# ``wil`` + ~15 métriques philologiques + reading order + NER + ALTO.
# Ce trigger remplace l'ancien import croisé Cercle 1 → Cercle 2 dans
# ``core/pipeline.py`` (violation B-1/B-2 du même esprit).
import picarones.measurements as _trigger_metric_registration  # noqa: F401, E402

__all__ = [
    "__version__",
    "__author__",
    # Corpus
    "Corpus",
    "Document",
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
    # Registre de métriques typées
    "MetricSpec",
    "compute_at_junction",
    "register_metric",
    "select_metrics",
]
