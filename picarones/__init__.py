"""Picarones — Plateforme de benchmark OCR/HTR pour documents patrimoniaux.

Licence Apache 2.0.

API publique des couches 1 & 3 (abstractions stables) ré-exportée
ici pour permettre :

>>> from picarones import Corpus, Document, BaseModule, ArtifactType
>>> from picarones import BenchmarkResult, EngineReport, DocumentResult

Pour les implémentations (calcul de métriques, runner, adapters OCR…),
utiliser les sous-packages explicites :

>>> from picarones.app.services.benchmark_runner import run_benchmark_via_service
>>> from picarones.evaluation.metrics.text_metrics import compute_metrics
>>> from picarones.adapters.ocr.tesseract import TesseractAdapter

Voir ``docs/explanation/architecture.md`` pour la cartographie complète des
8 couches, et ``docs/reference/api-stable.md`` pour le contrat de stabilité.
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
        __version__ = "2.0.0"

__author__ = "Picarones contributors"


# ──────────────────────────────────────────────────────────────────────────
# API publique — couches stables (domain + evaluation)
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

# ──────────────────────────────────────────────────────────────────────────
# API publique recommandée — Phase B3 migration Option B (mai 2026).
#
# ``RunOrchestrator`` remplace ``run_benchmark_via_service`` comme
# entry-point canonique pour lancer un benchmark.  Il consomme un
# ``RunSpec`` Pydantic et expose nativement les 4 fichiers JSONL
# (run_manifest, pipeline_results, artifacts_index, view_results) en
# plus du ``BenchmarkResult`` legacy (via ``spec.output_json``).
#
# La fonction ``run_benchmark_via_service`` reste disponible pour
# compat ascendante mais émet une ``DeprecationWarning`` à chaque
# appel.  Elle sera supprimée à la Phase B8 (post-deprecation
# release).
# ──────────────────────────────────────────────────────────────────────────

from picarones.app.schemas.run_spec import (
    RunSpec,
    RunSpecLoadError,
    load_run_spec_from_yaml,
)
from picarones.app.services.run_orchestrator import (
    OrchestrationResult,
    RunOrchestrator,
)

def register_default_metrics() -> None:
    """Charge le registre des métriques typées par défaut.

    Phase 5.3 audit code-quality (2026-05) : avant cette refonte,
    l'enregistrement passait par un ``import picarones.evaluation.metrics``
    side-effect en tête de module (anti-pattern dénoncé par
    ``test_no_side_effect_imports``).  La fonction ci-dessous extrait
    la logique en un appel explicite, **testable** et **idempotent**
    (Python met en cache les modules — un second appel ne déclenche
    aucun side-effect).

    L'enregistrement reste **auto-déclenché** au premier import du
    paquet ``picarones`` pour préserver l'API publique :

    >>> from picarones import register_metric  # déclenche l'auto-import
    >>> @register_metric(name="custom", ...)
    ... def my_metric(ref, hyp): ...

    Pour les consumers qui veulent un contrôle explicite (par exemple
    avant de fork() un worker), appeler ``register_default_metrics()``
    une fois au démarrage est sûr.

    Importe les sous-modules :

    - ``picarones.evaluation.metrics`` — déclencheur global des
      ``@register_metric``, ``@register_document_metric`` et
      ``@register_corpus_aggregator`` (~37 métriques scalaires +
      ~25 agrégateurs corpus-level).
    """
    # ``importlib.import_module`` plutôt qu'``import`` en local pour
    # rendre l'intention explicite (« je veux le side-effect du
    # module »).  Idempotent grâce au cache ``sys.modules``.
    import importlib
    importlib.import_module("picarones.evaluation.metrics")


# Auto-déclenchement préservé pour compat des callers existants —
# voir docstring de ``register_default_metrics``.
register_default_metrics()

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
    # Phase 5.3 audit code-quality — API d'enregistrement explicite.
    "register_default_metrics",
    # Phase B3 migration Option B — entry-point canonique.
    "OrchestrationResult",
    "RunOrchestrator",
    "RunSpec",
    "RunSpecLoadError",
    "load_run_spec_from_yaml",
]
