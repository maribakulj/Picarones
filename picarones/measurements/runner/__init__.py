"""Orchestrateur du benchmark.

Exécute les moteurs OCR/HTR sur le corpus de manière parallèle :

- ``ProcessPoolExecutor`` pour les moteurs CPU-bound (Tesseract, Pero OCR,
  Kraken) — les workers picklables vivent dans :mod:`workers`.
- ``ThreadPoolExecutor`` pour les moteurs IO-bound / API (Mistral, Google,
  Azure, LLMs).

Avant le sprint « découpage de runner.py » (mai 2026) ce module était
un fichier unique de 1019 lignes. Le sous-package éclate la
responsabilité par concern :

- :mod:`document` — calcul d'un :class:`DocumentResult` à partir d'un
  OCR (métriques principales + hooks via ``run_document_hooks(profile)``).
- :mod:`workers` — fonctions de niveau module pour ``ProcessPoolExecutor``
  (:func:`_cpu_doc_worker`) et ``ThreadPoolExecutor`` (:func:`_io_doc_worker`).
- :mod:`partial` — persistance NDJSON des résultats partiels pour
  reprise sur interruption.
- :mod:`orchestration` — :func:`run_benchmark` (boucle principale,
  pools, agrégation par moteur) + :func:`_build_pipeline_info`.
- :mod:`aggregation` — délégations rétrocompat vers les agrégateurs de
  ``builtin_hooks`` (chantier 2 post-Sprint 97).
- :mod:`ner_attach` — câblage NER au post-process (Sprint 40).

Ce ``__init__.py`` ré-exporte toute l'API publique historique pour que
les ~25 fichiers qui importent depuis ``picarones.measurements.runner``
continuent à fonctionner sans modification. Les symboles privés
``_compute_document_result``, ``_load_partial``, ``_partial_path``,
``_aggregate_*``, ``_calibration_from_engine_result`` sont ré-exportés
car les tests Sprint 13/40/42 les consomment directement.
"""

from picarones.measurements.runner.aggregation import (
    _aggregate_calibration,
    _aggregate_char_scores,
    _aggregate_confusion,
    _aggregate_hallucination,
    _aggregate_image_quality,
    _aggregate_line_metrics,
    _aggregate_structure,
    _aggregate_taxonomy,
)
from picarones.measurements.runner.document import (
    _calibration_from_engine_result,
    _compute_document_result,
    _make_error_doc_result,
    _make_timeout_doc_result,
)
from picarones.measurements.runner.ner_attach import (
    _aggregate_ner,
    _attach_ner_metrics,
)
from picarones.measurements.runner.orchestration import (
    _build_pipeline_info,
    run_benchmark,
)
from picarones.measurements.runner.partial import (
    _delete_partial,
    _load_partial,
    _partial_path,
    _partial_write_lock,
    _sanitize_filename,
    _save_partial_line,
)
from picarones.measurements.runner.workers import (
    _cpu_doc_worker,
    _io_doc_worker,
)

__all__ = [
    # API publique principale
    "run_benchmark",
    # Helpers calcul document
    "_compute_document_result",
    "_calibration_from_engine_result",
    "_make_error_doc_result",
    "_make_timeout_doc_result",
    # Workers picklables
    "_cpu_doc_worker",
    "_io_doc_worker",
    # Persistance partial
    "_partial_path",
    "_load_partial",
    "_save_partial_line",
    "_delete_partial",
    "_sanitize_filename",
    "_partial_write_lock",
    # Orchestration helper
    "_build_pipeline_info",
    # Délégations agrégation (rétrocompat tests Sprint 13/42)
    "_aggregate_calibration",
    "_aggregate_char_scores",
    "_aggregate_confusion",
    "_aggregate_hallucination",
    "_aggregate_image_quality",
    "_aggregate_line_metrics",
    "_aggregate_structure",
    "_aggregate_taxonomy",
    # NER (Sprint 40)
    "_aggregate_ner",
    "_attach_ner_metrics",
]
