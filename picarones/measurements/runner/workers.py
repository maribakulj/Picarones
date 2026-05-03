"""Workers de niveau module pour les pools d'exécution.

Deux workers correspondant aux deux modes d'exécution :

- :func:`_cpu_doc_worker` — pour ``ProcessPoolExecutor`` (moteurs
  CPU-bound, instanciés dans le sous-processus). Doit être picklable :
  c'est pour ça qu'il est défini au niveau module.
- :func:`_io_doc_worker` — pour ``ThreadPoolExecutor`` (moteurs
  IO-bound / API HTTP). L'instance du moteur est partagée entre les
  threads.

Les deux finissent par appeler :func:`_compute_document_result` du
sous-module :mod:`document` pour calculer toutes les métriques.
"""

from __future__ import annotations

from typing import Optional

from picarones.core.results import DocumentResult
from picarones.engines.base import BaseOCREngine
from picarones.measurements.runner.document import _compute_document_result


def _cpu_doc_worker(args: tuple) -> "DocumentResult":
    """Worker pour ProcessPoolExecutor (moteurs CPU-bound).

    Instancie le moteur dans le sous-processus, exécute l'OCR et calcule
    toutes les métriques.  Doit être une fonction de niveau module pour être
    sérialisable par ``pickle``.

    Le tuple ``args`` peut contenir, par compatibilité ascendante :
    - 7 éléments : legacy (Sprint 13)
    - 8 éléments : + ``corpus_lang`` (Sprint 87)
    - 9 éléments : + ``profile`` (chantier 2 post-Sprint 97)
    - 10 éléments : + ``normalization_profile`` (Sprint A14-S1, A.I.0 P0)
    """
    norm_profile = None
    if len(args) == 10:
        (engine_module, engine_class_name, engine_config, doc_id,
         image_path, ground_truth, char_exclude_chars, corpus_lang,
         profile, norm_profile) = args
    elif len(args) == 9:
        (engine_module, engine_class_name, engine_config, doc_id,
         image_path, ground_truth, char_exclude_chars, corpus_lang,
         profile) = args
    elif len(args) == 8:
        (engine_module, engine_class_name, engine_config, doc_id,
         image_path, ground_truth, char_exclude_chars, corpus_lang) = args
        profile = "standard"
    else:
        (engine_module, engine_class_name, engine_config, doc_id,
         image_path, ground_truth, char_exclude_chars) = args
        corpus_lang = "fr"
        profile = "standard"
    import importlib
    mod = importlib.import_module(engine_module)
    engine_cls = getattr(mod, engine_class_name)
    engine = engine_cls(config=engine_config)
    ocr_result = engine.run(image_path)
    char_exclude = frozenset(char_exclude_chars) if char_exclude_chars else None
    return _compute_document_result(
        doc_id=doc_id,
        image_path=image_path,
        ground_truth=ground_truth,
        ocr_result=ocr_result,
        char_exclude=char_exclude,
        corpus_lang=corpus_lang,
        profile=profile,
        normalization_profile=norm_profile,
    )


def _io_doc_worker(
    engine: BaseOCREngine,
    doc: object,
    char_exclude: Optional[frozenset],
    corpus_lang: str = "fr",
    profile: str = "standard",
    normalization_profile: Optional[object] = None,
) -> "DocumentResult":
    """Worker pour ThreadPoolExecutor (moteurs IO-bound / API).

    Exécute l'OCR et calcule les métriques dans un thread.  L'instance du
    moteur est partagée entre les threads — les adaptateurs HTTP sont
    généralement sans état mutable entre les appels.

    Si le document possède un texte OCR pré-calculé (corpus triplet) et que
    le moteur est un pipeline OCR+LLM, utilise ``run_with_ocr_text()`` pour
    court-circuiter l'étape OCR et tester directement la post-correction LLM.
    """
    doc_ocr_text = getattr(doc, "ocr_text", None)
    if doc_ocr_text is not None:
        # Corpus triplet — vérifier si le moteur supporte run_with_ocr_text
        run_with = getattr(engine, "run_with_ocr_text", None)
        if run_with is not None:
            ocr_result = run_with(doc.image_path, doc_ocr_text)  # type: ignore[attr-defined]
        else:
            # Moteur OCR classique — ignorer le texte OCR pré-calculé
            ocr_result = engine.run(doc.image_path)  # type: ignore[attr-defined]
    else:
        ocr_result = engine.run(doc.image_path)  # type: ignore[attr-defined]

    return _compute_document_result(
        doc_id=doc.doc_id,  # type: ignore[attr-defined]
        image_path=str(doc.image_path),  # type: ignore[attr-defined]
        ground_truth=doc.ground_truth,  # type: ignore[attr-defined]
        ocr_result=ocr_result,
        char_exclude=char_exclude,
        corpus_lang=corpus_lang,
        profile=profile,
        normalization_profile=normalization_profile,
    )


__all__ = ["_cpu_doc_worker", "_io_doc_worker"]
