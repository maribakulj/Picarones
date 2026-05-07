"""Utilitaires d'exécution de benchmark côté web.

API publique
------------
- ``sse_format`` : sérialisation d'un événement Server-Sent Events
  avec ``Last-Event-ID``.
- ``run_benchmark_thread`` / ``run_benchmark_thread_v2`` : workers
  threadés qui exécutent le benchmark, émettent des événements SSE
  via le ``BenchmarkJob``, génèrent le rapport HTML final.

Helpers internes (préfixe ``_``)
--------------------------------
- ``_build_llm_adapter`` : factory adapter LLM depuis une config
  ``CompetitorConfig``.
- ``_engine_from_competitor`` : factory moteur OCR ou pipeline
  OCR+LLM depuis une ``CompetitorConfig``.

Ces utilitaires sont consommés par le router ``/api/benchmark/*``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from picarones.web.models import (
    BenchmarkRequest,
    BenchmarkRunRequest,
    CompetitorConfig,
)
from picarones.web.state import BenchmarkJob, iso_now


def sse_format(event_type: str, data: Any, seq: Optional[int] = None) -> str:
    """Format Server-Sent Events.

    Émet une ligne ``id: <seq>`` quand le ``seq`` est connu.
    C'est la valeur que le navigateur renvoie automatiquement dans
    ``Last-Event-ID`` à la prochaine connexion (cf.
    https://html.spec.whatwg.org/multipage/server-sent-events.html).
    """
    payload = json.dumps(data, ensure_ascii=False)
    head = f"id: {seq}\n" if seq is not None else ""
    return f"{head}event: {event_type}\ndata: {payload}\n\n"


def _build_llm_adapter(comp: CompetitorConfig) -> Any:
    """Instancie un adaptateur LLM depuis la config d'un concurrent."""
    if comp.llm_provider == "openai":
        from picarones.llm.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(model=comp.llm_model or None)
    elif comp.llm_provider == "anthropic":
        from picarones.llm.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(model=comp.llm_model or None)
    elif comp.llm_provider == "mistral":
        from picarones.llm.mistral_adapter import MistralAdapter
        return MistralAdapter(model=comp.llm_model or None)
    elif comp.llm_provider == "ollama":
        from picarones.llm.ollama_adapter import OllamaAdapter
        return OllamaAdapter(model=comp.llm_model or None)
    else:
        raise ValueError(f"Provider LLM inconnu : {comp.llm_provider}")


def _engine_from_competitor(comp: CompetitorConfig) -> Any:
    """Instancie un moteur OCR (ou pipeline OCR+LLM) depuis une CompetitorConfig.

    Modes supportés :

    - ``ocr_engine`` = ``tesseract``, ``mistral_ocr``, … → moteur OCR seul.
    - ``ocr_engine`` + ``llm_provider`` → pipeline OCR live + LLM.
    - ``ocr_engine`` = ``corpus`` + ``llm_provider`` → post-correction LLM
      avec OCR pré-calculé (fichiers ``.ocr.txt`` du corpus triplet).
    - ``ocr_engine`` = ``""`` + ``llm_provider`` → LLM seul (zero-shot
      ou post-correction).
    """
    engine_id = comp.ocr_engine
    is_corpus_ocr = engine_id in ("corpus", "")

    if is_corpus_ocr and not comp.llm_provider:
        raise ValueError(
            "ocr_engine='corpus' nécessite un llm_provider "
            "(pour la post-correction ou le zero-shot)"
        )

    ocr = None
    if not is_corpus_ocr:
        from picarones.engines.tesseract import TesseractEngine
        from picarones.engines.mistral_ocr import MistralOCREngine

        if engine_id == "tesseract":
            ocr = TesseractEngine(config={"lang": comp.ocr_model or "fra", "psm": 6})
        elif engine_id == "mistral_ocr":
            ocr = MistralOCREngine(config={"model": comp.ocr_model or "mistral-ocr-latest"})
        elif engine_id == "google_vision":
            try:
                from picarones.engines.google_vision import GoogleVisionEngine
                ocr = GoogleVisionEngine(
                    config={"detection_type": comp.ocr_model or "document_text_detection"},
                )
            except ImportError as exc:
                raise RuntimeError("Google Vision non disponible.") from exc
        elif engine_id == "azure_doc_intel":
            try:
                from picarones.engines.azure_doc_intel import AzureDocIntelEngine
                ocr = AzureDocIntelEngine(
                    config={"model": comp.ocr_model or "prebuilt-document"},
                )
            except ImportError as exc:
                raise RuntimeError("Azure Document Intelligence non disponible.") from exc
        else:
            raise ValueError(f"Moteur OCR inconnu : {engine_id}")

        if not comp.llm_provider:
            return ocr

    # Pipeline OCR+LLM (live ou post-correction)
    mode_map = {
        "text_only": "text_only",
        "post_correction_text": "text_only",
        "text_and_image": "text_and_image",
        "post_correction_image": "text_and_image",
        "zero_shot": "zero_shot",
    }
    mode = mode_map.get(comp.pipeline_mode, "text_only")

    llm = _build_llm_adapter(comp)

    from picarones.pipelines.base import OCRLLMPipeline
    prompt = comp.prompt_file or "correction_medieval_french.txt"

    if is_corpus_ocr:
        pipeline_name = comp.name or f"corpus_ocr → {comp.llm_model or comp.llm_provider}"
    else:
        pipeline_name = comp.name or f"{engine_id} → {comp.llm_model or comp.llm_provider}"

    return OCRLLMPipeline(
        ocr_engine=ocr,
        llm_adapter=llm,
        mode=mode,
        prompt=prompt,
        pipeline_name=pipeline_name,
    )


def run_benchmark_thread_v2(job: BenchmarkJob, req: BenchmarkRunRequest) -> None:
    """Exécute un benchmark à partir d'une liste de ``CompetitorConfig``."""
    job.set_status("running")
    job.started_at = iso_now()
    job.add_event("start", {"message": "Démarrage du benchmark…", "corpus": req.corpus_path})

    try:
        from picarones.evaluation.corpus import load_corpus_from_directory
        from picarones.measurements.runner import run_benchmark

        corpus = load_corpus_from_directory(req.corpus_path)
        job.total_docs = len(corpus)
        job.add_event("log", {"message": f"{job.total_docs} documents chargés."})

        if job.status == "cancelled":
            return

        engines = []
        for comp in req.competitors:
            try:
                eng = _engine_from_competitor(comp)
                engines.append(eng)
                job.add_event("log", {"message": f"Concurrent : {eng.name}"})
            except Exception as exc:  # noqa: BLE001
                job.add_event("warning", {
                    "message": f"Concurrent ignoré '{comp.name or comp.ocr_engine}' : {exc}"
                })

        if not engines:
            raise ValueError("Aucun concurrent valide disponible.")

        # Sprint A14-S1 — A.I.0 P0 : ``output_dir`` a déjà été validé
        # par le router (validated_path).  ``report_name`` est sanitizé
        # ici pour défense en profondeur (refuse ``../``, séparateurs,
        # caractères de contrôle) avant concaténation à output_dir.
        from picarones.web.security import safe_report_name
        output_dir = Path(req.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_name = req.report_name or f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_name = safe_report_name(raw_name)
        output_json = str(output_dir / f"{report_name}.json")
        output_html = str(output_dir / f"{report_name}.html")

        n_engines = len(engines)
        total_steps = job.total_docs * n_engines
        step_counter = [0]

        def _progress_callback(engine_name: str, doc_idx: int, doc_id: str) -> None:
            if job.status == "cancelled":
                return
            step_counter[0] += 1
            job.current_engine = engine_name
            job.processed_docs = doc_idx
            job.progress = step_counter[0] / max(total_steps, 1)
            job.add_event("progress", {
                "engine": engine_name,
                "doc_idx": doc_idx,
                "doc_id": doc_id,
                "progress": job.progress,
                "processed": step_counter[0],
                "total": total_steps,
            })

        from picarones.measurements.normalization import _parse_exclude_chars
        char_excl = _parse_exclude_chars(req.char_exclude) if req.char_exclude else None

        result = run_benchmark(
            corpus=corpus,
            engines=engines,
            output_json=output_json,
            show_progress=False,
            progress_callback=_progress_callback,
            char_exclude=char_excl,
            cancel_event=job._cancel_event,
            normalization_profile=req.normalization_profile,
        )

        if job.status == "cancelled":
            return

        job.add_event("log", {"message": "Génération du rapport HTML…"})
        from picarones.reports_v2.html.generator import ReportGenerator
        gen = ReportGenerator(result, lang=req.report_lang)
        gen.generate(output_html)

        job.output_path = output_html
        job.progress = 1.0
        job.set_status("complete")

        ranking = result.ranking()
        job.add_event("complete", {
            "message": "Benchmark terminé.",
            "output_html": output_html,
            "output_json": output_json,
            "ranking": ranking,
        })

    except Exception as exc:  # noqa: BLE001
        job.set_status("error", error=str(exc))
        job.add_event("error", {"message": f"Erreur : {exc}"})


def run_benchmark_thread(job: BenchmarkJob, req: BenchmarkRequest) -> None:
    """Exécute le benchmark legacy (route ``/api/benchmark/start``)."""
    job.set_status("running")
    job.started_at = iso_now()
    job.add_event("start", {"message": "Démarrage du benchmark…", "corpus": req.corpus_path})

    try:
        from picarones.evaluation.corpus import load_corpus_from_directory
        from picarones.measurements.runner import run_benchmark

        # Charger le corpus
        job.add_event("log", {"message": f"Chargement du corpus : {req.corpus_path}"})
        corpus = load_corpus_from_directory(req.corpus_path)
        job.total_docs = len(corpus)
        job.add_event("log", {"message": f"{job.total_docs} documents chargés."})

        if job.status == "cancelled":
            return

        # Instancier les moteurs via la factory cercle 2 (pas de
        # dépendance ``click`` dans le code web).
        from picarones.engines.factory import engine_from_name

        ocr_engines = []
        for engine_name in req.engines:
            try:
                eng = engine_from_name(engine_name, lang=req.lang, psm=6)
                ocr_engines.append(eng)
                job.add_event("log", {"message": f"Moteur chargé : {engine_name}"})
            except Exception as exc:
                job.add_event("warning", {"message": f"Moteur ignoré '{engine_name}' : {exc}"})

        if not ocr_engines:
            raise ValueError("Aucun moteur valide disponible.")

        # Répertoire de sortie
        # Sprint A14-S1 — A.I.0 P0 : ``output_dir`` a déjà été validé
        # par le router (validated_path).  ``report_name`` est sanitizé
        # ici pour défense en profondeur (refuse ``../``, séparateurs,
        # caractères de contrôle) avant concaténation à output_dir.
        from picarones.web.security import safe_report_name
        output_dir = Path(req.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_name = req.report_name or f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_name = safe_report_name(raw_name)
        output_json = str(output_dir / f"{report_name}.json")
        output_html = str(output_dir / f"{report_name}.html")

        # Callback de progression
        n_engines = len(ocr_engines)
        total_steps = job.total_docs * n_engines
        step_counter = [0]

        def _progress_callback(engine_name: str, doc_idx: int, doc_id: str) -> None:
            if job.status == "cancelled":
                return
            step_counter[0] += 1
            job.current_engine = engine_name
            job.processed_docs = doc_idx
            job.progress = step_counter[0] / max(total_steps, 1)
            job.add_event("progress", {
                "engine": engine_name,
                "doc_idx": doc_idx,
                "doc_id": doc_id,
                "progress": job.progress,
                "processed": step_counter[0],
                "total": total_steps,
            })

        from picarones.measurements.normalization import _parse_exclude_chars
        char_excl = _parse_exclude_chars(req.char_exclude) if req.char_exclude else None

        result = run_benchmark(
            corpus=corpus,
            engines=ocr_engines,
            output_json=output_json,
            show_progress=False,
            progress_callback=_progress_callback,
            char_exclude=char_excl,
            cancel_event=job._cancel_event,
            normalization_profile=req.normalization_profile,
        )

        if job.status == "cancelled":
            return

        job.add_event("log", {"message": "Génération du rapport HTML…"})
        from picarones.reports_v2.html.generator import ReportGenerator
        report_lang = getattr(req, "report_lang", "fr")
        gen = ReportGenerator(result, lang=report_lang)
        gen.generate(output_html)

        job.output_path = output_html
        job.progress = 1.0
        job.set_status("complete")

        ranking = result.ranking()
        job.add_event("complete", {
            "message": "Benchmark terminé.",
            "output_html": output_html,
            "output_json": output_json,
            "ranking": ranking,
        })

    except Exception as exc:  # noqa: BLE001
        job.set_status("error", error=str(exc))
        job.add_event("error", {"message": f"Erreur : {exc}"})


__all__ = [
    "sse_format",
    "run_benchmark_thread",
    "run_benchmark_thread_v2",
]
