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
  ``PipelineConfig``.
- ``_engine_from_competitor`` : factory moteur OCR ou pipeline
  OCR+LLM depuis une ``PipelineConfig``.

Ces utilitaires sont consommés par le router ``/api/benchmark/*``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from picarones.interfaces.web.models import (
    BenchmarkRunRequest,
    PipelineConfig,
)
from picarones.interfaces.web.state import BenchmarkJob, iso_now

#: Répertoire de la bibliothèque de prompts embarquée — la même
#: que celle validée par ``validated_prompt_filename`` côté router.
_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _load_prompt_content(prompt_filename: str) -> str:
    """Charge le contenu d'un prompt embarqué depuis
    ``picarones/prompts/``.

    Avant le rewrite v2.0, l'``OCRLLMPipeline`` legacy lisait elle-
    même le fichier depuis disque.  Au cours du sprint H.2.c-d, ce
    chargement a disparu — le pipeline canonique
    ``OCRLLMPipelineConfig`` accepte un ``prompt_template`` string
    et n'a aucune connaissance du système de fichiers, donc le
    factory web (``_engine_from_competitor``) doit lire le fichier
    AVANT d'instancier le pipeline.

    Sans ce loader, le LLM recevait le filename brut comme prompt
    (par ex. ``"correction_early_modern_english.txt"``) et répondait
    avec du méta-discours sur le fichier au lieu de corriger l'OCR.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas dans ``picarones/prompts/``.
    """
    prompt_path = _PROMPTS_DIR / prompt_filename
    # Défense en profondeur : refuse de remonter hors du dossier
    # prompts (le filename est censé être déjà validé par
    # ``validated_prompt_filename`` côté router, mais on re-vérifie
    # car ce factory est aussi appelable directement).
    resolved = prompt_path.resolve()
    if not resolved.is_relative_to(_PROMPTS_DIR.resolve()):
        raise ValueError(
            f"Prompt filename invalide : {prompt_filename!r} pointe "
            f"hors de la bibliothèque embarquée.",
        )
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Prompt introuvable : {prompt_filename!r} dans "
            f"{_PROMPTS_DIR}.  Fichiers disponibles : "
            f"{sorted(p.name for p in _PROMPTS_DIR.glob('*.txt'))}",
        )
    return resolved.read_text(encoding="utf-8")


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


def _build_llm_adapter(comp: PipelineConfig) -> Any:
    """Instancie un adaptateur LLM depuis la config d'un concurrent."""
    if comp.llm_provider == "openai":
        from picarones.adapters.llm.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(model=comp.llm_model or None)
    elif comp.llm_provider == "anthropic":
        from picarones.adapters.llm.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(model=comp.llm_model or None)
    elif comp.llm_provider == "mistral":
        from picarones.adapters.llm.mistral_adapter import MistralAdapter
        return MistralAdapter(model=comp.llm_model or None)
    elif comp.llm_provider == "ollama":
        from picarones.adapters.llm.ollama_adapter import OllamaAdapter
        return OllamaAdapter(model=comp.llm_model or None)
    else:
        raise ValueError(f"Provider LLM inconnu : {comp.llm_provider}")


def _sanitize_name_suffix(value: str) -> str:
    """Réduit ``value`` à un suffixe d'identifiant alphanum + ``_-``.

    Les adapters OCR canoniques (``TesseractAdapter`` etc.) valident
    ``name`` contre ce charset au constructeur — on doit pré-sanitizer
    avant de leur passer un name dérivé d'``ocr_model`` qui peut
    contenir ``.``, ``:``, espaces, etc.  Exemples :
    ``"mistral-ocr-latest"`` → ``"mistral-ocr-latest"`` (intact),
    ``"prebuilt-read"`` → ``"prebuilt-read"``,
    ``"DOCUMENT_TEXT_DETECTION"`` → idem.
    """
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in value)


def _ocr_adapter_name(engine_id: str, ocr_model: str) -> str:
    """Nom canonique de l'adapter OCR pour un couple ``(engine, model)``.

    Deux ``PipelineConfig`` qui partagent exactement le même couple
    obtiennent le même ``name`` (donc le resolver les déduplique
    proprement).  Deux configs différentes obtiennent des noms
    distincts — pas de collision silencieuse, pas de bricolage côté
    resolver.

    Convention : ``{engine_id}_{model_sanitized}`` quand ``model`` est
    non vide ; sinon ``{engine_id}`` seul (cas de l'engine OCR seul
    en mode corpus ou avec model par défaut implicite).
    """
    if not ocr_model:
        return engine_id
    suffix = _sanitize_name_suffix(ocr_model)
    if not suffix:
        return engine_id
    return f"{engine_id}_{suffix}"


#: Sprint S9 — registry centralisée des engines OCR supportés par
#: l'UI web.  Chaque entrée mappe ``engine_id`` → fonction qui
#: transforme l'``ocr_model`` reçu de l'UI en dict de kwargs pour
#: ``ocr_adapter_from_name(engine_id, **kwargs)``.
#:
#: Pourquoi une registry plutôt que des elif
#: -----------------------------------------
#: Avant S9, la fonction ``_engine_from_competitor`` avait 4
#: branches ``elif`` qui répétaient le pattern
#: ``ocr_adapter_from_name(engine_id, name=adapter_name, ...)``.
#: Si un dev ajoutait une 5e branche en oubliant ``name=...``, le
#: bug de collision resolver Tesseract pouvait revenir pour le
#: nouveau moteur.  Avec la registry :
#:
#: 1. Le ``name`` est injecté automatiquement par la fonction
#:    appelante — il n'est plus possible de l'oublier dans une
#:    branche.
#: 2. Le test paramétré ``test_ocr_kwargs_for_*`` itère cette
#:    table directement → ajouter un engine sans test associé
#:    est impossible (le test échoue immédiatement sur la nouvelle
#:    entrée).
_OCR_KWARGS_BUILDERS: dict[str, Any] = {
    "tesseract": lambda model: {
        "lang": model or "fra",
        "psm": 6,
    },
    "pero_ocr": lambda model: {
        "config_path": model or "",
    },
    # Phase 3 chantier post-rewrite : kraken/calamari étaient annoncés
    # par ``/api/engines`` mais sans factory branchée → benchmark web
    # échouait silencieusement.  Le ``ocr_model`` côté UI véhicule
    # désormais le chemin du modèle (Kraken ``.mlmodel`` ou Calamari
    # checkpoint).  Si vide, l'adapter lève une OCRAdapterError
    # explicite à ``execute`` — pas de fallback silencieux.
    "kraken": lambda model: {
        "model_path": model or "",
    },
    "calamari": lambda model: {
        "checkpoint": model or "",
    },
    "mistral_ocr": lambda model: {
        "model": model or "mistral-ocr-latest",
    },
    "google_vision": lambda model: {
        "feature_type": (model or "DOCUMENT_TEXT_DETECTION").upper(),
    },
    "azure_doc_intel": lambda model: {
        "model_id": model or "prebuilt-read",
    },
}


def _build_ocr_kwargs(engine_id: str, ocr_model: str) -> dict[str, Any]:
    """Construit le dict complet de kwargs pour
    ``ocr_adapter_from_name(engine_id, **kwargs)`` à partir de
    la config UI ``(engine_id, ocr_model)``.

    Le ``name`` est dérivé via ``_ocr_adapter_name`` et toujours
    inclus — c'est la garantie systémique que deux competitors
    avec des configs distinctes auront des names distincts au
    resolver (cf. Sprint S9 — bug Tesseract collision).
    """
    builder = _OCR_KWARGS_BUILDERS.get(engine_id)
    if builder is None:
        raise ValueError(f"Moteur OCR inconnu : {engine_id}")
    kwargs = builder(ocr_model)
    kwargs["name"] = _ocr_adapter_name(engine_id, ocr_model)
    return kwargs


def _engine_from_competitor(comp: PipelineConfig) -> Any:
    """Instancie un moteur OCR (ou pipeline OCR+LLM) depuis une PipelineConfig.

    Modes supportés :

    - ``ocr_engine`` = ``tesseract``, ``mistral_ocr``, … → moteur OCR seul.
    - ``ocr_engine`` + ``llm_provider`` → pipeline OCR live + LLM.
    - ``ocr_engine`` = ``corpus`` + ``llm_provider`` → post-correction LLM
      avec OCR pré-calculé (fichiers ``.ocr.txt`` du corpus triplet).
    - ``ocr_engine`` = ``""`` + ``llm_provider`` → LLM seul (zero-shot
      ou post-correction).
    """
    engine_id = comp.engine_name
    is_corpus_ocr = engine_id in ("corpus", "")

    if is_corpus_ocr and not comp.llm_provider:
        raise ValueError(
            "engine_name='corpus' nécessite un llm_provider "
            "(pour la post-correction ou le zero-shot)"
        )

    # Sprint H.2.b.4 — instanciation OCR via la factory canonique
    # ``ocr_adapter_from_name`` (retourne ``BaseOCRAdapter``) au lieu
    # des constructeurs ``BaseOCREngine`` legacy.  Les adapters
    # canoniques ont des kwargs nommés (pas de dict ``config``) — la
    # conversion se fait ici en respectant les noms historiques des
    # champs ``PipelineConfig.ocr_model``.
    ocr = None
    if not is_corpus_ocr:
        from picarones.adapters.ocr.factory import ocr_adapter_from_name

        # Sprint S9 — dispatch uniforme via ``_OCR_KWARGS_BUILDERS``.
        # Le ``name`` est dérivé systématiquement de
        # ``(engine_id, ocr_model)`` par ``_build_ocr_kwargs`` — il
        # n'est plus possible de l'oublier pour un nouveau moteur.
        try:
            kwargs = _build_ocr_kwargs(engine_id, comp.ocr_model)
            # Phase B3-final corr-B (mai 2026) — propage expose_alto
            # à Tesseract (les autres adapters ignorent ce kwarg via
            # validation du factory).
            if comp.expose_alto and engine_id.lower() in {"tesseract", "tess"}:
                kwargs["expose_alto"] = True
            ocr = ocr_adapter_from_name(engine_id, **kwargs)
        except ValueError as exc:
            # Adapter indisponible (dépendance optionnelle absente)
            # → message utilisateur, comme avant la migration.
            raise RuntimeError(str(exc)) from exc

        if not comp.llm_provider:
            return ocr

    # Pipeline OCR+LLM (live ou post-correction) — ``OCRLLMPipelineConfig``
    # canonique remplace l'ex-``OCRLLMPipeline`` legacy.
    #
    # Phase 2 chantier post-rewrite : suppression de l'ancien ``mode_map``
    # qui aliasait silencieusement (``post_correction_text`` →
    # ``text_only``, valeur inconnue → ``text_only``).  Désormais le
    # typage Pydantic ``PipelineMode`` rejette en 422 toute chaîne hors
    # de la matrice {``text_only``, ``text_and_image``, ``zero_shot``},
    # et un éventuel client API qui passerait outre la validation
    # (test legacy, payload forgé) reçoit ici une ``ValueError``.
    mode = comp.pipeline_mode
    if mode not in ("text_only", "text_and_image", "zero_shot"):
        raise ValueError(
            f"pipeline_mode invalide : {comp.pipeline_mode!r}.  "
            "Valeurs acceptées : 'text_only', 'text_and_image', 'zero_shot'.",
        )

    llm = _build_llm_adapter(comp)

    from picarones.pipeline.llm_pipeline_config import OCRLLMPipelineConfig

    # Le ``prompt_file`` reçu de l'UI est un NOM de fichier ; le
    # pipeline canonique attend le CONTENU du prompt (string brute).
    # On charge ici, sinon le LLM reçoit le filename comme prompt
    # et répond avec du méta-discours au lieu de corriger l'OCR.
    prompt_filename = comp.prompt_file or "correction_medieval_french.txt"
    prompt_content = _load_prompt_content(prompt_filename)

    if is_corpus_ocr:
        pipeline_name = comp.name or f"corpus_ocr → {comp.llm_model or comp.llm_provider}"
    else:
        pipeline_name = comp.name or f"{engine_id} → {comp.llm_model or comp.llm_provider}"

    return OCRLLMPipelineConfig(
        ocr_adapter=ocr,
        llm_adapter=llm,
        mode=mode,
        prompt_template=prompt_content,
        pipeline_name=pipeline_name,
    )


def run_benchmark_thread_v2(job: BenchmarkJob, req: BenchmarkRunRequest) -> None:
    """Exécute un benchmark à partir d'une liste de ``PipelineConfig``."""
    job.set_status("running")
    job.started_at = iso_now()
    job.add_event("start", {"message": "Démarrage du benchmark…", "corpus": req.corpus_path})

    try:
        import tempfile
        from pathlib import Path

        from picarones.app.services import (
            RunOrchestrator,
            prepare_preset_args,
            run_result_to_benchmark_result,
        )
        from picarones.evaluation.corpus import load_corpus_from_directory

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
                    "message": f"Concurrent ignoré '{comp.name or comp.engine_name}' : {exc}"
                })

        if not engines:
            raise ValueError("Aucun concurrent valide disponible.")

        # Sprint A14-S1 — A.I.0 P0 : ``output_dir`` a déjà été validé
        # par le router (validated_path).  ``report_name`` est sanitizé
        # ici pour défense en profondeur (refuse ``../``, séparateurs,
        # caractères de contrôle) avant concaténation à output_dir.
        from picarones.interfaces.web.security import safe_report_name
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

        from picarones.evaluation.metrics.normalization import _parse_exclude_chars
        char_excl = _parse_exclude_chars(req.char_exclude) if req.char_exclude else None

        # Phase B3-final migration Option B (2026-05) — pattern 3
        # étapes explicite (Option 10) :
        #     1. prepare_preset_args     (conversion vers domain)
        #     2. execute_preset          (run du benchmark)
        #     3. run_result_to_benchmark_result (BenchmarkResult legacy)
        # Pattern explicite — chaque étape est unitairement testable.
        with tempfile.TemporaryDirectory(prefix="picarones_web_") as _ws:
            _ws_path = Path(_ws)
            _run_dir = _ws_path / "run"
            # Phase B3-final corr-A/B/C (mai 2026) — propage les
            # nouveaux champs ``BenchmarkRunRequest`` (views, profile,
            # partial_dir, entity_extractor, output_json).
            _views_tuple = tuple(req.views) if req.views else ("text_final",)
            _preset = prepare_preset_args(
                corpus, engines,
                workspace_dir=_ws_path / "gt",
                output_dir=_run_dir,
                views=_views_tuple,
                char_exclude=char_excl,
                normalization_profile=req.normalization_profile,
                profile=req.profile,
                partial_dir=req.partial_dir or None,
                entity_extractor=req.entity_extractor or None,
                output_json=req.output_json or output_json,
            )
            _orch_result = RunOrchestrator(_run_dir).execute_preset(
                spec=_preset.spec,
                corpus_spec=_preset.corpus_spec,
                extracted_dir=_preset.extracted_dir,
                pipeline_specs=_preset.pipeline_specs,
                adapter_resolver=_preset.adapter_resolver,
                adapter_kwargs=_preset.adapter_kwargs,
                progress_callback=_progress_callback,
                cancel_event=job._cancel_event,
            )
            result = run_result_to_benchmark_result(
                _orch_result.run_result,
                corpus=corpus, engines=engines,
                char_exclude=char_excl,
                normalization_profile=req.normalization_profile,
                profile=req.profile,
            )

        if job.status == "cancelled":
            return

        job.add_event("log", {"message": "Génération du rapport HTML…"})
        from picarones.reports.html.generator import ReportGenerator
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


# ──────────────────────────────────────────────────────────────────────
# Phase 4.2 audit code-quality (2026-05) — ``_legacy_request_to_run_request``
# et ``run_benchmark_thread`` supprimés avec l'endpoint
# ``POST /api/benchmark/start``.  Les clients utilisent désormais
# ``POST /api/benchmark/run`` avec ``BenchmarkRunRequest`` directement.
# Rupture API documentée dans CHANGELOG v2.0.
# ──────────────────────────────────────────────────────────────────────


__all__ = [
    "sse_format",
    "run_benchmark_thread_v2",
]
