"""Helpers stateless du ``RunOrchestrator``.

Extraits de ``run_orchestrator.py`` (god-module, audit prod P1) :
fonctions et proxy *sans état* — aucune dépendance vers
``RunOrchestrator`` ni vers l'état d'instance.  Réimportés par
``run_orchestrator`` pour préserver l'API (``from
picarones.app.services.run_orchestrator import _default_gt_factory``
reste valide ; les call sites internes utilisent toujours le nom
module-global, donc ``monkeypatch.setattr(run_orchestrator, …)``
fonctionne aussi).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

from picarones.app.services.corpus_service import CorpusImportError
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.documents import DocumentRef
from picarones.formats.alto.parser import parse_alto
from picarones.pipeline import RunContext

logger = logging.getLogger(__name__)


class _PipelineEngineProxy:
    """Proxy léger ``PipelineSpec → engine`` pour le converter legacy.

    Phase B2.7 — quand on persiste un ``BenchmarkResult`` legacy via
    le converter ``run_result_to_benchmark_result``, ce dernier attend
    une liste d'``engines`` qui exposent ``.name`` et ``.config`` (le
    modèle mental legacy est OCR adapter / OCRLLMPipeline).

    Le ``RunOrchestrator`` raisonne en termes de ``PipelineSpec``
    (déclaration YAML), pas d'instances d'engines.  Ce proxy fournit
    juste le contrat minimal pour que le converter produise un
    ``EngineReport`` cohérent — sans introduire de couplage entre
    ``run_orchestrator`` et le code legacy.

    Le ``.name`` exposé est celui de la pipeline (``"ocr_then_correct"``)
    et non du premier step (``"tesseract"``) — pour que l'``EngineReport``
    porte le nom canonique du pipeline.
    """

    __slots__ = ("_spec",)

    def __init__(self, pipeline_spec: Any) -> None:
        self._spec = pipeline_spec

    @property
    def name(self) -> str:
        return str(self._spec.name)

    @property
    def config(self) -> dict[str, Any]:
        """Config sérialisable : noms des steps + leurs types I/O.

        Permet aux consommateurs du ``BenchmarkResult`` legacy de
        savoir quel pipeline a produit chaque ``EngineReport`` sans
        connaître les détails d'implémentation des adapters.
        """
        return {
            "pipeline_name": self._spec.name,
            "steps": [
                {
                    "id": step.id,
                    "input_types": sorted(t.value for t in step.input_types),
                    "output_types": sorted(t.value for t in step.output_types),
                }
                for step in self._spec.steps
            ],
        }


def _resolve_entity_extractor(
    dotted_path: str,
) -> Callable[[str], list[dict]] | None:
    """Phase B2.4 — résout un dotted path vers un extracteur d'entités.

    Format attendu (validé en B1.1 via ``_DOTTED_PATH_RE`` du
    ``RunSpec``) :

    - ``module.submodule:Symbol`` (PEP 621 entry points / setuptools)
    - ``module.submodule.Symbol`` (import classique)

    Le symbole résolu doit être soit :

    - une **factory zéro-arg** qui retourne un callable ``(text: str)
      -> list[dict]`` (pattern legacy CLI : ``SpacyEntityExtractor``
      avec config par défaut),
    - soit directement un callable ``(text: str) -> list[dict]``
      (pattern test : fonction mock).

    On essaie d'abord d'appeler le symbole sans argument ; si ça
    renvoie un callable, on l'utilise.  Sinon, on suppose que le
    symbole est déjà un callable.

    Returns
    -------
    Callable ou ``None`` si la résolution échoue.  Un échec ne
    casse pas le bench (warning loggé, NER skippé) — cohérent avec
    le legacy ``_attach_ner_metrics_to_benchmark`` qui dégrade
    proprement.
    """
    import importlib

    # Normalise le séparateur final : ``:`` ou ``.`` indifféremment.
    if ":" in dotted_path:
        module_path, _, symbol_name = dotted_path.rpartition(":")
    else:
        module_path, _, symbol_name = dotted_path.rpartition(".")

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        logger.warning(
            "[run_orchestrator] entity_extractor : module %r introuvable "
            "(%s) — NER sauté pour ce run.",
            module_path, exc,
        )
        return None

    symbol = getattr(module, symbol_name, None)
    if symbol is None:
        logger.warning(
            "[run_orchestrator] entity_extractor : symbole %r absent de %r "
            "— NER sauté pour ce run.",
            symbol_name, module_path,
        )
        return None

    # Pattern legacy : si ``symbol`` est une factory (classe ou
    # fonction zéro-arg), l'instancier.  Sinon, l'utiliser tel quel.
    if callable(symbol):
        try:
            candidate = symbol()
            if callable(candidate):
                return candidate
            # ``symbol()`` retourne autre chose qu'un callable —
            # ``symbol`` est probablement déjà la fonction d'extraction.
            return symbol
        except TypeError:
            # ``symbol`` n'accepte pas zéro-arg : c'est probablement
            # la fonction d'extraction directe.
            return symbol

    logger.warning(
        "[run_orchestrator] entity_extractor : %r n'est pas callable.",
        dotted_path,
    )
    return None


def _kwargs_signature(kwargs: dict[str, Any]) -> str:
    """Signature stable d'un dict de kwargs (ordre tri-stable)."""
    return "|".join(f"{k}={kwargs[k]!r}" for k in sorted(kwargs))


def _default_gt_factory(
    doc: DocumentRef, art_type: ArtifactType,
) -> Artifact | None:
    """Factory GT par défaut.

    Convention : un candidat ``CORRECTED_TEXT`` est comparé contre
    la GT ``RAW_TEXT`` (les deux sont du texte plat — la distinction
    de type ne porte que sur le côté candidat).  Cas typique : un
    pipeline OCR + post-correction LLM produit un ``CORRECTED_TEXT``
    qu'on compare au ``.gt.txt`` original.
    """
    effective_type = (
        ArtifactType.RAW_TEXT
        if art_type == ArtifactType.CORRECTED_TEXT
        else art_type
    )
    gt_ref = doc.gt_for(effective_type)
    if gt_ref is None:
        return None
    return Artifact(
        id=f"{doc.id}:gt:{effective_type.value}",
        document_id=doc.id,
        type=effective_type,
        uri=gt_ref.uri,
    )


def _default_inputs_factory(doc: DocumentRef) -> dict[ArtifactType, Artifact]:
    """``{IMAGE: artifact_image}``.  Lève si ``doc.image_uri`` absent."""
    if doc.image_uri is None:
        raise CorpusImportError(
            f"Document {doc.id!r} sans ``image_uri`` — la pipeline "
            "par défaut consomme une IMAGE en entrée.",
        )
    return {ArtifactType.IMAGE: Artifact(
        id=f"{doc.id}:image",
        document_id=doc.id,
        type=ArtifactType.IMAGE,
        uri=doc.image_uri,
    )}


def _make_context_factory(
    code_version: str,
    *,
    progress_callback: Callable[[str, int, str], None] | None = None,
    workspace_uri: str | None = None,
) -> Callable[[DocumentRef, str], RunContext]:
    """Phase B2.1 — factory de ``RunContext`` avec callback de progression.

    Pattern strictement copié de
    ``_benchmark_execution.py:109-139`` (legacy) pour garantir
    l'équivalence numérique du compteur ``doc_idx`` à
    ``run_benchmark_via_service``.

    Le ``counter_lock`` partagé empêche les race conditions quand le
    ``CorpusRunner`` traite plusieurs documents en parallèle
    (``max_in_flight > 1``).  Le compteur est **global au run**, pas
    par pipeline — un benchmark à 2 pipelines × 5 docs émet 10
    notifications ``doc_idx ∈ {0..9}``.

    Phase B4 — ``workspace_uri`` propagé au ``RunContext`` pour que les
    adapters qui en ont besoin (PrecomputedTextAdapter,
    TesseractAdapter, etc.) puissent écrire leurs artefacts
    intermédiaires.  Cohérent avec ``_benchmark_execution.py:134-139``.
    """
    counter_lock = threading.Lock()
    counter_state = {"doc_idx": 0}

    def _factory(doc: DocumentRef, pipeline_name: str) -> RunContext:
        if progress_callback is not None:
            with counter_lock:
                idx = counter_state["doc_idx"]
                counter_state["doc_idx"] = idx + 1
            try:
                progress_callback(pipeline_name, idx, doc.id)
            except Exception:
                # On ignore silencieusement les erreurs du callback :
                # un caller qui crashe ne doit pas faire tomber le
                # benchmark.  Cohérent avec
                # ``_benchmark_execution.py:126-133``.
                import logging
                logging.getLogger(__name__).debug(
                    "[run_orchestrator] progress_callback levé — "
                    "ignoré pour ne pas tomber le bench.",
                    exc_info=True,
                )
        return RunContext(
            document_id=doc.id,
            code_version=code_version,
            pipeline_name=pipeline_name,
            workspace_uri=workspace_uri,
        )
    return _factory


def _filesystem_payload_loader(art: Artifact) -> Any:
    """Loader filesystem : lit RAW_TEXT/CORRECTED_TEXT depuis le
    fichier pointé par l'URI, parse ALTO_XML depuis le fichier pointé.

    Les artefacts projetés (sans URI) ne passent pas par ce loader —
    l'executor utilise directement le payload retourné par le
    projecteur.
    """
    if art.uri is None:
        raise FileNotFoundError(
            f"Loader filesystem : artifact {art.id!r} sans URI ; "
            "un projecteur aurait dû fournir le payload.",
        )
    path = Path(art.uri)
    if art.type == ArtifactType.ALTO_XML:
        return parse_alto(path.read_bytes())
    if art.type in (ArtifactType.RAW_TEXT, ArtifactType.CORRECTED_TEXT):
        return path.read_text(encoding="utf-8")
    raise ValueError(
        f"Loader filesystem : type {art.type.value!r} non géré.",
    )


def _persist_legacy_benchmark_json(
    *,
    run_result: Any,
    extracted_dir: Path,
    pipeline_specs: list[Any],
    corpus_name: str,
    output_json: Path,
    char_exclude: str | None,
    normalization_profile: str | None,
    profile: str,
    entity_extractor: str | None = None,
    corpus_legacy: Any | None = None,
) -> None:
    """Phase B2.7 — converti ``RunResult`` → ``BenchmarkResult`` legacy
    et persiste en JSON.

    Délègue à
    :func:`picarones.app.services._benchmark_converter.run_result_to_benchmark_result`
    (utilisé aussi par ``run_benchmark_via_service``) pour
    garantir l'équivalence numérique du format de sortie.

    Le caller fournit :

    - ``run_result`` : le ``RunResult`` produit par le ``BenchmarkService``.
    - ``extracted_dir`` : où le corpus a été extrait — sert à
      recharger un ``Corpus`` legacy via
      ``load_corpus_from_directory`` **quand** ``corpus_legacy``
      n'est pas fourni (mode ``execute()`` avec extraction réelle
      d'un zip/dir).  Le converter attend des ``Document`` legacy
      avec ``image_path`` et ``ground_truth``.
    - ``pipeline_specs`` : la liste des pipelines exécutées, dans
      l'ordre soumis à ``BenchmarkService.run``.  Chaque spec est
      wrappée en ``_PipelineEngineProxy`` qui expose le contrat
      minimal attendu par le converter (``name``, ``config``).
    - ``output_json`` : chemin de sortie ; les répertoires parents
      sont créés.
    - ``char_exclude``, ``normalization_profile``, ``profile`` :
      paramètres legacy propagés au converter (qui les passe à
      ``compute_metrics`` et aux hooks document-level).
    - ``corpus_legacy`` *(Phase B3-final hotfix mai 2026)* : Corpus
      legacy déjà en mémoire.  Quand fourni, court-circuite le
      ``load_corpus_from_directory(extracted_dir)`` qui échoue dans
      le path ``execute_preset`` : en mode preset, ``extracted_dir``
      pointe vers le ``workspace_dir`` qui ne contient que les
      ``.gt.txt`` synthétisés par ``document_to_document_ref``, pas
      les images sources — ``load_corpus_from_directory`` itère
      alors sur zéro image et lève ``ValueError: Aucun document
      valide trouvé``.  Symptôme observé en prod : le benchmark
      web/CLI échouait silencieusement après la 1re exécution OCR
      avec ce message trompeur.

    Notes
    -----
    Le format produit est strictement identique à celui de
    ``run_benchmark_via_service(output_json=...)`` (testé via le
    snapshot d'invariance ``test_migration_invariance.py``).
    """
    from picarones.app.services._benchmark_converter import (
        run_result_to_benchmark_result,
    )
    from picarones.app.services._benchmark_persistence import (
        persist_benchmark_result_json,
    )

    if corpus_legacy is not None:
        # Mode preset : le caller a déjà le ``Corpus`` en mémoire
        # (typiquement chargé depuis ``uploads/`` côté web ou via
        # ``load_corpus_from_directory(corpus_arg)`` côté CLI).
        # Pas de reload — évite la divergence ``extracted_dir`` ≠
        # vrai source dir documentée plus haut.
        corpus = corpus_legacy
    else:
        from picarones.evaluation.corpus import load_corpus_from_directory

        # Mode ``execute()`` classique : le corpus est physiquement
        # disponible dans ``extracted_dir`` (zip extrait ou dossier
        # source).  ``name`` passé explicitement pour matcher
        # ``corpus_spec.name`` (sinon le loader retourne
        # ``"Corpus"`` par défaut, ce qui casserait le snapshot
        # d'invariance).
        try:
            corpus = load_corpus_from_directory(
                extracted_dir, name=corpus_name,
            )
        except (ValueError, FileNotFoundError) as exc:
            # Audit B3-final mai 2026, trou #9 : si ``extracted_dir``
            # est en fait un ``workspace_dir`` synthétisé par
            # ``prepare_preset_args`` (= gt-only, pas d'images), le
            # reload lève ``ValueError: Aucun document valide
            # trouvé`` — message cryptique qui masque le vrai
            # problème (caller direct à ``execute_preset(...,
            # output_json=set)`` sans passer ``corpus_legacy``).
            # On enrichit le message pour pointer le caller.
            raise ValueError(
                "_persist_legacy_benchmark_json : impossible de "
                f"reloader le corpus depuis {extracted_dir!r}.\n"
                "Si vous êtes en mode preset (corpus chargé en "
                "mémoire avant ``execute_preset()``), passer "
                "``corpus_legacy=corpus`` à ``execute_preset()`` "
                "pour éviter ce reload — le ``workspace_dir`` "
                "synthétisé par ``prepare_preset_args`` ne "
                "contient que les .gt.txt, pas les images "
                f"sources.\nErreur originale : {exc}",
            ) from exc

    # Wrappe chaque PipelineSpec en proxy minimal pour le converter.
    # Le converter ne consomme que ``.name``, ``.config`` et tolère
    # l'absence de ``.version`` (cf. ``_safe_engine_version``).
    engines = [_PipelineEngineProxy(spec) for spec in pipeline_specs]

    # Phase B2.5 — le converter legacy passe ``normalization_profile``
    # à ``compute_metrics`` qui attend un objet ``NormalizationProfile``,
    # pas une string.  Résolution explicite ici pour aligner avec ce que
    # font les call sites legacy (CLI ``_workflows.py`` via
    # ``resolve_normalization_profile``).  ``char_exclude`` reste string —
    # ``compute_metrics`` le traite comme un set/frozenset implicite.
    resolved_profile: Any = None
    if normalization_profile:
        from picarones.formats.text.normalization import get_builtin_profile
        try:
            resolved_profile = get_builtin_profile(normalization_profile)
        except KeyError:
            # Profil inconnu — on laisse ``None`` (le converter
            # tombera dans son default ``DEFAULT_DIPLOMATIC_PROFILE``).
            # Cohérent avec le legacy qui logge un warning sans
            # casser le run.
            logger.warning(
                "[run_orchestrator] profil normalisation %r inconnu "
                "pour output_json — fallback default diplomatique.",
                normalization_profile,
            )

    benchmark_result = run_result_to_benchmark_result(
        run_result,
        corpus=corpus,
        engines=engines,
        char_exclude=frozenset(char_exclude) if char_exclude else None,
        normalization_profile=resolved_profile,
        profile=profile,
    )

    # Phase B2.4 — NER attach post-process si un entity_extractor
    # est fourni.  Pattern identique à
    # ``run_benchmark_via_service:261-264`` :  on résout le dotted
    # path, on instancie la factory, on attache au BenchmarkResult.
    if entity_extractor:
        extractor_callable = _resolve_entity_extractor(entity_extractor)
        if extractor_callable is not None:
            from picarones.app.services._benchmark_ner import (
                attach_ner_metrics_to_benchmark,
            )
            attach_ner_metrics_to_benchmark(
                benchmark_result, corpus, extractor_callable,
            )

    persist_benchmark_result_json(benchmark_result, output_json)


__all__ = [
    "_PipelineEngineProxy",
    "_default_gt_factory",
    "_default_inputs_factory",
    "_filesystem_payload_loader",
    "_kwargs_signature",
    "_make_context_factory",
    "_persist_legacy_benchmark_json",
    "_resolve_entity_extractor",
]
