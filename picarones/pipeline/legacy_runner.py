"""Banc d'essai de pipelines composées — Sprint 63 (axe B).

Phase 5.C.batch7 — module relocalisé depuis
``picarones.core.pipeline`` vers ``picarones.evaluation.pipeline``.
Shim ``picarones.core.pipeline`` retiré au Lot C (2026-05-07).

Phase 7.B.2 — module relocalisé une seconde fois
------------------------------------------------
``picarones.evaluation.pipeline`` → ``picarones.pipeline.legacy_runner``.
La délégation à :class:`PipelineExecutor` (ci-dessous) exige d'importer
la couche ``pipeline/``, ce que la règle d'architecture concentrique
interdit à ``evaluation/`` (whitelist externe restreinte, pas de
dépendance sortante vers une couche plus externe — cf. CLAUDE.md
§ "architecture des couches").  Le module bridge legacy ↔ canonique
vit donc dans la couche ``pipeline/``.  ``picarones.evaluation.pipeline``
reste exposé en re-export shim le temps que les callers historiques
migrent.

Phase 7.B.2 — délégation au ``PipelineExecutor`` canonique
----------------------------------------------------------
Depuis 2026-05, ``PipelineRunner.run`` ne porte **plus** sa propre
boucle d'exécution.  Le corps de la méthode délègue intégralement à
:class:`picarones.pipeline.executor.PipelineExecutor` via le wrapper
:class:`picarones.pipeline._legacy_module_adapter._BaseModuleAdapter`
(créé en 7.B.1).  Le runner ne conserve que :

1. La validation amont legacy (préservation des messages d'erreur
   français du Sprint 63 — ``"étape N (X) demande Y qui n'est ni…"``).
2. La traduction des résultats canoniques (``pipeline.types.StepResult``
   Pydantic) vers les types legacy (``StepResult``, ``PipelineResult``
   dataclass) attendus par les ~440 tests existants.
3. Le calcul des ``junction_metrics`` aux jonctions GT-vs-sortie —
   le canonique laisse cette responsabilité au caller (`MetricRegistry`
   intégré au planner mais évaluation déférée).

Cela élimine la duplication de moteur d'exécution (un seul code
path) tout en préservant intégralement l'API publique du Sprint 63
le temps que la sub-phase 7.C migre les tests vers le canonique
direct, puis 7.D supprime le runner legacy.

Sprint 63 — Étape 4 / axe B du plan d'évolution 2026 : démarrage du
banc d'essai de pipelines.

Philosophie
-----------
Picarones est un **banc d'essai**, pas un atelier de production.
Cette infrastructure permet d'**évaluer des pipelines composées de
modules tiers** que l'utilisateur amène — par exemple :

- ``[OCR(image→texte)] → [reconstructeur ALTO tiers(texte→ALTO)]``
- ``[VLM(image→ALTO)] → [post-processing tiers(ALTO→ALTO)]``
- ``[OCR(image→texte)] → [LLM correcteur(texte→texte)]``

Picarones **ne fournit aucun module métier** (pas de
reconstructeur ALTO, pas de correcteur, pas de re-segmenteur).
L'utilisateur branche ses propres ``BaseModule`` (Sprint 33), le
runner orchestre l'exécution séquentielle, valide les types aux
jonctions et **évalue automatiquement** chaque artefact produit
contre la GT du même niveau (Sprint 32) en sélectionnant les
métriques pertinentes du registre typé (Sprint 34).

Périmètre Sprint 63
-------------------
Inclus :

- Spécification déclarative d'une pipeline séquentielle.
- Exécution sur un seul document avec passage typé d'artefacts.
- Validation des types aux jonctions inter-modules.
- Évaluation automatique aux jonctions GT-vs-sortie pour chaque
  niveau de GT disponible sur le document.
- Mesure du temps par étape.
- Capture gracieuse des erreurs (un module qui lève n'arrête pas
  les étapes suivantes — leur entrée manquante est rapportée
  comme erreur explicite).

Reporté à des sprints dédiés :

- DAG branchant non séquentiel (1 → {2, 3} → 4) — Sprint 64+.
- Orchestration corpus-wide + agrégation par pipeline — Sprint 65+.
- Vue HTML dédiée aux pipelines composées — Sprint 66+.
- Cache d'artefacts intermédiaires — non prévu.
- Parallélisation inter-étapes — non prévue (les modules
  ``execution_mode`` sont déjà respectés par le runner historique
  pour le bench OCR mono-étage).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from picarones.evaluation.corpus import Document, GTLevel
from picarones.evaluation.metric_registry import compute_at_junction
from picarones.domain.artifacts import ArtifactType
from picarones.domain.documents import DocumentRef
from picarones.domain.module_protocol import BaseModule
from picarones.domain.pipeline_spec import (
    PipelineSpec as _DomainPipelineSpec,
    PipelineStep as _DomainPipelineStep,
)
from picarones.pipeline._legacy_module_adapter import (
    _BaseModuleAdapter,
    _PayloadRegistry,
    wrap_initial_inputs,
)
from picarones.pipeline.executor import PipelineExecutor
from picarones.pipeline.types import (
    RunContext,
    StepResult as _CanonicalStepResult,
)

# Sprint A3 (renforce la règle Cercle 1 → Cercle 1 uniquement) — la
# cérémonie d'eager-load des métriques typées (Sprint 34) qui vivait
# ici a été déplacée dans ``picarones/measurements/__init__.py``. Tout
# consommateur de ``compute_at_junction`` (typiquement la classe
# ``PipelineRunner`` ci-dessous) doit avoir importé
# ``picarones.measurements`` au moins une fois — c'est le cas dans
# l'API publique via ``picarones.__init__`` qui déclenche le trigger.

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Conversion ArtifactType <-> GTLevel
# ──────────────────────────────────────────────────────────────────────────


#: Map ``ArtifactType`` canonique → ``GTLevel`` legacy.  Phase 4-bis :
#: ``ArtifactType`` a été migré vers ``domain/artifacts.py`` qui
#: distingue ``RAW_TEXT``/``CORRECTED_TEXT`` (vs ``TEXT`` legacy) et
#: ``ALTO_XML``/``PAGE_XML`` (vs ``ALTO``/``PAGE`` legacy).  Les
#: valeurs canoniques ne matchent donc plus celles de ``GTLevel``.
#: Ce mapping explicite fait le pont — sera retiré en 2.0 quand
#: ``GTLevel`` aura aussi été retiré au profit de la projection
#: ``ArtifactType → niveau d'évaluation`` du rewrite.
_ARTIFACT_TO_GT_LEVEL: dict[ArtifactType, GTLevel] = {
    ArtifactType.RAW_TEXT: GTLevel.TEXT,
    ArtifactType.CORRECTED_TEXT: GTLevel.TEXT,
    ArtifactType.ALTO_XML: GTLevel.ALTO,
    ArtifactType.PAGE_XML: GTLevel.PAGE,
    ArtifactType.ENTITIES: GTLevel.ENTITIES,
    ArtifactType.READING_ORDER: GTLevel.READING_ORDER,
}


def _artifact_type_to_gt_level(at: ArtifactType) -> Optional[GTLevel]:
    """Retourne le ``GTLevel`` correspondant à un ``ArtifactType``.

    ``IMAGE`` n'a pas de correspondance GT (on n'évalue pas une
    image en sortie d'un module — c'est typiquement une entrée).
    Les types ``CONFIDENCES``, ``ALIGNMENT``, ``CANONICAL_DOCUMENT``
    n'ont pas non plus de niveau de GT direct dans le legacy.
    """
    return _ARTIFACT_TO_GT_LEVEL.get(at)


# ──────────────────────────────────────────────────────────────────────────
# PipelineStep + PipelineSpec
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class PipelineStep:
    """Une étape dans une pipeline composée.

    L'étape porte un nom lisible (utile pour le rapport et le
    diagnostic) et une instance de ``BaseModule`` fournie par
    l'utilisateur.  Les types d'entrée et de sortie ne sont pas
    redéclarés ici : ils sont lus depuis le module lui-même
    (``module.input_types`` / ``module.output_types``).

    Sprint 66 — DAG branchant
    -------------------------
    ``inputs_from`` permet de désigner explicitement, pour chaque
    type d'entrée, l'étape source dont on veut consommer l'artefact.
    Utile quand plusieurs étapes antérieures produisent le même
    type et qu'on veut éviter l'écrasement implicite (par exemple
    deux correcteurs LLM en parallèle qui partent du même OCR).

    - ``inputs_from = {}`` (défaut) : pour chaque type d'entrée,
      le runner prend la version **la plus récente** disponible
      dans le bag (comportement Sprint 63, rétrocompat stricte).
    - ``inputs_from = {ArtifactType.TEXT: "ocr"}`` : exige la
      version du ``TEXT`` produite par l'étape nommée ``"ocr"``.
      Si cette étape n'existe pas ou n'a pas produit ce type,
      ``PipelineSpec.validate`` remonte un problème explicite et
      le runner remonte une erreur d'entrée manquante.

    La chaîne spéciale ``"__initial__"`` désigne les artefacts
    fournis dans ``initial_inputs`` (par exemple ``IMAGE``).
    """

    name: str
    module: BaseModule
    inputs_from: dict[ArtifactType, str] = field(default_factory=dict)

    @property
    def input_types(self) -> tuple[ArtifactType, ...]:
        return tuple(self.module.input_types)

    @property
    def output_types(self) -> tuple[ArtifactType, ...]:
        return tuple(self.module.output_types)

    def __repr__(self) -> str:
        ins = ",".join(t.value for t in self.input_types) or "·"
        outs = ",".join(t.value for t in self.output_types) or "·"
        if self.inputs_from:
            refs = ",".join(
                f"{t.value}@{src}" for t, src in self.inputs_from.items()
            )
            return f"PipelineStep({self.name}: [{refs}] → {outs})"
        return f"PipelineStep({self.name}: {ins} → {outs})"


@dataclass
class PipelineSpec:
    """DAG séquentiel de ``PipelineStep``.

    Sprint 63 — séquentiel uniquement : l'étape ``i+1`` consomme
    les artefacts produits par l'étape ``i`` (et tous les artefacts
    initiaux fournis au runner, par exemple l'image source).

    Le DAG branchant arrive dans un sprint dédié.
    """

    name: str
    steps: list[PipelineStep] = field(default_factory=list)

    def validate(self, initial_inputs: tuple[ArtifactType, ...]) -> list[str]:
        """Vérifie que les types s'enchaînent et retourne la liste
        des problèmes détectés (vide si la pipeline est valide).

        Une pipeline est valide si, pour chaque étape, tous les
        ``input_types`` sont disponibles : soit dans les
        ``initial_inputs`` (typiquement ``IMAGE``), soit produits
        par une étape antérieure.

        Sprint 66 — validation des références ``inputs_from`` :
        si une étape déclare ``inputs_from[type] = "foo"``,
        l'étape ``foo`` doit exister parmi les étapes antérieures
        et avoir ce type dans ses ``output_types``.  La chaîne
        spéciale ``"__initial__"`` désigne les entrées initiales.
        """
        problems: list[str] = []
        if not self.steps:
            problems.append("pipeline vide : au moins une étape est requise")
            return problems
        # Map type → set des steps qui ont produit ce type
        # ("__initial__" pour les entrées initiales) — utilisé pour
        # valider les références ``inputs_from``.
        producers: dict[ArtifactType, set[str]] = {
            t: {"__initial__"} for t in initial_inputs
        }
        # Map step_name → set des types produits, pour la validation
        # des références.
        step_outputs: dict[str, set[ArtifactType]] = {
            "__initial__": set(initial_inputs),
        }
        # Set des types disponibles à un instant t (latest seulement).
        available: set[ArtifactType] = set(initial_inputs)

        for i, step in enumerate(self.steps):
            # 1. Toutes les entrées doivent être disponibles
            missing = [t for t in step.input_types if t not in available]
            if missing:
                miss_str = ",".join(t.value for t in missing)
                problems.append(
                    f"étape {i} ({step.name}) demande {miss_str} "
                    f"qui n'est ni dans les entrées initiales "
                    f"ni produit par une étape antérieure"
                )
            # 2. Vérification des références ``inputs_from``
            for ref_type, ref_step in step.inputs_from.items():
                if ref_type not in step.input_types:
                    problems.append(
                        f"étape {i} ({step.name}) déclare "
                        f"inputs_from[{ref_type.value}]={ref_step!r} "
                        f"mais le module ne consomme pas ce type"
                    )
                    continue
                if ref_step not in step_outputs:
                    problems.append(
                        f"étape {i} ({step.name}) référence "
                        f"inputs_from[{ref_type.value}]={ref_step!r} "
                        f"qui n'est pas une étape antérieure connue"
                    )
                    continue
                if ref_type not in step_outputs[ref_step]:
                    problems.append(
                        f"étape {i} ({step.name}) référence "
                        f"inputs_from[{ref_type.value}]={ref_step!r} "
                        f"mais cette étape ne produit pas ce type"
                    )
            # 3. Mise à jour pour les étapes suivantes
            available.update(step.output_types)
            step_outputs[step.name] = set(step.output_types)
            for out_type in step.output_types:
                producers.setdefault(out_type, set()).add(step.name)
        return problems

    def is_valid(self, initial_inputs: tuple[ArtifactType, ...]) -> bool:
        return not self.validate(initial_inputs)

    def __repr__(self) -> str:
        chain = " → ".join(str(s) for s in self.steps)
        return f"PipelineSpec({self.name}: {chain})"


# ──────────────────────────────────────────────────────────────────────────
# StepResult + PipelineResult
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class StepResult:
    """Résultat de l'exécution d'une étape sur un document.

    Champs
    ------
    step_name:
        Nom de l'étape (cf. ``PipelineStep.name``).
    duration_seconds:
        Temps d'exécution de ``module.process`` mesuré en wall-clock.
    output_types:
        Types effectivement présents dans la sortie (peut être un
        sous-ensemble de ``module.output_types`` si le module a
        omis un type — cas reporté ici comme info pour diagnostic).
    junction_metrics:
        Pour chaque type produit qui correspond à un ``GTLevel``
        dont le document porte une GT : dictionnaire ``{type: dict
        métriques}`` retourné par ``compute_at_junction``.
    error:
        ``None`` si l'étape s'est bien déroulée ; sinon message
        d'erreur (le module a levé, l'entrée est manquante, ou la
        validation des types a échoué).
    """

    step_name: str
    duration_seconds: float
    output_types: tuple[ArtifactType, ...]
    junction_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Map ``{artifact_type_value: {metric_name: value}}``.

    La clé est la valeur string du ``ArtifactType`` (ex. ``"text"``,
    ``"alto"``) et non l'enum lui-même, pour faciliter la
    sérialisation JSON.
    """
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Résultat complet d'une exécution de pipeline sur un document.

    On capture la durée totale, la durée par étape et les
    métriques aux jonctions pour chaque artefact produit qui a une
    GT correspondante.
    """

    pipeline_name: str
    doc_id: str
    steps: list[StepResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    error: Optional[str] = None
    """Erreur fatale au niveau pipeline (ex. validation des types
    en amont avant la première étape).  ``None`` n'implique pas
    qu'aucune étape n'a échoué — voir ``StepResult.error`` pour le
    détail par étape."""

    @property
    def succeeded(self) -> bool:
        """Vrai si la pipeline s'est exécutée jusqu'au bout sans
        qu'aucune étape ne lève d'erreur."""
        if self.error is not None:
            return False
        return all(s.error is None for s in self.steps)

    @property
    def failing_steps(self) -> list[str]:
        """Noms des étapes ayant levé une erreur."""
        return [s.step_name for s in self.steps if s.error is not None]

    def junction_metrics_for(
        self, artifact_type: ArtifactType,
    ) -> Optional[dict[str, Any]]:
        """Retourne les métriques de la **dernière** étape qui a
        produit ``artifact_type``, ou ``None`` si aucune étape ne
        l'a produit avec succès.

        Utile pour comparer plusieurs pipelines qui produisent in
        fine le même type (ex. deux DAG aboutissant à du texte
        corrigé).
        """
        from picarones.domain.artifacts import LEGACY_VALUE_ALIASES
        legacy_alias = LEGACY_VALUE_ALIASES.get(artifact_type.value)
        for step in reversed(self.steps):
            if step.error is not None:
                continue
            metrics = step.junction_metrics.get(artifact_type.value)
            if metrics is None and legacy_alias is not None:
                # Phase 4-bis : un caller legacy peut avoir construit
                # le dict avec la clé pré-rewrite ("text" au lieu de
                # "raw_text").  expand_legacy_keys synchronise les deux
                # côtés sur les sites d'écriture du runner, mais des
                # StepResult construits à la main par les tests ou par
                # un caller externe peuvent encore avoir une seule
                # clé — on tolère.
                metrics = step.junction_metrics.get(legacy_alias)
            if metrics is not None:
                return metrics
        return None


# ──────────────────────────────────────────────────────────────────────────
# Exécuteur
# ──────────────────────────────────────────────────────────────────────────


class PipelineRunner:
    """Exécute une ``PipelineSpec`` sur un document.

    Sprint 63 — un seul document à la fois.  L'orchestration
    corpus-wide et l'agrégation par pipeline sont reportées à un
    sprint dédié.

    Phase 7.B.2 — délégation au canonique
    --------------------------------------
    L'API publique (``run`` statique, types de retour ``PipelineResult``
    et ``StepResult`` legacy, format des messages d'erreur en français)
    est rigoureusement préservée pour rétrocompat.  Le corps de
    ``run`` délègue à :class:`picarones.pipeline.executor.PipelineExecutor`
    via :class:`_BaseModuleAdapter` — il n'y a plus de code de
    boucle d'exécution dupliqué.

    Usage typique
    -------------

    >>> spec = PipelineSpec(
    ...     name="ocr_then_rewrite",
    ...     steps=[
    ...         PipelineStep("ocr", my_ocr_module),
    ...         PipelineStep("rewrite", my_llm_rewriter),
    ...     ],
    ... )
    >>> runner = PipelineRunner()
    >>> result = runner.run(spec, document, {ArtifactType.IMAGE: "/path/img.png"})
    >>> result.succeeded
    True
    >>> result.junction_metrics_for(ArtifactType.TEXT)
    {'cer': 0.05, 'wer': 0.12, ...}
    """

    @staticmethod
    def run(
        spec: PipelineSpec,
        document: Document,
        initial_inputs: dict[ArtifactType, Any],
    ) -> PipelineResult:
        """Exécute ``spec`` sur ``document`` à partir de
        ``initial_inputs``.

        Parameters
        ----------
        spec:
            Spécification de la pipeline.
        document:
            Document du corpus, porteur de zéro ou plusieurs niveaux
            de GT (Sprint 32).
        initial_inputs:
            Artefacts initiaux par type — typiquement
            ``{ArtifactType.IMAGE: "/path/img.png"}`` pour une
            pipeline qui démarre par un OCR.

        Returns
        -------
        PipelineResult
            Résultat complet : durée totale, résultat par étape,
            métriques aux jonctions évaluées contre la GT.
        """
        result = PipelineResult(
            pipeline_name=spec.name, doc_id=document.doc_id,
        )

        # Validation amont legacy : si la pipeline est statiquement
        # invalide, on n'exécute aucune étape.  Cette validation
        # produit des messages français spécifiques au Sprint 63
        # (cf. ``PipelineSpec.validate``) que les tests vérifient ;
        # le canonique a sa propre ``ValidationError`` au format
        # différent — d'où la double validation tant que les tests
        # legacy ne sont pas migrés (sub-phase 7.C).
        problems = spec.validate(tuple(initial_inputs.keys()))
        if problems:
            result.error = " ; ".join(problems)
            return result

        canonical_result, registry = _delegate_to_canonical_executor(
            spec, document, initial_inputs,
        )

        for legacy_step, canonical_sr in zip(
            spec.steps, canonical_result.step_results,
        ):
            result.steps.append(
                _build_legacy_step_result(
                    legacy_step, canonical_sr, registry, document,
                ),
            )
        result.total_duration_seconds = canonical_result.duration_seconds
        return result


# ──────────────────────────────────────────────────────────────────────────
# Phase 7.B.2 — délégation au PipelineExecutor canonique
# ──────────────────────────────────────────────────────────────────────────


def _delegate_to_canonical_executor(
    legacy_spec: PipelineSpec,
    legacy_doc: Document,
    initial_inputs: dict[ArtifactType, Any],
) -> tuple[Any, _PayloadRegistry]:
    """Exécute ``legacy_spec`` via :class:`PipelineExecutor`.

    Construit la ``_DomainPipelineSpec`` canonique équivalente, un
    ``adapter_resolver`` ad-hoc qui mappe ``step.name → _BaseModuleAdapter``,
    et délègue à l'executor.  Retourne le ``PipelineResult`` canonique
    + le registre de payloads (dont le caller a besoin pour reconstruire
    les ``junction_metrics`` du contrat legacy).
    """
    registry = _PayloadRegistry()
    canonical_inputs = wrap_initial_inputs(
        initial_inputs, registry, legacy_doc.doc_id,
    )

    adapter_map: dict[str, _BaseModuleAdapter] = {}
    canonical_steps: list[_DomainPipelineStep] = []
    for step in legacy_spec.steps:
        adapter_map[step.name] = _BaseModuleAdapter(step.module, registry)
        canonical_steps.append(
            _DomainPipelineStep(
                id=step.name,
                kind="legacy_module",
                adapter_name=step.name,
                input_types=tuple(step.input_types),
                output_types=tuple(step.output_types),
                inputs_from=dict(step.inputs_from),
            ),
        )
    canonical_spec = _DomainPipelineSpec(
        name=legacy_spec.name,
        initial_inputs=tuple(initial_inputs.keys()),
        steps=tuple(canonical_steps),
    )

    document_ref = DocumentRef(id=legacy_doc.doc_id)
    # ``code_version`` est libre (str non vide).  Le wrapper
    # ``_BaseModuleAdapter`` ne produit pas de ``ProvenanceRecord``
    # détaillée — la couche pipeline ne peut pas importer
    # ``picarones.__version__`` (whitelist externe restreinte).
    # On étiquette les runs legacy avec un sentinel constant ; la
    # traçabilité fine reviendra avec le canonique pur en 7.D.
    context = RunContext(
        document_id=legacy_doc.doc_id,
        code_version="legacy_runner",
        pipeline_name=legacy_spec.name,
    )
    executor = PipelineExecutor(adapter_resolver=adapter_map.__getitem__)
    canonical_result = executor.run(
        canonical_spec, document_ref, canonical_inputs, context,
    )
    return canonical_result, registry


def _build_legacy_step_result(
    legacy_step: PipelineStep,
    canonical_sr: _CanonicalStepResult,
    registry: _PayloadRegistry,
    document: Document,
) -> StepResult:
    """Reconstruit un ``StepResult`` legacy depuis le canonique.

    Trois responsabilités :

    1. Traduire le format des messages d'erreur (``adapter_raised:``,
       ``missing_input:``, ``missing_output:``) vers le format français
       attendu par les tests legacy (``"Type: msg"``,
       ``"entrée manquante : ..."``, ``"sortie manquante : ..."``).
    2. Reconstruire le tuple ``output_types`` à partir de
       ``produced_artifacts`` du canonique.
    3. Calculer les ``junction_metrics`` en lisant les payloads
       depuis ``registry`` et en appelant ``compute_at_junction``
       contre la GT du document — comportement automatique du
       Sprint 63 que le canonique laisse au caller.
    """
    error = _translate_canonical_error(canonical_sr.error)

    produced_at: list[ArtifactType] = []
    for type_value in canonical_sr.produced_artifacts:
        try:
            produced_at.append(ArtifactType(type_value))
        except ValueError:
            continue

    junction_metrics = _compute_junction_metrics_for_step(
        produced_at, canonical_sr, registry, document,
    )

    return StepResult(
        step_name=legacy_step.name,
        duration_seconds=canonical_sr.duration_seconds,
        output_types=tuple(produced_at),
        junction_metrics=junction_metrics,
        error=error,
    )


def _compute_junction_metrics_for_step(
    produced_at: list[ArtifactType],
    canonical_sr: _CanonicalStepResult,
    registry: _PayloadRegistry,
    document: Document,
) -> dict[str, dict[str, Any]]:
    """Calcule ``junction_metrics`` en post-traitant les outputs.

    Pour chaque ``ArtifactType`` produit, retrouve le payload via
    ``registry`` (les ``Artifact`` du canonique ne portent pas de
    ``content`` direct — voir ``_BaseModuleAdapter``) puis appelle
    ``compute_at_junction(gt, payload, (T, T))`` exactement comme le
    Sprint 63.  Les exceptions par jonction sont logguées et la
    jonction est silencieusement ignorée — comportement historique.
    """
    junction_metrics: dict[str, dict[str, Any]] = {}
    for at in produced_at:
        gt_level = _artifact_type_to_gt_level(at)
        if gt_level is None:
            continue
        gt_payload = document.get_gt(gt_level)
        if gt_payload is None:
            continue
        artifact_id = canonical_sr.produced_artifacts.get(at.value)
        if artifact_id is None or artifact_id not in registry:
            continue
        payload = registry.get(artifact_id)
        try:
            metrics = compute_at_junction(
                _gt_payload_to_value(gt_payload),
                payload,
                (at, at),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[pipeline_runner] évaluation à la jonction %s "
                "a levé : %s",
                at.value, exc,
            )
            continue
        if metrics:
            junction_metrics[at.value] = metrics

    # Phase 4-bis : double-clé pour rétrocompat.  Les tests
    # legacy cherchent junction_metrics["text"] mais le runner
    # peut produire junction_metrics["raw_text"] si l'enum est
    # migré (ArtifactType.TEXT alias de RAW_TEXT, valeur
    # "raw_text").  expand_legacy_keys ajoute la clé legacy
    # ("text") à côté de la canonique ("raw_text") sans écraser.
    from picarones.domain.artifacts import expand_legacy_keys
    expand_legacy_keys(junction_metrics)
    return junction_metrics


def _translate_canonical_error(canonical_error: str | None) -> Optional[str]:
    """Traduit un message d'erreur canonique vers le format legacy.

    Le ``PipelineExecutor`` produit des messages structurés avec un
    préfixe (``adapter_raised:``, ``missing_input:``, ``missing_output:``,
    ``adapter_not_found:``).  Les tests legacy s'attendent à des
    messages français du Sprint 63 — on convertit pour préserver
    rétrocompat strict tant que la sub-phase 7.C n'a pas migré les
    tests.
    """
    if canonical_error is None:
        return None
    if canonical_error.startswith("adapter_raised: "):
        # "adapter_raised: TypeError: bla" → "TypeError: bla"
        return canonical_error[len("adapter_raised: "):]
    if canonical_error.startswith("missing_input: "):
        miss = canonical_error[len("missing_input: "):]
        return f"entrée manquante : {miss}"
    if canonical_error.startswith("missing_output: "):
        # Format canonique : "missing_output: ['raw_text', 'alto_xml']"
        # On parse cette repr de liste pour produire le format legacy
        # "sortie manquante : raw_text,alto_xml".
        miss_repr = canonical_error[len("missing_output: "):]
        miss = miss_repr.strip("[]").replace("'", "").replace(" ", "")
        return f"sortie manquante : {miss}"
    if canonical_error.startswith("adapter_not_found: "):
        adapter = canonical_error[len("adapter_not_found: "):]
        return f"adapter introuvable : {adapter}"
    if canonical_error.startswith("adapter_resolver_failed: "):
        msg = canonical_error[len("adapter_resolver_failed: "):]
        return f"résolution adapter échouée : {msg}"
    return canonical_error


def _gt_payload_to_value(payload: Any) -> Any:
    """Extrait la valeur exploitable d'un ``GTPayload`` typé.

    Pour ``TextGT`` on veut juste la chaîne ; pour les autres
    payloads on retourne le payload entier (la métrique sait quoi
    en faire selon sa signature de types).
    """
    # Import paresseux pour éviter une dépendance cyclique
    from picarones.evaluation.corpus import (
        AltoGT, EntitiesGT, PageGT, ReadingOrderGT, TextGT,
    )
    if isinstance(payload, TextGT):
        return payload.text
    if isinstance(payload, EntitiesGT):
        return payload.entities
    if isinstance(payload, ReadingOrderGT):
        return payload.region_order
    if isinstance(payload, (AltoGT, PageGT)):
        return payload
    return payload


__all__ = [
    "PipelineRunner",
    "PipelineResult",
    "PipelineSpec",
    "PipelineStep",
    "StepResult",
]
