"""Banc d'essai de pipelines composées — Sprint 63 (axe B).

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
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from picarones.evaluation.corpus import Document, GTLevel
from picarones.evaluation.metric_registry import compute_at_junction
from picarones.domain.artifacts import ArtifactType
from picarones.domain.module_protocol import BaseModule

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

        # Validation amont : si la pipeline est statiquement
        # invalide, on n'exécute aucune étape.
        problems = spec.validate(tuple(initial_inputs.keys()))
        if problems:
            result.error = " ; ".join(problems)
            return result

        # Sprint 66 — bag versionné : ``versioned[(type, src_step)]``
        # contient l'artefact produit par ``src_step`` pour ``type``.
        # ``src_step`` vaut ``"__initial__"`` pour les entrées
        # initiales fournies par l'utilisateur.  ``latest[type]``
        # désigne le nom de l'étape qui a produit la version la plus
        # récente du type — utilisé en l'absence d'``inputs_from``
        # explicite (rétrocompat Sprint 63).
        versioned: dict[tuple[ArtifactType, str], Any] = {
            (t, "__initial__"): v for t, v in initial_inputs.items()
        }
        latest: dict[ArtifactType, str] = {
            t: "__initial__" for t in initial_inputs
        }

        pipeline_t0 = time.monotonic()
        for step in spec.steps:
            step_result = PipelineRunner._run_step(
                step, versioned, latest, document,
            )
            result.steps.append(step_result)
        result.total_duration_seconds = time.monotonic() - pipeline_t0
        return result

    @staticmethod
    def _run_step(
        step: PipelineStep,
        versioned: dict[tuple[ArtifactType, str], Any],
        latest: dict[ArtifactType, str],
        document: Document,
    ) -> StepResult:
        # Sprint 66 — résolution des entrées : pour chaque type
        # demandé, on consulte ``inputs_from`` ; sinon on prend la
        # dernière version disponible (rétrocompat Sprint 63).
        resolved: dict[ArtifactType, Any] = {}
        missing: list[str] = []
        for t in step.input_types:
            src = step.inputs_from.get(t, latest.get(t))
            if src is None:
                missing.append(t.value)
                continue
            key = (t, src)
            if key not in versioned:
                # Référence explicite vers une étape qui n'a pas
                # produit cet artefact (ex. l'étape source a échoué).
                missing.append(f"{t.value}@{src}")
                continue
            resolved[t] = versioned[key]
        if missing:
            miss_str = ",".join(missing)
            return StepResult(
                step_name=step.name,
                duration_seconds=0.0,
                output_types=(),
                error=f"entrée manquante : {miss_str}",
            )
        inputs_for_module = resolved
        # Exécution chronométrée
        t0 = time.monotonic()
        try:
            outputs = step.module.process(inputs_for_module)
        except Exception as exc:  # noqa: BLE001
            duration = time.monotonic() - t0
            logger.warning(
                "[pipeline_runner] étape '%s' a levé : %s",
                step.name, exc,
            )
            return StepResult(
                step_name=step.name,
                duration_seconds=duration,
                output_types=(),
                error=f"{type(exc).__name__}: {exc}",
            )
        duration = time.monotonic() - t0

        # Validation des sorties : le module est censé déclarer ses
        # output_types, on vérifie qu'il les a tous produits.  Si
        # ce n'est pas le cas, on remonte une erreur explicite mais
        # on conserve les sorties effectivement présentes (utile
        # pour le diagnostic).
        if not isinstance(outputs, dict):
            return StepResult(
                step_name=step.name,
                duration_seconds=duration,
                output_types=(),
                error=(
                    f"le module a retourné {type(outputs).__name__}, "
                    f"un dict[ArtifactType, Any] est attendu"
                ),
            )
        produced = tuple(t for t in step.output_types if t in outputs)
        missing_outputs = [t for t in step.output_types if t not in outputs]
        error: Optional[str] = None
        if missing_outputs:
            miss_str = ",".join(t.value for t in missing_outputs)
            error = f"sortie manquante : {miss_str}"

        # Mise à jour du bag versionné : on stocke la sortie sous
        # une clé (type, step.name) ET on met à jour ``latest`` pour
        # que les étapes suivantes la récupèrent par défaut.
        for t in produced:
            versioned[(t, step.name)] = outputs[t]
            latest[t] = step.name

        # Évaluation aux jonctions : pour chaque type produit, si
        # la GT du même niveau existe, on calcule les métriques.
        junction_metrics: dict[str, dict[str, Any]] = {}
        for at in produced:
            gt_level = _artifact_type_to_gt_level(at)
            if gt_level is None:
                continue
            gt_payload = document.get_gt(gt_level)
            if gt_payload is None:
                continue
            try:
                metrics = compute_at_junction(
                    _gt_payload_to_value(gt_payload),
                    outputs[at],
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

        return StepResult(
            step_name=step.name,
            duration_seconds=duration,
            output_types=produced,
            junction_metrics=junction_metrics,
            error=error,
        )


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
