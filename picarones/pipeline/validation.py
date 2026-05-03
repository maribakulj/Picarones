"""``validate_spec`` — Sprint A14-S6.

Validation statique d'une ``PipelineSpec`` : vérifier que les
types s'enchaînent, qu'il n'y a pas d'IDs dupliqués, que les
références ``inputs_from`` pointent bien vers des étapes
antérieures qui produisent le bon type, et (optionnellement) que
les ``adapter_name`` existent dans un registre fourni.

S'exécute **sans instancier aucun adapter** — c'est le bénéfice
clé de la séparation déclaratif/runtime du S6.

API :

    >>> errors = validate_spec(spec)
    >>> if errors:
    ...     for e in errors:
    ...         print(f"{e.step_id}: {e.message}")

Le caller décide de la suite — typiquement un service applicatif
refuse de démarrer un run si la spec a des erreurs.

Anti-sur-ingénierie
-------------------
Pas de détection de cycles graphes complexe (le DAG est exprimé
par ordre des steps, donc impossible de référencer une étape
postérieure : si tu as une boucle, c'est qu'une référence pointe
vers un nom inconnu, déjà détecté).

Pas de validation des params (chaque adapter validera les siens
au moment de l'exécution — le format libre des params est un
choix assumé).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from picarones.domain.artifacts import ArtifactType
from picarones.pipeline.spec import INITIAL_STEP_ID, PipelineSpec, PipelineStep


class ValidationError(BaseModel):
    """Une erreur de validation d'une ``PipelineSpec``.

    Format structuré pour faciliter le rendu (CLI, rapport, JSON).
    Volontairement plat — pas de hiérarchie d'erreurs ; on ajoute
    un ``code`` discriminant si un caller en a besoin.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: str | None
    """Step concerné, ou ``None`` pour les erreurs globales (DAG vide,
    ID dupliqué détecté entre deux steps...)."""

    code: str
    """Identifiant court (``"duplicate_id"``, ``"unknown_adapter"``,
    ``"missing_input"``, ``"unknown_input_source"``, ...).  Permet
    à un test d'asserter sur le code plutôt que sur le message
    français.
    """

    message: str
    """Description humainement lisible (français)."""


def validate_spec(
    spec: PipelineSpec,
    available_adapters: set[str] | None = None,
) -> list[ValidationError]:
    """Vérifie une ``PipelineSpec`` et retourne la liste des erreurs.

    Parameters
    ----------
    spec:
        La spec à valider.
    available_adapters:
        Set des noms d'adapters connus.  Si fourni, chaque
        ``adapter_name`` du DAG est vérifié.  Si ``None`` (défaut),
        cette validation est sautée — utile pour les tests qui
        valident la cohérence d'un YAML sans avoir le runtime
        chargé.

    Returns
    -------
    list[ValidationError]
        Liste vide si la spec est valide ; sinon un ou plusieurs
        problèmes (ne s'arrête pas à la première erreur — le
        caller veut tout voir d'un coup).
    """
    errors: list[ValidationError] = []

    # -- 0. Steps absents
    if not spec.steps:
        errors.append(ValidationError(
            step_id=None,
            code="empty_pipeline",
            message="pipeline vide : au moins une étape est requise",
        ))
        return errors  # impossible de continuer

    # -- 1. IDs dupliqués
    seen_ids: dict[str, int] = {}
    for i, step in enumerate(spec.steps):
        if step.id in seen_ids:
            errors.append(ValidationError(
                step_id=step.id,
                code="duplicate_id",
                message=(
                    f"id dupliqué : '{step.id}' apparaît à l'étape {i} "
                    f"et précédemment à {seen_ids[step.id]}"
                ),
            ))
        else:
            seen_ids[step.id] = i

    # -- 2. Adapter inconnu (si registre fourni)
    if available_adapters is not None:
        for step in spec.steps:
            if step.adapter_name not in available_adapters:
                errors.append(ValidationError(
                    step_id=step.id,
                    code="unknown_adapter",
                    message=(
                        f"adapter '{step.adapter_name}' non disponible.  "
                        f"Adapters connus : {sorted(available_adapters)}"
                    ),
                ))

    # -- 3. Cohérence des types et des références inputs_from
    #    On simule un parcours topologique en ordre de spec.steps.
    #    À chaque étape :
    #    a) Tout type de input_types doit être disponible (soit
    #       initial, soit produit par une étape antérieure).
    #    b) Si inputs_from[type] = "src", "src" doit être une étape
    #       antérieure connue (ou "__initial__") qui produit ce type.

    # Map { step_id (ou "__initial__") -> set(types qu'elle produit) }.
    step_outputs: dict[str, set[ArtifactType]] = {
        INITIAL_STEP_ID: set(spec.initial_inputs),
    }
    # Set des types disponibles à un instant t (latest seulement).
    available: set[ArtifactType] = set(spec.initial_inputs)

    for step in spec.steps:
        errors.extend(_validate_step_against_state(
            step=step,
            step_outputs=step_outputs,
            available=available,
        ))
        # Mise à jour de l'état pour les étapes suivantes.
        step_outputs[step.id] = set(step.output_types)
        available.update(step.output_types)

    return errors


def _validate_step_against_state(
    *,
    step: PipelineStep,
    step_outputs: dict[str, set[ArtifactType]],
    available: set[ArtifactType],
) -> list[ValidationError]:
    """Valide une étape donnée contre l'état des types
    disponibles et des outputs des étapes antérieures."""
    errors: list[ValidationError] = []

    # 3.a — entrées disponibles
    missing = [t for t in step.input_types if t not in available]
    if missing:
        errors.append(ValidationError(
            step_id=step.id,
            code="missing_input",
            message=(
                f"types d'entrée non disponibles : "
                f"{[t.value for t in missing]}.  "
                f"Disponibles : {sorted(t.value for t in available)}"
            ),
        ))

    # 3.b — références inputs_from
    for ref_type, ref_step in step.inputs_from.items():
        if ref_type not in step.input_types:
            errors.append(ValidationError(
                step_id=step.id,
                code="inputs_from_unused",
                message=(
                    f"inputs_from[{ref_type.value}]={ref_step!r} "
                    "mais l'étape ne consomme pas ce type "
                    f"(input_types = {[t.value for t in step.input_types]})"
                ),
            ))
            continue
        if ref_step not in step_outputs:
            errors.append(ValidationError(
                step_id=step.id,
                code="unknown_input_source",
                message=(
                    f"inputs_from[{ref_type.value}]={ref_step!r} "
                    "ne désigne pas une étape antérieure connue "
                    f"({INITIAL_STEP_ID!r} pour les entrées initiales)"
                ),
            ))
            continue
        if ref_type not in step_outputs[ref_step]:
            errors.append(ValidationError(
                step_id=step.id,
                code="source_does_not_produce_type",
                message=(
                    f"inputs_from[{ref_type.value}]={ref_step!r} "
                    f"mais '{ref_step}' ne produit pas {ref_type.value!r}"
                ),
            ))

    return errors


__all__ = ["validate_spec", "ValidationError"]
