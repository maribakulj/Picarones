"""Registre typé de métriques (couche 3 — evaluation).

Pattern et données
------------------
Registre **module-level** alimenté par effet de bord d'import via
le décorateur ``@register_metric``.  Chaque métrique enregistre une
``MetricSpec`` (nom + signature de types + callable) ; la sélection
typée à une jonction se fait via ``select_metrics(input_types)``.

Différence avec ``picarones.evaluation.registry.MetricRegistry``
----------------------------------------------------------------
Le présent module est le pattern **module-level** : un registre
unique global, alimenté par les imports des sous-packages
(``picarones.evaluation.metrics.__init__`` charge tous les modules
définissant des ``@register_metric``).

``picarones.evaluation.registry.MetricRegistry`` est une **classe
instanciable** — un service applicatif l'instancie explicitement
et y enregistre les métriques sans side-effect d'import.  Les
deux patterns coexistent : le module-level fonctionne pour les
~37 métriques existantes, l'instance-based est réservé aux
contributions tierces et au cadre des ``EvaluationView``.

Exemple d'usage
---------------
>>> from picarones.domain.artifacts import ArtifactType
>>> from picarones.evaluation.metric_registry import (
...     register_metric, select_metrics, compute_at_junction,
... )
>>>
>>> @register_metric(
...     name="my_word_count_ratio",
...     input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
...     description="Rapport du nombre de mots OCR / GT",
... )
... def word_count_ratio(reference: str, hypothesis: str) -> float:
...     ref = max(1, len(reference.split()))
...     return len(hypothesis.split()) / ref
>>>
>>> applicable = select_metrics(
...     (ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
... )
>>> any(spec.name == "my_word_count_ratio" for spec in applicable)
True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from picarones.domain.artifacts import ArtifactType

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Spécification d'une métrique typée
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricSpec:
    """Description déclarative d'une métrique enregistrée.

    Attributs
    ---------
    name:
        Identifiant unique du registre (ex. ``"cer"``,
        ``"reading_order_f1"``).  Deux enregistrements avec le même
        ``name`` lèvent ``ValueError`` à l'enregistrement.
    func:
        Fonction de calcul ``f(reference, hypothesis) -> Any``.  Le type
        des deux arguments doit correspondre à ``input_types``.
    input_types:
        Couple ``(reference_type, hypothesis_type)`` indiquant ce que la
        métrique attend.  Le runner sélectionne par cette signature.
    description:
        Phrase courte affichée dans le rapport / le glossaire.
    higher_is_better:
        ``True`` si une valeur plus élevée signale une meilleure qualité
        (ex : F1, recall) ; ``False`` pour les métriques d'erreur (CER,
        WER).  Utilisé par le moteur narratif pour orienter ses
        comparaisons.
    tags:
        Étiquettes libres pour grouper les métriques (ex. ``{"text",
        "edit_distance"}`` ou ``{"structure", "icdar"}``).
    """

    name: str
    func: Callable[..., Any]
    input_types: tuple[ArtifactType, ArtifactType]
    description: str = ""
    higher_is_better: bool = False
    tags: frozenset[str] = field(default_factory=frozenset)


# ──────────────────────────────────────────────────────────────────────────
# Registre global
# ──────────────────────────────────────────────────────────────────────────


_METRIC_REGISTRY: dict[str, MetricSpec] = {}


def register_metric(
    *,
    name: str,
    input_types: tuple[ArtifactType, ArtifactType],
    description: str = "",
    higher_is_better: bool = False,
    tags: frozenset[str] | set[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Décorateur d'enregistrement d'une métrique typée.

    Parameters
    ----------
    name:
        Identifiant unique.
    input_types:
        Couple ``(reference_artifact_type, hypothesis_artifact_type)``.
    description:
        Aide courte (≤ une phrase).
    higher_is_better:
        ``True`` pour les métriques de qualité, ``False`` pour les
        métriques d'erreur.
    tags:
        Étiquettes pour grouper.

    Raises
    ------
    ValueError
        Si ``name`` est déjà enregistré ou si ``input_types`` n'a pas
        exactement deux éléments.
    """
    if len(input_types) != 2:
        raise ValueError(
            f"input_types doit être un couple (ref, hyp) — reçu {input_types!r}"
        )

    frozen_tags = frozenset(tags) if tags is not None else frozenset()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in _METRIC_REGISTRY:
            existing = _METRIC_REGISTRY[name]
            if existing.func is func:
                # Ré-import du module : on tolère silencieusement.
                return func
            raise ValueError(
                f"Métrique '{name}' déjà enregistrée par "
                f"{existing.func.__module__}.{existing.func.__qualname__}"
            )
        spec = MetricSpec(
            name=name,
            func=func,
            input_types=input_types,
            description=description,
            higher_is_better=higher_is_better,
            tags=frozen_tags,
        )
        _METRIC_REGISTRY[name] = spec
        return func

    return decorator


def get_metric(name: str) -> MetricSpec:
    """Retourne la spec enregistrée pour ``name``.

    Raises
    ------
    KeyError
        Si la métrique n'est pas enregistrée.
    """
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Métrique '{name}' non enregistrée")
    return _METRIC_REGISTRY[name]


def all_metrics() -> list[MetricSpec]:
    """Liste toutes les métriques enregistrées (ordre d'enregistrement)."""
    return list(_METRIC_REGISTRY.values())


def select_metrics(
    input_types: tuple[ArtifactType, ArtifactType],
) -> list[MetricSpec]:
    """Retourne les métriques applicables à une jonction donnée.

    Parameters
    ----------
    input_types:
        Couple ``(reference_type, hypothesis_type)`` à la jonction.

    Returns
    -------
    list[MetricSpec]
        Liste (potentiellement vide) des métriques dont la signature
        correspond exactement.
    """
    return [spec for spec in _METRIC_REGISTRY.values() if spec.input_types == input_types]


def compute_at_junction(
    reference: Any,
    hypothesis: Any,
    input_types: tuple[ArtifactType, ArtifactType],
    *,
    skip_on_error: bool = True,
) -> dict[str, Any]:
    """Calcule toutes les métriques applicables à une jonction.

    Parameters
    ----------
    reference:
        Artefact de référence (typiquement la GT au niveau attendu).
    hypothesis:
        Artefact à évaluer (sortie d'un module).
    input_types:
        Signature de la jonction.  Détermine quelles métriques sont
        sélectionnées.
    skip_on_error:
        Si ``True`` (défaut), une exception levée par une métrique est
        loggée en warning et la métrique est absente du résultat.  Si
        ``False``, l'exception est propagée — utile pour les tests.

    Returns
    -------
    dict[str, Any]
        Dictionnaire ``{metric_name: value}`` pour chaque métrique
        applicable qui s'est calculée sans erreur.
    """
    selected = select_metrics(input_types)
    results: dict[str, Any] = {}
    for spec in selected:
        try:
            value = spec.func(reference, hypothesis)
        except Exception as exc:  # noqa: BLE001
            if skip_on_error:
                logger.warning(
                    "[metric_registry] '%s' a échoué : %s — métrique ignorée",
                    spec.name, exc,
                )
                continue
            raise
        # Audit scientifique (Classe B) : ``None`` = métrique **non
        # applicable** (aucun signal exploitable dans la GT — p. ex.
        # zéro caractère MUFI, zéro chiffre romain).  On l'**omet** au
        # lieu de l'agréger : un document sans signal ne doit pas être
        # compté comme un échec (0.0) ni comme une réussite (1.0).
        if value is None:
            continue
        results[spec.name] = value
    return results


def _reset_registry_for_tests() -> None:
    """Vide le registre global.  **Réservé aux tests** — ne pas appeler
    en production sous peine de désactiver toutes les métriques."""
    _METRIC_REGISTRY.clear()


__all__ = [
    "MetricSpec",
    "register_metric",
    "get_metric",
    "all_metrics",
    "select_metrics",
    "compute_at_junction",
]
