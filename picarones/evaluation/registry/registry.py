"""``MetricRegistry`` — Sprint A14-S5.

Container mutable qui associe chaque ``MetricSpec`` à son callable
de calcul.  **Instancié explicitement** par un service au démarrage
de l'application (cf. ``picarones/app/services/registry_service.py``
au S20) — pas de singleton global, pas de side-effect d'import,
pas de décorateur magique.

Différence avec l'existant ``picarones.core.metric_registry``
-------------------------------------------------------------
L'ancien module utilise un dict module-level
``_METRIC_REGISTRY`` rempli par un décorateur ``@register_metric``
appliqué au top-level d'autres modules.  Conséquence : un
``import picarones`` charge ~50 sous-modules pour amorcer le
registre — anti-pattern documenté dans
``BACKLOG_POST_LIVRAISON.md`` §2.4.

Ici, ``MetricRegistry`` est une classe instanciable :

.. code-block:: python

    from picarones.domain import ArtifactType
    from picarones.domain.evaluation_spec import MetricSpec
    from picarones.evaluation.registry import MetricRegistry

    reg = MetricRegistry()
    reg.register(
        MetricSpec(name="cer", input_types=(
            ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT,
        )),
        compute_cer,  # callable
    )
    selected = reg.select(
        ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT,
    )

Anti-sur-ingénierie
-------------------
Pas de gestion de versions de métrique, pas de namespace, pas de
recherche par tag.  Si un caller a besoin de ces features, il les
implémentera quand le besoin sera concret (probablement S15+).
"""

from __future__ import annotations

from typing import Any, Callable

from picarones.domain.artifacts import ArtifactType
from picarones.domain.errors import PicaronesError
from picarones.domain.evaluation_spec import MetricSpec


class MetricRegistrationError(PicaronesError):
    """Tentative d'enregistrement invalide d'une métrique."""


class MetricNotFoundError(PicaronesError):
    """La métrique demandée n'est pas enregistrée."""


class MetricRegistry:
    """Container mutable de ``MetricSpec`` + callables.

    Thread-safe en lecture après initialisation ; la séquence
    d'enregistrement attendue est : un seul service, au démarrage,
    enregistre toutes les métriques en une fois, puis l'instance
    est figée par convention (lecture seule depuis les services
    consommateurs).

    Pas de mécanisme de freeze technique pour l'instant — si un
    caller modifie le registre après le bootstrap, c'est de sa
    responsabilité.
    """

    def __init__(self) -> None:
        self._specs: dict[str, MetricSpec] = {}
        self._callables: dict[str, Callable[..., Any]] = {}

    # ──────────────────────────────────────────────────────────────────
    # Enregistrement
    # ──────────────────────────────────────────────────────────────────

    def register(self, spec: MetricSpec, func: Callable[..., Any]) -> None:
        """Enregistre une métrique.

        Raises
        ------
        MetricRegistrationError
            Si une métrique du même nom est déjà enregistrée
            (sauf re-enregistrement strict du même couple
            ``(spec, func)``, toléré pour les tests qui re-instancient).
        """
        if not callable(func):
            raise MetricRegistrationError(
                f"register({spec.name!r}) : func n'est pas callable."
            )
        if spec.name in self._specs:
            existing_spec = self._specs[spec.name]
            existing_func = self._callables[spec.name]
            if existing_spec == spec and existing_func is func:
                return  # idempotent
            raise MetricRegistrationError(
                f"Métrique {spec.name!r} déjà enregistrée avec une "
                "autre spec ou un autre callable."
            )
        self._specs[spec.name] = spec
        self._callables[spec.name] = func

    # ──────────────────────────────────────────────────────────────────
    # Lecture
    # ──────────────────────────────────────────────────────────────────

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def names(self) -> list[str]:
        """Liste des noms enregistrés (ordre d'enregistrement)."""
        return list(self._specs.keys())

    def get_spec(self, name: str) -> MetricSpec:
        if name not in self._specs:
            raise MetricNotFoundError(
                f"Métrique {name!r} non enregistrée. "
                f"Disponibles : {sorted(self._specs)}."
            )
        return self._specs[name]

    def get_callable(self, name: str) -> Callable[..., Any]:
        if name not in self._callables:
            raise MetricNotFoundError(
                f"Callable de métrique {name!r} non enregistré."
            )
        return self._callables[name]

    def select(
        self,
        reference_type: ArtifactType,
        hypothesis_type: ArtifactType,
    ) -> list[MetricSpec]:
        """Métriques applicables à une jonction donnée (signature exacte)."""
        target = (reference_type, hypothesis_type)
        return [s for s in self._specs.values() if s.input_types == target]

    # ──────────────────────────────────────────────────────────────────
    # Calcul
    # ──────────────────────────────────────────────────────────────────

    def compute(
        self,
        name: str,
        reference: Any,
        hypothesis: Any,
    ) -> Any:
        """Calcule la métrique nommée sur la paire (référence, hypothèse).

        Aucune capture d'exception : si la métrique lève, l'exception
        remonte au caller (qui est typiquement un
        ``EvaluationViewExecutor`` qui décide quoi en faire dans son
        ``ProjectionReport``).
        """
        func = self.get_callable(name)
        return func(reference, hypothesis)

    def compute_at_junction(
        self,
        reference: Any,
        hypothesis: Any,
        reference_type: ArtifactType,
        hypothesis_type: ArtifactType,
    ) -> dict[str, Any]:
        """Calcule **toutes** les métriques applicables à la jonction.

        Retourne ``{metric_name: value}``.  Une métrique qui lève
        est absente du dict (warning loggé au niveau caller via
        l'EvaluationViewExecutor — ici on remonte l'exception pour
        que les tests détectent les bugs).
        """
        results: dict[str, Any] = {}
        for spec in self.select(reference_type, hypothesis_type):
            results[spec.name] = self.compute(spec.name, reference, hypothesis)
        return results


__all__ = [
    "MetricRegistry",
    "MetricRegistrationError",
    "MetricNotFoundError",
]
