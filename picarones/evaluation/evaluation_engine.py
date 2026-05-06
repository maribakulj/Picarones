"""``EvaluationEngine`` — Sprint A14-S27.

Pendant de ``ProjectionEngine`` (cf. ``projection_engine.py``).
Le S13 fusionnait dans ``DefaultEvaluationViewExecutor`` projection
**et** évaluation ; la cible architecturale les sépare en deux
moteurs spécialisés à responsabilité unique.

``EvaluationEngine`` calcule un ensemble nommé de métriques sur
une paire ``(reference, hypothesis)`` de payloads.  Une métrique
qui lève en interne va dans ``failed_metrics`` au lieu de planter
l'évaluation complète — l'erreur est capturée et associée au nom
de la métrique.

Pourquoi cette séparation
-------------------------
- **Réutilisation** : le ``PipelineExecutor`` (S28+) peut appeler
  ``EvaluationEngine.evaluate`` pour des métriques de jonction
  intra-pipeline (ex : « score de stabilité entre deux étapes ») sans
  passer par un ``EvaluationView``.
- **Testabilité** : on teste la collecte d'erreurs (métrique cassée,
  métrique inconnue) sans instancier de vue ni de projecteur.
- **Découplage** : ``EvaluationEngine`` ne sait rien des artefacts,
  des projections, des vues — il prend des payloads bruts.

Anti-sur-ingénierie
-------------------
Pas de batch (évaluer N paires en une passe), pas de cache de
payload normalisé, pas de pré-tri des métriques.  Le moteur est
volontairement minimal — la complexité vit dans les métriques
elles-mêmes (cf. ``picarones/evaluation/metrics/``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from picarones.evaluation.registry import (
    MetricNotFoundError,
    MetricRegistry,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Résultat d'un appel à ``EvaluationEngine.evaluate``.

    Attributes
    ----------
    metric_values:
        Métriques calculées avec succès, ``{name: value}``.
    failed_metrics:
        Métriques qui ont échoué, ``{name: error_message}``.  Les
        deux dicts sont disjoints : une métrique apparaît dans l'un
        ou l'autre, jamais les deux.

    Notes
    -----
    Frozen dataclass : container immuable ; les dicts internes le
    sont aussi grâce à ``field(default_factory=dict)`` qu'on ne
    mute pas après construction.  Le caller doit considérer les
    dicts comme lecture seule.
    """

    metric_values: dict[str, Any] = field(default_factory=dict)
    failed_metrics: dict[str, str] = field(default_factory=dict)

    @property
    def n_succeeded(self) -> int:
        return len(self.metric_values)

    @property
    def n_failed(self) -> int:
        return len(self.failed_metrics)

    @property
    def all_succeeded(self) -> bool:
        return self.n_failed == 0

    def with_global_failure(self, error: str) -> "EvaluationResult":
        """Retourne un nouveau ``EvaluationResult`` où **toutes** les
        métriques portent le même message d'erreur global.  Utile à
        un caller qui constate qu'un payload n'a pas pu être chargé
        et veut marquer l'évaluation entière en échec."""
        return EvaluationResult(
            metric_values={},
            failed_metrics={
                name: error
                for name in (
                    list(self.metric_values) + list(self.failed_metrics)
                )
            },
        )


class EvaluationEngine:
    """Moteur de calcul de métriques sur une paire de payloads.

    Responsabilité unique : prendre un ``MetricRegistry``, une liste
    de noms de métriques, et une paire ``(reference, hypothesis)``,
    retourner un ``EvaluationResult``.  Pas de connaissance des
    artefacts, des projections, des vues.

    Parameters
    ----------
    metric_registry:
        Registre des métriques, instancié explicitement au démarrage
        (pas de singleton global, pas de side-effect d'import).
    """

    def __init__(self, metric_registry: MetricRegistry) -> None:
        if not isinstance(metric_registry, MetricRegistry):
            raise TypeError(
                "metric_registry doit être un MetricRegistry."
            )
        self._metrics = metric_registry

    @property
    def metrics(self) -> MetricRegistry:
        """Accès en lecture au registre sous-jacent (utile aux tests)."""
        return self._metrics

    def evaluate(
        self,
        metric_names: tuple[str, ...] | list[str],
        reference: Any,
        hypothesis: Any,
    ) -> EvaluationResult:
        """Calcule chaque métrique nommée sur la paire (référence, hypothèse).

        Comportement :

        - Une métrique enregistrée et qui retourne une valeur → entrée
          dans ``metric_values``.
        - Une métrique enregistrée qui lève une exception → entrée
          dans ``failed_metrics`` avec le message ``f"{type}: {message}"``.
        - Un nom de métrique non enregistré → entrée dans
          ``failed_metrics`` avec un message explicite.

        L'ordre d'évaluation suit l'ordre de ``metric_names`` ; les
        deux dicts résultats préservent cet ordre (Python 3.7+
        garantit l'ordre d'insertion sur les ``dict``).
        """
        metric_values: dict[str, Any] = {}
        failed_metrics: dict[str, str] = {}

        for name in metric_names:
            try:
                value = self._metrics.compute(name, reference, hypothesis)
                metric_values[name] = value
            except MetricNotFoundError as exc:
                failed_metrics[name] = (
                    f"métrique non enregistrée dans le MetricRegistry : "
                    f"{exc}"
                )
            except Exception as exc:  # noqa: BLE001
                failed_metrics[name] = f"{type(exc).__name__}: {exc}"

        return EvaluationResult(
            metric_values=metric_values,
            failed_metrics=failed_metrics,
        )

    def evaluate_one(
        self,
        metric_name: str,
        reference: Any,
        hypothesis: Any,
    ) -> EvaluationResult:
        """Cas particulier : une seule métrique.  Sucre syntaxique sur
        ``evaluate``.  Utile aux callers qui pilotent une jonction
        unique (typiquement le pipeline executor sur une métrique de
        jonction)."""
        return self.evaluate((metric_name,), reference, hypothesis)


__all__ = ["EvaluationEngine", "EvaluationResult"]
