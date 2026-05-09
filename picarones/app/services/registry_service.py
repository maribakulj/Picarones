"""``RegistryService`` — bootstrap explicite des registres.

Sprint A14-S23 du rewrite ciblé.

Le service applicatif qui **construit** explicitement le
``MetricRegistry`` et le ``ProjectorRegistry`` au démarrage, en
remplacement de l'anti-pattern legacy ``import picarones.evaluation.metrics
as _trigger`` (où l'import par effet de bord déclenchait
l'enregistrement via décorateurs au top-level d'un package, chargeant
des dizaines de modules optionnels au moment d'un simple
``import picarones``).

Pourquoi explicite
------------------
- **Pas de chargement transitif** : un test du domain n'a pas besoin
  de `jiwer`, `numpy`, `scipy` parce qu'il importe quelque chose qui
  importe quelque chose qui amorce un registre.
- **Failure mode lisible** : si une métrique optionnelle ne peut pas
  être enregistrée (dépendance absente), on obtient une erreur
  explicite au moment du bootstrap, pas une erreur runtime
  trois layers plus loin.
- **Multi-instances** : un test peut construire SON registre,
  enregistrer EXACTEMENT les métriques dont il a besoin, sans
  partager d'état avec d'autres tests.
- **Inversion de dépendance** : les services consommateurs reçoivent
  des registres injectés, ils ne les importent pas.

Convention
----------
- ``bootstrap_default_registries()`` retourne ``RegistriesBundle``
  (les deux registres pleinement peuplés).
- ``RegistryService(metrics, projectors)`` (constructeur) accepte
  des registres pré-construits ou pré-bootstrappés.
- ``RegistryService.bootstrap_defaults()`` (classmethod) fait le
  bootstrap + construit l'instance en un appel.

Anti-sur-ingénierie
-------------------
- Pas de plugin discovery via entry_points (responsabilité
  ``BACKLOG_POST_LIVRAISON``).
- Pas de versioning du contenu du registre.
- Pas de freeze technique — convention : un seul bootstrap au
  démarrage, lecture seule depuis les services consommateurs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from picarones.domain.artifacts import ArtifactType
from picarones.domain.evaluation_spec import MetricSpec
from picarones.evaluation.metrics.alto_structural import (
    compute_alto_validity,
    compute_line_count_ratio,
    compute_word_box_coverage,
)
from picarones.evaluation.metrics.search import (
    numerical_sequence_preservation,
    searchability_recall,
)
from picarones.evaluation.projectors import (
    AltoToText,
    CanonicalToText,
    PageToText,
    ProjectorRegistry,
)
from picarones.evaluation.registry import MetricRegistry

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Bundle des deux registres
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RegistriesBundle:
    """Paquet de registres prêts à être injectés.

    Frozen pour signaler au caller que les références sont immuables
    une fois bootstrapée — chaque registre individuel reste mutable
    si on veut ajouter à la marge, mais le bundle ne re-pointe pas
    ses champs.
    """

    metrics: MetricRegistry
    projectors: ProjectorRegistry


# ──────────────────────────────────────────────────────────────────────
# Wrappers des métriques texte (jiwer optionnel)
# ──────────────────────────────────────────────────────────────────────


def _safe_jiwer(name: str):
    """Retourne un wrapper qui appelle ``jiwer.{name}`` avec garde-fous
    sur GT/hypothèse vides.  ``jiwer`` est importé à la première
    invocation — si absent, une ``RuntimeError`` claire est levée."""

    def _wrapped(reference: str, hypothesis: str) -> float:
        try:
            import jiwer
        except ImportError as exc:  # pragma: no cover — jiwer est core
            raise RuntimeError(
                f"Métrique {name!r} indisponible : jiwer non installé. "
                "Installer avec ``pip install jiwer``."
            ) from exc
        if not reference:
            return 0.0 if not hypothesis else 1.0
        if not hypothesis:
            return 1.0
        return float(getattr(jiwer, name)(reference, hypothesis))

    _wrapped.__name__ = f"_safe_jiwer_{name}"
    return _wrapped


# ──────────────────────────────────────────────────────────────────────
# Tables canoniques — ce qui est enregistré par défaut
# ──────────────────────────────────────────────────────────────────────


#: Métriques canoniques (RAW_TEXT, RAW_TEXT) — basées sur jiwer.
#: ``higher_is_better=False`` car ce sont des taux d'erreur.
_DEFAULT_TEXT_METRICS: tuple[tuple[MetricSpec, "callable"], ...] = (
    (
        MetricSpec(
            name="cer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
            description="Character Error Rate (jiwer).",
            higher_is_better=False,
        ),
        _safe_jiwer("cer"),
    ),
    (
        MetricSpec(
            name="wer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
            description="Word Error Rate (jiwer).",
            higher_is_better=False,
        ),
        _safe_jiwer("wer"),
    ),
    (
        MetricSpec(
            name="mer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
            description="Match Error Rate (jiwer).",
            higher_is_better=False,
        ),
        _safe_jiwer("mer"),
    ),
    (
        MetricSpec(
            name="wil",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
            description="Word Information Lost (jiwer).",
            higher_is_better=False,
        ),
        _safe_jiwer("wil"),
    ),
)


#: Métriques canoniques de recherche (RAW_TEXT, RAW_TEXT).  Rappel
#: et préservation → ``higher_is_better=True``.
_DEFAULT_SEARCH_METRICS: tuple[tuple[MetricSpec, "callable"], ...] = (
    (
        MetricSpec(
            name="searchability_recall",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
            description=(
                "Rappel fuzzy : fraction des tokens GT retrouvés à "
                "distance de Levenshtein ≤ 2 dans l'hypothèse."
            ),
            higher_is_better=True,
        ),
        searchability_recall,
    ),
    (
        MetricSpec(
            name="numerical_sequence_preservation",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
            description=(
                "Fraction des années 4-chiffres de la GT préservées "
                "strictement dans l'hypothèse."
            ),
            higher_is_better=True,
        ),
        numerical_sequence_preservation,
    ),
)


#: Métriques canoniques structurelles (ALTO_XML, ALTO_XML).
_DEFAULT_ALTO_METRICS: tuple[tuple[MetricSpec, "callable"], ...] = (
    (
        MetricSpec(
            name="alto_validity",
            input_types=(ArtifactType.ALTO_XML, ArtifactType.ALTO_XML),
            description=(
                "1.0 si l'ALTO hypothèse a au moins 1 page, 1 bloc "
                "et 1 ligne ; 0.0 sinon."
            ),
            higher_is_better=True,
        ),
        compute_alto_validity,
    ),
    (
        MetricSpec(
            name="alto_line_count_ratio",
            input_types=(ArtifactType.ALTO_XML, ArtifactType.ALTO_XML),
            description=(
                "min(n_hyp, n_ref) / max(n_hyp, n_ref) sur le nombre "
                "de lignes ALTO.  ∈ [0, 1]."
            ),
            higher_is_better=True,
        ),
        compute_line_count_ratio,
    ),
    (
        MetricSpec(
            name="alto_word_box_coverage",
            input_types=(ArtifactType.ALTO_XML, ArtifactType.ALTO_XML),
            description=(
                "Fraction des ``String`` de l'hypothèse qui portent "
                "une bbox non triviale.  Mesure la qualité de la "
                "détection spatiale."
            ),
            higher_is_better=True,
        ),
        compute_word_box_coverage,
    ),
)


# ──────────────────────────────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────────────────────────────


class RegistryService:
    """Encapsule deux registres + accessors typés.

    Parameters
    ----------
    metrics:
        ``MetricRegistry`` (peut être vide ou pré-rempli).
    projectors:
        ``ProjectorRegistry`` (peut être vide ou pré-rempli).
    """

    def __init__(
        self,
        metrics: MetricRegistry,
        projectors: ProjectorRegistry,
    ) -> None:
        if not isinstance(metrics, MetricRegistry):
            raise TypeError("metrics doit être un MetricRegistry.")
        if not isinstance(projectors, ProjectorRegistry):
            raise TypeError("projectors doit être un ProjectorRegistry.")
        self._metrics = metrics
        self._projectors = projectors

    @property
    def metrics(self) -> MetricRegistry:
        return self._metrics

    @property
    def projectors(self) -> ProjectorRegistry:
        return self._projectors

    @property
    def bundle(self) -> RegistriesBundle:
        return RegistriesBundle(
            metrics=self._metrics, projectors=self._projectors,
        )

    @classmethod
    def bootstrap_defaults(cls) -> "RegistryService":
        """Construit le service avec tous les registres canoniques.

        C'est l'entry point principal : un caller (CLI, web, test
        d'intégration) appelle ``RegistryService.bootstrap_defaults()``
        au démarrage et injecte le résultat dans les services
        consommateurs.
        """
        bundle = bootstrap_default_registries()
        return cls(bundle.metrics, bundle.projectors)


# ──────────────────────────────────────────────────────────────────────
# Bootstrap fonctionnel
# ──────────────────────────────────────────────────────────────────────


def bootstrap_default_registries() -> RegistriesBundle:
    """Construit deux registres pleinement peuplés.

    Pas d'effet de bord : appeler la fonction crée une nouvelle
    instance à chaque fois.  Les anciens callers qui partageaient un
    registre global doivent le maintenir eux-mêmes (ou réutiliser
    la même instance ``RegistryService``).
    """
    metrics = MetricRegistry()
    for spec, func in (
        *_DEFAULT_TEXT_METRICS,
        *_DEFAULT_SEARCH_METRICS,
        *_DEFAULT_ALTO_METRICS,
    ):
        metrics.register(spec, func)

    projectors = ProjectorRegistry()
    projectors.register(AltoToText())
    projectors.register(PageToText())
    projectors.register(CanonicalToText())

    return RegistriesBundle(metrics=metrics, projectors=projectors)


__all__ = [
    "RegistriesBundle",
    "RegistryService",
    "bootstrap_default_registries",
]
