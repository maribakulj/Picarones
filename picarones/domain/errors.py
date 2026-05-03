"""Exceptions du domaine — Sprint A14-S4.

Hiérarchie centrée sur ``PicaronesError`` pour qu'un caller puisse
attraper "toute erreur métier Picarones" avec un seul ``except``.

Règle d'or : ne JAMAIS attraper ``PicaronesError`` dans le code
métier sans le re-lever — c'est le rôle de la couche transport
(``app/services/`` puis ``interfaces/``) de mapper ces erreurs
vers HTTP 4xx / sortie CLI explicite.

Volontairement plat (pas de hiérarchie profonde) : on ajoute des
sous-classes au cas par cas quand un caller a besoin de discriminer.
"""

from __future__ import annotations


class PicaronesError(Exception):
    """Racine de la hiérarchie d'erreurs métier de Picarones.

    Tout sous-package du nouveau code (``domain/``, ``evaluation/``,
    ``pipeline/``, ``formats/``, ``adapters/``, ``app/``) doit lever
    une sous-classe de ``PicaronesError`` plutôt qu'un ``Exception``
    générique ou un ``ValueError`` quand l'erreur a un sens métier.

    L'ancien code (``picarones.core``, ``picarones.measurements``,
    etc.) garde son comportement actuel jusqu'à sa migration.
    """


class ArtifactValidationError(PicaronesError):
    """Un artefact ne respecte pas les invariants de son type.

    Exemples : un ``Artifact`` typé ``ALTO_XML`` dont le ``content_hash``
    est absent ; un ``Artifact`` dont le ``produced_by_step`` référence
    une étape qui n'existe pas dans la pipeline.
    """


class ProjectionError(PicaronesError):
    """Un projecteur ne peut pas convertir l'artefact source.

    Levée typiquement par les projecteurs ALTO→texte / PAGE→texte
    quand le XML d'entrée n'est pas parsable, n'a pas de TextLine,
    ou que l'ordre de lecture est ambigu.

    Le caller (``EvaluationViewExecutor``) doit propager cette erreur
    dans le ``ProjectionReport`` plutôt que de l'absorber silencieusement.
    """


class CorpusSpecError(PicaronesError):
    """Le ``CorpusSpec`` est mal formé.

    Exemples : ``DocumentRef.id`` dupliqués, chemins relatifs
    ambigus sans racine, GT déclarée pour un niveau non supporté.
    """


__all__ = [
    "PicaronesError",
    "ArtifactValidationError",
    "ProjectionError",
    "CorpusSpecError",
]
