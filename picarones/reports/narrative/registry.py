"""Registre déclaratif des détecteurs narratifs (Sprint 29).

Avant le Sprint 29, ajouter un nouveau type de fait imposait de toucher
**quatre** fichiers :

  1. ``facts.py``    — ajouter une valeur à ``FactType`` ;
  2. ``detectors.py`` — écrire ``def detect_xxx(data) -> list[Fact]`` ;
  3. ``detectors.py`` — l'inscrire dans le dict ``DETECTORS_BY_TYPE`` ;
  4. ``arbiter.py``  — ajouter le type à la séquence ``DEFAULT_TYPE_ORDER``
                       au bon endroit pour la priorité éditoriale.

Sprint 29 ramène le nombre de modifications à **deux** :

  1. ``facts.py``    — toujours nécessaire pour le type énuméré ;
  2. ``detectors.py`` — décorer la fonction avec ``@register_detector(...)``.

Le décorateur :
  - enregistre la fonction dans un registre global trié par ``priority`` ;
  - vérifie qu'aucun détecteur ne se réenregistre sur le même ``FactType`` ;
  - laisse la fonction utilisable telle quelle (rétrocompatibilité) ;
  - alimente automatiquement ``arbiter.DEFAULT_TYPE_ORDER``.

Conventions de priorité (« politique éditoriale » du rapport)
-------------------------------------------------------------
Plus la valeur est petite, plus le fait remonte tôt en synthèse à
importance égale. Pour conserver l'ordre historique du Sprint 23, on
utilise un pas de 10 pour laisser de la place à des insertions futures :

  10  GLOBAL_LEADER_CER       qui gagne globalement
  20  STATISTICAL_TIE         y a-t-il un ex-aequo
  30  SIGNIFICANT_GAP         à quel point l'écart est solide
  40  STRATUM_WINNER          qui domine sur quel sous-corpus
  50  STRATUM_COLLAPSE        qui s'effondre sur quoi
  60  ERROR_PROFILE_OUTLIER   qui se trompe différemment
  70  LLM_HALLUCINATION_FLAG  hallucinations VLM
  80  ROBUSTNESS_FRAGILE      sensibilité aux dégradations
  90  PARETO_ALTERNATIVE      compromis coût/qualité
 100  SPEED_WINNER            vitesse
 110  COST_OUTLIER            coût aberrant
 120  CONFIDENCE_WARNING      mise en garde sur la fiabilité

Le décorateur n'impose **pas** de pas — un détecteur tiers peut très
bien utiliser ``priority=42`` pour s'insérer entre STRATUM_WINNER et
STRATUM_COLLAPSE par exemple.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from picarones.domain.facts import (
    DetectorFn,
    DetectorRegistry,
    FactImportance,
    FactType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Métadonnées d'un détecteur
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectorEntry:
    """Métadonnées d'un détecteur enregistré."""
    fact_type: FactType
    fn: DetectorFn
    priority: int
    importance: FactImportance


# ---------------------------------------------------------------------------
# Registre global
# ---------------------------------------------------------------------------

_REGISTRY: dict[FactType, DetectorEntry] = {}
_REGISTRY_LOCK = threading.Lock()


def register_detector(
    fact_type: FactType,
    *,
    priority: int,
    importance: FactImportance = FactImportance.MEDIUM,
) -> Callable[[DetectorFn], DetectorFn]:
    """Décorateur d'enregistrement.

    Usage::

        @register_detector(FactType.GLOBAL_LEADER_CER, priority=10,
                           importance=FactImportance.CRITICAL)
        def detect_global_leader_cer(data: dict) -> list[Fact]:
            ...

    Le décorateur :
      - vérifie qu'aucun autre détecteur n'est déjà enregistré sur
        ``fact_type`` (sinon ``ValueError``) ;
      - vérifie que ``priority`` est un entier ;
      - retourne la fonction inchangée pour ne pas casser les imports
        existants.

    L'``importance`` mémorisée ici sert de **métadonnée** au registre :
    chaque détecteur reste libre d'émettre des ``Fact`` avec une
    importance différente selon le contexte (ex. CRITICAL si l'écart
    est gigantesque, HIGH sinon).
    """
    def _decorator(fn: DetectorFn) -> DetectorFn:
        with _REGISTRY_LOCK:
            if fact_type in _REGISTRY:
                raise ValueError(
                    f"Détecteur déjà enregistré pour {fact_type.value!r} : "
                    f"{_REGISTRY[fact_type].fn.__name__}. Désenregistrer "
                    "explicitement avant de réassigner."
                )
            entry = DetectorEntry(
                fact_type=fact_type,
                fn=fn,
                priority=int(priority),
                importance=importance,
            )
            _REGISTRY[fact_type] = entry
        logger.debug(
            "[narrative.registry] enregistré %s priority=%s importance=%s",
            fact_type.value, priority, importance.name,
        )
        return fn

    return _decorator


def unregister(fact_type: FactType) -> None:
    """Retire un détecteur du registre — utilisé par les tests."""
    with _REGISTRY_LOCK:
        _REGISTRY.pop(fact_type, None)


def iter_detectors() -> list[DetectorEntry]:
    """Retourne tous les détecteurs enregistrés, triés par ``priority``.

    Le tri est stable : à ``priority`` égale, l'ordre d'enregistrement
    est préservé (utile en présence d'extensions tierces).
    """
    with _REGISTRY_LOCK:
        entries = list(_REGISTRY.values())
    entries.sort(key=lambda e: e.priority)
    return entries


def detector_for(fact_type: FactType) -> Optional[DetectorEntry]:
    with _REGISTRY_LOCK:
        return _REGISTRY.get(fact_type)


def clear_registry() -> None:
    """Vide le registre — réservé aux tests d'isolation."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


def default_type_order() -> tuple[FactType, ...]:
    """Calcule l'ordre canonique des types depuis le registre courant.

    Source de vérité de ``arbiter.DEFAULT_TYPE_ORDER`` depuis le Sprint 29.
    """
    return tuple(e.fact_type for e in iter_detectors())


# ---------------------------------------------------------------------------
# Pont avec ``DetectorRegistry`` historique
# ---------------------------------------------------------------------------

def populate_legacy_registry(registry: DetectorRegistry) -> None:
    """Synchronise le ``DetectorRegistry`` historique depuis le décorateur.

    L'objet ``DetectorRegistry`` reste l'API publique pour les
    consommateurs externes (cf. ``DetectorRegistry.run``) ; cette
    fonction l'alimente depuis le registre déclaratif courant.
    """
    for entry in iter_detectors():
        registry.register(entry.fact_type, entry.fn)


__all__ = [
    "DetectorEntry",
    "register_detector",
    "unregister",
    "iter_detectors",
    "detector_for",
    "clear_registry",
    "default_type_order",
    "populate_legacy_registry",
]


# ---------------------------------------------------------------------------
# Sentinel — sans usage direct ; vérifie au build qu'on n'introduit pas
# de valeur ``priority`` dupliquée par accident parmi les builtins.
# ---------------------------------------------------------------------------

def _verify_unique_priorities() -> None:
    seen: dict[int, FactType] = {}
    for entry in iter_detectors():
        if entry.priority in seen:
            logger.warning(
                "[narrative.registry] priority %s dupliquée : "
                "%s et %s — ordre indéterministe à priorité égale.",
                entry.priority,
                seen[entry.priority].value,
                entry.fact_type.value,
            )
        else:
            seen[entry.priority] = entry.fact_type
