"""Journal en mémoire des fallbacks d'importer (Sprint A3, item B-3).

Quand un importer (HuggingFace, HTR-United, Gallica, eScriptorium…)
bascule en mode dégradé (timeout réseau, JSON mal formé, ZIP corrompu,
catalogue distant indisponible…), il enregistre un incident ici via
:func:`record_fallback`. Le moteur narratif consomme ces incidents via
:func:`consume_fallback_log`, qui **vide** la liste pour qu'un benchmark
suivant ne remonte pas les incidents du précédent.

Conception volontairement minimale :

- Pas de persistance disque (les incidents sont contextuels à un run).
- Pas de structure complexe (juste un ``list[dict]`` thread-safe).
- Le runner / le rapport peuvent ignorer la liste sans casser.

Le détecteur de Fact correspondant (``FactType.IMPORTER_FALLBACK_TRIGGERED``)
est implémenté dans
:mod:`picarones.evaluation.metrics.narrative.detectors.history`.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_fallbacks: list[dict[str, Any]] = []


def record_fallback(
    importer: str,
    operation: str,
    error: BaseException | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Enregistre un incident de mode dégradé.

    Logge également via ``logger.warning`` pour qu'un opérateur voit
    l'incident en temps réel sans dépendre du rapport.

    Parameters
    ----------
    importer:
        Nom court de l'importer (ex : ``"huggingface"``, ``"htr_united"``).
    operation:
        Description courte de l'opération (ex : ``"yaml_catalogue_parse"``,
        ``"image_save"``, ``"hub_search"``).
    error:
        Exception originelle (utilisée pour le message log et stockée dans
        le payload sous forme de chaîne — pas l'objet, pour éviter les
        références persistantes).
    extra:
        Champs additionnels (URL distante, identifiant dataset…) qui peuvent
        être utiles à un détecteur de Fact ultérieur.
    """
    error_repr = repr(error) if error is not None else None
    logger.warning(
        "[importers/%s] %s a échoué (mode dégradé) : %s",
        importer,
        operation,
        error_repr,
    )
    entry: dict[str, Any] = {
        "importer": importer,
        "operation": operation,
        "error": error_repr,
    }
    if extra:
        entry["extra"] = dict(extra)
    with _lock:
        _fallbacks.append(entry)


def consume_fallback_log() -> list[dict[str, Any]]:
    """Retourne ET vide la liste des incidents accumulés.

    Le moteur narratif appelle cette fonction au moment de construire
    la synthèse pour transformer chaque incident en ``Fact``."""
    with _lock:
        out = list(_fallbacks)
        _fallbacks.clear()
    return out


def peek_fallback_log() -> list[dict[str, Any]]:
    """Retourne une copie sans vider — utile pour les tests."""
    with _lock:
        return list(_fallbacks)


def reset_fallback_log() -> None:
    """Vide la liste sans rien retourner — utile pour les fixtures pytest."""
    with _lock:
        _fallbacks.clear()
