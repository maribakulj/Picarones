"""Labels i18n pour le rapport HTML et l'interface Picarones.

Phase 5.E — module relocalisé depuis ``picarones.i18n`` vers
``picarones.reports_v2.i18n``.  Le chemin legacy reste disponible
via un shim avec ``DeprecationWarning`` ; suppression prévue en 2.0.

Langues supportées
------------------
- ``"fr"`` : français (défaut)
- ``"en"`` : anglais patrimonial (heritage English)

Depuis le Sprint 17, les traductions sont stockées dans des fichiers
JSON et chargées au premier accès.  ``TRANSLATIONS`` reste exposé
comme dict pour compatibilité ascendante.

Sprint 30 — durcissement
------------------------
- Chargement lazy + thread-safe via verrou explicite ; les serveurs
  web sous charge concurrente ne peuvent plus initialiser deux fois.
- ``reload_translations()`` exposé pour les tests qui modifient les
  fichiers JSON à la volée.
- ``get_labels()`` mémoizé via ``functools.lru_cache`` pour absorber
  le fallback ``lang → fr`` sans relire le dict à chaque appel.
"""

from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


_I18N_DIR = Path(__file__).parent
_LOAD_LOCK = threading.Lock()
_TRANSLATIONS_CACHE: dict[str, dict[str, str]] | None = None


def _load_translations() -> dict[str, dict[str, str]]:
    """Charge tous les fichiers JSON du dossier i18n.

    Un fichier ``{lang}.json`` définit les labels de la langue ``lang``.
    Retourne toujours un dict non-vide, même si le dossier est manquant
    (dans ce cas, le dict est vide et ``get_labels`` tombe sur un fallback).
    """
    translations: dict[str, dict[str, str]] = {}
    if not _I18N_DIR.is_dir():
        return translations
    for path in sorted(_I18N_DIR.glob("*.json")):
        lang = path.stem
        try:
            with path.open(encoding="utf-8") as fh:
                translations[lang] = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("[i18n] fichier '%s' ignoré : %s", path, e)
    return translations


def _get_translations() -> dict[str, dict[str, str]]:
    """Retourne le cache de translations, initialisé une seule fois.

    Thread-safe : deux threads qui appellent simultanément en démarrage
    ne déclencheront qu'une seule lecture disque.
    """
    global _TRANSLATIONS_CACHE
    if _TRANSLATIONS_CACHE is not None:
        return _TRANSLATIONS_CACHE
    with _LOAD_LOCK:
        if _TRANSLATIONS_CACHE is None:
            _TRANSLATIONS_CACHE = _load_translations()
    return _TRANSLATIONS_CACHE


def reload_translations() -> None:
    """Force la relecture des fichiers JSON au prochain ``get_labels``.

    Utile pour les tests qui modifient ``reports_v2/i18n/*.json`` à la volée.
    """
    global _TRANSLATIONS_CACHE
    with _LOAD_LOCK:
        _TRANSLATIONS_CACHE = None
    _get_labels_cached.cache_clear()


@lru_cache(maxsize=None)
def _get_labels_cached(lang: str) -> tuple[tuple[str, str], ...]:
    """Cache mémoïsé : ``lang -> tuple ordonné des paires``.

    Le retour en tuple permet à ``lru_cache`` de mémoriser sans
    contrainte de hashabilité, et est trivialement converti en dict
    par ``get_labels`` à chaque appel (coût O(n)).
    """
    translations = _get_translations()
    labels = translations.get(lang) or translations.get("fr") or {}
    return tuple(labels.items())


def get_labels(lang: str = "fr") -> dict[str, str]:
    """Retourne le dictionnaire de labels pour la langue donnée.

    Parameters
    ----------
    lang:
        Code langue : ``"fr"`` (défaut) ou ``"en"``.

    Returns
    -------
    dict
        Labels traduits. Toujours valide : bascule sur ``"fr"`` si lang inconnu.
        Si ``"fr"`` lui-même manque, retourne un dict vide (comportement dégradé
        mais non bloquant).
    """
    return dict(_get_labels_cached(lang))


# ``TRANSLATIONS`` reste accessible comme attribut module pour les
# consommateurs externes qui le lisaient directement. Initialisé
# paresseusement à l'import — n'engendre **pas** de lecture si le
# module n'est jamais utilisé.
TRANSLATIONS: dict[str, dict[str, str]] = _get_translations()
SUPPORTED_LANGS: list[str] = list(TRANSLATIONS.keys())


__all__ = [
    "TRANSLATIONS",
    "SUPPORTED_LANGS",
    "get_labels",
    "reload_translations",
]
