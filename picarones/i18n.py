"""Labels i18n pour le rapport HTML et l'interface Picarones.

Langues supportées
------------------
- ``"fr"`` : français (défaut)
- ``"en"`` : anglais patrimonial (heritage English)

Depuis le Sprint 16, les traductions sont stockées dans
``picarones/report/i18n/{lang}.json`` et chargées au premier accès.
``TRANSLATIONS`` reste exposé comme dict pour compatibilité ascendante.
"""

from __future__ import annotations

import json
from pathlib import Path


_I18N_DIR = Path(__file__).parent / "report" / "i18n"


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
            import logging
            logging.getLogger(__name__).warning(
                "[i18n] fichier '%s' ignoré : %s", path, e,
            )
    return translations


TRANSLATIONS: dict[str, dict[str, str]] = _load_translations()


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
    return TRANSLATIONS.get(lang, TRANSLATIONS.get("fr", {}))


SUPPORTED_LANGS: list[str] = list(TRANSLATIONS.keys())
