"""i18n FR/EN — Sprint A14-S38.

Loader minimaliste pour l'internationalisation des templates Jinja2.
Charge ``fr.json`` et ``en.json`` au démarrage de l'app et expose une
fonction ``translate(key, lang)`` qui retourne la chaîne traduite,
ou la clé elle-même si la traduction est absente (avec warning).

Pas de fallback automatique entre langues — chaque langue est
indépendante.  Les deux fichiers JSON doivent partager les mêmes clés
(test garde-fou ``test_i18n_completeness`` au S38).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent

#: Langues supportées.  Ajout d'une langue = ajout d'un fichier JSON
#: avec les mêmes clés + ajout dans cette liste.
SUPPORTED_LANGUAGES: tuple[str, ...] = ("fr", "en")

DEFAULT_LANGUAGE = "fr"


def _load(language: str) -> dict[str, str]:
    """Charge un fichier de traductions JSON ; lève si introuvable."""
    path = _DIR / f"{language}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"i18n : fichier de traductions absent pour {language!r} "
            f"({path}).",
        )
    return json.loads(path.read_text(encoding="utf-8"))


_TRANSLATIONS: dict[str, dict[str, str]] = {
    lang: _load(lang) for lang in SUPPORTED_LANGUAGES
}


def translate(key: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Retourne la traduction de ``key`` dans ``language``.

    Si la langue est inconnue, fallback silencieux sur la langue par
    défaut (warning loggé).  Si la clé est absente, retourne la clé
    elle-même (warning loggé) — convention « graceful degradation ».
    """
    if language not in _TRANSLATIONS:
        logger.warning(
            "[i18n] langue %r inconnue, fallback %r.",
            language, DEFAULT_LANGUAGE,
        )
        language = DEFAULT_LANGUAGE
    table = _TRANSLATIONS[language]
    if key not in table:
        logger.warning(
            "[i18n] clé %r absente pour %r — utilisation de la clé.",
            key, language,
        )
        return key
    return table[key]


def all_keys(language: str = DEFAULT_LANGUAGE) -> list[str]:
    """Liste des clés disponibles pour une langue (utile aux tests)."""
    return list(_TRANSLATIONS.get(language, {}).keys())


__all__ = [
    "SUPPORTED_LANGUAGES",
    "DEFAULT_LANGUAGE",
    "translate",
    "all_keys",
]
