"""Glossaire contextuel — loader YAML pour le rapport.

Le glossaire est affiché dans un panneau latéral du rapport HTML. Chaque
terme a sa propre entrée structurée (definition / measures / usage /
limits / reference) dans ``{lang}.yaml``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_GLOSSARY_DIR = Path(__file__).parent
_CACHE: dict[str, dict[str, dict]] = {}


def load_glossary(lang: str = "fr") -> dict[str, dict]:
    """Charge le glossaire pour la langue donnée.

    Retourne un dict ``{term_key: {title, definition, measures, usage,
    limits, reference}}``. Si la langue demandée n'existe pas, retombe sur
    le français. Si même ``fr.yaml`` est absent, retourne ``{}`` (dégradé
    non bloquant — le bouton ``?`` n'apparaîtra simplement pas).
    """
    if lang in _CACHE:
        return _CACHE[lang]

    path = _GLOSSARY_DIR / f"{lang}.yaml"
    if not path.exists():
        if lang != "fr":
            return load_glossary("fr")
        _CACHE[lang] = {}
        return _CACHE[lang]

    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as e:
        logger.warning("[glossary] échec parsing %s : %s", path, e)
        _CACHE[lang] = {}
        return _CACHE[lang]

    if not isinstance(data, dict):
        logger.warning("[glossary] %s n'est pas un dict YAML — ignoré", path)
        _CACHE[lang] = {}
    else:
        _CACHE[lang] = {str(k): v for k, v in data.items() if isinstance(v, dict)}
    return _CACHE[lang]


SUPPORTED_LANGS: list[str] = sorted(
    p.stem for p in _GLOSSARY_DIR.glob("*.yaml")
)


__all__ = ["load_glossary", "SUPPORTED_LANGS"]
