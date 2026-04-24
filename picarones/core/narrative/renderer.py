"""Rendu des faits narratifs en texte lisible.

Les templates sont chargés depuis ``templates/{lang}.yaml`` au premier accès.
Le rendu utilise ``str.format_map`` sur le ``payload`` du ``Fact``. Aucun LLM,
aucune génération : la sortie est la concaténation de templates remplis avec
des valeurs venant strictement du JSON d'entrée.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import yaml

from picarones.core.narrative.facts import Fact, FactType

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_TEMPLATES_CACHE: dict[str, dict[str, str]] = {}


def _load_templates(lang: str) -> dict[str, str]:
    """Charge et met en cache les templates de la langue demandée.

    Fallback : si la langue n'existe pas, retourne les templates FR. Si FR
    est également absent (incident d'installation), retourne un dict vide.
    """
    if lang in _TEMPLATES_CACHE:
        return _TEMPLATES_CACHE[lang]

    path = _TEMPLATES_DIR / f"{lang}.yaml"
    if not path.exists():
        if lang != "fr":
            return _load_templates("fr")
        _TEMPLATES_CACHE[lang] = {}
        return _TEMPLATES_CACHE[lang]

    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            logger.warning("[narrative] %s n'est pas un dict YAML — ignoré", path)
            _TEMPLATES_CACHE[lang] = {}
        else:
            _TEMPLATES_CACHE[lang] = {str(k): str(v).strip() for k, v in data.items()}
    except yaml.YAMLError as e:
        logger.warning("[narrative] échec parsing %s : %s", path, e)
        _TEMPLATES_CACHE[lang] = {}

    return _TEMPLATES_CACHE[lang]


class _SafeFormatMap(dict):
    """Dict qui retourne ``'?'`` pour les clés manquantes dans un template.

    Évite qu'un détecteur mal documenté fasse crasher le rendu. En pratique
    les tests couvrent les clés attendues, mais la robustesse prévaut.
    """

    def __missing__(self, key: str) -> str:
        logger.warning("[narrative] clé manquante dans payload : %r", key)
        return "?"


def render_fact(fact: Fact, lang: str = "fr") -> str:
    """Rend un Fact en une phrase selon la langue.

    Retourne ``""`` si le template est absent pour ce type.
    """
    templates = _load_templates(lang)
    tpl = templates.get(fact.type.value)
    if not tpl:
        return ""

    try:
        return tpl.format_map(_SafeFormatMap(fact.payload))
    except (ValueError, KeyError) as e:
        logger.warning(
            "[narrative] rendu impossible pour %s : %s", fact.type.value, e,
        )
        return ""


def render_synthesis(facts: Iterable[Fact], lang: str = "fr") -> list[str]:
    """Rend une liste de Fact en liste de phrases (ordre préservé)."""
    out: list[str] = []
    for fact in facts:
        phrase = render_fact(fact, lang)
        phrase = re.sub(r"\s+", " ", phrase).strip()
        if phrase:
            out.append(phrase)
    return out


def extract_numbers(text: str) -> list[str]:
    """Extrait les nombres (décimaux ou entiers) présents dans une phrase.

    Utilisé par le test de traçabilité : chaque nombre remonté en synthèse
    doit être présent dans le JSON d'entrée.
    """
    return re.findall(r"\d+(?:[.,]\d+)?", text)
