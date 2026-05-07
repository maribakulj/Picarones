"""Métriques typées ``(ALTO, ALTO)`` — Chantier 1.

Pourquoi ce module
------------------
Le registre typé du Sprint 34 prévoit une signature ``(input_type,
output_type)`` pour chaque métrique.  ``builtin_metrics.py`` enregistre
les quatre métriques scalaires sur ``(TEXT, TEXT)`` et un stub sur
``(TEXT, ALTO)``.  Aucune métrique n'était enregistrée sur la jonction
``(ALTO, ALTO)`` — pourtant indispensable dès qu'une pipeline produit
un ALTO et qu'une GT ALTO est disponible (Sprint 32).

Ce module comble cette lacune.  Il expose un helper
:func:`extract_text_from_alto` qui parse l'ALTO XML et reconstruit le
texte plat dans l'ordre ``Page → TextBlock → TextLine → String``, et
enregistre quatre métriques natives (``alto_text_cer``,
``alto_text_wer``, ``alto_text_mer``, ``alto_text_wil``) qui appliquent
les opérateurs jiwer historiques sur le texte extrait des deux côtés.

L'approche est strictement additive vis-à-vis de
:mod:`picarones.measurements.metrics` : ce module ne touche pas le chemin de
calcul historique (``compute_metrics``), il enrichit uniquement le
registre typé pour les pipelines composées.

Robustesse
----------
- L'ALTO peut être passé sous forme :
    * ``str`` (XML brut),
    * :class:`picarones.evaluation.corpus.AltoGT` (porteur d'un ``xml_content``),
    * tout objet exposant un attribut ``xml_content`` typé.
- Le parser tolère les ALTO sans namespace, ALTO 2.x, ALTO 3.x, ALTO
  4.x — il cherche les balises locales par leur nom court (``Page``,
  ``TextLine``, ``String``).
- Un ALTO illisible ou vide → texte extrait ``""``.  Le calcul de CER
  reste possible (la couche jiwer sait gérer une référence non vide
  vs hypothèse vide).
- Aucune dépendance externe : utilise ``xml.etree.ElementTree`` du
  stdlib.

Cas typique d'usage
-------------------
Un VLM produit un ALTO via un reconstructeur (par exemple
:class:`picarones.modules.TextToAltoMonoRegion`).  La GT
:class:`picarones.evaluation.corpus.AltoGT` du document est confrontée à la
sortie via :func:`picarones.evaluation.metric_registry.compute_at_junction`,
qui sélectionne automatiquement les métriques ``(ALTO, ALTO)``
ci-dessous.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from picarones.core.xml_utils import safe_parse_xml

from picarones.evaluation.metric_registry import register_metric
from picarones.domain.artifacts import ArtifactType

logger = logging.getLogger(__name__)


try:
    import jiwer
    _JIWER_AVAILABLE = True
except ImportError:
    _JIWER_AVAILABLE = False


_LOCAL_NAME_RE = re.compile(r"\{[^}]*\}")


def _local(tag: str) -> str:
    """Retire le préfixe de namespace XML pour ne garder que le nom local.

    ElementTree expose les tags sous la forme ``{namespace}LocalName``
    quand un namespace est déclaré.  On normalise pour pouvoir
    matcher uniformément les ALTO avec ou sans namespace.
    """
    return _LOCAL_NAME_RE.sub("", tag)


def _coerce_alto_to_str(payload: Any) -> str:
    """Accepte plusieurs formes d'ALTO et retourne le XML brut."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    xml_content = getattr(payload, "xml_content", None)
    if isinstance(xml_content, str):
        return xml_content
    # Dernier recours — l'utilisateur a passé un objet avec str()
    # raisonnable (tests, mocks).  On ne lève pas, on retourne ""
    # pour ne pas faire échouer une jonction sur un input bizarre.
    return ""


def extract_text_from_alto(payload: Any) -> str:
    """Extrait le texte plat d'un ALTO XML.

    L'ordre suivi reproduit la lecture naturelle ALTO :
    ``Page → PrintSpace → TextBlock → TextLine → String``, avec
    insertion d'un espace entre les ``String`` d'une même ligne et
    d'un saut de ligne entre lignes.  Les ``SP`` (espaces explicites)
    sont implicites — on n'en a pas besoin si on met un espace entre
    chaque ``String``.

    Parameters
    ----------
    payload:
        ALTO sous forme ``str``, :class:`AltoGT`, ou tout objet
        exposant ``xml_content``.

    Returns
    -------
    str
        Texte reconstruit, ``""`` si l'ALTO est invalide ou vide.

    Notes
    -----
    Cette fonction est délibérément tolérante : un ALTO partiellement
    valide produit le texte qu'il a pu extraire avant l'erreur de
    parsing.  Cela évite de faire échouer une jonction parce que la
    GT a un défaut mineur (encodage, déclaration manquante).
    """
    xml = _coerce_alto_to_str(payload).strip()
    if not xml:
        return ""
    # ``safe_parse_xml`` neutralise XXE / Billion Laughs / DTD
    # retrieval — l'ALTO peut venir d'un module ``BaseModule`` tiers
    # qui n'a pas de garantie de provenance.
    root = safe_parse_xml(xml.encode("utf-8") if isinstance(xml, str) else xml)
    if root is None:
        logger.warning(
            "[alto_metrics] ALTO non parsable (XML invalide ou défense XXE "
            "déclenchée) — texte extrait vide",
        )
        return ""

    lines_text: list[str] = []
    # Itère sur tous les TextLine, peu importe leur profondeur.
    for line in root.iter():
        if _local(line.tag) != "TextLine":
            continue
        words: list[str] = []
        for s in line.iter():
            if _local(s.tag) != "String":
                continue
            content = s.attrib.get("CONTENT", "")
            if content:
                words.append(content)
        lines_text.append(" ".join(words))
    return "\n".join(lines_text).strip()


def _safe_jiwer_call(fn, reference: str, hypothesis: str) -> float:
    if not _JIWER_AVAILABLE:
        raise RuntimeError(
            "jiwer n'est pas installé — installer avec `pip install jiwer`"
        )
    if not reference:
        return 0.0 if not hypothesis else 1.0
    if not hypothesis:
        return 1.0
    return fn(reference, hypothesis)


# ──────────────────────────────────────────────────────────────────────────
# Métriques (ALTO, ALTO) — opèrent sur le texte extrait de chaque ALTO
# ──────────────────────────────────────────────────────────────────────────


@register_metric(
    name="alto_text_cer",
    input_types=(ArtifactType.ALTO, ArtifactType.ALTO),
    description=(
        "CER calculé sur le texte plat extrait des ALTO (référence vs "
        "hypothèse).  Permet de mesurer la qualité d'un reconstructeur "
        "ALTO sur l'axe textuel, indépendamment du layout."
    ),
    higher_is_better=False,
    tags={"alto", "text", "edit_distance"},
)
def alto_text_cer(reference_alto: Any, hypothesis_alto: Any) -> float:
    return _safe_jiwer_call(
        jiwer.cer,
        extract_text_from_alto(reference_alto),
        extract_text_from_alto(hypothesis_alto),
    )


@register_metric(
    name="alto_text_wer",
    input_types=(ArtifactType.ALTO, ArtifactType.ALTO),
    description="WER calculé sur le texte plat extrait des ALTO.",
    higher_is_better=False,
    tags={"alto", "text", "edit_distance"},
)
def alto_text_wer(reference_alto: Any, hypothesis_alto: Any) -> float:
    return _safe_jiwer_call(
        jiwer.wer,
        extract_text_from_alto(reference_alto),
        extract_text_from_alto(hypothesis_alto),
    )


@register_metric(
    name="alto_text_mer",
    input_types=(ArtifactType.ALTO, ArtifactType.ALTO),
    description="MER calculé sur le texte plat extrait des ALTO.",
    higher_is_better=False,
    tags={"alto", "text"},
)
def alto_text_mer(reference_alto: Any, hypothesis_alto: Any) -> float:
    return _safe_jiwer_call(
        jiwer.mer,
        extract_text_from_alto(reference_alto),
        extract_text_from_alto(hypothesis_alto),
    )


@register_metric(
    name="alto_text_wil",
    input_types=(ArtifactType.ALTO, ArtifactType.ALTO),
    description="WIL calculé sur le texte plat extrait des ALTO.",
    higher_is_better=False,
    tags={"alto", "text"},
)
def alto_text_wil(reference_alto: Any, hypothesis_alto: Any) -> float:
    return _safe_jiwer_call(
        jiwer.wil,
        extract_text_from_alto(reference_alto),
        extract_text_from_alto(hypothesis_alto),
    )


__all__ = [
    "extract_text_from_alto",
    "alto_text_cer",
    "alto_text_wer",
    "alto_text_mer",
    "alto_text_wil",
]
