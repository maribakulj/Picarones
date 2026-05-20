"""Backends d'extraction d'entités nommées

Suite directe du Sprint 38 : la couche de calcul (`compute_ner_metrics`)
prend deux listes d'entités, ce module fournit le moyen d'**obtenir** la
liste d'entités d'un côté à partir d'un texte (généralement la sortie
OCR du moteur).

Architecture
------------
- ``EntityExtractor`` : Protocol Python qui décrit l'interface ; tout
  callable ``(text: str) -> list[dict]`` est un extracteur valide.  Le
  format de sortie est compatible ``EntitiesGT`` (Sprint 32) et
  ``compute_ner_metrics``
- ``SpacyEntityExtractor`` : implémentation par défaut, lazy-import de
  spaCy.  Si spaCy n'est pas installé OU si le modèle n'est pas
  téléchargé, retourne ``[]`` avec un ``logger.warning`` explicite
  (cf. règle CLAUDE.md : pas de ``except: pass``).
- ``SPACY_PROFILES`` : dict de profils nommés vers noms de modèles
  spaCy (FR, EN, multilingue, HIPE pour les corpus historiques).
- ``get_extractor(profile)`` : factory qui retourne l'extracteur
  correspondant au profil demandé.

Découplage runner ↔ backend
---------------------------
Le runner reçoit un ``EntityExtractor`` en paramètre — il n'importe
**jamais** spaCy directement.  Cela permet :

1. de **tester** sans dépendance externe (le test injecte un callable
   qui simule l'extraction) ;
2. de **brancher** des backends alternatifs (Stanza, HIPE custom,
   modèle fine-tuné maison) sans modifier le runner ;
3. de **désactiver** la métrique en passant ``None`` — comportement
   par défaut, rétrocompat stricte.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Interface
# ──────────────────────────────────────────────────────────────────────────


class EntityExtractor(Protocol):
    """Tout callable ``(text) -> list[dict]`` est un extracteur valide.

    Format de sortie attendu : liste de dicts
    ``{"label": str, "start": int, "end": int, "text": str}``
    compatibles avec ``compute_ner_metrics`` (Sprint 38) et
    ``EntitiesGT``
    """

    def __call__(self, text: str) -> list[dict[str, Any]]: ...


# ──────────────────────────────────────────────────────────────────────────
# Profils spaCy nommés
# ──────────────────────────────────────────────────────────────────────────


SPACY_PROFILES: dict[str, str] = {
    "fr": "fr_core_news_sm",
    "fr_lg": "fr_core_news_lg",
    "en": "en_core_web_sm",
    "en_lg": "en_core_web_lg",
    "multilingual": "xx_ent_wiki_sm",
    # HIPE 2022 — modèle historique multilingue (Hugging Face).  Pas
    # toujours disponible via ``spacy.load`` direct ; documenté pour
    # mémoire, l'utilisateur peut le wrapper dans un EntityExtractor
    # custom si besoin.
    "hipe": "fr_core_news_lg",
}


# ──────────────────────────────────────────────────────────────────────────
# Backend spaCy
# ──────────────────────────────────────────────────────────────────────────


class SpacyEntityExtractor:
    """Extracteur d'entités basé sur spaCy.

    Lazy-import : ``spacy`` n'est importé qu'au premier appel.  Le
    modèle est chargé une seule fois et mis en cache sur l'instance.

    Si spaCy n'est pas installé OU si le modèle demandé n'est pas
    téléchargé, l'extracteur tombe en mode dégradé (retourne ``[]``
    pour chaque appel) et émet un ``logger.warning`` au premier
    appel.

    Parameters
    ----------
    model_name:
        Nom du modèle spaCy à charger (ex. ``"fr_core_news_sm"``).
    label_mapping:
        Dict optionnel ``{spacy_label: target_label}`` pour
        normaliser les labels (ex. spaCy utilise ``"PERSON"``,
        on veut ``"PER"``).  Si ``None``, garde les labels tels
        quels.

    Examples
    --------
    >>> extractor = SpacyEntityExtractor("fr_core_news_sm")
    >>> entities = extractor("Marie de Bourgogne, en 1477.")
    >>> # liste de dicts {label, start, end, text}, ou [] si spaCy absent
    """

    # Mapping par défaut spaCy → conventions HIPE/CoNLL courtes
    DEFAULT_LABEL_MAPPING: dict[str, str] = {
        "PERSON": "PER",
        "PER": "PER",
        "LOC": "LOC",
        "GPE": "LOC",       # Geo-Political Entity → LOC
        "ORG": "ORG",
        "DATE": "DATE",
        "TIME": "DATE",
        "MISC": "MISC",
    }

    def __init__(
        self,
        model_name: str = "fr_core_news_sm",
        label_mapping: dict[str, str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.label_mapping = (
            dict(label_mapping)
            if label_mapping is not None
            else dict(self.DEFAULT_LABEL_MAPPING)
        )
        self._nlp: Any | None = None
        self._loaded: bool = False
        self._available: bool = False

    def _load(self) -> None:
        """Charge spaCy + modèle au premier appel.  Idempotent."""
        if self._loaded:
            return
        self._loaded = True
        try:
            import spacy  # type: ignore[import-untyped]
        except ImportError as exc:
            logger.warning(
                "[ner_backends] spaCy non installé (%s) — extraction NER "
                "désactivée. Installer avec `pip install picarones[ner]`.",
                exc,
            )
            return
        try:
            self._nlp = spacy.load(self.model_name)
            self._available = True
        except OSError as exc:
            logger.warning(
                "[ner_backends] Modèle spaCy %r introuvable (%s) — extraction "
                "NER désactivée. Télécharger avec `python -m spacy download %s`.",
                self.model_name, exc, self.model_name,
            )

    @property
    def available(self) -> bool:
        """``True`` si spaCy + le modèle sont chargés et utilisables."""
        if not self._loaded:
            self._load()
        return self._available

    def __call__(self, text: str) -> list[dict[str, Any]]:
        if not text:
            return []
        if not self.available or self._nlp is None:
            return []
        doc = self._nlp(text)
        results: list[dict[str, Any]] = []
        for ent in doc.ents:
            label = self.label_mapping.get(ent.label_, ent.label_)
            results.append({
                "label": label,
                "start": int(ent.start_char),
                "end": int(ent.end_char),
                "text": ent.text,
            })
        return results


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────


def get_extractor(profile: str = "fr") -> SpacyEntityExtractor:
    """Retourne un extracteur spaCy pour le profil demandé.

    Le profil peut être :

    - une clé de ``SPACY_PROFILES`` (ex. ``"fr"``, ``"en"``,
      ``"multilingual"``)
    - un nom de modèle spaCy direct (ex. ``"fr_core_news_lg"``)

    L'extracteur est instancié paresseusement (le modèle n'est chargé
    qu'au premier appel).  Si le modèle n'est pas disponible,
    l'extracteur tombe en mode dégradé silencieux (retourne ``[]``).
    """
    model_name = SPACY_PROFILES.get(profile, profile)
    return SpacyEntityExtractor(model_name=model_name)


def is_spacy_available() -> bool:
    """``True`` si la librairie ``spacy`` est importable, sans charger
    de modèle."""
    try:
        import spacy  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = [
    "EntityExtractor",
    "SpacyEntityExtractor",
    "SPACY_PROFILES",
    "get_extractor",
    "is_spacy_available",
]
