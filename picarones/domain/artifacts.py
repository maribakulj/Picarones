"""``Artifact`` et ``ArtifactType`` — Sprint A14-S4.

Toute sortie d'une étape de pipeline est un **artefact traçable** :
identifiant stable, type explicite, hash du contenu, provenance.

Différences avec ``picarones.core.modules.ArtifactType`` (Sprint 33)
-------------------------------------------------------------------
L'ancien ``ArtifactType`` historique a 6 valeurs :
``IMAGE, TEXT, ALTO, PAGE, ENTITIES, READING_ORDER``.  Le nouveau
en a 9, avec deux distinctions importantes pour les vues d'évaluation
introduites aux Sprints S13-S18 :

- **``RAW_TEXT`` vs ``CORRECTED_TEXT``** — un OCR brut et un texte
  corrigé par un LLM ont la même structure (string) mais des contrats
  différents : seul le second peut être projeté vers ``ALTO_XML``
  via reconstruction.  Cette distinction permet à ``TextView`` de
  comparer honnêtement les deux types dans la même vue tout en
  signalant à l'utilisateur que la projection a un sens différent.
- **``ALTO_XML`` vs ``PAGE_XML`` vs ``CANONICAL_DOCUMENT``** — les
  trois formats spatiaux sont conceptuellement distincts ; un
  ``CANONICAL_DOCUMENT`` (markdown ou JSON canonique produit par un
  VLM) n'a pas de coordonnées et ne peut pas être projeté vers
  ``ALTO_XML`` sans étape de reconstruction.

Anti-sur-ingénierie
-------------------
``Artifact`` ne porte que les champs nécessaires aux vues actuelles.
Champs reportés (à ajouter quand un caller en a concrètement besoin) :
``media_type``, ``cost``, ``latency``, ``warnings``, ``model_version``,
``parent_artifact_ids`` (DAG d'origine).
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ArtifactType(str, Enum):
    """Type d'un artefact produit ou consommé par une étape de pipeline.

    Volontairement extensible : si une nouvelle vue (post-livraison)
    nécessite un type supplémentaire (ex : ``LAYOUT_HEATMAP``), on
    l'ajoute ici avec un commentaire indiquant la vue qui le
    consomme.

    Convention de nommage : ``UPPER_SNAKE_CASE`` pour le nom Python,
    ``lower_snake_case`` pour la valeur string sérialisée (utilisée
    dans les YAML de pipeline et dans les exports JSON).
    """

    #: Image source (PNG, TIFF, JPEG).  Entrée typique d'un OCR.
    IMAGE = "image"

    #: Texte brut produit par un OCR (avant correction LLM).
    RAW_TEXT = "raw_text"

    #: Texte corrigé par un LLM ou un module de post-correction.
    #: Distinct de ``RAW_TEXT`` parce que les vues d'évaluation
    #: doivent pouvoir signaler "ce texte a été modifié par un
    #: modèle après l'OCR" (impact sur over-normalisation,
    #: hallucination, fidélité philologique).
    CORRECTED_TEXT = "corrected_text"

    #: ALTO XML 4.x avec lignes, mots, coordonnées, ordre de lecture.
    ALTO_XML = "alto_xml"

    #: PAGE XML (PRIMA / Transkribus).
    PAGE_XML = "page_xml"

    #: Représentation canonique structurée sans coordonnées.
    #: Typique d'une sortie VLM (markdown, JSON canonique).  Peut
    #: être reconstruit en ALTO via un module dédié, mais n'a pas
    #: nativement les coordonnées spatiales.
    CANONICAL_DOCUMENT = "canonical_document"

    #: Liste d'entités nommées (PER, LOC, ORG, DATE, MISC...).
    ENTITIES = "entities"

    #: Liste ordonnée d'IDs de régions documentaires définissant
    #: l'ordre de lecture (essentiel pour les manuscrits glosés et
    #: les journaux multi-colonnes).
    READING_ORDER = "reading_order"

    #: Alignement entre deux artefacts (typiquement ``RAW_TEXT`` →
    #: ``CORRECTED_TEXT`` produit par un module de post-correction
    #: ou de remapping ALTO).  Utilisé par ``HallucinationView`` et
    #: ``error_absorption``.
    ALIGNMENT = "alignment"

    #: Confidences OCR au niveau token.  Sidecar JSON produit par les
    #: adapters OCR qui exposent des scores natifs (Tesseract
    #: image_to_data, Pero transcription_confidence, Mistral OCR API
    #: confidences, Google Vision Word.confidence, Azure DI
    #: Word.confidence).
    #:
    #: Schéma JSON : ``{"tokens": [{"text": str, "confidence":
    #: float ∈ [0, 1]}], "extractor": str, "model_version": str |
    #: null}``.  Consommé par les vues de calibration (ECE/MCE,
    #: reliability diagram).
    CONFIDENCES = "confidences"


def compute_content_hash(payload: bytes) -> str:
    """SHA-256 hex (64 chars) d'un payload binaire.

    Helper exposé au domain pour que les adapters puissent calculer
    un hash compatible avec ``Artifact.content_hash`` sans dépendre
    d'un détail d'implémentation.
    """
    return hashlib.sha256(payload).hexdigest()


# Validation des identifiants.  On veut un ``id`` stable et
# filesystem-safe (utilisable comme nom de fichier dans
# ``ArtifactStore``) sans imposer un format trop restrictif.
_ID_RE = re.compile(r"^[A-Za-z0-9_.\-:/]+$")


class Artifact(BaseModel):
    """Une sortie traçable d'une étape de pipeline.

    Immuable (``frozen=True``) : un artefact ne change pas après
    création.  Pour produire un artefact "modifié", une étape produit
    un nouvel ``Artifact`` distinct.

    Sérialisation déterministe : ``model_dump_json()`` produit les
    mêmes octets pour le même contenu (champs Pydantic ordonnés).
    Indispensable pour le cache d'artefacts.

    Attributs
    ---------
    id:
        Identifiant unique de l'artefact dans le contexte d'un run.
        Convention : ``"<doc_id>:<step_name>:<artifact_type>"``,
        mais le caller est libre du format tant que c'est unique
        et que ``_ID_RE`` matche.
    document_id:
        ``DocumentRef.id`` du document auquel cet artefact appartient.
    type:
        Type de l'artefact (cf. ``ArtifactType``).
    uri:
        Chemin filesystem ou URI distant vers le contenu.  ``None``
        si l'artefact est stocké inline (cas des petits artefacts
        comme un texte court produit en mémoire).  Le caller (typiquement
        ``ArtifactStore``, S7) est responsable de la résolution.
    content_hash:
        SHA-256 hex (64 chars) du contenu.  ``None`` autorisé seulement
        pour les artefacts initiaux fournis par l'utilisateur (image,
        GT) qui n'ont pas encore été lus.  Une fois calculé, immuable.
    produced_by_step:
        Nom de l'étape de pipeline qui a produit l'artefact.  ``None``
        pour les artefacts initiaux (entrées du pipeline, GT).
    provenance:
        ``ProvenanceRecord`` portant ``code_version`` et
        ``parameters_hash``.  ``None`` pour les artefacts initiaux.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=512)
    document_id: str = Field(min_length=1, max_length=256)
    type: ArtifactType
    uri: str | None = Field(default=None, max_length=2048)
    content_hash: str | None = Field(default=None, min_length=64, max_length=64)
    produced_by_step: str | None = Field(default=None, max_length=256)
    # ``provenance`` typé en str pour éviter import croisé pydantic
    # avec ProvenanceRecord ; remplacé par le vrai type via __init__
    # plus bas.
    provenance: "ProvenanceRecord | None" = Field(default=None)

    @field_validator("id", "document_id")
    @classmethod
    def _validate_filesystem_safe_id(cls, v: str) -> str:
        if not _ID_RE.match(v):
            from picarones.domain.errors import ArtifactValidationError
            raise ArtifactValidationError(
                f"id invalide : {v!r}.  "
                f"Doit matcher {_ID_RE.pattern!r} (alphanum + ``_.-:/``)."
            )
        return v

    @field_validator("content_hash")
    @classmethod
    def _validate_hex_hash(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            int(v, 16)
        except ValueError:
            from picarones.domain.errors import ArtifactValidationError
            raise ArtifactValidationError(
                f"content_hash doit être hex SHA-256 64 chars : {v!r}"
            )
        return v.lower()


# Forward reference pour ``provenance``.
from picarones.domain.provenance import ProvenanceRecord  # noqa: E402

Artifact.model_rebuild()


__all__ = [
    "Artifact",
    "ArtifactType",
    "compute_content_hash",
]
