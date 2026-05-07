"""``CorpusSpec`` — Sprint A14-S4.

Description **immuable et déclarative** d'un corpus à benchmarker.
Construit par un adapter de corpus (``picarones.adapters.corpus.*``),
consommé par les services applicatifs et le pipeline executor.

Différence avec ``picarones.evaluation.corpus.Corpus`` :
``CorpusSpec`` est volontairement minimaliste — il décrit la
**structure** d'un corpus (liste de documents + métadonnées
contextuelles).  La logique de chargement, parsing, détection des
patterns de nommage GT vit ailleurs (dans ``adapters/corpus/``,
puis ``app/services/corpus_service.py`` au S20).

Au Sprint S10, un convertisseur ``CorpusSpec ↔ Corpus`` permettra
au nouveau code d'utiliser les fixtures historiques sans
réimplémentation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from picarones.domain.documents import DocumentRef


class CorpusSpec(BaseModel):
    """Description immuable d'un corpus à benchmarker.

    Attributs
    ---------
    name:
        Nom court du corpus (utilisé dans les rapports, le cache,
        les logs).  Ex : ``"bnf_etat_civil_xviiie"``.
    documents:
        Liste ordonnée des ``DocumentRef``.  L'ordre est respecté
        par le runner (utile pour des comparaisons reproductibles).
        Les ``id`` ne peuvent pas être dupliqués.
    metadata:
        Dictionnaire libre de contexte.  Conventions actuelles :

        - ``"language"`` : ``"fr"`` ou ``"en"`` (utilisé par le delta
          Flesch et les profils de normalisation).
        - ``"period"`` : étiquette éditoriale (``"medieval"``,
          ``"early_modern"``, ``"modern_archives"``).
        - ``"source"`` : ``"local"``, ``"iiif"``, ``"htr_united"``, ...

        Pas de validation stricte sur les clés — les conventions
        évolueront (cf. ``BACKLOG_POST_LIVRAISON.md``).

    Note méthodologique
    -------------------
    Un ``CorpusSpec`` ne contient **pas** la racine du filesystem
    (les ``DocumentRef.image_uri`` doivent être absolus ou résoluble
    sans contexte).  C'est volontaire : ça permet à un service
    applicatif de réécrire les chemins (sandbox utilisateur, cache,
    etc.) sans muter le ``CorpusSpec`` lui-même.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    documents: tuple[DocumentRef, ...] = Field(default_factory=tuple)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("documents")
    @classmethod
    def _validate_unique_doc_ids(
        cls, v: tuple[DocumentRef, ...],
    ) -> tuple[DocumentRef, ...]:
        seen: set[str] = set()
        for doc in v:
            if doc.id in seen:
                from picarones.domain.errors import CorpusSpecError
                raise CorpusSpecError(
                    f"document id dupliqué : {doc.id!r}.  "
                    "Les id de DocumentRef doivent être uniques au sein "
                    "d'un CorpusSpec."
                )
            seen.add(doc.id)
        return v

    def __len__(self) -> int:
        return len(self.documents)

    def doc_by_id(self, doc_id: str) -> DocumentRef | None:
        """Retourne le ``DocumentRef`` correspondant ou ``None``."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None


__all__ = ["CorpusSpec"]
