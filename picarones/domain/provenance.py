"""Provenance d'un artefact

Empreinte minimale attachée à chaque ``Artifact`` produit par une
étape de pipeline.  Permet la reproductibilité : même corpus + même
``code_version`` + même ``parameters_hash`` = mêmes artefacts à hash
près.

Règle anti-sur-ingénierie : on ne déclare ici que les champs qui
ont un cas d'usage **immédiat** dans les Sprints S5-S18.  Les extras
attendus (cost, latency, model_version) seront ajoutés quand un
caller en aura concrètement besoin (probablement S15-S17 quand on
introduit les vues économiques).
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class ProvenanceRecord(BaseModel):
    """Empreinte de production d'un artefact.

    Immuable (``frozen=True``) : un artefact ne change pas de
    provenance après création — pour modifier une provenance, on crée
    un nouvel ``Artifact`` qui référence le précédent via
    ``parent_artifact_ids``.

    Attributs
    ---------
    timestamp:
        Date/heure UTC de production.  Défaut : ``utcnow()`` au
        moment de l'instanciation.
    code_version:
        Version du code Picarones qui a produit l'artefact.
        Typiquement ``picarones.__version__`` (au format setuptools_scm
        ``1.2.3.dev4+g<sha>`` hors release tag).  Stocké comme str
        opaque pour ne pas imposer un format particulier.
    parameters_hash:
        Hash SHA-256 hex (64 chars) des paramètres de l'étape qui a
        produit l'artefact.  Permet de détecter qu'on a relancé la
        même étape avec d'autres params (cf. cache d'artefacts du
        Sprint S7).  ``None`` autorisé pour les artefacts initiaux
        (image fournie par l'utilisateur, GT lue depuis le corpus).
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
    )
    code_version: str
    parameters_hash: str | None = None

    def is_compatible_with(self, other: "ProvenanceRecord") -> bool:
        """Deux artefacts produits par le **même contexte de calcul**.

        Utilisé par le cache d'artefacts (Sprint S7) pour décider si
        une étape peut être sautée.  Le timestamp n'entre pas dans la
        comparaison — seule la combinaison ``(code_version,
        parameters_hash)`` détermine la compatibilité de cache.
        """
        return (
            self.code_version == other.code_version
            and self.parameters_hash == other.parameters_hash
        )


__all__ = ["ProvenanceRecord"]
