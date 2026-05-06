"""``ArtifactKey`` — Sprint A14-S29, migré dans ``domain/`` au S47.

Le S29 livrait ``ArtifactKey`` dans ``picarones/adapters/storage/``
avec le store qui le consomme.  Au S47 (branchement du store dans
``PipelineExecutor``), on découvre que ``ArtifactKey`` est un type
**pur** (dataclass frozen, méthodes de sérialisation déterministe,
calcul de hash) — il appartient au cercle 1 (``domain/``).

Migration : ``ArtifactKey`` vit désormais ici.
``picarones.adapters.storage.ArtifactKey`` reste exposé en re-export
(alias de chemin pur, pas un shim).

Pourquoi cette migration
------------------------
La couche ``pipeline/`` doit pouvoir calculer une clé pour interroger
le cache (cf. ``pipeline/cache_helpers.py``), mais ne peut pas
importer depuis ``adapters/`` (couche plus externe).  L'inversion
de dépendance demandait un Protocol.  Plus simple et plus correct :
constater que ``ArtifactKey`` est un type domaine et le placer dans
le bon cercle.

``StoredArtifact``, ``ArtifactStore`` (ABC), ``InMemoryArtifactStore``,
``FilesystemArtifactStore`` restent dans ``adapters/storage/`` — ce
sont des infrastructures, pas des types purs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ArtifactKey:
    """Composition immuable de tous les paramètres qui déterminent
    l'identité d'un artefact dans le store.

    Sérialisable JSON déterministe via ``to_canonical_json``.

    Attributes
    ----------
    input_hashes:
        Tuple ``((type, content_hash), ...)`` des inputs, trié par
        type.  ``None`` ou vide → la clé n'est pas calculable
        (cas d'un input sans content_hash).
    adapter_name:
        ``step.adapter_name`` (ex : ``"tesseract"``,
        ``"openai:gpt-4o"``).
    adapter_version:
        Version du modèle / binaire de l'adapter.  ``None`` si
        l'adapter ne sait pas la fournir (warning loggé une fois).
    step_params:
        Dict ``{name: scalar}`` du step, sérialisé en JSON canonique
        (clés triées).
    code_version:
        Version du code Picarones (cf. ``RunContext.code_version``).
    normalization_profile:
        Profil de normalisation appliqué en aval (le cas échéant).
        Pour les jonctions textuelles avec normalisation.
    projection_name:
        Nom du projecteur appliqué (le cas échéant).
    projection_params:
        Params du projecteur (le cas échéant).
    metric_version:
        Version du module de métriques (rare ; reporté à la phase
        où on aura un versioning explicite des métriques).

    Notes
    -----
    Frozen dataclass : aucune mutation possible.  Le hash canonique
    est calculé à la demande via ``hash_hex()``.
    """

    input_hashes: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    adapter_name: str = ""
    adapter_version: str | None = None
    step_params: dict[str, str | int | float | bool] = field(default_factory=dict)
    code_version: str = ""
    normalization_profile: str | None = None
    projection_name: str | None = None
    projection_params: dict[str, str | int | float | bool] = field(
        default_factory=dict,
    )
    metric_version: str | None = None

    def to_canonical_json(self) -> str:
        """Sérialise la clé en JSON déterministe.

        - Clés du dict triées (``sort_keys=True``).
        - ``ensure_ascii=False`` pour préserver l'Unicode brut.
        - Séparateurs compacts pour minimiser les variations de
          whitespace entre OS.
        """
        # Trier les input_hashes par type pour déterminisme
        # cross-platform (les Python du même version trient les
        # tuples par leur premier élément, mais on l'explicite).
        sorted_inputs = sorted(self.input_hashes)
        payload = {
            "inputs": sorted_inputs,
            "adapter": self.adapter_name,
            "adapter_version": self.adapter_version,
            "step_params": self.step_params,
            "code_version": self.code_version,
            "normalization_profile": self.normalization_profile,
            "projection_name": self.projection_name,
            "projection_params": self.projection_params,
            "metric_version": self.metric_version,
        }
        return json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def hash_hex(self) -> str | None:
        """Calcule la clé hex SHA-256 (64 chars).

        Retourne ``None`` si **un seul** ``input_hash`` est ``None``
        ou vide — convention « ne pas servir un résultat douteux ».
        Les autres champs peuvent être ``None`` (ils sont sérialisés
        comme ``null`` dans le JSON canonique → entrent dans le hash).
        """
        for _, h in self.input_hashes:
            if h is None or h == "":
                return None
        canonical = self.to_canonical_json()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["ArtifactKey"]
