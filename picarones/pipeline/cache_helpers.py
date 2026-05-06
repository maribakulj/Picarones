"""Helpers de cache d'artefacts pour le ``PipelineExecutor`` — Sprint A14-S47.

Fix de l'audit #1 du rewrite ciblé : avant ce sprint,
``picarones/adapters/storage/artifact_store.py`` (S29) existait sans
être consommé par aucun runtime — promesse de « reprise par hash »
non tenue.

Ce module fournit les **fonctions pures** qui transforment un
``(PipelineStep, inputs, RunContext)`` en ``ArtifactKey`` et en clés
de stockage par output_type, pour que le ``PipelineExecutor`` puisse :

1. Avant d'exécuter un step : calculer la clé, interroger le store,
   et si toutes les sorties attendues sont présentes ET valides,
   sauter l'exécution en retournant les artefacts cachés.
2. Après une exécution réussie : persister chaque output dans le store
   sous une clé dérivée.

Stratégie de clé multi-output
-----------------------------
Un ``PipelineStep`` peut produire plusieurs ``ArtifactType``.
``ArtifactStore.put/get`` opère sur **un** Artifact à la fois.  Pour
gérer cela sans étendre l'API du store, on dérive une **clé composite**
par output_type :

::

    store_key = f"{step_hash}:{output_type.value}"

où ``step_hash`` est ``ArtifactKey(...).hash_hex()`` qui dépend des
inputs, du step et du code_version.  À la lecture, on demande au store
toutes les clés ``{step_hash}:<type>`` pour les ``output_types`` du
step ; si une seule manque, c'est un miss complet (cache partiel
n'est pas exploitable — on relance le step pour cohérence).

Pas de stockage du payload bytes
--------------------------------
On stocke uniquement les **métadonnées** ``Artifact`` (id, type,
content_hash, uri, provenance).  Le payload (texte, ALTO XML, image)
reste sur le filesystem au chemin pointé par ``Artifact.uri``.

Conséquence : si le workspace a été nettoyé entre deux runs, l'URI
cachée pointe vers un fichier disparu → cache miss (la fonction
``read_cached_outputs`` vérifie l'existence des URIs).  C'est le
comportement attendu : le store est un **cache**, pas une source de
vérité du contenu.

Anti-sur-ingénierie
-------------------
- Pas de TTL, pas d'éviction LRU.  Le caller appelle ``store.clear()``
  s'il veut forcer un re-run complet.
- Pas de support des artefacts inline (sans URI).  Si un step produit
  un artefact dont le contenu vit en RAM seulement, le cache est
  inopérant — c'est documenté.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from picarones.domain.artifact_key import ArtifactKey
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.pipeline.cache_protocol import ArtifactCachePort

if TYPE_CHECKING:
    from picarones.pipeline.spec import PipelineStep
    from picarones.pipeline.types import RunContext

logger = logging.getLogger(__name__)


def compute_step_artifact_key(
    step: "PipelineStep",
    inputs: dict[ArtifactType, Artifact],
    context: "RunContext",
) -> ArtifactKey:
    """Calcule la ``ArtifactKey`` d'un step pour le cache d'artefacts.

    La clé combine :

    - les ``content_hash`` des inputs (triés par type pour
      déterminisme — délégué à ``ArtifactKey.to_canonical_json``) ;
    - ``step.adapter_name`` ;
    - ``step.params`` (dict scalaire) ;
    - ``context.code_version``.

    Les autres champs de ``ArtifactKey`` (normalization_profile,
    projection_name, metric_version) restent ``None`` — ils sont
    spécifiques aux jonctions d'évaluation, pas aux steps de pipeline.

    La clé peut retourner ``None`` à ``hash_hex()`` si **un seul**
    input n'a pas de ``content_hash`` (cf. la convention « ne pas
    servir un résultat douteux » d'``ArtifactKey``).  Le caller doit
    tester ``key.hash_hex() is None`` avant d'utiliser la clé.
    """
    input_hashes: tuple[tuple[str, str], ...] = tuple(
        (art_type.value, artifact.content_hash or "")
        for art_type, artifact in inputs.items()
    )
    return ArtifactKey(
        input_hashes=input_hashes,
        adapter_name=step.adapter_name,
        adapter_version=None,  # adapters ne déclarent pas (encore) de version
        step_params=dict(step.params),
        code_version=context.code_version,
    )


def storage_key_for_output(step_hash: str, output_type: ArtifactType) -> str:
    """Construit la clé de stockage composite pour un output donné."""
    return f"{step_hash}:{output_type.value}"


def read_cached_outputs(
    store: ArtifactCachePort,
    step: "PipelineStep",
    step_hash: str,
) -> dict[ArtifactType, Artifact] | None:
    """Tente de lire les outputs cachés d'un step.

    Retourne ``None`` si :

    - une seule sortie attendue n'est pas dans le store
      (cache partiel) ;
    - une URI cachée pointe vers un fichier disparu
      (cache orphelin).

    Sinon, retourne le dict ``{output_type: Artifact}`` complet,
    prêt à être réinjecté dans le bag du runner.
    """
    cached: dict[ArtifactType, Artifact] = {}
    for output_type in step.output_types:
        store_key = storage_key_for_output(step_hash, output_type)
        stored = store.get(store_key)
        if stored is None:
            logger.debug(
                "[cache] miss partiel sur step %r : %s manquant.",
                step.id, output_type.value,
            )
            return None
        # Vérifie que l'URI cachée pointe vers un fichier qui existe
        # encore.  Sinon, le payload a disparu (workspace nettoyé,
        # mount débranché, etc.) — on doit re-exécuter.
        if stored.artifact.uri is not None:
            uri_path = Path(stored.artifact.uri)
            if not uri_path.exists():
                logger.debug(
                    "[cache] orphelin sur step %r : URI %s disparu.",
                    step.id, uri_path,
                )
                return None
        cached[output_type] = stored.artifact
    return cached


def write_outputs_to_cache(
    store: ArtifactCachePort,
    step: "PipelineStep",
    step_hash: str,
    outputs: dict[ArtifactType, Artifact],
) -> None:
    """Persiste tous les outputs d'un step réussi dans le store.

    Idempotent : ``store.put`` écrase silencieusement une entrée
    existante (cf. la convention de ``InMemoryArtifactStore`` et
    ``FilesystemArtifactStore``).
    """
    for output_type, artifact in outputs.items():
        store_key = storage_key_for_output(step_hash, output_type)
        store.put(store_key, artifact, payload=None)


__all__ = [
    "compute_step_artifact_key",
    "read_cached_outputs",
    "storage_key_for_output",
    "write_outputs_to_cache",
]
