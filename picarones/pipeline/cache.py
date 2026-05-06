"""``ArtifactCache`` minimal in-memory — Sprint A14-S7.

Cache d'outputs d'étape indexé par ``(content_hashes des inputs +
spec hash + code_version)``.  Permet de sauter une étape coûteuse
(typiquement un appel LLM cloud) si elle a déjà été exécutée avec
exactement les mêmes inputs et la même spec.

S7 livre la couche de calcul ; le branchement avec
``PipelineExecutor`` viendra quand un cas d'usage concret de
réutilisation se présentera (probablement S8 quand on aura
l'orchestration corpus-wide qui peut bénéficier d'un cache pour
les retries idempotents).

Garde-fous
----------
- Si **un seul** input n'a pas de ``content_hash``, la clé n'est
  pas calculable → ``compute_key`` retourne ``None`` →
  ``get`` retourne ``None`` (équivalent à un cache miss).  Pas de
  fallback hasardeux qui pourrait servir des résultats faux.
- Pas de TTL, pas d'éviction LRU — c'est un cache in-memory
  simple, taille gardée par le caller (qui peut appeler ``clear()``
  s'il veut libérer la mémoire).
- Pas de persistance disque pour S7.  Si un caller en a besoin,
  on l'ajoutera quand le besoin sera concret (S20+ probablement).
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.pipeline_spec import PipelineStep


class ArtifactCache:
    """Cache in-memory d'outputs d'étape.

    Thread-safe en lecture/écriture **après** l'init (les opérations
    mutantes se font sur un dict — Python GIL garantit l'atomicité
    des set/del sur un dict).  Pas de mécanisme de freeze technique.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[ArtifactType, Artifact]] = {}

    # ──────────────────────────────────────────────────────────────────
    # Calcul de clé
    # ──────────────────────────────────────────────────────────────────

    def compute_key(
        self,
        step: PipelineStep,
        input_artifacts: dict[ArtifactType, Artifact],
        code_version: str,
    ) -> str | None:
        """Calcule la clé canonique du cache pour cette exécution.

        Retourne ``None`` si **un seul** input n'a pas de
        ``content_hash`` — convention "ne sert pas un résultat
        douteux".

        La clé combine :

        - les ``content_hash`` triés par ``ArtifactType.value``,
        - le hash de la spec du step (sérialisée JSON déterministe),
        - le ``code_version``.

        Deux exécutions avec exactement les mêmes inputs (au sens
        ``content_hash``), la même spec et la même version de code
        produisent la même clé.
        """
        # 1. Inputs : (type → content_hash), tous obligatoires.
        try:
            input_hashes = sorted(
                (t.value, input_artifacts[t].content_hash)
                for t in input_artifacts
            )
        except KeyError:
            return None
        if any(h is None for _, h in input_hashes):
            return None

        # 2. Spec du step : on hash la sérialisation pydantic de
        #    PipelineStep (params, kind, adapter_name, etc.).  Tout
        #    changement dans la spec invalide le cache.
        step_payload = step.model_dump(mode="json")
        step_blob = json.dumps(
            step_payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # 3. Composition.
        material = json.dumps(
            {
                "inputs": input_hashes,
                "step": step_blob,
                "code_version": code_version,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    # ──────────────────────────────────────────────────────────────────
    # Get / Put / Clear
    # ──────────────────────────────────────────────────────────────────

    def get(self, key: str | None) -> dict[ArtifactType, Artifact] | None:
        """Retourne les outputs cachés pour la clé, ou ``None``.

        Tolère ``key=None`` pour faciliter le pattern :

            key = cache.compute_key(...)
            cached = cache.get(key)
            if cached is not None:
                return cached
        """
        if key is None:
            return None
        return self._store.get(key)

    def put(
        self,
        key: str | None,
        outputs: dict[ArtifactType, Artifact],
    ) -> None:
        """Stocke les outputs sous la clé donnée.  No-op si
        ``key=None`` (alignement avec la convention "ne pas servir
        un résultat douteux")."""
        if key is None:
            return
        self._store[key] = dict(outputs)  # copie défensive

    def clear(self) -> None:
        """Vide complètement le cache."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def keys(self) -> Iterable[str]:
        """Liste des clés actuellement en cache (utile pour les tests)."""
        return list(self._store.keys())


__all__ = ["ArtifactCache"]
