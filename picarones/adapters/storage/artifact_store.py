"""``ArtifactStore`` — Sprint A14-S29.

Le S7 livrait ``ArtifactCache`` (in-memory, hash basique sur
inputs + step + code_version).  S29 introduit un ``ArtifactStore``
plus robuste qui adresse la critique d'audit n° 14 (« hash
multi-paramètres + reprise par hash ») :

1. **Hash multi-paramètres** : la clé canonique d'un artefact
   inclut les ``content_hash`` des inputs, le nom + version du
   model utilisé, les ``params`` du step, le ``code_version``,
   l'éventuel profil de normalisation, et l'éventuelle spec de
   projection.  Tout changement d'un paramètre éditorial invalide
   la cache.

2. **Reprise par hash** : si un artefact avec exactement la même
   clé existe déjà dans le store, le caller peut l'utiliser
   directement plutôt que de re-exécuter l'étape coûteuse.

3. **Persistance optionnelle** : ``InMemoryArtifactStore`` pour
   les tests et les workflows éphémères ; ``FilesystemArtifactStore``
   pour les longs runs où on veut survivre à un crash.

Pas de shim
-----------
``ArtifactCache`` (S7) reste exposé pour les callers qui en
dépendent en interne, mais la nouvelle API canonique est
``ArtifactStore``.  Le ``PipelineExecutor`` peut consommer un
``ArtifactStore`` via le paramètre optionnel ``artifact_store=``
au constructeur ; sans store, l'executor s'exécute comme avant
(pas d'effet de cache).

Anti-sur-ingénierie
-------------------
- Pas de TTL ni d'éviction LRU dans la version in-memory.  La
  taille est gérée par le caller (qui peut appeler ``clear()``).
- Pas de compression des payloads dans la version filesystem.
- Pas de namespacing par run — un store partagé entre runs est
  censé converger, c'est précisément la propriété de la reprise.
- Pas de support distribué (S3, GCS, …) — viendra quand un
  caller en aura concrètement besoin.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from picarones.domain.artifacts import Artifact

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Clé canonique multi-paramètres
# ──────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────
# Conteneur du store
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StoredArtifact:
    """Entrée du store : un artefact + son payload + sa clé.

    Le payload est stocké en bytes brutes — le caller décide de la
    désérialisation (texte UTF-8, ALTO XML, image PNG, etc.) en se
    basant sur ``artifact.type``.

    Attributes
    ----------
    key:
        Hash hex de la ``ArtifactKey`` qui a produit l'artefact.
    artifact:
        ``Artifact`` complet (id, type, content_hash, provenance).
    payload:
        Bytes du contenu, ou ``None`` si le store ne stocke que
        les métadonnées (cas d'un artefact dont l'``uri`` pointe
        vers un fichier externe).
    """

    key: str
    artifact: Artifact
    payload: bytes | None = None


# ──────────────────────────────────────────────────────────────────────
# Interface ABC
# ──────────────────────────────────────────────────────────────────────


class ArtifactStore(ABC):
    """Contrat abstrait d'un store d'artefacts indexé par hash.

    Implémentations livrées au S29 :

    - ``InMemoryArtifactStore`` (tests, runs éphémères) ;
    - ``FilesystemArtifactStore`` (workspaces persistants).

    Une implémentation tierce (S3, Postgres, …) est attendue post-
    livraison ; elle hérite de cette ABC et passe les tests de
    contrat.
    """

    @abstractmethod
    def get(self, key: str) -> StoredArtifact | None:
        """Récupère un artefact par sa clé hex, ou ``None``.

        Tolère les clés inexistantes — le retour ``None`` indique
        un cache miss, pas une erreur.
        """

    @abstractmethod
    def put(
        self,
        key: str,
        artifact: Artifact,
        payload: bytes | None = None,
    ) -> None:
        """Stocke un artefact sous la clé donnée.

        Convention idempotente : ``put(k, ...)`` deux fois avec la
        même clé écrase la valeur précédente sans erreur.  L'ABC
        n'impose pas de comportement en concurrence multi-process
        — chaque implémentation documente ses garanties.
        """

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        """Vrai si la clé est connue du store."""

    @abstractmethod
    def clear(self) -> None:
        """Supprime toutes les entrées du store.

        Implémentations filesystem : supprime les fichiers de
        l'index et des payloads.  Implémentations in-memory :
        vide les dicts.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Nombre d'entrées dans le store."""


# ──────────────────────────────────────────────────────────────────────
# InMemoryArtifactStore
# ──────────────────────────────────────────────────────────────────────


class InMemoryArtifactStore(ArtifactStore):
    """Store in-memory thread-safe pour tests et runs éphémères.

    Performances : O(1) en lecture/écriture.  Aucune persistance —
    toutes les données disparaissent à la sortie du process.

    Thread-safety : un ``threading.Lock`` protège les opérations
    mutantes (put, clear).  Lecture (get, __contains__, __len__)
    est sans lock car les dict Python sont atomiques par opération
    sur clé.
    """

    def __init__(self) -> None:
        self._store: dict[str, StoredArtifact] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> StoredArtifact | None:
        return self._store.get(key)

    def put(
        self,
        key: str,
        artifact: Artifact,
        payload: bytes | None = None,
    ) -> None:
        if not key:
            raise ValueError("ArtifactStore.put : key vide non autorisé")
        with self._lock:
            self._store[key] = StoredArtifact(
                key=key, artifact=artifact, payload=payload,
            )

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def keys(self) -> tuple[str, ...]:
        """Liste figée des clés connues (utile aux tests)."""
        return tuple(self._store.keys())


# ──────────────────────────────────────────────────────────────────────
# FilesystemArtifactStore
# ──────────────────────────────────────────────────────────────────────


class FilesystemArtifactStore(ArtifactStore):
    """Store persistant sur le filesystem.

    Layout
    ------

    ``<root>/``
        ``index.jsonl``                   — un JSON par ligne
                                            ``{"key": ..., "artifact_id": ...,
                                            "has_payload": bool, "type": ...,
                                            "timestamp": ISO8601}``
        ``artifacts/<key>.json``          — métadonnées de l'``Artifact``
                                            sérialisées via
                                            ``model_dump_json()``
        ``payloads/<key>.bin``            — bytes du payload (le cas
                                            échéant)

    Concurrence
    -----------
    Un ``threading.Lock`` interne protège les opérations mutantes
    dans le même process.  Multi-process : pas de garantie ; le
    layout est conçu pour qu'un read-only multi-process soit
    sûr (les fichiers individuels sont écrits atomiquement via
    ``write_text(... newline=...)`` et un rename).

    Garbage / corruption
    --------------------
    Si l'index pointe vers un fichier disparu, le ``get`` retourne
    ``None`` et logge un warning.  ``clear()`` supprime tout —
    un caller peut aussi reconstruire l'index en parsant les
    fichiers ``artifacts/*.json``.

    Pas de shim
    -----------
    Cette implémentation n'a pas de migration depuis l'``ArtifactCache``
    in-memory du S7 — c'est un store distinct, instanciable
    explicitement par un service applicatif (typiquement
    ``WorkspaceManager`` au S30+).
    """

    INDEX_FILENAME = "index.jsonl"
    ARTIFACTS_DIR = "artifacts"
    PAYLOADS_DIR = "payloads"

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / self.ARTIFACTS_DIR).mkdir(exist_ok=True)
        (self._root / self.PAYLOADS_DIR).mkdir(exist_ok=True)
        self._index_path = self._root / self.INDEX_FILENAME
        self._lock = threading.Lock()
        # In-memory index of known keys reconstructed from disk.
        # On sait qu'on est seul écrivain dans un process donné, mais
        # un autre process peut aussi écrire — on ne fait pas de
        # garantie multi-process ici.
        self._known_keys: set[str] = self._reconstruct_known_keys()

    # ──────────────────────────────────────────────────────────────
    # API ABC
    # ──────────────────────────────────────────────────────────────

    def get(self, key: str) -> StoredArtifact | None:
        if key not in self._known_keys:
            return None
        artifact_path = self._root / self.ARTIFACTS_DIR / f"{key}.json"
        if not artifact_path.exists():
            logger.warning(
                "[artifact_store] index pointe vers %s mais le fichier "
                "n'existe plus — entrée corrompue, retour None.",
                artifact_path,
            )
            return None
        try:
            artifact = Artifact.model_validate_json(
                artifact_path.read_text(encoding="utf-8"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[artifact_store] échec de désérialisation de %s : %s",
                artifact_path, exc,
            )
            return None
        payload_path = self._root / self.PAYLOADS_DIR / f"{key}.bin"
        payload = (
            payload_path.read_bytes() if payload_path.exists() else None
        )
        return StoredArtifact(key=key, artifact=artifact, payload=payload)

    def put(
        self,
        key: str,
        artifact: Artifact,
        payload: bytes | None = None,
    ) -> None:
        if not key:
            raise ValueError("ArtifactStore.put : key vide non autorisé")
        with self._lock:
            artifact_path = self._root / self.ARTIFACTS_DIR / f"{key}.json"
            tmp_path = artifact_path.with_suffix(".json.tmp")
            tmp_path.write_text(
                artifact.model_dump_json(),
                encoding="utf-8",
            )
            tmp_path.replace(artifact_path)
            if payload is not None:
                payload_path = self._root / self.PAYLOADS_DIR / f"{key}.bin"
                tmp_payload = payload_path.with_suffix(".bin.tmp")
                tmp_payload.write_bytes(payload)
                tmp_payload.replace(payload_path)
            self._append_index_line(key, artifact, payload is not None)
            self._known_keys.add(key)

    def __contains__(self, key: str) -> bool:
        return key in self._known_keys

    def clear(self) -> None:
        with self._lock:
            for sub in (self.ARTIFACTS_DIR, self.PAYLOADS_DIR):
                d = self._root / sub
                if d.exists():
                    for f in d.iterdir():
                        f.unlink()
            if self._index_path.exists():
                self._index_path.unlink()
            self._known_keys.clear()

    def __len__(self) -> int:
        return len(self._known_keys)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._known_keys)

    # ──────────────────────────────────────────────────────────────
    # Helpers internes
    # ──────────────────────────────────────────────────────────────

    def _append_index_line(
        self, key: str, artifact: Artifact, has_payload: bool,
    ) -> None:
        """Append-only JSONL : une nouvelle ligne par put.  Lit le
        rapport d'index au démarrage, recompose ``_known_keys``."""
        from datetime import datetime, timezone
        line = json.dumps(
            {
                "key": key,
                "artifact_id": artifact.id,
                "type": artifact.type.value,
                "has_payload": has_payload,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            },
            ensure_ascii=False,
        )
        with self._index_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _reconstruct_known_keys(self) -> set[str]:
        """Lit ``index.jsonl`` et reconstruit l'ensemble des clés
        connues.  Tolère les lignes corrompues (warning + skip).

        Si l'index n'existe pas, recompose depuis le contenu du
        sous-répertoire ``artifacts/`` (cas d'un store partiellement
        copié sans son index).
        """
        keys: set[str] = set()
        if self._index_path.exists():
            for line_no, raw_line in enumerate(
                self._index_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not raw_line.strip():
                    continue
                try:
                    rec = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "[artifact_store] index ligne %d corrompue, "
                        "ignorée : %s", line_no, exc,
                    )
                    continue
                if "key" in rec and isinstance(rec["key"], str):
                    keys.add(rec["key"])
        else:
            # Recompose depuis les fichiers d'artefacts.
            artifacts_dir = self._root / self.ARTIFACTS_DIR
            if artifacts_dir.exists():
                for f in artifacts_dir.iterdir():
                    if f.suffix == ".json":
                        keys.add(f.stem)
        return keys


__all__ = [
    "ArtifactKey",
    "ArtifactStore",
    "FilesystemArtifactStore",
    "InMemoryArtifactStore",
    "StoredArtifact",
]
