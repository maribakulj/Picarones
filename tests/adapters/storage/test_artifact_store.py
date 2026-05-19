"""Sprint A14-S29 — ``ArtifactStore`` + ``ArtifactKey``.

Tests du store et du hash multi-paramètres introduits par S29
pour adresser la critique d'audit n° 14 (« hash multi-paramètres
+ reprise par hash »).

Couvre :

1. ``ArtifactKey`` :
   - frozen dataclass ;
   - sérialisation JSON canonique déterministe ;
   - hash hex SHA-256 stable cross-platform ;
   - sensibilité à chaque champ (changement → hash change) ;
   - ``hash_hex()`` retourne ``None`` si un input_hash est manquant.

2. ``InMemoryArtifactStore`` :
   - get/put/contains/clear/len ;
   - rejet des clés vides ;
   - put idempotent (écrase silencieusement) ;
   - thread-safety basique (pas de race évidente).

3. ``FilesystemArtifactStore`` :
   - get/put/contains/clear/len ;
   - persistance disque (relire après ré-instanciation) ;
   - layout (index.jsonl + artifacts/<key>.json + payloads/<key>.bin) ;
   - tolérance aux fichiers manquants (warning + None) ;
   - reconstruction depuis artifacts/ si index manquant ;
   - écriture atomique via .tmp + rename.

4. Contrat ABC : les deux implémentations passent les mêmes tests
   de comportement.
"""

from __future__ import annotations

import json
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from picarones.adapters.storage import (
    ArtifactKey,
    ArtifactStore,
    ArtifactStoreError,
    FilesystemArtifactStore,
    InMemoryArtifactStore,
    StoredArtifact,
)
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.provenance import ProvenanceRecord


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_artifact(
    artifact_id: str = "d1:ocr:raw_text",
    document_id: str = "d1",
    artifact_type: ArtifactType = ArtifactType.RAW_TEXT,
    content_hash: str | None = "0" * 64,
) -> Artifact:
    return Artifact(
        id=artifact_id,
        document_id=document_id,
        type=artifact_type,
        content_hash=content_hash,
        produced_by_step="ocr",
        provenance=ProvenanceRecord(
            code_version="1.0.0",
            parameters_hash="a" * 64,
        ),
    )


def _basic_key() -> ArtifactKey:
    return ArtifactKey(
        input_hashes=(("image", "f" * 64),),
        adapter_name="tesseract",
        adapter_version="5.3.0",
        step_params={"lang": "fra"},
        code_version="1.0.0",
    )


# ──────────────────────────────────────────────────────────────────────
# ArtifactKey
# ──────────────────────────────────────────────────────────────────────


class TestArtifactKeyDataclass:
    def test_default_values(self) -> None:
        k = ArtifactKey()
        assert k.input_hashes == ()
        assert k.adapter_name == ""
        assert k.adapter_version is None
        assert k.step_params == {}
        assert k.code_version == ""
        assert k.normalization_profile is None
        assert k.projection_name is None
        assert k.projection_params == {}
        assert k.metric_version is None

    def test_frozen(self) -> None:
        k = _basic_key()
        with pytest.raises(FrozenInstanceError):
            k.adapter_name = "different"  # type: ignore[misc]


class TestArtifactKeyCanonicalJson:
    def test_deterministic(self) -> None:
        """Deux clés équivalentes produisent le même JSON."""
        k1 = ArtifactKey(
            input_hashes=(("image", "a" * 64),),
            adapter_name="x",
            step_params={"a": 1, "b": 2},
            code_version="v1",
        )
        k2 = ArtifactKey(
            input_hashes=(("image", "a" * 64),),
            adapter_name="x",
            step_params={"b": 2, "a": 1},  # ordre différent
            code_version="v1",
        )
        assert k1.to_canonical_json() == k2.to_canonical_json()

    def test_inputs_sorted(self) -> None:
        """L'ordre des input_hashes ne change pas le JSON canonique."""
        k1 = ArtifactKey(
            input_hashes=(("image", "a" * 64), ("text", "b" * 64)),
            adapter_name="x",
            code_version="v",
        )
        k2 = ArtifactKey(
            input_hashes=(("text", "b" * 64), ("image", "a" * 64)),
            adapter_name="x",
            code_version="v",
        )
        assert k1.to_canonical_json() == k2.to_canonical_json()

    def test_unicode_preserved(self) -> None:
        k = ArtifactKey(
            input_hashes=(),
            adapter_name="modèle",
            step_params={"prompt": "français médiéval"},
            code_version="v",
        )
        canonical = k.to_canonical_json()
        assert "modèle" in canonical
        assert "français médiéval" in canonical


class TestArtifactKeyHash:
    def test_hash_is_64_hex_chars(self) -> None:
        h = _basic_key().hash_hex()
        assert h is not None
        assert len(h) == 64
        int(h, 16)  # valide hex

    def test_hash_stable_across_calls(self) -> None:
        k = _basic_key()
        assert k.hash_hex() == k.hash_hex()

    def test_hash_changes_with_adapter_version(self) -> None:
        k1 = ArtifactKey(
            input_hashes=(("image", "a" * 64),),
            adapter_name="x",
            adapter_version="1.0",
            code_version="v",
        )
        k2 = ArtifactKey(
            input_hashes=(("image", "a" * 64),),
            adapter_name="x",
            adapter_version="2.0",  # change
            code_version="v",
        )
        assert k1.hash_hex() != k2.hash_hex()

    def test_hash_changes_with_step_params(self) -> None:
        k1 = ArtifactKey(
            input_hashes=(("image", "a" * 64),),
            adapter_name="x",
            step_params={"lang": "fra"},
            code_version="v",
        )
        k2 = ArtifactKey(
            input_hashes=(("image", "a" * 64),),
            adapter_name="x",
            step_params={"lang": "eng"},  # change
            code_version="v",
        )
        assert k1.hash_hex() != k2.hash_hex()

    def test_hash_changes_with_normalization(self) -> None:
        k1 = ArtifactKey(
            input_hashes=(("image", "a" * 64),),
            adapter_name="x",
            code_version="v",
        )
        k2 = ArtifactKey(
            input_hashes=(("image", "a" * 64),),
            adapter_name="x",
            code_version="v",
            normalization_profile="medieval_french",
        )
        assert k1.hash_hex() != k2.hash_hex()

    def test_hash_changes_with_projection(self) -> None:
        k1 = ArtifactKey(
            input_hashes=(("alto", "a" * 64),),
            adapter_name="x",
            code_version="v",
        )
        k2 = ArtifactKey(
            input_hashes=(("alto", "a" * 64),),
            adapter_name="x",
            code_version="v",
            projection_name="alto_to_text",
        )
        assert k1.hash_hex() != k2.hash_hex()

    def test_hash_returns_none_if_input_hash_missing(self) -> None:
        # Cas pathologique : un tuple avec hash vide.
        k = ArtifactKey(
            input_hashes=(("image", ""),),
            adapter_name="x",
            code_version="v",
        )
        assert k.hash_hex() is None

    def test_empty_inputs_yields_valid_hash(self) -> None:
        """Pas d'inputs (tuple vide) ne signifie pas missing — c'est
        valide pour les artefacts sans dépendance externe."""
        k = ArtifactKey(
            adapter_name="x",
            code_version="v",
        )
        assert k.hash_hex() is not None


# ──────────────────────────────────────────────────────────────────────
# InMemoryArtifactStore
# ──────────────────────────────────────────────────────────────────────


class _SharedStoreContract:
    """Mixin abstrait : partage les tests entre InMemory et Filesystem."""

    def make_store(self, tmp_path: Path) -> ArtifactStore:
        raise NotImplementedError

    def test_empty_store(self, tmp_path: Path) -> None:
        store = self.make_store(tmp_path)
        assert len(store) == 0
        assert "any-key" not in store
        assert store.get("any-key") is None

    def test_put_then_get(self, tmp_path: Path) -> None:
        store = self.make_store(tmp_path)
        artifact = _make_artifact()
        store.put("k1", artifact, payload=b"hello world")
        assert "k1" in store
        assert len(store) == 1
        retrieved = store.get("k1")
        assert retrieved is not None
        assert retrieved.key == "k1"
        assert retrieved.artifact.id == artifact.id
        assert retrieved.payload == b"hello world"

    def test_put_without_payload(self, tmp_path: Path) -> None:
        store = self.make_store(tmp_path)
        artifact = _make_artifact()
        store.put("k1", artifact, payload=None)
        retrieved = store.get("k1")
        assert retrieved is not None
        assert retrieved.payload is None

    def test_put_idempotent_overwrites(self, tmp_path: Path) -> None:
        store = self.make_store(tmp_path)
        store.put("k1", _make_artifact(), payload=b"v1")
        store.put("k1", _make_artifact(), payload=b"v2")
        assert len(store) == 1
        assert store.get("k1").payload == b"v2"

    def test_clear(self, tmp_path: Path) -> None:
        store = self.make_store(tmp_path)
        store.put("k1", _make_artifact(), payload=b"x")
        store.put("k2", _make_artifact(), payload=b"y")
        assert len(store) == 2
        store.clear()
        assert len(store) == 0
        assert "k1" not in store
        assert "k2" not in store

    def test_empty_key_rejected(self, tmp_path: Path) -> None:
        store = self.make_store(tmp_path)
        with pytest.raises(ArtifactStoreError, match="vide"):
            store.put("", _make_artifact(), payload=b"x")

    def test_multiple_artifacts_independent(self, tmp_path: Path) -> None:
        store = self.make_store(tmp_path)
        a1 = _make_artifact(artifact_id="d1:art1", content_hash="1" * 64)
        a2 = _make_artifact(artifact_id="d2:art2", content_hash="2" * 64)
        store.put("k1", a1, payload=b"alpha")
        store.put("k2", a2, payload=b"beta")
        assert store.get("k1").artifact.id == "d1:art1"
        assert store.get("k2").artifact.id == "d2:art2"
        assert store.get("k1").payload == b"alpha"
        assert store.get("k2").payload == b"beta"


class TestInMemoryArtifactStore(_SharedStoreContract):
    def make_store(self, tmp_path: Path) -> ArtifactStore:
        return InMemoryArtifactStore()

    def test_keys_helper(self) -> None:
        store = InMemoryArtifactStore()
        store.put("k1", _make_artifact(), payload=b"x")
        store.put("k2", _make_artifact(), payload=b"y")
        keys = store.keys()
        assert set(keys) == {"k1", "k2"}

    def test_thread_safe_disjoint_keys(self) -> None:
        """100 threads écrivent chacun 10 clés disjointes → 1000."""
        store = InMemoryArtifactStore()
        artifact = _make_artifact()

        def writer(i: int) -> None:
            for j in range(10):
                store.put(f"k_{i}_{j}", artifact, payload=b"x")

        threads = [
            threading.Thread(target=writer, args=(i,))
            for i in range(100)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(store) == 1000

    def test_thread_safe_concurrent_overwrites_same_key(self) -> None:
        """Sprint S56 (audit #29) : test de concurrence sur la MÊME
        clé.  Avec 50 threads qui put la même clé en parallèle, le
        store doit converger sur une valeur (last-write-wins) sans
        crash, sans corruption, sans clé fantôme."""
        store = InMemoryArtifactStore()

        def writer(i: int) -> None:
            for _ in range(20):
                store.put(
                    "shared_key",
                    _make_artifact(artifact_id=f"d{i}:art"),
                    payload=f"payload_{i}".encode(),
                )

        threads = [
            threading.Thread(target=writer, args=(i,))
            for i in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Une seule clé "shared_key" — pas de duplication.
        assert len(store) == 1
        # Le stored est cohérent (artifact + payload appartiennent
        # au même writer, pas un mix).
        stored = store.get("shared_key")
        assert stored is not None
        # L'id de l'artefact détermine quel writer a gagné ; le
        # payload doit correspondre au même writer.
        assert stored.artifact.id.startswith("d")
        winner_idx = stored.artifact.id.split(":")[0][1:]
        assert stored.payload == f"payload_{winner_idx}".encode()


class TestFilesystemArtifactStore(_SharedStoreContract):
    def make_store(self, tmp_path: Path) -> ArtifactStore:
        return FilesystemArtifactStore(tmp_path / "store")

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        """Le store sait re-charger ses entrées après ré-instanciation."""
        root = tmp_path / "store"
        s1 = FilesystemArtifactStore(root)
        s1.put("k1", _make_artifact(), payload=b"persisted")

        # Nouvelle instance pointant vers le même root.
        s2 = FilesystemArtifactStore(root)
        assert "k1" in s2
        assert s2.get("k1").payload == b"persisted"
        assert s2.get("k1").artifact.id == "d1:ocr:raw_text"

    def test_layout(self, tmp_path: Path) -> None:
        """Vérifie le layout sur disque."""
        root = tmp_path / "store"
        s = FilesystemArtifactStore(root)
        s.put("k1", _make_artifact(), payload=b"hello")
        assert (root / "index.jsonl").exists()
        assert (root / "artifacts" / "k1.json").exists()
        assert (root / "payloads" / "k1.bin").exists()
        # L'index contient une ligne JSON.
        index_lines = (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(index_lines) == 1
        rec = json.loads(index_lines[0])
        assert rec["key"] == "k1"
        assert rec["artifact_id"] == "d1:ocr:raw_text"
        assert rec["has_payload"] is True

    def test_artifact_metadata_preserved(self, tmp_path: Path) -> None:
        """Les métadonnées de l'Artifact survivent au round-trip."""
        root = tmp_path / "store"
        s = FilesystemArtifactStore(root)
        artifact = Artifact(
            id="d1:complex",
            document_id="d1",
            type=ArtifactType.ALTO_XML,
            content_hash="b" * 64,
            uri="/tmp/some.xml",
            produced_by_step="alto_step",
            provenance=ProvenanceRecord(
                code_version="2.5.1",
                parameters_hash="c" * 64,
            ),
        )
        s.put("k1", artifact, payload=b"<alto/>")
        s2 = FilesystemArtifactStore(root)
        retrieved = s2.get("k1")
        assert retrieved is not None
        assert retrieved.artifact.id == artifact.id
        assert retrieved.artifact.type == ArtifactType.ALTO_XML
        assert retrieved.artifact.content_hash == artifact.content_hash
        assert retrieved.artifact.uri == "/tmp/some.xml"
        assert retrieved.artifact.provenance.code_version == "2.5.1"
        assert retrieved.payload == b"<alto/>"

    def test_corrupted_index_line_skipped(self, tmp_path: Path) -> None:
        """Une ligne corrompue de l'index ne plante pas le store."""
        root = tmp_path / "store"
        s1 = FilesystemArtifactStore(root)
        s1.put("k1", _make_artifact(), payload=b"x")
        # Corrompre l'index par ajout d'une ligne garbage.
        (root / "index.jsonl").open("a", encoding="utf-8").write(
            "this is not json\n"
        )
        s2 = FilesystemArtifactStore(root)
        assert "k1" in s2  # Toujours présent malgré ligne corrompue
        assert s2.get("k1") is not None

    def test_artifact_file_missing_returns_none_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Si l'index pointe vers un fichier supprimé, get retourne
        None avec warning explicite (pas un crash)."""
        root = tmp_path / "store"
        s = FilesystemArtifactStore(root)
        s.put("k1", _make_artifact(), payload=b"x")
        # Supprimer le fichier d'artefact pour simuler corruption.
        (root / "artifacts" / "k1.json").unlink()
        result = s.get("k1")
        assert result is None
        assert any(
            "n'existe plus" in r.message for r in caplog.records
        )

    def test_reconstruct_from_artifacts_dir_when_index_missing(
        self, tmp_path: Path,
    ) -> None:
        """Si index.jsonl est manquant, reconstruction depuis
        artifacts/."""
        root = tmp_path / "store"
        s1 = FilesystemArtifactStore(root)
        s1.put("k1", _make_artifact(), payload=b"a")
        s1.put("k2", _make_artifact(), payload=b"b")
        # Effacer l'index, garder les artefacts.
        (root / "index.jsonl").unlink()
        s2 = FilesystemArtifactStore(root)
        assert "k1" in s2
        assert "k2" in s2
        assert len(s2) == 2

    def test_clear_removes_all_files(self, tmp_path: Path) -> None:
        root = tmp_path / "store"
        s = FilesystemArtifactStore(root)
        s.put("k1", _make_artifact(), payload=b"x")
        s.put("k2", _make_artifact(), payload=b"y")
        s.clear()
        assert len(s) == 0
        # Les sous-répertoires existent toujours, juste vides.
        assert (root / "artifacts").exists()
        assert list((root / "artifacts").iterdir()) == []
        assert list((root / "payloads").iterdir()) == []
        assert not (root / "index.jsonl").exists()


# ──────────────────────────────────────────────────────────────────────
# Intégration ArtifactKey + Store
# ──────────────────────────────────────────────────────────────────────


class TestKeyStoreIntegration:
    def test_store_keyed_by_artifact_key_hash(self, tmp_path: Path) -> None:
        """Le pattern d'usage attendu : compute key, then put with
        key.hash_hex() as the store key."""
        store = InMemoryArtifactStore()
        key = _basic_key()
        hash_hex = key.hash_hex()
        assert hash_hex is not None
        store.put(hash_hex, _make_artifact(), payload=b"raw text")
        assert hash_hex in store
        retrieved = store.get(hash_hex)
        assert retrieved is not None
        assert retrieved.payload == b"raw text"

    def test_different_params_yield_different_keys_and_no_collision(
        self, tmp_path: Path,
    ) -> None:
        """Deux clés conceptuellement différentes ne collisent pas."""
        store = InMemoryArtifactStore()
        k_fra = ArtifactKey(
            input_hashes=(("image", "f" * 64),),
            adapter_name="tess",
            step_params={"lang": "fra"},
            code_version="v",
        )
        k_eng = ArtifactKey(
            input_hashes=(("image", "f" * 64),),
            adapter_name="tess",
            step_params={"lang": "eng"},
            code_version="v",
        )
        store.put(k_fra.hash_hex(), _make_artifact(artifact_id="art:fra"))
        store.put(k_eng.hash_hex(), _make_artifact(artifact_id="art:eng"))
        assert len(store) == 2
        assert store.get(k_fra.hash_hex()).artifact.id == "art:fra"
        assert store.get(k_eng.hash_hex()).artifact.id == "art:eng"


# ──────────────────────────────────────────────────────────────────────
# StoredArtifact dataclass
# ──────────────────────────────────────────────────────────────────────


class TestStoredArtifactDataclass:
    def test_frozen(self) -> None:
        sa = StoredArtifact(
            key="k", artifact=_make_artifact(), payload=b"x",
        )
        with pytest.raises(FrozenInstanceError):
            sa.payload = b"y"  # type: ignore[misc]
