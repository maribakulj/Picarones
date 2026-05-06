"""Sprint A14-S47 — branchement ``ArtifactStore`` dans ``PipelineExecutor``.

Fix de l'audit #1 : avant ce sprint, ``ArtifactStore`` (S29) était
livré comme module standalone sans consommateur runtime — la promesse
de « reprise par hash » n'était pas tenue.

Tests vérifient :

1. Sans ``artifact_store`` injecté : comportement identique à l'avant
   (pas de régression sur les 115 tests existants).
2. Avec store : premier run → exécution normale + persistance.
3. Avec store : second run même inputs+spec+code_version → cache hit,
   ``StepResult.duration_seconds=0.0``, adapter NON appelé.
4. Cache miss si un seul ``content_hash`` manque sur les inputs.
5. Cache miss si un output_type promis n'est pas dans le store
   (cache partiel rejeté).
6. Cache miss si une URI cachée pointe vers un fichier disparu
   (cache orphelin → re-run).
7. Cache miss si ``code_version`` change (key change).
8. Cache miss si ``step.params`` change.
9. Cache hit ne re-exécute PAS l'adapter (vérifie via spy).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picarones.adapters.storage import (
    FilesystemArtifactStore,
    InMemoryArtifactStore,
)
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.domain.documents import DocumentRef
from picarones.pipeline.executor import PipelineExecutor
from picarones.domain.pipeline_spec import PipelineSpec, PipelineStep
from picarones.pipeline.types import RunContext


# ──────────────────────────────────────────────────────────────────────
# Adapter de test : compte ses appels et écrit un fichier déterministe
# ──────────────────────────────────────────────────────────────────────


class _CountingOCRAdapter:
    """Stub OCR qui produit RAW_TEXT et compte ses exécutions.

    Écrit le texte sur disque (URI valide) pour que le check
    ``read_cached_outputs`` (vérification existence URI) trouve le
    fichier.
    """

    name = "counting_ocr"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def __init__(self, output_dir: Path, response_text: str = "hello") -> None:
        self.output_dir = output_dir
        self.response_text = response_text
        self.call_count = 0

    def execute(self, inputs, params, context):
        self.call_count += 1
        out_path = self.output_dir / f"{context.document_id}.txt"
        out_path.write_text(self.response_text, encoding="utf-8")
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:{self.name}:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                content_hash="b" * 64,
                produced_by_step="ocr",
                uri=str(out_path),
            ),
        }


def _make_spec() -> PipelineSpec:
    return PipelineSpec(
        name="cache_test",
        initial_inputs=(ArtifactType.IMAGE,),
        steps=(
            PipelineStep(
                id="ocr",
                kind="ocr",
                adapter_name="counting_ocr",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),
        ),
    )


def _make_initial_inputs(image_uri: str = "/tmp/img.png") -> dict:
    return {
        ArtifactType.IMAGE: Artifact(
            id="d1:image",
            document_id="d1",
            type=ArtifactType.IMAGE,
            content_hash="a" * 64,
            uri=image_uri,
        ),
    }


def _make_context(code_version: str = "1.0.0") -> RunContext:
    return RunContext(
        document_id="d1",
        code_version=code_version,
        pipeline_name="cache_test",
    )


# ──────────────────────────────────────────────────────────────────────
# Comportement par défaut (sans store) — pas de régression
# ──────────────────────────────────────────────────────────────────────


class TestNoStoreNoRegression:
    def test_executor_works_without_store(self, tmp_path: Path) -> None:
        adapter = _CountingOCRAdapter(tmp_path)
        executor = PipelineExecutor(adapter_resolver=lambda n: adapter)
        # Pas d'artifact_store → comportement identique à l'avant-S47.
        result = executor.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=_make_initial_inputs(),
            context=_make_context(),
        )
        assert result.succeeded
        assert adapter.call_count == 1

    def test_rejects_non_store_in_constructor(self) -> None:
        from picarones.domain.errors import PicaronesError
        with pytest.raises(PicaronesError, match="artifact_store"):
            PipelineExecutor(
                adapter_resolver=lambda n: None,
                artifact_store="not a store",  # type: ignore[arg-type]
            )


# ──────────────────────────────────────────────────────────────────────
# Cache hit — second run avec mêmes inputs+spec+code_version
# ──────────────────────────────────────────────────────────────────────


class TestCacheHit:
    def test_second_run_hits_cache(self, tmp_path: Path) -> None:
        adapter = _CountingOCRAdapter(tmp_path)
        store = InMemoryArtifactStore()
        executor = PipelineExecutor(
            adapter_resolver=lambda n: adapter,
            artifact_store=store,
        )

        # Premier run : exécute, persiste.
        result1 = executor.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=_make_initial_inputs(),
            context=_make_context(),
        )
        assert result1.succeeded
        assert adapter.call_count == 1
        assert len(store) >= 1  # au moins une entrée persistée

        # Second run identique : doit hit le cache.
        result2 = executor.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=_make_initial_inputs(),
            context=_make_context(),
        )
        assert result2.succeeded
        # L'adapter n'a PAS été ré-appelé.
        assert adapter.call_count == 1, (
            "Cache hit raté : l'adapter a été ré-exécuté."
        )
        # Le step est marqué succeeded avec duration ≈ 0.
        cached_step = result2.step_results[0]
        assert cached_step.succeeded
        assert cached_step.duration_seconds == 0.0

    def test_cache_hit_returns_same_artifact(self, tmp_path: Path) -> None:
        adapter = _CountingOCRAdapter(tmp_path)
        store = InMemoryArtifactStore()
        executor = PipelineExecutor(
            adapter_resolver=lambda n: adapter,
            artifact_store=store,
        )

        result1 = executor.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=_make_initial_inputs(),
            context=_make_context(),
        )
        result2 = executor.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=_make_initial_inputs(),
            context=_make_context(),
        )
        # Même artefact retourné (mêmes id, même content_hash).
        a1 = [a for a in result1.artifacts if a.type == ArtifactType.RAW_TEXT][0]
        a2 = [a for a in result2.artifacts if a.type == ArtifactType.RAW_TEXT][0]
        assert a1.id == a2.id
        assert a1.content_hash == a2.content_hash
        assert a1.uri == a2.uri


# ──────────────────────────────────────────────────────────────────────
# Cache miss — invariants de la clé
# ──────────────────────────────────────────────────────────────────────


class TestCacheMissOnKeyChange:
    def test_miss_when_code_version_differs(self, tmp_path: Path) -> None:
        adapter = _CountingOCRAdapter(tmp_path)
        store = InMemoryArtifactStore()
        executor = PipelineExecutor(
            adapter_resolver=lambda n: adapter,
            artifact_store=store,
        )

        executor.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=_make_initial_inputs(),
            context=_make_context(code_version="1.0.0"),
        )
        executor.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=_make_initial_inputs(),
            context=_make_context(code_version="2.0.0"),  # change !
        )
        # Le code_version fait partie de la clé → 2 exécutions distinctes.
        assert adapter.call_count == 2

    def test_miss_when_step_params_differ(self, tmp_path: Path) -> None:
        adapter = _CountingOCRAdapter(tmp_path)
        store = InMemoryArtifactStore()
        executor = PipelineExecutor(
            adapter_resolver=lambda n: adapter,
            artifact_store=store,
        )

        spec_a = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr",
                    kind="ocr",
                    adapter_name="counting_ocr",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                    params={"lang": "fra"},
                ),
            ),
        )
        spec_b = PipelineSpec(
            name="x",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(
                PipelineStep(
                    id="ocr",
                    kind="ocr",
                    adapter_name="counting_ocr",
                    input_types=(ArtifactType.IMAGE,),
                    output_types=(ArtifactType.RAW_TEXT,),
                    params={"lang": "eng"},  # change !
                ),
            ),
        )

        executor.run(
            spec=spec_a,
            document=DocumentRef(id="d1"),
            initial_inputs=_make_initial_inputs(),
            context=_make_context(),
        )
        executor.run(
            spec=spec_b,
            document=DocumentRef(id="d1"),
            initial_inputs=_make_initial_inputs(),
            context=_make_context(),
        )
        assert adapter.call_count == 2

    def test_miss_when_input_content_hash_differs(self, tmp_path: Path) -> None:
        adapter = _CountingOCRAdapter(tmp_path)
        store = InMemoryArtifactStore()
        executor = PipelineExecutor(
            adapter_resolver=lambda n: adapter,
            artifact_store=store,
        )

        inputs_a = {
            ArtifactType.IMAGE: Artifact(
                id="d1:image", document_id="d1", type=ArtifactType.IMAGE,
                content_hash="a" * 64, uri="/tmp/img.png",
            ),
        }
        inputs_b = {
            ArtifactType.IMAGE: Artifact(
                id="d1:image", document_id="d1", type=ArtifactType.IMAGE,
                content_hash="c" * 64,  # change !
                uri="/tmp/img.png",
            ),
        }

        executor.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=inputs_a,
            context=_make_context(),
        )
        executor.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=inputs_b,
            context=_make_context(),
        )
        assert adapter.call_count == 2


# ──────────────────────────────────────────────────────────────────────
# Cache miss — invariants de validité
# ──────────────────────────────────────────────────────────────────────


class TestCacheMissOnInvalidState:
    def test_miss_when_input_has_no_content_hash(self, tmp_path: Path) -> None:
        """Si un input n'a pas de content_hash, la clé n'est pas
        calculable → bypass complet du cache (pas de hit, pas de
        persistence)."""
        adapter = _CountingOCRAdapter(tmp_path)
        store = InMemoryArtifactStore()
        executor = PipelineExecutor(
            adapter_resolver=lambda n: adapter,
            artifact_store=store,
        )

        inputs_no_hash = {
            ArtifactType.IMAGE: Artifact(
                id="d1:image", document_id="d1", type=ArtifactType.IMAGE,
                content_hash=None,  # pas de hash !
                uri="/tmp/img.png",
            ),
        }

        executor.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=inputs_no_hash,
            context=_make_context(),
        )
        executor.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=inputs_no_hash,
            context=_make_context(),
        )
        # Sans hash, on n'a ni hit ni miss déterministe — on
        # exécute systématiquement.
        assert adapter.call_count == 2
        # Le store reste vide (rien n'a été persisté).
        assert len(store) == 0

    def test_miss_when_cached_uri_disappeared(self, tmp_path: Path) -> None:
        """Si le fichier pointé par l'URI cachée a été supprimé entre
        les deux runs (workspace nettoyé), on doit re-exécuter."""
        adapter = _CountingOCRAdapter(tmp_path)
        store = InMemoryArtifactStore()
        executor = PipelineExecutor(
            adapter_resolver=lambda n: adapter,
            artifact_store=store,
        )

        executor.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=_make_initial_inputs(),
            context=_make_context(),
        )
        assert adapter.call_count == 1

        # Simule un nettoyage du workspace.
        for f in tmp_path.iterdir():
            if f.is_file():
                f.unlink()

        executor.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=_make_initial_inputs(),
            context=_make_context(),
        )
        # URI cachée pointe vers fichier disparu → cache miss → ré-exec.
        assert adapter.call_count == 2


# ──────────────────────────────────────────────────────────────────────
# Persistance filesystem — survie inter-process
# ──────────────────────────────────────────────────────────────────────


class TestFilesystemStorePersistence:
    def test_cache_survives_executor_recreation(self, tmp_path: Path) -> None:
        """Avec un FilesystemArtifactStore partagé, deux instances
        d'executor distinctes (simule un redémarrage) hit le cache
        de la première."""
        store_root = tmp_path / "store"
        adapter = _CountingOCRAdapter(tmp_path / "outputs")
        (tmp_path / "outputs").mkdir()

        # Premier executor.
        store1 = FilesystemArtifactStore(store_root)
        exe1 = PipelineExecutor(
            adapter_resolver=lambda n: adapter,
            artifact_store=store1,
        )
        exe1.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=_make_initial_inputs(),
            context=_make_context(),
        )
        assert adapter.call_count == 1

        # Second executor avec un NOUVEAU store pointant vers le même
        # filesystem root (simule un redémarrage du process).
        store2 = FilesystemArtifactStore(store_root)
        exe2 = PipelineExecutor(
            adapter_resolver=lambda n: adapter,
            artifact_store=store2,
        )
        exe2.run(
            spec=_make_spec(),
            document=DocumentRef(id="d1"),
            initial_inputs=_make_initial_inputs(),
            context=_make_context(),
        )
        # Le cache filesystem a survécu → hit.
        assert adapter.call_count == 1, (
            "Le cache filesystem n'a pas survécu au re-démarrage."
        )
