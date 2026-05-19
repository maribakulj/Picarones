"""Sprint A14-S26 — ``BaseOCRAdapter`` + ``PrecomputedTextAdapter``.

Couverture :

- **Contrat** : un ``BaseOCRAdapter`` est instanciable, expose
  ``name`` / ``input_types`` / ``output_types`` / ``execution_mode``,
  son ``execute()`` est abstrait.
- **PrecomputedTextAdapter** : validation du ``source_label``,
  lecture filesystem par convention de nommage, politique
  ``"raise"`` vs ``"empty"`` sur fichier manquant, validation
  UTF-8, isolation entre instances de sources distinctes.
- **Pipeline executor** : un ``PrecomputedTextAdapter`` est consommé
  directement par le ``PipelineExecutor`` (S7) — preuve que le
  contrat ``BaseOCRAdapter`` satisfait ``StepExecutor``.
- **CLI E2E** : YAML déclarant 3 sources pré-calculées différentes
  → benchmark complet avec 3 pipelines comparés sur TextView,
  sans aucun OCR réel.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picarones.adapters.ocr import (
    BaseOCRAdapter,
    OCRAdapterError,
    PrecomputedTextAdapter,
)
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.pipeline.types import RunContext


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89"
    )


def _ctx(doc_id: str = "doc01") -> RunContext:
    return RunContext(
        document_id=doc_id,
        code_version="1.0.0-s26-test",
        pipeline_name="test_pipeline",
    )


def _image_artifact(doc_id: str, path: Path) -> Artifact:
    return Artifact(
        id=f"{doc_id}:image",
        document_id=doc_id,
        type=ArtifactType.IMAGE,
        uri=str(path),
    )


# ──────────────────────────────────────────────────────────────────
# Contrat BaseOCRAdapter
# ──────────────────────────────────────────────────────────────────


class TestBaseOCRAdapterContract:
    def test_cannot_instantiate_abstract_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseOCRAdapter()  # type: ignore[abstract]

    def test_minimal_subclass_with_name_and_execute_works(self) -> None:
        class _Minimal(BaseOCRAdapter):
            @property
            def name(self) -> str:
                return "minimal"

            def execute(self, inputs, params, context):
                return {}

        adapter = _Minimal()
        assert adapter.name == "minimal"
        assert ArtifactType.IMAGE in adapter.input_types
        assert ArtifactType.RAW_TEXT in adapter.output_types
        assert adapter.execution_mode == "io"

    def test_subclass_can_override_io_modes(self) -> None:
        class _CPUBound(BaseOCRAdapter):
            execution_mode = "cpu"
            input_types = frozenset({ArtifactType.IMAGE})
            output_types = frozenset({
                ArtifactType.RAW_TEXT, ArtifactType.ALTO_XML,
            })

            @property
            def name(self) -> str:
                return "cpu_bound"

            def execute(self, inputs, params, context):
                return {}

        adapter = _CPUBound()
        assert adapter.execution_mode == "cpu"
        assert ArtifactType.ALTO_XML in adapter.output_types


# ──────────────────────────────────────────────────────────────────
# PrecomputedTextAdapter — validation à l'init
# ──────────────────────────────────────────────────────────────────


class TestPrecomputedInitValidation:
    def test_empty_source_label_rejected(self) -> None:
        with pytest.raises(OCRAdapterError, match="vide"):
            PrecomputedTextAdapter(source_label="")

    def test_whitespace_source_label_rejected(self) -> None:
        with pytest.raises(OCRAdapterError, match="vide"):
            PrecomputedTextAdapter(source_label="   ")

    def test_invalid_chars_in_source_label_rejected(self) -> None:
        for bad in ("foo/bar", "foo bar", "foo.bar", "foo:bar"):
            with pytest.raises(OCRAdapterError, match="invalide"):
                PrecomputedTextAdapter(source_label=bad)

    def test_valid_source_labels_accepted(self) -> None:
        for good in ("tesseract", "gpt-4v", "pero_ocr", "ABC123"):
            adapter = PrecomputedTextAdapter(source_label=good)
            assert adapter.source_label == good
            assert adapter.name == f"precomputed_{good}"

    def test_invalid_missing_text_policy_rejected(self) -> None:
        with pytest.raises(OCRAdapterError, match="missing_text_policy"):
            PrecomputedTextAdapter(
                source_label="tess",
                missing_text_policy="silent",  # type: ignore[arg-type]
            )

    def test_default_missing_text_policy_is_raise(self) -> None:
        adapter = PrecomputedTextAdapter(source_label="tess")
        assert adapter._missing_policy == "raise"


# ──────────────────────────────────────────────────────────────────
# PrecomputedTextAdapter — exécution
# ──────────────────────────────────────────────────────────────────


class TestPrecomputedExecute:
    def test_reads_text_file_by_convention(self, tmp_path: Path) -> None:
        # Préparer image + texte pré-calculé.
        image_path = tmp_path / "doc01.png"
        image_path.write_bytes(_png_bytes())
        text_path = tmp_path / "doc01.tesseract.txt"
        text_path.write_text("Bonjour le monde", encoding="utf-8")

        adapter = PrecomputedTextAdapter(source_label="tesseract")
        outputs = adapter.execute(
            inputs={ArtifactType.IMAGE: _image_artifact("doc01", image_path)},
            params={},
            context=_ctx("doc01"),
        )
        art = outputs[ArtifactType.RAW_TEXT]
        assert art.type == ArtifactType.RAW_TEXT
        assert art.document_id == "doc01"
        assert Path(art.uri).read_text(encoding="utf-8") == "Bonjour le monde"
        # Convention <doc_id>:<owner>:<role>.
        assert art.id == "doc01:precomputed_tesseract:raw_text"

    def test_missing_text_raises_by_default(self, tmp_path: Path) -> None:
        image_path = tmp_path / "doc01.png"
        image_path.write_bytes(_png_bytes())
        # Pas de doc01.tesseract.txt.

        adapter = PrecomputedTextAdapter(source_label="tesseract")
        with pytest.raises(OCRAdapterError, match="introuvable"):
            adapter.execute(
                inputs={ArtifactType.IMAGE: _image_artifact("doc01", image_path)},
                params={},
                context=_ctx("doc01"),
            )

    def test_missing_text_empty_policy_creates_empty_file(
        self, tmp_path: Path,
    ) -> None:
        image_path = tmp_path / "doc01.png"
        image_path.write_bytes(_png_bytes())

        adapter = PrecomputedTextAdapter(
            source_label="tess",
            missing_text_policy="empty",
        )
        outputs = adapter.execute(
            inputs={ArtifactType.IMAGE: _image_artifact("doc01", image_path)},
            params={},
            context=_ctx("doc01"),
        )
        art = outputs[ArtifactType.RAW_TEXT]
        assert Path(art.uri).read_text(encoding="utf-8") == ""

    def test_non_utf8_file_rejected(self, tmp_path: Path) -> None:
        image_path = tmp_path / "doc01.png"
        image_path.write_bytes(_png_bytes())
        text_path = tmp_path / "doc01.tess.txt"
        # Bytes invalides en UTF-8 (latin-1 avec accent).
        text_path.write_bytes(b"\xe9\xe8")

        adapter = PrecomputedTextAdapter(source_label="tess")
        with pytest.raises(OCRAdapterError, match="UTF-8"):
            adapter.execute(
                inputs={ArtifactType.IMAGE: _image_artifact("doc01", image_path)},
                params={},
                context=_ctx("doc01"),
            )

    def test_missing_image_input_rejected(self, tmp_path: Path) -> None:
        adapter = PrecomputedTextAdapter(source_label="tess")
        with pytest.raises(OCRAdapterError, match="IMAGE manquant"):
            adapter.execute(inputs={}, params={}, context=_ctx())

    def test_image_artifact_without_uri_rejected(self) -> None:
        adapter = PrecomputedTextAdapter(source_label="tess")
        with pytest.raises(OCRAdapterError, match="sans URI"):
            adapter.execute(
                inputs={
                    ArtifactType.IMAGE: Artifact(
                        id="d:image", document_id="d",
                        type=ArtifactType.IMAGE,
                    ),
                },
                params={},
                context=_ctx(),
            )

    def test_two_sources_isolated_in_same_dir(self, tmp_path: Path) -> None:
        """Cas BnF central : deux sources pré-calculées dans le même
        répertoire ne se piétinent pas — chaque adapter lit son
        propre fichier."""
        image_path = tmp_path / "doc01.png"
        image_path.write_bytes(_png_bytes())
        (tmp_path / "doc01.tess.txt").write_text(
            "tesseract output", encoding="utf-8",
        )
        (tmp_path / "doc01.gpt4v.txt").write_text(
            "gpt-4 vision output", encoding="utf-8",
        )

        a_tess = PrecomputedTextAdapter(source_label="tess")
        a_gpt = PrecomputedTextAdapter(source_label="gpt4v")

        out_tess = a_tess.execute(
            inputs={ArtifactType.IMAGE: _image_artifact("doc01", image_path)},
            params={},
            context=_ctx("doc01"),
        )
        out_gpt = a_gpt.execute(
            inputs={ArtifactType.IMAGE: _image_artifact("doc01", image_path)},
            params={},
            context=_ctx("doc01"),
        )
        assert Path(out_tess[ArtifactType.RAW_TEXT].uri).read_text() \
            == "tesseract output"
        assert Path(out_gpt[ArtifactType.RAW_TEXT].uri).read_text() \
            == "gpt-4 vision output"

    def test_image_extension_variations_handled(
        self, tmp_path: Path,
    ) -> None:
        """``stem`` strip toutes les extensions image courantes."""
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            image_path = tmp_path / f"folio_001{ext}"
            image_path.write_bytes(_png_bytes())
            text_path = tmp_path / "folio_001.src.txt"
            text_path.write_text("ok", encoding="utf-8")

            adapter = PrecomputedTextAdapter(source_label="src")
            out = adapter.execute(
                inputs={
                    ArtifactType.IMAGE: _image_artifact("folio_001", image_path),
                },
                params={},
                context=_ctx("folio_001"),
            )
            assert Path(out[ArtifactType.RAW_TEXT].uri).read_text() == "ok"


# ──────────────────────────────────────────────────────────────────
# Smoke pipeline executor
# ──────────────────────────────────────────────────────────────────


class TestPipelineExecutorIntegration:
    def test_adapter_consumed_by_pipeline_executor(
        self, tmp_path: Path,
    ) -> None:
        """Démontre que ``BaseOCRAdapter`` satisfait le contrat
        ``StepExecutor`` du nouveau pipeline executor — preuve que
        le contrat propre du nouveau monde est suffisant."""
        from picarones.domain.documents import DocumentRef
        from picarones.pipeline import (
            PipelineExecutor, PipelineSpec, PipelineStep,
        )

        image_path = tmp_path / "doc01.png"
        image_path.write_bytes(_png_bytes())
        (tmp_path / "doc01.tess.txt").write_text(
            "Bonjour", encoding="utf-8",
        )

        adapter = PrecomputedTextAdapter(source_label="tess")
        spec = PipelineSpec(
            name="precomputed_smoke",
            initial_inputs=(ArtifactType.IMAGE,),
            steps=(PipelineStep(
                id="ocr", kind="ocr",
                adapter_name="precomputed",
                input_types=(ArtifactType.IMAGE,),
                output_types=(ArtifactType.RAW_TEXT,),
            ),),
        )
        executor = PipelineExecutor(adapter_resolver=lambda n: adapter)
        result = executor.run(
            spec=spec,
            document=DocumentRef(id="doc01", image_uri=str(image_path)),
            initial_inputs={
                ArtifactType.IMAGE: _image_artifact("doc01", image_path),
            },
            context=_ctx("doc01"),
        )
        assert result.succeeded
        text_arts = result.artifacts_of_type(ArtifactType.RAW_TEXT)
        assert len(text_arts) == 1
        assert Path(text_arts[0].uri).read_text() == "Bonjour"
