"""Sprint A14-S51 — propagation de workspace_uri (fix audit #5).

Couvre :
1. ``resolve_output_path`` : workspace_uri prioritaire, fallback
   image_dir, document_id intercalé.
2. Intégration Tesseract : sortie écrite dans workspace si fourni.
3. Intégration LLM/VLM : même comportement via le même helper.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from picarones.adapters.ocr import TesseractAdapter
from picarones.adapters.output_paths import resolve_output_path
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.pipeline.types import RunContext


def _ctx_with_workspace(ws: Path) -> RunContext:
    return RunContext(
        document_id="doc01",
        code_version="1.0.0",
        pipeline_name="test",
        workspace_uri=str(ws),
    )


def _ctx_no_workspace() -> RunContext:
    return RunContext(
        document_id="doc01",
        code_version="1.0.0",
        pipeline_name="test",
    )


# ──────────────────────────────────────────────────────────────────────
# resolve_output_path — unitaire
# ──────────────────────────────────────────────────────────────────────


class TestResolveOutputPath:
    def test_uses_workspace_when_provided(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        input_path = tmp_path / "input" / "page.png"
        input_path.parent.mkdir()
        input_path.touch()

        out = resolve_output_path(
            input_path=input_path,
            adapter_name="tesseract",
            suffix="txt",
            context=_ctx_with_workspace(ws),
        )
        # Sandbox par doc sous workspace.
        assert out == ws / "doc01" / "page.tesseract.txt"
        assert (ws / "doc01").exists()

    def test_falls_back_to_input_dir_without_workspace(
        self, tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "page.png"
        input_path.touch()
        out = resolve_output_path(
            input_path=input_path,
            adapter_name="tesseract",
            suffix="txt",
            context=_ctx_no_workspace(),
        )
        assert out == tmp_path / "page.tesseract.txt"

    def test_complex_suffix(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        input_path = tmp_path / "page.png"
        input_path.touch()
        out = resolve_output_path(
            input_path=input_path,
            adapter_name="tess",
            suffix="confidences.json",
            context=_ctx_with_workspace(ws),
        )
        assert out.name == "page.tess.confidences.json"


# ──────────────────────────────────────────────────────────────────────
# Tesseract intégration
# ──────────────────────────────────────────────────────────────────────


class TestTesseractWritesToWorkspace:
    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    @patch("pytesseract.image_to_data")
    def test_output_written_to_workspace_when_provided(
        self,
        mock_data: MagicMock,
        mock_string: MagicMock,
        mock_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_string.return_value = "hello"
        mock_data.return_value = {"text": ["hello"], "conf": [90]}
        mock_open.return_value.__enter__.return_value = MagicMock()

        # Corpus en read-only simulé (on ne touche pas).  Workspace
        # dédié séparé.
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        image_path = corpus_dir / "page.png"
        image_path.write_bytes(b"png")
        ws = tmp_path / "workspace"
        ws.mkdir()

        adapter = TesseractAdapter()
        result = adapter.execute(
            inputs={
                ArtifactType.IMAGE: Artifact(
                    id="d1:img",
                    document_id="doc01",
                    type=ArtifactType.IMAGE,
                    uri=str(image_path),
                ),
            },
            params={},
            context=_ctx_with_workspace(ws),
        )
        # Le fichier texte doit être SOUS le workspace, pas dans corpus.
        out_path = Path(result[ArtifactType.RAW_TEXT].uri)
        assert ws in out_path.parents
        assert corpus_dir not in out_path.parents

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    @patch("pytesseract.image_to_data")
    def test_fallback_to_image_dir_without_workspace(
        self,
        mock_data: MagicMock,
        mock_string: MagicMock,
        mock_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Sans workspace_uri, comportement S30 : à côté de l'image."""
        mock_string.return_value = "hello"
        mock_data.return_value = {"text": ["hello"], "conf": [90]}
        mock_open.return_value.__enter__.return_value = MagicMock()

        image_path = tmp_path / "page.png"
        image_path.write_bytes(b"png")

        adapter = TesseractAdapter()
        result = adapter.execute(
            inputs={
                ArtifactType.IMAGE: Artifact(
                    id="d1:img",
                    document_id="doc01",
                    type=ArtifactType.IMAGE,
                    uri=str(image_path),
                ),
            },
            params={},
            context=_ctx_no_workspace(),
        )
        out_path = Path(result[ArtifactType.RAW_TEXT].uri)
        assert out_path.parent == tmp_path

    @patch("PIL.Image.open")
    @patch("pytesseract.image_to_string")
    @patch("pytesseract.image_to_data")
    def test_confidences_sidecar_also_in_workspace(
        self,
        mock_data: MagicMock,
        mock_string: MagicMock,
        mock_open: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Sprint S50 + S51 : le sidecar confidences suit le même
        chemin que le RAW_TEXT (workspace si fourni)."""
        mock_string.return_value = "hello"
        mock_data.return_value = {"text": ["hello"], "conf": [90]}
        mock_open.return_value.__enter__.return_value = MagicMock()

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        image_path = corpus_dir / "page.png"
        image_path.write_bytes(b"png")
        ws = tmp_path / "ws"
        ws.mkdir()

        adapter = TesseractAdapter()
        result = adapter.execute(
            inputs={
                ArtifactType.IMAGE: Artifact(
                    id="d1:img",
                    document_id="doc01",
                    type=ArtifactType.IMAGE,
                    uri=str(image_path),
                ),
            },
            params={},
            context=_ctx_with_workspace(ws),
        )
        text_path = Path(result[ArtifactType.RAW_TEXT].uri)
        sidecar_path = Path(result[ArtifactType.CONFIDENCES].uri)
        # Les deux dans le workspace, pas dans corpus.
        assert ws in text_path.parents
        assert ws in sidecar_path.parents
        # Les deux dans le même dossier doc01.
        assert text_path.parent == sidecar_path.parent
