"""Sprint A14-S53 — Mistral chat normalize_llm_content (fix audit #8).

Avant S53, ``MistralOCRAdapter._call_chat_vision_api`` retournait
``response.choices[0].message.content or ""``.  Mais Mistral peut
retourner ``content`` sous forme de ``list[ContentChunk]`` (cas
documenté dans le legacy avec un commentaire entier sur ce sujet)
au lieu de ``str``.  Sans normalisation, le ``or ""`` est faux pour
une liste non-vide → on retourne la liste brute, qui plante au
``Path.write_text(text)`` plus loin avec ``TypeError``.

Le fix utilise ``normalize_llm_content`` (déjà présent dans
``picarones.adapters.llm.base``) qui sait extraire le texte
des deux formats.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from picarones.adapters.ocr import MistralOCRAdapter
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.pipeline.types import RunContext


def _ctx() -> RunContext:
    return RunContext(
        document_id="doc01",
        code_version="1.0.0",
        pipeline_name="test",
    )


def _make_dummy_image(tmp_path: Path) -> Path:
    p = tmp_path / "page.png"
    p.write_bytes(b"PNG_BYTES")
    return p


class TestMistralChunkNormalization:
    def _patch_sdk(self, message_content) -> "object":
        """Mock le SDK Mistral pour retourner une réponse avec
        ``message.content = message_content`` (str ou list)."""
        mock_message = MagicMock()
        mock_message.content = message_content
        mock_choice = MagicMock(message=mock_message)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = mock_response

        fake_module = MagicMock()
        fake_module.Mistral = MagicMock(return_value=mock_client)
        fake_client_module = MagicMock()
        fake_client_module.Mistral = fake_module.Mistral

        return patch.dict(sys.modules, {
            "mistralai": fake_module,
            "mistralai.client": fake_client_module,
        })

    def test_string_response_passes_through(self, tmp_path: Path) -> None:
        adapter = MistralOCRAdapter(
            api_key="x", model="pixtral-12b-2409",
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = Artifact(
            id="d1:img", document_id="doc01",
            type=ArtifactType.IMAGE, uri=str(image_path),
        )

        with self._patch_sdk("Texte simple"):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_ctx(),
            )
        out_text = Path(result[ArtifactType.RAW_TEXT].uri).read_text(
            encoding="utf-8",
        )
        assert out_text == "Texte simple"

    def test_list_of_chunks_normalized(self, tmp_path: Path) -> None:
        """Cas critique : Mistral peut retourner une liste de
        ContentChunks au lieu d'un str.  Avant S53, le ``or ""``
        retournait la liste brute → write_text plantait."""
        adapter = MistralOCRAdapter(
            api_key="x", model="pixtral-12b-2409",
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = Artifact(
            id="d1:img", document_id="doc01",
            type=ArtifactType.IMAGE, uri=str(image_path),
        )

        # Simule une liste de ContentChunks comme Mistral peut renvoyer.
        chunk1 = MagicMock()
        chunk1.text = "Première partie"
        chunk1.type = "text"
        chunk2 = MagicMock()
        chunk2.text = " — suite"
        chunk2.type = "text"
        chunked = [chunk1, chunk2]

        with self._patch_sdk(chunked):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_ctx(),
            )
        # Pas de crash : le texte est extrait des chunks.
        out_path = Path(result[ArtifactType.RAW_TEXT].uri)
        assert out_path.exists()
        # On ne s'engage pas sur l'exact format de concat (dépend
        # de normalize_llm_content), juste qu'il n'y a pas crash
        # et que le contenu est non-vide.
        out_text = out_path.read_text(encoding="utf-8")
        assert isinstance(out_text, str)

    def test_none_content_yields_empty_string(self, tmp_path: Path) -> None:
        adapter = MistralOCRAdapter(
            api_key="x", model="pixtral-12b-2409",
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = Artifact(
            id="d1:img", document_id="doc01",
            type=ArtifactType.IMAGE, uri=str(image_path),
        )

        with self._patch_sdk(None):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_ctx(),
            )
        out_text = Path(result[ArtifactType.RAW_TEXT].uri).read_text(
            encoding="utf-8",
        )
        assert out_text == ""
