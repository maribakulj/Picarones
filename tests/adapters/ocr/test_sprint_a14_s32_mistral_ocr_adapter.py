"""Sprint A14-S32 — ``MistralOCRAdapter`` natif au contrat S26."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from picarones.adapters.ocr import (
    BaseOCRAdapter,
    MistralOCRAdapter,
    OCRAdapterError,
)
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.pipeline.types import RunContext


def _make_image_artifact(uri: str) -> Artifact:
    return Artifact(
        id="d1:initial:image",
        document_id="d1",
        type=ArtifactType.IMAGE,
        uri=uri,
    )


def _make_context() -> RunContext:
    return RunContext(
        document_id="d1",
        code_version="1.0.0",
        pipeline_name="test",
    )


def _make_dummy_image(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nfakeimagebytes")
    return path


# ──────────────────────────────────────────────────────────────────────
# Constructeur
# ──────────────────────────────────────────────────────────────────────


class TestMistralOCRAdapterConstructor:
    def test_defaults(self) -> None:
        adapter = MistralOCRAdapter()
        assert adapter.name == "mistral_ocr"
        assert adapter.model == "mistral-ocr-latest"

    def test_custom_name(self) -> None:
        adapter = MistralOCRAdapter(name="my_mistral")
        assert adapter.name == "my_mistral"

    def test_custom_model(self) -> None:
        adapter = MistralOCRAdapter(model="pixtral-12b-2409")
        assert adapter.model == "pixtral-12b-2409"

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(OCRAdapterError, match="vide"):
            MistralOCRAdapter(name="")

    def test_rejects_invalid_chars_in_name(self) -> None:
        with pytest.raises(OCRAdapterError, match="invalide"):
            MistralOCRAdapter(name="bad name")

    def test_rejects_non_positive_max_tokens(self) -> None:
        with pytest.raises(OCRAdapterError, match="max_tokens"):
            MistralOCRAdapter(max_tokens=0)
        with pytest.raises(OCRAdapterError, match="max_tokens"):
            MistralOCRAdapter(max_tokens=-1)

    def test_rejects_non_positive_timeout(self) -> None:
        with pytest.raises(OCRAdapterError, match="timeout_seconds"):
            MistralOCRAdapter(timeout_seconds=0)
        with pytest.raises(OCRAdapterError, match="timeout_seconds"):
            MistralOCRAdapter(timeout_seconds=-1.0)


# ──────────────────────────────────────────────────────────────────────
# Contrat BaseOCRAdapter
# ──────────────────────────────────────────────────────────────────────


class TestMistralOCRAdapterContract:
    def test_inherits_base_adapter(self) -> None:
        adapter = MistralOCRAdapter()
        assert isinstance(adapter, BaseOCRAdapter)

    def test_input_types(self) -> None:
        assert MistralOCRAdapter.input_types == frozenset({ArtifactType.IMAGE})

    def test_output_types(self) -> None:
        assert MistralOCRAdapter.output_types == frozenset({ArtifactType.RAW_TEXT})

    def test_execution_mode_is_io(self) -> None:
        """Mistral OCR fait des appels HTTP — IO-bound, ThreadPool."""
        assert MistralOCRAdapter.execution_mode == "io"


# ──────────────────────────────────────────────────────────────────────
# API key resolution
# ──────────────────────────────────────────────────────────────────────


class TestMistralOCRApiKey:
    def test_explicit_key_takes_priority(self) -> None:
        adapter = MistralOCRAdapter(api_key="explicit_key")
        # Mock l'env pour s'assurer qu'on n'utilise pas la valeur env.
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "env_key"}):
            assert adapter._resolve_api_key() == "explicit_key"

    def test_env_key_used_when_no_explicit(self) -> None:
        adapter = MistralOCRAdapter()
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "env_key"}):
            assert adapter._resolve_api_key() == "env_key"

    def test_no_key_raises(self) -> None:
        adapter = MistralOCRAdapter()
        # Vide l'env de MISTRAL_API_KEY.
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(OCRAdapterError, match="MISTRAL_API_KEY"):
                adapter._resolve_api_key()


# ──────────────────────────────────────────────────────────────────────
# Encoding
# ──────────────────────────────────────────────────────────────────────


class TestMistralOCREncoding:
    def test_png_extension_yields_png_mime(self, tmp_path: Path) -> None:
        adapter = MistralOCRAdapter()
        image_path = _make_dummy_image(tmp_path)
        encoded = adapter._encode_image(image_path)
        assert encoded.startswith("data:image/png;base64,")

    def test_jpg_extension_yields_jpeg_mime(self, tmp_path: Path) -> None:
        adapter = MistralOCRAdapter()
        path = tmp_path / "img.jpg"
        path.write_bytes(b"jpegbytes")
        encoded = adapter._encode_image(path)
        assert encoded.startswith("data:image/jpeg;base64,")

    def test_unknown_extension_defaults_to_jpeg(self, tmp_path: Path) -> None:
        adapter = MistralOCRAdapter()
        path = tmp_path / "img.xyz"
        path.write_bytes(b"random")
        encoded = adapter._encode_image(path)
        assert encoded.startswith("data:image/jpeg;base64,")


# ──────────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────────


class TestMistralOCRInputValidation:
    def test_missing_image_input_raises(self) -> None:
        adapter = MistralOCRAdapter(api_key="x")
        with pytest.raises(OCRAdapterError, match="IMAGE manquant"):
            adapter.execute(inputs={}, params={}, context=_make_context())

    def test_image_artifact_without_uri_raises(self) -> None:
        adapter = MistralOCRAdapter(api_key="x")
        artifact = Artifact(
            id="d1:img",
            document_id="d1",
            type=ArtifactType.IMAGE,
            uri=None,
        )
        with pytest.raises(OCRAdapterError, match="sans URI"):
            adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )

    def test_image_path_does_not_exist_raises(self) -> None:
        adapter = MistralOCRAdapter(api_key="x")
        artifact = _make_image_artifact("/nonexistent/img.png")
        with pytest.raises(OCRAdapterError, match="introuvable"):
            adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )

    def test_no_api_key_raises(self, tmp_path: Path) -> None:
        adapter = MistralOCRAdapter()  # pas d'api_key explicite
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(OCRAdapterError, match="MISTRAL_API_KEY"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )


# ──────────────────────────────────────────────────────────────────────
# /v1/ocr API (mistral-ocr-* models)
# ──────────────────────────────────────────────────────────────────────


class TestMistralOCRNativeAPI:
    def _mock_urlopen_ok(self, response_json: dict):
        """Helper : retourne un context manager qui mock urlopen."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = repr(response_json).encode()
        # On ne peut pas json.dumps un dict avec json.dumps directement
        # à cause du repr ; on encode proprement.
        import json as _json
        mock_resp.read.return_value = _json.dumps(response_json).encode()
        mock_resp.__enter__.return_value = mock_resp
        return patch("urllib.request.urlopen", return_value=mock_resp)

    def test_native_api_concatenates_pages(self, tmp_path: Path) -> None:
        adapter = MistralOCRAdapter(api_key="x")
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        response_json = {
            "pages": [
                {"markdown": "Page 1 contenu"},
                {"markdown": "Page 2 contenu"},
            ],
        }

        with self._mock_urlopen_ok(response_json):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        out_text = Path(result[ArtifactType.RAW_TEXT].uri).read_text(
            encoding="utf-8",
        )
        assert out_text == "Page 1 contenu\n\nPage 2 contenu"

    def test_native_api_writes_to_stem_name_pattern(self, tmp_path: Path) -> None:
        adapter = MistralOCRAdapter(api_key="x", name="my_mistral")
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with self._mock_urlopen_ok({"pages": [{"markdown": "x"}]}):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        out_path = Path(result[ArtifactType.RAW_TEXT].uri)
        assert out_path.name == "page.my_mistral.txt"

    def test_native_api_raises_on_http_error(self, tmp_path: Path) -> None:
        adapter = MistralOCRAdapter(api_key="x")
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with patch(
            "urllib.request.urlopen",
            side_effect=ConnectionError("API down"),
        ):
            with pytest.raises(OCRAdapterError, match="ConnectionError"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )


# ──────────────────────────────────────────────────────────────────────
# Vision/chat API (pixtral-* models)
# ──────────────────────────────────────────────────────────────────────


class TestMistralOCRVisionAPI:
    def test_pixtral_routes_to_vision_api(self, tmp_path: Path) -> None:
        adapter = MistralOCRAdapter(
            api_key="x",
            model="pixtral-12b-2409",
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        # Mock le SDK mistralai.
        mock_message = MagicMock()
        mock_message.content = "Texte transcrit par pixtral."
        mock_choice = MagicMock(message=mock_message)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = mock_response

        fake_module = MagicMock()
        fake_module.Mistral = MagicMock(return_value=mock_client)
        fake_client_module = MagicMock()
        fake_client_module.Mistral = fake_module.Mistral

        with patch.dict(sys.modules, {
            "mistralai": fake_module,
            "mistralai.client": fake_client_module,
        }):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )

        out_text = Path(result[ArtifactType.RAW_TEXT].uri).read_text(
            encoding="utf-8",
        )
        assert out_text == "Texte transcrit par pixtral."

    def test_pixtral_sdk_missing_raises_clean_error(
        self, tmp_path: Path,
    ) -> None:
        adapter = MistralOCRAdapter(api_key="x", model="pixtral-12b")
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with patch.dict(sys.modules, {
            "mistralai": None,
            "mistralai.client": None,
        }):
            with pytest.raises(OCRAdapterError, match="mistralai"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )

    def test_pixtral_api_error_wrapped(self, tmp_path: Path) -> None:
        adapter = MistralOCRAdapter(api_key="x", model="pixtral-12b")
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        mock_client = MagicMock()
        mock_client.chat.complete.side_effect = RuntimeError("API error")

        fake_module = MagicMock()
        fake_module.Mistral = MagicMock(return_value=mock_client)
        fake_client_module = MagicMock()
        fake_client_module.Mistral = fake_module.Mistral

        with patch.dict(sys.modules, {
            "mistralai": fake_module,
            "mistralai.client": fake_client_module,
        }):
            with pytest.raises(OCRAdapterError, match="RuntimeError.*API error"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )


# ──────────────────────────────────────────────────────────────────────
# Artifact ID
# ──────────────────────────────────────────────────────────────────────


class TestMistralOCRArtifactID:
    def test_artifact_id_uses_adapter_name(self, tmp_path: Path) -> None:
        adapter = MistralOCRAdapter(api_key="x", name="custom")
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        mock_resp = MagicMock()
        import json as _json
        mock_resp.read.return_value = _json.dumps(
            {"pages": [{"markdown": "x"}]},
        ).encode()
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        produced = result[ArtifactType.RAW_TEXT]
        assert produced.id == "d1:custom:raw_text"
        assert produced.document_id == "d1"
        assert produced.produced_by_step == "ocr"
