"""Sprint A14-S33 — ``GoogleVisionAdapter`` natif au contrat S26."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from picarones.adapters.ocr import (
    BaseOCRAdapter,
    GoogleVisionAdapter,
    OCRAdapterError,
)
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.pipeline.types import RunContext


def _make_image_artifact(uri: str) -> Artifact:
    return Artifact(
        id="d1:img",
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
    path.write_bytes(b"PNG_FAKE_BYTES")
    return path


# ──────────────────────────────────────────────────────────────────────
# Constructeur
# ──────────────────────────────────────────────────────────────────────


class TestGoogleVisionConstructor:
    def test_defaults(self) -> None:
        adapter = GoogleVisionAdapter()
        assert adapter.name == "google_vision"
        assert adapter.feature_type == "DOCUMENT_TEXT_DETECTION"

    def test_custom_name(self) -> None:
        adapter = GoogleVisionAdapter(name="my_gv")
        assert adapter.name == "my_gv"

    def test_text_detection_feature(self) -> None:
        adapter = GoogleVisionAdapter(feature_type="TEXT_DETECTION")
        assert adapter.feature_type == "TEXT_DETECTION"

    def test_rejects_invalid_feature_type(self) -> None:
        with pytest.raises(OCRAdapterError, match="feature_type"):
            GoogleVisionAdapter(feature_type="UNKNOWN_FEATURE")

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(OCRAdapterError, match="vide"):
            GoogleVisionAdapter(name="")

    def test_rejects_invalid_chars_in_name(self) -> None:
        with pytest.raises(OCRAdapterError, match="invalide"):
            GoogleVisionAdapter(name="bad name")

    def test_rejects_non_positive_timeout(self) -> None:
        with pytest.raises(OCRAdapterError, match="timeout"):
            GoogleVisionAdapter(timeout_seconds=0)

    def test_default_language_hints(self) -> None:
        adapter = GoogleVisionAdapter()
        # Vérifier que les hints sont stockés (privé mais accessible).
        assert adapter._language_hints == ["fr"]

    def test_custom_language_hints(self) -> None:
        adapter = GoogleVisionAdapter(language_hints=["en", "lat"])
        assert adapter._language_hints == ["en", "lat"]


# ──────────────────────────────────────────────────────────────────────
# Contrat BaseOCRAdapter
# ──────────────────────────────────────────────────────────────────────


class TestGoogleVisionContract:
    def test_inherits_base_adapter(self) -> None:
        adapter = GoogleVisionAdapter()
        assert isinstance(adapter, BaseOCRAdapter)

    def test_input_types(self) -> None:
        assert GoogleVisionAdapter.input_types == frozenset({ArtifactType.IMAGE})

    def test_output_types(self) -> None:
        assert GoogleVisionAdapter.output_types == frozenset({ArtifactType.RAW_TEXT})

    def test_execution_mode_is_io(self) -> None:
        assert GoogleVisionAdapter.execution_mode == "io"


# ──────────────────────────────────────────────────────────────────────
# Auth resolution
# ──────────────────────────────────────────────────────────────────────


class TestGoogleVisionAuth:
    def test_no_auth_raises(self, tmp_path: Path) -> None:
        adapter = GoogleVisionAdapter()
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(OCRAdapterError, match="authentification manquante"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )

    def test_explicit_credentials_path_takes_priority(self) -> None:
        adapter = GoogleVisionAdapter(credentials_path="/explicit/creds.json")
        with patch.dict(
            "os.environ",
            {"GOOGLE_APPLICATION_CREDENTIALS": "/env/creds.json"},
        ):
            assert adapter._resolve_credentials_path() == "/explicit/creds.json"

    def test_env_credentials_fallback(self) -> None:
        adapter = GoogleVisionAdapter()
        with patch.dict(
            "os.environ",
            {"GOOGLE_APPLICATION_CREDENTIALS": "/env/creds.json"},
        ):
            assert adapter._resolve_credentials_path() == "/env/creds.json"

    def test_explicit_api_key_takes_priority(self) -> None:
        adapter = GoogleVisionAdapter(api_key="explicit_key")
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env_key"}):
            assert adapter._resolve_api_key() == "explicit_key"


# ──────────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────────


class TestGoogleVisionInputValidation:
    def test_missing_image_input_raises(self) -> None:
        adapter = GoogleVisionAdapter(api_key="x")
        with pytest.raises(OCRAdapterError, match="IMAGE manquant"):
            adapter.execute(inputs={}, params={}, context=_make_context())

    def test_image_artifact_without_uri_raises(self) -> None:
        adapter = GoogleVisionAdapter(api_key="x")
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
        adapter = GoogleVisionAdapter(api_key="x")
        artifact = _make_image_artifact("/nonexistent/img.png")
        with pytest.raises(OCRAdapterError, match="introuvable"):
            adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )


# ──────────────────────────────────────────────────────────────────────
# REST API path (api_key)
# ──────────────────────────────────────────────────────────────────────


class TestGoogleVisionREST:
    def _mock_urlopen(self, response_dict: dict):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_dict).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        return patch("urllib.request.urlopen", return_value=mock_resp)

    def test_document_text_detection_returns_full_text(
        self, tmp_path: Path,
    ) -> None:
        adapter = GoogleVisionAdapter(api_key="x")
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        response = {
            "responses": [{
                "fullTextAnnotation": {"text": "Bonjour\nle monde"},
            }],
        }

        with self._mock_urlopen(response):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        out_text = Path(result[ArtifactType.RAW_TEXT].uri).read_text(
            encoding="utf-8",
        )
        assert out_text == "Bonjour\nle monde"

    def test_text_detection_returns_first_annotation(
        self, tmp_path: Path,
    ) -> None:
        adapter = GoogleVisionAdapter(
            api_key="x", feature_type="TEXT_DETECTION",
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        response = {
            "responses": [{
                "textAnnotations": [
                    {"description": "Texte court"},
                ],
            }],
        }

        with self._mock_urlopen(response):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        out_text = Path(result[ArtifactType.RAW_TEXT].uri).read_text(
            encoding="utf-8",
        )
        assert out_text == "Texte court"

    def test_empty_responses_returns_empty_text(self, tmp_path: Path) -> None:
        adapter = GoogleVisionAdapter(api_key="x")
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with self._mock_urlopen({"responses": [{}]}):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        out_text = Path(result[ArtifactType.RAW_TEXT].uri).read_text(
            encoding="utf-8",
        )
        assert out_text == ""

    def test_api_error_in_response_raises(self, tmp_path: Path) -> None:
        adapter = GoogleVisionAdapter(api_key="x")
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        response = {
            "responses": [{
                "error": {"code": 7, "message": "Permission denied"},
            }],
        }

        with self._mock_urlopen(response):
            with pytest.raises(OCRAdapterError, match="Permission denied"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )

    def test_writes_to_stem_name_pattern(self, tmp_path: Path) -> None:
        adapter = GoogleVisionAdapter(api_key="x", name="my_gv")
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        response = {"responses": [{"fullTextAnnotation": {"text": "x"}}]}

        with self._mock_urlopen(response):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        out_path = Path(result[ArtifactType.RAW_TEXT].uri)
        assert out_path.name == "page.my_gv.txt"


# ──────────────────────────────────────────────────────────────────────
# SDK path (credentials_path)
# ──────────────────────────────────────────────────────────────────────


class TestGoogleVisionSDK:
    def test_credentials_path_routes_to_sdk(self, tmp_path: Path) -> None:
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        adapter = GoogleVisionAdapter(credentials_path=str(creds_path))
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        # Mock du SDK google.cloud.vision
        mock_response = MagicMock()
        mock_response.full_text_annotation.text = "SDK output text"
        mock_client = MagicMock()
        mock_client.document_text_detection.return_value = mock_response

        fake_vision = MagicMock()
        fake_vision.ImageAnnotatorClient = MagicMock(return_value=mock_client)
        fake_vision.Image = MagicMock(return_value="image_obj")
        fake_vision.ImageContext = MagicMock(return_value="ctx_obj")
        fake_module = MagicMock()
        fake_module.vision = fake_vision

        with patch.dict(sys.modules, {
            "google": fake_module,
            "google.cloud": fake_module,
            "google.cloud.vision": fake_vision,
        }):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        out_text = Path(result[ArtifactType.RAW_TEXT].uri).read_text(
            encoding="utf-8",
        )
        assert out_text == "SDK output text"

    def test_sdk_missing_raises_clean_error(self, tmp_path: Path) -> None:
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        adapter = GoogleVisionAdapter(credentials_path=str(creds_path))
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with patch.dict(sys.modules, {
            "google.cloud.vision": None,
            "google.cloud": None,
        }):
            with pytest.raises(OCRAdapterError, match="google-cloud-vision"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )

    def test_sdk_internal_error_wrapped(self, tmp_path: Path) -> None:
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        adapter = GoogleVisionAdapter(credentials_path=str(creds_path))
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        mock_client = MagicMock()
        mock_client.document_text_detection.side_effect = RuntimeError(
            "SDK boom",
        )

        fake_vision = MagicMock()
        fake_vision.ImageAnnotatorClient = MagicMock(return_value=mock_client)
        fake_vision.Image = MagicMock(return_value="image_obj")
        fake_vision.ImageContext = MagicMock(return_value="ctx_obj")
        fake_module = MagicMock()
        fake_module.vision = fake_vision

        with patch.dict(sys.modules, {
            "google": fake_module,
            "google.cloud": fake_module,
            "google.cloud.vision": fake_vision,
        }):
            with pytest.raises(OCRAdapterError, match="RuntimeError.*SDK boom"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )


# ──────────────────────────────────────────────────────────────────────
# Artifact ID
# ──────────────────────────────────────────────────────────────────────


class TestGoogleVisionArtifactID:
    def test_artifact_id_uses_adapter_name(self, tmp_path: Path) -> None:
        adapter = GoogleVisionAdapter(api_key="x", name="custom_gv")
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        response = {"responses": [{"fullTextAnnotation": {"text": "x"}}]}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        produced = result[ArtifactType.RAW_TEXT]
        assert produced.id == "d1:custom_gv:raw_text"
        assert produced.document_id == "d1"
        assert produced.produced_by_step == "ocr"
