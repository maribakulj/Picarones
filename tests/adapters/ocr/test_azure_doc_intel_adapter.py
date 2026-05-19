"""Sprint A14-S34 — ``AzureDocIntelAdapter`` natif au contrat S26."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from picarones.adapters.ocr import (
    AzureDocIntelAdapter,
    BaseOCRAdapter,
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


class TestAzureDocIntelConstructor:
    def test_defaults(self) -> None:
        adapter = AzureDocIntelAdapter()
        assert adapter.name == "azure_doc_intel"
        assert adapter.model_id == "prebuilt-read"

    def test_custom_name(self) -> None:
        adapter = AzureDocIntelAdapter(name="my_azure")
        assert adapter.name == "my_azure"

    def test_custom_model_id(self) -> None:
        adapter = AzureDocIntelAdapter(model_id="prebuilt-document")
        assert adapter.model_id == "prebuilt-document"

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(OCRAdapterError, match="vide"):
            AzureDocIntelAdapter(name="")

    def test_rejects_invalid_chars_in_name(self) -> None:
        with pytest.raises(OCRAdapterError, match="invalide"):
            AzureDocIntelAdapter(name="bad name")

    def test_rejects_non_positive_timeout(self) -> None:
        with pytest.raises(OCRAdapterError, match="timeout_seconds"):
            AzureDocIntelAdapter(timeout_seconds=0)

    def test_rejects_non_positive_max_polling(self) -> None:
        with pytest.raises(OCRAdapterError, match="max_polling_attempts"):
            AzureDocIntelAdapter(max_polling_attempts=0)

    def test_rejects_negative_polling_interval(self) -> None:
        with pytest.raises(OCRAdapterError, match="polling_interval_base"):
            AzureDocIntelAdapter(polling_interval_base=-1.0)


# ──────────────────────────────────────────────────────────────────────
# Contrat BaseOCRAdapter
# ──────────────────────────────────────────────────────────────────────


class TestAzureDocIntelContract:
    def test_inherits_base_adapter(self) -> None:
        adapter = AzureDocIntelAdapter()
        assert isinstance(adapter, BaseOCRAdapter)

    def test_input_types(self) -> None:
        assert AzureDocIntelAdapter.input_types == frozenset(
            {ArtifactType.IMAGE},
        )

    def test_output_types(self) -> None:
        assert AzureDocIntelAdapter.output_types == frozenset(
            {ArtifactType.RAW_TEXT},
        )

    def test_execution_mode_is_io(self) -> None:
        assert AzureDocIntelAdapter.execution_mode == "io"


# ──────────────────────────────────────────────────────────────────────
# Auth resolution
# ──────────────────────────────────────────────────────────────────────


class TestAzureDocIntelAuth:
    def test_explicit_api_key_takes_priority(self) -> None:
        adapter = AzureDocIntelAdapter(api_key="explicit")
        with patch.dict("os.environ", {"AZURE_DOC_INTEL_KEY": "env"}):
            assert adapter._resolve_api_key() == "explicit"

    def test_env_api_key_fallback(self) -> None:
        adapter = AzureDocIntelAdapter()
        with patch.dict("os.environ", {"AZURE_DOC_INTEL_KEY": "env_key"}):
            assert adapter._resolve_api_key() == "env_key"

    def test_no_api_key_raises(self) -> None:
        adapter = AzureDocIntelAdapter()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(OCRAdapterError, match="AZURE_DOC_INTEL_KEY"):
                adapter._resolve_api_key()

    def test_explicit_endpoint_takes_priority(self) -> None:
        adapter = AzureDocIntelAdapter(endpoint="https://explicit.azure.com")
        with patch.dict(
            "os.environ", {"AZURE_DOC_INTEL_ENDPOINT": "https://env.azure.com"},
        ):
            assert adapter._resolve_endpoint() == "https://explicit.azure.com"

    def test_env_endpoint_fallback(self) -> None:
        adapter = AzureDocIntelAdapter()
        with patch.dict(
            "os.environ", {"AZURE_DOC_INTEL_ENDPOINT": "https://env.azure.com/"},
        ):
            # Note : .rstrip("/") supprime le trailing slash.
            assert adapter._resolve_endpoint() == "https://env.azure.com"

    def test_no_endpoint_raises(self) -> None:
        adapter = AzureDocIntelAdapter()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(
                OCRAdapterError, match="AZURE_DOC_INTEL_ENDPOINT",
            ):
                adapter._resolve_endpoint()


# ──────────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────────


class TestAzureDocIntelInputValidation:
    def test_missing_image_input_raises(self) -> None:
        adapter = AzureDocIntelAdapter(
            api_key="x", endpoint="https://test.azure.com",
        )
        with pytest.raises(OCRAdapterError, match="IMAGE manquant"):
            adapter.execute(inputs={}, params={}, context=_make_context())

    def test_image_artifact_without_uri_raises(self) -> None:
        adapter = AzureDocIntelAdapter(
            api_key="x", endpoint="https://test.azure.com",
        )
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
        adapter = AzureDocIntelAdapter(
            api_key="x", endpoint="https://test.azure.com",
        )
        artifact = _make_image_artifact("/nonexistent/img.png")
        with pytest.raises(OCRAdapterError, match="introuvable"):
            adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )


# ──────────────────────────────────────────────────────────────────────
# REST path
# ──────────────────────────────────────────────────────────────────────


class TestAzureDocIntelREST:
    def _patch_no_sdk(self):
        """Mock le SDK Azure comme absent → fallback REST."""
        return patch.dict(sys.modules, {
            "azure.ai.documentintelligence": None,
            "azure.core.credentials": None,
        })

    def _make_initial_response(self):
        """Mock initial POST response retournant Operation-Location."""
        mock_resp = MagicMock()
        mock_resp.headers = {"Operation-Location": "https://op-status-url"}
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    def _make_polling_response(self, status: str, text_lines: list[str] | None = None):
        """Mock polling response avec le status donné."""
        result = {"status": status}
        if status == "succeeded":
            result["analyzeResult"] = {
                "pages": [{
                    "lines": [{"content": line} for line in (text_lines or [])],
                }],
            }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(result).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    def test_succeeded_returns_text(self, tmp_path: Path) -> None:
        adapter = AzureDocIntelAdapter(
            api_key="k", endpoint="https://e.azure.com",
            polling_interval_base=0,  # pas de sleep dans les tests
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        initial = self._make_initial_response()
        succeeded = self._make_polling_response(
            "succeeded", text_lines=["Ligne 1", "Ligne 2"],
        )

        with self._patch_no_sdk(), patch(
            "urllib.request.urlopen",
            side_effect=[initial, succeeded],
        ):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        out_text = Path(result[ArtifactType.RAW_TEXT].uri).read_text(
            encoding="utf-8",
        )
        assert out_text == "Ligne 1\nLigne 2"

    def test_running_then_succeeded(self, tmp_path: Path) -> None:
        adapter = AzureDocIntelAdapter(
            api_key="k", endpoint="https://e.azure.com",
            polling_interval_base=0,
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with self._patch_no_sdk(), patch(
            "urllib.request.urlopen",
            side_effect=[
                self._make_initial_response(),
                self._make_polling_response("running"),
                self._make_polling_response("running"),
                self._make_polling_response(
                    "succeeded", text_lines=["Done"],
                ),
            ],
        ):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        out_text = Path(result[ArtifactType.RAW_TEXT].uri).read_text(
            encoding="utf-8",
        )
        assert out_text == "Done"

    def test_failed_status_raises(self, tmp_path: Path) -> None:
        adapter = AzureDocIntelAdapter(
            api_key="k", endpoint="https://e.azure.com",
            polling_interval_base=0,
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with self._patch_no_sdk(), patch(
            "urllib.request.urlopen",
            side_effect=[
                self._make_initial_response(),
                self._make_polling_response("failed"),
            ],
        ):
            with pytest.raises(OCRAdapterError, match="failed"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )

    def test_canceled_status_raises(self, tmp_path: Path) -> None:
        adapter = AzureDocIntelAdapter(
            api_key="k", endpoint="https://e.azure.com",
            polling_interval_base=0,
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with self._patch_no_sdk(), patch(
            "urllib.request.urlopen",
            side_effect=[
                self._make_initial_response(),
                self._make_polling_response("canceled"),
            ],
        ):
            with pytest.raises(OCRAdapterError, match="canceled"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )

    def test_polling_timeout_raises(self, tmp_path: Path) -> None:
        adapter = AzureDocIntelAdapter(
            api_key="k", endpoint="https://e.azure.com",
            polling_interval_base=0,
            max_polling_attempts=2,
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with self._patch_no_sdk(), patch(
            "urllib.request.urlopen",
            side_effect=[
                self._make_initial_response(),
                self._make_polling_response("running"),
                self._make_polling_response("running"),
            ],
        ):
            with pytest.raises(OCRAdapterError, match="timeout polling"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )

    def test_no_operation_location_raises(self, tmp_path: Path) -> None:
        adapter = AzureDocIntelAdapter(
            api_key="k", endpoint="https://e.azure.com",
            polling_interval_base=0,
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        # Initial POST sans Operation-Location.
        bad_initial = MagicMock()
        bad_initial.headers = {}
        bad_initial.__enter__.return_value = bad_initial

        with self._patch_no_sdk(), patch(
            "urllib.request.urlopen",
            side_effect=[bad_initial],
        ):
            with pytest.raises(OCRAdapterError, match="Operation-Location"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )

    def test_writes_to_stem_name_pattern(self, tmp_path: Path) -> None:
        adapter = AzureDocIntelAdapter(
            api_key="k", endpoint="https://e.azure.com",
            polling_interval_base=0,
            name="my_azure",
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        with self._patch_no_sdk(), patch(
            "urllib.request.urlopen",
            side_effect=[
                self._make_initial_response(),
                self._make_polling_response("succeeded", text_lines=["x"]),
            ],
        ):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        out_path = Path(result[ArtifactType.RAW_TEXT].uri)
        from picarones.adapters.output_paths import _pipeline_path_segment
        seg = _pipeline_path_segment(_make_context())
        assert out_path.name == f"page.{seg}.my_azure.txt"


# ──────────────────────────────────────────────────────────────────────
# SDK path
# ──────────────────────────────────────────────────────────────────────


class TestAzureDocIntelSDK:
    def test_sdk_call_succeeds(self, tmp_path: Path) -> None:
        adapter = AzureDocIntelAdapter(
            api_key="k", endpoint="https://e.azure.com",
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        # Mock du résultat SDK avec pages.lines.content.
        mock_line_a = MagicMock()
        mock_line_a.content = "Ligne A"
        mock_line_b = MagicMock()
        mock_line_b.content = "Ligne B"
        mock_page = MagicMock()
        mock_page.lines = [mock_line_a, mock_line_b]
        mock_result = MagicMock()
        mock_result.pages = [mock_page]

        mock_poller = MagicMock()
        mock_poller.result.return_value = mock_result
        mock_client = MagicMock()
        mock_client.begin_analyze_document.return_value = mock_poller

        fake_di_module = MagicMock()
        fake_di_module.DocumentIntelligenceClient = MagicMock(
            return_value=mock_client,
        )
        fake_creds_module = MagicMock()
        fake_creds_module.AzureKeyCredential = MagicMock(return_value="creds")

        with patch.dict(sys.modules, {
            "azure": MagicMock(),
            "azure.ai": MagicMock(),
            "azure.ai.documentintelligence": fake_di_module,
            "azure.core": MagicMock(),
            "azure.core.credentials": fake_creds_module,
        }):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        out_text = Path(result[ArtifactType.RAW_TEXT].uri).read_text(
            encoding="utf-8",
        )
        assert out_text == "Ligne A\nLigne B"

    def test_sdk_internal_error_wrapped(self, tmp_path: Path) -> None:
        adapter = AzureDocIntelAdapter(
            api_key="k", endpoint="https://e.azure.com",
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        mock_client = MagicMock()
        mock_client.begin_analyze_document.side_effect = RuntimeError(
            "Azure boom",
        )

        fake_di_module = MagicMock()
        fake_di_module.DocumentIntelligenceClient = MagicMock(
            return_value=mock_client,
        )
        fake_creds_module = MagicMock()
        fake_creds_module.AzureKeyCredential = MagicMock(return_value="creds")

        with patch.dict(sys.modules, {
            "azure": MagicMock(),
            "azure.ai": MagicMock(),
            "azure.ai.documentintelligence": fake_di_module,
            "azure.core": MagicMock(),
            "azure.core.credentials": fake_creds_module,
        }):
            with pytest.raises(OCRAdapterError, match="RuntimeError.*Azure boom"):
                adapter.execute(
                    inputs={ArtifactType.IMAGE: artifact},
                    params={},
                    context=_make_context(),
                )


# ──────────────────────────────────────────────────────────────────────
# Artifact ID
# ──────────────────────────────────────────────────────────────────────


class TestAzureDocIntelArtifactID:
    def test_artifact_id_uses_adapter_name(self, tmp_path: Path) -> None:
        adapter = AzureDocIntelAdapter(
            api_key="k", endpoint="https://e.azure.com",
            polling_interval_base=0,
            name="custom_az",
        )
        image_path = _make_dummy_image(tmp_path)
        artifact = _make_image_artifact(str(image_path))

        mock_resp_initial = MagicMock()
        mock_resp_initial.headers = {"Operation-Location": "https://op"}
        mock_resp_initial.__enter__.return_value = mock_resp_initial

        result_payload = {
            "status": "succeeded",
            "analyzeResult": {
                "pages": [{"lines": [{"content": "x"}]}],
            },
        }
        mock_resp_polling = MagicMock()
        mock_resp_polling.read.return_value = json.dumps(
            result_payload,
        ).encode("utf-8")
        mock_resp_polling.__enter__.return_value = mock_resp_polling

        with patch.dict(sys.modules, {
            "azure.ai.documentintelligence": None,
            "azure.core.credentials": None,
        }), patch(
            "urllib.request.urlopen",
            side_effect=[mock_resp_initial, mock_resp_polling],
        ):
            result = adapter.execute(
                inputs={ArtifactType.IMAGE: artifact},
                params={},
                context=_make_context(),
            )
        produced = result[ArtifactType.RAW_TEXT]
        assert produced.id == "d1:custom_az:raw_text"
        assert produced.document_id == "d1"
        assert produced.produced_by_step == "ocr"
