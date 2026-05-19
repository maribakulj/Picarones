"""Sprint A14-S52 — hiérarchie d'erreurs unifiée (fix audit #7 + #11).

Avant S52 :
- LLM/VLM levaient OCRAdapterError (mauvaise classe).
- JobStoreError héritait de Exception (pas de PicaronesError).
- Pas de racine commune AdapterStepError pour catcher OCR+LLM+VLM.

Après S52 :
- AdapterStepError(PicaronesError) est la racine commune.
- OCRAdapterError, LLMAdapterError, VLMAdapterError héritent.
- JobStoreError hérite de PicaronesError.
"""

from __future__ import annotations

import pytest

from picarones.adapters.llm.base import LLMAdapterError
from picarones.adapters.ocr.base import OCRAdapterError
from picarones.adapters.storage import JobStoreError
from picarones.adapters.vlm.base import VLMAdapterError
from picarones.domain.errors import AdapterStepError, PicaronesError


class TestErrorInheritance:
    def test_ocr_inherits_adapter_step_error(self) -> None:
        assert issubclass(OCRAdapterError, AdapterStepError)
        assert issubclass(OCRAdapterError, PicaronesError)

    def test_llm_inherits_adapter_step_error(self) -> None:
        assert issubclass(LLMAdapterError, AdapterStepError)
        assert issubclass(LLMAdapterError, PicaronesError)

    def test_vlm_inherits_adapter_step_error(self) -> None:
        assert issubclass(VLMAdapterError, AdapterStepError)
        assert issubclass(VLMAdapterError, PicaronesError)

    def test_jobstore_inherits_picarones_error(self) -> None:
        # Avant S52, héritait de Exception → un caller `except
        # PicaronesError` ratait JobStoreError.  Maintenant inclus.
        assert issubclass(JobStoreError, PicaronesError)


class TestPolymorphicCatch:
    """Un caller peut catcher AdapterStepError pour gérer toute
    erreur d'adapter sans connaître la sous-classe."""

    def test_catches_ocr(self) -> None:
        with pytest.raises(AdapterStepError):
            raise OCRAdapterError("ocr boom")

    def test_catches_llm(self) -> None:
        with pytest.raises(AdapterStepError):
            raise LLMAdapterError("llm boom")

    def test_catches_vlm(self) -> None:
        with pytest.raises(AdapterStepError):
            raise VLMAdapterError("vlm boom")

    def test_picarones_catches_all_adapter_errors(self) -> None:
        for cls in (OCRAdapterError, LLMAdapterError, VLMAdapterError):
            with pytest.raises(PicaronesError):
                raise cls("boom")

    def test_picarones_catches_jobstore(self) -> None:
        with pytest.raises(PicaronesError):
            raise JobStoreError("store boom")
