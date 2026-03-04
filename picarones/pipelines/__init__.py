"""Pipelines OCR+LLM : combinent un moteur OCR avec un LLM de correction."""

from picarones.pipelines.base import OCRLLMPipeline, PipelineMode
from picarones.pipelines.over_normalization import (
    OverNormalizationResult,
    detect_over_normalization,
)

__all__ = [
    "OCRLLMPipeline",
    "PipelineMode",
    "OverNormalizationResult",
    "detect_over_normalization",
]
