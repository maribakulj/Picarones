# `picarones.adapters` — implémentations concrètes

## OCR

::: picarones.adapters.ocr.base
    options:
      show_root_heading: true

::: picarones.adapters.ocr.tesseract
    options:
      show_root_heading: true
      members: ["TesseractAdapter"]

::: picarones.adapters.ocr.pero_ocr
    options:
      show_root_heading: true
      members: ["PeroOCRAdapter"]

::: picarones.adapters.ocr.mistral_ocr
    options:
      show_root_heading: true
      members: ["MistralOCRAdapter"]

::: picarones.adapters.ocr.google_vision
    options:
      show_root_heading: true
      members: ["GoogleVisionAdapter"]

::: picarones.adapters.ocr.azure_doc_intel
    options:
      show_root_heading: true
      members: ["AzureDocIntelAdapter"]

## LLM

::: picarones.adapters.llm.base
    options:
      show_root_heading: true
      members: ["BaseLLMAdapter", "LLMAdapterError", "LLMResult", "normalize_llm_content"]

::: picarones.adapters.llm.anthropic_adapter
    options:
      show_root_heading: true

::: picarones.adapters.llm.openai_adapter
    options:
      show_root_heading: true

::: picarones.adapters.llm.mistral_adapter
    options:
      show_root_heading: true

::: picarones.adapters.llm.ollama_adapter
    options:
      show_root_heading: true

## VLM

::: picarones.adapters.vlm.base
    options:
      show_root_heading: true
      members: ["BaseVLMAdapter", "VLMAdapterError"]

## Storage

::: picarones.adapters.storage.artifact_store
    options:
      show_root_heading: true

::: picarones.adapters.storage.job_store
    options:
      show_root_heading: true

## Helpers

::: picarones.adapters.output_paths
    options:
      show_root_heading: true

::: picarones.adapters._retry
    options:
      show_root_heading: true
