"""Index thématique : tests des engines OCR et adapters LLM.

Chantier 6 du plan d'évolution post-Sprint 97.

Tests couvrant cette feature
----------------------------
- :mod:`tests.test_engines` — 5 adapters OCR (Tesseract, Pero,
  Mistral OCR, Google Vision, Azure DI), `BaseOCREngine` factorisé.
- :mod:`tests.test_engines_cloud` — tests cloud-only (gated).
- :mod:`tests.test_chantier4` (sous-classes ``TestNormalizeLlmContent``,
  ``TestLogHttpError``, ``TestLlmAdaptersInheritEnvVar``) — helpers
  factorisés `picarones.llm.base`.
- :mod:`tests.test_alto_baseline` (chantier 1) — `BaseOCREngine`
  refondu (hooks `_run_with_native` + `_extract_raw_confidences`).

Sprints d'origine
-----------------
- Sprint 1 : adapters Tesseract et Pero OCR (texte historique).
- Sprint 4 : adapter Mistral OCR (endpoint /v1/ocr dédié).
- Sprint 4 : adapter Google Vision et Azure DI.
- Sprint 15 : ``test_sprint15_llm_pipeline_bugs.py`` — fix
  normalisation `ContentChunk` Mistral (propagé aux 4 adapters
  par chantier 4).
- Sprints 47-51 : ``test_sprint{47,48,49,50,51}_*_confidences.py``
  — exposition `token_confidences` natifs des 5 adapters
  (refondue par chantier 1 sur les hooks unifiés).

Pour exécuter :

.. code-block:: bash

    pytest tests/test_engines*.py \\
           tests/test_sprint{47,48,49,50,51}_*.py \\
           tests/test_chantier4.py::TestNormalizeLlmContent \\
           tests/test_chantier4.py::TestLogHttpError
"""

# Index documentaire — pas de tests propres.
