# Pipelines composées — exemples YAML

Ce dossier contient des spécifications déclaratives de pipelines à
exécuter via `picarones pipeline run` (Sprint 70). Chaque YAML
référence des classes `BaseModule` accessibles dans le `PYTHONPATH`.

## Pipelines disponibles

### `ocr_to_alto.yaml`

Pipeline de référence livrée par le **chantier 1** post-Sprint 97 : un
moteur OCR produit du texte, puis le reconstructeur ALTO de référence
(`TextToAltoMonoRegion`) en dérive un ALTO XML mono-région.

Le rapport produit montre **deux jonctions** évaluées contre la GT
multi-niveaux du document :

- `(TEXT, TEXT)` : CER/WER/MER/WIL après OCR
- `(ALTO, ALTO)` : `alto_text_cer`/`alto_text_wer` sur le texte
  reconstruit depuis l'ALTO

### Utilisation

```bash
picarones pipeline run examples/pipelines/ocr_to_alto.yaml \
    --corpus ./mon_corpus \
    --output-html rapport.html
```

Le corpus doit contenir au moins une image avec sa GT texte ; pour
voir la jonction `(ALTO, ALTO)`, ajouter un fichier `.gt.alto.xml` à
côté de l'image (cf. Sprint 32 — multi-level GT).

## Composer sa propre pipeline

Les modules `BaseModule` sont identifiés par leur **chemin Python
pointé**. Picarones livre :

| Module | Chemin | Types |
|---|---|---|
| Tesseract OCR | `picarones.engines.tesseract.TesseractEngine` | `IMAGE → TEXT` |
| Pero OCR | `picarones.engines.pero_ocr.PeroOCREngine` | `IMAGE → TEXT` |
| Mistral OCR | `picarones.engines.mistral_ocr.MistralOCREngine` | `IMAGE → TEXT` |
| Google Vision | `picarones.engines.google_vision.GoogleVisionEngine` | `IMAGE → TEXT` |
| Azure DI | `picarones.engines.azure_doc_intel.AzureDocIntelEngine` | `IMAGE → TEXT` |
| Reconstructeur ALTO baseline | `picarones.modules.alto_text_to_mono_region.TextToAltoMonoRegion` | `IMAGE, TEXT → ALTO` |

Pour brancher votre propre module (correcteur LLM, reconstructeur
ALTO plus avancé, mappeur VLM…), exposez une classe qui hérite de
`picarones.core.modules.BaseModule` et déclarez son chemin dans le
YAML. Voir `docs/tutorials/writing-a-pipeline-module.md` pour les détails.
