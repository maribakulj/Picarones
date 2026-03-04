# Picarones

> **Plateforme de comparaison de moteurs OCR/HTR pour documents patrimoniaux**
> BnF — Département numérique · Apache 2.0

Picarones permet d'évaluer et de comparer rigoureusement des moteurs OCR (Tesseract, Pero OCR, Kraken, APIs cloud…) ainsi que des pipelines OCR+LLM sur des corpus de documents historiques — manuscrits, imprimés anciens, archives.

---

## Sprint 1 — Ce qui est implémenté

- Structure complète du projet Python (`picarones/`)
- Adaptateur **Tesseract 5** (`pytesseract`)
- Adaptateur **Pero OCR** (necessite `pero-ocr`)
- Interface abstraite `BaseOCREngine` pour ajouter facilement de nouveaux moteurs
- Calcul **CER** et **WER** via `jiwer` (brut, NFC, caseless, normalisé, MER, WIL)
- Chargement de **corpus** depuis dossier local (paires image / `.gt.txt`)
- **Export JSON** structuré des résultats avec classement
- **CLI** `click` : commandes `run`, `metrics`, `engines`, `info`

---

## Installation

```bash
pip install -e .

# Pour Tesseract, installer aussi le binaire système :
# Ubuntu/Debian : sudo apt install tesseract-ocr tesseract-ocr-fra
# macOS         : brew install tesseract

# Pour Pero OCR (optionnel) :
pip install pero-ocr
```

## Usage rapide

```bash
# Lancer un benchmark sur un corpus local
picarones run --corpus ./mon_corpus/ --engines tesseract --output resultats.json

# Plusieurs moteurs
picarones run --corpus ./corpus/ --engines tesseract,pero_ocr --lang fra

# Calculer CER/WER entre deux fichiers
picarones metrics --reference gt.txt --hypothesis ocr.txt

# Lister les moteurs disponibles
picarones engines

# Infos de version
picarones info
```

## Structure du projet

```
picarones/
├── __init__.py
├── cli.py                  # CLI Click
├── core/
│   ├── corpus.py           # Chargement corpus
│   ├── metrics.py          # CER/WER (jiwer)
│   ├── results.py          # Modèles de données + export JSON
│   └── runner.py           # Orchestrateur benchmark
└── engines/
    ├── base.py             # Interface abstraite BaseOCREngine
    ├── tesseract.py        # Adaptateur Tesseract
    └── pero_ocr.py         # Adaptateur Pero OCR
tests/
├── test_metrics.py
├── test_corpus.py
├── test_engines.py
└── test_results.py
```

## Format du corpus

Un corpus local est un dossier contenant des paires :

```
corpus/
├── page_001.jpg
├── page_001.gt.txt    ← vérité terrain UTF-8
├── page_002.png
├── page_002.gt.txt
└── ...
```

## Format de sortie JSON

```json
{
  "picarones_version": "0.1.0",
  "run_date": "2025-03-04T...",
  "corpus": { "name": "...", "document_count": 50 },
  "ranking": [
    { "engine": "tesseract", "mean_cer": 0.043, "mean_wer": 0.112 }
  ],
  "engine_reports": [...]
}
```

## Lancer les tests

```bash
pytest
```

## Roadmap

| Sprint | Livrables |
|--------|-----------|
| **Sprint 1** ✅ | Structure, adaptateurs Tesseract + Pero OCR, CER/WER, JSON, CLI |
| Sprint 2 | Rapport HTML interactif avec diff coloré |
| Sprint 3 | Pipelines OCR+LLM (GPT-4o, Claude) |
| Sprint 4 | APIs cloud OCR, import IIIF, normalisation diplomatique |
| Sprint 5 | Métriques avancées : matrice de confusion unicode, ligatures |
| Sprint 6 | Interface web FastAPI, import HTR-United / HuggingFace |
