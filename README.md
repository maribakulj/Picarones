# Picarones

> **Plateforme de comparaison de moteurs OCR/HTR pour documents patrimoniaux**
Apache 2.0

[![CI](https://github.com/bnf/picarones/actions/workflows/ci.yml/badge.svg)](https://github.com/bnf/picarones/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

**Picarones** est un outil open-source conçu pour comparer rigoureusement des moteurs OCR et HTR
(Tesseract, Pero OCR, Kraken, APIs cloud…) ainsi que des pipelines OCR+LLM sur des corpus de
documents historiques — manuscrits, imprimés anciens, archives.

---

*[English version below](#english)*

---

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Installation rapide](#installation-rapide)
- [Usage rapide](#usage-rapide)
- [Moteurs supportés](#moteurs-supportés)
- [Structure du projet](#structure-du-projet)
- [Variables d'environnement](#variables-denvironnement)
- [Roadmap](#roadmap)
- [English](#english)

---

## Fonctionnalités

### Métriques adaptées aux documents patrimoniaux

- **CER** (Character Error Rate) : brut, NFC, caseless, diplomatique (ſ=s, u=v, i=j…)
- **WER**, MER, WIL avec tokenisation historique
- **Matrice de confusion unicode** — fingerprint de chaque moteur
- **Scores ligatures** : fi, fl, ff, œ, æ, ꝑ, ꝓ…
- **Scores diacritiques** : accents, cédilles, trémas
- **Taxonomie des erreurs** en 10 classes (confusion visuelle, abréviation, ligature, casse…)
- **Intervalles de confiance à 95%** par bootstrap — tests de Wilcoxon pour la significativité
- **Score de difficulté intrinsèque** par document (indépendant des moteurs)

### Pipelines OCR+LLM

- Chaînes composables : `tesseract → gpt-4o`, `pero_ocr → claude-sonnet`, LLM zero-shot…
- Modes : texte seul, image+texte, zero-shot
- Détection de **sur-normalisation LLM** : le LLM modernise-t-il à tort la graphie médiévale ?
- Bibliothèque de prompts pour manuscrits médiévaux, imprimés anciens, latin…

### Import de corpus

| Source | Commande |
|--------|----------|
| Dossier local | `picarones run --corpus ./corpus/` |
| IIIF (Gallica, Bodleian, BL…) | `picarones import iiif <url>` |
| Gallica (API BnF + OCR) | `GallicaClient` / `picarones import iiif` |
| HuggingFace Datasets | `picarones import hf <dataset>` |
| HTR-United | `picarones import htr-united` |
| eScriptorium | `EScriptoriumClient` |

### Rapport HTML interactif

- Fichier HTML **auto-contenu**, lisible hors-ligne
- Tableau de classement trié, graphiques radar, histogrammes
- Vue galerie avec filtres dynamiques et badges CER colorés
- Diff coloré façon GitHub, scroll synchronisé N-way
- Vue spécifique OCR+LLM : diff triple GT / OCR brut / après correction
- Vue Caractères : matrice de confusion unicode interactive
- Export CSV, JSON, ALTO XML, PAGE XML, images annotées

### Suivi longitudinal & robustesse

- **Base SQLite** optionnelle pour historiser les runs
- **Courbes d'évolution CER** dans le temps par moteur
- **Détection automatique des régressions** entre deux runs
- **Analyse de robustesse** : bruit, flou, rotation, réduction de résolution, binarisation
- Commandes `picarones history`, `picarones robustness`

---

## Installation rapide

```bash
# Cloner et installer
git clone https://github.com/bnf/picarones.git
cd picarones
pip install -e .

# Tesseract (binaire système, obligatoire pour le moteur Tesseract)
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-lat

# macOS
brew install tesseract

# Vérifier l'installation
picarones engines
```

Voir [INSTALL.md](INSTALL.md) pour un guide détaillé (Linux, macOS, Windows, Docker).

---

## Usage rapide

```bash
# Rapport de démonstration (sans moteur OCR installé)
picarones demo

# Benchmark sur un corpus local
picarones run --corpus ./mon_corpus/ --engines tesseract --output resultats.json

# Générer le rapport HTML interactif
picarones report --results resultats.json --output rapport.html

# Calculer CER/WER entre deux fichiers
picarones metrics --reference gt.txt --hypothesis ocr.txt

# Importer depuis Gallica (IIIF)
picarones import iiif https://gallica.bnf.fr/ark:/12148/xxx/manifest.json --pages 1-10

# Suivi longitudinal (historique des runs)
picarones history --demo
picarones history --engine tesseract --regression

# Analyse de robustesse
picarones robustness --corpus ./gt/ --engine tesseract --demo

# Interface web locale
picarones serve
```

---

## Moteurs supportés

| Moteur | Type | Installation |
|--------|------|--------------|
| **Tesseract 5** | Local CLI | `pip install pytesseract` + binaire système |
| **Pero OCR** | Local Python | `pip install pero-ocr` |
| **Kraken** | Local Python | `pip install kraken` |
| **Mistral OCR** | API REST | Clé `MISTRAL_API_KEY` |
| **GPT-4o** (LLM) | API REST | Clé `OPENAI_API_KEY` |
| **Claude Sonnet** (LLM) | API REST | Clé `ANTHROPIC_API_KEY` |
| **Mistral Large** (LLM) | API REST | Clé `MISTRAL_API_KEY` |
| **Google Vision** | API REST | Credentials JSON Google |
| **AWS Textract** | API REST | Credentials AWS |
| **Azure Doc. Intel.** | API REST | Endpoint + clé Azure |
| **Ollama** (LLM local) | Local | `ollama serve` |
| **Moteur custom** | CLI/API YAML | Déclaration YAML, sans code |

---

## Structure du projet

```
picarones/
├── cli.py                      # CLI Click (run, demo, report, history, robustness…)
├── fixtures.py                 # Données de test fictives réalistes
├── core/
│   ├── corpus.py               # Chargement corpus (dossier, ALTO, PAGE XML…)
│   ├── metrics.py              # CER, WER, MER, WIL (jiwer)
│   ├── normalization.py        # Normalisation unicode, profils diplomatiques
│   ├── statistics.py           # Bootstrap CI, Wilcoxon, corrélations
│   ├── confusion.py            # Matrice de confusion unicode
│   ├── char_scores.py          # Scores ligatures et diacritiques
│   ├── taxonomy.py             # Taxonomie des erreurs (10 classes)
│   ├── structure.py            # Analyse structurelle
│   ├── image_quality.py        # Métriques qualité image
│   ├── difficulty.py           # Score de difficulté intrinsèque
│   ├── history.py              # Suivi longitudinal SQLite
│   ├── robustness.py           # Analyse de robustesse
│   ├── results.py              # Modèles de données + export JSON
│   └── runner.py               # Orchestrateur benchmark
├── engines/                    # Adaptateurs moteurs OCR
├── llm/                        # Adaptateurs LLM (OpenAI, Anthropic, Mistral, Ollama)
├── importers/                  # Sources d'import (IIIF, Gallica, eScriptorium, HF…)
├── pipelines/                  # Orchestrateur OCR+LLM
├── report/                     # Générateur rapport HTML
└── web/                        # Interface web FastAPI
tests/                          # Tests unitaires et d'intégration (743 tests)
```

---

## Variables d'environnement

```bash
# APIs LLM (selon les moteurs utilisés)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export MISTRAL_API_KEY="..."

# APIs OCR cloud (optionnel)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="eu-west-1"
export AZURE_DOC_INTEL_ENDPOINT="https://..."
export AZURE_DOC_INTEL_KEY="..."
```

---

## Roadmap

| Sprint | Statut | Livrables |
|--------|--------|-----------|
| Sprint 1 | ✅ | Structure, Tesseract, Pero OCR, CER/WER, CLI |
| Sprint 2 | ✅ | Rapport HTML v1, diff coloré, galerie |
| Sprint 3 | ✅ | Pipelines OCR+LLM, GPT-4o, Claude |
| Sprint 4 | ✅ | APIs cloud, import IIIF, normalisation diplomatique |
| Sprint 5 | ✅ | Métriques avancées : confusion unicode, ligatures, taxonomie |
| Sprint 6 | ✅ | Interface web FastAPI, HTR-United, HuggingFace, Ollama |
| Sprint 7 | ✅ | Rapport HTML v2 : Wilcoxon, clustering, scatter plots |
| Sprint 8 | ✅ | eScriptorium, Gallica API, historique longitudinal, robustesse |
| Sprint 9 | ✅ | Documentation, packaging, Docker, CI/CD |

---

## Contribuer

Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour ajouter un moteur OCR, un adaptateur LLM, ou soumettre une pull request.

---

## Licence

Apache License 2.0 — © BnF — Département numérique

---

---

# English

## Picarones — OCR/HTR Benchmark Platform for Heritage Documents

**Picarones** is an open-source platform for rigorously comparing OCR and HTR engines (Tesseract,
Pero OCR, Kraken, cloud APIs…) and OCR+LLM pipelines on historical document corpora — manuscripts,
early printed books, archives.

### Key Features

- **Metrics tailored to historical documents**: CER (raw, NFC, caseless, diplomatic), WER, MER,
  WIL; unicode confusion matrix; ligature and diacritic scores; 10-class error taxonomy; bootstrap
  confidence intervals; Wilcoxon significance tests
- **OCR+LLM pipelines**: composable chains (`tesseract → gpt-4o`), three modes (text-only,
  image+text, zero-shot), LLM over-normalisation detection
- **Corpus import**: local folder, IIIF (Gallica, Bodleian, BL…), Gallica API + OCR, HuggingFace
  Datasets, HTR-United, eScriptorium
- **Interactive HTML report**: self-contained file, sortable ranking, gallery, coloured diff,
  unicode character view, CSV/JSON/ALTO/PAGE XML export
- **Longitudinal tracking**: SQLite benchmark history, CER evolution curves, automatic regression
  detection
- **Robustness analysis**: degraded image versions (noise, blur, rotation, resolution,
  binarisation), critical threshold detection

### Quick Start

```bash
pip install -e .
sudo apt install tesseract-ocr tesseract-ocr-fra   # Ubuntu/Debian
picarones demo          # demo report without any engine installed
picarones engines       # list available engines
picarones run --corpus ./corpus/ --engines tesseract --output results.json
picarones report --results results.json
```

See [INSTALL.md](INSTALL.md) for detailed installation on Linux, macOS, Windows, and Docker.

### Supported Engines

Tesseract 5 · Pero OCR · Kraken · Mistral OCR · GPT-4o · Claude Sonnet · Mistral Large ·
Google Vision · AWS Textract · Azure Document Intelligence · Ollama (local LLMs) · Custom YAML engine

### License

Apache License 2.0 — © BnF — Département numérique
