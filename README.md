---
title: Picarones
emoji: 📜
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Picarones

> **OCR/HTR Benchmarking Platform for Heritage Documents**

[![CI](https://github.com/maribakulj/Picarones/actions/workflows/ci.yml/badge.svg)](https://github.com/maribakulj/Picarones/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace%20Space-yellow.svg)](https://huggingface.co/spaces/Ma-Ri-Ba-Ku/Picarones)

---

**Picarones** is an open-source platform for rigorously comparing OCR and HTR engines
(Tesseract, Pero OCR, Kraken, cloud APIs...) and OCR+LLM pipelines on historical document
corpora -- manuscripts, early printed books, and archives.

It provides heritage-specific metrics (diplomatic CER, ligature scores, medieval abbreviation
handling), composable OCR+LLM pipelines, interactive HTML reports, and multiple corpus import
sources including IIIF, Gallica, HuggingFace, and eScriptorium.

---

## Table of Contents

- [Features](#features)
  - [Heritage-Specific Metrics](#heritage-specific-metrics)
  - [OCR+LLM Pipelines](#ocr-llm-pipelines)
  - [Corpus Import](#corpus-import)
  - [Interactive HTML Report](#interactive-html-report)
  - [Longitudinal Tracking & Robustness](#longitudinal-tracking--robustness)
  - [Web Interface](#web-interface)
- [Quick Start](#quick-start)
- [Installation](#installation)
  - [From Source](#from-source)
  - [Docker](#docker)
  - [Optional Extras](#optional-extras)
- [Usage](#usage)
  - [CLI Commands](#cli-commands)
  - [Web Interface](#web-interface-1)
  - [Pipeline Modes](#pipeline-modes)
- [Supported Engines](#supported-engines)
- [Normalization Profiles](#normalization-profiles)
- [Error Taxonomy](#error-taxonomy)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [CI/CD](#cicd)
- [Development](#development)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Heritage-Specific Metrics

- **CER** (Character Error Rate) in four variants: raw, NFC-normalized, caseless, and
  **diplomatic** (historical equivalences: long s = s, u = v, i = j, etc.)
- **WER**, **MER**, **WIL** with historical-aware tokenization (via [jiwer](https://github.com/jitsi/jiwer))
- **Unicode confusion matrix** -- fingerprint each engine's character-level errors
- **Ligature and diacritic scores** -- track handling of fi, fl, ff, oe, ae, p-bar, and other
  medieval glyphs
- **10-class error taxonomy** -- automatic classification of every error (visual confusion,
  abbreviation, segmentation, lacuna, over-normalization, etc.)
- **Bootstrap 95% confidence intervals** and **Wilcoxon signed-rank tests** for statistical
  significance
- **Intrinsic difficulty score** per document, independent of engine performance
- **Line-level error distribution** with Gini coefficient and percentile analysis
- **VLM hallucination detection** -- anchor score and length ratio to flag fabricated output

### OCR+LLM Pipelines

- Composable chains: `tesseract -> gpt-4o`, `pero_ocr -> claude-sonnet`, zero-shot VLM, etc.
- Three pipeline modes: text-only post-correction, image+text post-correction, and zero-shot
- **Over-normalization detection** -- does the LLM silently modernize historical spellings?
- Versioned prompt library for medieval French, early modern French, medieval Latin, medieval
  English, and early modern English -- both correction and zero-shot variants

### Corpus Import

| Source | Method |
|--------|--------|
| Local folder | `picarones run --corpus ./corpus/` |
| IIIF manifests (Gallica, Bodleian, BL...) | `picarones import iiif <manifest-url>` |
| Gallica API (SRU + OCR) | `GallicaClient` / `picarones import iiif` |
| HuggingFace Datasets | `picarones import hf <dataset-id>` |
| HTR-United catalogue | `picarones import htr-united` |
| eScriptorium | `EScriptoriumClient` |
| ZIP upload (browser) | Web interface upload endpoint |

Supported corpus formats: plain text pairs (image + ground truth), **ALTO XML**, and **PAGE XML**.

### Interactive HTML Report

- **Self-contained HTML file** -- works offline, no server needed
- Sortable ranking table, radar charts, histograms (powered by Chart.js)
- Gallery view with dynamic filters and color-coded CER badges
- GitHub-style colored diff with synchronized N-way scrolling
- Triple diff view for OCR+LLM: ground truth / raw OCR / post-correction
- Unicode character view: interactive confusion matrix explorer
- Export to **CSV**, **JSON**, **ALTO XML**, **PAGE XML**, and annotated images

### Longitudinal Tracking & Robustness

- Optional **SQLite database** to record benchmark history across runs
- **CER evolution curves** over time, per engine
- **Automatic regression detection** between consecutive runs
- **Robustness analysis**: measure engine resilience to noise, blur, rotation, resolution
  reduction, and binarization
- Critical degradation threshold identification

### Web Interface

- **FastAPI** application with real-time **Server-Sent Events** (SSE) progress streaming
- Upload corpus as a **ZIP file** directly from the browser
- Dynamic engine and normalization profile selectors
- Browse and re-download generated HTML reports
- Bilingual **French/English** interface
- Deployable on HuggingFace Spaces (Docker, port 7860)

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/maribakulj/Picarones.git
cd Picarones
pip install -e .

# Install Tesseract (system binary, required for the Tesseract engine)
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-lat

# macOS
brew install tesseract

# Generate a demo report (no OCR engine needed)
picarones demo --output demo_report.html

# List available engines
picarones engines

# Run a benchmark
picarones run --corpus ./corpus/ --engines tesseract --output results.json

# Generate HTML report
picarones report --results results.json --output report.html

# Launch the web interface
picarones serve --port 8080
```

---

## Installation

### From Source

```bash
git clone https://github.com/maribakulj/Picarones.git
cd Picarones
pip install -e ".[dev,web]"    # includes test and web dependencies
```

**System requirements:**

- Python >= 3.11
- [Tesseract OCR 5](https://github.com/tesseract-ocr/tesseract) (for the Tesseract engine)

### Docker

```bash
docker build -t picarones .
docker run -p 7860:7860 \
  -e MISTRAL_API_KEY=... \
  -e OPENAI_API_KEY=... \
  picarones
```

The Docker image is based on Python 3.11-slim, includes Tesseract 5 with language packs
(fra, lat, eng, deu, ita, spa), and runs as a non-root user. A health check polls
`/health` every 30 seconds.

The [HuggingFace Space](https://huggingface.co/spaces/Ma-Ri-Ba-Ku/Picarones) uses this
same Docker image.

### Optional Extras

| Extra | Install command | What it adds |
|-------|----------------|--------------|
| `dev` | `pip install -e ".[dev]"` | pytest, pytest-cov, httpx, linting |
| `web` | `pip install -e ".[web]"` | FastAPI, uvicorn, python-multipart |
| `llm` | `pip install -e ".[llm]"` | OpenAI, Anthropic, Mistral SDKs |
| `hf` | `pip install -e ".[hf]"` | HuggingFace Datasets |
| `pero` | `pip install -e ".[pero]"` | Pero OCR engine |
| `kraken` | `pip install -e ".[kraken]"` | Kraken engine |
| `ocr-cloud` | `pip install -e ".[ocr-cloud]"` | Google Vision, AWS Textract, Azure Doc Intelligence |
| `all` | `pip install -e ".[all]"` | Everything except ocr-cloud |

See [INSTALL.md](INSTALL.md) for detailed instructions on Linux, macOS, Windows, and Docker.

---

## Usage

### CLI Commands

| Command | Description |
|---------|-------------|
| `picarones run` | Run a full benchmark on a corpus |
| `picarones report` | Generate an HTML report from JSON results |
| `picarones demo` | Generate a demo report with synthetic data (no engine required) |
| `picarones metrics` | Calculate CER/WER between two text files |
| `picarones engines` | List all available OCR engines and LLM adapters |
| `picarones info` | Display version and system information |
| `picarones serve` | Launch the FastAPI web interface |
| `picarones history` | Query longitudinal benchmark history (SQLite) |
| `picarones robustness` | Run robustness analysis with degraded images |
| `picarones import` | Import corpus from IIIF, HuggingFace, or HTR-United |

**Examples:**

```bash
# Benchmark with Tesseract, French language, PSM 6
picarones run --corpus ./manuscripts/ --engines tesseract --lang fra --psm 6 \
  --output results.json --verbose

# Compare two text files
picarones metrics --reference ground_truth.txt --hypothesis ocr_output.txt

# Import 10 pages from a Gallica IIIF manifest
picarones import iiif https://gallica.bnf.fr/ark:/12148/xxx/manifest.json --pages 1-10

# Import a HuggingFace dataset
picarones import hf medieval-ocr/dataset-name

# View benchmark history with regression detection
picarones history --engine tesseract --regression

# Robustness demo (noise, blur, rotation, resolution)
picarones robustness --corpus ./gt/ --engine tesseract --demo

# Fail CI if CER exceeds threshold
picarones run --corpus ./corpus/ --engines tesseract --fail-if-cer-above 0.15
```

### Web Interface

```bash
picarones serve --host 0.0.0.0 --port 8080
```

**API endpoints include:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main single-page application |
| `/api/status` | GET | Version and application status |
| `/api/engines` | GET | Available OCR/LLM engines |
| `/api/normalization/profiles` | GET | Normalization profiles (read dynamically) |
| `/api/benchmark/start` | POST | Start a benchmark job (returns `job_id`) |
| `/api/benchmark/{job_id}/stream` | GET | SSE real-time progress stream |
| `/api/benchmark/{job_id}/cancel` | POST | Cancel a running benchmark |
| `/api/corpus/browse` | GET | Browse server-side corpus folders |
| `/api/htr-united/catalogue` | GET | Browse HTR-United catalogue |
| `/api/huggingface/search` | GET | Search HuggingFace datasets |
| `/reports/{filename}` | GET | Download generated HTML reports |

### Pipeline Modes

Picarones supports three modes for OCR+LLM pipelines:

| Mode | Description | Model type |
|------|-------------|------------|
| `zero_shot` | LLM receives the image directly and transcribes without prior OCR | VLM (vision) |
| `post_correction_texte` | OCR produces raw text, then LLM corrects it | Text-only LLM |
| `post_correction_image_texte` | OCR produces raw text, then LLM receives both image and text for correction | VLM (vision) |

**Example:** `ministral-3b-latest` is a text-only model and should use `post_correction_texte`.
GPT-4o and Claude support all three modes.

---

## Supported Engines

| Engine | Type | Execution Mode | Installation |
|--------|------|---------------|-------------|
| **Tesseract 5** | Local CLI | CPU (ProcessPool) | `pip install pytesseract` + system binary |
| **Pero OCR** | Local Python | CPU (ProcessPool) | `pip install pero-ocr` |
| **Kraken** | Local Python | CPU (ProcessPool) | `pip install kraken` |
| **Mistral OCR** | Cloud API | IO (ThreadPool) | `MISTRAL_API_KEY` env var |
| **Google Vision** | Cloud API | IO (ThreadPool) | `GOOGLE_APPLICATION_CREDENTIALS` env var |
| **Azure Doc Intelligence** | Cloud API | IO (ThreadPool) | `AZURE_DOC_INTEL_ENDPOINT` + `AZURE_DOC_INTEL_KEY` |
| **GPT-4o** (VLM) | LLM API | IO (ThreadPool) | `OPENAI_API_KEY` env var |
| **Claude Sonnet** (VLM) | LLM API | IO (ThreadPool) | `ANTHROPIC_API_KEY` env var |
| **Mistral Large** (LLM) | LLM API | IO (ThreadPool) | `MISTRAL_API_KEY` env var |
| **Ollama** (local LLM) | Local LLM | IO (ThreadPool) | `ollama serve` running locally |
| **Custom engine** | CLI or API | Configurable | YAML declaration, no code required |

Engines declare their `execution_mode` (`"io"` or `"cpu"`), allowing the runner to use
`ThreadPoolExecutor` for IO-bound engines and `ProcessPoolExecutor` for CPU-bound engines
simultaneously.

---

## Normalization Profiles

Picarones includes eight built-in diplomatic normalization profiles designed for historical
text comparison. These reduce noise from expected orthographic variation so metrics reflect
genuine OCR errors, not historical spelling differences.

| Profile | Period | Key equivalences |
|---------|--------|-----------------|
| `nfc` | Any | Unicode NFC normalization only |
| `minimal` | Any | NFC + long s (ſ -> s) |
| `medieval_french` | 12th-15th c. | ſ=s, u=v, i=j, y=i, ae=ae, oe=oe, p-bar=per, etc. |
| `early_modern_french` | 16th-18th c. | ſ=s, ae=ae, oe=oe, y-tilde=yn, &=et |
| `medieval_latin` | 12th-15th c. | ſ=s, u=v, i=j, ae=ae, oe=oe, p-bar=per, q-bar=que |
| `medieval_english` | 12th-15th c. | ſ=s, u=v, i=j, thorn=th, eth=th, yogh=y, p-bar=per |
| `early_modern_english` | 16th-18th c. | ſ=s, u=v, i=j, vv=w, thorn=th, eth=th, yogh=y |
| `secretary_hand` | 16th-17th c. | Early modern English + secretary hand visual confusions |

Custom profiles can be loaded from YAML files with user-defined diplomatic tables.

---

## Error Taxonomy

Every character-level error is automatically classified into one of 10 categories:

| Class | Name | Description |
|-------|------|-------------|
| 1 | `visual_confusion` | Morphologically similar characters (rn/m, l/1, O/0, u/n) |
| 2 | `diacritic_error` | Missing, incorrect, or spurious diacritical mark |
| 3 | `case_error` | Case difference only (A/a) |
| 4 | `ligature_error` | Ligature not resolved or incorrectly resolved |
| 5 | `abbreviation_error` | Medieval abbreviation not expanded |
| 6 | `hapax` | Word not found in any reference lexicon |
| 7 | `segmentation_error` | Token fusion or fragmentation (words/lines) |
| 8 | `oov_character` | Character outside the engine's vocabulary |
| 9 | `lacuna` | Text present in ground truth but absent from OCR output |
| 10 | `over_normalization` | LLM silently modernized a historical spelling |

---

## Project Structure

```
picarones/
├── __init__.py                 # Version (1.0.0), package metadata
├── cli.py                      # Click CLI: run, demo, report, metrics, engines, info,
│                               #   serve, import, history, robustness
├── fixtures.py                 # Realistic synthetic test data (medieval documents)
│
├── core/
│   ├── corpus.py               # Corpus loading (folder, ALTO XML, PAGE XML)
│   ├── metrics.py              # CER, WER, MER, WIL (via jiwer)
│   ├── normalization.py        # Unicode normalization, 8 diplomatic profiles
│   ├── statistics.py           # Bootstrap CI 95%, Wilcoxon test, correlations
│   ├── runner.py               # Benchmark orchestrator (ThreadPool + ProcessPool)
│   ├── results.py              # DocumentResult, BenchmarkResults, JSON export
│   ├── confusion.py            # Unicode confusion matrix
│   ├── char_scores.py          # Ligature and diacritic scores
│   ├── taxonomy.py             # 10-class error taxonomy
│   ├── structure.py            # Structural analysis (blocks, lines, words)
│   ├── image_quality.py        # Image quality metrics (contrast, noise, resolution)
│   ├── difficulty.py           # Intrinsic difficulty score per document
│   ├── hallucination.py        # VLM hallucination detection
│   ├── line_metrics.py         # Line-level error distribution (Gini, percentiles)
│   ├── history.py              # SQLite longitudinal tracking
│   └── robustness.py           # Robustness analysis (noise, blur, rotation, resolution)
│
├── engines/
│   ├── base.py                 # BaseOCREngine (execution_mode: "io" | "cpu")
│   ├── tesseract.py            # Tesseract 5 adapter (CPU)
│   ├── pero_ocr.py             # Pero OCR adapter (CPU)
│   ├── mistral_ocr.py          # Mistral OCR API (/v1/ocr endpoint)
│   ├── google_vision.py        # Google Cloud Vision adapter
│   └── azure_doc_intel.py      # Azure Document Intelligence adapter
│
├── llm/
│   ├── base.py                 # BaseLLMAdapter interface
│   ├── openai_adapter.py       # OpenAI / GPT-4o adapter
│   ├── anthropic_adapter.py    # Anthropic / Claude adapter
│   ├── mistral_adapter.py      # Mistral chat completions adapter
│   └── ollama_adapter.py       # Ollama local LLM adapter
│
├── pipelines/
│   ├── base.py                 # OCRLLMPipeline orchestrator
│   └── over_normalization.py   # Over-normalization detection
│
├── prompts/                    # Versioned prompt templates (FR + EN)
│   ├── correction_medieval_french.txt
│   ├── zero_shot_medieval_french.txt
│   ├── correction_imprime_ancien.txt
│   ├── zero_shot_imprime_ancien.txt
│   ├── correction_medieval_english.txt
│   ├── zero_shot_medieval_english.txt
│   ├── correction_early_modern_english.txt
│   └── correction_image_medieval_french.txt
│
├── report/
│   ├── generator.py            # Self-contained HTML report (Chart.js + diff2html)
│   └── diff_utils.py           # Diff computation utilities
│
├── web/
│   ├── app.py                  # FastAPI app (SSE, ZIP upload, dynamic endpoints)
│   └── static/                 # CSS assets
│
└── importers/
    ├── iiif.py                 # IIIF manifest importer
    ├── gallica.py              # Gallica API client (BnF)
    ├── htr_united.py           # HTR-United catalogue importer
    ├── huggingface.py          # HuggingFace Datasets importer
    └── escriptorium.py         # eScriptorium client

tests/                          # ~1020 unit and integration tests
.github/workflows/
├── ci.yml                      # CI: Python 3.11/3.12, Linux/macOS/Windows
└── sync_to_huggingface.yml     # Auto-sync to HuggingFace Space on push to main
Dockerfile                      # Multi-stage Docker build for HuggingFace Spaces
```

---

## Environment Variables

Configure API keys depending on which engines and LLM adapters you use:

```bash
# LLM APIs
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export MISTRAL_API_KEY="..."

# Cloud OCR APIs (optional)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="eu-west-1"
export AZURE_DOC_INTEL_ENDPOINT="https://..."
export AZURE_DOC_INTEL_KEY="..."
```

For deployment on HuggingFace Spaces, set these in **Settings > Variables and secrets**.

---

## CI/CD

### GitHub Actions (`ci.yml`)

- **Triggers:** push to `main`/`develop`/`feature/*`/`sprint/*`/`claude/*`, PRs to
  `main`/`develop`, manual dispatch
- **Matrix:** Python 3.11 + 3.12 on Linux, macOS, and Windows
- **Jobs:**
  1. **Tests** -- full pytest suite with coverage, uploaded to Codecov
  2. **Demo** -- end-to-end demo report generation with history and robustness
  3. **Build** -- wheel and sdist with twine validation
  4. **Lint** -- ruff check for errors (E, W, F; ignores E501, E402)

### HuggingFace Sync (`sync_to_huggingface.yml`)

- Automatically pushes `main` to the HuggingFace Space `Ma-Ri-Ba-Ku/Picarones`
- Requires the `HF_TOKEN` secret in GitHub repository settings

---

## Development

```bash
# Install with dev + web dependencies
pip install -e ".[dev,web]"

# Run the test suite
pytest tests/ -q --tb=short

# Run with coverage
pytest tests/ --cov=picarones --cov-report=term-missing

# Generate a demo report
picarones demo --output demo_report.html

# Launch the web UI in development mode
picarones serve --port 8080

# Full refresh (useful in Codespaces)
git pull && pip install -e ".[dev,web]" && picarones demo --output demo.html
```

**Key development conventions:**

- Never use bare `except Exception: pass` -- always log with `logger.warning()`
- Normalization profiles are read dynamically from `picarones/core/normalization.py` --
  never hardcode them in endpoint handlers
- Engines declare their `execution_mode` (`"io"` or `"cpu"`) so the runner can select the
  appropriate executor
- `python-multipart` must remain in dependencies (FastAPI checks at import time)

---

## Roadmap

| Sprint | Status | Deliverables |
|--------|--------|-------------|
| 1 | Done | Project structure, Tesseract, Pero OCR, CER/WER, CLI |
| 2 | Done | HTML report v1: Chart.js, colored diff, gallery |
| 3 | Done | OCR+LLM pipelines, GPT-4o, Claude, Mistral, Ollama |
| 4 | Done | Cloud OCR APIs, IIIF import, diplomatic normalization |
| 5 | Done | Advanced metrics: confusion matrix, ligatures, 9-class taxonomy |
| 6 | Done | FastAPI web interface, HTR-United, HuggingFace, bilingual UI |
| 7 | Done | HTML report v2: Wilcoxon, bootstrap, clustering, difficulty score |
| 8 | Done | eScriptorium, Gallica API, SQLite history, robustness analysis |
| 9 | Done | Documentation, packaging, Docker, CI/CD, PyInstaller, v1.0.0-Beta |
| 10 | Done | Line error distribution (Gini), VLM hallucination detection |
| 11 | Done | Internationalization FR/EN, English normalization profiles |
| 12 | Done | Browser ZIP upload, macOS file filtering, dynamic model selector |
| 13 | Done | pyproject.toml cleanup, runner parallelization, NDJSON streaming, Wilcoxon validation |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on adding an OCR engine, an LLM
adapter, or submitting a pull request.

---

## License

[Apache License 2.0](LICENSE)

Copyright 2024 Picarones contributors.
