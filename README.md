---
title: Picarones
emoji: 📜
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Picarones

> **Heritage OCR / HTR / VLM and post-correction benchmarking — bring your golden dataset, plug in the AIs.**

> **Banc d'essai d'OCR / HTR / VLM et de post-correction pour documents patrimoniaux — amenez votre golden dataset, branchez vos IA.**

[![CI](https://github.com/maribakulj/Picarones/actions/workflows/ci.yml/badge.svg)](https://github.com/maribakulj/Picarones/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace%20Space-yellow.svg)](https://huggingface.co/spaces/Ma-Ri-Ba-Ku/Picarones)

---

**Picarones** is an open-source benchmarking platform for OCR, HTR, VLM
and post-correction pipelines on heritage documents.

The input is a folder of `(image, ground truth)` pairs — ground truth in
plain text (`.gt.txt`), ALTO XML, or PAGE XML, hand-annotated or sourced
from a reference corpus. Picarones runs the AIs you plug in (OCR engines,
VLMs, OCR+LLM pipelines, ALTO mappers, ensembles…) on every page,
compares each output to the ground truth at every relevant level (text,
ALTO, PAGE, entities, reading order), and produces a self-contained HTML
report with factual numbers, statistical tests and a reproducibility
snapshot. Without ground truth, no benchmark — Picarones measures how
well an AI matches a known reference, not how well it transcribes an
arbitrary document.

Typical workflow: assemble a small golden dataset whose profile (script
type, period, language) matches the production corpus you intend to
process; benchmark candidate AIs on it; read the report; decide which AI
to deploy. Picarones does not yet ship a curated library of standard
datasets — the importers (IIIF, Gallica, HuggingFace, HTR-United,
eScriptorium, ZIP) help fetch existing data, curation remains yours.

Heritage-specific metrics (diplomatic CER, ligature and diacritic scores,
medieval abbreviations, Roman numerals, foliation, fuzzy searchability,
philological marker fidelity), composable pipelines, factual narrative
synthesis at the top of the report, multi-engine Friedman/Nemenyi tests
with critical difference diagram, cost / speed / CO₂ Pareto analysis,
per-junction error absorption, multi-run stability, controlled per-slot
comparison.

> *Version française ci-dessous.*

---

## Use case

An archive, a digital library or a heritage service plans to OCR a
production corpus — say, several thousand 17th-century parish registers,
19th-century newspapers, or medieval glossed manuscripts. Several
candidate pipelines are on the table (alternative OCR, LLM correction,
ALTO mappers, ensembles); the question is which one to deploy.

The candidates cannot be benchmarked on the production corpus itself
(no ground truth). A small golden dataset matching the target profile is
assembled; Picarones runs each candidate on it and reports CER gain,
recovered fuzzy searchability, preserved numerical sequences, errors
introduced by post-correctors and statistical significance. The numbers
inform the deployment decision.

---

## En français

**Picarones** est une plateforme open source de banc d'essai pour des IA
d'OCR, HTR, VLM et des pipelines de post-correction sur documents
patrimoniaux.

L'entrée est un dossier de paires `(image, vérité terrain)` — VT en
texte brut (`.gt.txt`), ALTO XML, ou PAGE XML, annotée à la main ou
issue d'un corpus de référence. Picarones exécute les IA que vous
branchez (moteurs OCR, VLM, pipelines OCR+LLM, mappeurs ALTO,
ensembles…) sur chaque page, compare la sortie à la VT à tous les
niveaux pertinents (texte, ALTO, PAGE, entités, ordre de lecture) et
produit un rapport HTML autonome avec chiffres factuels, tests
statistiques et snapshot de reproductibilité. Sans vérité terrain, pas
de benchmark — Picarones mesure à quel point une IA matche une référence
connue, pas à quel point elle transcrit un document quelconque.

Workflow type : constituer un golden dataset dont le profil (type
d'écriture, période, langue) correspond au corpus de production à
traiter ; benchmarker les IA candidates dessus ; lire le rapport ;
décider quelle IA déployer. Picarones ne fournit pas encore de
bibliothèque curatée de datasets standards — les importers (IIIF,
Gallica, HuggingFace, HTR-United, eScriptorium, ZIP) aident à récupérer
des données existantes, la curation reste à votre charge.

Métriques spécifiques aux corpus patrimoniaux (CER diplomatique, scores
de ligatures et diacritiques, abréviations médiévales, numéraux romains,
foliotation, recherchabilité fuzzy, fidélité aux marqueurs
philologiques), pipelines composables, synthèse narrative factuelle au
sommet du rapport, tests Friedman/Nemenyi multi-moteurs avec diagramme
de différence critique, analyse Pareto coût/vitesse/CO₂, absorption
d'erreur par jonction, stabilité multi-runs, comparaison contrôlée par
slot.

### Cas d'usage type

Une archive, une bibliothèque numérique ou un service patrimonial
prévoit d'OCRiser un corpus de production — par exemple plusieurs
milliers de registres paroissiaux du XVIIᵉ, de presse du XIXᵉ ou de
manuscrits glosés médiévaux. Plusieurs pipelines candidats sont sur la
table (OCR alternatif, correction LLM, mappeurs ALTO, ensembles) ;
reste à décider lequel déployer.

Les candidats ne peuvent pas être benchmarkés sur le corpus de
production lui-même (pas de VT). On constitue un golden dataset
matching le profil cible ; Picarones exécute chaque candidat dessus et
remonte le gain CER, la recherchabilité fuzzy gagnée, les séquences
numériques préservées, les erreurs introduites par les post-correcteurs
et la significativité statistique. Les chiffres nourrissent la décision
de déploiement.

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
- **Bootstrap 95% confidence intervals**, **Wilcoxon signed-rank tests**, and the
  **Friedman test + Nemenyi post-hoc** with a **Critical Difference Diagram** (Demšar 2006)
  for rigorous multi-engine comparison
- **Intrinsic difficulty score** per document, independent of engine performance
- **Line-level error distribution** with Gini coefficient and percentile analysis
- **VLM hallucination detection** -- anchor score and length ratio to flag fabricated output
- **Cost / speed / carbon Pareto front** (local vs cloud, per-token pricing model)

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
| IIIF manifests (institutional repositories) | `picarones import iiif <manifest-url>` |
| Gallica API (SRU + OCR) | `GallicaClient` / `picarones import iiif` |
| HuggingFace Datasets | `picarones import hf <dataset-id>` |
| HTR-United catalogue | `picarones import htr-united` |
| eScriptorium | `EScriptoriumClient` |
| ZIP upload (browser) | Web interface upload endpoint |

Supported corpus formats: plain text pairs (image + ground truth), **ALTO XML**, and **PAGE XML**.

### Interactive HTML Report

- **Self-contained HTML file** -- works offline, no server needed (Jinja2-templated since Sprint 17)
- **Factual narrative synthesis** at the top of the report (Sprint 19): 12 deterministic
  detectors extract salient facts (global leader, significant gap, stratum collapse, VLM
  hallucination flag, speed winner, cost outlier, Pareto alternative, ...) and render them
  as short sentences -- every number is traceable to the source payload, no LLM, no
  hallucination risk
- **Critical Difference Diagram** (CDD) rendered server-side as static SVG -- no JS required
- **Cost / speed / carbon Pareto chart** with toggleable axes and highlighted Pareto front
- **Contextual glossary**: a `?` icon next to every metric header opens a side panel with
  definition, what it measures, usage, limits, and reference (25 bilingual entries)
- **Advanced mode panel**: visible-column picker, per-stratum filter, and opt-in personal
  composite score (sliders default to 0, formula always visible, explicit warning that no
  universal weighting exists). State is persisted in the URL.
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
| `dev` | `pip install -e ".[dev]"` | pytest, pytest-cov, httpx, FastAPI, uvicorn, python-multipart |
| `web` | `pip install -e ".[web]"` | FastAPI, uvicorn, python-multipart, httpx |
| `stats` | `pip install -e ".[stats]"` | scipy (exact Wilcoxon/Friedman/Nemenyi -- otherwise pure-Python fallback) |
| `llm` | `pip install -e ".[llm]"` | OpenAI, Anthropic, Mistral SDKs |
| `hf` | `pip install -e ".[hf]"` | HuggingFace Datasets |
| `pero` | `pip install -e ".[pero]"` | Pero OCR engine |
| `kraken` | `pip install -e ".[kraken]"` | Kraken engine |
| `ocr-cloud` | `pip install -e ".[ocr-cloud]"` | Google Vision, AWS (boto3), Azure Doc Intelligence |
| `all` | `pip install -e ".[all]"` | `web` + `hf` + `llm` + `dev` (no `ocr-cloud`) |

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
| `picarones import iiif` | Import corpus from an IIIF manifest (any institutional repository). HTR-United and HuggingFace imports are exposed through the web interface (`/api/htr-united/import`, `/api/huggingface/import`). |

**Examples:**

```bash
# Benchmark with Tesseract, French language, PSM 6
picarones run --corpus ./manuscripts/ --engines tesseract --lang fra --psm 6 \
  --output results.json --verbose

# Compare two text files
picarones metrics --reference ground_truth.txt --hypothesis ocr_output.txt

# Import 10 pages from any IIIF manifest URL
picarones import iiif https://institution.example/iiif/xxx/manifest.json --pages 1-10

# HuggingFace and HTR-United imports are available via the web UI at
#   http://localhost:8000/  (endpoints POST /api/huggingface/import and /api/htr-united/import)

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

Picarones ships **11 built-in normalization profiles** designed for historical text comparison.
These reduce noise from expected orthographic variation so metrics reflect genuine OCR errors,
not historical spelling differences. The canonical list is defined in
[`picarones/core/normalization.py`](picarones/core/normalization.py) (`NORMALIZATION_PROFILES`)
and is exposed dynamically via `/api/normalization/profiles`.

| Profile | Period | Key equivalences |
|---------|--------|-----------------|
| `nfc` | Any | Unicode NFC normalization only |
| `caseless` | Any | NFC + case folding (`casefold`) |
| `minimal` | Any | NFC + long s (ſ -> s) |
| `medieval_french` | 12th-15th c. | ſ=s, u=v, i=j, y=i, æ=ae, œ=oe, ꝑ=per, & = et |
| `early_modern_french` | 16th-18th c. | ſ=s, æ=ae, œ=oe |
| `medieval_latin` | 12th-15th c. | ſ=s, u=v, i=j, ꝑ=per, ꝓ=pro |
| `medieval_english` | 12th-15th c. | ſ=s, u=v, i=j, þ=th, ȝ=y, ꝑ=per, ꝓ=pro |
| `early_modern_english` | 16th-18th c. | ſ=s, u=v, i=j, vv=w, þ=th, ð=th, ȝ=y |
| `secretary_hand` | 16th-17th c. | Early Modern English + secretary hand visual confusions |
| `sans_ponctuation` | Any | NFC + strips `. , ; : ! ? ' " - – — ( ) [ ]` |
| `sans_apostrophes` | Any | NFC + strips straight (`'`) and typographic (`’`) apostrophes |

Custom profiles can be loaded from YAML files with user-defined diplomatic tables and/or
`exclude_chars` sets.

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
├── __main__.py                 # `python -m picarones`
├── cli.py                      # Click CLI: run, demo, report, metrics, engines, info,
│                               #   serve, import iiif, history, robustness
├── fixtures.py                 # Realistic synthetic test data (medieval documents)
├── i18n.py                     # Back-compat shim loading report/i18n/{fr,en}.json
│
├── core/
│   ├── corpus.py               # Corpus loading (folder, ALTO XML, PAGE XML)
│   ├── metrics.py              # CER, WER, MER, WIL (via jiwer)
│   ├── normalization.py        # Unicode normalization, 11 diplomatic/exclusion profiles
│   ├── statistics.py           # Bootstrap CI, Wilcoxon, Friedman, Nemenyi, CDD SVG
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
│   ├── robustness.py           # Robustness analysis (noise, blur, rotation, resolution)
│   ├── pricing.py              # Cost model, EngineCost, Pareto front
│   └── narrative/              # Factual narrative engine (Sprint 16-19)
│       ├── facts.py            # Fact model, 12 FactType, DetectorRegistry
│       ├── detectors.py        # 12 detectors (global_leader_cer, significant_gap,
│       │                       #   stratum_winner/collapse, error_profile_outlier,
│       │                       #   llm_hallucination_flag, robustness_fragile,
│       │                       #   speed_winner, confidence_warning,
│       │                       #   statistical_tie, pareto_alternative, cost_outlier)
│       ├── arbiter.py          # Sort by importance, dedup, anti-contradiction
│       ├── renderer.py         # YAML template rendering via str.format_map
│       └── templates/{fr,en}.yaml
│
├── data/
│   └── pricing.yaml            # Indicative cost table (OCR local/cloud + LLM)
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
├── prompts/                    # 8 versioned prompt templates
│   ├── correction_medieval_french.txt
│   ├── correction_image_medieval_french.txt
│   ├── correction_imprime_ancien.txt
│   ├── correction_medieval_english.txt
│   ├── correction_early_modern_english.txt
│   ├── zero_shot_medieval_french.txt
│   ├── zero_shot_imprime_ancien.txt
│   └── zero_shot_medieval_english.txt
│
├── report/
│   ├── generator.py            # Orchestrates Jinja2 rendering (617 lines since Sprint 17)
│   ├── diff_utils.py           # Diff computation utilities
│   ├── templates/              # Jinja2 partials (Sprint 17)
│   │   ├── base.html.j2        # assembles everything via {% include %}
│   │   ├── _header.html, _footer.html, _styles.css, _app.js
│   │   ├── _critical_difference.html, _narrative_summary.html, _side_panels.html
│   │   └── view_ranking.html, view_gallery.html, view_document.html,
│   │       view_analyses.html, view_characters.html
│   ├── i18n/                   # FR/EN translations (Sprint 17 -- extracted from i18n.py)
│   │   ├── fr.json
│   │   └── en.json
│   ├── glossary/               # Contextual glossary (Sprint 21)
│   │   ├── fr.yaml             # 25 bilingual entries (definition, measures, usage,
│   │   └── en.yaml             #   limits, reference)
│   └── vendor/                 # Vendored Chart.js
│
├── web/
│   ├── app.py                  # FastAPI app (SSE, ZIP upload, dynamic endpoints)
│   └── static/                 # CSS assets
│
└── importers/
    ├── iiif.py                 # IIIF manifest importer
    ├── gallica.py              # Gallica API client (institutional digital library)
    ├── htr_united.py           # HTR-United catalogue importer
    ├── huggingface.py          # HuggingFace Datasets importer
    └── escriptorium.py         # eScriptorium client

docs/                           # User + developer documentation (Sprint 22)
├── case-studies/               # Two labelled case studies ("Cas d'école")
│   ├── 01-registres-paroissiaux.md
│   └── 02-edition-critique.md
├── user/
│   └── reading-a-report.md     # Anatomy, suggested reading order, advanced panel
└── developer/
    ├── index.md
    ├── narrative-engine.md
    ├── extending-glossary.md
    └── extending-i18n.md

tests/                          # 1242 tests (1 skipped: scipy optional)
.github/workflows/
├── ci.yml                      # CI: Python 3.11/3.12, Linux/macOS/Windows, ruff lint
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
  1. **Tests** -- full pytest suite (1242 passing, 1 skipped when scipy is absent) with
     coverage uploaded to Codecov
  2. **Demo** -- end-to-end demo report generation with history and robustness
  3. **Build** -- wheel and sdist with twine validation
  4. **Lint** -- `ruff check picarones/ tests/` (E, W, F; ignores E501, E402). The ruff
     config lives in `pyproject.toml` under `[tool.ruff]` so CI, `make lint` and direct
     invocation all produce the same result -- blocking on F401 / E741.

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

**Test suite:** `pytest tests/` -> **1242 passed, 1 skipped** (the skip is intentional
when the optional `scipy` extra is not installed).

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
| 14 | Done | Robust engine filtering, corpus validation |
| 15 | Done | Fix empty OCR+LLM pipeline output (Mistral ContentChunk normalization, `finish_reason` logging) |
| 16 | Done | `line_metrics` + `hallucination` wired into runner/`EngineReport`; narrative engine foundations (`core/narrative/` with `Fact` / `DetectorRegistry`); Pillow `getdata`->`tobytes`, silent excepts -> explicit warnings |
| 17 | Done | Report refactor: `generator.py` 3690 -> 617 lines via Jinja2; monolithic HTML template split into 10 files under `picarones/report/templates/`; i18n migrated to `report/i18n/{fr,en}.json`; +16 non-regression tests |
| 18 | Done | Friedman test + Nemenyi post-hoc + Critical Difference Diagram (Demšar 2006); `detect_statistical_tie` enabled; SVG rendered server-side; +41 tests |
| 19 | Done | Factual narrative engine complete: 9 new detectors, arbiter (importance + anti-contradiction), YAML templates renderer, `_narrative_summary.html` partial, anti-hallucination traceability test; +32 tests |
| 20 | Done | Cost model + Pareto view: `core/pricing.py` + `data/pricing.yaml`, `compute_pareto_front`, Chart.js Pareto chart with cost/speed/carbon toggles, `pareto_alternative` and `cost_outlier` detectors; +28 tests |
| 21 | Done | Contextual glossary (25 bilingual entries) + advanced-mode side panel (visible columns, strata filters, opt-in composite score, URL state persistence); +21 tests |
| 22 | Done | Case studies (`docs/case-studies/`), user guide (`docs/user/reading-a-report.md`), three developer guides (`docs/developer/`); +18 tests |

---

## Known Issues & Improvement Opportunities

This section captures the findings of the Sprint 22 audit. None of them block the current
release (all 1242 tests pass, lint clean), but each represents a sensible next step.

### Architecture / refactor

- **`picarones/web/app.py` is 3072 lines** (FastAPI routes, corpus upload, SSE, ZIP flattening,
  HTML delivery, model registry all in one module). Candidate split: `app_routes.py` /
  `app_corpus.py` / `app_jobs.py` / `app_models.py`.
- **`picarones/core/statistics.py` is 1127 lines** mixing bootstrap CI, Wilcoxon, Friedman,
  Nemenyi table, Pareto front and CDD SVG. Splitting into `statistics/bootstrap.py`,
  `statistics/tests.py`, `statistics/pareto.py`, `statistics/cdd_svg.py` would shorten
  import graphs and ease review.
- **`picarones/cli.py` is 971 lines** — each Click command could live in its own module under
  `picarones/cli/` and be registered via `cli.add_command(...)`.
- **`picarones/core/runner.py` is 847 lines** — orchestrator is reasonable but edges past the
  500-line guideline; extracting the per-document worker + the partial-NDJSON writer would
  reduce mental load.
- **`picarones/core/narrative/detectors.py` is 680 lines** — all 12 detectors live together;
  one file per `FactType` (or per importance tier) would make additions safer.

### Back-compat shim

- **`picarones/i18n.py`** is a 66-line shim that reads `picarones/report/i18n/{fr,en}.json`.
  Since Sprint 17 the JSON files are the source of truth and only
  `picarones/report/generator.py:654` still imports through the shim. Either promote the
  shim to `picarones.report.i18n` (renaming the import) or delete the file and import the
  loader directly.

### Explicit engine declarations

- `MistralOCREngine`, `GoogleVisionEngine` and `AzureDocIntelEngine` inherit the implicit
  `execution_mode = "io"` default from `BaseOCREngine`. For clarity and to protect against a
  future default flip, declare it explicitly (as `TesseractEngine` and `PeroOCREngine` already
  do for `"cpu"`).

### Test coverage gaps

- No dedicated unit tests for `picarones/core/char_scores.py` (exercised only transitively).
- No unit tests for the cloud engine adapters themselves (`mistral_ocr.py`,
  `google_vision.py`, `azure_doc_intel.py`) — they are only reached via integration fixtures.
- `pytest` installed as a `uv` tool doesn't see project dependencies automatically; document
  `pip install -e ".[dev,web,stats]"` in the pytest environment or switch to an in-repo venv
  to avoid "`ModuleNotFoundError: No module named 'yaml'`" surprises.

### Documentation

- `CHANGELOG.md` stops at Sprint 9 (2025-03). Sprints 10-22 are described in `CLAUDE.md` and
  this README but should be back-ported into `CHANGELOG.md` to follow Keep-a-Changelog.
- `SPECS.md` predates the narrative engine, Pareto view and glossary — worth a pass.
- Some code comments and docstrings are still in French while user-facing strings are
  bilingual; harmonising module docstrings in English would make the project more
  contributor-friendly.

### CI / packaging

- `sync_to_huggingface.yml` uses `git push --force hf main` unconditionally — safe today but
  worth documenting because a non-main branch push would silently rewrite the Space.
- `picarones.spec` (PyInstaller) is still present but not exercised in CI — either add a
  `build-exe` job or mark the spec as community-maintained.

### Security (nothing critical)

- ZIP upload flattening in `web/app.py` rejects absolute paths and `..` traversal but does
  not check for symlinks inside archives. Python's `zipfile` doesn't extract symlinks, so
  the risk is theoretical; adding an explicit check (`ZipInfo.external_attr & 0xA000`) is a
  belt-and-braces improvement.
- API keys are read from environment variables only (no hardcoded fallback) — good.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on adding an OCR engine, an LLM
adapter, or submitting a pull request.

---

## License

[Apache License 2.0](LICENSE)

Copyright 2024 Picarones contributors.
