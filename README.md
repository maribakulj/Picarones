---
title: Picarones
emoji: 📜
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Picarones

> **Heritage OCR / HTR / VLM and post-correction benchmarking tool**
>
> **Outil de comparaison d'OCR / HTR / VLM et de post-correction pour documents patrimoniaux**

**Status (May 2026)** — version 1.x, scientific prototype under
consolidation.  The core (corpus, runner, metrics, HTML report) is
usable to compare transcription pipelines on a ground-truth corpus.
A targeted rewrite (see
[`docs/roadmap/rewrite-2026.md`](docs/roadmap/rewrite-2026.md))
rebuilds the orchestration layer and evaluation views for a stable
2.0 release by the end of 2026.

[![CI](https://github.com/maribakulj/Picarones/actions/workflows/ci.yml/badge.svg)](https://github.com/maribakulj/Picarones/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/maribakulj/Picarones/graph/badge.svg)](https://codecov.io/gh/maribakulj/Picarones)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-46aef7.svg)](https://github.com/astral-sh/ruff)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace%20Space-yellow.svg)](https://huggingface.co/spaces/Ma-Ri-Ba-Ku/Picarones)

---

## What is Picarones?

**Picarones** is an open-source comparison tool for OCR, HTR, VLM and
post-correction pipelines on **heritage documents** (manuscripts,
early printed books, archives).

The input is a folder of `(image, ground truth)` pairs — ground truth
in plain text, ALTO XML, or PAGE XML. Picarones runs the AIs you plug
in (OCR engines, VLMs, OCR+LLM pipelines, ALTO mappers, ensembles…) on
every page, compares each output to the ground truth, and produces an
HTML report with the numerical results.

**Without ground truth, no benchmark** — Picarones measures how well
an AI matches a known reference, not how it transcribes an arbitrary
document.

> **Limits to keep in mind.** Picarones is a tool, not a verdict
> machine. CER/WER and the philological metrics measure agreement with
> a single reference; the choice of reference, normalization profile
> and metric is an editorial decision the user must own.

> *Version française ci-dessous.*

### Use case

A digital library plans to OCR a production corpus — say, several
thousand 17th-century parish registers, 19th-century newspapers, or
medieval glossed manuscripts. Several pipelines are on the table
(alternative OCR, LLM correction, ALTO mappers, ensembles); the
question is which one to deploy.

The candidates cannot be benchmarked on the production corpus itself
(no ground truth). A small **golden dataset** matching the target
profile is assembled; Picarones runs each candidate on it and reports
CER, recovered fuzzy searchability, preserved numerical sequences,
errors introduced by post-correctors, and statistical significance.
The numbers inform the deployment decision.

### En français

**Picarones** est une plateforme open-source de banc d'essai pour des
IA d'OCR, HTR, VLM et des pipelines de post-correction sur documents
patrimoniaux.

L'entrée est un dossier de paires `(image, vérité terrain)` — VT en
texte brut, ALTO XML ou PAGE XML. Picarones exécute les IA que vous
branchez sur chaque page, compare la sortie à la VT à tous les
niveaux pertinents et produit un rapport HTML autonome avec chiffres
factuels, tests statistiques et snapshot de reproductibilité. Sans
vérité terrain, pas de benchmark.

---

## Features

### Heritage-specific metrics

Three families of metrics calibrated for historical documents:

- **Classical OCR/HTR** — CER (raw, NFC, caseless, **diplomatic**),
  WER, MER, WIL via [jiwer](https://github.com/jitsi/jiwer); 10-class
  error taxonomy; bootstrap 95% CIs; line-level Gini distribution.
- **Philological** — MUFI coverage, abbreviation expansion (Capelli),
  early-modern typography (long-s, ligatures, tilde nasals), modern
  archives markers, Roman numerals, Unicode block accuracy, NER
  precision (HIPE), reading-order F1 (ICDAR 2015), layout F1.
- **Comparison & decision** — Friedman + Nemenyi + **Critical
  Difference Diagram** (Demšar 2006); cross-engine taxonomic
  divergence + oracle complementarity; cost / speed / CO₂ Pareto
  front; multi-run stability (Cohen κ, Krippendorff α); longitudinal
  trend with change-point detection; controlled per-slot ANOVA-like
  comparison.

For the full list with definitions, see [`docs/reference/views.md`](docs/reference/views.md)
and the contextual glossary embedded in every report (25 bilingual
entries).

### OCR+LLM pipelines

Composable chains: `tesseract -> gpt-4o`, `pero_ocr -> claude-sonnet`,
zero-shot VLM, etc. Three pipeline modes: text-only post-correction,
image+text post-correction, and zero-shot. **Over-normalisation
detection** flags LLMs that silently modernise historical spellings.
A composed-pipeline benchmarking layer (Sprint 63+) runs N candidate
pipelines on the same corpus and ranks them by a chosen metric.

### Corpus import

| Source | Method |
|--------|--------|
| Local folder | `picarones run --corpus ./corpus/` |
| IIIF manifests (any institutional repository) | `picarones import iiif <manifest-url>` |
| Gallica API (BnF SRU + IIIF) | `GallicaClient` / `picarones import iiif` |
| HuggingFace Datasets | Web UI: `POST /api/huggingface/import` |
| HTR-United catalogue | Web UI: `POST /api/htr-united/import` |
| eScriptorium | `EScriptoriumClient` |
| ZIP upload (browser) | Web upload endpoint |

Supported corpus formats: plain text pairs, ALTO XML, PAGE XML.

### Interactive HTML report

A single self-contained HTML file (or with `--lazy-images` for large
corpora). Five views:

- **Ranking** — sortable table of all engines and metrics.
- **Gallery** — color-coded CER badges per document.
- **Document** — synchronized N-way diff, triple diff for OCR+LLM.
- **Analyses** — distribution charts, Pareto, calibration, robustness
  projection, philological profile, longitudinal trends, levers.
- **Characters** — Unicode confusion matrix, ligature analysis.

Above the views: factual narrative synthesis (20+ deterministic
detectors, every number traceable to the input — anti-hallucination
proven by tests), Critical Difference Diagram, Pareto front. Side
panels for contextual glossary and Advanced mode (visible columns,
strata filters, opt-in personal composite score).

### Web interface

FastAPI application with real-time SSE progress streaming, ZIP
upload from the browser, dynamic engine and normalization profile
selectors, browse and re-download generated reports, bilingual
French/English UI. Deployable on HuggingFace Spaces (Docker, port
7860) **and** on institutional infrastructure (see
[`docs/operations/deployment-institutional.md`](docs/operations/deployment-institutional.md)).

### Longitudinal tracking & robustness

Optional SQLite database recording benchmark history across runs.
CER evolution curves per engine, automatic regression detection
between consecutive runs (Pettitt change-point analysis, Sprint 92).
**Robustness analysis** measures engine resilience to noise, blur,
rotation, resolution and binarization, projected on the real corpus
quality profile (Sprint 81).

---

## Quick start

```bash
# Install
pip install -e ".[dev,web]"

# Tesseract (system binary, required for the Tesseract engine)
sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-lat   # Debian/Ubuntu
brew install tesseract tesseract-lang                                # macOS

# Generate a demo report (no engine needed)
picarones demo --output demo_report.html

# Run a benchmark
picarones run --corpus ./corpus/ --engines tesseract --output results.json
picarones report --results results.json --output report.html

# Web UI
picarones serve --port 8080
```

For Docker, institutional deployment, or HuggingFace Spaces, see
[`docs/how-to/install.md`](docs/how-to/install.md) and
[`docs/operations/deployment-institutional.md`](docs/operations/deployment-institutional.md).

---

## Supported engines

<!-- generated:engines -->

| Engine | Type | Installation |
|--------|------|-------------|
| **Azure Doc Intelligence** | Cloud API | `AZURE_DOC_INTEL_ENDPOINT` + `AZURE_DOC_INTEL_KEY` |
| **Calamari OCR** | Local Python | `pip install -e .[calamari]` + checkpoint |
| **Google Vision** | Cloud API | `GOOGLE_APPLICATION_CREDENTIALS` env var |
| **Kraken HTR** | Local Python | `pip install -e .[kraken]` + modèle `.mlmodel` |
| **Mistral OCR** | Cloud API | `MISTRAL_API_KEY` env var |
| **Pero OCR** | Local Python | `pip install -e .[pero]` |
| **Tesseract 5** | Local CLI | `pip install pytesseract` + system binary |

<!-- /generated:engines -->

LLM/VLM adapters (used through pipelines, not as standalone OCR
engines): GPT-4o, Claude, Mistral Large, Ollama (local). See
[`docs/how-to/cli-workflows.md`](docs/how-to/cli-workflows.md).

The `Engine` table is regenerated automatically by
`scripts/gen_readme_tables.py` — adding a new adapter under
`picarones/adapters/ocr/` makes the next CI run update this table
or fail.

---

## CLI commands

<!-- generated:cli -->

| Command | Description |
|---------|-------------|
| `picarones compare` | Compare two benchmark JSON runs and flag regressions (Sprint 28) |
| `picarones demo` | Generate a demo report with synthetic data (no engine required) |
| `picarones diagnose` | Pre-wired workflow: bench + improvement levers + factual recommendations |
| `picarones economics` | Pre-wired workflow: bench + effective throughput + cost projection |
| `picarones edition` | Pre-wired workflow: bench + philological metrics for critical editing |
| `picarones engines` | List available OCR engines and LLM adapters |
| `picarones history` | Query longitudinal benchmark history (SQLite) |
| `picarones import` | Import a corpus from a remote source (IIIF, HF, HTR-United) |
| `picarones info` | Display version and system information |
| `picarones metrics` | Compute CER/WER between two text files |
| `picarones report` | Generate an HTML report from JSON results |
| `picarones robustness` | Run robustness analysis with degraded images |
| `picarones run` | Run a full benchmark on a corpus |
| `picarones serve` | Launch the FastAPI web interface |

<!-- /generated:cli -->

Each command supports `--help` for full options. See
[`docs/how-to/cli-workflows.md`](docs/how-to/cli-workflows.md) for end-to-end
examples.

---

## Web API endpoints

The web app exposes a documented OpenAPI spec at `/docs` (Swagger UI)
when running. Summary:

<!-- generated:endpoints -->

| Method | Endpoint | Summary |
|--------|----------|---------|
| `GET` | `/` | Index |
| `POST` | `/api/benchmark/run` | Api Benchmark Run |
| `POST` | `/api/benchmark/{job_id}/cancel` | Api Benchmark Cancel |
| `GET` | `/api/benchmark/{job_id}/status` | Api Benchmark Status |
| `GET` | `/api/benchmark/{job_id}/stream` | Api Benchmark Stream |
| `GET` | `/api/benchmark/{job_id}/synthesis_preview` | Api Benchmark Synthesis Preview |
| `POST` | `/api/config/load` | Api Config Load |
| `POST` | `/api/config/save` | Api Config Save |
| `GET` | `/api/corpus/browse` | Api Corpus Browse |
| `GET` | `/api/corpus/image/{upload_id}/{filename}` | Api Corpus Image |
| `POST` | `/api/corpus/upload` | Api Corpus Upload |
| `GET` | `/api/corpus/uploads` | Api Corpus Uploads |
| `DELETE` | `/api/corpus/uploads/{corpus_id}` | Api Corpus Delete |
| `GET` | `/api/csrf/token` | Api Csrf Token |
| `GET` | `/api/engines` | Api Engines |
| `GET` | `/api/history/regressions` | Api History Regressions |
| `GET` | `/api/htr-united/catalogue` | Api Htr United Catalogue |
| `POST` | `/api/htr-united/import` | Api Htr United Import |
| `POST` | `/api/huggingface/import` | Api Huggingface Import |
| `GET` | `/api/huggingface/search` | Api Huggingface Search |
| `GET` | `/api/lang` | Api Get Lang |
| `POST` | `/api/lang/{lang_code}` | Api Set Lang |
| `GET` | `/api/models/{provider}` | Api Models |
| `GET` | `/api/normalization/profiles` | Api Normalization Profiles |
| `POST` | `/api/normalization/profiles/preview` | Api Normalization Profile Preview |
| `GET` | `/api/reports` | Api Reports |
| `GET` | `/api/status` | Api Status |
| `GET` | `/health` | Health |
| `GET` | `/metrics` | Metrics Endpoint |
| `GET` | `/reports/{filename}` | Serve Report |

<!-- /generated:endpoints -->

The complete OpenAPI JSON is also exposed at `/openapi.json` for
client generation.

---

## Normalization profiles

Picarones ships **11 built-in normalization profiles** for historical
text comparison (defined in
[`picarones/formats/text/normalization.py`](picarones/formats/text/normalization.py),
exposed via `/api/normalization/profiles`):

`nfc`, `caseless`, `minimal`, `medieval_french`,
`early_modern_french`, `medieval_latin`, `medieval_english`,
`early_modern_english`, `secretary_hand`, `sans_ponctuation`,
`sans_apostrophes`.

Custom profiles can be loaded from YAML files with user-defined
diplomatic tables and `exclude_chars` sets. See
[`docs/reference/normalization-profiles.md`](docs/reference/normalization-profiles.md).

A traceability table mapping each profile to its source standard
(MUFI v4.0, TEI P5, DEAF) will ship in Sprint A12 (B-6).

---

## Project structure

```
picarones/
├── domain/         Layer 1 — pure types (Pydantic, stdlib only)
├── formats/        Layer 2 — parsing/serialization (ALTO, PAGE XML)
├── evaluation/     Layer 3 — metrics & analyses
├── pipeline/       Layer 4 — canonical pipeline executor
├── adapters/       Layer 5 — external libs (OCR, LLM, VLM, corpus)
├── app/            Layer 6 — application services
├── reports/     Layer 7 — HTML / JSON / CSV report renderers
└── interfaces/     Layer 8 — CLI Click, Web FastAPI
```

Strict 8-layer architecture: imports flow outer → inner. Enforced
by `tests/architecture/test_layer_dependencies.py`. The v2.0
release (May 2026) removed all legacy top-level packages (`core/`,
`measurements/`, `engines/`, `llm/`, `pipelines/`, `report/`,
`modules/`, `cli/`, `web/`, `extras/`) and the transitional
sub-packages (`adapters/legacy_engines/`, `adapters/legacy_pipelines/`,
`interfaces/{cli,web}/_legacy/`). See
[`docs/explanation/architecture.md`](docs/explanation/architecture.md)
for the full manifesto and migration history under
`docs/archives/migration/`.

---

## Environment variables

See [`.env.example`](.env.example) for the complete list. Key
variables:

```bash
# Security & mode (cf. SECURITY.md)
PICARONES_PUBLIC_MODE=         # 1/true/yes for HF Space (no cloud OCR)
PICARONES_CSRF_REQUIRED=       # 1 for institutional deployment
PICARONES_BROWSE_ROOTS=        # restrict browse to specific paths

# Cloud API keys (optional)
MISTRAL_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_APPLICATION_CREDENTIALS=
AZURE_DOC_INTEL_ENDPOINT=
AZURE_DOC_INTEL_KEY=

# RGPD retention (Sprint A11)
PICARONES_UPLOAD_RETENTION_DAYS=7
```

For HuggingFace Spaces, set these in **Settings → Variables and secrets**.

---

## CI/CD

GitHub Actions: `.github/workflows/`

- `ci.yml` — tests on Python 3.11/3.12/3.13 × Linux/macOS/Windows,
  ruff, mypy strict on core/, security scanners (bandit + pip-audit
  + trivy), coverage gate `--cov-fail-under=85`, pytest-timeout
  300s.
- `precommit.yml` — replays pre-commit hooks (catches `--no-verify`
  bypass).
- `release.yml` — on tag `v*.*.*`: PyPI + ghcr.io multi-arch +
  GitHub Release with notes from CHANGELOG.
- `perf_regression.yml` — weekly cron + PR-triggered: CER
  anti-regression on a synthetic reference corpus.
- `sync_to_huggingface.yml` — auto-syncs `main` to the HF Space.

---

## Development

```bash
pip install -e ".[dev,web]"
pre-commit install
python -m pytest tests/ -q          # ``python -m`` requis si pytest est uv-installé
ruff check picarones/ tests/
python -m mypy picarones/domain/    # strict mode (Layer 1)
python -m mypy picarones/           # lax mode (full tree)
```

**Test suite**: ~4800 tests, ~3 min on a modern laptop. Coverage
floor at 85% (currently ~87%). The `network` marker excludes tests
requiring live HTTP. A handful of tests depend on optional engines
(`pero-ocr`, `pytesseract`) and are skipped/fail gracefully when
those binaries are not installed in the local environment — the CI
matrix runs them in a fully provisioned image.

For end-to-end developer guides, see
[`docs/developer/index.md`](docs/developer/index.md) (FR) /
[`docs/developer/index.en.md`](docs/developer/index.en.md) (EN).

### Conventions

- Never `except Exception: pass` — use
  `logger.warning("[module] degraded feature: %s", e)`.
- One canonical home per module — circle dependency direction
  enforced by tests.
- Engines declare `execution_mode` (`"io"` or `"cpu"`) so the
  runner picks `ThreadPoolExecutor` vs `ProcessPoolExecutor`
  appropriately.
- Hardcoded UI strings forbidden — always go through i18n
  (cf. [`docs/developer/extending-i18n.md`](docs/developer/extending-i18n.md)).

---

## Roadmap

Detailed history and current direction live in:

- [`CHANGELOG.md`](CHANGELOG.md) — Keep a Changelog format,
  one entry per sprint up to the latest release.
- [`docs/roadmap/evolution-2026.md`](docs/roadmap/evolution-2026.md) —
  technical evolution roadmap (axes A and B for 2026+).
- [`docs/roadmap/rewrite-2026.md`](docs/roadmap/rewrite-2026.md) —
  targeted rewrite plan (S1–S26) restructuring orchestration around
  `Pipeline → Artifacts → Projection → EvaluationView`. Target: end of 2026.
- [`docs/audits/`](docs/audits/) — internal audit notes ; [`BACKLOG_POST_LIVRAISON.md`](BACKLOG_POST_LIVRAISON.md) — promises **not** in scope.

**Honest status (May 2026).** Several items historically presented as
"institutional readiness complete" are not at the level the README
previously claimed and remain on the post-delivery backlog:

- RGPD documentation is a draft, not a validated policy.
- Governance / COI policies are documented but not exercised by an
  external review.
- `CITATION.cff` + Zenodo DOI + JOSS submission are planned, not done.
- Accessibility (WCAG 2.1 AA) and security pentest are scoped but
  not externally audited.

The **rewrite-2026** plan (S1–S26) prioritises stabilising the
benchmark core and the security boundary of the web layer over
adding new features. Until S26 ships, treat the web app as an
experimental demonstrator and the CLI as the supported interface.

---

## Documentation

| Audience | Entry point |
|----------|-------------|
| **End user** | [`docs/tutorials/reading-a-report.md`](docs/tutorials/reading-a-report.md) ([EN](docs/tutorials/reading-a-report.en.md)) |
| **Developer** | [`docs/developer/index.md`](docs/developer/index.md) ([EN](docs/developer/index.en.md)) |
| **Operations / DSI** | [`docs/operations/deployment-institutional.md`](docs/operations/deployment-institutional.md), [`docs/operations/data-retention-rgpd.md`](docs/operations/data-retention-rgpd.md), [`docs/operations/release-process.md`](docs/operations/release-process.md) |
| **Architect** | [`docs/explanation/architecture.md`](docs/explanation/architecture.md), [`docs/reference/api-stable.md`](docs/reference/api-stable.md) |
| **Researcher** | [`docs/case-studies/`](docs/case-studies/), [`docs/reference/reproducibility-snapshots.md`](docs/reference/reproducibility-snapshots.md) |
| **Contributor** | [`CONTRIBUTING.md`](CONTRIBUTING.md), [`GOVERNANCE.md`](GOVERNANCE.md), [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) |
| **Security** | [`SECURITY.md`](SECURITY.md) |
| **Accessibility** | [`ACCESSIBILITY.md`](ACCESSIBILITY.md) |

The complete functional specification is in
[`SPECS.md`](SPECS.md) (full refresh planned in Sprint A14).

---

## Citation

A `CITATION.cff` file and a Zenodo DOI are **planned**, not yet
shipped (see [`BACKLOG_POST_LIVRAISON.md`](BACKLOG_POST_LIVRAISON.md)).
Cite the GitHub repository with the commit SHA used in your benchmark.
Every Picarones report embeds the commit hash and a snapshot of the
parameters used (cf.
[`docs/reference/reproducibility-snapshots.md`](docs/reference/reproducibility-snapshots.md))
so the cited commit is sufficient to attribute the result.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) (FR) /
[`CONTRIBUTING.en.md`](CONTRIBUTING.en.md) (EN).
Code of conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
(Contributor Covenant 2.1).
Governance & maintainership: [`GOVERNANCE.md`](GOVERNANCE.md).

---

## License

[Apache License 2.0](LICENSE)

Copyright 2024–2026 Picarones contributors.
