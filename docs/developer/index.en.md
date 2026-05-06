<!-- translation: machine + human review pending -->
<!-- canonical: docs/developer/index.md (FR) -->

# Developer guide — Picarones

> 🇫🇷 [Version française](index.md)

This documentation targets developers who want to **extend
Picarones**: add a new metric, a new narrative detector, a new
glossary entry, a new translation, or write a custom benchmark
module.

## Architecture

Picarones uses a **3-circle architecture** (manifesto in
[`docs/explanation/architecture.md`](../architecture.md)):

```
   Circle 3 (extras, report, cli, web)
   │
   ▼
   Circle 2 (measurements, engines, llm, pipelines, modules)
   │
   ▼
   Circle 1 (core)
```

Strict dependency rule: imports only flow from outer to inner. No
shim — a module has a single canonical location.

## Extension points

| What | Reference doc |
|---|---|
| Add a typed metric (registered for a junction) | [`narrative-engine.en.md`](narrative-engine.en.md) (parallel pattern) |
| Add a narrative detector + template | [`narrative-engine.en.md`](narrative-engine.en.md) |
| Enrich the contextual glossary | [`extending-glossary.en.md`](extending-glossary.en.md) |
| Add a UI translation (FR/EN/...) | [`extending-i18n.en.md`](extending-i18n.en.md) |
| Write a custom benchmark module (axis B) | [`../user/writing-a-pipeline-module.md`](../user/writing-a-pipeline-module.md) |
| Module policy (manifest + audit) | [`module-policy.md`](module-policy.md) |
| Documentation consistency contract | [`doc-consistency.md`](doc-consistency.md) |

## Running tests

```bash
pip install -e ".[dev,web]"
pytest tests/ -q --tb=short
```

The full suite runs in ~3 minutes. Use `pytest tests/<subdir>/` to
focus on a domain. Tests marked `network` are excluded by default
(`pytest -m network` to include them).

## Code style

- **ruff** lints: `ruff check picarones/ tests/` (config in
  `pyproject.toml`).
- **mypy** strict on `picarones/core/`, lax elsewhere (Sprint A1).
- **No `except Exception: pass`** — replace by
  `logger.warning("[module] degraded feature: %s", e)`.
- **Logger per module**: `logger = logging.getLogger(__name__)`.
- **No emoji** in code or commit messages unless explicitly requested.

## Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

Replays `ruff check`, trailing whitespace, YAML/JSON/TOML check,
secret detection, large files (>500 KB) — same gates as the CI
`lint` job. CI also re-runs the hooks via `precommit.yml` to
catch `--no-verify` bypasses.

## Releasing

See [`docs/operations/release-process.md`](../operations/release-process.md)
for the complete release procedure. Short version:

```bash
git checkout main && git pull
# Update CHANGELOG.md
git tag -a v1.2.0 -m "Picarones 1.2.0"
git push origin main v1.2.0
# The release.yml workflow handles PyPI + ghcr.io + GitHub Release.
```

## Help and support

- Issues: <https://github.com/maribakulj/Picarones/issues>
- Governance: [`../../GOVERNANCE.md`](../../GOVERNANCE.md)
- Code of conduct: [`../../CODE_OF_CONDUCT.md`](../../CODE_OF_CONDUCT.md)
