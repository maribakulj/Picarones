<!-- translation: machine + human review pending -->
<!-- canonical: CONTRIBUTING.md (FR) -->

# Contributing to Picarones

> 🇫🇷 [Version française](CONTRIBUTING.md)

Thanks for considering a contribution! Picarones is an open-source
project (Apache-2.0) that benefits from contributions across the
spectrum: code, documentation, translations, case studies,
accessibility audits.

## Pre-requisites

- Python 3.11 or 3.12 (3.13 informational, Sprint A1).
- Git, GitHub account.
- A working Picarones dev install:
  ```bash
  git clone https://github.com/maribakulj/Picarones.git
  cd Picarones
  pip install -e ".[dev,web]"
  pre-commit install
  ```

## Workflow

1. **Fork** the repo on GitHub.
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature
   ```
3. **Make your change**, write tests, run them locally:
   ```bash
   pytest tests/ -q --tb=short
   ruff check picarones/ tests/
   ```
4. **Update the CHANGELOG.md** under the `[Unreleased]` section
   in the appropriate category (Added / Changed / Fixed /
   Deprecated / Security).
5. **Push and open a PR** against `main`.

## What we look for in a PR

- **Clear motivation** in the PR description: what problem does
  this solve, why this approach.
- **Tests** that exercise the new code path.
- **Documentation** updates (README, SPECS, user/dev guides) when
  relevant. The CI doc-consistency tests (`tests/docs/`) will fail
  if you add a new engine, CLI command or endpoint without
  updating the README tables.
- **Editorial neutrality** in the narrative engine: factual
  statements only, no recommendations (cf. CLAUDE.md philosophy).

## What we don't accept

- **Breaking changes** without a major version bump and 2-version
  deprecation notice.
- **Bare `except Exception: pass`** — replace by
  `logger.warning("[module] degraded feature: %s", e)`.
- **Hardcoded UI strings** — use the i18n mechanism (see
  `docs/developer/extending-i18n.md` — French canonical).
- **Commits with `--no-verify`** — bypassing pre-commit hooks is
  detected by the CI `precommit.yml` workflow (Sprint A1).

## Reviewing

PRs are reviewed by the maintainer(s) listed in `.github/CODEOWNERS`
for the path you touched. Best-effort SLO of 7 working days for
the first review (cf. `GOVERNANCE.md`).

The reviewer may ask:

- to split a large PR into smaller logical units ;
- to refactor for circular dependency cleanliness ;
- to add an editorial note in the synthesis (anti-hallucination
  test will catch any fancy derived metric not present in input).

## Code of conduct

All interactions on the project are governed by our
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) (Contributor Covenant
2.1). In short: be kind, accept constructive feedback gracefully,
focus on what's best for the community.

## Reporting a security issue

**Don't** open a public GitHub issue for security problems. See
[`SECURITY.md`](SECURITY.md) for the responsible disclosure
procedure.

## Licensing of your contribution

By submitting a PR, you agree that your contribution will be
licensed under the same license as the project (Apache-2.0).

## Help and questions

- Quick questions: open a GitHub Discussion.
- Bug reports: open a GitHub Issue with reproduction steps.
- Architecture decisions / philosophical changes: open a
  Discussion first to gauge consensus before writing code.
