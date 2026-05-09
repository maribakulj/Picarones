<!-- translation: machine + human review pending -->
<!-- canonical: docs/developer/extending-glossary.md (FR) -->

# Extending the contextual glossary

> 🇫🇷 [Version française](extending-glossary.md)

The contextual glossary (Sprint 21) provides definitions, usage
notes and primary references for the metrics shown in the report.
A small `?` icon next to each metric column header opens a side
panel with the entry's full content.

## Adding an entry

The glossary lives in **two YAML files** that must stay in sync:

- `picarones/reports/html/glossary/fr.yaml`
- `picarones/reports/html/glossary/en.yaml`

Each entry has the following schema:

```yaml
your_metric_key:
  title: "Display name"
  definition: |
    Concise mathematical/operational definition of what this metric
    computes.
  measures: |
    What aspect of OCR/HTR quality this metric captures.
  usage: |
    When to look at this metric ; what threshold is meaningful.
  limits: |
    Known limitations, edge cases, sensitivity to corpus size, etc.
  reference:
    citation: "Author et al. (Year). Title. Journal, vol(num), pages."
    url: "https://doi.org/..."
```

## Editorial rules

- **One reference per metric** (primary citation), not a textbook
  excerpt. If multiple references are essential, use a top-level
  list and explain in `definition`.
- **No recommendation** in `usage`. Describe when the metric is
  observable, not whether the user should trust it.
- **Limits are mandatory** — every metric has limitations.
  Honesty about edge cases is what makes the glossary useful for
  scientific publication.

## Linking from the report

The `?` icon is rendered automatically when a `<th>` carries the
`data-glossary-key="your_metric_key"` attribute. See for example
`picarones/reports/html/templates/view_ranking.html` line ~13.

## Tests

`tests/report/test_sprint21_glossary_customize.py` validates:

- Both languages have the same exact set of keys.
- Every entry has the 4 required fields (definition, measures,
  usage, limits) + reference.
- No HTML injection in field values (rendered via `html.escape`).
- 25+ entries currently shipped (spot-checked common metrics).

## Adding a new language

If you want to ship a third language (e.g. Spanish, German), see
[`extending-i18n.en.md`](extending-i18n.en.md) for the i18n side
and create a third `glossary/<lang>.yaml` with the same keys.
