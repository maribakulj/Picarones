<!-- translation: machine + human review pending -->
<!-- canonical: docs/tutorials/reading-a-report.md (FR) -->

# Reading a Picarones report

> 🇫🇷 [Version française](reading-a-report.md)

This guide explains how to read a Picarones HTML report — the
self-contained file produced by `picarones report --output report.html`.
It is the primary deliverable of a benchmark and is intended to be
read both by engineers and by domain experts (archivists,
paleographers, project managers).

## Anatomy

A report is structured as **5 main views** (tabs in the navigation):

1. **Ranking** — sortable table of all engines with CER, WER, MER,
   WIL, ligature/diacritic scores, anchor score, etc.
2. **Gallery** — grid view of all documents with color-coded CER
   badges per engine.
3. **Document** — per-document detail with synchronized N-way diff
   between ground truth and each engine output.
4. **Analyses** — statistical charts: CER histogram, radar chart,
   correlation plots, calibration diagrams, Pareto front, etc.
5. **Characters** — Unicode confusion matrix and ligature analysis.

Above the tabs, you'll find:

- The **factual narrative synthesis** (Sprint 19): 3–5 sentences
  summarizing the salient facts (global leader, statistical ties,
  outliers, regression flags). Every number cited in the synthesis
  is traceable to the underlying JSON data — no LLM, no
  hallucination risk.
- The **Critical Difference Diagram** (Sprint 18, Demšar 2006):
  visual representation of which engines are statistically
  indistinguishable.
- The **Pareto front** (Sprint 20): cost vs CER trade-off analysis.

## Suggested reading order

1. **Read the synthesis at the top** (3–5 sentences) — it points
   to the salient facts.
2. **Look at the CDD**: if all engines are connected by a single
   horizontal bar, your corpus does not discriminate them
   sufficiently — increase the corpus or refine its homogeneity.
3. **Open the ranking** sorted by CER median (default since
   Sprint 44). Identify the leader and the gap to second place.
4. **Switch to Gallery** and click on the "Worst cases" filter to
   see what specifically goes wrong.
5. **For an OCR+LLM pipeline**: open Document view and toggle the
   triple diff (GT / raw OCR / post-correction).

## Side panels

Two side panels enrich the report:

- **Glossary** (`?` icon next to each metric) — definition, what
  it measures, usage, limits, primary reference. 25 bilingual
  entries, opens via click on `?`.
- **Advanced mode** (`⚙` button in nav) — visible columns picker,
  per-stratum filters (script type), opt-in personal composite
  score with explicit "no universal weighting" warning.
  All settings are URL-stateful (shareable).

## Export

A "⬇ CSV" button in the navigation exports the current view (with
all customization filters applied) to CSV for Excel/LibreOffice.
JSON, ALTO XML and PAGE XML exports are available via CLI flags
on `picarones run` and `picarones report`.

## `--lazy-images` mode for large corpora

Sprint A5 (item M-16). By default, the HTML report is a **single
file** transportable: all images are embedded as base64 within the
HTML. Convenient for sharing by email, but the file becomes heavy
beyond ~50 documents:

| Corpus size | Inline HTML | Lazy HTML |
|---|---|---|
|   10 docs |  ~5 MB | ~3 MB + ~2 MB assets |
|   50 docs | ~50 MB | ~3 MB + ~10 MB assets |
|  500 docs | ~250 MB (slow to load) | ~3 MB + ~100 MB lazy-loaded |

For digital libraries benchmarking thousands of documents, enable
the lazy mode:

```bash
picarones report --results results.json --output report.html --lazy-images
```

The report stays **self-contained**: copy `report.html` AND the
`report-assets/` folder side by side. Images are referenced by
relative path and loaded by the browser on-demand
(`loading="lazy"` HTML5).

## Further reading

- [Glossary] (embedded in report, accessible via `?` icons)
- [docs/explanation/narrative-engine.md](../explanation/narrative-engine.md)
  — adding a detector (French canonical)
- [docs/developer/extending-glossary.md](../developer/extending-glossary.md)
  — enriching the glossary (French canonical)
- [docs/reference/specification.md](../reference/specification.md) — full project specifications
