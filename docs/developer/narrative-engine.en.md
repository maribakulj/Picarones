<!-- translation: machine + human review pending -->
<!-- canonical: docs/developer/narrative-engine.md (FR) -->

# Extending the narrative engine

> 🇫🇷 [Version française](narrative-engine.md)

The narrative engine produces the **factual synthesis** at the top
of each report (Sprint 19). It detects salient facts via 20+
deterministic detectors, arbitrates them (importance, anti-
contradiction), and renders them through YAML templates with
`str.format_map` — guaranteed traceability and zero hallucination.

## Add a new detector in 5 steps

### 1. Add a `FactType` in `picarones/core/facts.py`

```python
class FactType(str, Enum):
    # ... existing ...
    YOUR_NEW_FACT = "your_new_fact"
    """Short docstring describing what triggers this fact."""
```

### 2. Add the FR + EN templates

`picarones/measurements/narrative/templates/fr.yaml`:

```yaml
your_new_fact: >-
  Phrase factuelle citant {engine} et {value_pct} % — pas de chiffres
  en dur, tous viennent du payload du Fact.
```

Same in `en.yaml` with the English version.

### 3. Implement the detector

In an existing detector module (e.g.
`picarones/measurements/narrative/detectors/quality.py` for
quality-related facts) or a new one if a new family is justified:

```python
@register_detector(
    FactType.YOUR_NEW_FACT,
    priority=85,  # ordering in the synthesis
    importance=FactImportance.MEDIUM,
)
def detect_your_new_fact(benchmark_data: dict) -> list[Fact]:
    """Decide whether to emit Facts based on benchmark_data.

    Read the keys you need from benchmark_data; never invent values.
    """
    # ... your logic ...
    return [Fact(
        type=FactType.YOUR_NEW_FACT,
        importance=FactImportance.MEDIUM,
        payload={"engine": engine_name, "value_pct": round(value * 100, 2)},
        engines_involved=(engine_name,),
    )]
```

**Rule**: every value in `payload` MUST come from `benchmark_data`.
Never compute a fancy derived metric here that isn't already in the
input — the anti-hallucination test would catch it.

### 4. Register the detector in the package `__init__`

`picarones/measurements/narrative/detectors/__init__.py`:

```python
from picarones.measurements.narrative.detectors.quality import (
    # ...
    detect_your_new_fact,
)
```

And add it to `__all__`.

### 5. Update the arbiter ordering

`picarones/measurements/narrative/arbiter.py` — append your new
type to `_FALLBACK_TYPE_ORDER` at the right position.

### 6. Write tests

In `tests/measurements/`:

- A unit test of your detector (3+ canonical cases: triggers,
  doesn't trigger, edge case).
- A traceability test (FR + EN): `build_synthesis(...)` produces
  output where every number is in the payload.

Update `tests/integration/test_chantier5.py` and
`tests/measurements/test_sprint29_detector_registry.py` to bump
the detector count.

## Editorial rules

- **Factual only**: no recommendation, no value judgment. "Engine
  X has a CER of 5.2%" — yes. "Engine X is the best for archives" —
  no.
- **Symmetric thresholds**: thresholds are public in the detector
  source code, not hidden. They apply equally to all engines.
- **Anti-contradiction**: if your detector contradicts another
  (e.g., Wilcoxon-uncorrected gap vs Nemenyi-corrected tie), the
  arbiter handles it via the `_COMPLEMENTARY_PAIRS` mechanism — add
  your pair if needed.

## Testing the synthesis

```bash
pytest tests/measurements/test_sprint19_narrative_engine.py
pytest tests/measurements/test_sprint23_anti_hallucination.py
```

The anti-hallucination test parses the rendered synthesis and
verifies that every number is traceable to a Fact payload. If it
fails after your changes, you've likely cited a value not present
in `benchmark_data`.
