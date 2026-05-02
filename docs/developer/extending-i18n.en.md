<!-- translation: machine + human review pending -->
<!-- canonical: docs/developer/extending-i18n.md (FR) -->

# Extending i18n (UI translations)

> 🇫🇷 [Version française](extending-i18n.md)

Picarones is bilingual French / English by default (Sprint 17).
This guide covers:

1. Adding a new key (label, button text, error message).
2. Adding a third language.

## Architecture

Two JSON files store all UI strings:

- `picarones/report/i18n/fr.json` — French (canonical)
- `picarones/report/i18n/en.json` — English

Both files **must have the exact same set of keys** — any drift
fails `tests/report/test_a11y_level_aa.py::test_i18n_fr_en_have_same_keys`.

## Adding a key

1. Add the key + value to `fr.json` (canonical):
   ```json
   "your_new_key": "Texte en français"
   ```
2. Add the same key to `en.json` with the English value:
   ```json
   "your_new_key": "Text in English"
   ```
3. Use it in templates via `data-i18n="your_new_key"`:
   ```html
   <button data-i18n="your_new_key">Texte en français</button>
   ```
   The fallback text inside the tag is shown if the key is missing
   in the active locale (defensive). The JS swaps it at load time.

## Locale formatting (Sprint A7)

The key `locale` carries the BCP-47 code:

- `fr.json`: `"locale": "fr-FR"`
- `en.json`: `"locale": "en-GB"`

Used by `fmtNum()` / `fmtInt()` in `_app.js` for `toLocaleString`
(thousands separators, decimals). For a third language, choose an
appropriate BCP-47 code (e.g. `de-DE`, `es-ES`, `it-IT`).

## Adding a third language

1. Copy `fr.json` to `<lang>.json` (e.g. `de.json`) and translate
   every value. Keep the keys identical.
2. Add the language to `picarones/web/state.SUPPORTED_LANGS`:
   ```python
   SUPPORTED_LANGS = frozenset({"fr", "en", "de"})
   ```
3. Add the same key in `picarones/report/glossary/<lang>.yaml`
   with translated definitions (cf. [`extending-glossary.en.md`](extending-glossary.en.md)).
4. Add `<lang>` to the narrative templates:
   `picarones/measurements/narrative/templates/<lang>.yaml`.
5. Add the language switcher entry in the report header (if any).

## Tests

```bash
pytest tests/report/test_a11y_level_aa.py::test_i18n_fr_en_have_same_keys
pytest tests/report/test_sprint17_jinja_refactor.py
```

The first ensures key parity. The second verifies that the
templates compile with the new locale.

## Pitfalls

- **Don't hardcode strings** in templates or JS — always use the
  i18n mechanism, even for "simple" labels. The Sprint A6 scan
  found 11 hardcoded FR fallbacks; we don't want regressions.
- **Don't translate file paths or function names** — they are
  language-neutral identifiers.
- **Test in both languages** — generate a report with `--lang fr`
  AND `--lang en` and visually inspect.
