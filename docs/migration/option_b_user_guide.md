# Guide de migration — Option B (RunOrchestrator)

Document utilisateur pour migrer du legacy `run_benchmark_via_service`
vers le nouveau service `RunOrchestrator`.  Phase B3 du chantier de
migration Option B (mai 2026).

## TL;DR

```python
# ❌ Legacy (deprecated depuis Phase B3, retrait en B8)
from picarones.app.services.benchmark_runner import run_benchmark_via_service
result = run_benchmark_via_service(
    corpus=corp,
    engines=[engine1, engine2],
    char_exclude="!?.,",
    normalization_profile="caseless",
    profile="standard",
    output_json="bm.json",
)

# ✅ Nouveau (Phase B3+)
from picarones import RunOrchestrator, load_run_spec_from_yaml

spec_yaml = """
corpus_dir: ./corpus_extracted
pipelines:
  - name: tesseract_only
    initial_inputs: [image]
    steps:
      - id: ocr
        adapter_class: picarones.adapters.ocr.tesseract.TesseractAdapter
        adapter_kwargs: {lang: fra, psm: 6}
        input_types: [image]
        output_types: [raw_text]
views: [text_final]
output_dir: ./runs/r1
char_exclude: "!?.,"
normalization_profile: caseless
profile: standard
output_json: ./runs/r1/bm.json
"""
spec = load_run_spec_from_yaml(spec_yaml)
orch_result = RunOrchestrator("./runs/r1").execute(spec)
# orch_result.run_result : RunResult typé (JSONL natif)
# orch_result.persisted_files : map {kind: Path} des 4 fichiers JSONL
# orch_result.run_result.view_results_for("text_final") : ViewResult tuples

# Le BenchmarkResult legacy reste disponible via output_json :
from picarones import BenchmarkResult
import json
loaded_legacy = json.loads((Path("./runs/r1/bm.json")).read_text(encoding="utf-8"))
```

## Pourquoi migrer

### Avantages immédiats

1. **API déclarative** : un `RunSpec` Pydantic validé au lieu de 12
   kwargs hétérogènes.  Sérialisable YAML pour la reproductibilité.
2. **JSONL streamable** : 4 fichiers séparés (`run_manifest.json`,
   `pipeline_results.jsonl`, `artifacts_index.jsonl`,
   `view_results.jsonl`) au lieu d'un JSON monolithique.  Indispensable
   pour les corpus 1000+ docs.
3. **Vues natives** : les 3 vues canoniques (`text_final`,
   `alto_documentary`, `searchability`) produisent des `ViewResult`
   typés exposés directement dans le `RunResult` — pas seulement des
   métriques CER/WER plates.
4. **Cohabitation** : si vous avez besoin du `BenchmarkResult` legacy
   (rapport HTML, narrative engine, downstream tooling), il suffit de
   spécifier `output_json` dans le `RunSpec` — le format produit est
   strictement identique à `run_benchmark_via_service(output_json=...)`.

### Avantages à long terme

- **Audit de reproductibilité** : `RunManifest` capture le verrou de
  dépendances + binaires système + version du code.  Un run peut être
  rejoué bit-à-bit.
- **Extensions natives** : ajouter une nouvelle vue
  (`HallucinationView`, `CalibrationView`) demande une seule entrée
  dans `_build_views()` — pas de fork du runner.

## Mapping des paramètres

| `run_benchmark_via_service` (legacy) | `RunSpec` (B3+) |
|---|---|
| `corpus: Corpus` (en mémoire) | `corpus_dir: str` ou `corpus_zip: str` (chargé par l'orchestrateur) |
| `engines: list[Any]` (instances) | `pipelines: tuple[PipelineSpecYaml, ...]` (dotted paths + kwargs YAML) |
| `char_exclude: str \| None` | `char_exclude: str \| None` (identique) |
| `normalization_profile: NormalizationProfile \| str \| None` | `normalization_profile: str \| None` (résolu via `get_builtin_profile` dans l'orchestrateur) |
| `output_json: str \| None` | `output_json: str \| None` (identique) |
| `code_version: str \| None` | `code_version: str` (défaut `"0.0.0-unset"`) |
| `progress_callback: Callable` | kwarg de `execute()` (non-sérialisable) |
| `cancel_event: threading.Event` | kwarg de `execute()` (non-sérialisable) |
| `partial_dir: str \| Path \| None` | `partial_dir: str \| None` (identique) |
| `entity_extractor: Callable[[str], list[dict]]` | `entity_extractor: str` (dotted path Python, résolu lazy) |
| `timeout_seconds: float` | `timeout_seconds_per_doc: float` (renommé pour clarté) |
| `profile: str` | `profile: str` (identique) |

**Retour** : 
- Legacy retourne `BenchmarkResult` (legacy dataclass).
- Nouveau retourne `OrchestrationResult(run_result, persisted_files, ...)`.
  Le `BenchmarkResult` legacy peut être récupéré en désérialisant
  `spec.output_json` si fourni.

## Cas concrets

### Cas 1 — Corpus déjà chargé en mémoire (CLI / Python script)

Si vous avez un objet `Corpus` legacy (chargé via
`load_corpus_from_directory`), persistez-le dans un dossier ou un
zip avant de construire le `RunSpec`.  En pratique, le caller a
déjà le chemin du dossier source — il suffit de le réutiliser :

```python
# Avant
from picarones.evaluation.corpus import load_corpus_from_directory
from picarones.app.services.benchmark_runner import run_benchmark_via_service

corpus = load_corpus_from_directory("./my_corpus/")
result = run_benchmark_via_service(corpus, [tesseract_adapter])

# Après
from picarones import RunOrchestrator, load_run_spec_from_yaml

spec = load_run_spec_from_yaml(f"""
corpus_dir: ./my_corpus/
pipelines:
  - name: tesseract_only
    initial_inputs: [image]
    steps:
      - id: ocr
        adapter_class: picarones.adapters.ocr.tesseract.TesseractAdapter
        adapter_kwargs: {{lang: fra, psm: 6}}
        input_types: [image]
        output_types: [raw_text]
views: [text_final]
output_dir: ./runs/r1
""")
result = RunOrchestrator("./runs/r1").execute(spec)
```

### Cas 2 — Reprise sur interruption (`partial_dir`)

```python
spec = load_run_spec_from_yaml(f"""
corpus_dir: ./my_corpus/
pipelines: [...]
views: [text_final]
output_dir: ./runs/r1
partial_dir: ./runs/r1/partial
""")

# Si le run crashe, le partial est conservé.  Re-lancer reprend là où
# c'était.  Le fingerprint (config + corpus mtime/size + code version)
# garantit qu'on ne réutilise pas un partial incompatible.
result = RunOrchestrator("./runs/r1").execute(spec)
```

### Cas 3 — NER attach (entity_extractor)

```python
spec = load_run_spec_from_yaml(f"""
corpus_dir: ./my_corpus/  # avec gt.entities.json
pipelines: [...]
views: [text_final]
output_dir: ./runs/r1
entity_extractor: mypackage.ner:SpacyExtractor
output_json: ./runs/r1/bm.json
""")
# Le dotted path est résolu lazy à execute().  Doit pointer vers une
# factory zéro-arg qui retourne un Callable[[str], list[dict]], ou
# directement vers une fonction (text -> entities).
result = RunOrchestrator("./runs/r1").execute(spec)
```

### Cas 4 — Cancellation

```python
import threading
from picarones import RunOrchestrator, load_run_spec_from_yaml

spec = load_run_spec_from_yaml(...)
cancel_event = threading.Event()

# Dans un autre thread :
#   cancel_event.set()  → le run s'arrête au prochain poll
def cb(engine: str, idx: int, doc_id: str) -> None:
    print(f"[{engine}] doc {idx}: {doc_id}")
    # Décision : annuler après 5 docs
    if idx > 5:
        cancel_event.set()

result = RunOrchestrator("./runs/r1").execute(
    spec, progress_callback=cb, cancel_event=cancel_event,
)
```

## Calendrier de retrait

- **Phase B3** (mai 2026, ce commit) : `run_benchmark_via_service`
  émet une `DeprecationWarning` à chaque appel.  Fonction toujours
  opérationnelle.
- **Phase B4** : migration interne des 25 fichiers de tests.
- **Phase B5-B7** : adaptation des consommateurs (rapport HTML,
  narrative engine, partial_store legacy).
- **Phase B8** (post-deprecation release) : suppression complète de
  `run_benchmark_via_service` et des modules `_benchmark_*.py`
  associés (-1500 LOC nets).

Si vous bloquez sur un cas qui n'est pas couvert ci-dessus, ouvrir
une issue GitHub avec le tag `migration-option-b`.
