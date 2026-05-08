# Sprint D — Audit du retrait de `measurements/runner/`

**Sprint D du plan v2.0** — migration du runner legacy
(`measurements/runner/`) vers `app/services/BenchmarkService`.
Préparation à la suppression du sous-package, qui débloque
ensuite Phase 9 (Web → `interfaces/web/`) et Phase 10 (CLI →
`interfaces/cli/`).

Ce document est le pré-requis du Sprint D — il identifie
exhaustivement les **gaps** entre les deux services et le
**plan ordonné** des sub-phases D.1 à D.6.

## 1. Surface du retrait

### 1.1 — Modules à supprimer (`measurements/runner/`)

| Fichier | LOC | Rôle |
|---|---:|---|
| `__init__.py` | 103 | Re-exports |
| `orchestration.py` | 545 | `run_benchmark()` + boucle principale |
| `document.py` | 200 | `_compute_document_result()` |
| `partial.py` | 140 | Reprise sur interruption |
| `workers.py` | 116 | Pool process/thread |
| `aggregation.py` | 82 | Agrégation EngineReport |
| `ner_attach.py` | 133 | Calcul NER (Sprint 40) |
| **Total** | **1 319** | |

### 1.2 — Callers à migrer

| Caller | Sites | Bloquant |
|---|---:|---|
| `cli/_workflows.py` | 2 (run_cmd, compare_cmd) | bloque Phase 10 |
| `web/benchmark_utils.py` | 2 (run_benchmark_thread + thread_v2) | bloque Phase 9 |
| `picarones/__init__.py` | 1 (docstring) | cosmétique |
| `picarones/measurements/__init__.py` | 1 (docstring) | supprimé avec measurements/ |
| `picarones/adapters/corpus/iiif.py` | 1 (docstring) | cosmétique |

**Total : 4 call-sites de production à migrer.** `fixtures.py` ne
consomme PAS `run_benchmark()` (il fabrique un `BenchmarkResult`
synthétique en pur Python).

## 2. API legacy `run_benchmark()`

### 2.1 — Signature

```python
def run_benchmark(
    corpus: Corpus,                                    # legacy
    engines: list[BaseOCREngine],                      # legacy (avec OCRLLMPipeline)
    output_json: Optional[str | Path] = None,          # I/O
    show_progress: bool = True,                        # tqdm
    progress_callback: Optional[callable] = None,      # SSE web
    char_exclude: Optional[frozenset] = None,          # filtre
    max_workers: int = 4,                              # parallélisme
    timeout_seconds: float = 60.0,                     # timeout/doc
    partial_dir: Optional[str | Path] = None,          # reprise
    cancel_event: Optional[threading.Event] = None,    # annulation web
    entity_extractor: Optional[callable] = None,       # NER opt-in
    profile: str = "standard",                         # profil mesures
    normalization_profile: Optional[str] = None,       # normalisation
) -> BenchmarkResult:
```

### 2.2 — Features fournies

| Feature | Module | Mécanisme |
|---|---|---|
| Boucle (engines × corpus) | `orchestration.py` | imbriquée séquentielle |
| Parallélisme intra-engine | `workers.py` | `Process` ou `ThreadPoolExecutor` selon `engine.execution_mode` |
| Timeout par document | `workers.py` | Future + `concurrent.futures.wait` |
| Reprise interruption | `partial.py` | JSON par engine, document-par-document |
| Annulation propre | `orchestration.py` | `threading.Event` propagé |
| Progress bar | `orchestration.py` | tqdm + callback |
| OCR confidences | `document.py` | `EngineResult.token_confidences` |
| OCR+LLM metadata | `orchestration.py:520` | duck-typing `is_pipeline` (Sprint C.1 ✅) |
| Over-normalization | `document.py` | `detect_over_normalization(gt, ocr_int, hyp)` |
| NER calculations | `ner_attach.py` | Optionnel via `entity_extractor` |
| Profil normalisation | `orchestration.py` | `validate_profile` + `normalization_profile` |
| Aggregation EngineReport | `aggregation.py` | per-engine corpus stats |
| Fail-if-CER seuil | callers (CLI) | post-hoc sur `BenchmarkResult` |

## 3. API rewrite `BenchmarkService`

### 3.1 — Signature

```python
class BenchmarkService:
    def run(
        self,
        *,
        corpus: CorpusSpec,                            # rewrite
        pipelines: Iterable[PipelineSpec],             # rewrite (déclaratif)
        views: Iterable[EvaluationView],
        ground_truth_factory: GroundTruthFactory,
        pipeline_inputs_factory: PipelineInputsFactory,
        context_factory: ContextFactory,
        run_id: str | None = None,
        dependencies_lock: dict[str, str] | None = None,
        adapter_kwargs: dict[str, dict[str, Any]] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> RunResult:
```

### 3.2 — Features fournies

| Feature | Disponible ? | Source |
|---|---|---|
| Parallélisme intra-pipeline | ✅ | `pipeline.runner.CorpusRunner` (max_in_flight) |
| Timeout par document | ✅ | `CorpusRunner.timeout_seconds_per_doc` |
| Annulation propre | ✅ | `threading.Event` (CorpusRunner) |
| `RunManifest` provenance | ✅ | `RunResult.manifest` |
| `EvaluationView` (S13+) | ✅ | natif |
| Run ID stable | ✅ | `run_id` arg |

### 3.3 — Features manquantes

| Feature | Effort | Sub-phase |
|---|---|---|
| Progress callback compat web | Faible | D.2.a |
| tqdm progress bar | Faible | D.2.a |
| Reprise sur interruption (`partial.py`) | Moyen | D.2.b |
| `output_json` sérialisation directe | Faible | D.2.c |
| Conversion `BenchmarkResult` → `RunResult` | Élevé | D.3 |
| `over_normalization` aggregation | Moyen | D.2.d (déjà migré au volet 1) |
| NER attach via `entity_extractor` | Moyen | D.2.e |
| `profile` validation | Faible | D.2.f |
| `normalization_profile` | Faible | D.2.f |
| `char_exclude` filter | Faible | D.2.f |
| `fail_if_cer` (callers) | Faible | côté caller |

## 4. Plan ordonné

### D.0 — Audit (ce document) — fait ✅

### D.1 — Adapter de compat `run_benchmark_via_service`

Fonction qui présente l'API legacy (`Corpus`, `engines`,
`output_json`, etc.) et construit en interne :

1. `CorpusSpec` à partir du `Corpus` legacy (mapping
   `Document` → `DocumentRef`).
2. `PipelineSpec` à partir de chaque `BaseOCREngine` :
   - OCR seul → spec mono-step via le builder
     (`adapter_name=engine.name`, params=engine.config).
   - `OCRLLMPipeline` → utilise déjà `make_ocr_llm_pipeline_spec`
     en interne via Sprint B.
3. Adapter resolver (`name → instance`) qui retrouve les engines
   par leur `name`.
4. Factories par défaut (ground_truth, pipeline_inputs, context).
5. Lance `BenchmarkService.run(...)`.
6. Convertit `RunResult` → `BenchmarkResult` legacy
   (mapping inverse : `RunDocumentResult` → `DocumentResult`,
   `pipeline_results` → `EngineReport`).

**Effort** : 2-3 j. **Risque** : la conversion bidirectionnelle
``Corpus ↔ CorpusSpec`` et ``RunResult ↔ BenchmarkResult`` est la
partie délicate (les structures sont différentes par design).

### D.2 — Combler les gaps `BenchmarkService`

| Sub-phase | Gap | Effort |
|---|---|---|
| D.2.a | progress callback + tqdm | 0.5 j |
| D.2.b | reprise interruption | 1 j |
| D.2.c | `output_json` sérialisation | 0.3 j |
| D.2.d | over_normalization aggregation | 0.5 j |
| D.2.e | NER attach via entity_extractor | 0.5 j |
| D.2.f | profile + normalization + char_exclude | 0.5 j |

**Total D.2** : ~3.3 j.

### D.3 — Migrer `web/benchmark_utils.py:run_benchmark_thread_v2`

Le caller le plus simple à migrer (le plus récent, code propre) :
remplacer `run_benchmark(...)` par `run_benchmark_via_service(...)`.
Tests `tests/web/test_sprint28_ux_save_compare.py` doivent rester
verts.

**Effort** : 0.5 j.

### D.4 — Migrer `web/benchmark_utils.py:run_benchmark_thread` (legacy)

Cette fonction est plus ancienne et utilise un format de competitor
configuration différent. Probablement redondante avec `_v2` —
candidat à la **suppression pure** plutôt qu'à la migration.

**Effort** : 0.3 j (suppression) ou 1 j (migration si conservé).

### D.5 — Migrer `cli/_workflows.py`

5 commandes : `run`, `diagnose`, `economics`, `edition`, `compare`.
Toutes appellent `run_benchmark()` directement.  Migration par
commande, en commençant par la plus simple (`run`).

**Effort** : 1.5 j.

### D.6 — Suppression `measurements/runner/`

Une fois tous les callers migrés :

```bash
rm -r picarones/measurements/runner/
```

Plus mise à jour des tests qui importaient depuis `runner` (51
fichiers) et des baselines architecturaux.

**Effort** : 0.5 j.

## 5. Ordre d'enchaînement et durée

```
D.0 (audit)        ✅ fait
  ↓
D.1 (adapter)      ←─────┐
  ↓                       │
D.2 (gaps)         ←──────┤  parallélisable
  ↓                       │
D.3 (web v2)              │
  ↓                       │
D.4 (web v1, opt.)        │
  ↓                       │
D.5 (CLI)                 │
  ↓                       │
D.6 (suppression)  ←──────┘
```

| Sub-phase | Effort |
|---|---:|
| D.1 | 2-3 j |
| D.2 | 3-4 j |
| D.3 | 0.5 j |
| D.4 | 0.3-1 j |
| D.5 | 1.5 j |
| D.6 | 0.5 j |
| **Total Sprint D** | **8-10 j** |

## 6. Risques et mitigations

| Risque | Probabilité | Mitigation |
|---|---|---|
| Conversion `RunResult ↔ BenchmarkResult` perd des champs | Élevée | tests round-trip détaillés en D.1 |
| Performance dégradée du runner rewrite | Moyenne | benchmark de comparaison sur fixtures |
| Reprise sur interruption manque dans rewrite | Élevée | D.2.b prioritaire |
| Tests Sprint 15 (warnings LLM vide) cassent | Faible | Sprint B a déjà préservé les warnings |
| Web SSE callback signature incompatible | Moyenne | D.2.a en premier |
| CLI fail-if-cer logique côté caller | Faible | Reste côté CLI, ne touche pas le runner |

## 7. Critères d'acceptation Sprint D

À l'issue de D.6 :

- [ ] `picarones/measurements/runner/` n'existe plus.
- [ ] `from picarones.measurements.runner import run_benchmark`
      → ImportError.
- [ ] `web/benchmark_utils.py` consomme `BenchmarkService` (ou son
      adapter).
- [ ] `cli/_workflows.py` consomme `BenchmarkService` (ou son
      adapter).
- [ ] Tests CLI (Sprint 9, 11) verts.
- [ ] Tests Web (Sprint 6, 28) verts.
- [ ] Tests metrics (Sprint 3, 15) verts.
- [ ] Performance : pas de régression > 10 % sur fixtures
      (corpus de 5 documents, 1 engine Tesseract).
- [ ] Reprise sur interruption : test `test_partial_resume.py`
      vert (à créer).
- [ ] Phase 9 (Web → `interfaces/web/`) débloquée — plus aucun
      import `measurements.runner` dans `web/`.
- [ ] Phase 10 (CLI → `interfaces/cli/`) débloquée — plus aucun
      import `measurements.runner` dans `cli/`.

## 8. Non-objectifs (hors-scope Sprint D)

- ❌ Refactor de `app/services/run_orchestrator.py` (déjà
  canonique).
- ❌ Migration des métriques `measurements/*.py` (Sprint E).
- ❌ Migration des routes web (Sprint F).
- ❌ Migration des commandes CLI (Sprint G).
- ❌ Suppression de `OCRLLMPipeline` (Sprint D.6 inclura sa
  suppression car `pipelines/_executor_runner.py` n'aura plus
  d'utilité — mais c'est un effet de bord, pas l'objectif).

---

**Document de référence** pour le Sprint D.  Toute déviation du
plan ci-dessus doit être documentée en commit message
`docs(sprint-D): ajustement <raison>`.
