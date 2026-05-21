# Équivalence numérique — ancien runner ↔ nouveau pipeline executor

Ce document décrit comment le `CorpusRunner` introduit au Sprint S8
(combiné au `PipelineExecutor` du S7) reproduit les mêmes chiffres
CER/WER que l'ancien `picarones.measurements.runner.run_benchmark`.

C'est le **critère go/no-go de fin de Phase 2** du rewrite ciblé
(cf. `docs/roadmap/rewrite-2026.md`).  Sans cette équivalence, on
ne peut pas basculer la BnF vers le nouveau runner sans surprise.

## Architecture des deux orchestrations

### Ancien runner (`picarones.measurements.runner`)

```
Corpus[Document(image, GT)]
     │
     ▼
run_benchmark(corpus, [BaseOCREngine])
     │
     ▼ ProcessPoolExecutor / ThreadPoolExecutor
BaseOCREngine.run(image)  →  EngineResult(text, ...)
     │
     ▼
compute_metrics(GT, text)  →  MetricsResult(cer, wer, ...)
     │
     ▼
aggregate_metrics([MetricsResult, ...])  →  {"cer": {"mean": 0.05}, ...}
     │
     ▼
EngineReport(mean_cer=0.05, ...)
```

### Nouveau pipeline (`picarones.pipeline`)

```
[DocumentRef], initial_inputs={IMAGE: Artifact}
     │
     ▼
CorpusRunner.run(spec, docs, factory_inputs, factory_ctx)
     │
     ▼ ThreadPoolExecutor avec backpressure
PipelineExecutor.run(spec, doc, inputs, ctx)
     │
     ▼ pour chaque step
StepExecutor.execute(inputs, params, ctx)  →  {RAW_TEXT: Artifact}
     │
     ▼ (S13+ : EvaluationViewExecutor)
TextView.evaluate(candidate, ground_truth)  →  ViewResult(metric_values)
```

Le S12 ne livre pas encore l'`EvaluationViewExecutor` — il vérifie
juste que **si on appelle ``compute_metrics`` directement sur les
artefacts produits par le nouveau pipeline**, on obtient les mêmes
valeurs.  Le S13-S14 livrera la couche `TextView` qui fera ce
calcul automatiquement.

## Méthode de vérification (test d'équivalence)

Le test `tests/integration/test_sprint_a14_s12_executor_equivalence.py`
implémente l'équivalence :

1. **Construit deux orchestrations** consommant exactement le même
   corpus :
   - `_FakeOCREngine` (héritant de `BaseOCREngine`) pour l'ancien
     runner.
   - `_FakeStepExecutor` (satisfaisant le protocole `StepExecutor`)
     pour le nouveau.
   - Les deux retournent **le même texte** par document, indexé par
     `doc_id`.

2. **Lance les deux runners** sur le même corpus.

3. **Calcule CER/WER avec le même `compute_metrics`** sur les
   sorties des deux runners.

4. **Compare** les moyennes CER et WER.

## Tolérance : 1e-6, pas 1e-9

Le plan d'origine prévoyait une tolérance de **1e-9** ("équivalence
numérique stricte").  La réalité du code montre une divergence de
l'ordre de **1e-7** sur certaines fixtures, **uniquement à cause
d'un arrondi à 6 décimales** dans `aggregate_metrics` de l'ancien
runner :

```python
# picarones/core/metrics.py — _stats()
return {
    "mean": round(statistics.mean(values), 6),
    "median": round(statistics.median(values), 6),
    ...
}
```

Les valeurs brutes (avant `round`) sont identiques bit-à-bit
entre les deux runners.  La divergence observée provient
strictement du `round(..., 6)`.

Le test S12 utilise donc une tolérance **1e-6** (cohérente avec les
6 décimales d'arrondi) et documente cette décision.  Quand
l'agrégation finale passera par les types non-arrondis du nouveau
code (S22), la tolérance pourra être resserrée à 1e-9.

## 5 fixtures patrimoniales testées

Le test couvre 5 cas de difficulté croissante :

| Fixture | Description |
|---|---|
| `fixture_1_court` | Mots isolés, hypothèse parfaite |
| `fixture_2_paragraphe` | Phrases avec une coquille |
| `fixture_3_multi_lignes` | Multi-lignes + accents perdus |
| `fixture_4_abreviations` | Bibliographie + date erronée |
| `fixture_5_mix_langues` | Latin + français, multiples coquilles |

Plus deux cas limites :

- `test_equivalence_with_perfect_hypothesis` — CER == WER == 0
- `test_equivalence_with_empty_hypothesis` — texte produit vide

Total : **7 tests d'équivalence**, tous verts.

## Conséquences pour la migration BnF

À partir du S12, on peut affirmer que :

- Basculer un benchmark BnF du runner legacy vers le nouveau
  `CorpusRunner` ne change pas les chiffres rapportés au-delà de
  l'arrondi à 6 décimales.
- Les rapports HTML produits depuis le nouveau pipeline (S22)
  afficheront les mêmes CER que les rapports historiques (modulo
  arrondi).
- Le nouveau `CorpusRunner` apporte **trois améliorations** non
  visibles côté chiffres :
  1. Backpressure (RAM bornée même sur 1000+ docs).
  2. Timeout depuis le **début d'exécution** (pas la queue).
  3. Annulation propre via `threading.Event`.

## Limites du S12

L'équivalence vérifiée ici porte uniquement sur :

- Le pipeline OCR seul (un step → un texte → CER/WER).
- Les métriques principales `mean_cer` / `mean_wer`.

Restent à vérifier dans des sprints suivants :

- **S13** : équivalence des projecteurs (ALTO → texte) — couvert
  par les tests unitaires de `formats.alto.projector` mais pas
  encore comparé à `extract_text_from_alto` legacy.
- **S15** : équivalence des métriques structurelles (Layout F1,
  reading order F1) — non testées en S12 car elles vivent dans
  des fichiers `measurements/*.py` non encore migrés.
- **S20** : équivalence des métriques philologiques (MUFI,
  abbreviations, etc.) — idem.

Quand ces sprints ajouteront leurs tests d'équivalence, le critère
"équivalence numérique fin Phase 3 / Phase 4" sera complet.

## Statut

- **Fin de Phase 2 (S12)** — équivalence runner OCR ✅
- **Fin de Phase 3 (S18)** — équivalence views ouverte (S13-S18)
- **Fin de Phase 4 (S22)** — équivalence rapport HTML ouverte
