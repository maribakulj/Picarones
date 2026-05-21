# Profils de calcul

Picarones expose **7 profils de calcul** qui modulent les métriques
calculées par le runner selon le use case. Chaque profil active un
sous-ensemble des **12 hooks document-level** et **12 agrégateurs
corpus-level** du registre central
([`picarones/evaluation/metric_hooks.py`](../picarones/evaluation/metric_hooks.py)).

## Synoptique

| Profil | Hooks doc | Cible | CLI dédiée |
|---|---|---|---|
| `minimal` | 0 | Tests rapides, bench massif (CER/WER seuls) | — |
| `standard` (défaut) | 12 | Comportement historique, rétrocompat | `picarones run` |
| `philological` | 12 | Édition critique, paléographie | `picarones edition` |
| `diagnostics` | 12 | Comprendre **pourquoi** un moteur produit ses résultats | `picarones diagnose` |
| `economics` | 12 | Décision budget : pages/h utilisable, coût marginal | `picarones economics` |
| `pipeline` | 12 | Bench de pipelines composées (axe B) | `picarones pipeline run` |
| `full` | 12 | R&D — tout ce qui peut être calculé | — |

> **Note rétrocompat** : aujourd'hui les profils `philological`, `diagnostics`,
> `economics`, `pipeline` et `full` activent **le même ensemble** que `standard`
> côté hooks calculés. Ce qui change, c'est la **vue HTML rendue** : chaque
> profil active des sous-sections différentes du rapport (cf. `docs/reference/views.md`).
> Les profils sont volontairement génériques pour permettre aux contributeurs
> futurs d'ajouter des hooks spécifiques sans casser l'API.

## Détail des hooks par profil

### `minimal`

Aucun hook. Le runner calcule uniquement les métriques de base
(`cer`, `wer`, `mer`, `wil` via `compute_metrics`). Les `DocumentResult`
n'ont **aucun champ** `confusion_matrix`, `taxonomy`, `calibration`, etc.

**Cas d'usage** : bench de régression sur 10 000 documents pour mesurer
le CER d'une mise à jour de modèle, sans avoir besoin du diagnostic
philologique. Réduit drastiquement le temps de calcul (∼5× plus rapide).

```bash
picarones run --corpus ./corpus --profile minimal
```

### `standard` (défaut)

Active les 12 hooks document-level historiques :

| Hook | Sprint | Champ DocumentResult | Notes |
|---|---|---|---|
| `confusion` | 5 | `confusion_matrix` | Matrice unicode |
| `char_scores` | 5 | `char_scores` | Ligatures + diacritiques |
| `taxonomy` | 5 | `taxonomy` | 9 classes d'erreurs |
| `structure` | 5 | `structure` | Lignes / blocs / mots |
| `image_quality` | 5 | `image_quality` | Contraste, bruit, flou… |
| `line_metrics` | 10 | `line_metrics` | Distribution CER + Gini |
| `hallucination` | 10 | `hallucination_metrics` | Détection VLM |
| `calibration` | 42 | `calibration_metrics` | ECE/MCE (si confidences) |
| `philological` | 61 | `philological_metrics` | 6 modules philologiques |
| `searchability` | 86 | `searchability_metrics` | Fuzzy recall |
| `numerical_sequences` | 86 | `numerical_sequence_metrics` | Repérage et alignement de séquences numériques |
| `readability` | 87 | `readability_metrics` | Δ Flesch |

12 agrégateurs corpus-level correspondants remplissent les attributs
`aggregated_*` de chaque `EngineReport`.

### `philological` (édition critique)

Active les 12 hooks `standard` + déclenchera (chantiers futurs)
l'enrichissement par des hooks dédiés à la paléographie médiévale,
l'imprimé ancien et les archives modernes (déjà calculés via le module
`philological` du `standard`, mais valorisés différemment dans la vue
HTML « Taxonomie avancée »).

```bash
picarones edition --corpus ./manuscrits --engines tesseract,pero_ocr
```

Vue HTML supplémentaire activée : `view_advanced_taxonomy.html` avec
diagramme miroir leader vs runner-up, classes par récupérabilité
éditoriale.

### `diagnostics` (analyse approfondie)

Identique à `standard` côté hooks. Déclenche la vue HTML « Diagnostic
approfondi » avec **leviers d'amélioration** factuels (jamais
prescriptifs) calculés par `picarones.domain.levers.detect_levers()`.

```bash
picarones diagnose --corpus ./corpus --engines tess,pero
```

Vue HTML supplémentaire : `view_diagnostics.html` avec :

- Leviers d'amélioration (`dominant_recoverable_class`,
  `pareto_concentration`, etc.).
- Profil d'image du corpus (complexité paléographique + homogénéité,
  opt-in).
- Comparaison historique (baseline, opt-in si historique SQLite).

### `economics` (décision budget)

Active la vue HTML « Coût et performance » avec **throughput effectif**
calculé selon la formule HTR-United (5 s/erreur de correction humaine).
Discrimine entre un moteur cloud rapide mais imprécis (drag élevé) et
un local lent mais fiable.

```bash
picarones economics --corpus ./corpus --engines mistral_ocr,tesseract
```

### `pipeline` (axe B)

Profil utilisé par `picarones pipeline run` pour les benchmarks de
pipelines composées (Sprints 32-34, 53-54, 63-68, 94-97 + chantier 1).
La vue HTML produite contient le DAG visuel de la pipeline,
l'absorption d'erreur par jonction, et la comparaison incrémentale par
slot.

```bash
picarones pipeline run examples/pipelines/ocr_to_alto.yaml --corpus DIR
```

### `full`

Active **tous** les hooks et toutes les vues. Coût maximal mais
reproductibilité scientifique maximale.

## Comment ajouter un hook personnalisé

Voir [`docs/explanation/narrative-engine.md`](developer/narrative-engine.md)
pour le détail. Pattern de base :

```python
from picarones.evaluation.metric_hooks import (
    register_document_metric, PROFILE_DIAGNOSTICS, PROFILE_FULL,
)

@register_document_metric(
    name="my_metric",
    attribute="my_metric",       # nom du champ DocumentResult
    profiles=(PROFILE_DIAGNOSTICS, PROFILE_FULL),
    requires_success=True,
)
def my_hook(*, ground_truth, hypothesis, image_path, corpus_lang, ocr_result):
    from my_module import compute_my_metric
    return compute_my_metric(ground_truth, hypothesis)
```

## Code source

- [`picarones/evaluation/metric_hooks.py`](../picarones/evaluation/metric_hooks.py)
  — registre, profils, `run_document_hooks()`, `run_corpus_aggregators()`.
- [`picarones/evaluation/metrics/builtin_hooks.py`](../picarones/evaluation/metrics/builtin_hooks.py)
  — les 12 hooks doc + 12 agrégateurs natifs Picarones.
- [`tests/test_metric_hooks.py`](../tests/test_metric_hooks.py)
  — tests unitaires + rétrocompat profil `standard`.
