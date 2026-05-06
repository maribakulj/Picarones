# Tolérances de régression — legacy ↔ rewrite

> **Audience** : développeur qui migre une fonctionnalité legacy
> vers le rewrite, reviewer qui relit la PR.
>
> **Référence** : [`legacy-retirement-plan.md`](legacy-retirement-plan.md).
>
> **Contrat** : le harness `tests/regression/legacy_vs_rewrite/`
> exécute legacy + rewrite sur les mêmes corpus de référence et
> compare leurs sorties.  Toute divergence au-delà de la tolérance
> ε définie ici est une **régression à corriger avant merge**.
>
> Une régression peut être :
>
> - **Intentionnelle** : la phase de migration corrige un bug
>   historique → la tolérance est temporairement relâchée AVEC
>   commentaire pointant vers l'issue.
> - **Inattendue** : c'est ce que ce document est censé empêcher.

## Principe général

Pour une fonctionnalité donnée, la sortie du rewrite **doit être
égale** à celle du legacy à la tolérance ε près.  L'égalité est :

- **Bit-for-bit** quand l'output est déterministe (texte, hash, JSON).
- **Sémantique** quand l'output structurel a des libertés (ordre des
  éléments d'un set, indentation HTML, ordre des facts narratifs
  équivalents).

## Table des tolérances par type d'output

### Métriques numériques

| Métrique | ε | Justification |
|----------|---|---------------|
| `cer_raw`, `cer_nfc`, `cer_caseless`, `cer_diplomatic` | **0** (bit-for-bit) | jiwer est déterministe ; toute différence = changement de pré/post-processing |
| `wer`, `mer`, `wil` | **0** | idem |
| `bleu`, `chrf` | **1e-9** | flottants — réordonnancements internes acceptables |
| `precision`, `recall`, `f1` (NER) | **1e-9** | flottants |
| `mufi_coverage`, `abbreviation_expansion_score` | **0** | comptage entier sur ensembles fermés |
| `roman_numerals_accuracy` | **0** | parsing déterministe |
| `unicode_blocks_accuracy` | **0** | tables Unicode déterministes |
| `reading_order_f1` (ICDAR 2015) | **1e-9** | algorithme déterministe, flottants |
| `layout_f1` | **1e-9** | flottants |
| `confusion_matrix.entries` | **0** | comptage entier |
| `taxonomy.error_class_*` | **0** | classification déterministe sur règles |

### Tests statistiques

| Test | ε | Justification |
|------|---|---------------|
| Wilcoxon `p_value` | **1e-9** | scipy `wilcoxon` est déterministe à entrée constante |
| Friedman `chi2`, `p_value` | **1e-9** | idem |
| Nemenyi (matrice p-values) | **1e-9** | dérivé de Friedman |
| Bootstrap CI 95 % | **1e-3** | random seed FIXÉ explicitement (cf. `bootstrap.py` du legacy : `seed=42`) ; la tolérance laisse une marge minuscule pour les ré-implémentations qui itéreraient dans un ordre différent à seed identique |
| Pareto front (set d'engines dominants) | **0** (bit-for-bit en tant qu'ensemble) | dominance Pareto stable sur entrées identiques |
| CDD (Critical Difference Diagram) coordonnées SVG | **1e-3** sur les positions (px) | rendu Matplotlib peut varier sur des sub-pixels selon backend |
| Clustering (labels) | **0** sur l'**ensemble** des classes (l'étiquetage interne 0/1/2 peut différer mais la partition doit être identique) | un test custom compare les partitions, pas les labels |
| Corrélation Spearman / Pearson | **1e-9** | flottants |

### Calibration

| Output | ε | Justification |
|--------|---|---------------|
| ECE, MCE | **1e-9** | flottants, pas d'aléatoire |
| Reliability diagram (bins, freq, conf) | **0** sur les bins, **1e-9** sur les valeurs | binning déterministe |

### Confidences sidecar (S50 sur Tesseract)

| Output | ε |
|--------|---|
| `tokens[].text` | **0** (string identique) |
| `tokens[].confidence` | **0** | Tesseract retourne un entier 0-100 ; division exacte par 100 → flottant binairement identique en IEEE-754 |
| `extractor`, `model_version` | **0** |

### HTML (rapport `reports_v2/html/render.py`)

Le diff HTML est **structurel**, pas lexical :

- Mêmes éléments DOM avec mêmes attributs sémantiques (`data-*`,
  `aria-*`, `id`, `class`).
- Mêmes valeurs textuelles dans les nœuds de texte.
- L'**ordre** des sections doit être identique.
- L'indentation et le whitespace inter-éléments sont **ignorés**.
- Le contenu d'un `<script>` est comparé après normalisation
  d'espace blanc.

Implémenté via une fonction `assert_html_semantically_equal(a, b)`
qui parse les deux HTML avec `lxml` (ou `html.parser` fallback) et
compare l'arbre.

### CSV (`reports_v2/csv/render.py`)

| Output | ε |
|--------|---|
| Header row | **0** (identique exact) |
| Data rows (set non ordonné) | **0** sur l'ensemble |
| Ordre des lignes | autorisé à différer | les renderers triaient parfois différemment ; seule l'égalité ensembliste est exigée |
| Format des nombres | **0** (le rewrite formate à 6 décimales `f"{v:.6f}"`) | déterministe |

### JSON (`reports_v2/json/render.py`)

| Output | ε |
|--------|---|
| Bit-for-bit identique | **0** | le rewrite utilise `model_dump(mode="json")` Pydantic + `json.dumps(sort_keys=True, indent=2, ensure_ascii=False)` ; le legacy doit être amené au même contrat dans la phase concernée |

### Narrative facts (Phase 3)

| Aspect | ε |
|--------|---|
| Ensemble des `Fact` produits (par `FactType`) | **0** sur l'ensemble | l'arbitre peut réordonner mais pas inventer ni rater un fact |
| Payload de chaque fact (les valeurs numériques citées) | **0** (bit-for-bit) | garde-fou anti-hallucination |
| Templates rendus FR + EN | **0** sur le texte | déterministe par `str.format_map` |
| Ordre final des facts dans la synthèse | **autorisé à différer** | l'arbitre du rewrite peut choisir un ordre différent si la priorité est respectée — un test custom valide « les facts HIGH apparaissent avant les MEDIUM » plutôt que l'ordre exact |

### Rapport HTML — sections legacy spécifiques (Phase 5)

Pour chaque renderer migré (calibration, NER, Pareto, narrative,
philological, etc.), un cas-test de régression dédié vit dans
`tests/regression/legacy_vs_rewrite/test_phase5_<renderer>.py`.
Le snapshot legacy est figé en début de phase.

## Aléatoire — politique

Tout module qui utilise `random` doit :

1. Accepter un argument `seed: int` ou utiliser une seed fixée
   explicitement.
2. Documenter la seed dans son docstring.
3. Le harness de régression utilise toujours **seed=42**.

Modules concernés au legacy :

- `measurements/statistics/bootstrap.py` (seed=42)
- `measurements/runner/workers.py` (pas d'aléatoire — confirmé)
- `core/results.py` (pas d'aléatoire — confirmé)

## Adaptateurs cloud (Mistral, OpenAI, Anthropic, Google, Azure)

Les appels réseau ne sont **pas** rejoués pendant la régression —
le test serait non-déterministe et coûteux.  Stratégie :

1. Le harness utilise des **fixtures de réponses figées** (JSON
   capturé en local lors de la création du corpus de référence).
2. Le legacy et le rewrite reçoivent **la même fixture** ; le test
   vérifie que tous deux produisent le même output structurel.
3. Si une dépendance SDK change la sérialisation (rare), le test
   pète bruyamment et la PR doit re-frigorifier la fixture.

Aucune tolérance non triviale n'est nécessaire — l'égalité
bit-for-bit est tenable parce que l'aléatoire vient du cloud, pas
du parser.

## Procédure d'exception (régression intentionnelle)

Quand une migration corrige un bug historique légitime :

1. Ouvrir une issue GitHub avec le label `regression-intentional`.
2. Référencer le numéro d'issue dans le commit qui modifie la
   tolérance.
3. Ajouter une entrée dans la section *« Régressions intentionnelles
   acceptées »* ci-dessous, **avant** le merge.
4. La tolérance peut être relâchée temporairement ; au merge, soit
   le snapshot legacy est mis à jour pour refléter le nouveau
   comportement (correct), soit la tolérance reste serrée pour les
   prochaines migrations.

## Régressions intentionnelles acceptées

| Date | Issue | Phase | Module | Description |
|------|-------|-------|--------|-------------|
| (aucune à ce jour) |  |  |  |  |

## Révisions

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-05 | Création initiale (Phase 0 du plan de retrait legacy) |
