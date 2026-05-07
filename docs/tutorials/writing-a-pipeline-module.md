# Écrire un module pour le banc d'essai de pipelines

> **Public visé** : chercheurs, ingénieurs, équipes patrimoniales
> qui veulent **évaluer leurs propres modules** (correcteur LLM,
> reconstructeur ALTO, classifieur d'entités, re-segmenteur…)
> dans Picarones.

> **Ce que Picarones est et ce qu'il n'est pas**
>
> Picarones est un **banc d'essai**, pas un atelier de production.
> Il fournit l'infrastructure pour exécuter, mesurer et comparer
> vos modules sur un corpus avec sa GT — il ne fournit **aucun
> module métier** (pas de reconstructeur ALTO « maison », pas de
> correcteur LLM intégré, pas de re-segmenteur).  C'est à vous
> d'amener vos modules ; Picarones les juge.

## TL;DR

```python
from picarones.domain.artifacts import ArtifactType
from picarones.domain.module_protocol import BaseModule
from picarones.evaluation.pipeline import (
    PipelineRunner, PipelineSpec, PipelineStep,
)

class MyCorrector(BaseModule):
    input_types = (ArtifactType.TEXT,)
    output_types = (ArtifactType.TEXT,)
    execution_mode = "io"          # ou "cpu"

    @property
    def name(self) -> str:
        return "my-corrector-v1"

    def process(self, inputs):
        text = inputs[ArtifactType.TEXT]
        # … votre logique : appel LLM, regex, modèle local, etc.
        return {ArtifactType.TEXT: text.replace("teh", "the")}

spec = PipelineSpec(
    name="ocr_then_corrector",
    steps=[
        PipelineStep("ocr",  my_ocr_module),       # votre module OCR
        PipelineStep("fix",  MyCorrector()),
    ],
)
result = PipelineRunner.run(
    spec, document, {ArtifactType.IMAGE: "/path/to/img.png"},
)
print(result.junction_metrics_for(ArtifactType.TEXT))
# → {"cer": 0.05, "wer": 0.12, ...}
```

## 1. Le contrat `BaseModule`

Un module Picarones est une classe qui hérite de `BaseModule` et
déclare quatre choses :

| Champ            | Rôle                                                   | Exemple                                |
| ---------------- | ------------------------------------------------------ | -------------------------------------- |
| `input_types`    | Tuple des types d'artefacts consommés                  | `(ArtifactType.TEXT,)`                 |
| `output_types`   | Tuple des types d'artefacts produits                   | `(ArtifactType.TEXT,)`                 |
| `execution_mode` | `"io"` (réseau, disque) ou `"cpu"` (calcul intensif)   | `"io"` pour un appel LLM cloud         |
| `name`           | Identifiant lisible pour le rapport et les logs        | `"my-corrector-v1"`                    |
| `process`        | La méthode qui transforme un dict d'inputs en outputs  | voir TL;DR                             |

Les `ArtifactType` disponibles aujourd'hui :

- `IMAGE` — typiquement chemin vers le fichier image
- `TEXT` — chaîne de caractères (transcription)
- `ALTO` / `PAGE` — XML structurel (objets `AltoGT` / `PageGT`)
- `ENTITIES` — liste d'entités nommées
- `READING_ORDER` — ordre de lecture des régions

## 2. Exemples pédagogiques (à NE PAS copier en production)

> Ces exemples sont **mockés** — leur seul rôle est d'illustrer
> le contrat.  Pour évaluer un vrai module, vous écrirez votre
> propre classe qui appelle votre vraie logique.

### 2.a Correcteur LLM TEXT → TEXT

```python
class LLMCorrector(BaseModule):
    """Mock pédagogique d'un correcteur LLM."""
    input_types = (ArtifactType.TEXT,)
    output_types = (ArtifactType.TEXT,)
    execution_mode = "io"

    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def name(self) -> str:
        return f"llm-correcteur-{self._model}"

    def process(self, inputs):
        text = inputs[ArtifactType.TEXT]
        # Production : appel à votre LLM ici (OpenAI, Mistral, …)
        # corrected = my_llm_client.correct(text, model=self._model)
        corrected = text  # mock pédagogique : passthrough
        return {ArtifactType.TEXT: corrected}
```

### 2.b Reconstructeur TEXT → ALTO (mock)

```python
class TextToAltoReconstructor(BaseModule):
    """Mock pédagogique d'un reconstructeur ALTO depuis du texte.

    En production : votre code reconstruit la structure XML ALTO
    en plaçant chaque mot dans une boîte selon une heuristique
    (longueur de mot, hauteur de ligne, …).
    """
    input_types = (ArtifactType.TEXT,)
    output_types = (ArtifactType.ALTO,)
    execution_mode = "cpu"

    @property
    def name(self) -> str:
        return "text-to-alto-mock"

    def process(self, inputs):
        text = inputs[ArtifactType.TEXT]
        # Production : votre logique de reconstruction
        alto_payload = my_reconstruct(text)
        return {ArtifactType.ALTO: alto_payload}
```

### 2.c Classifieur TEXT → ENTITIES

```python
class NERExtractor(BaseModule):
    input_types = (ArtifactType.TEXT,)
    output_types = (ArtifactType.ENTITIES,)
    execution_mode = "cpu"

    @property
    def name(self) -> str:
        return "ner-extractor-spacy-fr"

    def process(self, inputs):
        text = inputs[ArtifactType.TEXT]
        # Production : votre extracteur (spaCy, HuggingFace, HIPE, …)
        entities = my_ner(text)  # liste de dicts {label, start, end, text}
        return {ArtifactType.ENTITIES: entities}
```

## 3. Orchestrer une pipeline

### 3.a Mono-document (Sprint 63)

```python
from picarones.evaluation.pipeline import (
    PipelineRunner, PipelineSpec, PipelineStep,
)

spec = PipelineSpec(
    name="ocr_then_correct",
    steps=[
        PipelineStep("ocr",     my_ocr_module),
        PipelineStep("correct", LLMCorrector(model="my-model")),
    ],
)
result = PipelineRunner.run(
    spec, document, {ArtifactType.IMAGE: document.image_path},
)
print(result.succeeded)                                   # True / False
print(result.junction_metrics_for(ArtifactType.TEXT))     # CER, WER…
print(result.failing_steps)                               # noms des étapes en erreur
```

À chaque sortie d'étape, Picarones évalue **automatiquement**
l'artefact contre la GT du même niveau (via `compute_at_junction`,
Sprint 34).  Vous n'avez rien à câbler explicitement — il suffit
que `Document.ground_truths` porte une `TextGT` (ou `AltoGT`,
`EntitiesGT`…) au niveau correspondant.

### 3.b Corpus complet (Sprint 64)

```python
from picarones.evaluation.pipeline_benchmark import run_pipeline_benchmark

bench = run_pipeline_benchmark(spec, my_corpus)
print(bench.n_pipelines_succeeded, "/", bench.n_docs)

agg = bench.aggregate_for_step("correct")
print(agg.duration_seconds_mean)                          # durée moyenne
print(agg.junction_metrics["text"]["cer"]["median"])      # CER médian
print(agg.error_breakdown)                                # types d'erreur
```

Pour les pipelines qui ne démarrent pas par `IMAGE` (par exemple
un re-segmenteur ALTO qui démarre depuis un ALTO pré-existant),
vous fournissez votre propre factory :

```python
def my_factory(doc):
    return {ArtifactType.ALTO: my_load_alto(doc)}

bench = run_pipeline_benchmark(spec, corpus, initial_inputs_factory=my_factory)
```

### 3.c Comparer N pipelines (Sprint 65)

```python
from picarones.evaluation.pipeline_comparison import compare_pipelines

comparison = compare_pipelines(
    [spec_baseline, spec_with_correcteur_a, spec_with_correcteur_b],
    corpus,
)

# Classement par CER (plus bas = meilleur)
for name, cer in comparison.ranking_by_final_metric(
    ArtifactType.TEXT, "cer",
):
    print(f"{name}: {cer:.4f}" if cer else f"{name}: N/A")

# Gain vs baseline
gains = comparison.gain_table(
    ArtifactType.TEXT, "cer", baseline_pipeline="baseline",
)
```

### 3.d DAG branchant via `inputs_from` (Sprint 66)

Quand plusieurs étapes produisent le même type, la sortie de la
plus récente écrase les précédentes par défaut.  Pour comparer
deux corrections d'un même OCR dans une **seule** pipeline, vous
désignez explicitement l'étape source :

```python
spec = PipelineSpec(
    name="fork",
    steps=[
        PipelineStep("ocr", MyOCR()),
        PipelineStep(
            "correct_a", CorrecteurA(),
            inputs_from={ArtifactType.TEXT: "ocr"},     # depuis OCR
        ),
        PipelineStep(
            "correct_b", CorrecteurB(),
            inputs_from={ArtifactType.TEXT: "ocr"},     # depuis OCR aussi
        ),
    ],
)
```

Sans `inputs_from`, `correct_b` aurait reçu la sortie de
`correct_a` (chaîne).

> Pour comparer **plusieurs pipelines distinctes** apple-to-apple,
> préférez `compare_pipelines` (Sprint 65) — c'est plus clair et
> ça produit un rapport HTML dédié (Sprint 68).

## 4. Générer un rapport HTML autonome

### 4.a Pipeline unique (Sprint 67)

```python
from pathlib import Path
from picarones.reports_v2.html.renderers.pipeline import build_pipeline_report_html

bench = run_pipeline_benchmark(spec, corpus)
Path("rapport_pipeline.html").write_text(
    build_pipeline_report_html(bench, lang="fr"),
)
```

### 4.b Comparaison de N pipelines (Sprint 68)

```python
from picarones.domain.artifacts import ArtifactType
from picarones.reports_v2.html.renderers.pipeline import (
    RankingSpec, build_pipeline_comparison_report_html,
)

html = build_pipeline_comparison_report_html(
    comparison,
    ranking_specs=[
        RankingSpec(ArtifactType.TEXT, "cer", label="CER"),
        RankingSpec(ArtifactType.TEXT, "wer", label="WER"),
    ],
    baseline_pipeline="baseline",
    lang="fr",
)
Path("comparaison.html").write_text(html)
```

L'utilisateur déclare **explicitement** ce qu'il veut voir : les
classements (`ranking_specs`) et la baseline éventuelle.  Pas
d'auto-détection magique — vous pilotez.

## 5. Bonnes pratiques

### 5.a Discipline des types

- Un module qui consomme `TEXT` doit accepter une chaîne, pas un
  `TextGT`.  Picarones extrait automatiquement le payload d'une
  GT typée avant de l'évaluer ; mais à l'intérieur d'un module,
  on travaille avec les types « bruts » :
  - `TEXT`     → `str`
  - `ENTITIES` → `list[dict]`
  - `ALTO` / `PAGE` → objet structuré (à vous de choisir le
    schéma — Picarones n'impose pas)
- Si votre module produit un type différent de ses inputs (par
  ex. `TEXT` → `ALTO`), déclarez-le dans `output_types`.  Le runner
  validera automatiquement et évaluera contre la `AltoGT` si elle
  existe.

### 5.b Erreurs gracieuses

Un module qui lève une exception **n'arrête pas** la pipeline :
le runner capture l'exception, marque l'étape en erreur, et
continue avec les étapes suivantes (qui rapporteront « entrée
manquante » si elles dépendaient de la sortie de l'étape échouée).

Vous n'avez pas besoin de capturer vos propres exceptions — laissez
Picarones le faire pour bénéficier de la trace dans
`StepResult.error`.

### 5.c Mesure du temps

Picarones chronomètre **wall-clock** chaque appel `process`.  Si
votre module fait du caching interne ou du batching, c'est sa
durée réelle vue par l'utilisateur qui est mesurée — c'est ce
qu'on veut pour comparer fairement.

### 5.d Pas de seuils éditoriaux dans votre module

Si votre module classe en interne (ex. « ce texte semble
diplomatique »), ne reportez pas ce verdict dans Picarones — c'est
au chercheur qui lit le rapport de juger selon ses critères
éditoriaux.  Votre module produit un artefact, Picarones mesure
l'écart à la GT, le chercheur conclut.

## 6. Anti-patterns

### 6.a « Et si Picarones avait un correcteur LLM intégré ? »

Non.  Picarones est un **banc d'essai** : si on intégrait un
correcteur, on devrait le maintenir, le faire évoluer, le calibrer,
le documenter — et au final on **biaiserait** les benchmarks (parce
qu'on connaîtrait mieux notre correcteur que les autres).

À la place, vous écrivez votre `BaseModule` qui wrappe le
correcteur que vous voulez évaluer.  Picarones se contente de le
brancher dans la pipeline et de mesurer.

### 6.b « Et si je veux juste tester une pipeline OCR seule, sans étapes en aval ? »

C'est exactement ce que fait le runner OCR historique
(`run_benchmark` dans `picarones/measurements/runner/`) — il est
toujours là, n'a pas changé, et reste la voie recommandée pour
les benchmarks d'OCR mono-étage.

L'axe B (pipelines composées) sert quand vous avez **plusieurs
modules tiers** à enchaîner ou à comparer.

### 6.c « Mon module a besoin d'un état mutable entre documents »

Possible mais à vos risques.  Les `BaseModule` sont instanciés
une fois et leur méthode `process` est appelée pour chaque
document du corpus.  Si vous gardez de l'état dans `self`, il
persistera — utile pour un cache, dangereux si vous ne savez pas
ce que vous faites.

Pour la parallélisation future (à arbitrer dans un sprint dédié),
mieux vaut concevoir vos modules **stateless**.

## 7. Référence rapide des sprints axe B

| Sprint | Sujet                                            |
| ------ | ------------------------------------------------ |
| 32     | GT multi-niveaux (`Document.ground_truths`)      |
| 33     | Interface `BaseModule` + `ArtifactType`          |
| 34     | Registre typé de métriques (`compute_at_junction`) |
| 63     | `PipelineRunner` mono-document                   |
| 64     | `run_pipeline_benchmark` corpus-wide             |
| 65     | `compare_pipelines` apple-to-apple               |
| 66     | DAG branchant via `inputs_from`                  |
| 67     | `build_pipeline_report_html` autonome            |
| 68     | `build_pipeline_comparison_report_html`          |
| 69     | Ce guide                                         |
