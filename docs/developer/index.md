# Documentation développeur Picarones

Guides courts pour étendre Picarones sans casser les invariants
fondamentaux du projet.

## Architecture

Voir [CLAUDE.md](../../CLAUDE.md) et
[`docs/explanation/architecture.md`](../explanation/architecture.md)
pour la cartographie complète.  En résumé : architecture **8
couches concentriques** (post-rewrite, canonique) :

```
picarones/
├── domain/              # Layer 1 — types purs (Pydantic, stdlib only)
│   ├── artifacts.py     # Artifact, ArtifactType (10 types)
│   ├── corpus.py        # CorpusSpec
│   ├── documents.py     # DocumentRef
│   ├── pipeline_spec.py # PipelineSpec, PipelineStep (Pydantic immutable)
│   ├── module_protocol.py # BaseModule (ABC, en cours de retrait au profit de StepExecutor)
│   ├── facts.py         # Fact, FactType, registre narratif
│   └── …
├── formats/             # Layer 2 — parsing/serialization (ALTO 4, PAGE XML, JSON)
├── evaluation/          # Layer 3 — métriques et calcul
│   ├── metrics/         # ~30 métriques (CER/WER, MUFI, philological, NER, …)
│   ├── statistics/      # Wilcoxon, Friedman/Nemenyi, bootstrap, Pareto
│   ├── views/, projectors/  # EvaluationView (S13+), projecteurs Alto/Page/CanonicalToText
│   ├── corpus.py        # Document, Corpus, GTLevel (legacy en cours de retrait)
│   ├── pipeline.py      # PipelineRunner legacy (en cours de retrait)
│   └── benchmark_result.py # BenchmarkResult, EngineReport, DocumentResult
├── pipeline/            # Layer 4 — PipelineExecutor canonique (instance-based)
├── adapters/            # Layer 5 — adapters externes (libs externes autorisées)
│   ├── ocr/             # Tesseract, Pero, Mistral OCR, Google Vision, Azure DI
│   ├── llm/             # OpenAI, Anthropic, Mistral, Ollama
│   ├── vlm/             # Adapters VLM (zero-shot)
│   ├── corpus/          # IIIF, Gallica, HTR-United, HuggingFace
│   ├── storage/         # ArtifactStore, JobStore
│   └── legacy_engines/, legacy_modules/  # legacy BaseModule-based, en retrait
├── app/                 # Layer 6 — services applicatifs (BenchmarkService, …)
├── reports_v2/          # Layer 7 — rendu HTML / JSON / CSV (22 renderers + 5 vues)
└── interfaces/          # Layer 8 — CLI Click, Web FastAPI

# Arborescence legacy en cours de retrait (cf. docs/migration/) :
# core/, measurements/, engines/, llm/, pipelines/, report/, modules/
```

Règle d'import stricte : les flèches d'import vont uniquement
de l'extérieur vers l'intérieur (de bas en haut dans le diagramme).
Vérifié par `tests/architecture/test_layer_dependencies.py`.

## Guides d'extension

- [Étendre le moteur narratif](narrative-engine.md) — ajouter un type
  de fait, ses templates, l'enregistrer dans le registre.
- [Étendre le glossaire](extending-glossary.md) — documenter une
  nouvelle métrique, l'attacher à une colonne.
- [Étendre l'i18n](extending-i18n.md) — ajouter une nouvelle langue
  ou une clé d'interface.

## Invariants à respecter

1. **Pas de LLM dans le chemin critique** du rapport. La synthèse
   factuelle est rendue par des templates `str.format_map`. Tout LLM
   au moment de la génération est à proscrire (reproductibilité,
   coût, dépendance externe).
2. **Pas de prescription dans l'interface**. Le glossaire est factuel
   (« utilisé historiquement pour X »), pas prescriptif (« à choisir
   si vous êtes Y »). Le panneau de personnalisation a un warning
   explicite sur l'absence de pondération universelle.
3. **Toute valeur numérique remontée dans la synthèse doit être
   traçable au JSON d'entrée**. Le test
   `test_every_number_in_synthesis_is_traceable` vérifie ce contrat.
4. **Symétrie FR/EN** garantie par les tests. Toute nouvelle clé
   d'interface ou entrée de glossaire doit exister dans les deux
   langues.
5. **Déterminisme du rapport** : deux générations sur les mêmes
   données produisent le même HTML (octet à octet pour la synthèse).
   Aucun timestamp, ID aléatoire ou ordre non-trié dans le HTML
   généré.

## Lancer la suite de tests

```bash
pip install -e ".[dev,web]"
pytest tests/ -q --tb=short
```

À la date du Sprint 21 : **1244 tests passent, 2 sont skip** (dépendance
scipy optionnelle). Toute contribution doit conserver le statut "0
failed".

## Démo rapide

```bash
picarones demo --output /tmp/demo.html --docs 8
```

Génère un rapport sur des données fictives. Utile pour vérifier visuellement
qu'un nouveau composant s'intègre proprement.
