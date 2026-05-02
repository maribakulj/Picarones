# Documentation développeur Picarones

Guides courts pour étendre Picarones sans casser les invariants
fondamentaux du projet.

## Architecture

Voir [CLAUDE.md](../../CLAUDE.md) pour la cartographie complète des
modules. En résumé :

```
picarones/
├── core/                # cœur analytique pur Python (Cercle 1)
│   ├── pipeline.py      # PipelineRunner pour pipelines composées
│   ├── corpus.py        # Document, Corpus, GTLevel
│   ├── results.py       # DocumentResult, EngineReport, BenchmarkResult
│   ├── modules.py       # BaseModule, ArtifactType
│   ├── facts.py         # Fact, FactType, registre narratif
│   └── …
├── measurements/        # métriques officielles (Cercle 2)
│   ├── runner.py        # orchestration ThreadPool/ProcessPool
│   ├── metrics.py       # CER/WER/MER/WIL via jiwer
│   ├── statistics/      # Wilcoxon, Friedman, Nemenyi, Pareto
│   │   (sous-package depuis le sprint « découpage statistics.py »)
│   ├── narrative/       # moteur de synthèse factuelle
│   ├── pricing.py       # modèle de coût pour la vue Pareto
│   └── …
├── engines/             # adaptateurs OCR (Tesseract, Pero, Mistral OCR…)
├── llm/                 # adaptateurs LLM (OpenAI, Anthropic, Mistral, Ollama)
├── pipelines/           # OCRLLMPipeline (3 modes)
├── report/              # générateur HTML + templates Jinja2 + i18n + glossaire
└── web/                 # FastAPI + SPA vanilla JS
```

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
