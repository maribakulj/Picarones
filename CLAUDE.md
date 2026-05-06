# CLAUDE.md — Picarones

Plateforme de benchmark OCR/HTR pour documents patrimoniaux.
Repo : github.com/maribakulj/Picarones
HuggingFace Space : huggingface.co/spaces/Ma-Ri-Ba-Ku/Picarones (Docker, port 7860)

---

## Architecture — coexistence legacy + rewrite

Voir le manifeste complet dans [`docs/explanation/architecture.md`](docs/explanation/architecture.md).

Le projet a deux arborescences qui cohabitent **par design** depuis le
**rewrite ciblé S27-S46** (cf. [`docs/migration/rewrite-status-s46.md`](docs/migration/rewrite-status-s46.md)) :

**Arbo cible (post-rewrite)** — chemin canonique pour tout nouveau code :

```
domain → formats → evaluation → pipeline → adapters → app → reports_v2 → interfaces
```

Règle d'import stricte : les dépendances vont uniquement de l'extérieur
vers l'intérieur.  Vérifié par `tests/architecture/test_layer_dependencies.py`.

**Arbo legacy (pré-rewrite)** — `picarones/{cli,web,engines,llm,
pipelines,report,measurements,extras,modules,core}/` reste exécutable
le temps que les callers externes (HuggingFace Space, scripts BnF)
migrent.  Ne pas y ajouter de nouveau code.  Calendrier de retrait
documenté dans le CHANGELOG.

---

## Setup

```bash
pip install -e ".[dev,web]"          # toujours inclure [web] pour les tests
pytest tests/ -q --tb=short          # lancer les tests
picarones demo --output rapport.html # rapport démo sans moteur installé
picarones serve --port 8080          # interface web locale
```

---

## Structure

```
picarones/
├── core/                       Cercle 1 — abstractions pures (7 modules)
│   ├── modules.py              BaseModule, ArtifactType
│   ├── corpus.py               Document, Corpus, GTLevel, payloads typés
│   ├── results.py              DocumentResult, EngineReport, BenchmarkResult
│   ├── metric_registry.py      MetricSpec, register_metric, compute_at_junction
│   ├── metric_hooks.py         register_document_metric, register_corpus_aggregator
│   ├── pipeline.py             PipelineRunner, PipelineSpec, PipelineStep
│   └── facts.py                Fact, FactType, FactImportance, DetectorRegistry
│
├── measurements/               Cercle 2 — métriques officielles (~55 modules)
│   ├── runner.py               run_benchmark (orchestration)
│   ├── statistics/             sous-package (Wilcoxon, Friedman/Nemenyi, bootstrap, Pareto, clustering, corrélation, distributions, CDD)
│   ├── metrics.py / normalization.py / builtin_hooks.py
│   ├── confusion.py / taxonomy.py / calibration.py / line_metrics.py / ...
│   ├── readability.py / reliability.py / searchability.py / ner.py / ...
│   ├── mufi.py / abbreviations.py / unicode_blocks.py / roman_numerals.py
│   ├── pipeline_benchmark.py / pipeline_comparison.py / pipeline_spec_loader.py
│   └── narrative/              moteur narratif (arbiter, renderer, registry,
│                                18 détecteurs en 6 familles : ranking, pareto,
│                                stratum, quality, history, ensemble)
│
├── engines/                    Cercle 2 — adapters OCR (5)
│   ├── base.py                 BaseOCREngine (hérite de BaseModule)
│   ├── tesseract.py / pero_ocr.py
│   ├── mistral_ocr.py / google_vision.py / azure_doc_intel.py
│
├── llm/                        Cercle 2 — adapters LLM (4)
│   ├── base.py / mistral_adapter.py / openai_adapter.py
│   └── anthropic_adapter.py / ollama_adapter.py
│
├── pipelines/                  Cercle 2 — pipelines OCR+LLM intégrés
│   ├── base.py (OCRLLMPipeline) / over_normalization.py
│
├── modules/                    Cercle 2 — modules BaseModule officiels
│   └── alto_text_to_mono_region.py
│
├── extras/                     Cercle 3 — plugins / extensions
│   └── importers/              IIIF, Gallica, HTR-United, HuggingFace, eScriptorium
│
├── report/                     Cercle 3 — rendu HTML
│   ├── generator.py / colors.py / diff_utils.py
│   ├── views/                  5 vues thématiques
│   ├── templates/ / i18n/ / glossary/ / vendor/
│   └── *_render.py             ~22 renderers (calibration, NER, Pareto, etc.)
│
├── cli/                        Cercle 3 — Click (7 fichiers)
├── web/                        Cercle 3 — FastAPI (app.py, jobs.py)
├── prompts/                    8 fichiers .txt FR+EN
├── data/                       Tables indicatives (pricing.yaml)
└── fixtures.py                 Corpus de test fictifs
```

---

## État des tests et bugs historiques

`pytest tests/` → **5040 passed, 12 skipped, 8 deselected, 0 failed**
(post-S59).  Les deselected sont les markers `live` (5 tests d'intégration
contre vraie API/binaire) + `network` (3 tests qui hit le réseau réel),
opt-in en local via `pytest -m live` ou `pytest -m network`.  Le
compteur en prose est synchronisé automatiquement par
`scripts/gen_readme_tables.py` — toute modification manuelle sera
ré-écrasée au prochain `make lint`.

### Bugs documentés antérieurement — tous résolus

| Bug | Statut | Sprint de résolution |
|-----|--------|---------------------|
| Pipeline OCR+LLM sortie vide (`tesseract → ministral-3b-latest`) | ✅ Résolu | Sprint 15 — adapter Mistral logue `finish_reason`, `completion_tokens`, normalise les ContentChunk |
| CI `python-multipart` manquant | ✅ Résolu | `pyproject.toml` expose `python-multipart>=0.0.9` dans les extras `dev` ET `web`; `ci.yml:71` installe `.[dev,web]` |
| Tests fixtures post-Sprint 10 (counts moteurs, flag `is_pipeline`) | ✅ Résolu | Fixtures mises à jour |
| Test Windows SQLite `test_history_empty_db` | ✅ Résolu | `try/except OSError` + `gc.collect()` avant `unlink` |
| Test HuggingFace `test_search_language_filter` | ✅ Résolu | Assertion corrigée |

En cas de régression sur un de ces bugs, chercher les fichiers de test
correspondants (`test_sprint15_llm_pipeline_bugs.py`, `test_sprint8_escriptorium_gallica.py`,
`test_sprint6_web_interface.py`) avant de ré-ouvrir une enquête.

---

## Règles importantes — ne pas toucher

- **Ne jamais retirer `python-multipart` des dépendances** : FastAPI vérifie sa présence à
  l'import du module (décoration `@app.post` avec `UploadFile`), pas à l'exécution. Ça casse
  tous les tests web au setup.
- **Ne jamais mettre `except Exception: pass`** : remplacer par
  `logger.warning("[module] fonctionnalité dégradée : %s", e)`.
- **Toujours utiliser `logger.warning` avec message explicite** quand une fonctionnalité optionnelle
  échoue (confusion, taxonomy, structure, image_quality, etc.).
- **Avant tout push, lancer `make lint`** (ou `ruff check picarones/ tests/`).
  La config est centralisée dans `pyproject.toml` sous `[tool.ruff]`, donc
  CI, Makefile et invocation directe produisent le même résultat. Le job
  `lint` du CI est bloquant — un F401 (import inutilisé) ou un E741
  (variable ambiguë) fait échouer la PR, par design.
- **Les profils de normalisation** sont dans `picarones/measurements/normalization.py` — l'endpoint
  `/api/normalization/profiles` doit les lire dynamiquement depuis ce fichier, pas depuis une
  liste statique.

---

## Variables d'environnement

```bash
# Clés API LLM (configurées dans HuggingFace Space Settings → Variables and secrets)
MISTRAL_API_KEY=...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# OCR cloud (optionnel)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json
AZURE_DOC_INTEL_ENDPOINT=https://...
AZURE_DOC_INTEL_KEY=...
```

---

## Pipelines OCR+LLM — modes

| Mode | Description |
|------|-------------|
| **zero_shot** | Le LLM reçoit l'image directement et transcrit sans OCR préalable (VLM) |
| **post_correction_texte** | OCR → texte brut → LLM corrige le texte (modèles texte seul) |
| **post_correction_image_texte** | OCR → LLM reçoit image + texte brut pour correction (VLM) |

`ministral-3b-latest` = modèle texte pur → utiliser mode `post_correction_texte` uniquement.

---

## CI/CD

- **CI GitHub Actions** : `.github/workflows/ci.yml` — Python 3.11/3.12, Linux/macOS/Windows
- **Sync HuggingFace** : `.github/workflows/sync_to_huggingface.yml` — push auto sur main
  (nécessite secret `HF_TOKEN` dans GitHub Settings → Secrets → Actions)
- **HuggingFace Space** : Docker sur port 7860

---

## Sprints réalisés

L'historique détaillé des **97+ sprints** du projet (de la fondation
S1 jusqu'au rewrite ciblé S27-S46 puis l'audit institutionnel
S47-S59) est dans le CHANGELOG.md à la racine.  Cette page,
auparavant pléthorique, ne le duplique plus — un seul endroit où
chercher.

Pour le travail courant, ce qui compte :

- **Phase active** : audit institutionnel post-S57 vers la
  release 1.3.0 (cf. section [Unreleased] du CHANGELOG).
- **Documents de référence** : docs/migration/rewrite-status-s46.md
  (état du rewrite), docs/audits/ (audits historiques figés),
  docs/roadmap/evolution-2026.md (plan stratégique).

## Moteur narratif

Le modèle de données (`Fact`, `FactType`, `FactImportance`,
`DetectorRegistry`) vit en cercle 1 dans
[`picarones/core/facts.py`](picarones/core/facts.py). Les détecteurs et
le rendu vivent en cercle 2 :

```
picarones/measurements/narrative/
├── __init__.py              API publique + pipeline build_synthesis
├── arbiter.py               Tri par importance, non-redondance, anti-contradiction
├── renderer.py              Rendu templates YAML par str.format_map (déterministe)
├── registry.py              Registre par défaut des détecteurs
├── templates/{fr,en}.yaml   18 templates × 2 langues
└── detectors/               18 détecteurs en 6 familles
    ├── ranking.py           5 (global_leader, statistical_tie, significant_gap,
    │                          speed_winner, median_mean_gap_warning)
    ├── pareto.py            2 (pareto_alternative, cost_outlier)
    ├── stratum.py           3 (stratum_winner, stratum_collapse,
    │                          stratification_recommended)
    ├── quality.py           4 (error_profile_outlier, llm_hallucination_flag,
    │                          robustness_fragile, confidence_warning)
    ├── history.py           3 (engine_off_baseline, engine_unstable,
    │                          regression_in_history)
    └── ensemble.py          1 (ensemble_opportunity)
```

**Principe anti-hallucination** : chaque valeur numérique ou nom d'entité
dans le `payload` d'un `Fact` provient du JSON d'entrée. Le test
`test_sprint19_narrative_engine.py` parse la synthèse rendue et vérifie
la traçabilité.

**Règle anti-contradiction** (arbitre) : si `SIGNIFICANT_GAP` (Wilcoxon
non corrigé) et `STATISTICAL_TIE` (Nemenyi corrigé) concernent les mêmes
moteurs, Nemenyi l'emporte.

**Pipeline** : `build_synthesis(benchmark_data, lang, max_facts=5)`
détecte, arbitre, rend.

---

## Contexte développement

- **Environnement** : GitHub Codespaces, Python 3.11+
- **Tests** : `pytest tests/ -q` → ~5040 passed, 2 skipped, 0 failed.
- **Plan d'évolution actif** : [`docs/roadmap/evolution-2026.md`](docs/roadmap/evolution-2026.md).
- **Manifeste architecture** : [`docs/explanation/architecture.md`](docs/explanation/architecture.md).
- **API publique stable** : [`docs/reference/api-stable.md`](docs/reference/api-stable.md).
- **Branche active** : `claude/code-quality-audit-ACnhK`.
