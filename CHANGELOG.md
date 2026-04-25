# Changelog — Picarones

Tous les changements notables de ce projet sont documentés dans ce fichier.

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).
La numérotation de version suit [Semantic Versioning](https://semver.org/lang/fr/).

---

## [1.1.x] — Sprints 23-30 — 2026-04

### Ajouté

- **Sprint 23** — intégrité anti-hallucination du moteur narratif :
  whitelist `{"95", "100"}` vidée, `confidence_level=95` propagé dans
  `CONFIDENCE_WARNING`, `cost_unit_pages=1000` propagé dans
  `PARETO_ALTERNATIVE`/`COST_OUTLIER`, paramètre `select_facts(..., type_order=...)`,
  test stabilité bootstrap (±0,5 pp inter-seeds), test E2E synthèse EN.
  Doc « Politique éditoriale » dans `docs/developer/narrative-engine.md`.
- **Sprint 24** — durcissement sécurité institutionnelle : mode public
  (`PICARONES_PUBLIC_MODE=1`), `PICARONES_BROWSE_ROOTS`, validation Pillow
  sur upload (CVE-2023-50447), rate limit + sémaphore concurrence,
  middleware CSP + en-têtes durcis, `SECURITY.md` à la racine.
- **Sprint 25** — refactor frontend en Jinja2 : `_HTML_TEMPLATE` (3000 L)
  → 8 partials `picarones/web/templates/` + `static/web-app.js`. CSP
  durcie en partie (script externalisé).
- **Sprint 26** — persistance jobs SQLite : `picarones/core/jobs.py`,
  `JobStore` thread-safe (WAL), `BenchmarkJob` persiste chaque event,
  endpoint SSE supporte `Last-Event-ID`, jobs orphelins marqués
  `interrupted` au boot, fallback DB sur `/api/benchmark/{id}/status`.
- **Sprint 27** — snapshots de reproductibilité dans le rapport HTML :
  `picarones/report/snapshot.py` embarque YAML brut de `pricing.yaml`,
  glossaire trié, profil de normalisation, version Picarones+Python+
  commit+deps figées.
- **Sprint 28** — UX : save/load config (`/api/config/save|load`),
  comparaison de runs (`picarones compare`, exit code 2 si régression),
  synthesis preview (`/api/benchmark/{id}/synthesis_preview`),
  `/api/history/regressions` qui surface l'infra Sprint 8.
- **Sprint 29** — registre déclaratif des détecteurs narratifs :
  `@register_detector(fact_type, priority, importance)` ;
  `DEFAULT_TYPE_ORDER` dérivé du registre. Ajouter un détecteur passe
  de 4 fichiers à 2.
- **Sprint 30** — polish/accessibilité/DX : `.pre-commit-config.yaml`
  avec ruff + check YAML/JSON/secrets, badges CER WCAG (icône + bordure
  pattern + `aria-label`), `i18n.py` thread-safe avec `lru_cache`,
  `_safe_version` log la stacktrace en DEBUG, backport CHANGELOG
  Sprints 10-22, mise à jour SPECS pour narrative/Pareto/glossaire.

### Tests

- 1242 → 1426 tests (+184 sur les Sprints 23-30).

---

## [1.0.x] — Sprints 10-22 — 2025-04 → 2026-03

### Ajouté

- **Sprint 10** — distribution erreurs par ligne (Gini, percentiles)
  dans `picarones/core/line_metrics.py`, détection hallucinations VLM
  dans `picarones/core/hallucination.py` (anchor score, length ratio).
- **Sprint 11** — internationalisation FR/EN, profils de normalisation
  anglais (`early_modern_english`, `medieval_english`, `secretary_hand`).
- **Sprint 12** — upload ZIP depuis le navigateur, filtrage fichiers
  macOS `._*`, profils d'exclusion de caractères (`sans_ponctuation`,
  `sans_apostrophes`), sélecteur dynamique de modèles via
  `/api/models/{provider}`.
- **Sprint 13** — nettoyage `pyproject.toml`, parallélisation runner
  (ThreadPool/ProcessPool selon `execution_mode`), timeout par doc,
  résultats partiels NDJSON, validation statistique Wilcoxon.
- **Sprint 14** — filtrage robuste des moteurs côté CLI/web, validation
  corpus avant lancement.
- **Sprint 15** — fix bug pipeline OCR+LLM sortie vide : `mistral_adapter`
  normalise les `ContentChunk`, log `finish_reason` + tokens.
- **Sprint 16** — câblage de `line_metrics` et `hallucination` dans
  `runner` et l'agrégation `EngineReport` ; fondations du moteur
  narratif (`core/narrative/` avec `Fact`/`DetectorRegistry`) ;
  Pillow `getdata()` → `tobytes()` ; deux `except: pass` → warnings.
- **Sprint 17** — refactor du rapport monolithique : `generator.py`
  3690 → 617 lignes via Jinja2, 10 fichiers externes dans
  `picarones/report/templates/`, i18n migrée vers
  `report/i18n/{fr,en}.json`. +16 tests de non-régression.
- **Sprint 18** — test de Friedman multi-moteurs + Nemenyi post-hoc +
  Critical Difference Diagram (Demšar 2006) ; `core/statistics.py`
  étendu, fallback pur Python, scipy optionnel via extra `[stats]`.
  Détecteur narratif `STATISTICAL_TIE` activé. +41 tests.
- **Sprint 19** — moteur narratif complet : 9 détecteurs implémentés
  (global_leader_cer, significant_gap, stratum_winner/collapse,
  error_profile_outlier, llm_hallucination_flag, robustness_fragile,
  speed_winner, confidence_warning), arbitre, renderer YAML,
  `_narrative_summary.html`. Garde-fou anti-hallucination testé.
  +32 tests.
- **Sprint 20** — modélisation coût + vue Pareto : `core/pricing.py`,
  `data/pricing.yaml`, `compute_pareto_front` multi-objectifs,
  Chart.js Pareto avec axes coût/vitesse/carbone. Détecteurs
  `pareto_alternative` + `cost_outlier` activés. +28 tests.
- **Sprint 21** — glossaire contextuel (25 entrées bilingues) +
  panneau « Mode avancé » : choix de colonnes, filtres par strate,
  vue opt-in « score composite personnel » avec curseurs à 0 par défaut
  et formule visible. État persisté en URL. +21 tests.
- **Sprint 22** — études de cas (`docs/case-studies/`),
  `docs/user/reading-a-report.md`, trois guides développeur dans
  `docs/developer/`. Garde-fou « pas de fausses études prétendant
  être réelles ». +18 tests.

### Modifié

- `pyproject.toml` : extras `[stats]`, `[hf]`, mises à jour de
  `dev`/`web` pour `python-multipart`.
- `picarones/core/runner.py` : refactor pour gérer le `execution_mode`
  des moteurs (IO-bound vs CPU-bound).

### Corrigé

- `python-multipart` durablement présent dans `[dev]` et `[web]`
  (FastAPI vérifie l'import au décorateur `@app.post`).
- Tests Windows SQLite `test_history_empty_db` (gc.collect avant
  unlink).
- `test_search_language_filter` (HuggingFace) — assertion corrigée.

---

## [1.0.0] — Sprint 9 — 2025-03

### Ajouté
- `README.md` complet bilingue (français + anglais) avec badges CI, description des fonctionnalités, tableau des moteurs, variables d'environnement
- `INSTALL.md` — guide d'installation détaillé pour Linux (Ubuntu/Debian), macOS et Windows, incluant Tesseract, Pero OCR, Ollama, configuration des clés API, Docker
- `CHANGELOG.md` — historique des sprints 1 à 9
- `CONTRIBUTING.md` — guide pour contribuer : ajouter un moteur OCR, un adaptateur LLM, soumettre une PR
- `Makefile` — commandes `make install`, `make test`, `make demo`, `make serve`, `make build`, `make build-exe`, `make docker-build`, `make lint`, `make clean`
- `Dockerfile` — image Docker multi-étape basée sur Python 3.11-slim, Tesseract pré-installé, `CMD ["picarones", "serve", "--host", "0.0.0.0"]`
- `docker-compose.yml` — service Picarones + service Ollama optionnel (profil `ollama`)
- `.github/workflows/ci.yml` — pipeline GitHub Actions : tests sur Python 3.11/3.12, Linux/macOS/Windows, rapport de couverture
- `picarones.spec` — configuration PyInstaller pour générer des exécutables standalone (Linux, macOS, Windows)
- `picarones/__main__.py` — permet l'exécution via `python -m picarones`
- Version bumped à `1.0.0` dans `pyproject.toml` et `__init__.py`
- Extras PyPI `[llm]`, `[ocr-cloud]`, `[all]` dans `pyproject.toml`
- Tests Sprint 9 : `tests/test_sprint9_packaging.py` (30 tests)

### Modifié
- `pyproject.toml` : version 1.0.0, nouveaux extras, classifiers mis à jour, URLs projet ajoutées

---

## [0.8.0] — Sprint 8 — 2025-03

### Ajouté
- **eScriptorium** (`picarones/importers/escriptorium.py`)
  - `EScriptoriumClient` : connexion par token API, listing projets/documents/pages, gestion de la pagination
  - `import_document()` : import d'un document avec ses transcriptions comme corpus Picarones
  - `export_benchmark_as_layer()` : export des résultats benchmark comme couche OCR nommée dans eScriptorium
  - `connect_escriptorium()` : connexion avec validation automatique
- **Gallica API** (`picarones/importers/gallica.py`)
  - `GallicaClient` : recherche SRU par cote/titre/auteur/date/langue/type
  - Récupération OCR Gallica texte brut (`f{n}.texteBrut`)
  - Import IIIF Gallica avec enrichissement OCR comme vérité terrain de référence
  - Métadonnées OAI-PMH (`/services/OAIRecord`)
  - `search_gallica()`, `import_gallica_document()` — fonctions de commodité
- **Suivi longitudinal** (`picarones/core/history.py`)
  - `BenchmarkHistory` : base SQLite horodatée par run, moteur, corpus, CER/WER
  - `record()` depuis `BenchmarkResult`, `record_single()` pour imports manuels
  - `query()` avec filtres engine/corpus/since/limit
  - `get_cer_curve()` : données prêtes pour Chart.js
  - `detect_regression()` / `detect_all_regressions()` : seuil configurable en points de CER
  - `export_json()` — export complet de l'historique
  - `generate_demo_history()` : 8 runs fictifs avec régression simulée au run 5
- **Analyse de robustesse** (`picarones/core/robustness.py`)
  - 5 types de dégradation : bruit gaussien, flou, rotation, réduction de résolution, binarisation
  - `degrade_image_bytes()` : Pillow (préféré) ou fallback pur Python
  - `RobustnessAnalyzer.analyze()` : CER par niveau, seuil critique automatique
  - `DegradationCurve`, `RobustnessReport`, `_build_summary()`
  - `generate_demo_robustness_report()` : rapport fictif réaliste sans moteur réel
- **CLI Sprint 8**
  - `picarones history` : historique avec filtres, détection de régression, export JSON, mode `--demo`
  - `picarones robustness` : analyse de robustesse, barres ASCII, export JSON, mode `--demo`
  - `picarones demo --with-history --with-robustness` : démonstration intégrée
- `picarones/importers/__init__.py` mis à jour pour exporter les nouveaux importeurs

### Tests
- `tests/test_sprint8_escriptorium_gallica.py` : 74 tests (eScriptorium, Gallica, CLI)
- `tests/test_sprint8_longitudinal_robustness.py` : 86 tests (history, robustesse, CLI)
- **Total** : 743 tests (anciennement 583)

---

## [0.7.0] — Sprint 7 — 2025-02

### Ajouté
- **Rapport HTML v2**
  - Intervalles de confiance Bootstrap à 95% (`bootstrap_ci()`)
  - Tests de Wilcoxon et matrices de tests par paires (`wilcoxon_test()`, `pairwise_stats()`)
  - Courbes de fiabilité (CER cumulatif par percentile de qualité)
  - Diagrammes de Venn des erreurs communes/exclusives entre concurrents (2 et 3 ensembles)
  - Clustering des patterns d'erreurs (k-means simplifié sur n-grammes d'erreur)
  - Matrice de corrélation entre métriques (Pearson)
  - Score de difficulté intrinsèque par document (`compute_difficulty()`, `compute_all_difficulties()`)
  - Scatter plots interactifs qualité image vs CER, colorés par type de script
  - Heatmaps de confusion unicode améliorées
- `picarones/core/statistics.py` : module dédié aux tests statistiques
- `picarones/core/difficulty.py` : score de difficulté intrinsèque

### Tests
- `tests/test_sprint7_advanced_report.py` : 100 tests (bootstrap, Wilcoxon, Venn, clustering, difficulté)
- **Total** : 583 tests (anciennement 483)

---

## [0.6.0] — Sprint 6 — 2025-02

### Ajouté
- **Interface web FastAPI** (`picarones/web/app.py`)
  - Endpoints REST pour lancer des benchmarks, consulter les résultats, lister les moteurs
  - Streaming des logs en temps réel (Server-Sent Events)
  - `picarones serve` — lancement du serveur uvicorn
- **Import HuggingFace Datasets** (`picarones/importers/huggingface.py`)
  - Recherche, filtrage et import partiel de datasets OCR/HTR
  - Datasets patrimoniaux pré-référencés : IAM, RIMES, READ-BAD, Esposalles…
  - Cache local avec gestion des versions
- **Import HTR-United** (`picarones/importers/htr_united.py`)
  - Listing et import depuis le catalogue HTR-United
  - Lecture des métadonnées : langue, script, institution, époque
- **Adaptateurs Ollama** (`picarones/llm/ollama_adapter.py`)
  - Support de Llama 3, Gemma, Phi et tout modèle Ollama local
  - Mode texte seul (LLMs non multimodaux)
- **Profils de normalisation pré-configurés**
  - Français médiéval, Français moderne, Latin médiéval, Imprimés anciens
  - Profil personnalisé exportable/importable

### Tests
- `tests/test_sprint6_web_interface.py` : 90 tests
- **Total** : 483 tests (anciennement 393)

---

## [0.5.0] — Sprint 5 — 2025-02

### Ajouté
- **Matrice de confusion unicode** (`picarones/core/confusion.py`)
  - `build_confusion_matrix()`, `aggregate_confusion_matrices()`
  - Affichage compact trié par fréquence d'erreur
- **Scores ligatures et diacritiques** (`picarones/core/char_scores.py`)
  - `compute_ligature_score()` : fi, fl, ff, ffi, ffl, st, ct, œ, æ, ꝑ, ꝓ…
  - `compute_diacritic_score()` : accents, cédilles, trémas, diacritiques combinants
- **Taxonomie des erreurs en 10 classes** (`picarones/core/taxonomy.py`)
  - Confusion visuelle, erreur diacritique, casse, ligature, abréviation, hapax, segmentation, hors-vocabulaire, lacune, sur-normalisation LLM
- **Analyse structurelle** (`picarones/core/structure.py`)
  - Score d'ordre de lecture, taux de segmentation des lignes, conservation des sauts de paragraphe
- **Métriques de qualité image** (`picarones/core/image_quality.py`)
  - Netteté (Laplacien), niveau de bruit, contraste (Michelson), détection rotation résiduelle
  - Corrélations image ↔ CER
- Intégration de toutes ces métriques dans le rapport HTML (vue Analyse, vue Caractères)
- Scatter plots qualité image vs CER

### Tests
- `tests/test_sprint5_advanced_metrics.py` : 100 tests
- **Total** : 393 tests (anciennement 293)

---

## [0.4.0] — Sprint 4 — 2025-01

### Ajouté
- **Adaptateurs APIs cloud OCR**
  - Mistral OCR (`picarones/engines/mistral_ocr.py`) — Mistral OCR 3, multimodal
  - Google Vision (`picarones/engines/google_vision.py`) — Document AI
  - Azure Document Intelligence (`picarones/engines/azure_doc_intel.py`)
- **Import IIIF v2/v3** (`picarones/importers/iiif.py`)
  - Sélecteur de pages (`"1-10"`, `"1,3,5"`, `"all"`)
  - Téléchargement images et extraction des annotations de transcription si disponibles
  - Compatibilité : Gallica, Bodleian, British Library, BSB, e-codices, Europeana
  - `picarones import iiif <url>` — commande CLI
- **Normalisation unicode** (`picarones/core/normalization.py`)
  - NFC, caseless, diplomatique (tables ſ=s, u=v, i=j, æ=ae, œ=oe…)
  - Profils configurables via YAML
  - CER diplomatique dans les métriques

### Tests
- `tests/test_sprint4_normalization_iiif.py` : 100 tests
- **Total** : 293 tests (anciennement 193)

---

## [0.3.0] — Sprint 3 — 2025-01

### Ajouté
- **Pipelines OCR+LLM** (`picarones/pipelines/base.py`)
  - Mode 1 — Post-correction texte brut (LLM reçoit la sortie OCR)
  - Mode 2 — Post-correction avec image (LLM reçoit image + OCR)
  - Mode 3 — Zero-shot LLM (LLM reçoit uniquement l'image)
  - Chaînes composables multi-étapes
- **Adaptateurs LLM**
  - OpenAI (`picarones/llm/openai_adapter.py`) — GPT-4o, GPT-4o mini
  - Anthropic (`picarones/llm/anthropic_adapter.py`) — Claude Sonnet, Haiku
  - Mistral (`picarones/llm/mistral_adapter.py`) — Mistral Large, Pixtral
- **Détection de sur-normalisation LLM** (`picarones/pipelines/over_normalization.py`)
  - Mesure du taux de modification sur des passages déjà corrects
  - Classe 10 dans la taxonomie des erreurs
- **Bibliothèque de prompts**
  - Prompts pour manuscrits médiévaux, imprimés anciens, latin
  - Versionning des prompts dans les métadonnées du rapport
- Vue spécifique OCR+LLM dans le rapport : diff triple GT / OCR brut / après correction

### Tests
- `tests/test_sprint3_llm_pipelines.py` : 100 tests
- **Total** : 193 tests (anciennement 93)

---

## [0.2.0] — Sprint 2 — 2025-01

### Ajouté
- **Rapport HTML interactif** (`picarones/report/generator.py`)
  - Fichier HTML auto-contenu, lisible hors-ligne
  - Tableau de classement des concurrents (CER, WER, scores), tri par colonne
  - Graphique radar (spider chart) : CER / WER / Précision diacritiques / Ligatures
  - Vue Galerie : toutes les images avec badges CER colorés (vert→rouge), filtres
  - Vue Document : image zoomable + diff coloré façon GitHub, scroll synchronisé N-way
  - Vue Analyse : histogrammes de distribution CER, scatter plots
  - Recommandation automatique de moteur
  - Exports CSV, JSON, ALTO XML depuis le rapport
- **Diff coloré** (`picarones/report/diff_utils.py`)
  - Diff au niveau caractère et mot
  - Insertions (vert), suppressions (rouge), substitutions (orange)
  - Bascule diplomatique / normalisé
- `picarones demo` — rapport de démonstration avec données fictives réalistes
- `picarones report --results results.json` — génère le HTML depuis un JSON existant
- `picarones/fixtures.py` — générateur de benchmarks fictifs (12 textes médiévaux, 4 concurrents)

### Tests
- `tests/test_report.py`, `tests/test_diff_utils.py` : 93 tests
- **Total** : 93 tests (anciennement 20)

---

## [0.1.0] — Sprint 1 — 2025-01

### Ajouté
- **Structure complète du projet** Python avec `pyproject.toml`, `setup`, packaging
- **Adaptateur Tesseract 5** (`picarones/engines/tesseract.py`) via `pytesseract`
  - Configuration lang, PSM, DPI
  - Récupération de la version
- **Adaptateur Pero OCR** (`picarones/engines/pero_ocr.py`)
  - Chargement de modèle, traitement d'image
- **Interface abstraite** `BaseOCREngine` avec `process_image()`, `get_version()`, propriétés
- **Calcul CER et WER** (`picarones/core/metrics.py`) via `jiwer`
  - CER brut, NFC, caseless
  - WER, WER normalisé, MER, WIL
  - Longueurs de référence et hypothèse
- **Chargement de corpus** (`picarones/core/corpus.py`)
  - Dossier local : paires image / `.gt.txt`
  - Détection automatique des extensions image (jpg, png, tif, bmp…)
  - Classe `Corpus`, `Document`
- **Export JSON** (`picarones/core/results.py`)
  - `BenchmarkResult`, `EngineReport`, `DocumentResult`
  - `ranking()` : classement par CER moyen
  - `to_json()` avec horodatage et métadonnées
- **Orchestrateur benchmark** (`picarones/core/runner.py`)
  - Traitement séquentiel des documents par moteur
  - Barre de progression `tqdm`
  - Cache des sorties par hash SHA-256
- **CLI Click** (`picarones/cli.py`)
  - `picarones run` — benchmark complet
  - `picarones metrics` — CER/WER entre deux fichiers
  - `picarones engines` — liste des moteurs avec statut
  - `picarones info` — version et dépendances
  - `--fail-if-cer-above` pour intégration CI/CD

### Tests
- `tests/test_metrics.py`, `test_corpus.py`, `test_engines.py`, `test_results.py` : 20 tests
