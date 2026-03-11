# Changelog — Picarones

Tous les changements notables de ce projet sont documentés dans ce fichier.

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).
La numérotation de version suit [Semantic Versioning](https://semver.org/lang/fr/).

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
