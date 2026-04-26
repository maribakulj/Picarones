# CLAUDE.md — Picarones

Plateforme de benchmark OCR/HTR pour documents patrimoniaux.
Repo : github.com/maribakulj/Picarones
HuggingFace Space : huggingface.co/spaces/Ma-Ri-Ba-Ku/Picarones (Docker, port 7860)

---

## Setup

```bash
pip install -e ".[dev,web]"          # IMPORTANT : toujours inclure [web] pour les tests
pytest tests/ -q --tb=short          # lancer les tests
picarones demo --output rapport.html # rapport démo sans moteur installé
picarones serve --port 8080          # interface web locale
```

Mise à jour Codespace complète :
```bash
git pull && pip install -e ".[dev,web]" && picarones demo --output rapport_demo.html && picarones serve --port 8080
```

---

## Architecture

```
picarones/
├── cli.py                  # CLI Click : run, metrics, engines, info, demo, serve, import, history, robustness
├── fixtures.py             # Données de test fictives (documents médiévaux)
├── core/
│   ├── corpus.py           # Chargement corpus (dossier local, ALTO XML, PAGE XML)
│   ├── metrics.py          # CER, WER, MER, WIL (via jiwer)
│   ├── normalization.py    # Profils : nfc, caseless, minimal, medieval_french, early_modern_french,
│   │                       #           medieval_latin, early_modern_english, medieval_english
│   ├── statistics.py       # Bootstrap CI 95%, Wilcoxon (scipy optionnel), corrélations
│   ├── runner.py           # Orchestrateur benchmark (ThreadPool IO-bound, ProcessPool CPU-bound)
│   ├── results.py          # Modèles de données DocumentResult, BenchmarkResults + export JSON
│   ├── confusion.py        # Matrice de confusion unicode
│   ├── char_scores.py      # Scores ligatures (fi, fl, œ, æ, ꝑ…) et diacritiques
│   ├── taxonomy.py         # Taxonomie erreurs 9 classes (confusion visuelle, abréviation…)
│   ├── structure.py        # Analyse structurelle (blocs, lignes, mots)
│   ├── image_quality.py    # Métriques qualité image (contraste, bruit, résolution…)
│   ├── difficulty.py       # Score difficulté intrinsèque par document
│   ├── hallucination.py    # Détection hallucinations VLM (score ancrage, ratio longueur)
│   ├── line_metrics.py     # Distribution erreurs par ligne (Gini, percentiles)
│   ├── history.py          # Suivi longitudinal SQLite
│   ├── robustness.py       # Analyse robustesse (bruit, flou, rotation, résolution)
│   └── narrative/          # Moteur narratif factuel (Sprint 16) — modèle Fact + registre
│       ├── facts.py        # Fact, FactType (12 types), FactImportance, DetectorRegistry
│       └── detectors.py    # Stubs des 12 détecteurs, implémentations par sprint
├── engines/
│   ├── base.py             # BaseEngine avec execution_mode ("io" ou "cpu")
│   ├── tesseract.py        # execution_mode = "cpu"
│   ├── pero_ocr.py         # execution_mode = "cpu"
│   ├── mistral_ocr.py      # endpoint /v1/ocr dédié (pas chat/completions)
│   ├── google_vision.py
│   └── azure_doc_intel.py
├── llm/
│   ├── base.py
│   ├── mistral_adapter.py
│   ├── openai_adapter.py
│   ├── anthropic_adapter.py
│   └── ollama_adapter.py
├── pipelines/
│   ├── base.py             # OCRLLMPipeline (interface BaseOCREngine)
│   └── over_normalization.py
├── prompts/                # 8 fichiers .txt FR+EN
│   ├── medieval_french.txt
│   ├── medieval_french_zero_shot.txt
│   ├── early_modern_french.txt
│   ├── early_modern_french_zero_shot.txt
│   ├── medieval_english.txt
│   ├── early_modern_english.txt
│   ├── medieval_latin.txt
│   └── zero_shot.txt
├── report/
│   ├── generator.py        # Orchestration Jinja2 (617 lignes depuis Sprint 17)
│   ├── diff_utils.py
│   ├── templates/          # Templates Jinja2 (Sprint 17)
│   │   ├── base.html.j2    # assemble tout via {% include %}
│   │   ├── _header.html, _footer.html, _styles.css, _app.js
│   │   └── view_ranking.html, view_gallery.html, view_document.html,
│   │       view_analyses.html, view_characters.html
│   ├── i18n/               # Traductions FR/EN (Sprint 17 — extraites de i18n.py)
│   │   ├── fr.json
│   │   └── en.json
│   └── vendor/             # Chart.js vendorisé
├── web/
│   └── app.py              # FastAPI, SSE, upload corpus ZIP, endpoints modèles dynamiques
└── importers/
    ├── iiif.py
    ├── htr_united.py
    ├── huggingface.py
    ├── gallica.py
    └── escriptorium.py
```

---

## État des tests et bugs historiques

**État actuel (Sprint 16)** : `pytest tests/` → **1072 passed, 2 skipped, 0 failed**.
Les deux tests skip sont volontaires (dépendance scipy optionnelle).

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
- **Les profils de normalisation** sont dans `picarones/core/normalization.py` — l'endpoint
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

| Sprint | Contenu |
|--------|---------|
| 1 | Structure Python, Tesseract, Pero OCR, CER/WER, CLI |
| 2 | Rapport HTML v1 (Chart.js, diff coloré, galerie) |
| 3 | Pipelines OCR+LLM (3 modes), GPT-4o/Claude/Mistral/Ollama, prompts versionnés |
| 4 | Adaptateurs API OCR (Mistral OCR, Google Vision, Azure), import IIIF, CER diplomatique |
| 5 | Métriques avancées (unicode, ligatures, structure, qualité image, taxonomie 9 classes) |
| 6 | Interface web FastAPI, HTR-United/HuggingFace, bilingue FR/EN, upload ZIP |
| 7 | Rapport HTML v2 (Wilcoxon, bootstrap, clustering, score difficulté, URL stateful, CSV) |
| 8 | eScriptorium, Gallica API, suivi longitudinal SQLite, analyse robustesse |
| 9 | Documentation, packaging, Docker, CI/CD GitHub Actions, PyInstaller, version 1.0.0-Beta |
| 10 | Distribution erreurs par ligne (Gini, percentiles), détection hallucinations VLM |
| 11 | Internationalisation FR/EN, profils normalisation anglais (early_modern, medieval, secretary_hand) |
| 12 | Upload ZIP depuis navigateur, filtrage fichiers macOS `._*`, profils exclusion caractères, sélecteur modèles dynamique |
| 13 | Nettoyage pyproject.toml, exceptions silencieuses → warnings, parallélisation runner (ThreadPool/ProcessPool), timeout par doc, résultats partiels NDJSON, validation statistique Wilcoxon |
| 14 | Filtrage robuste des moteurs, validation corpus |
| 15 | Correction du bug pipeline OCR+LLM sortie vide (normalisation ContentChunk Mistral, logs finish_reason/tokens) |
| 16 | **Sprint 1 du plan rapport** : câblage de `line_metrics` et `hallucination` dans le runner et l'agrégation `EngineReport`, fondations du moteur narratif (`core/narrative/` avec modèle `Fact` et registre de détecteurs), correctifs qualité (deprecation Pillow `getdata` → `tobytes`, deux `except Exception: pass` remplacés par warnings explicites) |
| 17 | **Sprint 2 du plan rapport** : refactor de `generator.py` (3690 → 617 lignes) via Jinja2. Le monolithe `_HTML_TEMPLATE` est découpé en 10 fichiers externes dans `picarones/report/templates/` (base + 5 vues + header/footer + CSS + JS). L'i18n `i18n.py` (dict Python 101 clés) migré vers `picarones/report/i18n/{fr,en}.json` chargés à l'import. Ajout de 16 tests de non-régression (structure, déterminisme, i18n, garde-fous contre balises dupliquées). |
| 18 | **Sprint 3 du plan rapport** : test de Friedman multi-moteurs + post-hoc Nemenyi + Critical Difference Diagram (Demšar 2006). Nouveau module `core/statistics.py` : `friedman_test`, `nemenyi_posthoc`, `build_critical_difference_svg` avec table Nemenyi (k=2 à 50, α=0,05 et 0,01), fallback pur Python (Wilson-Hilferty pour chi²), support scipy optionnel (extra `stats`). Partial `_critical_difference.html` inséré en tête du rapport, SVG rendu server-side (pas de JS), i18n FR/EN pour les aides. Détecteur narratif `detect_statistical_tie` activé (lit `nemenyi.tied_groups`). 41 tests ajoutés (cas canoniques, dégénérés, SVG, intégration rapport). |
| 19 | **Sprint 4 du plan rapport** : moteur narratif complet + synthèse factuelle en tête. 9 détecteurs implémentés (global_leader_cer, significant_gap, stratum_winner/collapse, error_profile_outlier, llm_hallucination_flag, robustness_fragile, speed_winner, confidence_warning). Arbitre (`arbiter.py`) avec tri par importance, non-redondance, suppression des contradictions Wilcoxon/Nemenyi. Renderer (`renderer.py`) lit templates YAML `core/narrative/templates/{fr,en}.yaml` (10 templates par langue) et rend par `str.format_map` déterministe. Nouveau partial `_narrative_summary.html` placé en tête du rapport (entre header et CDD). Garde-fou anti-hallucination testé : chaque nombre rendu est traçable au payload du Fact associé. 32 tests (détecteurs unitaires, arbitre, renderer, E2E, traçabilité, intégration HTML). `pareto_alternative` et `cost_outlier` restent stubs pour Sprint 5. |
| 20 | **Sprint 5 du plan rapport** : modélisation coût + vue Pareto. Nouveau module `core/pricing.py` (`EngineCost`, `estimate_cost`, `build_costs_for_benchmark`) lit la table indicative `picarones/data/pricing.yaml` (OCR locaux + APIs cloud + LLM). Nouvel algo `compute_pareto_front` dans `statistics.py`, multi-objectifs (min/max), N dimensions. Vue Chart.js dans `view_analyses.html` avec front Pareto en surbrillance et 3 toggles d'axe : coût € / vitesse / carbone (dernier étiqueté ⚗ expérimental). Détecteurs `pareto_alternative` et `cost_outlier` activés. Templates FR/EN ajoutés. Bloc "hypothèses détaillées" replié sous le graphique avec liens vers les sources de prix. 28 tests (pricing local vs cloud, override taux horaire, pareto canonique/dégénéré/3D, détecteurs, intégration HTML). |
| 21 | **Sprint 6 du plan rapport** : glossaire contextuel + panneau « Mode avancé ». Nouveau module `picarones/report/glossary/` avec loader YAML et 25 entrées bilingues (CER et variantes, WER/MER/WIL, ligatures, diacritiques, taxonomie, Gini, hallucinations, bootstrap, Wilcoxon, Friedman, Nemenyi, CDD, Pareto, difficulté, normalisation, structure, qualité image) — chaque entrée porte `definition`, `measures`, `usage`, `limits`, `reference`. Dans le rapport, un petit `?` apparaît à côté de chaque en-tête de colonne pertinente ; un clic ouvre un panneau latéral avec l'entrée complète. Bouton « ⚙ Avancé » dans la nav ouvre un second panneau latéral avec : choix de colonnes visibles, filtres par strate (script_type), et vue opt-in « score composite personnel » — tous les curseurs à 0 par défaut, formule affichée en permanence, warning explicite « il n'existe pas de pondération universellement valide ». État persisté en URL (`?hidden=…&strata_off=…&w=…`). 19 nouvelles clés i18n (`glossary_*`, `customize_*`). 21 tests (loader, complétude FR/EN, structure des entrées, pas de HTML injecté, intégration rapport, garde-fou anti-prescription). |
| 22 | **Sprint 7 du plan rapport (clôture phase 0)** : études de cas, documentation utilisateur, documentation développeur. Création de `docs/case-studies/` avec 2 cas d'école explicitement étiquetés (registres paroissiaux XVIIᵉ-XVIIIᵉ pour archivistes ; édition critique d'un manuscrit médiéval pour philologues). Encart sous la synthèse pointant vers le dossier. Documentation utilisateur `docs/user/reading-a-report.md` (anatomie du rapport, ordre de lecture suggéré, panneau avancé). Trois guides développeur (`docs/developer/index.md`, `narrative-engine.md`, `extending-glossary.md`, `extending-i18n.md`) couvrant l'extension de chaque sous-système. Tests E2E sur petits/grands corpus + locale EN, garde-fou « pas de fausses études prétendant être réelles » (chaque .md case-study doit contenir « Cas d'école »). 18 tests Sprint 22. |
| 23-31 | Sprints intermédiaires : anti-hallucination, sécurité institutionnelle, refactor frontend Jinja2, persistance SQLite des jobs, snapshots reproductibilité, save/load config + comparaison de runs, registre déclaratif des détecteurs, polish/a11y/DX, couverture des modules sous-testés. Voir `CHANGELOG.md` [1.1.x] pour le détail. |
| 32 | **Sprint 1 du plan d'évolution 2026 — Phase 0.1 : GT multi-niveaux**. Refonte de `picarones/core/corpus.py` pour porter une vérité terrain à plusieurs niveaux (`GTLevel.{TEXT,ALTO,PAGE,ENTITIES,READING_ORDER}`), payloads typés (`TextGT`, `AltoGT`, `PageGT`, `EntitiesGT`, `ReadingOrderGT`) avec `source_path` traçable. Le champ `Document.ground_truth: str` reste la source de vérité historique et est synchronisé automatiquement avec `Document.ground_truths[GTLevel.TEXT]` — rétrocompatibilité stricte (1478 tests existants passent sans modification). Le loader détecte automatiquement `.gt.alto.xml`, `.gt.page.xml`, `.gt.entities.json`, `.gt.reading_order.json` à côté de l'image. `Corpus.gt_level_coverage()` et `Corpus.available_gt_levels` exposent la couverture. Erreurs de parse dégradées en `logger.warning` (jamais `except: pass`). +17 tests dans `test_sprint32_multi_level_gt.py`. **Verrou levé** : ce sprint débloque l'évaluation des modules qui produisent ou consomment ALTO/PAGE/entités (axe B du plan, à venir Sprint 35+) et plusieurs métriques de l'axe A (Layout F1, reading order F1, NER). |
| 33 | **Sprint 2 du plan d'évolution 2026 — Phase 0.2 : interface module générique**. Nouveau module `picarones/core/modules.py` avec l'enum `ArtifactType` (IMAGE, TEXT, ALTO, PAGE, ENTITIES, READING_ORDER) et la classe abstraite `BaseModule` qui déclare `input_types`/`output_types`, `execution_mode` (`"io"`/`"cpu"`), une méthode `process(dict[ArtifactType, Any]) → dict[ArtifactType, Any]`, et des helpers `validate_inputs`/`validate_outputs`. `BaseOCREngine` (`picarones/engines/base.py`) hérite désormais de `BaseModule` avec `input_types=(IMAGE,)` et `output_types=(TEXT,)` ; sa nouvelle méthode `process` wrappe l'API historique `run()`. Aucun adaptateur OCR existant n'est touché — `test_engines.py` passe à 20/20 sans modification. +23 tests dans `test_sprint33_module_interface.py` (contrat, validation, MockModule TEXT→ALTO démonstratif comme demandé par le plan, délégation `BaseOCREngine.process → run`, cohérence ArtifactType/GTLevel). **Verrou levé** : un même runner peut maintenant exécuter un OCR (image→texte), un mappeur VLM→ALTO, un rewriter ALTO→ALTO, un module NER (texte→entités), etc. — fondation directe pour l'axe B du plan. |
| 34 | **Sprint 3 du plan d'évolution 2026 — Phase 0.3 : registre typé de métriques (clôture Phase 0)**. Nouveaux modules `picarones/core/metric_registry.py` (`MetricSpec`, `@register_metric`, `select_metrics`, `compute_at_junction`) et `picarones/core/builtin_metrics.py` qui enregistre `cer`, `wer`, `mer`, `wil` sur `(TEXT, TEXT)` plus un stub `text_preservation_after_reconstruction` sur `(TEXT, ALTO)` comme preuve de concept de jonction hétérogène. **Approche strictement additive** : ni `metrics.py` ni `compute_metrics` ne sont modifiés, le rapport HTML reste identique octet par octet. La sélection par signature de types est exacte (pas de coercion). +21 tests dans `test_sprint34_metric_registry.py`, dont une parité numérique CER/WER/MER/WIL avec `compute_metrics` legacy à 1e-9 près sur 4 paires de textes. **Verrou levé** : le runner d'une pipeline composée peut maintenant calculer automatiquement la métrique adéquate à chaque jonction de son DAG selon les types d'artefacts produits/attendus — fondation directe pour la métrique d'absorption d'erreur (acte B.3) et toutes les métriques structurelles à venir (Layout F1, reading order F1, NER). |
| 35 | **Sprint 4 du plan d'évolution 2026 — Étape 2 / axe A : métriques inter-moteurs (couche de calcul)**. Nouveau module `picarones/core/inter_engine.py` qui répond à deux questions distinctes mais liées : *(a) à quel point les moteurs font-ils des erreurs de natures différentes ?* via `kl_divergence`, `jensen_shannon_divergence` (symétrique, bornée `[0, 1]`), et `taxonomy_divergence_matrix` qui construit la matrice triangulaire inter-moteurs ; *(b) quel CER serait atteignable si on combinait les moteurs ?* via `oracle_token_recall` (proxy bag-of-words, borne supérieure du recall atteignable), `complementarity_gap` (oracle vs meilleur moteur seul, gap absolu/relatif), et `pairwise_disagreement_rate`. Fonctions pures, sans I/O ni intégration runner — la couche de calcul est livrée indépendamment, le câblage narratif (`ENSEMBLE_OPPORTUNITY`) et HTML (matrice de divergence, badge oracle) suit au Sprint 36. +27 tests couvrant les invariants mathématiques (KL ≥ 0, KL(p,p) = 0, JS symétrique et bornée, oracle ≥ best_single, multiplicité respectée), les cas concrets (deux moteurs spécialisés sortent comme candidats ensemble, complémentarité parfaite atteint oracle = 1), et les garde-fous (référence vide, hypothèses vides, métrique inconnue). |
| 42 | **Sprint 11 du plan d'évolution 2026 — Étape 2 / axe A.II.1.b : exposition `token_confidences` + câblage runner**. Suite du Sprint 39 (couche de calcul). `EngineResult` gagne un champ optionnel `token_confidences: Optional[list[dict[str, Any]]]` (`None` par défaut → rétrocompat stricte). `DocumentResult.calibration_metrics` et `EngineReport.aggregated_calibration` ajoutés (sérialisation dans `as_dict` conditionnelle, libérés par `compact()`). Nouveau helper `_calibration_from_engine_result` qui aligne par bag-of-words avec multiplicité (proxy oracle, comme `oracle_token_recall`), normalise les confidences en pourcentage à `[0, 1]`, ignore les confidences négatives (Tesseract met -1 pour les non-mots) ; appelé dans `_compute_document_result` quand `token_confidences` est non-vide. Helper `_aggregate_calibration` combine les bins de tous les docs en somme pondérée par count, recalcule ECE/MCE micro. **L'adaptation de chaque adapter (Tesseract, Pero OCR, Mistral OCR, Google Vision, Azure DI) à exposer ses confidences natives est reportée à des sprints dédiés** : ce sprint pose l'infrastructure complète et la teste avec un mock. +17 tests dans `test_sprint42_calibration_runner.py` (champ EngineResult, sérialisation/compact, helper d'alignement avec calibration parfaite + normalisation % + skip négatifs + bag-of-words multiplicité, agrégation multi-docs, rétrocompat sans confidences). **Verrou levé** : un moteur qui expose ses confidences (cas réel à venir) verra automatiquement ses métriques de calibration calculées et agrégées par le runner — il manque uniquement la vue HTML reliability et l'adaptation des engines un par un. |
| 41 | **Sprint 10 du plan d'évolution 2026 — Étape 2 / axe A.II.1.a : vue HTML NER (clôture A.II.1.a)**. Nouveau module `picarones/report/ner_render.py` : `build_ner_summary_html` rend un tableau résumé (F1 global, P, R, docs évalués, hallucinations, missed) avec cellule F1 colorée par gradient rouge → jaune → vert ; `build_ner_per_category_html` rend la heatmap moteur × catégorie d'entité (PER, LOC, ORG, DATE, MISC…) avec tooltip `support=N`, cellule vide marquée `—` pour les catégories non observées. Rendu server-side, pas de JS, déterministe. Anti-injection HTML via `html.escape`. `_build_report_data` expose `aggregated_ner` par moteur. `ReportGenerator.generate` calcule les deux blocs et les passe au template `view_analyses.html` qui les affiche dans une `chart-card` à largeur pleine **uniquement si ≥ 1 moteur a un `aggregated_ner`**. +12 clés i18n FR/EN. +38 tests dans `test_sprint41_ner_html.py` (rendu, masquage adaptatif, anti-injection, intégration FR + EN, complétude i18n). **Verrou levé** : A.II.1.a (NER) est désormais livré bout-en-bout — couche de calcul (Sprint 38) + backend + câblage runner (Sprint 40) + vue HTML (Sprint 41). Reste la calibration A.II.1.b à finir bout-en-bout (extraction des token_confidences depuis les engines + vue HTML reliability diagram). |
| 40 | **Sprint 9 du plan d'évolution 2026 — Étape 2 / axe A.II.1.a : NER backend + câblage runner**. Suite du Sprint 38 (couche de calcul). Nouveau module `picarones/core/ner_backends.py` : `EntityExtractor` (Protocol, tout callable `(text) → list[dict]` est valide), `SpacyEntityExtractor` (lazy-import spaCy, charge le modèle au premier appel, fallback gracieux silencieux + warning explicite si spaCy/modèle absent, mapping par défaut spaCy → conventions HIPE : PERSON→PER, GPE→LOC, etc.), `SPACY_PROFILES` (6 profils nommés), `get_extractor(profile)`, `is_spacy_available()`. `DocumentResult.ner_metrics: Optional[dict]` et `EngineReport.aggregated_ner` ajoutés (sérialisés dans `as_dict` quand renseignés, libérés par `compact()`). `runner.run_benchmark` accepte un nouveau paramètre optionnel `entity_extractor` ; si fourni, helpers `_attach_ner_metrics` et `_aggregate_ner` calculent les métriques en post-process (main process pour éviter de pickler spaCy dans les sous-processus). Rétrocompat stricte : sans `entity_extractor`, aucun calcul ni champ ajouté. Nouveau extra `[ner]` dans `pyproject.toml` (spacy>=3.7.0). +16 tests dans `test_sprint40_ner_runner.py` (fallback sans spaCy + warning, idempotence load, profils + factory, sérialisation nouveaux champs, câblage runner avec mock injecté, agrégation micro-F1, rétrocompat sans extracteur, robustesse à un extracteur qui lève). **Verrou levé** : un benchmark dont le corpus a une GT entités produit maintenant des métriques NER bout-en-bout — il manque uniquement la vue HTML dédiée (Sprint 41 à venir). |
| 39 | **Sprint 8 du plan d'évolution 2026 — Étape 2 / axe A.II.1.b : Calibration (couche de calcul)**. Nouveau module `picarones/core/calibration.py` avec dataclass `CalibrationBin` (`bin_low/high`, `avg_confidence`, `accuracy`, `count`, propriété `gap`), `reliability_diagram`, `expected_calibration_error` (ECE — moyenne pondérée par bin de `\|conf - accuracy\|`, ∈ [0, 1]), `maximum_calibration_error` (MCE — pire écart sur les bins non vides), `compute_calibration_metrics` (vue agrégée). Calcul d'index de bin par multiplication `int(c * n_bins)` plutôt que division pour éviter le piège IEEE 754 (`0.6 / 0.1 = 5.999…`). Aucune dépendance externe — les listes `confidences` ∈ [0, 1] et `is_correct` ∈ {0,1} sont fournies en entrée ; l'extraction depuis les engines existants est reportée à un sprint dédié. +32 tests couvrant calibration parfaite (ECE = 0), cas extrêmes (sur/sous-confiance → ECE = 0,5), biais constant (ECE = `\|c-a\|`), binning correct (0.6 placé dans le bon bin), bins vides (`gap = None`), garde-fous, monotonie `n_bins` plus fins → ECE ne décroît pas. **Verrou levé** : un workflow patrimonial peut maintenant répondre à *« quand le moteur dit qu'il est sûr, est-il vraiment sûr ? »* — différence entre vérification humaine systématique (100 %) et ciblée (15 %) sur les passages à faible confiance. |
| 38 | **Sprint 7 du plan d'évolution 2026 — Étape 2 / axe A.II.1.a : NER (couche de calcul)**. Nouveau module `picarones/core/ner.py` : dataclass `Entity(label, start, end, text)` (validation de span), fonction `compute_ner_metrics(reference, hypothesis, iou_threshold=0.5)` qui aligne par chevauchement IoU (greedy, IoU décroissant, chaque entité matchée au plus une fois) et retourne precision/recall/F1 globaux + par catégorie + listes `hallucinated_entities` / `missed_entities`. Format dict compatible `EntitiesGT` du Sprint 32. Métrique `ner_f1` enregistrée dans le registre typé Sprint 34 pour la jonction `(ENTITIES, ENTITIES)`. Aucune dépendance externe : les listes d'entités sont fournies en entrée — le backend extracteur (spaCy/Stanza/HIPE) suivra dans un sprint dédié. +19 tests dans `test_sprint38_ner_metrics.py` (cas standards, label case-insensitive, IoU sous/sur seuil, multi-catégorie, alignement greedy, cas dégénérés, validation Entity, intégration registre). **Verrou levé** : un benchmark dont le corpus a une GT entités peut maintenant mesurer l'utilité aval pour l'indexation prosopographique — métrique critique pour les bibliothèques numériques. |
| 37 | **Sprint 6 du plan d'évolution 2026 — Étape 2 / axe A : section inter-moteurs dans le rapport HTML**. Nouveau module `picarones/report/inter_engine_render.py` qui produit deux blocs HTML serveur-side (pas de JS) : `build_divergence_matrix_html` rend une table heatmap CSS inline (gradient blanc → rouge sur le max hors-diagonale, diagonale étiquetée, paire la plus divergente annoncée en sous-titre) ; `build_oracle_gap_html` rend l'encart factuel best engine / recall / oracle / gap absolu+relatif / doc count. Le `ReportGenerator` les calcule et les passe au template `view_analyses.html` qui les affiche dans une `chart-card` à largeur pleine **uniquement si présents** — principe du rapport adaptatif (< 2 moteurs ou pas de taxonomie → section omise). +14 clés i18n FR/EN (`h_inter_engine`, `inter_engine_note`, `divergence_*`, `oracle_*`). Anti-injection HTML via `html.escape`. +42 tests dans `test_sprint37_inter_engine_html.py` couvrant le rendu (valeurs, paire max), le masquage adaptatif sur 4 cas dégénérés, l'anti-injection (engine name `<script>` correctement échappé), l'intégration rapport FR + EN, la complétude i18n sur les 14 clés × 2 langues. **Verrou levé** : ce que le moteur narratif annonce dans la synthèse (« tess et pero ont des profils divergents… ») est maintenant aussi visible dans la vue analyses sous forme de matrice et d'encart factuel — le lecteur peut vérifier visuellement le chiffre. |
| 36 | **Sprint 5 du plan d'évolution 2026 — Étape 2 / axe A : câblage inter-moteurs au runner et au moteur narratif**. Suite du Sprint 35 : `inter_engine.py` gagne `compute_inter_engine_analysis` (agrégation corpus-wide doc par doc, structure stable consommable par les détecteurs et le rapport HTML — oracle global, recall par moteur, per_doc top 50 trié par gap, matrice de divergence, paire la plus divergente). `BenchmarkResult` expose un nouveau champ optionnel `inter_engine_analysis` ; le runner (`run_benchmark`) collecte les hypothèses brutes par moteur avant `compact()` et calcule l'analyse si ≥ 2 moteurs (sinon `None`). Nouveau `FactType.ENSEMBLE_OPPORTUNITY` (priority 130, importance MEDIUM, HIGH si `relative_gap` ≥ 50 %) avec détecteur `detect_ensemble_opportunity` qui fallback sur `per_engine_recall` quand la divergence taxonomique est absente. Templates FR/EN ajoutés à `narrative/templates/{fr,en}.yaml`. `report_data["inter_engine_analysis"]` exposé pour la consommation par le rapport HTML (matrice de divergence Sprint 37 à venir). +22 tests dans `test_sprint36_ensemble_narrative.py` couvrant l'agrégation, l'exposition `BenchmarkResult.as_dict`, les seuils du détecteur, le fallback paire sans taxonomie, l'intégration `build_synthesis` FR + EN, la traçabilité anti-hallucination (chaque nombre rendu est dans le payload, template sans chiffres en dur). |

---

## Moteur narratif (Sprint 16)

Fondations en place dans `picarones/core/narrative/` :

```
core/narrative/
├── __init__.py              # API publique + pipeline build_synthesis
├── facts.py                 # Modèle Fact, FactType (12 types), FactImportance, DetectorRegistry
├── detectors.py             # 10 détecteurs implémentés (Sprint 19) + 2 stubs (Sprint 5)
├── arbiter.py               # Tri par importance, non-redondance, anti-contradiction
├── renderer.py              # Rendu templates YAML par str.format_map (déterministe)
└── templates/
    ├── fr.yaml              # 10 templates français
    └── en.yaml              # 10 templates anglais
```

**Principe anti-hallucination** : chaque valeur numérique ou nom d'entité dans le
`payload` d'un `Fact` doit provenir du JSON d'entrée. Test `test_sprint19_narrative_engine.py`
parse la synthèse rendue et vérifie que chaque nombre est traçable au payload
(via `_numbers_in_payload`) augmenté d'une liste blanche limitative de constantes
de template (`95`, `100`).

**Détecteurs activés dans le registre par défaut (Sprint 20)** — les 12 sont opérationnels :
- Sprint 3 : `statistical_tie`
- Sprint 4 : `global_leader_cer`, `significant_gap`, `stratum_winner`, `stratum_collapse`,
  `error_profile_outlier`, `llm_hallucination_flag`, `robustness_fragile`,
  `speed_winner`, `confidence_warning`
- Sprint 5 : `pareto_alternative`, `cost_outlier`

**Règle anti-contradiction** (arbitre) : si `SIGNIFICANT_GAP` (Wilcoxon non corrigé)
et `STATISTICAL_TIE` (Nemenyi corrigé) concernent les mêmes moteurs, Nemenyi
l'emporte — on ne veut pas dire en même temps "A bat B significativement" ET
"A et B sont indiscernables".

**Pipeline** : `build_synthesis(benchmark_data, lang, max_facts=5)` détecte,
arbitre, rend. Le `ReportGenerator.generate` l'appelle et passe le résultat
au template `_narrative_summary.html` (placé entre `_header.html` et `_critical_difference.html`).

---

## Contexte développement

- **Environnement** : GitHub Codespaces (`/workspaces/Picarones`), Python 3.12
- **Tests** : 1752 passed, 2 skipped (Sprints 32-34 = Phase 0 close ; Sprints 35-37 = inter-moteurs livrés bout-en-bout ; Sprints 38+40+41 = NER livré bout-en-bout calcul → runner → HTML ; Sprints 39+42 = calibration couche de calcul + câblage runner, vue HTML reliability + adaptation engines à venir)
- **Plan d'évolution actif** : [`docs/roadmap/evolution-2026.md`](docs/roadmap/evolution-2026.md)
- **Branche active** : `claude/analyze-project-evolution-KOA56`
- **Transcript de la conversation de développement** :
  `/mnt/transcripts/2026-03-11-14-01-41-picarones-ocr-bench-project.txt`
