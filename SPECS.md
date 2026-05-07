# Picarones — Spécifications fonctionnelles et techniques

> **Plateforme de banc d'essai d'OCR / HTR / VLM et de pipelines de
> post-correction pour documents patrimoniaux.**
>
> Version **2.0** — Mai 2026. Refonte intégrale (Sprint A14 du plan
> de remédiation institutionnelle, item **B-12**).

> **Note de lecture** : ce document décrit ce que Picarones **fait
> aujourd'hui**, dans la version 1.x. Pour les **non-fonctionnalités
> assumées** (ce que Picarones *ne fait pas et ne fera pas* dans
> la v1.x — par exemple la recommandation prescriptive, l'export PDF,
> les adapters Kraken/AWS Textract), voir la section §10.
>
> Pour la cartographie technique du code et les règles de
> contribution interne, voir [`CLAUDE.md`](CLAUDE.md). SPECS.md
> reste tourné « public » (vocabulaire bibliothécaire, exemples
> patrimoniaux). Les deux documents sont complémentaires, pas
> redondants.

---

## Table des matières

1. [Vision et positionnement](#1-vision-et-positionnement)
2. [Architecture en 8 couches concentriques](#2-architecture-en-8-couches-concentriques)
3. [Module 1 — Corpus et imports](#3-module-1--corpus-et-imports)
4. [Module 2 — Adaptateurs OCR / HTR](#4-module-2--adaptateurs-ocr--htr)
5. [Module 3 — Pipelines OCR+LLM et pipelines composables](#5-module-3--pipelines-ocrllm-et-pipelines-composables)
6. [Module 4 — Métriques et analyses](#6-module-4--métriques-et-analyses)
7. [Module 5 — Rapport HTML interactif](#7-module-5--rapport-html-interactif)
8. [Module 6 — Interface web et CLI](#8-module-6--interface-web-et-cli)
9. [Reproductibilité et sécurité](#9-reproductibilité-et-sécurité)
10. [Limites assumées et non-fonctionnalités](#10-limites-assumées-et-non-fonctionnalités)
11. [Roadmap d'évolution](#11-roadmap-dévolution)
12. [Migration v1 → v2 — annexe historique](#12-migration-v1--v2--annexe-historique)

---

## 1. Vision et positionnement

### 1.1 Problématique

Les équipes OCR/HTR travaillant sur des fonds patrimoniaux
(manuscrits, imprimés anciens, archives) disposent d'un paysage
hétérogène — moteurs locaux (Tesseract, Pero OCR), services cloud
(Mistral OCR, Google Vision, Azure Document Intelligence), modèles
fine-tunés maison, VLMs (GPT-4o, Claude, Mistral Large) — sans
outil unifié pour les comparer rigoureusement sur leurs propres
corpus.

Les outils existants (ocrevalUAtion, dinglehopper) sont soit
obsolètes, soit limités au CER/WER, soit non adaptés aux
spécificités des documents historiques : glyphes anciens,
ligatures, abréviations, graphies variables, pathologies d'image,
ordre de lecture multi-colonnes, structure ALTO/PAGE.

À cela s'ajoute une question de recherche émergente : **est-ce
qu'une couche de correction par LLM améliore réellement la sortie
OCR, de combien, sur quels types d'erreurs, et sans introduire
d'over-normalisation moderne ?** Aucun outil existant ne permet
de tester et mesurer cela rigoureusement.

### 1.2 Philosophie : un banc d'essai, pas un atelier

Picarones est conçu comme un **banc d'essai** :

- L'utilisateur amène son **golden dataset** annoté (paires image +
  vérité terrain). Sans VT, pas de benchmark.
- Picarones exécute les IA candidates et **mesure** l'écart à la VT.
- Picarones **classe** les résultats avec rigueur statistique.
- Picarones **n'arbitre pas le débat éditorial**. Il ne dit pas si
  un moteur diplomatique vaut mieux qu'un moteur modernisant —
  il rapporte les chiffres et laisse le chercheur, l'archiviste
  ou le paléographe trancher selon ses critères propres.

Cette philosophie est tenue jusque dans le moteur narratif
factuel : chaque phrase de la synthèse en tête du rapport est
traçable à un payload de `Fact` qui provient du JSON d'entrée
(garde-fou anti-hallucination prouvé par tests).

### 1.3 Contributions scientifiques

Au-delà de la simple agrégation de moteurs, Picarones apporte
plusieurs briques nouvelles dans l'écosystème OCR/HTR open-source :

- **Registre typé de métriques** (Sprint 34) : chaque métrique
  est enregistrée pour une jonction de types `ArtifactType`
  (TEXT/ALTO/PAGE/ENTITIES/READING_ORDER) ; un pipeline composé
  peut alors calculer automatiquement la métrique adéquate à
  chaque jonction de son DAG.
- **Interface BaseModule générique** (Sprint 33) : OCR, mappeur
  VLM→ALTO, rewriter ALTO→ALTO, classifieur d'entités texte→entités
  partagent la même API ; le runner les enchaîne sans privilégier
  un type particulier.
- **GT multi-niveaux** (Sprint 32) : un document peut porter
  simultanément une vérité terrain texte, ALTO, PAGE, entités,
  et reading order — chacune calibrée à son niveau d'évaluation.
- **Moteur narratif factuel anti-hallucination** (Sprint 19+) :
  20+ détecteurs déterministes produisent une synthèse en
  langage naturel dont chaque chiffre est traçable au payload
  d'un `Fact`. Aucune intervention LLM, garde-fou prouvé par
  test (`test_sprint23_anti_hallucination`).
- **Test multi-moteurs Friedman + Nemenyi + Critical Difference
  Diagram** (Sprint 18, Demšar 2006) : référence canonique pour
  la comparaison statistique de classifieurs, transposée à l'OCR.
- **Pareto coût / vitesse / CO₂** (Sprint 20) : positionnement
  tri-objectifs avec front explicite, table de pricing
  surchargeable.
- **Métriques philologiques transversales** (Sprints 55–60) :
  six modules couvrant l'imprimé ancien, le médiéval, les
  archives modernes (XIXᵉ–XXᵉ), avec scores éditoriaux séparés
  (préservation stricte vs équivalence diplomatique).

### 1.4 Utilisateurs cibles

| Profil | Cas d'usage typique |
|---|---|
| **Ingénieur OCR/ML** | Pipeline programmatique, métriques fines, export JSON, intégration CI/CD via `picarones run --fail-if-cer-above` |
| **Chargé de numérisation** | Rapport HTML autonome, comparaison A vs B, lecture du Pareto coût/qualité |
| **Responsable de projet** | Vue agrégée multi-corpus, analyse coût/bénéfice des APIs cloud, suivi longitudinal SQLite |
| **Chercheur en humanités numériques** | Métriques philologiques, corpus HTR-United, taxonomie d'erreurs en 10 classes, glossaire contextuel |
| **Paléographe / éditeur critique** | Diff visuel par document, bascule diplomatique / normalisé, profil philologique séparant strict et expansion |
| **DSI institutionnel** | Déploiement intranet derrière SSO, RGPD, observabilité (cf. `docs/operations/`) |

---

## 2. Architecture en 8 couches concentriques

```
domain → formats → evaluation → pipeline → adapters → app → reports_v2 → interfaces
```

**Règle de dépendance** : les imports vont uniquement de
l'extérieur vers l'intérieur (de gauche à droite dans le
diagramme).  La règle est appliquée par
`tests/architecture/test_layer_dependencies.py` qui parse
l'AST de chaque fichier et bloque toute violation au merge.

> **Note sur le legacy** : le projet est en cours de retrait
> du legacy.  Une arborescence historique
> (``picarones/{core,measurements,engines,llm,pipelines,
> report,modules}``) cohabite encore et est en train de
> disparaître phase par phase.  Cf.
> [`docs/migration/legacy-retirement-plan.md`](docs/migration/legacy-retirement-plan.md)
> pour le statut et le calendrier.  Tout nouveau code va
> dans l'arborescence canonique ; les chemins legacy
> existants sont des shims minimaux destinés à être
> supprimés.

### 2.1 `picarones/domain/` — types purs

Cercle le plus interne.  Stdlib + Pydantic uniquement, aucune
I/O, aucun framework, aucun module legacy.

| Module | Contenu |
|---|---|
| `artifacts.py` | `Artifact`, `ArtifactType` (10 types : IMAGE, RAW_TEXT, CORRECTED_TEXT, ALTO_XML, PAGE_XML, CANONICAL_DOCUMENT, ENTITIES, READING_ORDER, ALIGNMENT, CONFIDENCES) |
| `corpus.py` | `CorpusSpec` |
| `documents.py` | `DocumentRef` |
| `evaluation_spec.py` | `MetricSpec`, `EvaluationView`, `EvaluationSpec` |
| `pipeline_spec.py` | `PipelineSpec`, `PipelineStep`, `INITIAL_STEP_ID` |
| `projection_spec.py` | `ProjectionSpec` |
| `provenance.py` | `ProvenanceRecord` |
| `run_manifest.py` | `RunManifest` |
| `module_protocol.py` | `BaseModule` (ABC, voie de retrait au profit de `StepExecutor`) |
| `facts.py` | `Fact`, `FactType`, `FactImportance`, `DetectorRegistry` |
| `errors.py` | Hiérarchie d'exceptions (`PicaronesError`, `AdapterStepError`, …) |

### 2.2 `picarones/formats/` — parsing / sérialisation

ALTO 4, PAGE XML, JSON, XML utilitaires.  Stdlib + lxml +
defusedxml.  Pas de logique métier.

### 2.3 `picarones/evaluation/` — métriques et calcul

Cœur de la valeur ajoutée.  Stdlib + numpy + scipy + jiwer +
spacy + rapidfuzz.

| Sous-paquet | Contenu |
|---|---|
| `metrics/` | ~30 métriques (CER, WER, MUFI, philological, NER, calibration, taxonomy, …) |
| `statistics/` | Wilcoxon, Friedman/Nemenyi, bootstrap, Pareto, clustering, CDD |
| `views/`, `projectors/` | EvaluationView (Sprint S13+), projecteurs `AltoToText`, `PageToText`, `CanonicalToText` |
| `corpus.py` | `Document`, `Corpus`, `GTLevel`, payloads (legacy en cours de retrait) |
| `metric_registry.py`, `metric_hooks.py`, `metric_result.py` | Registres typés + hooks + dataclasses résultats |
| `pipeline.py`, `pipeline_benchmark.py`, `pipeline_comparison.py` | `PipelineRunner` legacy + orchestration corpus-wide (en cours de convergence vers `pipeline.executor`) |
| `benchmark_result.py` | `BenchmarkResult`, `EngineReport`, `DocumentResult`, sérialisation JSON |
| `engines/` | OCR engines legacy (`BaseOCREngine`-based) — temporairement avant suppression complète |
| `_diff_utils.py` | `compute_word_diff`, `compute_char_diff`, `diff_stats` |

### 2.4 `picarones/pipeline/` — orchestration canonique

`PipelineExecutor` instance-based, `StepExecutor` Protocol,
`ExecutionPlan` immuable.  Cible canonique pour le bench
d'axe B (pipelines composées).

### 2.5 `picarones/adapters/` — adapters externes

Adapters OCR / LLM / VLM consommant des libs externes
(pytesseract, mistralai, openai, anthropic, google.cloud,
azure.*, pero_ocr, ollama).  Implémentent `StepExecutor`.

| Sous-paquet | Contenu |
|---|---|
| `ocr/` | `TesseractAdapter`, `PeroOCRAdapter`, `MistralOCRAdapter`, `GoogleVisionAdapter`, `AzureDocIntelAdapter`, `PrecomputedAdapter` |
| `llm/` | `BaseLLMAdapter` + Mistral / OpenAI / Anthropic / Ollama |
| `vlm/` | Adapters VLM (zero-shot OCR via vision-language models) |
| `corpus/` | Loaders externes : IIIF, Gallica, HTR-United, HuggingFace |
| `storage/` | `ArtifactStore`, `JobStore` (S29 + S47) |
| `legacy_engines/`, `legacy_modules/` | Engines + modules legacy `BaseModule`-based (en cours de retrait, cf. Phase 7.A) |

### 2.6 `picarones/app/` — services applicatifs

`BenchmarkService`, `CorpusRunner`, `RunOrchestrator`.
Orchestrent les pipelines canoniques sur corpus.

### 2.7 `picarones/reports_v2/` — rendu HTML / JSON / CSV

Rapport final consommant un `BenchmarkResult` ou `RunResult`.
22 renderers thématiques + 5 vues (advanced_taxonomy,
diagnostics, economics, pipeline, robustness) +
`ReportGenerator` orchestrateur + templates Jinja2 +
glossaire bilingue (25 entrées) + i18n FR/EN.

### 2.8 `picarones/interfaces/` — entrées utilisateur

CLI Click, Web FastAPI, IIIF/Gallica/eScriptorium importers
exposés en interface.

---

## 3. Module 1 — Corpus et imports

### 3.1 Formats de vérité terrain acceptés

| Format | Extension | Niveau GT | Usage typique |
|---|---|---|---|
| Texte brut | `image.gt.txt` | TEXT | Convention Tesseract, HTR-United |
| ALTO XML v4 | `.gt.alto.xml` | ALTO | Standard bibliothèques nationales, eScriptorium export |
| PAGE XML 2019 | `.gt.page.xml` | PAGE | Transkribus, OCRopus |
| Entités nommées | `.gt.entities.json` | ENTITIES | Format HIPE simplifié |
| Reading order | `.gt.reading_order.json` | READING_ORDER | Liste ordonnée de region IDs |

Le loader (`load_corpus_from_directory`) détecte automatiquement
chacun de ces niveaux à côté de l'image. Un même document peut
porter plusieurs niveaux simultanément (Sprint 32).

### 3.2 Sources d'import

#### Local
Import d'un dossier de paires image / GT. Détection automatique
du format. Filtrage des fichiers macOS `._*`.

#### IIIF
Import par URL de manifeste IIIF v2 et v3. Compatible Gallica
(BnF), Bodleian, BL, Vatican, e-codices, Europeana, et tout
entrepôt IIIF-compliant. Sélection par range de canvas.

#### HuggingFace Datasets
Recherche par filtre langue/script/époque/institution. Datasets
patrimoniaux pré-référencés (IAM, RIMES, READ-BAD, Esposalles,
HTR-United datasets). Statut : module
`extras/importers/huggingface.py` marqué expérimental
(`UserWarning` à l'import).

#### HTR-United
Listing du catalogue distant + import direct. Lecture des
métadonnées (langue, script, institution, époque). En cas
d'échec réseau ou parsing, fallback sur catalogue de démo +
émission d'un `Fact` `IMPORTER_FALLBACK_TRIGGERED` (Sprint A3).

#### Gallica (API BnF)
Recherche par cote, titre, auteur, date. Récupération des
images via API IIIF Gallica.

#### eScriptorium
Connexion à une instance distante via API. Statut
expérimental.

#### Upload ZIP via navigateur
Endpoint `POST /api/corpus/upload`. Validation Pillow
(décompression bombs), zip-slip prévenu, taille plafonnée
(`PICARONES_MAX_UPLOAD_MB`).

### 3.3 Gestion des corpus

- Corpus nommés et versionnés avec métadonnées descriptives.
- Tags : type de script, langue, siècle, institution, état de
  conservation.
- Stratification par `script_type` (Sprint 45-46) — vue stratifiée
  dans le rapport, détecteur narratif `STRATIFICATION_RECOMMENDED`
  qui invite l'utilisateur quand le corpus est hétérogène.

---

## 4. Module 2 — Adaptateurs OCR / HTR

### 4.1 Architecture des adaptateurs

Chaque moteur OCR est une classe Python qui hérite de
`BaseOCREngine` (`picarones/engines/base.py`), elle-même héritière
de `BaseModule` (Sprint 33). Une instance déclare son
`execution_mode` (`"io"` ou `"cpu"`) que le runner utilise pour
choisir entre `ThreadPoolExecutor` (cloud APIs) et
`ProcessPoolExecutor` (Tesseract, Pero).

Ajouter un nouveau moteur = créer une classe Python de ~50 lignes
qui implémente `_run_ocr(image_path) -> str` et déclare son
`execution_mode`.

### 4.2 Moteurs OCR livrés

| Moteur | Type | Mode d'exécution | Confidence native exposée ? |
|---|---|---|---|
| **Tesseract 5** | Local CLI | CPU (ProcessPool) | ✅ Sprint 47 (`image_to_data`) |
| **Pero OCR** | Local Python | CPU (ProcessPool) | ✅ Sprint 48 (`transcription_confidence` ligne) |
| **Mistral OCR** | Cloud API | IO (ThreadPool) | ✅ Sprint 49 (quand disponible côté API) |
| **Google Vision** | Cloud API | IO (ThreadPool) | ✅ Sprint 50 (`Word.confidence` en mode `DOCUMENT_TEXT_DETECTION`) |
| **Azure Doc Intelligence** | Cloud API | IO (ThreadPool) | ✅ Sprint 51 (`Word.confidence`) |

Quand un moteur expose ses confidences natives, le runner calcule
automatiquement les métriques de calibration (ECE, MCE, reliability
diagram — Sprint 39-43).

### 4.3 Robustesse runtime

- **Erreurs HTTP cloud** (4xx/5xx, timeout, body mal formé) :
  remontées dans `EngineResult.error` avec le code HTTP, jamais
  avalées silencieusement (Sprint A5 / m-10, 19 cas testés).
- **Crash isolé d'un document** : le runner continue avec les
  autres documents. Le doc en échec a `engine_error` rempli.
- **Cancel mid-run** : `cancel_event.set()` interrompt proprement.
- **Timeout par document** : configurable via paramètre
  `timeout_seconds`.

---

## 5. Module 3 — Pipelines OCR+LLM et pipelines composables

### 5.1 Pipelines OCR+LLM historiques (Sprint 3+)

L'unité de comparaison est le **concurrent** — pas forcément un
moteur seul, mais une chaîne produisant du texte à partir d'une
image.

| Mode | Description | Usage typique |
|---|---|---|
| `zero_shot` | Le LLM/VLM reçoit l'image directement et transcrit | Test si GPT-4o ou Claude peut remplacer un OCR sur des documents anciens |
| `post_correction_texte` | OCR → texte brut → LLM corrige le texte | LLM non multimodal (Llama local), grand volume |
| `post_correction_image_texte` | OCR → LLM reçoit image ET texte brut | Meilleure qualité ; le LLM voit le contexte visuel |

Les prompts sont **versionnés** dans `picarones/prompts/` (8 fichiers
FR + EN), embarqués dans le snapshot du rapport pour
reproductibilité.

### 5.2 Pipelines composables (Sprint 63+)

Au-delà des 3 modes historiques, Picarones livre une infrastructure
générique : un pipeline est une **liste d'étapes `BaseModule`**
qui produit un artefact à chaque étape (TEXT, ALTO, PAGE,
ENTITIES…) ; à chaque jonction, le runner calcule
**automatiquement** la métrique adéquate via `compute_at_junction`
(registre typé Sprint 34).

```yaml
# Spec YAML chargée par picarones pipeline run
name: ocr_then_corrector
steps:
  - name: ocr
    module: picarones.engines.tesseract.TesseractEngine
    args: { lang: "fra", psm: 6 }
  - name: post_correction
    module: my_module.MyLLMCorrector
    args: { model: "gpt-4o" }
```

`picarones pipeline compare specs.yaml --corpus ./scans --output rapport.html`
exécute N pipelines sur le même corpus et produit un rapport
comparatif. Conçu pour qu'un mainteneur tiers puisse contribuer
ses propres modules sans toucher au cœur de Picarones (cf.
`docs/developer/module-policy.md`, Sprint 97).

### 5.3 Détection d'over-normalisation LLM

Risque spécifique aux pipelines OCR+LLM : le LLM modernise à tort
des graphies historiques légitimes. Picarones mesure :

- **Modernisation lexicale** (Sprint 80) : top-N tokens GT
  systématiquement remplacés (`maistre → maître` dans 100 % des
  cas → signal exploitable).
- **Score d'absorption d'erreur** (Sprint 94) : à chaque jonction
  OCR→LLM, calcule le **taux de correction** (parmi les erreurs
  avant, combien corrigées) **et** le **taux d'introduction**
  (parmi les erreurs après, combien nouvelles). Distingue un
  module qui *corrige* d'un module qui *écrase*.
- **Delta Flesch** (Sprint 52) : sur les langues prises en
  charge, signale les LLM qui rendent le texte « trop moderne »
  par rapport à la GT.
- **Score d'ancrage** (Sprint 10) : proportion des trigrammes
  produits par le LLM qui s'ancrent dans la GT — score bas =
  hallucination probable.

---

## 6. Module 4 — Métriques et analyses

### 6.1 Catalogue exhaustif des métriques

Picarones livre **plus de 30 métriques** organisées en familles.
Pour chaque métrique : son nom, sa jonction de types, sa source,
ses limites — voir le **glossaire contextuel** intégré au rapport
HTML (25 entrées bilingues, ouvre via le `?` à côté du nom de
colonne) et `picarones/report/glossary/{fr,en}.yaml`.

#### Classique OCR/HTR

| Métrique | Jonction | Source primaire |
|---|---|---|
| CER (raw, NFC, caseless, diplomatique) | `(TEXT, TEXT)` | Levenshtein character / [jiwer](https://github.com/jitsi/jiwer) |
| WER, MER, WIL | `(TEXT, TEXT)` | jiwer |
| Bootstrap CI 95 % | dérivé | Efron (1979) |
| Distribution CER par ligne, Gini | dérivé | Sprint 10 |
| Détection hallucinations VLM (anchor score, length ratio) | dérivé | Sprint 10 |

#### Philologique (Sprints 52-60, 80, 84-85, 92-94)

| Métrique | Jonction | Cible patrimoniale |
|---|---|---|
| Couverture MUFI | `(TEXT, TEXT)` | Manuscrits médiévaux |
| Score d'expansion d'abréviations Capelli | `(TEXT, TEXT)` | Médiéval |
| Précision par bloc Unicode | `(TEXT, TEXT)` | Imprimés anciens / médiéval |
| Préservation des marqueurs typographiques de l'imprimé ancien (long-s, ligatures, tildes nasaux) | `(TEXT, TEXT)` | XVIᵉ-XVIIIᵉ |
| Marqueurs des archives modernes (titres, ordinaux, monnaies, état civil…) | `(TEXT, TEXT)` | XIXᵉ-XXᵉ |
| Préservation des numéraux romains (5 statuts) | `(TEXT, TEXT)` | Toutes périodes |
| Recherchabilité fuzzy (Levenshtein distance ≤ 2) | `(TEXT, TEXT)` | Indexation Elastic / full-text |
| Précision sur séquences numériques (dates, foliotation, monnaies) | `(TEXT, TEXT)` | Archives, économie historique |
| Modernisation lexicale (top-N tokens GT modernisés) | `(TEXT, TEXT)` | Pipelines OCR+LLM |
| Delta Flesch (FR + EN) | `(TEXT, TEXT)` | Repère VLM hallucinant du français moderne |
| Score d'absorption d'erreur par jonction | `(TEXT, TEXT)` | Pipelines composées |
| Précision sur entités nommées (HIPE) | `(ENTITIES, ENTITIES)` | Indexation prosopographique |
| Reading order F1 (ICDAR 2015) | `(READING_ORDER, READING_ORDER)` | Manuscrits glosés, journaux multi-colonnes |
| Layout F1 par type de région (IoU 0.5) | `(ALTO, ALTO)` | Texte/glose/marginalia |

#### Comparaison & décision (Sprints 18, 20, 35-37, 81, 89-92, 96)

| Métrique | Source primaire |
|---|---|
| Test multi-moteurs Friedman + post-hoc Nemenyi + CDD | Demšar (2006) |
| Test pairé Wilcoxon | Wilcoxon (1945) |
| Pareto coût / vitesse / CO₂ (multi-objectifs N dim) | Pareto (1896) |
| Divergence taxonomique inter-moteurs (Jensen-Shannon) | Lin (1991) |
| Oracle complementarity (recall borné supérieur) | Sprint 35 |
| Score de spécialisation inter-moteurs | Sprint 89 |
| Stabilité multi-runs (CV CER, accord identique) | Sprint 83 |
| Accord inter-annotateurs (Cohen κ, Krippendorff α) | Cohen (1960), Krippendorff (1970) |
| Tendance longitudinale + change-point Pettitt | Sprint 92 |
| Throughput effectif (pages/h après correction humaine 5s/erreur) | Sprint 91, HTR-United |
| Coût marginal par erreur évitée | Sprint 91 |
| Comparaison incrémentale ANOVA-like par slot | Sprint 96 |

**Note de traçabilité** : les références primaires (Demšar 2006,
Wilcoxon 1945, Efron 1979, etc.) sont citées dans les docstrings
de chaque fonction de `picarones/measurements/statistics/`.
Le glossaire contextuel relie chaque métrique à sa publication
canonique (champ `reference`).

### 6.2 Profils de normalisation

11 profils livrés (`picarones/measurements/normalization.py`,
exposés via `/api/normalization/profiles`) : `nfc`, `caseless`,
`minimal`, `medieval_french`, `early_modern_french`,
`medieval_latin`, `medieval_english`, `early_modern_english`,
`secretary_hand`, `sans_ponctuation`, `sans_apostrophes`.

Chaque profil applique un ensemble d'équivalences diplomatiques
(ſ=s, u=v, i=j, ꝑ=per, þ=th, etc.). Un profil custom peut être
chargé depuis YAML.

**Traçabilité aux standards éditoriaux** (MUFI v4.0, TEI P5
Unicode chapter 3.4, DEAF) : prévue Sprint A12 (item B-6 du
plan de remédiation institutionnelle).

### 6.3 Taxonomie des erreurs en 10 classes

Catégorisation automatique de chaque erreur (Sprint 5) :

1. Confusion visuelle (rn/m, l/1, O/0, u/n…)
2. Erreur diacritique
3. Erreur de casse
4. Ligature non résolue
5. Abréviation non développée
6. Hapax (mot absent du lexique)
7. Segmentation (fusion / fragmentation)
8. Hors-vocabulaire
9. Lacune (texte présent en GT, absent en OCR)
10. Sur-normalisation LLM

### 6.4 Score de difficulté intrinsèque

Indicateur calculé **indépendamment des moteurs** (Sprint 7) :

- variance du CER entre tous les concurrents (si tous ratent →
  document objectivement difficile),
- métriques de qualité image,
- densité de caractères spéciaux (ligatures, abréviations,
  diacritiques),
- longueur et densité du texte.

Sépare deux questions distinctes : *« est-ce que ce moteur est
mauvais ? »* vs *« est-ce que ce document est objectivement
difficile ? »*.

---

## 7. Module 5 — Rapport HTML interactif

Le rapport est un **fichier HTML auto-portant** (Jinja2 server-side,
Chart.js vendoré inline), lisible hors-ligne, embarquant toutes
les données et visualisations.

### 7.1 Cinq vues + sections globales

#### Sections globales (en tête)

- **Synthèse narrative factuelle** (Sprint 19+) : 3-5 phrases
  produites par 20+ détecteurs déterministes. Chaque chiffre
  rendu est traçable au payload du `Fact` correspondant
  (anti-hallucination prouvé par test).
- **Critical Difference Diagram** (Sprint 18) : SVG server-side,
  Friedman + post-hoc Nemenyi.
- **Section inter-moteurs** (Sprint 37) : matrice de divergence
  taxonomique + encart oracle complementarity.
- **Front Pareto** (Sprint 20) : coût / vitesse / CO₂ avec
  toggles d'axes.
- **Section leviers d'amélioration** (Sprint 51-82) : 5 leviers
  factuels (taxonomie récupérable, concentration Pareto,
  complémentarité, modernisation lexicale, déficit projeté de
  robustesse).

#### Vue Classement (Ranking)

Tableau triable : CER (médiane par défaut depuis Sprint 44),
WER, MER, WIL, scores ligatures et diacritiques, Gini, score
d'ancrage, sur-normalisation, etc. Vue stratifiée optionnelle
par `script_type` (Sprint 45-46).

#### Vue Galerie (Gallery)

Grille de vignettes avec badge CER coloré. Filtres dynamiques
(CER > X, qualité image, type de script, longueur GT). Tri
multi-critères. Vue **« Worst lines globale »** (Sprint 72)
qui transcende les documents et liste les lignes individuelles
les plus mal transcrites.

#### Vue Document

Image originale + diff token coloré façon GitHub par moteur,
scroll synchronisé N-way. Vue spécifique OCR+LLM : triple diff
GT / sortie OCR brute / sortie après LLM.

#### Vue Analyses

Distribution CER (histogramme + densité), scatter plots
qualité image vs CER, heatmap de confusion de caractères,
diagrammes de fiabilité (calibration ECE/MCE — Sprint 43),
graphiques de bootstrap CI 95 %, profil philologique par moteur
(Sprint 62), throughput effectif (Sprint 91), tendances
longitudinales (Sprint 92), DAG de pipeline composée (Sprint 95),
etc.

#### Vue Caractères

Matrice de confusion Unicode interactive, tableau des
caractères les plus souvent manqués par chaque moteur, CER par
bloc Unicode (Sprint 55), analyse des ligatures.

### 7.2 Panneaux latéraux

- **Glossaire contextuel** (Sprint 21) : `?` à côté de chaque
  en-tête de colonne ; clic ouvre un panneau avec définition,
  ce qu'on mesure, usage, limites, référence primaire (25
  entrées bilingues).
- **Mode avancé** (Sprint 21) : choix de colonnes visibles,
  filtres par strate, opt-in score composite personnel
  (curseurs à 0 par défaut, formule visible, warning explicite
  « il n'existe pas de pondération universellement valide »),
  toggle palette daltonien-friendly (Sprint A7), URL stateful.

### 7.3 Exports

| Format | Statut |
|---|---|
| HTML autonome | ✅ Livré |
| CSV (vue courante avec filtres) | ✅ Livré |
| JSON (BenchmarkResult complet) | ✅ Livré |
| Snapshot reproductibilité (versions, commit, lock) | ✅ Sprint 27 |
| Lazy images (rapport HTML + dossier `report-assets/`) | ✅ Sprint A5 / M-16 |
| PDF | ❌ Non livré (cf. §10) |
| ALTO XML / PAGE XML / images annotées | ❌ Non livré (cf. §10) |

### 7.4 Accessibilité

Conformité WCAG 2.1 niveau AA (cf.
[`ACCESSIBILITY.md`](ACCESSIBILITY.md)) :

- Skip-to-content link (WCAG 2.4.1).
- `role="img"` + `aria-label` + table de données jumelle
  pour chaque graphique Chart.js (WCAG 1.1.1).
- `scope="col"` sur tous les `<th>`.
- Palette par défaut Okabe-Ito (daltonien-friendly), toggle vers
  l'ancienne palette via panneau Avancé ou `?palette=classic`.
- Bilinguisme intégral (skip-link, ARIA labels, captions des
  tables jumelles).
- Audit RGAA externe planifié Sprint A15.

---

## 8. Module 6 — Interface web et CLI

### 8.1 Interface web FastAPI

- Configuration de benchmark : sélection corpus, moteurs,
  normalisation.
- Streaming SSE de la progression en temps réel (`Last-Event-ID`
  reconnexion supportée — Sprint 26).
- Persistance des jobs en SQLite (mode WAL, thread-safe), reprise
  des jobs orphelins au boot.
- Upload ZIP depuis le navigateur.
- Imports HTR-United / HuggingFace via formulaire.
- Bilingue FR/EN.
- Healthcheck minimal `/health` (Sprint A4 / M-3).
- Token CSRF (`/api/csrf/token`) + middleware (Sprint A4 / B-11)
  activable via `PICARONES_CSRF_REQUIRED=1` pour les déploiements
  institutionnels derrière SSO.

### 8.2 Interface en ligne de commande (Click)

15 commandes :

```bash
picarones run         # benchmark
picarones report      # rapport HTML depuis JSON
picarones demo        # rapport démo synthétique
picarones compare     # compare deux runs JSON, exit-code 2 si régression
picarones diagnose    # workflow bench + leviers + recommandations factuelles
picarones economics   # workflow bench + throughput + coût projeté
picarones edition     # workflow bench + métriques philologiques
picarones pipeline    # run/compare pipelines composées YAML
picarones import      # IIIF / HF / HTR-United
picarones serve       # interface web locale
picarones history     # historique longitudinal SQLite
picarones robustness  # courbes CER vs dégradation
picarones engines     # liste les moteurs disponibles
picarones metrics     # CER/WER entre deux fichiers texte
picarones info        # version + system info
```

Toutes les commandes supportent `--help`. Workflows pré-câblés
(`diagnose`, `economics`, `edition`) sont des combinaisons
canoniques pour les profils utilisateurs typiques.

### 8.3 Intégration CI/CD

- Mode headless (`--no-progress`).
- Output JSON machine-readable.
- Exit code 2 sur `picarones compare` si régression CER détectée.
- Workflow GitHub Actions `perf_regression.yml` (Sprint A5 /
  M-14) — cron hebdomadaire + sur PR touchant le runner.

---

## 9. Reproductibilité et sécurité

### 9.1 Snapshots de reproductibilité (Sprint 27)

Chaque rapport HTML embarque un dict `report_data["snapshots"]`
qui contient :

- **pricing** — YAML brut intégral de `picarones/data/pricing.yaml`
  utilisé.
- **glossary** — entrées du glossaire effectivement référencées.
- **normalization** — profil sérialisé.
- **environment** — version Picarones, Python, plateforme, commit
  git, paquets installés (top 200).

Procédure complète de re-jeu d'un benchmark à 5 ans d'écart :
[`docs/reference/reproducibility-snapshots.md`](docs/reference/reproducibility-snapshots.md)
(Sprint A8 / M-12).

### 9.2 Reproductibilité des builds

- Lock files `requirements.lock` + `requirements-dev.lock`
  générés via `uv pip compile` (Sprint A8).
- Image Docker épinglée à un patch précis via `ARG PYTHON_BASE_IMAGE`
  (rotation trimestrielle).
- Release pipeline GitHub Actions (Sprint A9) : tag `v*.*.*` →
  PyPI via OIDC trust + ghcr.io multi-arch + GitHub Release auto.

### 9.3 Sécurité institutionnelle

- **Mode public** (`PICARONES_PUBLIC_MODE=1`) : refuse les moteurs
  cloud mutualisés et les pipelines LLM facturés à la clef serveur.
- **CSRF** double-submit (`PICARONES_CSRF_REQUIRED=1`) — Sprint A4.
- **XML défendu** par `defusedxml` partout (XXE / Billion Laughs).
- **Zip-slip prévenu** par `Path(member.filename).name`.
- **Validation Pillow** systématique (CVE bombes de
  décompression).
- **Rate limiting** par IP + sémaphore de jobs concurrents.
- **CSP + en-têtes durcis** (X-Content-Type-Options,
  Referrer-Policy).

Voir [`SECURITY.md`](SECURITY.md) pour la procédure complète.

### 9.4 RGPD et rétention

Politique documentée dans
[`docs/operations/data-retention-rgpd.md`](docs/operations/data-retention-rgpd.md)
(Sprint A11 / M-8). Purge automatique des uploads anciens
configurable via `PICARONES_UPLOAD_RETENTION_DAYS=7` par défaut.

---

## 10. Limites assumées et non-fonctionnalités

Cette section décrit explicitement **ce que Picarones ne fait pas
et ne fera pas dans la v1.x**. Plusieurs items étaient promis dans
la SPECS v1 (Mars 2025) — leur abandon est un choix éditorial
documenté ci-dessous, pas un oubli.

<!-- specs-check: known-abandoned-start -->

> Toutes les fonctionnalités listées ci-dessous étaient promises ou
> évoquées dans SPECS v1 et sont **explicitement abandonnées,
> non implémentées ou reportées** dans la v2.0 de ce document.
> Le test ``tests/docs/test_specs_consistency.py`` (Sprint A2)
> détecte cette section comme la déclaration officielle des
> non-fonctionnalités du projet.

### 10.1 Adapters OCR non livrés

- **Kraken** : prévu v1.0 dans SPECS v1, jamais implémenté. Choix :
  ouverture en plugins externes via la politique de modules
  contribués (Sprint 97), pas un adapter intégré au cœur. Un
  utilisateur peut écrire son `KrakenEngine(BaseOCREngine)` et
  l'exécuter via les pipelines composables.
- **AWS Textract** : prévu v1.1, abandonné. Pas de DPA Amazon
  signé, et le périmètre patrimonial est mieux servi par les
  trois clouds déjà intégrés (Mistral OCR, Google Vision, Azure
  DI).
- **Calamari** : prévu v1.1, abandonné. Maintenance d'un adapter
  par moteur ≈ 50 PJ/an ; mieux vaut concentrer sur les 5 adapters
  livrés et ouvrir Calamari en plugin externe.
- **OCRopus4** : prévu v1.2, abandonné — projet historique en
  fin de vie.
- **Moteur custom YAML** (`type: cli` / `type: api`) : prévu en
  SPECS v1.0, abandonné. **Refondu en pipelines composables**
  (Sprint 63-70) qui permettent de brancher n'importe quel module
  via une spec YAML — plus puissant que la déclaration d'engine
  custom imaginée à l'origine.

### 10.2 Exports non livrés

- **Export PDF** du rapport. CSV + JSON + HTML autonome couvrent
  les usages observés. Reportable sur demande utilisateur si
  besoin tracé.
- **Export ALTO XML** des sorties OCR.
- **Export PAGE XML** des sorties OCR.
- **Export images annotées** (PNG avec zones d'erreur surlignées).

### 10.3 Fonctionnalités explicitement abandonnées

- **Recommandation automatique** « quel concurrent pour quel
  usage ». Promise dans SPECS v1 §7.1, **abandonnée** au profit
  du moteur narratif factuel (Sprint 19) et de la philosophie
  « Picarones mesure et classe — il ne tranche pas ». Les leviers
  d'amélioration (Sprint 51-82) restent factuels.
- **Score de consensus / vote majoritaire / ensemble** : Picarones
  livre l'**oracle borné supérieur** (Sprint 35) et le score de
  spécialisation inter-moteurs (Sprint 89) — observations
  factuelles. Pas de mécanisme de vote actif intégré ; au
  chercheur de combiner les sorties s'il le décide.
- **Clustering automatique k-means des erreurs**. Remplacé par
  la taxonomie discrète (Sprint 5) + co-occurrence Jaccard
  (Sprint 75) + heatmap intra-doc (Sprint 76).
- **Annotations inline du paléographe exportées en JSON**. Non
  implémentées.
- **Badge SVG de qualité OCR pour CI**. `picarones compare` avec
  exit code 2 sur régression couvre l'usage CI ; un badge SVG
  reste nice-to-have, non priorisé.
- **Dataset de référence embarqué de 100 documents
  patrimoniaux**. Picarones est volontairement un **banc d'essai
  sur votre golden dataset** — le 100-doc corpus de référence
  imaginé en SPECS v1 §3.3 entrerait en concurrence avec les
  corpus institutionnels existants (HTR-United, Esposalles,
  IAM, RIMES, READ-BAD) et en fragmenterait l'écosystème. Les
  5 documents synthétiques de Sprint A5 (`tests/fixtures/reference_corpus/`)
  servent uniquement à l'anti-régression CER en CI, pas à la
  valeur scientifique.

### 10.4 Fonctionnalités scientifiques planifiées

À livrer dans des sprints futurs :

- **CITATION.cff + DOI Zenodo + papier JOSS** (Sprint A12 du
  plan institutionnel) — débloque la citation académique propre.
- **Traçabilité des profils de normalisation aux standards
  éditoriaux** (MUFI v4.0, TEI P5, DEAF) — Sprint A12.
- **Citations primaires des méthodes statistiques** dans les
  docstrings (Demšar 2006, Wilcoxon 1945, Efron 1979) —
  Sprint A12.

<!-- specs-check: known-abandoned-end -->

---

## 11. Roadmap d'évolution

Trois documents complémentaires pilotent l'évolution :

- [`CHANGELOG.md`](CHANGELOG.md) — historique sprint par sprint,
  format Keep a Changelog.
- [`docs/roadmap/evolution-2026.md`](docs/roadmap/evolution-2026.md) —
  roadmap technique 2026+ (axes A et B : nouvelles métriques et
  pipelines composables).
- [`docs/audits/`](docs/audits/) — audits institutionnels et
  plans de remédiation (sprints A1 à A15 du plan en cours).

L'**état du plan institutionnel** au 2 mai 2026 :

| Phase | Sprints | Statut |
|---|---|---|
| Phase 0 — Garde-fous CI | A1, A2 | ✅ Terminée |
| Phase 1 — Hygiène architecturale | A3 | ✅ Terminée |
| Phase 2 — Robustesse runtime | A4, A5 | ✅ Terminée |
| Phase 3 — Accessibilité | A6, A7 | ✅ Terminée |
| Phase 4 — Reproductibilité ops | A8, A9 | ✅ Terminée |
| Phase 5 — Gouvernance | A10, A11 | ✅ Terminée |
| Phase 7 — Refonte doc produit | A13, **A14 (ce document)** | ✅ Terminée |
| Phase 6 — Publication scientifique | A12 | ⏳ Planifiée |
| Phase 8 — Validation externe | A15 | ⏳ Planifiée (calendrier externe) |

---

## 12. Migration v1 → v2 — annexe historique

Pour les lecteurs qui avaient pris connaissance de SPECS v1.0
(mars 2025) ou de l'Addendum Sprints 16-30, voici la table de
migration des promesses changées :

| SPECS v1 disait | SPECS v2 documente | Raison |
|---|---|---|
| Adapter Kraken (priorité v1.0) | Ouvert en plugin externe | Politique modules contribués Sprint 97 ; concentration sur 5 adapters cœur. |
| Adapter AWS Textract (v1.1) | Abandonné | Pas de DPA, périmètre couvert par 3 clouds existants. |
| Adapter Calamari (v1.1) | Abandonné | Maintenance par adapter ≈ 50 PJ/an ; mieux servi en plugin externe. |
| Adapter OCRopus4 (v1.2) | Abandonné | Projet historique en fin de vie. |
| Moteur custom YAML | Refondu en pipelines composables | Sprint 63-70 livre une infrastructure plus puissante. |
| Recommandation automatique | Remplacée par moteur narratif factuel | Pivot philosophique vers la neutralité éditoriale. |
| Export PDF | Abandonné | CSV + JSON + HTML couvrent les usages. |
| Export ALTO/PAGE/images annotées | Abandonné | Idem. |
| `picarones estimate` (preview coût) | Remplacé par vue Pareto post hoc | Sprint 20 livre la même information dans le rapport. |
| Score consensus / k-means | Remplacé par oracle borné + taxonomie discrète + Jaccard | Sprint 35, 5, 75 — équivalence fonctionnelle, formalisme différent. |
| Annotations inline JSON | Abandonné | Pas de demande utilisateur observée. |
| Badge SVG qualité OCR | Abandonné | `picarones compare` exit code 2 couvre la CI. |
| Dataset 100 docs embarqué | Abandonné | Banc d'essai sur votre golden dataset, pas un dataset de référence. |
| Prompt latin | Pas livré | Reportable sur demande. |

À l'inverse, **~25 modules majeurs ajoutés depuis Sprint 30** sont
documentés dans la nouvelle SPECS aux §6 (NER, reading order F1,
layout F1, recherchabilité fuzzy, séquences numériques, 6 modules
philologiques transversaux, narrative engine, Friedman+Nemenyi+CDD,
Pareto, glossaire, métriques inter-moteurs, absorption d'erreur,
pipelines composables, registre typé, audit modules, comparaison
de runs, stratification, calibration, longitudinal, throughput
effectif, etc.) — invisibles dans SPECS v1.

---

*Picarones est conçu pour devenir une référence open-source
d'évaluation OCR/HTR dans le champ patrimonial — métriques
adaptées aux documents historiques, pipelines composables,
intégration des standards bibliothéconomiques (IIIF, ALTO XML,
PAGE XML, HTR-United, eScriptorium, Gallica), rapport interactif
exportable, snapshot de reproductibilité.*

*Dernière mise à jour : 2 mai 2026 (Sprint A14, refonte v2.0).*
