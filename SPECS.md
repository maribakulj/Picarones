# Picarones — Spécifications Fonctionnelles et Techniques

> **Plateforme de comparaison et d'évaluation de moteurs OCR/HTR et de pipelines OCR+LLM pour documents patrimoniaux**
>
> Version 1.0 — Mars 2025

---

## Table des matières

1. [Vision générale et positionnement](#1-vision-générale-et-positionnement)
2. [Architecture générale](#2-architecture-générale)
3. [Module 1 — Gestion des corpus et imports](#3-module-1--gestion-des-corpus-et-imports)
4. [Module 2 — Adaptateurs moteurs OCR](#4-module-2--adaptateurs-moteurs-ocr)
5. [Module 3 — Pipelines OCR+LLM](#5-module-3--pipelines-ocrllm)
6. [Module 4 — Métriques et analyse](#6-module-4--métriques-et-analyse)
7. [Module 5 — Rapport interactif HTML](#7-module-5--rapport-interactif-html)
8. [Module 6 — Interface de lancement et CLI](#8-module-6--interface-de-lancement-et-cli)
9. [Fonctionnalités avancées](#9-fonctionnalités-avancées)
10. [Plan de développement](#10-plan-de-développement)
11. [Exigences non fonctionnelles](#11-exigences-non-fonctionnelles)

---

## 1. Vision générale et positionnement

### 1.1 Problématique

Les équipes OCR/HTR travaillant sur des fonds patrimoniaux (manuscrits, imprimés anciens, archives) disposent d'un paysage de moteurs hétérogène — moteurs locaux (Tesseract, Pero OCR, Kraken), solutions cloud (Mistral OCR, Google Vision, AWS Textract), modèles fine-tunés maison — sans outil unifié pour les comparer rigoureusement sur leurs propres corpus.

Les outils existants (ocrevalUAtion, dinglehopper) sont soit obsolètes, soit limités au CER/WER, soit non adaptés aux spécificités des documents historiques : glyphes anciens, ligatures, abréviations, graphies variables, pathologies d'image.

À cela s'ajoute une question de recherche émergente : **est-ce qu'une couche de correction par LLM améliore réellement la sortie OCR, de combien, sur quels types d'erreurs ?** Aucun outil existant ne permet de tester et mesurer cela rigoureusement.

**Picarones** comble ce vide en proposant une plateforme complète, open-source, pensée pour le milieu patrimonial, capable de comparer des moteurs OCR seuls **et** des pipelines OCR+LLM.

### 1.2 Objectifs stratégiques

- Permettre une évaluation rigoureuse, reproductible et multi-dimensionnelle des moteurs OCR/HTR sur des corpus patrimoniaux réels
- Évaluer l'apport réel des LLMs en post-correction OCR, en termes de qualité
- Produire des rapports interactifs exploitables par des profils variés : ingénieurs, chercheurs, responsables de projets de numérisation
- S'intégrer dans les workflows patrimoniaux existants (IIIF, eScriptorium, HTR-United, Gallica)
- Offrir une base extensible pour le suivi longitudinal de la qualité OCR dans le temps

### 1.3 Utilisateurs cibles

| Profil | Besoins principaux |
|---|---|
| Ingénieur OCR/ML | Pipeline programmatique, métriques fines, export JSON/CSV, CI/CD |
| Chargé de numérisation | Rapport visuel, comparaison simple A vs B, recommandation de moteur |
| Responsable de projet | Vue agrégée, graphiques, export PDF, analyse coût/bénéfice des APIs |
| Chercheur en humanités numériques | Métriques diplomatiques, corpus HTR-United, analyse des erreurs par catégorie |
| Paléographe | Diff visuel sur l'image, annotation inline des cas difficiles |

### 1.4 Proposition de valeur unique

Picarones est le seul outil combinant :
1. **Métriques adaptées aux documents historiques** (glyphes, ligatures, diacritiques, abréviations, normalisation diplomatique)
2. **Évaluation des pipelines OCR+LLM** avec mesure du delta de qualité
3. **Intégration native des standards bibliothéconomiques** (IIIF, ALTO, PAGE XML, HTR-United, eScriptorium, Gallica)
4. **Rapport interactif auto-contenu** exploitable sans compétences techniques

---

## 2. Architecture générale

### 2.1 Vue d'ensemble

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              PICARONES                                   │
│                                                                          │
│  ┌─────────────────┐   ┌──────────────────────┐   ┌─────────────────┐   │
│  │  Import /        │   │  Pipeline            │   │  Rapport        │   │
│  │  Corpus Mgmt    │──▶│  Orchestrator        │──▶│  Interactif     │   │
│  └─────────────────┘   └──────────────────────┘   └─────────────────┘   │
│         ▲                        │                        │              │
│   IIIF / Gallica          ┌──────▼──────┐           HTML self-contained  │
│   HTR-United              │  Moteurs    │           Export PDF/CSV/ALTO  │
│   HuggingFace             │  OCR        │                                │
│   eScriptorium            └──────┬──────┘                                │
│   Dossier local                  │                                       │
│                           ┌──────▼──────┐                                │
│                           │  Couche     │                                │
│                           │  LLM        │  ◀── optionnelle               │
│                           │  (optionel) │                                │
│                           └─────────────┘                                │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Stack technique

| Couche | Technologie | Justification |
|---|---|---|
| Backend / Pipeline | Python 3.11+ | Écosystème OCR mature, jiwer, Pillow, NumPy |
| API serveur | FastAPI | Async, auto-documentation OpenAPI, léger |
| Rapport interactif | HTML + Vanilla JS + Chart.js + diff2html | Zéro dépendance runtime, fichier unique transportable |
| Configuration | YAML | Déclaration simple des moteurs et paramètres |
| Stockage résultats | JSON + SQLite optionnel | Léger, portable, requêtable |
| CLI | Click (Python) | Usage sans interface, intégration CI/CD |

### 2.3 Principes d'architecture

- **Moteur-agnostique** : chaque moteur OCR ou LLM est un adaptateur interchangeable, déclaré en YAML
- **Pipeline composable** : un "concurrent" peut être un moteur seul (`tesseract`), un pipeline (`tesseract → gpt-4o`), ou un LLM seul en mode zero-shot (`gpt-4o`)
- **Pipeline idempotent** : les sorties sont cachées par hash d'image, relance partielle possible
- **Rapport auto-contenu** : le fichier HTML final embarque toutes les données, lisible hors-ligne
- **Traçabilité complète** : versions des moteurs, paramètres, prompts LLM utilisés, dates d'exécution — tout est loggé dans les métadonnées du rapport

---

## 3. Module 1 — Gestion des corpus et imports

### 3.1 Formats de vérité terrain acceptés

| Format | Extension | Usage typique |
|---|---|---|
| Texte brut parallèle | `image.jpg` + `image.gt.txt` | Convention Tesseract, HTR-United |
| ALTO XML | `.alto.xml` | Standard bibliothèques nationales, eScriptorium export |
| PAGE XML | `.page.xml` | Transkribus, OCRopus |
| JSON HuggingFace | `dataset.json` | HuggingFace Datasets |
| eScriptorium export | `.zip` (PAGE+images) | Export natif eScriptorium |
| CSV simple | `image,texte_gt` | Exports maison, tableaux Callico |

### 3.2 Sources d'import

#### 3.2.1 Dossier local
Import d'un dossier contenant des paires image/GT. Détection automatique du format. Prévisualisation du corpus avant lancement (nombre de documents, longueur moyenne de GT, aperçu des images).

#### 3.2.2 Import IIIF — fonctionnalité clé

L'intégration IIIF est la fonctionnalité d'import la plus stratégique pour le contexte patrimonial. Elle permet d'accéder directement aux fonds numérisés de toutes les grandes bibliothèques sans téléchargement manuel.

- Import par URL de manifeste IIIF v2 et v3
- Sélection des canvas (pages) via interface de sélection visuelle
- Récupération des annotations de transcription si le manifeste les contient
- Compatibilité : Gallica (BnF), Bodleian, BL, BSB, e-codices, Europeana et tout entrepôt IIIF-compliant
- Résolution configurable des images

> **Exemple :** coller l'URL du manifeste Gallica d'un incunable, sélectionner 50 pages, lancer le benchmark. La GT est fournie séparément ou issue d'un export eScriptorium.

#### 3.2.3 Import HuggingFace Datasets
- Recherche et prévisualisation de datasets OCR/HTR
- Filtrage par langue, type de script, époque, institution
- Datasets patrimoniaux pré-référencés : IAM, RIMES, READ-BAD, Esposalles, Bozen-Baptism, datasets HTR-United
- Import partiel : sous-ensemble aléatoire ou filtré
- Cache local avec gestion des versions

#### 3.2.4 Import HTR-United
- Listing et recherche dans le catalogue HTR-United
- Import direct des corpus publiés (Bréviaires, chartes médiévales, registres paroissiaux, presse ancienne...)
- Lecture des métadonnées : langue, script, institution, époque, nombre de lignes

#### 3.2.5 Import Gallica (API BnF)
- Recherche dans Gallica par cote, titre, auteur, date
- Récupération des images via API IIIF Gallica
- Récupération de l'OCR Gallica existant comme moteur de référence ou GT partielle

#### 3.2.6 Import eScriptorium
- Connexion à une instance eScriptorium locale ou distante via API
- Import de projets, documents et transcriptions
- Export des résultats de benchmark vers eScriptorium

### 3.3 Gestion des corpus

- Corpus nommés et versionnés avec métadonnées descriptives
- Tags : type de script (gothique, humanistique, caroline, textura, cursive, imprimé ancien...), langue, siècle, institution, état de conservation
- Statistiques : distribution des longueurs de GT, histogramme des scores de qualité image, aperçu des caractères unicode présents
- Partage de corpus (format JSON exportable/importable)
- Corpus de référence fournis avec Picarones : 100 documents représentatifs multi-scripts pour benchmarks rapides

---

## 4. Module 2 — Adaptateurs moteurs OCR

### 4.1 Architecture des adaptateurs

Chaque moteur OCR est un adaptateur Python standardisé exposant une interface commune. Ajouter un nouveau moteur = créer un fichier YAML de configuration et, si nécessaire, une classe Python de ~30 lignes.

### 4.2 Moteurs OCR supportés nativement

| Moteur | Type | Priorité | Notes |
|---|---|---|---|
| Tesseract 5 | Local CLI | v1.0 | `pytesseract`, multi-langues, LSTM |
| Pero OCR | Local Python | v1.0 | Excellent sur documents historiques |
| Kraken | Local Python | v1.0 | Référence HTR manuscrits, compatible eScriptorium |
| Mistral OCR | API REST | v1.0 | Mistral OCR 3, multimodal |
| Google Vision | API REST | v1.1 | Document AI, bonne couverture unicode |
| AWS Textract | API REST | v1.1 | Détection layout avancée |
| Azure Document Intelligence | API REST | v1.1 | Anciennement Form Recognizer |
| Calamari | Local Python | v1.1 | Basé TF, modèles pré-entraînés HTR |
| OCRopus4 | Local Python | v1.2 | Historique, utile pour comparaison |
| Moteur custom | CLI/API YAML | v1.0 | Déclaration YAML, aucun code requis |

### 4.3 Configuration d'un moteur (YAML)

```yaml
# Moteur custom via CLI
name: mon_ocr_interne
type: cli
command: "mon_ocr {input_image} --output {output_file} --lang fra"
output_format: txt
version_command: "mon_ocr --version"

# Moteur via API REST
name: api_ocr_bnf
type: api
endpoint: http://localhost:8080/ocr
method: POST
image_field: file
response_path: $.result.text
```

### 4.4 Gestion de l'exécution

- Parallélisation configurable : N moteurs tournent en parallèle
- Cache des sorties par hash SHA-256 de l'image — relance partielle possible
- Timeout configurable par moteur, avec rapport d'erreur si dépassé
- Retry automatique sur erreur transitoire (rate limit, timeout réseau) avec backoff exponentiel
- Rapport d'avancement en temps réel : barre de progression par moteur, ETA
- Mode dry-run : validation de la configuration sans lancer les moteurs

---

## 5. Module 3 — Pipelines OCR+LLM

> Ce module est la fonctionnalité la plus originale de Picarones. Il permet de tester l'apport réel d'une couche de correction LLM sur une sortie OCR, et de comparer des pipelines complets entre eux.

### 5.1 Concept de "concurrent"

Dans Picarones, l'unité de comparaison est le **concurrent** — pas forcément un moteur OCR seul, mais n'importe quelle combinaison produisant du texte à partir d'une image :

| Type de concurrent | Description | Exemple |
|---|---|---|
| OCR seul | Un moteur OCR classique | `tesseract` |
| LLM zero-shot | Le LLM reçoit uniquement l'image | `gpt-4o` en mode vision |
| OCR → LLM (texte) | Le LLM reçoit la sortie OCR brute et corrige | `tesseract → mistral-large` |
| OCR → LLM (image + texte) | Le LLM reçoit image ET sortie OCR | `pero_ocr → gpt-4o` |
| OCR → LLM → LLM | Chaîne de correction en deux passes | `tesseract → llm1 → llm2` |

Ce modèle composable permet de tester toutes les configurations imaginables et de mesurer l'apport exact de chaque couche.

### 5.2 LLMs supportés

| LLM | Type | Modes supportés | Priorité |
|---|---|---|---|
| GPT-4o / GPT-4o mini | API OpenAI | texte seul, image+texte, zero-shot | v1.0 |
| Claude Sonnet / Haiku | API Anthropic | texte seul, image+texte, zero-shot | v1.0 |
| Mistral Large / Pixtral | API Mistral | texte seul, image+texte, zero-shot | v1.0 |
| Llama 3 (via Ollama) | Local | texte seul | v1.1 |
| Gemma / Phi (via Ollama) | Local | texte seul | v1.1 |
| LLM custom | API REST YAML | configurable | v1.0 |

### 5.3 Modes de correction LLM

#### Mode 1 — Post-correction texte brut
Le LLM reçoit uniquement la sortie OCR textuelle et un prompt de correction. Le plus rapide.

```
[Sortie OCR brute] ──▶ [LLM + prompt] ──▶ [Texte corrigé]
```

**Usage typique :** correction rapide sur grand volume, LLM non multimodal (Llama local), test de la valeur ajoutée d'un LLM de correction pur.

#### Mode 2 — Post-correction avec image
Le LLM reçoit l'image originale ET la sortie OCR. Permet au LLM de vérifier visuellement les passages ambigus.

```
[Image] ──────────────┐
                       ▼
[Sortie OCR brute] ──▶ [LLM multimodal + prompt] ──▶ [Texte corrigé]
```

**Usage typique :** meilleure qualité, test de la valeur ajoutée du contexte visuel pour la correction.

#### Mode 3 — Zero-shot LLM
Le LLM reçoit uniquement l'image, sans sortie OCR préalable. Teste la capacité de transcription native du LLM.

```
[Image] ──▶ [LLM multimodal + prompt] ──▶ [Transcription]
```

**Usage typique :** évaluer si GPT-4o ou Claude peut remplacer un moteur OCR sur des documents patrimoniaux.

### 5.4 Système de prompts

Les prompts LLM sont configurables, versionnés et font partie intégrante des métadonnées du rapport.

```yaml
# Configuration d'un concurrent OCR+LLM
name: tesseract_gpt4o_correction
type: pipeline
steps:
  - engine: tesseract
    config:
      lang: fra
      psm: 6
  - llm: gpt-4o
    mode: text_and_image   # text_only | text_and_image | zero_shot
    prompt: prompts/correction_medieval_french.txt
    temperature: 0.0
    max_tokens: 4096
```

```
# prompts/correction_medieval_french.txt
Tu es un expert en paléographie et en transcription de documents en français médiéval.
On te fournit la sortie brute d'un moteur OCR et l'image originale du document.
Ta tâche est de corriger les erreurs de transcription en te basant sur :
- Le contexte linguistique (français médiéval, XVe siècle)
- Les confusions visuelles typiques de l'OCR (rn/m, l/1, u/n, ſ/f...)
- Les abréviations et ligatures médiévales visibles sur l'image

Retourne UNIQUEMENT le texte corrigé, sans commentaire ni explication.
Conserve fidèlement la graphie originale (ne modernise pas l'orthographe).

OCR BRUT :
{ocr_output}
```

- Bibliothèque de prompts intégrée : prompts optimisés pour manuscrits médiévaux, imprimés anciens, cursives administratives, latin, documents mixtes
- Versionning des prompts : le prompt exact utilisé est stocké dans le JSON de résultats
- Comparaison de prompts : tester différents prompts sur le même concurrent OCR+LLM

### 5.5 Questions de recherche adressées par ce module

Picarones permet de répondre empiriquement, sur vos propres corpus, à des questions qui font débat :

1. **Un LLM améliore-t-il systématiquement la sortie OCR ?** (Pas toujours — il peut halluciner)
2. **Le mode image+texte est-il meilleur que texte seul ?** (Coût plus élevé, apport variable)
3. **Un LLM zero-shot peut-il remplacer un moteur OCR sur des documents anciens ?**
4. **Sur quels types d'erreurs le LLM apporte-t-il le plus ?** (Diacritiques ? Abréviations ? Hapax ?)
5. **Y a-t-il un risque de sur-normalisation ?** Le LLM modernise-t-il à tort la graphie médiévale ?
6. **Quel est le seuil de CER OCR en dessous duquel un LLM n'apporte plus rien ?**
7. **Quel est le seuil de CER OCR en dessous duquel un LLM n'apporte plus rien ?**

---

## 6. Module 4 — Métriques et analyse

> L'objectif est de fournir la vision la plus complète possible, adaptée aux spécificités des documents patrimoniaux — bien au-delà du CER/WER brut.

### 6.1 Métriques de base

#### CER — Character Error Rate
- CER brut (distance d'édition caractère / longueur GT)
- CER avec normalisation Unicode NFC
- CER sans distinction de casse
- CER diplomatique : avec table de correspondances historiques (ſ=s, u=v, i=j...)
- CER par ligne : distribution, médiane, percentiles P90/P95
- Intervalles de confiance à 95% par bootstrap (1000 itérations)

#### WER — Word Error Rate
- WER brut et normalisé
- WER avec tokenisation historique (traits d'union, abréviations)
- Match Error Rate (MER) et Word Information Lost (WIL)

#### Métriques de précision/rappel
- Précision et rappel au niveau caractère, mot, ligne
- F1-score global et par classe de caractère

### 6.2 Métriques spécifiques aux documents patrimoniaux

#### Glyphes et caractères spéciaux
- **Matrice de confusion unicode** : quels caractères GT sont transcrits par quels caractères OCR — fingerprint de chaque moteur
- **CER par bloc Unicode** : Latin de Base / Latin Étendu A & B / Diacritiques combinants / Formes de présentation latines
- **Score ligatures** : fi, fl, ff, ffi, ffl, st, ct, œ, æ, ꝑ, ꝓ...
- **Score abréviations** : taux de restitution correcte des formes abrégées
- **Précision diacritiques** : taux de conservation des accents, cédilles, trémas
- **Précision chiffres romains et arabes** séparément

#### Analyse structurelle
- **Score d'ordre de lecture** : les blocs sont-ils dans l'ordre logique ? Critique pour documents multi-colonnes, marginalia, réclames
- **Taux de segmentation des lignes** : fusion abusive / fragmentation — indépendant du contenu
- **Conservation des sauts de paragraphe et de section**
- **Détection des transpositions de blocs**
- **Score de mise en page** (si bounding boxes disponibles) : IoU entre zones détectées et zones GT

#### Taxonomie des erreurs
Catégorisation automatique de chaque erreur :

| Classe | Description |
|---|---|
| 1 — Confusion visuelle | Caractères morphologiquement proches (rn/m, l/1, O/0, u/n...) |
| 2 — Erreur diacritique | Accent manquant, mauvais accent, cédille manquante |
| 3 — Erreur de casse | Majuscule/minuscule |
| 4 — Ligature | Non résolue ou mal résolue |
| 5 — Abréviation | Non développée ou mal développée |
| 6 — Hapax | Mot absent de tout dictionnaire moderne |
| 7 — Segmentation | Fusion ou fragmentation de tokens |
| 8 — Hors-vocabulaire | Caractère absent du modèle du moteur |
| 9 — Lacune | Zone non transcrite |
| 10 — Sur-normalisation LLM | Le LLM a modernisé à tort la graphie (spécifique pipelines LLM) |

#### Métriques sur entités et contenus critiques
- Précision sur les entités nommées (NER via spaCy multilingue) : personnes, lieux, dates
- Précision sur les séquences numériques (foliotation, pagination, montants)
- Taux de conservation de la ponctuation

### 6.3 Analyse qualité image et corrélations

#### Métriques de qualité image automatiques
- Score de netteté (variance du Laplacien)
- Niveau de bruit (écart-type sur région homogène)
- Détection du biais/rotation résiduel (transformée de Hough)
- Score de contraste (ratio Michelson encre/fond)
- Détection du show-through (transparence verso)
- Score de déformation géométrique (courbure de page)
- Détection des dégradations chimiques (taches, foxing)

#### Corrélations image ↔ performance
- Scatter plots interactifs : qualité image (X) vs CER (Y) par concurrent
- Corrélation de Pearson et Spearman, avec test de significativité
- Identification des concurrents robustes aux dégradations vs sensibles
- Segmentation du corpus en terciles qualité (bonne/moyenne/mauvaise)

### 6.4 Analyses statistiques et agrégées

#### Score de difficulté intrinsèque
Indicateur calculé indépendamment des moteurs, combinant :
- Variance du CER entre tous les concurrents (si tous ratent → document difficile)
- Métriques de qualité image
- Densité de caractères spéciaux (ligatures, abréviations, diacritiques)
- Longueur des lignes et densité du texte

**Valeur :** séparer deux questions distinctes — *est-ce que ce moteur est mauvais ?* vs *est-ce que ce document est objectivement difficile ?*

#### Tests statistiques
- Test de Wilcoxon (non-paramétrique) pour comparer deux concurrents
- Correction de Bonferroni pour comparaisons multiples (>2 concurrents)
- Intervalles de confiance à 95% par bootstrap sur toutes les métriques
- Test de Student apparié pour grands corpus

#### Analyse des séquences et clustering
- Longueur moyenne des séquences correctes entre deux erreurs
- Distribution des longueurs d'erreurs (erreurs isolées vs blocs)
- Clustering automatique des patterns d'erreurs (k-means) avec exemples représentatifs
- Export des clusters pour cibler le fine-tuning

#### Analyse inter-concurrents
- Score de consensus : vote majoritaire, souvent meilleur que n'importe quel moteur seul
- Carte d'accord : zones de consensus vs désaccord sur le corpus
- Complémentarité : quels concurrents ont des erreurs différentes (bons candidats pour ensemble) ?
- Analyse de dominance : pour quels types de documents le concurrent A bat-il systématiquement B ?

---

## 7. Module 5 — Rapport interactif HTML

Le rapport est un **fichier HTML unique auto-contenu**, lisible hors-ligne, embarquant toutes les données et visualisations. C'est la livrable principale de Picarones.

### 7.1 Structure du rapport

#### Page d'accueil — Tableau de bord exécutif
- Résumé de l'expérience : concurrents testés, corpus, date, paramètres de normalisation
- **Tableau de classement** des concurrents : CER, WER, score ligatures, score diacritiques — trié par colonne au clic
- **Graphique radar (spider chart)** : CER / WER / Précision diacritiques / Précision ligatures / Score mise en page — snapshot visuel des forces/faiblesses
- Histogrammes de distribution CER côte-à-côte
- Alertes : concurrents avec CER > seuil, tests statistiques non-significatifs
- Recommandation automatique : quel concurrent pour quel usage (manuscrits anciens, imprimés, grand volume...)

#### Vue Galerie — exploration du corpus
- Toutes les images en grille de vignettes avec badge CER par concurrent (code couleur vert→rouge)
- Filtres dynamiques : CER > X%, score qualité image, type de script, longueur GT, concurrent gagnant
- Tri multi-critères : CER, difficulté intrinsèque, longueur, date
- Vue **"Worst cases"** : top N documents les plus difficiles par concurrent, avec explication automatique
- Vue **"Consensus"** : documents où tous les concurrents s'accordent — les plus fiables
- Vue **"LLM gagne"** / **"LLM dégrade"** : documents où la couche LLM améliore vs détériore la sortie OCR

#### Vue Document — analyse détaillée
- Image originale zoomable (panneau gauche) avec superposition des zones en erreur si bounding boxes disponibles
- **Affichage N-way synchronisé** : GT + sortie de chaque concurrent en colonnes parallèles avec scroll synchronisé
- **Diff token coloré** façon GitHub : insertions (vert), suppressions (rouge), substitutions (orange)
- **Diff aligné sur l'image** : surlignage de la zone correspondante au survol d'une erreur (si bounding boxes)
- Bascule **"Diplomatique / Normalisé"** : diff exact vs diff avec normalisation configurée
- **Vue spécifique OCR+LLM** : trois colonnes — GT / Sortie OCR brute / Sortie après correction LLM — avec double diff pour voir exactement ce que le LLM a modifié
- Détail des métriques pour ce document

#### Vue Analyse — graphiques et statistiques
- Distribution complète des CER (histogramme + courbe de densité)
- Scatter plots interactifs : qualité image vs CER, colorés par type de script
- Courbes de fiabilité : pour les X% documents les plus faciles, quel CER ?
- **Heatmap de confusion de caractères** : cliquable — cliquer affiche tous les exemples
- Diagramme de Venn des erreurs (communes et exclusives entre concurrents)
- Visualisation des clusters d'erreurs avec exemples représentatifs
- Matrices de corrélation entre toutes les métriques
- Graphiques de significativité (p-values des tests de Wilcoxon)
- Analyse temporelle si métadonnées de date disponibles

#### Vue Caractères — analyse unicode
- Matrice de confusion unicode interactive, colorée par fréquence
- Tableau des caractères les plus souvent manqués par chaque concurrent
- CER par bloc unicode en diagramme à barres groupées
- Analyse des ligatures : taux de reconnaissance par ligature
- Caractères absents du vocabulaire d'un moteur


### 7.2 Fonctionnalités transversales

- Thème sombre / clair, interface responsive
- Recherche plein texte dans le corpus (GT et sorties de tous les concurrents)
- **URL stateful** : chaque vue filtrée accessible via URL — partager un cas précis avec un collègue
- **Mode présentation** : vue épurée pour contextes institutionnels
- **Annotations inline** : notes du paléographe exportées en JSON
- **Comparaison de rapports** : charger deux rapports (avant/après fine-tuning) et voir les deltas

### 7.3 Exports depuis le rapport

| Format | Contenu |
|---|---|
| CSV / Excel | Toutes les métriques par document et par concurrent |
| JSON | Données brutes complètes réutilisables |
| PDF | Rapport synthétique avec graphiques, pour non-techniciens |
| ALTO XML | Sorties OCR sélectionnées au format standard bibliothèques |
| PAGE XML | Format Transkribus / eScriptorium |
| Images annotées | Images originales avec zones d'erreur surlignées (PNG) |
| Corpus d'erreurs | Pires cas pour cibler le fine-tuning (image + GT + sortie OCR) |
| Prompts LLM | Export de tous les prompts utilisés avec leurs performances |

---

## 8. Module 6 — Interface de lancement et CLI

### 8.1 Interface web légère (FastAPI)

- Configuration du benchmark : sélection corpus, concurrents, paramètres de normalisation
- Visualisation de l'avancement en temps réel avec log streamé
- Gestion des configurations enregistrées (profils)
- Accès aux rapports générés précédemment
- Configuration des adaptateurs moteurs et LLM via formulaire

### 8.2 Interface en ligne de commande (CLI)

```bash
# Lancer un benchmark
picarones run --corpus ./mes_gt/ --engines tesseract,pero_ocr,mistral --output ./rapport/

# Avec pipeline OCR+LLM
picarones run --corpus ./gt/ --config pipelines/medieval_correction.yaml

# Import IIIF puis benchmark
picarones import iiif https://gallica.bnf.fr/ark:/12148/xxx/manifest.json --pages 1-50
picarones run --corpus iiif:xxx --engines tesseract,pero_ocr --llm gpt-4o

# Rapport seul depuis résultats existants
picarones report --results ./results.json --output ./rapport.html

# Mode CI/CD : exit code non-zero si CER > seuil
picarones run --corpus ./gt/ --engines mon_moteur --fail-if-cer-above 5.0

picarones estimate --corpus ./gt/ --config pipelines/gpt4o_correction.yaml
```

### 8.3 Intégration CI/CD

- Mode headless complet, exit code paramétrable selon les métriques
- Output JSON machine-readable pour intégration dans systèmes de monitoring
- Badge de qualité générable (SVG) affichant le CER du modèle courant
- Détection automatique des régressions (CER augmente par rapport au run précédent)

---

## 9. Fonctionnalités avancées

### 9.1 Profils de normalisation pré-configurés

| Profil | Règles principales |
|---|---|
| Français médiéval (XIIe-XVe) | u/v, i/j, ſ/s, abréviations courantes |
| Français moderne (XVIe-XVIIIe) | ſ/s, ligatures fi/fl, esperluettes |
| Latin médiéval | Abréviations, contractions, ligatures spécifiques |
| Imprimés anciens (XVe-XVIIe) | Conventions typographiques, réclames |
| Personnalisé | Configurable et exportable |

### 9.2 Analyse par type de script

Si les documents sont tagués, calcul automatique des métriques par catégorie :
- Gothique textura / rotunda / cursiva
- Minuscule caroline
- Humanistique / italique
- Imprimé romain / italique ancien
- Cursives administratives (XVIIe-XIXe)

### 9.3 Suivi longitudinal

- Base de données des benchmarks historiques (SQLite optionnel)
- Courbes d'évolution : CER dans le temps pour un modèle en développement
- Détection automatique des régressions entre deux versions
- Comparaison avant/après fine-tuning

### 9.4 Analyse de robustesse

- Génération automatique de versions dégradées des images (bruit, flou, rotation, réduction de résolution, binarisation)
- Courbes de robustesse : CER en fonction du niveau de dégradation
- Identification du seuil de dégradation critique pour chaque concurrent

### 9.5 Évaluation par sous-région

Si bounding boxes disponibles dans la GT (ALTO/PAGE XML) :
- CER par zone : corps du texte / titres courants / marginalia / initiales / notes de bas de page
- Heatmap de densité d'erreur sur l'image

### 9.6 Détection de la sur-normalisation LLM

Risque spécifique aux pipelines OCR+LLM : le LLM "corrige" à tort des graphies médiévales légitimes en les modernisant. Picarones mesure :
- Taux de modification introduites par le LLM sur des passages déjà corrects
- Score de sur-normalisation : combien de transcriptions correctes le LLM a-t-il dégradées ?
- Liste des interventions LLM non souhaitées pour affiner le prompt

---

## 10. Plan de développement

| Sprint | Durée | Livrables |
|---|---|---|
| **Sprint 1** | 1-2 sem. | Structure du projet, adaptateurs Tesseract + Pero OCR, calcul CER/WER avec `jiwer`, export JSON, CLI de base |
| **Sprint 2** | 1-2 sem. | Rapport HTML v1 : galerie, vue document avec diff coloré, tableau de classement, graphiques de base |
| **Sprint 3** | 1-2 sem. | Pipelines OCR+LLM (modes text_only et text_and_image), adaptateurs GPT-4o et Claude |
| **Sprint 4** | 1-2 sem. | Adaptateurs API OCR (Mistral OCR, Google Vision), import IIIF, normalisation unicode, CER diplomatique |
| **Sprint 5** | 1-2 sem. | Métriques avancées : matrice confusion unicode, ligatures, structure, qualité image, taxonomie des erreurs |
| **Sprint 6** | 1-2 sem. | Interface web FastAPI, import HTR-United / HuggingFace, profils de normalisation, Ollama (LLMs locaux) |
| **Sprint 7** | 1-2 sem. | Rapport HTML v2 : vue Caractères, scatter plots, heatmaps, clustering |
| **Sprint 8** | 2 sem. | Intégration eScriptorium et Gallica API, suivi longitudinal, analyse de robustesse, prompts bibliothèque |
| **Sprint 9+** | Continu | Tests utilisateurs, documentation, packaging Docker, CI/CD, publication open-source |

---

## 11. Exigences non fonctionnelles

### Performance
- Pipeline capable de traiter 1000 documents en moins de 30 minutes (moteurs locaux)
- Rapport HTML interactif fluide pour des corpus de 10 000 documents
- Calcul des métriques en moins de 1 seconde par document

### Interopérabilité
- Compatibilité Linux, macOS, Windows
- Docker fourni pour un déploiement reproductible
- API REST documentée (OpenAPI) pour intégration tierce
- Conformité IIIF, ALTO XML, PAGE XML, TEI

### Qualité et maintenabilité
- Tests unitaires pour toutes les métriques (vérification sur cas connus)
- Tests d'intégration sur corpus de référence
- Documentation de chaque métrique (définition, formule, interprétation)
- Code open-source (licence Apache 2.0)

### Sécurité et confidentialité
- Aucune donnée envoyée vers l'extérieur sans consentement explicite
- Mode entièrement hors-ligne possible (moteurs locaux + Ollama uniquement)
- Clés API dans variables d'environnement uniquement

---

*Picarones est conçu pour devenir la référence open-source de l'évaluation OCR/HTR dans le champ patrimonial — métriques adaptées aux documents historiques, pipelines OCR+LLM, intégration native des standards bibliothéconomiques, rapport interactif exportable.*
