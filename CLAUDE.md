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
│   └── robustness.py       # Analyse robustesse (bruit, flou, rotation, résolution)
├── engines/
│   ├── base.py             # BaseEngine avec execution_mode ("io" ou "cpu")
│   ├── tesseract.py        # execution_mode = "cpu"
│   ├── pero_ocr.py         # execution_mode = "cpu"
│   ├── mistral_ocr.py      # endpoint /v1/ocr dédié (pas chat/completions)
│   ├── google_vision.py
│   └── azure_doc_intel.py
├── llm/
│   ├── base.py
│   ├── mistral_adapter.py  # POST /v1/chat/completions — BUG ACTIF : sortie vide à corriger
│   ├── openai_adapter.py
│   ├── anthropic_adapter.py
│   └── ollama_adapter.py
├── pipelines/
│   ├── base.py             # OCRLLMPipeline — BUG ACTIF : résultats 0/0 documents
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
│   ├── generator.py        # Rapport HTML auto-contenu (Chart.js + diff2html)
│   └── diff_utils.py
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

---

## Moteur narratif (Sprint 16)

Fondations en place dans `picarones/core/narrative/` :

```
core/narrative/
├── __init__.py              # API publique : Fact, FactType, FactImportance, DetectorRegistry, detect_all
├── facts.py                 # Modèle de données : Fact dataclass, 12 FactType, 4 FactImportance, DetectorRegistry
└── detectors.py             # Stubs des 12 détecteurs (implémentations sprint par sprint)
```

**Principe anti-hallucination** : chaque valeur numérique ou nom d'entité dans le
`payload` d'un `Fact` doit provenir directement du JSON d'entrée du benchmark.
Test unitaire à ajouter au Sprint 4 : parser la synthèse rendue et vérifier que
tous les nombres qu'elle contient sont traçables au JSON source.

**Détecteurs** : les 12 stubs sont en place. L'activation dans le registre par
défaut se fait sprint par sprint au fur et à mesure de leur implémentation :
- Sprint 3 : `statistical_tie` (après Friedman-Nemenyi)
- Sprint 4 : `global_leader_cer`, `significant_gap`, `stratum_winner`, `stratum_collapse`,
  `error_profile_outlier`, `llm_hallucination_flag`, `robustness_fragile`, `speed_winner`,
  `confidence_warning`
- Sprint 5 : `pareto_alternative`, `cost_outlier`

---

## Contexte développement

- **Environnement** : GitHub Codespaces (`/workspaces/Picarones`), Python 3.12
- **Tests** : 1072 passed, 2 skipped (Sprint 16)
- **Branche active** : `claude/review-picarones-benchmarks-E3J42`
- **Transcript de la conversation de développement** :
  `/mnt/transcripts/2026-03-11-14-01-41-picarones-ocr-bench-project.txt`
