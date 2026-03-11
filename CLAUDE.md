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

## Bugs actifs à corriger en priorité

### 🔴 BUG CRITIQUE — Pipeline OCR+LLM sortie vide
**Symptôme** : le pipeline `tesseract → mistral:ministral-3b-latest` s'exécute (15s de traitement
visible dans les logs) mais produit une sortie vide `""` pour chaque document. Le rapport affiche
CER 100% avec "Aucune sortie" et 0/0 documents.

**Localisation probable** :
- `picarones/llm/mistral_adapter.py` : vérifier que `choices[0].message.content` est bien extrait
- `picarones/pipelines/base.py` : vérifier que `result.hypothesis` est bien mis à jour après
  l'appel LLM, et que les DocumentResult sont bien collectés par le runner
- Le modèle `ministral-3b-latest` supporte bien `POST /v1/chat/completions`

**À faire** : ajouter des logs DEBUG (prompt envoyé tronqué, statut HTTP, contenu brut réponse)
pour diagnostiquer sans modifier le comportement.

### 🟡 CI — python-multipart
**Symptôme** : 114 tests ERROR car `python-multipart` absent lors de l'import de `web/app.py`.
**Fix** : dans `.github/workflows/ci.yml`, remplacer `pip install -e ".[dev]"` par
`pip install -e ".[dev,web]"`.

### 🟡 Tests fixtures post-Sprint 10
5 tests échouent : counts de moteurs (4→5) et flag `is_pipeline` pour `gpt-4o-vision`.

### 🟡 Test Windows SQLite
`TestCLIHistory::test_history_empty_db` — PermissionError sur Windows (fichier encore ouvert
lors du `os.unlink`). À corriger avec `try/except` autour du `unlink`.

### 🟡 Test HuggingFace language filter
`TestHuggingFaceImporter::test_search_language_filter` — assertion sur `ds.language`.

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

---

## Contexte développement

- **Environnement** : GitHub Codespaces (`/workspaces/Picarones`), Python 3.12
- **Tests** : ~1020 tests (après sprint 13)
- **Branche active** : `main` (ou `claude/setup-picarones-project-FKKns` selon le contexte)
- **Transcript de la conversation de développement** :
  `/mnt/transcripts/2026-03-11-14-01-41-picarones-ocr-bench-project.txt`
