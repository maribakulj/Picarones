# Commandes CLI Picarones

Picarones expose **15 commandes/groupes** Click dans le package
[`picarones/interfaces/cli/`](../picarones/interfaces/cli/). Le découpage en sous-modules
(chantier 5) est transparent : toutes les commandes restent
accessibles via `picarones <cmd>` après `pip install -e .`.

## Synoptique

| Commande | Module | Profil | Cible |
|---|---|---|---|
| `run` | `_workflows.py` | `standard` | Bench classique mono- ou multi-moteur |
| `diagnose` | `_workflows.py` | `diagnostics` | Bench + leviers + image_predictive |
| `economics` | `_workflows.py` | `economics` | Bench + throughput effectif |
| `edition` | `_workflows.py` | `philological` | Bench + taxonomie miroir |
| `compare` | `_workflows.py` | — | Comparer 2 runs JSON existants |
| `serve` | `_serve.py` | — | Lance l'interface web FastAPI |
| `import iiif` | `_imports.py` | — | Importe un manifeste IIIF en corpus |
| `history` | `_history.py` | — | Consulte l'historique SQLite |
| `robustness` | `_robustness.py` | — | Analyse de robustesse aux dégradations |
| `metrics` | `__init__.py` | — | CER/WER entre 2 fichiers texte |
| `engines` | `__init__.py` | — | Liste les moteurs disponibles |
| `info` | `__init__.py` | — | Version + dépendances |
| `report` | `__init__.py` | — | Régénère un rapport HTML depuis JSON |
| `demo` | `__init__.py` | — | Génère un rapport démo (données fictives) |

## Workflows benchmark — chantier 4

Les 4 commandes `run` / `diagnose` / `economics` / `edition` partagent
le même squelette factorisé dans `_run_workflow()`. La seule différence :
le **profil de calcul** (chantier 2) qui détermine quelles métriques
sont calculées et quelle vue HTML est rendue.

### `picarones run` — bench standard

```bash
picarones run \
    --corpus ./corpus_test \
    --engines tesseract,pero_ocr \
    --output results.json \
    --lang fra \
    --profile standard
```

Profil par défaut : `standard` (les 12 hooks historiques).
Génère `results.json` puis `report.html` automatiquement.

### `picarones diagnose` — diagnostic approfondi

```bash
picarones diagnose --corpus ./corpus --engines tess,pero
```

Profil : `diagnostics`. Active la vue HTML « Diagnostic approfondi »
avec leviers, profil d'image, baseline historique (si SQLite chargé).

### `picarones economics` — décision budget

```bash
picarones economics --corpus ./corpus --engines mistral_ocr,tesseract
```

Profil : `economics`. Vue HTML « Coût et performance » : throughput
effectif (5 s/erreur HTR-United), pages/h utilisable.

### `picarones edition` — édition critique

```bash
picarones edition --corpus ./manuscrits --engines tesseract,pero_ocr
```

Profil : `philological`. Vue HTML « Taxonomie avancée » : diagramme
miroir leader vs runner-up, classes par récupérabilité.

### `picarones compare` — diff entre 2 runs

```bash
picarones compare run_a.json run_b.json --output diff.html
```

Compare deux fichiers JSON de bench (par exemple : avant/après mise à
jour modèle) et génère un rapport HTML de diff.

## Pipeline composée — axe B + chantier 1

### `picarones pipeline run` / `pipeline compare` — retirés en 7.D

Les commandes ``picarones pipeline run`` et ``pipeline compare`` ont
été retirées en Phase 7.D du retrait du legacy (mai 2026), avec le
``PipelineRunner`` legacy qu'elles enveloppaient.  Une CLI au-dessus
du ``PipelineExecutor`` canonique pourra être réintroduite
post-2.0.  En attendant, l'API Python est documentée dans
[`docs/reference/api-stable.md`](../reference/api-stable.md).

## Imports — chantier 4

### `picarones import iiif`

```bash
picarones import iiif \
    --manifest https://gallica.bnf.fr/ark:/12148/btv1b8453561w/manifest.json \
    --output ./corpus_gallica \
    --pages 1-10
```

Télécharge un manifeste IIIF v2/v3 (BnF Gallica, Bodleian, Vatican…) et
crée un corpus local avec `.gt.txt` extraits de l'OCR ALTO si présent.
Depuis le chantier 4, IIIF et Gallica utilisent les mêmes helpers HTTP
factorisés ([`picarones/adapters/corpus/_http.py`](../picarones/adapters/corpus/_http.py))
avec garde-fou `file://`/`ftp://`/`javascript://`.

## Outils utilitaires

### `picarones serve`

```bash
picarones serve --host 0.0.0.0 --port 7860
```

Lance l'interface web FastAPI (HuggingFace Space port 7860 par défaut
en prod). Permet l'upload de corpus ZIP, le bench live avec barre de
progression SSE, et l'export du rapport HTML.

### `picarones history`

```bash
picarones history --engine tesseract --corpus mon_corpus_xviii
```

Consulte l'historique SQLite des runs (Sprint 8). Affiche l'évolution
longitudinale du CER pour un moteur sur un corpus donné.

### `picarones robustness`

```bash
picarones robustness \
    --corpus ./corpus_test \
    --engine tesseract \
    --degradations noise,blur,rotation \
    --intensities 0.1,0.3,0.5
```

Re-OCR un corpus avec des dégradations synthétiques d'image et trace
la courbe CER vs intensité. Permet d'évaluer la robustesse d'un moteur
hors conditions optimales.

### `picarones metrics`

```bash
picarones metrics --reference gt.txt --hypothesis ocr.txt
```

Calcul rapide CER/WER entre deux fichiers texte, sans corpus ni rapport.

### `picarones engines`

Liste les engines OCR disponibles dans l'environnement courant
(détectés via leur import optionnel).

### `picarones info`

Affiche la version Picarones, Python, OS, et les dépendances optionnelles
détectées (`tesseract`, `pero-ocr`, `mistralai`, `openai`,
`google-cloud-vision`, `azure-ai-formrecognizer`, `scipy`, `spacy`).

### `picarones report`

```bash
picarones report --results results.json --output report.html
```

Régénère un rapport HTML depuis un JSON existant. Utile pour
re-rendre après une mise à jour de l'i18n ou des templates.

### `picarones demo`

```bash
picarones demo --output demo.html
```

Génère un rapport HTML à partir de données fictives (médiévales). Utile
pour découvrir la sortie sans corpus réel.

## Code source

- [`picarones/interfaces/cli/__init__.py`](../picarones/interfaces/cli/__init__.py) — groupe
  Click + helpers + commandes simples.
- [`picarones/interfaces/cli/_workflows.py`](../picarones/interfaces/cli/_workflows.py) —
  run, diagnose, economics, edition, compare + helper `_run_workflow`.
- Voir aussi [`docs/reference/normalization-profiles.md`](profiles.md) et
  [`docs/reference/views.md`](views.md).
