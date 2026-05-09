# Cohérence documentation — contrat d'écriture

> Sprint A2 du plan de remédiation institutionnelle
> ([`docs/audits/remediation-plan-2026-05.md`](../audits/remediation-plan-2026-05.md)).

Picarones expose plusieurs documents de premier contact (README.md,
SPECS.md, CHANGELOG.md, CITATION.cff, …). Pour qu'un primo-lecteur
ne soit jamais induit en erreur, **la documentation publiée doit
refléter le code réel**. Le suite de tests `tests/docs/` matérialise
ce contrat.

## TL;DR

```bash
make doc-check       # rapport en < 5 s
# ou directement :
pytest tests/docs/ -v
```

Si vous ajoutez un moteur OCR, une commande CLI, un endpoint web ou
modifiez le compteur de tests, lisez la suite ci-dessous.

---

## Tests posés en A2

### `test_readme_consistency.py`

Vérifie que :

| Item | Source de vérité | Sens du contrat |
|---|---|---|
| Moteurs OCR | `picarones/adapters/legacy_engines/*.py` | Tout moteur listé dans le tableau « Supported Engines » du README doit avoir un adapter |
| Commandes CLI | `picarones/interfaces/cli/_legacy/*.py` (Click) | Toute commande listée dans le README doit apparaître dans `picarones --help` |
| Endpoints API | `picarones/interfaces/web/_legacy/app.py` (`app.openapi()`) | Tout endpoint listé doit exister dans la spec OpenAPI |
| Compteur de tests | `pytest --collect-only` | Toute mention « N tests » ou « N passed » doit être à 5 % près du baseline |
| Variables `AWS_*` | `picarones/adapters/legacy_engines/aws*.py` | Si documentées, un adapter doit exister |

**Direction unidirectionnelle** : on vérifie que ce qui est *annoncé*
existe — pas que tout ce qui existe est annoncé. La direction réciproque
est posée en Sprint A13 (refonte intégrale du README).

### `test_specs_consistency.py`

Vérifie que :

- SPECS.md existe et déclare une version + une date.
- Toute promesse explicitement *abandonnée* depuis SPECS v1 (AWS Textract,
  Calamari, OCRopus, Recommandation automatique, Export PDF, k-means
  clustering, Annotations inline, Badge SVG) doit être marquée par un
  des trois mécanismes acceptés (cf. § « Mécanismes de tolérance » plus bas).

### `test_changelog_links.py`

Vérifie que :

- Le CHANGELOG existe, suit Keep-a-Changelog, contient des sections
  versionnées.
- Toute référence `Sprint NN` résout dans CHANGELOG ou CLAUDE.md.
- Tout lien interne (`docs/...`, `picarones/...`) pointe vers un
  fichier existant.

### `test_sprint_numbering.py`

Audit **informatif** (warnings, non bloquant) : trous de numérotation
des fichiers `test_sprintNN_*.py`, doublons, docstrings manquants.
Pour rendre bloquant ponctuellement : `pytest -W error::UserWarning
tests/docs/test_sprint_numbering.py`.

---

## Mécanismes de tolérance

Trois mécanismes permettent une exception ponctuelle :

### 1. Marqueur ligne par ligne (`test_readme_consistency.py`)

Pour autoriser une ligne de tableau temporairement non vérifiable :

```markdown
| Engine | Type | … |
|--------|------|---|
| **NewEngine** | Local Python | (en cours) | <!-- doc-check: skip-engine -->
```

Marqueurs reconnus : `skip-engine`, `skip-cli`, `skip-endpoint`,
`skip-env`. À utiliser avec modération — **tout `skip-*` doit être
expliqué en revue de PR**.

### 2. Bloc d'abandon global (`test_specs_consistency.py`)

Pour SPECS, les promesses explicitement abandonnées sont listées dans
un bloc unique en tête de l'addendum :

```markdown
<!-- specs-check: known-abandoned-start -->

- **AWS Textract** : adapter non implémenté ; reporté.
- **Calamari** : adapter non implémenté ; reporté.
- …

<!-- specs-check: known-abandoned-end -->
```

Le test accepte qu'une promesse listée dans ce bloc apparaisse aussi
ailleurs dans SPECS sans note de deprecation locale.

### 3. Note de deprecation locale (`test_specs_consistency.py`)

Alternative à la #2 quand on veut documenter une décision *à proximité*
de la mention :

```markdown
La recommandation automatique (§7.1) est **abandonnée** au profit du
moteur narratif factuel (Sprint 19) ; cf. la note de neutralité
éditoriale dans CLAUDE.md.
```

Le test scanne une fenêtre de 200 caractères autour de chaque mention
et accepte si l'un des mots `reporté`, `abandonné`, `non implémenté`,
`deferred`, `not implemented`, etc. est présent.

---

## Workflow de modification

| Vous modifiez… | Vous devez aussi… |
|---|---|
| `picarones/engines/<X>.py` (nouveau) | Ajouter une ligne dans le tableau « Supported Engines » du README |
| `picarones/cli/_<X>.py` (nouveau) | Ajouter la commande dans la table CLI du README |
| Un nouveau endpoint FastAPI | Ajouter dans la table « API endpoints » du README |
| Le nombre de tests | Mettre à jour les 3 mentions dans README (l/583, l/623, l/660) |
| Une promesse SPECS qui devient sans objet | Soit retirer la mention, soit ajouter dans le bloc `known-abandoned`, soit ajouter une note locale |

Le test `make doc-check` (ou directement `pytest tests/docs/`) tournera
automatiquement en CI à chaque PR. Un échec bloque le merge.

---

## À venir (Sprint A13)

La refonte du README en Sprint A13 ajoutera de la **génération
automatique** : les tableaux d'engines / CLI / endpoints / structure
projet seront générés depuis le code via `scripts/gen_readme_tables.py`,
insérés dans le README via des balises HTML
`<!-- generated:engines -->`. À ce moment-là, la direction réciproque
deviendra naturellement vérifiée (« tout ce qui existe est annoncé »).

D'ici là, ce contrat **uni-directionnel** est suffisant pour bloquer
les divergences les plus visibles.
