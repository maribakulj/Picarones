# Snapshots de reproductibilité

> Sprint A8 (item M-12 du plan de remédiation).

## Pourquoi des snapshots ?

Pour qu'un benchmark Picarones soit **citable scientifiquement**, un
lecteur doit pouvoir, des années plus tard, comprendre exactement
*ce qui a été mesuré*. Un rapport HTML qui dit « Tesseract 5.3.4
obtient un CER de 4,2 % sur ce corpus » est inutilisable s'il
n'indique pas :

- la **table de pricing** utilisée (qui a évolué entre temps),
- la **version exacte des prompts** appliqués aux pipelines OCR+LLM,
- le **profil de normalisation** effectivement appliqué (avec ses
  équivalences diplomatiques),
- le **commit Picarones** utilisé pour produire le rapport,
- les **paquets Python** installés au moment du run.

Le module `picarones.report.snapshot` agrège ces cinq dimensions et
les embarque **dans le JSON du rapport**, sous la clé
`report_data["snapshots"]`. Le rapport HTML reste auto-portant : un
lecteur peut tout retrouver sans accès au repo source.

## Ce qu'un snapshot contient

```python
from picarones.report.snapshot import snapshot_all

snap = snapshot_all(
    lang="fr",
    normalization_profile=profile,  # Profile dataclass ou None
)
```

Retourne un dict avec quatre clés top-level :

| Clé | Contenu | Source |
|---|---|---|
| `pricing` | YAML brut intégral de `picarones/data/pricing.yaml` | `pricing_snapshot()` |
| `glossary` | Entrées du glossaire effectivement référencées dans la synthèse (langue rendue) | `glossary_snapshot()` |
| `normalization` | Profil sérialisé (`diplomatic_table`, `exclude_chars`, drapeaux NFC/case-folding…) | `normalization_snapshot()` |
| `environment` | Version Picarones, Python, plateforme OS, commit git, paquets installés (top 200) | `environment_snapshot()` |

### `pricing`

Le YAML est embarqué **verbatim**. Si le tarif d'OpenAI change demain,
le rapport d'aujourd'hui reste lisible — l'analyse Pareto coût garde
sa valeur historique car elle pointe vers la table effectivement
utilisée.

### `glossary`

Pas le glossaire complet (~25 entrées) — seulement celles qui sont
effectivement référencées par la synthèse narrative ou les vues
ouvertes au moment du snapshot. Économie de poids ; un ancien rapport
reste documenté même si le glossaire évolue.

### `normalization`

Le profil contient :

```yaml
name: medieval_french
nfc: true
casefold: false
diplomatic_table:
  ſ: s
  u: v
  i: j
  ꝑ: per
  ⁊: et
exclude_chars:
  - "·"
  - "¶"
```

Permet à un relecteur de comprendre exactement quelles équivalences
ont été appliquées au moment du calcul de CER, **sans relancer**.

### `environment`

```yaml
picarones_version: "1.1.0"
python_version: "3.11.13"
platform: "Linux 6.18.5-x86_64-with-glibc2.39"
git_commit: "17cc5474..."
installed_packages:
  - "click==8.3.3"
  - "defusedxml==0.7.1"
  - "jiwer==3.1.0"
  # ... top 200 trié alpha
```

Le `git_commit` est lu via `git rev-parse HEAD` (subprocess timeout 2 s,
sans shell). Si le repo n'est pas un dépôt git, la clé reste `None`.

## Comment rejouer un benchmark à 5 ans d'écart

### Étape 1 — Récupérer le commit Picarones d'origine

Dans le rapport HTML, ouvrir la console JS et inspecter `DATA.snapshots.environment.git_commit` :

```javascript
> DATA.snapshots.environment.git_commit
"17cc5474abc..."
```

Puis :

```bash
git clone https://github.com/maribakulj/Picarones.git
cd Picarones
git checkout 17cc5474abc
```

### Étape 2 — Récréer l'environnement Python

Sprint A8 livre les lock files :

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.lock
```

Si vous avez besoin d'une version Python différente (par exemple un
ancien rapport rendu en Python 3.11.10), utiliser pyenv :

```bash
pyenv install 3.11.10
pyenv local 3.11.10
```

### Étape 3 — Récréer le corpus + GT

Picarones ne stocke **pas** les images du corpus dans le snapshot
(les images appartiennent au déposant). Il faut donc re-récupérer le
corpus original. Pour les imports IIIF, le manifeste est durable
(les bibliothèques nationales versionnent) ; pour les uploads ZIP,
l'utilisateur doit conserver son archive source.

Si le corpus a déjà été chargé dans Gallica / HTR-United, le
`metadata.source_url` de chaque `Document` permet le re-fetch.

### Étape 4 — Rejouer le benchmark

```bash
picarones run \
    --corpus ./corpus_recovered/ \
    --engines tesseract,pero_ocr \
    --output rerun.json \
    --normalization medieval_french
```

### Étape 5 — Vérifier la concordance

Le commit + le lock file + le profil de normalisation garantissent
que les métriques **CER / WER / MER / WIL** seront bit-à-bit
identiques.

Différences possibles légitimes :
- L'OCR cloud (Mistral / Google / Azure) peut avoir évolué côté
  serveur — les chiffres peuvent diverger même avec un client
  identique. Pour un benchmark scientifiquement reproductible,
  privilégier **Tesseract / Pero OCR** (modèles versionnés et locaux).
- Les LLMs évoluent constamment ; un pipeline OCR+LLM rejoué 6 mois
  plus tard peut donner d'autres résultats. Le snapshot des prompts
  reste utile mais ne reproduit pas le LLM lui-même.

## Snapshot et publication scientifique (préparation Sprint A12)

Pour un papier scientifique, citer Picarones doit indiquer :

```bibtex
@misc{picarones_2026,
  title  = {Picarones: Heritage OCR/HTR/VLM Benchmarking Platform},
  author = {<auteurs>},
  year   = {2026},
  doi    = {10.5281/zenodo.<id>},
  version= {1.1.0},
  url    = {https://github.com/maribakulj/Picarones}
}
```

Et dans le matériel supplémentaire, joindre :

1. Le rapport HTML autonome (qui contient tout le snapshot).
2. `requirements-dev.lock` du moment du benchmark (pour ré-instancier
   l'environnement).
3. Le digest Docker si vous avez utilisé l'image officielle :
   `docker inspect ghcr.io/maribakulj/picarones:1.1.0 --format='{{.Id}}'`.

Un évaluateur scientifique disposera ainsi des éléments matériels pour
vérifier l'analyse — c'est l'exigence minimale d'une publication
reproductible (cf. Stodden et al., *Computational reproducibility*).

## Limites assumées

- **Le code source n'est pas embarqué dans le snapshot**. On embarque
  *l'identifiant* du commit, pas le diff. Si le repo est rendu
  privé ou supprimé, le snapshot devient orphelin. Mitigation :
  publier les versions sur Zenodo (DOI durable, garanti 20 ans).
- **Les images du corpus** ne sont pas snapshottées (poids + droits
  d'auteur). Le déposant doit conserver son corpus.
- **Le LLM cloud** ne peut pas être snapshotté — c'est une dépendance
  externe non reproductible. Pour la science, préférer les modèles
  ouverts (Llama, Phi, Mistral via Ollama).

## Tests

`tests/report/test_sprint27_reproducibility_snapshots.py` (Sprint 27)
valide que `snapshot_all()` est :

- déterministe (même input → même bytes en sortie),
- complet (toutes les clés top-level présentes),
- robuste (ne crashe pas si git absent, si pricing.yaml manquant…).

`tests/test_reproducibility_ops.py` (Sprint A8) ajoute la validation
de la chaîne **lock file + Docker digest + snapshot** comme contrat
opérationnel.
