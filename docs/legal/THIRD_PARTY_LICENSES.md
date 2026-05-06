# Third-party licenses

> **Audience** : équipe juridique, DSI institutionnelle, mainteneur
> de release.  Audit des licences des dépendances tierces utilisées
> par Picarones, requis par Apache 2.0 §4(d) et par les politiques
> d'achat institutionnelles (BnF, LoC, BL).
>
> **Régénération** : ce fichier est censé être régénéré à chaque
> release par `scripts/gen_third_party_licenses.py` (à venir, cf.
> [`docs/roadmap/backlog.md`](../roadmap/backlog.md)).  Tant que le
> script n'existe pas, mise à jour manuelle au moment de la release.
>
> **Date du dernier rafraîchissement** : 2026-05.

## Politique générale

Picarones est distribué sous **Apache License 2.0**.  Cette licence
est compatible avec toutes les licences listées ci-dessous (MIT, BSD,
PSF, Apache 2.0 elles-mêmes ; pas de dépendance GPL/LGPL/AGPL en
runtime).

Les dépendances optionnelles (extras `[mistral]`, `[anthropic]`,
`[openai]`, `[ollama]`, `[google]`, `[azure]`, `[hf]`, `[escriptorium]`,
`[iiif]`, `[stats]`, `[ner]`) ne sont chargées qu'à la demande de
l'utilisateur ; elles n'affectent pas la licence du distribué de base.

## Dépendances de runtime (cœur)

| Paquet | Licence | Copyright | Usage |
|--------|---------|-----------|-------|
| [click](https://palletsprojects.com/p/click/) | BSD-3-Clause | © Pallets | CLI |
| [jiwer](https://github.com/jitsi/jiwer) | Apache-2.0 | © 8x8, Inc. | CER / WER |
| [Pillow](https://python-pillow.org/) | HPND (MIT-style) | © Jeffrey A. Clark + Pillow contributors | Images |
| [PyYAML](https://pyyaml.org/) | MIT | © Kirill Simonov | YAML |
| [pytesseract](https://github.com/madmaze/pytesseract) | Apache-2.0 | © Matthias A. Lee | OCR Tesseract wrapper |
| [tqdm](https://tqdm.github.io/) | MIT + MPL-2.0 | © tqdm contributors | Barres de progression |
| [numpy](https://numpy.org/) | BSD-3-Clause | © NumPy developers | Calculs numériques |
| [jinja2](https://palletsprojects.com/p/jinja/) | BSD-3-Clause | © Pallets | Templating HTML |
| [defusedxml](https://github.com/tiran/defusedxml) | PSF-2.0 | © Christian Heimes | Parsing XML sécurisé |
| [pydantic](https://docs.pydantic.dev/) | MIT | © Samuel Colvin and contributors | Modèles immuables |

## Dépendances de runtime — extras

### `[web]`

| Paquet | Licence | Usage |
|--------|---------|-------|
| [fastapi](https://fastapi.tiangolo.com/) | MIT | API web |
| [uvicorn](https://www.uvicorn.org/) | BSD-3-Clause | Serveur ASGI |
| [python-multipart](https://github.com/Kludex/python-multipart) | Apache-2.0 | Upload form-data |
| [starlette](https://www.starlette.io/) | BSD-3-Clause | (transitif via FastAPI) |
| [httpx](https://www.python-httpx.org/) | BSD-3-Clause | Client HTTP (tests) |

### `[mistral]`

| Paquet | Licence | Usage |
|--------|---------|-------|
| [mistralai](https://github.com/mistralai/client-python) | Apache-2.0 | SDK Mistral OCR + chat/vision |

### `[anthropic]`

| Paquet | Licence | Usage |
|--------|---------|-------|
| [anthropic](https://github.com/anthropics/anthropic-sdk-python) | MIT | SDK Claude |

### `[openai]`

| Paquet | Licence | Usage |
|--------|---------|-------|
| [openai](https://github.com/openai/openai-python) | Apache-2.0 | SDK OpenAI |

### `[ollama]`

| Paquet | Licence | Usage |
|--------|---------|-------|
| [ollama](https://github.com/ollama/ollama-python) | MIT | Client Ollama local |

### `[google]`

| Paquet | Licence | Usage |
|--------|---------|-------|
| [google-cloud-vision](https://github.com/googleapis/python-vision) | Apache-2.0 | OCR Google Vision |

### `[azure]`

| Paquet | Licence | Usage |
|--------|---------|-------|
| [azure-ai-documentintelligence](https://github.com/Azure/azure-sdk-for-python) | MIT | OCR Azure DI |

### `[hf]`

| Paquet | Licence | Usage |
|--------|---------|-------|
| [datasets](https://github.com/huggingface/datasets) | Apache-2.0 | Datasets HuggingFace |
| [huggingface-hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | Hub HuggingFace |

### `[ner]`

| Paquet | Licence | Usage |
|--------|---------|-------|
| [spacy](https://spacy.io/) | MIT | NER |

### `[stats]`

| Paquet | Licence | Usage |
|--------|---------|-------|
| [scipy](https://scipy.org/) | BSD-3-Clause | Tests statistiques (Friedman, Nemenyi) |

## Dépendances de développement

Les paquets utilisés uniquement en développement (tests, lint,
sécurité) ne sont pas redistribués avec Picarones et n'apparaissent
dans aucun wheel.  Pour traçabilité supply-chain :

| Paquet | Licence | Usage |
|--------|---------|-------|
| pytest | MIT | Tests unitaires |
| pytest-cov | MIT | Couverture |
| pytest-timeout | MIT | Timeout par test |
| ruff | MIT | Lint |
| mypy | MIT | Type checking |
| bandit | Apache-2.0 | Audit sécurité statique |
| pip-audit | Apache-2.0 | Audit CVE des dépendances |

## Modèles tiers

Picarones n'embarque **aucun modèle tiers** dans ses wheels.  Les
modèles sont :

- soit **téléchargés à l'usage** par l'utilisateur (Tesseract `*.traineddata`,
  Pero OCR via Zenodo, modèles spaCy via `python -m spacy download`) ;
- soit **invoqués via des APIs cloud** sous le contrat du fournisseur
  (Mistral AI, Anthropic, OpenAI, Google, Azure).

Les conditions d'utilisation de chaque modèle / API sont à la charge
de l'utilisateur et de l'institution déployant Picarones.

## Police d'écriture / fontes

Picarones n'embarque aucune fonte.  Les rapports HTML utilisent les
fontes système du navigateur.

## Données

Aucun corpus, aucune image, aucune vérité terrain n'est embarquée
dans les wheels.  Les fixtures de test (`tests/fixtures/`) sont
synthétiques (générées) ou citées depuis leur source originale (cf.
`tests/fixtures/reference_corpus/README.md`).

## Comment signaler une omission

Une dépendance manquante, une licence incorrecte, un copyright
mal attribué : ouvrir une issue avec le label `legal` ou écrire à
l'adresse de contact dans [`/SECURITY.md`](../../SECURITY.md).  Une
correction sera publiée dans la prochaine release patch.
