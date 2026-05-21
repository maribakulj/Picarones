# TextView — première vue canonique

`TextView` est la première vue d'évaluation canonique.  Elle répond
à la question patrimoniale la plus fréquente : **"quel pipeline
produit le meilleur texte final ?"**

## Cas d'usage central BnF

Une bibliothèque numérique veut comparer 3 pipelines hétérogènes
sur le même corpus :

1. **Tesseract** → texte brut (`RAW_TEXT`)
2. **OCR + LLM + remapping ALTO** → ALTO XML enrichi (`ALTO_XML`)
3. **VLM avec sortie markdown structurée** → `CANONICAL_DOCUMENT`

Sans `TextView`, comparer ces 3 pipelines est trompeur : ils ne
produisent pas le même type d'artefact.  Avec `TextView`, chaque
sortie est **projetée vers du texte plat** avant calcul de
CER/WER, et le rapport documente explicitement ce que la vue
**ignore** (géométrie, structure de blocs, ordre de lecture, IDs,
formatage).

## API

```python
from picarones.evaluation.views import build_text_view

# Vue canonique avec valeurs par défaut
view = build_text_view()

# Vue spécialisée (par exemple : OCR seul, sans ALTO/PAGE)
from picarones.domain import ArtifactType
view_ocr_only = build_text_view(
    candidate_types=frozenset({
        ArtifactType.RAW_TEXT,
        ArtifactType.CORRECTED_TEXT,
    }),
    metric_names=("cer", "wer"),
    normalization_profile="medieval_french",
)
```

## Types acceptés (par défaut)

| Type | Projection | Justification |
|---|---|---|
| `RAW_TEXT` | identité | déjà du texte |
| `CORRECTED_TEXT` | identité | déjà du texte (modifié par un LLM) |
| `ALTO_XML` | `AltoToText` | extraction par ordre de lecture, gestion césure |
| `PAGE_XML` | `PageToText` | extraction depuis `<TextEquiv><Unicode>` |
| `CANONICAL_DOCUMENT` | `CanonicalToText` | décode markdown, aplatit JSON canonique |

## Métriques (par défaut)

`cer`, `wer`, `mer`, `wil` — toutes typées `(RAW_TEXT, RAW_TEXT)`
puisque la comparaison se fait toujours après projection vers
texte plat.

## Dimensions explicitement ignorées

Le `ViewResult` propage dans `ignored_dimensions` les dimensions
que cette vue **ne mesure pas** :

- `geometry` — coordonnées HPOS/VPOS/WIDTH/HEIGHT des mots
- `block_structure` — découpage en `TextBlock` / `TextRegion`
- `reading_order` — ordre de lecture spatial
- `ids` — identifiants stables des éléments
- `confidence` — scores de confiance par mot
- `formatting` — gras / italique / titre

Ces dimensions sont éventuellement évaluées par d'autres vues :

- `geometry`, `block_structure`, `reading_order`, `ids` →
  **`AltoView`** (S15)
- `confidence` → vue calibration (existante via S5 metrics)

## Garde-fou méthodologique

Chaque `ViewResult` produit par `TextView` porte un `warnings`
explicite :

> Cette vue compare les sorties textuelles finales après
> projection éventuelle.  Les pipelines qui produisent
> ALTO/PAGE/markdown sont projetés vers du texte plat — leurs
> structures spatiale et documentaire ne sont PAS évaluées ici.
> Pour évaluer la qualité ALTO, voir AltoView (S15).

Ce warning sera affiché en tête du bloc TextView dans le rapport
HTML (S22) pour signaler à un lecteur exactement la portée de la
comparaison.

## Exemple de `ViewResult`

```python
ViewResult(
    view_name="text_final",
    candidate_artifact_id="bnf_doc:vlm:canonical_document",
    ground_truth_artifact_id="bnf_doc:gt:raw_text",
    metric_values={
        "cer": 0.04,
        "wer": 0.12,
        "mer": 0.04,
        "wil": 0.18,
    },
    failed_metrics={},
    projection_report=ProjectionReport(
        source_artifact_id="bnf_doc:vlm:canonical_document",
        source_type=ArtifactType.CANONICAL_DOCUMENT,
        target_type=ArtifactType.RAW_TEXT,
        projector_name="canonical_to_text",
        lossy=True,
        ignored_dimensions=("structure", "formatting", "headers", "links"),
        warnings=("Markdown / JSON canonique projeté en texte plat...",),
    ),
    warnings=(
        "Cette vue compare les sorties textuelles finales...",
        "Markdown / JSON canonique projeté en texte plat...",
    ),
    ignored_dimensions=(
        "geometry", "block_structure", "reading_order", "ids",
        "confidence", "formatting", "structure", "headers", "links",
    ),
)
```

## Limites assumées

- **Pas de comparaison fuzzy / search recall** — c'est `SearchView`
  (S16).
- **Pas d'évaluation structurelle ALTO** — c'est `AltoView` (S15).
- **`CANONICAL_DOCUMENT` peut perdre beaucoup de structure** ; le
  warning du `ProjectionReport` le signale.
- **Pas de pondération inter-pipelines** — chaque pipeline est
  évalué indépendamment ; le ranking et l'agrégation sont la
  responsabilité du caller (typiquement le rapport HTML S22).

## Statut

- ✅ `TextView` (codé + testé)
- ✅ `AltoView` (fidélité documentaire)
- ✅ `SearchView` (recherchabilité fuzzy)
- ⏳ Intégration runner + RunManifest
- ⏳ Tests E2E sur le cas BnF central avec 3 pipelines
