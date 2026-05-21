# AltoView — fidélité documentaire ALTO

`AltoView` est la deuxième vue canonique.  Elle répond à la
question : **"quel pipeline produit le meilleur ALTO
exploitable ?"**

## Distinct de TextView

| Aspect | TextView | AltoView |
|---|---|---|
| Question | "meilleur texte final ?" | "meilleur ALTO exploitable ?" |
| Types acceptés | RAW_TEXT, CORRECTED_TEXT, ALTO, PAGE, CANONICAL | ALTO_XML uniquement |
| Projection | tout → RAW_TEXT | aucune (compare ALTO direct) |
| Mesure | qualité linguistique | fidélité structurelle |
| Métriques | CER, WER, MER, WIL | alto_validity, line_count_ratio, word_box_coverage |

Un même pipeline peut être évalué dans les deux vues.  Le rapport
HTML (S22) présentera les deux côte-à-côte pour qu'un lecteur
comprenne pourquoi deux pipelines avec le même CER peuvent
produire des ALTO de qualités différentes.

## Pattern d'omission explicite

Un pipeline qui ne produit pas d'`ALTO_XML` (exemple : Tesseract
texte brut sans ALTO) ne peut **pas** être évalué dans `AltoView`.
Le caller (typiquement un service applicatif au S19) doit
**omettre** ce pipeline du résultat, plutôt que de lui attribuer
un score factice à 0.

```python
from picarones.evaluation.views import build_alto_view

view = build_alto_view()

pipelines = [
    ("tesseract",       ArtifactType.RAW_TEXT),       # PAS d'ALTO
    ("ocr_llm_alto",    ArtifactType.ALTO_XML),       # ALTO ✓
    ("vlm_alto",        ArtifactType.ALTO_XML),       # ALTO ✓
]

eligible = [(n, t) for n, t in pipelines if view.accepts(t)]
omitted  = [(n, t) for n, t in pipelines if not view.accepts(t)]

# eligible: [("ocr_llm_alto", ALTO_XML), ("vlm_alto", ALTO_XML)]
# omitted: [("tesseract", RAW_TEXT)]
```

Le caller affichera dans le rapport : *"Tesseract n'est pas
évalué dans AltoView (ne produit pas d'ALTO)."*  Pas de score
factice à 0 qui ferait passer Tesseract pour un mauvais ALTO,
alors qu'il n'a juste pas pris part à la compétition.

## Métriques par défaut

### `alto_validity`

L'hypothèse a-t-elle une structure ALTO cohérente ?  ≥ 1 page ET
≥ 1 bloc ET ≥ 1 ligne.  Détecte les ALTO vides, tronqués, ou
produits par un reconstructeur défaillant.

- 1.0 = structure cohérente
- 0.0 = vide ou tronqué

### `alto_line_count_ratio`

Ratio min/max du nombre de lignes : `min(n_hyp, n_ref) / max(n_hyp,
n_ref)` ∈ [0, 1].  1.0 = même nombre de lignes.

Permet de détecter un reconstructeur qui invente ou perd des
lignes.  Ne dit rien sur l'**alignement spatial** — c'est
`textline_alignment` (post-livraison) qui mesurera cette
dimension.

### `alto_word_box_coverage`

Fraction des `AltoString` de l'hypothèse qui ont une `bbox`
définie (HPOS, VPOS, WIDTH, HEIGHT).  1.0 = tous les mots ont
une boîte (cas idéal pour un reconstructeur ALTO).

Un VLM qui produit du markdown puis le reconstruit en ALTO sans
coordonnées aura un `word_box_coverage` proche de 0.

## Garde-fou méthodologique

Le `ViewResult` produit par `AltoView` porte un `warnings`
explicite :

> Cette vue mesure la fidélité STRUCTURELLE de l'ALTO produit
> (validité, nombre de lignes, bbox).  La qualité TEXTUELLE de
> ce qui est dans cet ALTO est mesurée par TextView ; les deux
> doivent être lues ensemble pour juger un pipeline.
>
> Les pipelines qui ne produisent pas d'ALTO sont OMIS de cette
> vue.  Aucun score factice n'est attribué à un pipeline absent.

## Limites assumées

Reportées à des sprints suivants :

- **`textline_alignment`** (IoU des bbox de lignes) — exige un
  algorithme d'alignement bipartite par bbox.
- **`reading_order_consistency`** (Kendall tau sur les IDs de
  lignes) — exige un mapping ID → position.
- **`layout_f1` (ICDAR 2015)** — déjà implémenté dans
  `evaluation/metrics/layout.py` (migré au S10) sur des `Region`
  génériques ; un wrapper ALTO peut être ajouté plus tard.

## Statut

- ✅ `AltoView` (3 métriques + pattern d'omission)
- ✅ `SearchView` (recherchabilité fuzzy)
- ⏳ Intégration runner + RunManifest
- ⏳ Tests E2E sur le cas BnF central
