# Plan d'évolution Picarones — 2026

> Synthèse d'une conversation de cadrage (avril 2026) entre l'équipe Picarones
> et un LLM externe. Couvre à la fois l'enrichissement du benchmark texte
> existant (axe A) et la bascule progressive vers un banc d'essai de pipelines
> composées (axe B), pensés comme **un seul produit à deux modes d'usage**.
>
> Ce document est un plan d'intention, pas une spécification figée. Chaque
> sprint réel devra rouvrir et challenger les hypothèses ci-dessous, en
> particulier celles de la Phase B qui dépendent du retour terrain BnF.

---

## 1. Principe directeur

### 1.1 Un seul produit, deux modes d'usage

Picarones reste **une seule base de code, un seul rapport, un seul runner**.
La distinction entre « benchmark texte » (axe A) et « banc d'essai de
pipelines composées » (axe B) n'est pas un fork du produit, c'est un
continuum :

> **Une pipeline à un seul module est un cas particulier d'une pipeline à
> N modules.**

Cette règle, déjà appliquée avec succès en Sprint 3 quand `OCRLLMPipeline`
a hérité de `BaseOCREngine`, doit être poussée un cran plus loin. Le mode
benchmark texte devient alors le **cas dégénéré** du banc d'essai
pipelines.

### 1.2 Trois modes d'entrée, un seul moteur

| Mode | Commande | Pour qui |
|---|---|---|
| **Legacy** | `picarones run --corpus ./c --engines tesseract,pero` | Chercheur qui n'a jamais entendu parler de pipelines. Aucune nouvelle option à apprendre. |
| **YAML composé** | `picarones pipeline run my-pipeline.yaml` | Ingénieur qui compose une chaîne OCR → reconstructeur → post-correcteur. |
| **Hybride** | `picarones run --pipeline-yaml post.yaml --engines tesseract,pero` | Qui veut comparer l'effet du choix d'OCR à pipeline post-OCR fixée. |

En interne, **les trois construisent la même structure** : un graphe orienté
de modules avec, pour le mode legacy, un graphe trivial à un seul nœud
par moteur.

### 1.3 Rapport adaptatif par défaut

Question à se poser à chaque vue : « si la pipeline a un seul nœud, est-ce
que cette vue apporte quelque chose ? ».

- **Non** → la vue est **absente** (pas vide, pas désactivée).
- **Oui** → la vue est affichée, éventuellement dégradée mais utile.

Application directe du principe du panneau « Avancé » du Sprint 21 :
l'utilisateur simple ne voit pas ce qu'il n'a pas demandé. La discipline du
masquage automatique est ce qui distingue un bon outil d'une usine à gaz.

### 1.4 La règle pratique à tenir

À chaque décision de design : *« est-ce que ça oblige l'utilisateur du mode
simple à apprendre quelque chose de nouveau ? »*. Si oui, mauvaise voie.
Repenser jusqu'à ce que la réponse soit non.

### 1.5 Ce qui ne change pas

Le différenciateur qui distingue déjà Picarones reste central : **rigueur
méthodologique, traçabilité, absence de prescription**. Tout ce que ce plan
ajoute doit servir cette ligne, pas la diluer. En particulier :

- Pas de LLM dans le chemin critique du rapport.
- Pas de score composite imposé. Tout score agrégé est opt-in et
  étiqueté comme tel.
- Chaque chiffre rendu dans la synthèse factuelle reste traçable au
  payload du `Fact` source (garde-fou anti-hallucination du Sprint 19).

---

## 2. Phase 0 — Fondation commune

Trois chantiers à mener avant le reste, sans lesquels tout le plan construit
sur du sable. Ils débloquent à la fois l'axe A (métriques structurelles,
philologiques) et l'axe B (pipelines composées).

### 2.1 Modèle de données multi-niveaux

**Pourquoi maintenant.** La GT actuelle est mono-niveau (texte plat dans
`.gt.txt`). Cette contrainte interdit l'évaluation de tout module qui
produit ou consomme une autre représentation : ALTO, PAGE XML, entités
nommées, ordre de lecture. C'est le verrou qui empêche tout l'axe B et qui
limite déjà des métriques de l'axe A (Layout F1, reading order F1).

**Refonte de `picarones/core/corpus.py`.** La classe `Document` actuelle
porte `image_path`, `ground_truth: str`, `ocr_text`. La nouvelle version
porte une GT structurée :

```python
class GTLevel(str, Enum):
    TEXT = "text"
    ALTO = "alto"
    PAGE = "page"
    ENTITIES = "entities"
    READING_ORDER = "reading_order"

@dataclass
class Document:
    image_path: Path
    ground_truths: dict[GTLevel, GTPayload]
    metadata: dict
```

Chaque payload est typé : `TextGT(str)`, `AltoGT(xml_root)`,
`EntitiesGT(list[Entity])`, etc. Le chargeur `load_corpus_from_directory`
détecte automatiquement les fichiers présents (`.gt.txt`, `.gt.alto.xml`,
`.gt.page.xml`, `.gt.entities.json`) et peuple les niveaux disponibles.

**Compatibilité ascendante stricte.** Un corpus avec uniquement `.gt.txt`
doit continuer à fonctionner exactement comme avant. Le runner consulte
`document.ground_truths[GTLevel.TEXT]`. Une `@property ground_truth` de
transition garantit zéro casse côté tests.

**Conséquences à anticiper.** Les fixtures (`fixtures.py`) doivent
générer des corpus mixtes : text-only et text+alto, pour que la suite de
tests valide les deux régimes. Le rapport HTML doit afficher dans la vue
Document quels niveaux de GT sont disponibles.

**Critère de réussite.** Les 1242 tests actuels passent sans modification.
Trois nouveaux tests valident le chargement d'un corpus avec GT ALTO
partielle (certains documents ont l'ALTO, d'autres non).

**Effort estimé.** Un sprint. Risque : les imports (HuggingFace, IIIF,
HTR-United) peuvent nécessiter de petits ajustements pour produire la
nouvelle structure.

### 2.2 Interface module générique

**Pourquoi maintenant.** Aujourd'hui `BaseOCREngine` est typé `image →
texte` de manière implicite. Pour qu'un même runner puisse exécuter un
mappeur ALTO ou un rewriter, il faut une interface plus générale dont
`BaseOCREngine` devient un cas particulier.

**Création de `picarones/core/modules.py`.**

```python
class ArtifactType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    ALTO = "alto"
    PAGE = "page"
    ENTITIES = "entities"
    READING_ORDER = "reading_order"

class BaseModule:
    name: str
    input_types: tuple[ArtifactType, ...]
    output_types: tuple[ArtifactType, ...]
    execution_mode: Literal["io", "cpu"]

    def process(self, inputs: dict[ArtifactType, Any]) -> dict[ArtifactType, Any]:
        raise NotImplementedError

    def metadata(self) -> dict: ...
```

Un OCR classique déclare `input_types=(IMAGE,)`, `output_types=(TEXT,)`.
Un mappeur VLM→ALTO déclare `input_types=(IMAGE,)`,
`output_types=(TEXT, ALTO)`. Un rewriter ALTO post-correction déclare
`input_types=(ALTO,)`, `output_types=(ALTO,)`.

**`BaseOCREngine` devient un alias de compatibilité** dont l'implémentation
de `process` wrappe l'ancien `_run_ocr` et retourne `{TEXT: result.text}`.
Aucun adaptateur OCR n'est touché à ce stade.

**Critère de réussite.** Tous les moteurs existants passent les tests sans
modification. Un test ajouté instancie un `MockModule` qui consomme `TEXT`
et produit `ALTO`, et vérifie que le runner peut l'invoquer.

**Effort estimé.** Un demi-sprint, principalement de la rigueur de typage.

### 2.3 Métriques composables

**Pourquoi maintenant.** Aujourd'hui une métrique est calculée une fois,
en sortie de moteur, sur la paire `(GT_text, hypothesis_text)`. Dans une
pipeline composée, on veut calculer une métrique **à chaque jonction** du
DAG, et la métrique change selon les types d'artefacts à la jonction.

**Registre de métriques typé.** Chaque métrique déclare ses types d'entrée :

```python
@register_metric(input_types=(TEXT, TEXT))
def cer(reference: str, hypothesis: str) -> float: ...

@register_metric(input_types=(ALTO, ALTO))
def reading_order_f1(ref_alto, hyp_alto) -> float: ...

@register_metric(input_types=(TEXT, ALTO))
def text_preservation_after_reconstruction(ref_text, hyp_alto) -> float: ...
```

Le runner, étant donné une jonction `(artifact_produced, gt_at_this_level)`,
sélectionne automatiquement les métriques applicables et les calcule.
C'est le mécanisme qui rend l'axe B possible.

**Compatibilité.** Tout `compute_metrics` actuel devient un appel orchestré
du registre. Le format de sortie de `MetricsResult` reste identique pour
les jonctions `text→text`.

**Critère de réussite.** Les rapports HTML existants restent strictement
identiques (déterminisme bit-à-bit) sur les fixtures de test. Un nouveau
test enregistre une métrique factice `(TEXT, ALTO)` et vérifie qu'elle est
sélectionnée à la bonne jonction.

**Effort estimé.** Un demi-sprint.

### 2.4 Bilan Phase 0

À la fin de cette phase :

- Tous les tests existants passent sans modification.
- Le rapport HTML est strictement identique sur les fixtures legacy.
- L'infrastructure peut accueillir des modules non-OCR, des GT
  multi-niveaux et des métriques typées par jonction.

**Aucune autre étape ne peut commencer tant que la Phase 0 n'est pas
stable.** Trois sprints consécutifs, sans dérive de scope.

---

## 3. Phase A — Enrichissement métrique et UX

L'axe A travaille sur le périmètre actuel (benchmark texte mono-pipeline)
et bénéficie immédiatement à tous les utilisateurs existants. Il se
décline en deux familles : **A.I — adresser les 9 critiques structurelles
identifiées dans la conversation de cadrage**, et **A.II — combler les
métriques absentes**. Une vue transversale (A.III stratification) clôt
l'axe.

### A.I — Réponses aux 9 critiques structurelles

Chaque sous-section nomme la critique, ce qui existe déjà, et le chantier
à mener.

#### A.I.1 — La granularité ne s'arrête plus à la page

**Critique.** CER/WER agrégés au document, agrégés au moteur. Le niveau
ligne existe (`line_metrics.py`, Gini, percentiles depuis Sprint 10) mais
n'est que décrit. Le niveau token rare est absent.

**Chantier 1 — Vue « Worst lines globale ».** Une nouvelle vue dans le
rapport qui transcende les documents : top-N lignes les plus mal
transcrites de tout le corpus, classées par CER ligne, avec image de la
ligne (extrait ALTO si disponible, sinon crop image), GT et hypothèse en
diff coloré, et lien vers le document parent. C'est le complément
opérationnel du percentile p95 abstrait.

Paramètres : N (défaut 20), filtre par moteur, filtre par strate
`script_type`. Les ingrédients existent (`line_metrics.py`,
`diff_utils.py`, vue galerie) — il manque la requête transversale et la
vue dédiée.

**Chantier 2 — Métrique « rare-token recall ».** Calcul du rappel sur les
tokens GT de fréquence ≤ 2 dans le corpus (hapax + dis legomena).
Implémentation : tokenisation Unicode-aware sur la GT, comptage de
fréquence corpus-wide, calcul du taux de tokens rares correctement
restitués par chaque moteur. Une variante stratifiée par `pos`-tag ou par
catégorie (NOUN, PROPN) est rendue par A.II.1 (NER).

Conjecture à tester sur la fixture : cette métrique discrimine plus les
moteurs que le CER global. Si confirmée, elle gagne sa place dans le
tableau de classement principal à côté du CER.

#### A.I.2 — La médiane devient le critère de classement par défaut

**Critique.** Le rapport affiche les deux mais classe sur la moyenne. Sur
des distributions asymétriques (80 % à 3 %, 20 % à 40 %), le moteur A peut
gagner sur la moyenne uniquement parce qu'il rate moins souvent les pages
catastrophiques, alors que B est strictement meilleur sur le quotidien.

**Chantier.** Le tri par défaut du tableau de classement passe sur la
**médiane CER**. La moyenne devient une colonne secondaire. Un
**avertissement** s'affiche automatiquement si l'écart relatif
`|moyenne − médiane| / médiane` dépasse un seuil (par défaut 30 %), avec
un message explicite : *« la distribution des CER est très asymétrique
sur ce corpus — la moyenne est tirée par quelques documents
catastrophiques. La médiane est plus représentative. »*

Cohérence supplémentaire : le test de Friedman travaille déjà sur les
rangs (Sprint 18). Trier sur la médiane aligne la lecture humaine sur la
lecture statistique.

#### A.I.3 — Vue « inhabituel ici » alimentée par l'historique SQLite

**Critique.** Le rapport décrit le corpus évalué mais ne le compare jamais
à une distribution de référence. L'historique SQLite (Sprint 8) existe
mais aucun détecteur narratif ne le lit.

**Chantier 1 — Encart « Ce corpus est-il habituel ? ».** En tête du
rapport, sous la synthèse factuelle, un mini-graphique qui place le score
de difficulté moyen du corpus courant sur la distribution des corpus
précédents stockés en SQLite (boxplot + point). Une phrase factuelle :
*« Difficulté observée 0,62 — au 88ᵉ percentile des 47 corpus précédents
de votre institution. Ce corpus est plus difficile que la moyenne. »*

**Chantier 2 — Détecteur narratif `engine_off_baseline`.** Pour chaque
moteur, comparer le CER observé à la moyenne historique des derniers
benchmarks. Templates FR/EN du type : *« Tesseract a obtenu 5,2 % CER ici,
vs 4,1 % en moyenne sur les 12 derniers benchmarks de votre institution.
Ce corpus lui est plus difficile que vos corpus habituels. »*

Garde-fous : ne déclenche que si N_runs ≥ 5 et même `corpus_signature`
(empreinte de strates). Sinon le détecteur reste silencieux.

#### A.I.4 — Taxonomie d'erreurs exploitée à 3 niveaux

**Critique.** 10 classes, mais le rapport montre un seul histogramme.

**Chantier 1 — Co-occurrence pattern.** Matrice de co-occurrence des
classes taxonomiques au niveau document. Si `ligature_error` et
`abbreviation_error` co-occurrent toujours, c'est un signal de scribe
particulier — utile pour stratifier le corpus *a posteriori*.
Visualisation : heatmap de Jaccard entre paires de classes.

**Chantier 2 — Évolution intra-document.** Étendre la heatmap de CER par
tranche de page (déjà disponible) à toutes les classes taxonomiques.
Permet de distinguer les erreurs de marge (concentrées en bordure) des
erreurs de scribe (uniformes).

**Chantier 3 — Taxonomie comparative.** Vue côte-à-côte des profils
taxonomiques de deux moteurs avec CER similaire. Diagramme en miroir
(barres horizontales) qui montre que A fait surtout des erreurs de casse
(récupérables) tandis que B fait des lacunes (irrécupérables). Le
détecteur `error_profile_outlier` existe mais ne génère pas cette vue
comparative — il faut la créer.

#### A.I.5 — Normalisation diplomatique en curseur fin

**Critique.** `cer_diplomatic` applique un profil entier. Un éditeur peut
vouloir « je tolère ſ=s mais pas u=v ».

**Chantier.** Le panneau « Avancé » (Sprint 21) gagne une section
**« Équivalences diplomatiques »** : liste de toutes les transformations
des profils (`medieval_french`, `early_modern_french`, etc.) sous forme
de cases à cocher granulaires. Quand l'utilisateur (dé)coche, le CER se
recalcule **côté client** (les hypothèses et GT pré-tokenisées sont déjà
embarquées) et le tableau de classement se met à jour en direct.

État persisté en URL (`?eq=oe-ae,longs-s,...`) comme les autres options
du panneau Avancé. Garde-fou : ne pas recalculer plus d'une fois par
seconde (debounce).

C'est précisément ce qui distingue un benchmark *outil de mesure* d'un
*outil de décision éditoriale*.

#### A.I.6 — Coût projeté en volume cible

**Critique.** La vue Pareto (Sprint 20) trace CER vs coût mais le coût
n'est pas linéaire en valeur — payer 50 € de plus sur 50 pages est
trivial, sur 5 millions ça change tout.

**Chantier.** Champ « **Volume cible** » dans le panneau Avancé : nombre
de pages que l'utilisateur projette de traiter en production. La vue
Pareto et le tableau Pareto recalculent le **coût total projeté** :

> *« Pour vos 80 000 pages BMS — Tesseract = 3 €, Pero = 0 € (local),
> Mistral OCR = 280 €, GPT-4o post-correction = 600 €. »*

Trois axes de la vue Pareto restent disponibles (coût, vitesse, carbone)
mais le coût se lit en valeur projetée plutôt qu'en valeur unitaire. Le
détecteur `cost_outlier` (Sprint 20) gagne un seuil paramétré par
volume — ce qui est trivial à 50 pages devient bloquant à 5 M.

#### A.I.7 — Sur-normalisation LLM en vue analytique dédiée

**Critique.** La classe taxonomique 10, le détecteur d'hallucination, la
vue triple-diff document existent. Mais le score agrégé (« 0,05 % ») ne
dit rien sur **quoi** corriger dans le prompt.

**Chantier.** Nouvelle vue **« Modernisation lexicale »** dédiée aux
moteurs LLM/VLM. Tableau de fréquences :

| Forme historique GT | Forme « modernisée » par le LLM | Occurrences GT | % de fois modernisé |
|---|---|---:|---:|
| maistre | maître | 47 | 85 % |
| nostre | nostre (préservé) | 92 | 8 % |
| veoir | voir | 23 | 100 % |

Calcul : alignement caractère par caractère GT/hypothèse, extraction des
substitutions sur tokens GT non présents dans un dictionnaire moderne
(stop-list paramétrable). Trie les substitutions les plus fréquentes par
moteur.

C'est exploitable pour ajuster le prompt — bien plus utile qu'un score
agrégé.

#### A.I.8 — Robustesse synthétique projetée sur le corpus réel

**Critique.** Le module `robustness.py` génère des dégradations
synthétiques (bruit, flou, rotation). Mais aucun lien avec les
caractéristiques réelles des images.

**Chantier.** Pour chaque document du corpus, mesurer (via
`image_quality.py`) les niveaux de bruit/flou/contraste réels. Projeter
ces niveaux sur la courbe de robustesse synthétique pour estimer le
**déficit attendu de CER** :

> *« 30 % de vos documents ont un bruit équivalent à σ=15 où Tesseract
> perd 8 points de CER — soit un déficit attendu global de 2,4 points. »*

Visualisation : sur le graphique de robustesse, ajouter un histogramme
des qualités réelles observées en arrière-plan. Le détecteur
`robustness_fragile` (Sprint 19) est étendu pour intégrer ce déficit
projeté dans son texte.

#### A.I.9 — Section « Leviers d'amélioration »

**Critique.** Le rapport dit « voici les performances » mais jamais
« voici ce qui changerait facilement le classement ».

**Chantier.** Nouvelle section en pied de rapport, factuelle, qui agrège
des observations actionnables à partir des données déjà calculées :

- *« 45 % des erreurs de Tesseract sont de la classe `casse` —
  applicable post-traitement gratuit. »* (source : taxonomie)
- *« 12 documents concentrent 60 % du CER total — leur rescan en haute
  résolution aurait plus d'impact que de changer de moteur. »* (source :
  distribution par document + `image_quality`)
- *« Mistral OCR et Pero OCR sont complémentaires : leurs erreurs ne se
  chevauchent qu'à 30 %. Un vote majoritaire entre les deux ferait
  passer le CER de 7 % à 4 %. »* (source : Venn d'erreurs +
  `cer_oracle`, voir A.II.8)

Format : liste de cartes, chacune avec un titre court, un chiffre saillant
et un lien vers la vue détaillée qui justifie l'observation.

Architecture : un nouveau registre `levers/` parallèle au registre des
détecteurs narratifs, avec la même contrainte de traçabilité des chiffres
(garde-fou anti-hallucination).

### A.II — Métriques absentes à ajouter

Classées par utilité réelle (haut ROI d'abord). Trois métriques marquent
le premier livrable parce qu'elles ouvrent une dimension d'utilité
nouvelle dans le rapport.

#### A.II.1 — Trois métriques prioritaires (premier livrable)

**A.II.1.a — Précision sur entités nommées (NER).**

Nouveau module `picarones/measurements/ner.py`. Backends : spaCy multilingue,
Stanza, modèle HIPE pour les corpus historiques. Choix paramétré par
profil (`fr_core_news_lg`, `xx_ent_wiki_sm`, `hipe2022`).

Métriques calculées sur la paire `(entities_GT, entities_OCR)` après
alignement par chevauchement de spans :

- Précision, rappel, F1 par catégorie (`PER`, `LOC`, `ORG`, `DATE`, `MISC`)
- F1 global pondéré
- Taux d'**hallucinations d'entité** : entités en OCR sans
  correspondance GT

Le rapport gagne une vue dédiée : tableau moteur × catégorie, drill-down
sur les `PER` ratés. C'est la métrique qui parle directement aux
indexeurs et aux prosopographes.

**Piège.** Les modèles NER hallucinent eux-mêmes. La métrique mesure
conjointement OCR + NER. Documenter explicitement ce biais dans la
glossaire (entrée `ner_score`).

**A.II.1.b — Score de calibration des moteurs.**

Nouveau module `picarones/measurements/calibration.py`. Tous les moteurs cibles
fournissent une confidence par token ou par ligne (Tesseract `tsv`
output, Pero OCR via `PageLayout`, Mistral OCR via `confidence`, Google
Vision via `Word.confidence`). Ajout d'un champ
`EngineResult.token_confidences: list[float]`.

Métriques :

- **ECE** (Expected Calibration Error) en 10 bins
- **MCE** (Maximum Calibration Error)
- **Reliability diagram** en SVG embarqué dans le rapport

Vue « Fiabilité de l'auto-évaluation » qui répond à *« quand le moteur
dit qu'il est sûr, est-il vraiment sûr ? »*. Pour un workflow de
validation humaine, c'est la différence entre vérifier 100 % vs 15 % du
corpus.

**Piège.** Les confidences VLM/LLM sont moins fiables et moins
standardisées (logprobs vs scores arbitraires). Marquer explicitement les
moteurs sans calibration calculable.

**A.II.1.c — Divergence taxonomique entre moteurs.**

Extension de `core/taxonomy.py`. À partir des distributions taxonomiques
agrégées (déjà calculées), ajout de la KL-divergence et de la
Jensen-Shannon-divergence pour chaque paire de moteurs.

Le rapport gagne une **matrice de divergence triangulaire**. Une
divergence élevée signale des moteurs spécialisés sur des erreurs
différentes — candidats pour un voting ensemble. Faible divergence =
mêmes faiblesses, pas de gain attendu.

Couplé au score de complémentarité quantifiée (A.II.8.a).

Détecteur narratif `ensemble_opportunity` qui remonte automatiquement :
*« Pero et Mistral sont fortement complémentaires (KL=0,82) — un voting
majoritaire pourrait faire passer le CER de 7 % à 4 %. »*

**Effort total A.II.1.** Un sprint et demi pour les trois métriques +
leurs vues + la mise à jour du moteur narratif et du glossaire.

#### A.II.2 — Métriques structurelles

Ces métriques ne sont calculables que quand la GT ALTO/PAGE est
disponible (Phase 0.1).

**Reading order F1 par région.** Implémentation de la métrique ICDAR 2015
(Antonacopoulos). Comparaison de l'ordre des `TextRegion` dans la GT et
dans l'hypothèse ALTO. F1 sur les paires consécutives correctement
préservées. Pour les pipelines qui produisent du texte plat, l'ordre est
extrait de l'ordre de concaténation (et la métrique signale cette
extraction comme approximative).

**Layout F1 par type de région.** Précision/rappel sur la détection de
`TextRegion`, `MarginNote`, `Header`, `Footer`, `Drop-Cap`. Métrique
critique pour les manuscrits glosés et les journaux multi-colonnes. Vue
qui répond à *« le moteur sépare-t-il bien le texte principal de la
glose ? »*.

**Différence de score Flesch.** Calcul du score Flesch (ou
Flesch-Kincaid) sur GT et sortie OCR. La différence est un signal
d'over-normalisation indépendant de toute classe taxonomique. **N'exige
aucun alignement**, donc fiable même quand l'OCR est très dégradé.
Particulièrement utile pour repérer les LLM qui « lissent » la langue
historique.

#### A.II.3 — Métriques philologiques

Pour les corpus médiévaux et les éditions critiques.

**Précision par bloc Unicode.** À partir de la matrice de confusion
existante, agrégation par bloc : *Latin de Base* (U+0000–U+007F),
*Latin-1 Supplément*, *Latin Étendu A et B*, *Diacritiques combinants*,
*Présentation latine*. Graphe à barres groupées qui dit immédiatement
*« ce moteur restitue 95 % du Latin de Base mais 12 % des formes de
présentation latines »*. Actionnable en un coup d'œil pour le choix
éditorial.

**Score d'expansion d'abréviations en deux variantes.** Reconnaissance
des abréviations médiévales par leur position Unicode (ꝑ, ꝓ, ꝗ, p̃, q̃)
et calcul de :

- **Strict** : taux de préservation de la forme abrégée Unicode
- **Expansion** : taux de développement correct (ꝑ → per, ꝓ → pro)

Le ratio des deux dit beaucoup sur la convention adoptée par le moteur.
Crucial pour distinguer un OCR diplomatique d'un OCR modernisant.

**Score de couverture MUFI.** Liste de référence : caractères MUFI v4.0
utilisés dans le GT. Mesure : taux de ces caractères correctement
préservés dans l'OCR. Pour les médiévistes, c'est un critère éditorial
central — déjà mentionné dans le glossaire (Sprint 21) mais non mesuré.

#### A.II.4 — Métriques de fiabilité

**Inter-annotator agreement.** Quand le corpus a plusieurs GT (par
exemple deux paléographes), calcul du Cohen κ ou Krippendorff α. Donne le
plafond atteignable. Affiché dans la synthèse factuelle :

> *« Le CER moyen de Pero (4,2 %) approche le plafond humain pour ce
> corpus (κ = 0,89). »*

Nécessite une extension du loader pour accepter `doc_001.gt.A.txt` et
`doc_001.gt.B.txt` comme GT multiples — ce qui s'inscrit naturellement
dans le modèle multi-niveaux de la Phase 0.1.

**Score de stabilité multi-runs.** Le runner gagne une option
`--repeats N` qui exécute N fois la même pipeline LLM sur les mêmes
documents et mesure :

- Variance du CER
- Taux de tokens divergents entre runs
- Divergence sémantique (BERTScore sur paires de sorties)

C'est critique pour la reproductibilité scientifique. Une publication qui
rapporte un CER LLM sans stabilité est méthodologiquement faible.
Détecteur narratif `engine_unstable` qui remonte les moteurs dont la
variance dépasse un seuil.

#### A.II.5 — Métriques d'utilisabilité aval

**OCR-friendliness pour la recherche plein-texte.** Pourcentage de mots
GT dont une variante orthographiquement proche (Levenshtein ≤ 2) existe
dans la sortie OCR. C'est ce que recherchent réellement Elastic et Solr
en mode fuzzy. Un CER de 8 % peut donner 95 % de findability si les
erreurs sont concentrées sur des caractères non-significatifs.

Affichée à côté du CER dans le tableau principal sous le nom
**« Recherchabilité »**.

**Précision sur séquences numériques.** Extraction par regex des dates
(formats classiques et historiques : MDCLXVIII, *mil cinq cens*, 1ᵉʳ
janvier 1789), foliotation, montants, années régnales. Mesure de
précision sur ces séquences. Pour un économiste-historien ou un éditeur
de chartes, c'est un proxy direct de la qualité éditoriale.

#### A.II.6 — Métriques de processus et économiques

**Throughput effectif.**

```
pages_par_heure_utilisable =
    pages_traitées / (durée_totale + temps_correction_humaine_estimé)
```

Le temps de correction humaine est estimé par
`temps_par_erreur × nombre_d_erreurs`. La constante `temps_par_erreur`
est paramétrable (défaut 5 secondes par erreur de mot, justifié par les
études HTR-United).

Discrimine fortement entre un cloud rapide à 30 % de timeouts et un local
lent à 100 % de fiabilité.

**Coût marginal par erreur évitée.** Extension naturelle de la vue
Pareto. Pour chaque paire de moteurs :

```
coût_marginal = (coût_B − coût_A) / (errors_A − errors_B)   # quand B est meilleur
```

Nouvelle colonne dans le tableau Pareto : *« Passer de Tesseract à
Mistral OCR : 0,83 € par erreur évitée »*. Couplé au volume cible (chantier
A.I.6), devient un outil de décision business.

#### A.II.7 — Métriques d'image prédictives

**Score de complexité paléographique.** Trois features simples
combinées :

- Densité de glyphes par cm² (à partir d'OCR amont approximatif)
- Variabilité de hauteur de ligne (écart-type sur les lignes détectées)
- Ratio encre/papier (Otsu)

Combinaison pondérée en un score [0, 1]. Corrélé avec le CER attendu
sur le corpus. Permet d'expliquer une partie de la difficulté observée.

**Score d'homogénéité du corpus.** Variance des features image entre
documents. Score bas = corpus uniforme = moyenne fiable. Score haut =
corpus hétérogène = la moyenne ment, il faut stratifier.

Avertissement automatique en haut de la vue Classement quand le score
d'homogénéité est bas : *« ⚠ Ce corpus est hétérogène (score 0,34) — la
moyenne globale masque des disparités importantes. Voir la vue
stratifiée. »* (renvoie vers A.III.)

#### A.II.8 — Métriques inter-moteurs

**A.II.8.a — Score de complémentarité quantifié.**

```
cer_oracle = tokens_GT_correctement_transcrits_par_AU_MOINS_un_moteur
             / total_tokens_GT
```

Borne inférieure atteignable par voting ensemble. Si `cer_oracle` est
très inférieur au meilleur moteur seul, ça justifie l'effort d'un
pipeline d'ensemble. Sinon non. Affiché à côté du meilleur CER observé
dans la synthèse factuelle.

**A.II.8.b — Score de spécialisation.** Pour chaque paire de moteurs,
la KL-divergence sur les distributions taxonomiques (déjà calculée en
A.II.1.c) sert de score de spécialisation. Moteurs spécialisés
différemment → candidats pour ensemble. Moteurs spécialisés
identiquement → pas de gain attendu.

#### A.II.9 — Métriques longitudinales

L'historique SQLite (`core/history.py`, Sprint 8) existe mais aucune
métrique n'en sort dans le rapport. C'est complémentaire du chantier
A.I.3.

**Pente de progression.** Régression linéaire sur les CER successifs
d'un moteur dans le temps. Trois nombres : pente, R², `n_runs`.
Nouvelle vue **« Évolution dans le temps »** du rapport, un graphe par
moteur avec zone de confiance bootstrap.

**Détection de point de rupture.** Algorithmes de change-point detection
(PELT, Bayesian via `ruptures` ou implémentation Python pure pour les
petits N) sur la série temporelle des CER. Identifie automatiquement la
version où un modèle a changé de comportement. Particulièrement utile
pour relier une régression à un changement de pipeline d'entraînement.

Détecteur narratif `regression_in_history` qui remonte *« Tesseract
montre une régression depuis le run du 12/02 (CER moyen +1,8 points sur
les 3 derniers benchmarks) »*.

### A.III — Stratification par `script_type` (vue transversale)

C'est probablement **la plus haute valeur ajoutée transversale** du plan
A. Aujourd'hui les détecteurs `stratum_winner` et `stratum_collapse`
(Sprint 19) calculent l'information par strate mais elle n'est pas
remontée dans le tableau de classement principal.

**Chantier.** Toggle dans la nav du rapport : *« Stratifier par
script_type »*. Quand activé, le tableau de classement se dédouble :

- Un sous-tableau par strate avec son propre top-3
- Ses propres tests statistiques (Wilcoxon, Friedman)
- Son propre Pareto

Le moteur narratif adapte sa synthèse :

> *« Sur les manuscrits gothiques (n=43), Pero domine ; sur les imprimés
> modernes (n=87), Tesseract suffit. Le classement global agrégé masque
> cette divergence. »*

**Architecture.** Le runner stocke déjà `script_type` par document
(metadata). L'agrégation actuelle est plate ; il faut introduire un
`StratifiedBenchmarkResults` qui contient une `BenchmarkResults` par
strate plus l'agrégat global. Le rapport HTML en mode stratifié remplace
les vues principales par des onglets de strates, avec un onglet « Tous »
qui reprend la vue actuelle.

**Contrainte UX.** Le toggle est désactivé automatiquement si une seule
strate est présente. Il est mis en évidence (badge `Recommandé`) si le
score d'homogénéité (A.II.7) est bas.

C'est le passage d'un rapport qui dit *« voici la performance moyenne »*
à un rapport qui dit *« voici quel moteur choisir pour quel type de
document »*.

**Effort A.III.** Un sprint complet, parce qu'il faut adapter toute la
chaîne de visualisation et plusieurs détecteurs narratifs.

---

## 4. Phase B — Banc d'essai de pipelines (mode optionnel)

Saut conceptuel qui répond à la question scientifique non-résolue
soulevée par le contexte BnF : **le classement des moteurs survit-il à
l'introduction d'une étape de reconstruction de structure ?**.

Ce mode reste **optionnel et masqué par défaut**. Un utilisateur du mode
benchmark texte simple ne voit jamais les vues B tant qu'il ne charge
pas une pipeline composée. Cf. § 1.3 (rapport adaptatif).

### B.1 — Acte 1 : premier cas BnF concret

**Pourquoi ne pas généraliser tout de suite.** L'erreur classique sur ce
type de projet : coder un framework avant d'avoir vu un cas d'usage
réel. On finit avec une abstraction élégante qui ne colle à aucun
workflow. Un cas BnF concret va révéler ce qui manque dans le modèle de
données, comment associer la GT par étape, quelles visualisations ont du
sens.

**Ce qu'il faut faire.** Choisir avec l'équipe BnF un corpus qui réunit
trois conditions :

- GT ALTO disponible et stable
- Volume modeste (50 à 200 pages) pour itérer vite
- Représentatif d'un cas réel (registres, presse, manuscrits)

Construire une première pipeline en YAML :

```yaml
pipeline:
  name: "pero_to_alto_v1"
  steps:
    - module: pero_ocr
      input: image
      output: text
    - module: alto_reconstructor_naive
      input: [image, text]
      output: alto

evaluation:
  junctions:
    - after: pero_ocr
      compare_to: gt.text
      metrics: [cer, wer]
    - after: alto_reconstructor_naive
      compare_to: gt.alto
      metrics: [reading_order_f1, layout_f1, text_preservation]
```

**Pas d'UI graphique à ce stade.** CLI uniquement :
`picarones pipeline run my-pipeline.yaml --corpus path/to/corpus`.

**Ce qui doit sortir.** Un rapport HTML qui montre, document par
document, ce que produit chaque étape de la pipeline et comment ça se
compare à la GT correspondante. Minimum viable.

**Critères de réussite.**

1. La pipeline tourne de bout en bout sur le corpus.
2. Au moins une métrique est calculée à chaque jonction.
3. On identifie au moins trois choses qui manquent dans le modèle de
   données ou dans l'interface module (entrée pour l'acte 2).
4. Le rapport est interprétable par un collègue BnF qui n'a pas codé la
   pipeline.

**Décision critique en fin d'acte 1.** Faire le bilan **avant** de passer
à l'acte 2. Si l'acte 1 a révélé des manques majeurs dans la Phase 0,
refactoriser maintenant. Si tout est solide, continuer.

**Effort B.1.** Deux à trois sprints — la marge couvre les découvertes.

### B.2 — Acte 2 : pipeline composée comme concept

À ce stade on a un cas qui marche. On peut généraliser.

**Extension du concept de « concurrent ».** Aujourd'hui un concurrent est
un moteur OCR ou un OCR+LLM. Demain c'est une chaîne de N modules. Le
runner traite uniformément les deux : un moteur OCR seul est une pipeline
à un seul nœud (cf. § 1.1).

**Métriques par jonction.** À chaque arête du DAG, calculer la métrique
adéquate selon les types d'artefacts. Le rapport décompose la performance
par étape :

> *« La pipeline A (Pero → reconstructor_v2) bat la pipeline B (Tesseract
> → reconstructor_v2) globalement (CER 4,2 % vs 7,8 %). Mais
> reconstructor_v2 produit un meilleur reading-order F1 quand il part de
> Tesseract (0,89 vs 0,82). La différence finale vient entièrement de la
> qualité OCR amont. »*

C'est cette analyse par étape qui transforme un benchmark en outil de
diagnostic.

**Synthèse factuelle adaptée.** Les détecteurs narratifs existants
gagnent des frères pour les pipelines :

- `pipeline_stage_collapse` — une étape effondre la performance
- `module_introduces_more_than_corrects` — voir B.3
- `pipeline_dominated_by_single_stage` — une étape porte tout le gain

Le moteur narratif compare des pipelines au lieu de comparer des moteurs
quand le mode pipeline est actif.

### B.3 — Acte 3 : métrique d'absorption d'erreur

**Pourquoi c'est critique.** Quand un module post-correction LLM aplatit
les différences entre OCR amont, ce n'est pas qu'il « améliore » tous
les moteurs — c'est qu'il introduit ses propres biais qui dominent ceux
de l'OCR. Mesurer la dégradation par étape ne suffit pas. Sans cette
métrique, on confond correction et écrasement, et la communauté
scientifique ne peut pas faire confiance aux conclusions.

**Ce qu'il faut faire.** À chaque jonction où un module transforme un
artefact, mesurer **deux flux séparément** :

- **Taux de correction** : parmi les erreurs présentes en entrée du
  module, combien sont corrigées en sortie ?
- **Taux d'introduction** : parmi les erreurs présentes en sortie,
  combien sont nouvelles (absentes en entrée) ?

C'est la généralisation du score de sur-normalisation existant
(chantier A.I.7) à toute jonction. La formule s'applique uniformément à
OCR→LLM, OCR→reconstructor, VLM→ALTO_mapper.

**Visualisation.** Un graphe **Sankey** par pipeline : à chaque jonction,
deux flux (corrections et introductions) avec leurs volumes. Permet de
voir au premier coup d'œil quels modules « ajoutent du bruit » sous
couvert d'amélioration globale.

**Détecteur narratif `module_writes_more_than_reads`.**

> *« Le module GPT-4o post-corrector introduit 1,3 erreur nouvelle pour
> chaque erreur corrigée — il écrit son propre texte plutôt que de
> corriger Tesseract. »*

C'est précisément la métrique qui légitimise une publication ICDAR/TPDL
sur le sujet.

### B.4 — Acte 4 : visualisation du DAG

**Outil d'inspection, pas de construction.** Le YAML reste source de
vérité.

**Ce qu'il faut faire.** Un nouveau module dans le rapport HTML :
**« Pipeline DAG »**. Affiche le graphe orienté de la pipeline. Chaque
nœud est un module. Chaque arête est annotée avec :

- Le type d'artefact transmis
- La métrique calculée à cette jonction (la plus pertinente)
- Une indication visuelle de qualité (vert/jaune/rouge selon seuils)

Clic sur une arête → drill-down par document : *« À cette étape,
document 47, voici ce qui sort du module en amont, voici ce qui sort en
aval, voici la GT correspondante. »*. Diff coloré façon GitHub à trois
colonnes.

**Pas de drag-and-drop, pas de notebook.** Le visuel sert à inspecter
et déboguer, pas à construire. Une institution sérieuse versionne ses
pipelines en YAML dans Git, pas en JSON exporté d'une UI.

### B.5 — Acte 5 : comparaison incrémentale

**Pourquoi c'est indispensable.** Avec 5 OCR × 3 reconstructeurs × 4
post-correcteurs × 3 mappeurs = 180 pipelines à comparer, le rapport
noie l'information. Il faut un mécanisme de comparaison contrôlée type
design d'expérience.

**Ce qu'il faut faire.** Une nouvelle vue **« Comparaison contrôlée »** :

- Fixer N−1 modules de la pipeline
- Faire varier le N-ième
- Voir l'effet isolé du module qui varie

Présentation : un tableau ANOVA-like qui isole l'effet du module variable
en contrôlant les autres. Tests statistiques associés (Friedman généralisé
sur les rangs des pipelines, Nemenyi pour les comparaisons par paires).

C'est presque un Latin square automatisé. Sans ça, le rapport sur 180
pipelines est inutilisable.

### B.6 — Acte 6 : politique de modules

**Avant d'ouvrir aux contributions externes.**

**Cadre de qualité pour les modules contribués.**

- Test unitaire obligatoire sur un corpus de référence (versionné dans
  le repo)
- Contrat d'entrée/sortie strict avec validation automatique au
  chargement
- Métadonnées obligatoires : licence, auteur, version, citation
  académique si applicable
- Score de qualité affiché dans le rapport pour chaque module utilisé

Un module qui ne passe pas l'audit n'est **pas exécutable**. Pas
exécutable = pas dans le rapport, pas dans la pipeline.

**Stratégie d'ouverture en deux temps.**

1. **Phase fermée.** Modules officiels uniquement, contributions via PR
   sur le repo principal. Tant que l'interface module n'est pas stable,
   accepter des contributions externes force à maintenir une compatibilité
   qu'on n'est pas prêt à offrir.
2. **Phase ouverte.** Une fois 5–6 modules officiels stables et un guide
   de contribution éprouvé, ouvrir via plugins (`picarones-module-X` sur
   PyPI avec mécanisme `entry_points`).

---

## 5. Trajectoire intégrée et calendrier

Le plan se déroule en **six étapes**, certaines en parallèle. Les axes A
et B partagent la même base de code et le même rapport — ils ne
s'opposent jamais.

### Étape 1 — Phase 0 dans son intégralité

**Durée.** 3 sprints (≈ 6 semaines).

Modèle de données multi-niveaux, interface module générique, métriques
composables. Non-négociable. **Aucune autre étape ne peut commencer tant
que ce n'est pas stable.**

À la fin : tous les tests existants passent, la rétrocompatibilité est
garantie, la base est posée pour les axes A et B.

### Étape 2 — Premier livrable de l'axe A

**Durée.** 1,5 sprint (≈ 3 semaines).

Trois métriques prioritaires (A.II.1 : NER, calibration, divergence
taxonomique) + **stratification par script_type (A.III)**, qui est
l'amélioration la plus visible pour les utilisateurs actuels.

Inclure aussi **A.I.2 (médiane par défaut)** parce qu'il s'agit d'un
changement à un seul endroit avec impact immédiat sur la lecture du
rapport.

À la fin : le rapport actuel a gagné une dimension d'utilité aval (NER),
une dimension de fiabilité (calibration), un classement plus honnête
(médiane), et une lecture stratifiée qui change la nature des
recommandations.

### Étape 3 — Acte 1 de l'axe B + suite de l'axe A en parallèle

**Durée.** 3 sprints (≈ 6 semaines).

- **Axe B** : cas BnF concret (B.1) — démarrage de la collaboration
  formalisée avec l'équipe BnF.
- **Axe A en parallèle** : critiques structurelles à fort ROI :
  - A.I.1 (worst lines + rare-token recall)
  - A.I.4 (taxonomie 3 niveaux)
  - A.I.9 (section « Leviers d'amélioration »)
  - A.II.2 (métriques structurelles, qui consommeront la GT ALTO du cas
    BnF)

Les deux axes ne se gênent pas : l'axe A travaille sur l'évaluation
texte+structure mono-pipeline, l'axe B sur l'évaluation pipeline
composée.

**Décision critique en fin d'étape 3.** Bilan du cas BnF avant de passer
à l'étape suivante. Refactoriser la Phase 0 si besoin.

### Étape 4 — Actes 2 et 3 de l'axe B + suite de l'axe A

**Durée.** 3,5 sprints (≈ 7 semaines).

- **Axe B** : pipeline composée comme concept (B.2) + métrique
  d'absorption d'erreur (B.3). C'est ici que la dimension publication
  ICDAR/TPDL devient réelle.
- **Axe A en parallèle** :
  - A.I.3 (vue « inhabituel ici » + détecteur `engine_off_baseline`)
  - A.I.5 (curseur normalisation diplomatique)
  - A.I.6 (volume cible projeté)
  - A.II.4 (fiabilité : IAA + stabilité multi-runs)

À la fin de l'étape 4, **un papier scientifique est rédactible** sur les
résultats du cas BnF avec absorption d'erreur quantifiée.

### Étape 5 — Visualisation DAG + métriques longitudinales et avancées

**Durée.** 3 sprints (≈ 6 semaines).

- **Axe B** : visualisation DAG (B.4)
- **Axe A** :
  - A.I.7 (sur-normalisation LLM en vue analytique)
  - A.I.8 (robustesse projetée)
  - A.II.3 (philologiques : Unicode/abbrev/MUFI)
  - A.II.5 (utilisabilité aval)
  - A.II.6 (économique)
  - A.II.7 (image prédictives)
  - A.II.8 (inter-moteurs : oracle + spécialisation)
  - A.II.9 (longitudinales)

À ce stade, l'outil est mature et commence à attirer l'attention de la
communauté patrimoniale.

### Étape 6 — Maturité institutionnelle

**Durée.** 3 sprints (≈ 6 semaines).

- B.5 (comparaison incrémentale)
- B.6 (politique de modules)
- Ouverture aux contributions externes en phase fermée → ouverte
- Documentation institutionnelle (guide d'écriture de modules,
  audit-bench)
- Premiers retours d'autres institutions

### Synthèse temporelle

| Étape | Durée | Cumul | Livrable saillant |
|---|---:|---:|---|
| 1 — Phase 0 | 3 sprints | 6 sem | Fondations multi-niveaux |
| 2 — A premier livrable | 1,5 sprint | 9 sem | NER + calibration + médiane + stratif |
| 3 — B acte 1 + A suite | 3 sprints | 15 sem | Premier cas BnF + worst lines + leviers |
| 4 — B actes 2-3 + A | 3,5 sprints | 22 sem | Absorption d'erreur (publication possible) |
| 5 — DAG + métriques restantes | 3 sprints | 28 sem | Outil mature |
| 6 — Maturité | 3 sprints | 34 sem | Ouverture externe |

**Total estimé.** 17 sprints, soit **8–9 mois** à un sprint de deux
semaines. Ambitieux mais cohérent avec ce qui a déjà été construit
(22 sprints terminés à date).

---

## 6. Décisions structurantes à trancher avant le premier sprint

Trois choix qui contraignent tout le reste. Mieux vaut les fixer
maintenant que les découvrir au sprint 8.

### 6.1 Périmètre de la marque « Picarones »

**Question.** Dans le mode legacy, Picarones reste un benchmark. Dans le
mode pipeline, Picarones devient un banc d'essai. Faut-il un seul
produit ou deux noms ?

**Recommandation.** **Un seul nom.** Les deux dimensions partagent la
même philosophie (rigueur, traçabilité, absence de prescription) et le
même socle technique. Documenter clairement les deux modes d'usage dans
le `README.md` mis à jour, avec une section *« À qui ça s'adresse »* qui
distingue les deux profils.

### 6.2 Partenariat BnF formalisé

**Question.** Le plan B est intenable sans un cas BnF en condition
réelle.

**Recommandation.** Co-écrire dès maintenant un document court (1 page)
avec l'équipe BnF qui définit :

- Quel corpus (référence stable, accord d'utilisation)
- Quelle GT disponible et à quel niveau (texte ? ALTO ? les deux ?)
- Quelle pipeline cible (que veut-on évaluer ?)
- Quel critère de succès (que doit produire le rapport ?)
- Quel calendrier (qui livre quoi à quelle date ?)

Sans ce document, on construit sur des hypothèses internes qui peuvent
diverger des besoins réels.

### 6.3 Politique de contribution externe

**Question.** Accepter ou non des modules tiers dès le départ ?

**Recommandation.** **Commencer fermé**, pour deux raisons :

1. Tant que l'interface module n'est pas stable, accepter des
   contributions externes force à maintenir une compatibilité qu'on
   n'est pas prêt à offrir.
2. Avant 5–6 modules officiels stables et bien testés, on n'a pas le
   recul pour juger de la qualité d'un module externe.

Ouvrir au sprint 17 (étape 6), pas avant.

---

## 7. Le cap à tenir

Le **différenciateur de fond** reste celui qui distingue déjà Picarones :
**rigueur méthodologique, traçabilité, absence de prescription**. Tout ce
qui s'ajoute doit servir cette ligne, pas la diluer.

### 7.1 Pièges identifiés à ne pas commettre

**Piège 1 — Coder le framework de l'axe B avant le cas d'usage.** La
Phase 0 prépare l'infrastructure, mais l'acte 1 de l'axe B doit
absolument commencer par un cas BnF concret avant toute généralisation.
Si on code `BaseModule`, le runner pipeline et la visualisation DAG en
partant d'intuitions, on finira avec une abstraction qui ne colle à
aucun workflow.

**Piège 2 — Le mode hybride en patchwork.** Tentation : ajouter des
options à `picarones run` au fil du temps (`--with-pipeline`,
`--enable-junction-metrics`, `--show-dag`). Au bout de six mois, quinze
flags qui interagissent mal. Mauvaise voie. La bonne voie : un seul
concept central — la pipeline — qui peut avoir un seul nœud (cas legacy
invisible) ou plusieurs (cas avancé explicite). La CLI legacy
`picarones run` construit en interne une pipeline triviale.
L'utilisateur ne voit jamais cette construction.

**Piège 3 — La progressivité comme concession.** Tentation inverse :
afficher tout dans le rapport, « au cas où l'utilisateur en aurait
besoin ». Si une vue n'apporte rien quand la pipeline est simple, elle
doit être **absente** — pas juste vide ou désactivée. La discipline du
masquage automatique est ce qui distingue un bon produit d'un produit
qui veut tout faire.

**Piège 4 — Aplatir les jonctions.** Une métrique calculée à une jonction
text→text (CER) et une métrique à une jonction image→ALTO (Layout F1) ne
sont pas comparables. Le rapport doit clairement séparer les deux dans
les vues. Le typage strict de la Phase 0.3 garantit cette séparation,
**à condition de ne jamais contourner le système**.

**Piège 5 — Régressions silencieuses.** À chaque sprint, garder un set
de tests qui exécute des configurations purement legacy (pas de YAML,
pas de pipeline composée, GT mono-niveau) et vérifie que le rapport
produit est strictement identique octet par octet à ce qui est produit
aujourd'hui. C'est le seul garde-fou contre les régressions silencieuses
quand on enrichit le runner.

### 7.2 Question à se poser à chaque PR

À chaque décision de design, à chaque PR, une seule question :

> *« Est-ce que ce changement oblige l'utilisateur du mode simple à
> apprendre quelque chose de nouveau ? »*

Si oui, **mauvais design**. Repenser jusqu'à ce que la réponse soit non.

C'est la règle qui garantit que les axes A et B coexistent vraiment au
lieu de cohabiter difficilement.

---

## 8. Annexe — Mapping critiques → chantiers

Vérification que chaque point soulevé dans la conversation de cadrage
trouve sa réponse dans ce plan.

### 8.1 Critiques structurelles

| # | Critique | Chantier(s) |
|---:|---|---|
| 1 | Granularité s'arrête à la page | A.I.1 (worst lines + rare-token recall) |
| 2 | CER moyen au lieu de médian | A.I.2 (médiane par défaut) |
| 3 | Pas de vue « inhabituel ici » | A.I.3 (encart historique + détecteur off-baseline) |
| 4 | Taxonomie sous-exploitée | A.I.4 (co-occurrence + intra-doc + comparative) |
| 5 | Normalisation diplomatique binaire | A.I.5 (curseur dans panneau Avancé) |
| 6 | Coût décorrélé de la valeur | A.I.6 (volume cible projeté) + A.II.6 (coût marginal) |
| 7 | Sur-normalisation LLM en colonne | A.I.7 (vue analytique + table fréquences) |
| 8 | Robustesse déconnectée du benchmark | A.I.8 (projection qualité réelle sur courbe) |
| 9 | Pas de signal sur leviers | A.I.9 (section dédiée + registre `levers/`) |

### 8.2 Métriques additionnelles

| Métrique | Chantier |
|---|---|
| NER (précision/rappel/F1 par catégorie) | A.II.1.a |
| Calibration des moteurs (ECE, MCE, reliability) | A.II.1.b |
| Divergence taxonomique inter-moteurs (KL/JS) | A.II.1.c |
| Reading order F1 (ICDAR 2015) | A.II.2 |
| Layout F1 par type de région | A.II.2 |
| Différence Flesch GT vs OCR | A.II.2 |
| Précision par bloc Unicode | A.II.3 |
| Score d'expansion d'abréviations (strict/expansion) | A.II.3 |
| Couverture MUFI | A.II.3 |
| Inter-annotator agreement (Cohen κ / Krippendorff α) | A.II.4 |
| Stabilité multi-runs | A.II.4 |
| Recherchabilité (Levenshtein ≤ 2) | A.II.5 |
| Précision sur séquences numériques | A.II.5 |
| Throughput effectif | A.II.6 |
| Coût marginal par erreur évitée | A.II.6 |
| Complexité paléographique | A.II.7 |
| Homogénéité du corpus | A.II.7 |
| CER oracle / complémentarité | A.II.8.a |
| Score de spécialisation (KL) | A.II.8.b |
| Pente de progression (régression linéaire) | A.II.9 |
| Détection de point de rupture (PELT/Bayesian) | A.II.9 |

### 8.3 Banc d'essai pipelines (contexte BnF)

| Question | Acte |
|---|---|
| GT multi-niveaux (texte + ALTO + entités) | Phase 0.1 |
| Module générique typé I/O | Phase 0.2 |
| Métriques par jonction | Phase 0.3 |
| Premier cas BnF (Pero → reconstructeur ALTO) | B.1 |
| Pipeline composée comme concurrent | B.2 |
| Absorption vs introduction d'erreur | B.3 |
| Visualisation DAG inspectable | B.4 |
| Comparaison contrôlée (Latin square) | B.5 |
| Politique de modules contribués | B.6 |

### 8.4 Synthèse de l'unification produit

| Préoccupation soulevée | Réponse architecturale |
|---|---|
| Coexister sans gêner les chercheurs actuels | § 1.1 (pipeline 1-nœud = cas dégénéré) |
| Trois modes d'entrée sans complexifier la CLI | § 1.2 (legacy / YAML / hybride) |
| Vues B masquées en mode simple | § 1.3 (rapport adaptatif) |
| Pas obliger à apprendre du nouveau | § 1.4 + § 7.2 (règle pratique) |
| Une seule base de code, pas deux produits | § 6.1 (un seul nom Picarones) |

---

## 9. Pour démarrer

Ordre concret des trois prochaines actions, dans cet ordre, sans
négociation :

1. **Co-écrire le mémo BnF (§ 6.2)** — 1 page, signée des deux côtés,
   définissant corpus / GT / pipeline cible / critère / calendrier.
2. **Ouvrir une branche `phase-0/multi-level-gt`** et coder la refonte
   du modèle `Document` (chantier 2.1) en gardant la rétrocompatibilité
   stricte (test legacy bit-à-bit dès le premier commit).
3. **Lister les modules officiels candidats pour la phase fermée
   (§ 6.3)** : OCR existants, premier reconstructeur ALTO BnF, premier
   post-correcteur LLM. Cette liste devient le périmètre de stabilisation
   de l'interface module pour les sprints 1 à 17.

Tout le reste découle.
