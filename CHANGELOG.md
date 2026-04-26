# Changelog — Picarones

Tous les changements notables de ce projet sont documentés dans ce fichier.

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).
La numérotation de version suit [Semantic Versioning](https://semver.org/lang/fr/).

---

## [1.2.x] — Sprints 32+ — 2026-04 → ongoing

> Démarrage de la **Phase 0** du [plan d'évolution 2026](docs/roadmap/evolution-2026.md) :
> fondations communes pour l'enrichissement métrique (axe A) et le banc
> d'essai de pipelines composées (axe B). Les deux axes restent
> rétrocompatibles avec le mode benchmark texte historique.

### Ajouté

- **Sprint 42 — A.II.1.b Calibration : exposition `token_confidences` +
  câblage runner.** Suite directe du Sprint 39 (couche de calcul). Le
  runner peut maintenant calculer ECE/MCE/reliability dès qu'un moteur
  expose des confidences au niveau token.
  - `EngineResult.token_confidences: Optional[list[dict[str, Any]]]`
    ajouté. Format attendu : `[{"token": str, "confidence": float}, …]`,
    confidence ∈ [0, 1] ou ∈ [0, 100] (normalisé par le runner).
    `None` par défaut → comportement strictement rétrocompat pour tous
    les adapters historiques (Tesseract, Pero, Mistral OCR, Google
    Vision, Azure DI). L'adaptation de chaque adapter à exposer ses
    confidences natives est reportée à des sprints dédiés (un par
    adapter).
  - `DocumentResult.calibration_metrics: Optional[dict]` ajouté
    (sérialisé dans `as_dict` quand renseigné, libéré par `compact()`).
  - `EngineReport.aggregated_calibration: Optional[dict]` ajouté.
  - Helper `_calibration_from_engine_result(ground_truth, token_confidences)` :
    aligne par bag-of-words avec multiplicité (proxy oracle, comme
    `oracle_token_recall` du Sprint 35), normalise les confidences en
    pourcentage à `[0, 1]`, ignore les confidences négatives
    (Tesseract met -1 pour les non-mots), retourne `None` sur entrée
    vide. Appelé dans `_compute_document_result` quand
    `EngineResult.token_confidences` est non-vide.
  - Helper `_aggregate_calibration(doc_results)` : combine les bins de
    tous les docs en somme pondérée par count, recalcule ECE/MCE micro
    sur l'ensemble. Renvoie `None` si aucun doc n'a de
    `calibration_metrics`.
  - +17 tests dans `test_sprint42_calibration_runner.py` couvrant le
    nouveau champ EngineResult, la sérialisation et compact des
    nouveaux champs DR/ER, l'helper d'alignement (calibration parfaite,
    normalisation %, skip négatifs, bag-of-words avec multiplicité,
    skip entrées invalides), l'agrégateur (combinaison de bins
    multi-docs, recalcul ECE/MCE micro), et la rétrocompat
    (pas de calcul sans token_confidences).

- **Sprint 41 — A.II.1.a NER : vue HTML dédiée (clôture A.II.1.a).**
  Suite directe des Sprints 38-40. Le moteur narratif et le runner ont
  déjà tout ce qu'il faut ; ce sprint rend les chiffres visibles et
  vérifiables dans le rapport.
  - Nouveau module `picarones/report/ner_render.py` :
    - `build_ner_summary_html(engines_summary, labels)` : tableau
      résumé F1 global / Précision / Rappel / Docs évalués /
      Hallucinations / Missed par moteur, cellule F1 colorée par
      gradient rouge → jaune → vert.
    - `build_ner_per_category_html(engines_summary, labels)` : heatmap
      moteur × catégorie d'entité (PER, LOC, DATE, ORG, MISC…).
      Cellules colorées par F1, cellule vide marquée d'un `—` pour
      les catégories non observées chez ce moteur. Tooltip `support=N`
      sur chaque cellule.
    - Rendu strictement serveur-side, déterministe, **pas de
      JavaScript**.
    - Anti-injection : noms de moteurs et labels de catégories passés
      à `html.escape`.
  - `_build_report_data` expose `aggregated_ner` par moteur dans
    `engines_summary`. `ReportGenerator.generate` calcule les deux
    blocs HTML et les passe au template `view_analyses.html` qui les
    affiche dans une `chart-card` à largeur pleine **uniquement si au
    moins un moteur a un `aggregated_ner`** (principe du rapport
    adaptatif).
  - +12 clés i18n FR/EN (`h_ner`, `ner_note`, `ner_summary_caption`,
    `ner_per_category_caption`, `ner_engine_label`, `ner_f1_label`,
    `ner_precision_label`, `ner_recall_label`, `ner_doc_count_label`,
    `ner_hallucinated_label`, `ner_missed_label`, `ner_no_data_label`).
  - +38 tests dans `test_sprint41_ner_html.py` couvrant le rendu
    (résumé, heatmap, multi-moteurs, union des catégories, cellule
    vide), le masquage adaptatif (3 cas dégénérés), l'anti-injection
    (engine et label avec balises HTML), l'intégration rapport (FR +
    EN), et la complétude i18n sur les 12 clés × 2 langues.

- **Sprint 40 — A.II.1.a NER : backend extracteur + câblage runner.**
  Suite directe du Sprint 38 (couche de calcul pure). Le runner peut
  maintenant calculer les métriques NER de bout-en-bout quand le corpus
  porte une GT entités (`EntitiesGT` du Sprint 32).
  - Nouveau module `picarones/core/ner_backends.py` :
    - `EntityExtractor` (Protocol) : tout callable
      `(text) -> list[dict]` est un extracteur valide.
    - `SpacyEntityExtractor(model_name, label_mapping=None)` : lazy-import
      de spaCy, charge le modèle au premier appel, met en cache. Si
      spaCy absent OU modèle non téléchargé, fallback gracieux silencieux
      (retourne `[]`) avec **warning explicite** au premier appel
      (cf. règle CLAUDE.md). Mapping par défaut spaCy → conventions HIPE
      (PERSON → PER, GPE → LOC, TIME → DATE, etc.).
    - `SPACY_PROFILES` : 6 profils nommés (fr, fr_lg, en, en_lg,
      multilingual, hipe).
    - `get_extractor(profile)` : factory qui accepte clé de profil ou
      nom de modèle direct.
    - `is_spacy_available()` : test sans charger de modèle.
  - `DocumentResult.ner_metrics: Optional[dict]` ajouté ; sérialisé
    dans `as_dict()` quand renseigné, libéré par `compact()`.
  - `EngineReport.aggregated_ner: Optional[dict]` ajouté avec micro-F1
    global recalculé à partir des sommes TP/FP/FN, détail par catégorie,
    totaux d'hallucinations et missed.
  - `runner.run_benchmark` accepte un nouveau paramètre optionnel
    `entity_extractor`. Si fourni, le runner appelle deux helpers en
    post-process (main process, **pas** dans les sous-processus pour
    éviter de pickler spaCy) :
    - `_attach_ner_metrics(corpus, doc_results, extractor)` : pour
      chaque doc avec `GTLevel.ENTITIES`, extrait les entités sur
      l'hypothèse et calcule `compute_ner_metrics`.
    - `_aggregate_ner(doc_results)` : agrège au niveau du moteur
      (micro-F1 + per_category + totaux).
    Les exceptions par doc sont dégradées en warning, le benchmark
    continue.
  - **Rétrocompat stricte** : sans `entity_extractor`, aucun calcul
    NER n'a lieu, aucun champ n'est ajouté, le rapport reste identique.
  - Nouveau extra `[ner]` dans `pyproject.toml` (`spacy>=3.7.0`) — non
    installé par défaut.
  - +16 tests dans `test_sprint40_ner_runner.py` couvrant le fallback
    sans spaCy + warning, l'idempotence du load, les profils + factory,
    la sérialisation des nouveaux champs (omis quand None, présents
    quand renseignés, libérés par compact), le câblage runner avec un
    extracteur mock injecté, l'agrégation micro-F1, la rétrocompat
    sans extracteur, et la robustesse à un extracteur qui lève.

- **Sprint 39 — A.II.1.b Calibration des moteurs : couche de calcul.**
  Deuxième brique des trois métriques prioritaires de l'Étape 2 (axe A —
  fiabilité). Stratégie identique aux Sprints 35-38 : couche de calcul
  pure, exposition des `token_confidences` sur les `EngineResult` et
  câblage runner+narratif+HTML aux sprints suivants.
  - Nouveau module `picarones/core/calibration.py` :
    - dataclass `CalibrationBin(bin_low, bin_high, avg_confidence,
      accuracy, count)` avec propriété `gap` (renvoie `None` si bin vide)
    - `reliability_diagram(confidences, is_correct, n_bins=10)` : binning
      équidistant de la confiance, calcul de la précision moyenne et de
      la confiance moyenne par bin
    - `expected_calibration_error` (ECE) : moyenne pondérée par bin de
      `|conf - accuracy|`, ∈ [0, 1], 0 = calibration parfaite
    - `maximum_calibration_error` (MCE) : pire écart sur tous les bins
      non vides
    - `compute_calibration_metrics` : vue agrégée
  - **Calcul d'index de bin par multiplication** (`int(c * n_bins)`)
    plutôt que division, pour éviter les pièges IEEE 754 (`0.6 / 0.1 =
    5.999…` en flottant). Cas testé.
  - Aucune dépendance externe ; les listes `confidences` et `is_correct`
    sont fournies en entrée. L'extraction depuis les engines existants
    (Tesseract `tsv`, Pero `PageLayout`, Mistral `confidence`, Google
    Vision `Word.confidence`) est explicitement reportée à un sprint
    dédié.
  - +32 tests dans `test_sprint39_calibration.py` couvrant la
    calibration parfaite (ECE = 0), les cas extrêmes (sur-confiance et
    sous-confiance → ECE = 0,5), le biais constant (ECE = `|conf - acc|`),
    le binning correct (bornes équidistantes, c=1.0 dans le dernier bin,
    affectation correcte y compris pour 0.6), les bins vides
    (avg/accuracy/gap = `None`), les listes vides, les garde-fous
    (longueurs incompatibles, conf hors [0, 1], n_bins ≤ 0), `n_bins`
    paramétrable + monotonie « ECE ne décroît pas avec un binning plus
    fin ».

- **Sprint 38 — A.II.1.a NER : couche de calcul.** Première brique
  des trois métriques prioritaires de l'Étape 2 du plan d'évolution
  (axe A — utilité aval). Stratégie de découpage analogue à la
  divergence taxonomique (Sprints 35-37) : couche de calcul d'abord,
  câblage runner+narratif+HTML aux sprints suivants.
  - Nouveau module `picarones/core/ner.py` :
    - dataclass `Entity(label, start, end, text)` avec validation de
      span ;
    - `compute_ner_metrics(reference_entities, hypothesis_entities,
      iou_threshold=0.5)` qui aligne par chevauchement IoU
      (greedy, IoU décroissant, chaque entité matchée une seule fois)
      et retourne precision/recall/F1 globaux + par catégorie, plus
      les listes des entités hallucinées (FP) et manquées (FN) ;
    - format de dict compatible `EntitiesGT` du Sprint 32 (clé
      `{label, start, end, text}`).
  - Métrique `ner_f1` enregistrée dans le registre typé Sprint 34
    pour la jonction `(ENTITIES, ENTITIES)` — peut être appelée par
    `compute_at_junction` aussi simplement que `cer` sur `(TEXT, TEXT)`.
  - Aucune dépendance externe (pas de spaCy ni Stanza dans ce sprint) :
    les listes d'entités sont fournies en entrée. Le backend extracteur
    suit dans un sprint dédié.
  - +19 tests dans `test_sprint38_ner_metrics.py` (cas standard parfait/FN/FP,
    label case-insensitive, IoU sous/sur seuil, custom threshold,
    multi-catégorie, alignement greedy avec une seule entité matchée,
    best-IoU wins, cas dégénérés vide/asymétrique, validation Entity,
    intégration registre).

- **Sprint 37 — Section inter-moteurs dans le rapport HTML.** Suite directe
  des Sprints 35-36 : la couche de calcul est maintenant visible côté
  rapport.
  - Nouveau module `picarones/report/inter_engine_render.py` :
    `build_divergence_matrix_html` (table HTML colorée — heatmap CSS
    inline, gradient blanc → rouge proportionnel au max hors-diagonale,
    diagonale étiquetée et grisée, paire la plus divergente annoncée
    en sous-titre) et `build_oracle_gap_html` (encart factuel avec
    best engine, recall préservé, oracle, gap absolu/relatif, doc count).
    Rendu strictement serveur-side, **pas de JavaScript** — déterministe
    comme le SVG du CDD (Sprint 18).
  - `ReportGenerator.generate` calcule les deux blocs et les passe au
    template `view_analyses.html` qui les insère dans une nouvelle
    `chart-card` à largeur pleine, **uniquement si présents**
    (principe du rapport adaptatif : moins de 2 moteurs ou taxonomie
    absente → section omise sans laisser de trace).
  - +14 clés i18n FR/EN (`h_inter_engine`, `inter_engine_note`,
    `divergence_*`, `oracle_*`).
  - Anti-injection HTML : tous les noms de moteurs et étiquettes sont
    passés à `html.escape` avant insertion.
  - +42 tests dans `test_sprint37_inter_engine_html.py` (rendu de la
    matrice avec valeurs et paire max, masquage adaptatif sur 4 cas
    dégénérés, anti-injection sur engine name `<script>`, intégration
    rapport FR + EN, complétude i18n sur les 14 clés × 2 langues).

- **Sprint 36 — Câblage inter-moteurs au runner et au moteur narratif.**
  Suite directe du Sprint 35 (couche de calcul) :
  - `picarones/core/inter_engine.py` gagne `compute_inter_engine_analysis`
    qui agrège la complémentarité doc par doc (oracle global + recall par
    moteur + per_doc top 50 trié par gap), et la divergence taxonomique
    (matrice + paire la plus divergente).
  - `BenchmarkResult.inter_engine_analysis: Optional[dict]` exposé dans
    `as_dict()` quand renseigné, absent sinon (rétrocompat stricte).
  - `runner.run_benchmark` collecte les hypothèses brutes par moteur
    avant `compact()`, calcule l'analyse inter-moteurs si ≥ 2 moteurs,
    et stocke le résultat sur le `BenchmarkResult`.
  - Nouveau `FactType.ENSEMBLE_OPPORTUNITY` + détecteur
    `detect_ensemble_opportunity` (priority 130, importance MEDIUM/HIGH
    selon `relative_gap`). Seuil de déclenchement à 25 % du gap relatif
    pour ne pas bruiter la synthèse.
  - Templates FR et EN ajoutés à `narrative/templates/{fr,en}.yaml` —
    aucun nombre en dur (vérifié par test), tout vient du payload.
  - `_build_report_data` du générateur HTML expose
    `report_data["inter_engine_analysis"]` afin que le détecteur le
    voie en mode "rapport".
  - +22 tests dans `tests/test_sprint36_ensemble_narrative.py`
    (agrégation, exposition `BenchmarkResult.as_dict`, déclenchement et
    seuils du détecteur, fallback paire sans taxonomie, intégration
    `build_synthesis` FR + EN, traçabilité anti-hallucination,
    template sans chiffres en dur).

- **Sprint 35 — Étape 2 du plan d'évolution : métriques inter-moteurs
  (couche de calcul).** Nouveau module `picarones/core/inter_engine.py`
  qui expose deux familles de mesures qui ne dépendent que des données
  déjà produites par le runner :
  - **Divergence taxonomique** : `kl_divergence`,
    `jensen_shannon_divergence` (symétrique, bornée dans `[0, 1]`),
    `taxonomy_divergence_matrix` qui construit la matrice triangulaire
    inter-moteurs sur les distributions de classes d'erreur (issues de
    `taxonomy.py`). Lissage epsilon des zéros pour éviter `log(0)`.
  - **Complémentarité** : `oracle_token_recall` (borne supérieure
    bag-of-words du recall atteignable par voting), `complementarity_gap`
    qui retourne aussi `best_single_recall` / `absolute_gap` /
    `relative_gap` / `best_engine`, `pairwise_disagreement_rate` pour
    quantifier le potentiel d'ensemble entre deux moteurs spécifiques.
  - Fonctions pures, sans I/O ni intégration runner — la couche de calcul
    est livrable indépendamment ; le câblage au moteur narratif
    (`ENSEMBLE_OPPORTUNITY`) et au rapport HTML (matrice de divergence,
    badge oracle gap) suit au Sprint 36.
  - +27 tests dans `tests/test_sprint35_inter_engine.py` couvrant les
    invariants mathématiques (KL ≥ 0, KL(p,p) = 0, JS symétrique et
    bornée, oracle ≥ best_single), les cas concrets (moteurs
    spécialisés ressortent comme candidats à un ensemble, complémentarité
    parfaite atteint oracle = 1), les garde-fous (référence vide,
    hypothèses vides, métrique inconnue).

- **Sprint 34 — Phase 0.3 : registre typé de métriques (clôture Phase 0).**
  Nouveaux modules `picarones/core/metric_registry.py` et
  `picarones/core/builtin_metrics.py` :
  - `MetricSpec` (dataclass figée) déclare `name`, `func`,
    `input_types: tuple[ArtifactType, ArtifactType]`, `description`,
    `higher_is_better`, `tags`
  - Décorateur `@register_metric(name=..., input_types=..., ...)`
    enregistre une métrique dans un registre global ; double
    enregistrement avec le même nom interdit, signature non-paire rejetée
  - `select_metrics(input_types)` retourne les métriques applicables à
    une jonction
  - `compute_at_junction(reference, hypothesis, input_types)` calcule
    toutes les métriques sélectionnées et tolère les erreurs unitaires
    (`logger.warning`, jamais `except: pass`)
  - `builtin_metrics.py` enregistre `cer`, `wer`, `mer`, `wil` sur
    `(TEXT, TEXT)` plus le stub `text_preservation_after_reconstruction`
    sur `(TEXT, ALTO)` comme preuve de concept de jonction hétérogène
  - **Approche additive stricte** : ni `metrics.py` ni `compute_metrics`
    ne sont modifiés ; le rapport HTML existant reste strictement
    identique octet par octet
  - +21 tests dans `tests/test_sprint34_metric_registry.py` couvrant
    l'enregistrement, la sélection par signature exacte, la résilience
    aux erreurs (`skip_on_error`), la **parité numérique** avec
    `compute_metrics` legacy sur 4 paires de textes (CER/WER/MER/WIL
    identiques à 1e-9 près), les garde-fous (double enregistrement,
    arité), et le stub TEXT→ALTO

- **Sprint 33 — Phase 0.2 : interface module générique.** Création de
  `picarones/core/modules.py` :
  - Enum `ArtifactType` (IMAGE, TEXT, ALTO, PAGE, ENTITIES, READING_ORDER) —
    valeurs string alignées sur `GTLevel` pour conversion triviale
  - Classe abstraite `BaseModule` avec `input_types`/`output_types`
    déclaratifs, `execution_mode: "io"|"cpu"`, méthode `process` typée
    `dict[ArtifactType, Any] → dict[ArtifactType, Any]`, helpers
    `validate_inputs`/`validate_outputs`, `metadata()` libre
  - `BaseOCREngine` hérite désormais de `BaseModule` avec
    `input_types=(IMAGE,)`, `output_types=(TEXT,)`. Sa nouvelle méthode
    `process` wrappe l'API historique `run()`. Aucun adaptateur OCR
    existant (Tesseract, Pero, Mistral OCR, Google Vision, Azure DI) n'est
    touché — le test_engines.py passe sans modification.
  - +23 tests dans `tests/test_sprint33_module_interface.py` couvrant le
    contrat (instanciation, validation I/O, repr), un `TextToAltoMock`
    démonstratif (TEXT→ALTO, critère explicite du plan), la délégation
    `BaseOCREngine.process → run`, et la cohérence ArtifactType/GTLevel.

- **Sprint 32 — Phase 0.1 : modèle de données GT multi-niveaux.**
  Refonte de `picarones/core/corpus.py` :
  - Enum `GTLevel` (TEXT, ALTO, PAGE, ENTITIES, READING_ORDER)
  - Payloads typés `TextGT`, `AltoGT`, `PageGT`, `EntitiesGT`,
    `ReadingOrderGT` avec `source_path` traçable
  - Champ `Document.ground_truths: dict[GTLevel, GTPayload]` synchronisé
    automatiquement avec le champ historique `ground_truth: str`
    (rétrocompatibilité stricte — toute API publique inchangée)
  - Détection automatique au chargement des fichiers `.gt.alto.xml`,
    `.gt.page.xml`, `.gt.entities.json`, `.gt.reading_order.json` à
    côté de l'image
  - `Corpus.gt_level_coverage()` et `Corpus.available_gt_levels` pour
    interroger la couverture par niveau
  - Erreurs de parse dégradées en `logger.warning` (CLAUDE.md :
    pas de `except: pass`) — le document conserve les niveaux qui ont
    pu être chargés
  - +17 tests dans `tests/test_sprint32_multi_level_gt.py` couvrant
    rétrocompat, détection, couverture partielle, synchronisation
    bidirectionnelle, JSON cassé

### Tests

- 1478 → 1752 tests (+17 Sprint 32, +23 Sprint 33, +21 Sprint 34,
  +27 Sprint 35, +22 Sprint 36, +42 Sprint 37, +19 Sprint 38,
  +32 Sprint 39, +16 Sprint 40, +38 Sprint 41, +17 Sprint 42).
  Aucune régression. **Phase 0 close ; Étape 2 du plan d'évolution :
  inter-moteurs (A.II.1.c) et NER (A.II.1.a) livrés bout-en-bout
  calcul → runner → narratif → HTML ; calibration (A.II.1.b) couche
  de calcul + câblage runner livrés (Sprints 39+42), il manque la vue
  HTML reliability diagram et l'adaptation des engines pour exposer
  leurs confidences natives.**

---

## [1.1.x] — Sprints 23-30 — 2026-04

### Ajouté

- **Sprint 23** — intégrité anti-hallucination du moteur narratif :
  whitelist `{"95", "100"}` vidée, `confidence_level=95` propagé dans
  `CONFIDENCE_WARNING`, `cost_unit_pages=1000` propagé dans
  `PARETO_ALTERNATIVE`/`COST_OUTLIER`, paramètre `select_facts(..., type_order=...)`,
  test stabilité bootstrap (±0,5 pp inter-seeds), test E2E synthèse EN.
  Doc « Politique éditoriale » dans `docs/developer/narrative-engine.md`.
- **Sprint 24** — durcissement sécurité institutionnelle : mode public
  (`PICARONES_PUBLIC_MODE=1`), `PICARONES_BROWSE_ROOTS`, validation Pillow
  sur upload (CVE-2023-50447), rate limit + sémaphore concurrence,
  middleware CSP + en-têtes durcis, `SECURITY.md` à la racine.
- **Sprint 25** — refactor frontend en Jinja2 : `_HTML_TEMPLATE` (3000 L)
  → 8 partials `picarones/web/templates/` + `static/web-app.js`. CSP
  durcie en partie (script externalisé).
- **Sprint 26** — persistance jobs SQLite : `picarones/core/jobs.py`,
  `JobStore` thread-safe (WAL), `BenchmarkJob` persiste chaque event,
  endpoint SSE supporte `Last-Event-ID`, jobs orphelins marqués
  `interrupted` au boot, fallback DB sur `/api/benchmark/{id}/status`.
- **Sprint 27** — snapshots de reproductibilité dans le rapport HTML :
  `picarones/report/snapshot.py` embarque YAML brut de `pricing.yaml`,
  glossaire trié, profil de normalisation, version Picarones+Python+
  commit+deps figées.
- **Sprint 28** — UX : save/load config (`/api/config/save|load`),
  comparaison de runs (`picarones compare`, exit code 2 si régression),
  synthesis preview (`/api/benchmark/{id}/synthesis_preview`),
  `/api/history/regressions` qui surface l'infra Sprint 8.
- **Sprint 29** — registre déclaratif des détecteurs narratifs :
  `@register_detector(fact_type, priority, importance)` ;
  `DEFAULT_TYPE_ORDER` dérivé du registre. Ajouter un détecteur passe
  de 4 fichiers à 2.
- **Sprint 30** — polish/accessibilité/DX : `.pre-commit-config.yaml`
  avec ruff + check YAML/JSON/secrets, badges CER WCAG (icône + bordure
  pattern + `aria-label`), `i18n.py` thread-safe avec `lru_cache`,
  `_safe_version` log la stacktrace en DEBUG, backport CHANGELOG
  Sprints 10-22, mise à jour SPECS pour narrative/Pareto/glossaire.

### Tests

- 1242 → 1426 tests (+184 sur les Sprints 23-30).

---

## [1.0.x] — Sprints 10-22 — 2025-04 → 2026-03

### Ajouté

- **Sprint 10** — distribution erreurs par ligne (Gini, percentiles)
  dans `picarones/core/line_metrics.py`, détection hallucinations VLM
  dans `picarones/core/hallucination.py` (anchor score, length ratio).
- **Sprint 11** — internationalisation FR/EN, profils de normalisation
  anglais (`early_modern_english`, `medieval_english`, `secretary_hand`).
- **Sprint 12** — upload ZIP depuis le navigateur, filtrage fichiers
  macOS `._*`, profils d'exclusion de caractères (`sans_ponctuation`,
  `sans_apostrophes`), sélecteur dynamique de modèles via
  `/api/models/{provider}`.
- **Sprint 13** — nettoyage `pyproject.toml`, parallélisation runner
  (ThreadPool/ProcessPool selon `execution_mode`), timeout par doc,
  résultats partiels NDJSON, validation statistique Wilcoxon.
- **Sprint 14** — filtrage robuste des moteurs côté CLI/web, validation
  corpus avant lancement.
- **Sprint 15** — fix bug pipeline OCR+LLM sortie vide : `mistral_adapter`
  normalise les `ContentChunk`, log `finish_reason` + tokens.
- **Sprint 16** — câblage de `line_metrics` et `hallucination` dans
  `runner` et l'agrégation `EngineReport` ; fondations du moteur
  narratif (`core/narrative/` avec `Fact`/`DetectorRegistry`) ;
  Pillow `getdata()` → `tobytes()` ; deux `except: pass` → warnings.
- **Sprint 17** — refactor du rapport monolithique : `generator.py`
  3690 → 617 lignes via Jinja2, 10 fichiers externes dans
  `picarones/report/templates/`, i18n migrée vers
  `report/i18n/{fr,en}.json`. +16 tests de non-régression.
- **Sprint 18** — test de Friedman multi-moteurs + Nemenyi post-hoc +
  Critical Difference Diagram (Demšar 2006) ; `core/statistics.py`
  étendu, fallback pur Python, scipy optionnel via extra `[stats]`.
  Détecteur narratif `STATISTICAL_TIE` activé. +41 tests.
- **Sprint 19** — moteur narratif complet : 9 détecteurs implémentés
  (global_leader_cer, significant_gap, stratum_winner/collapse,
  error_profile_outlier, llm_hallucination_flag, robustness_fragile,
  speed_winner, confidence_warning), arbitre, renderer YAML,
  `_narrative_summary.html`. Garde-fou anti-hallucination testé.
  +32 tests.
- **Sprint 20** — modélisation coût + vue Pareto : `core/pricing.py`,
  `data/pricing.yaml`, `compute_pareto_front` multi-objectifs,
  Chart.js Pareto avec axes coût/vitesse/carbone. Détecteurs
  `pareto_alternative` + `cost_outlier` activés. +28 tests.
- **Sprint 21** — glossaire contextuel (25 entrées bilingues) +
  panneau « Mode avancé » : choix de colonnes, filtres par strate,
  vue opt-in « score composite personnel » avec curseurs à 0 par défaut
  et formule visible. État persisté en URL. +21 tests.
- **Sprint 22** — études de cas (`docs/case-studies/`),
  `docs/user/reading-a-report.md`, trois guides développeur dans
  `docs/developer/`. Garde-fou « pas de fausses études prétendant
  être réelles ». +18 tests.

### Modifié

- `pyproject.toml` : extras `[stats]`, `[hf]`, mises à jour de
  `dev`/`web` pour `python-multipart`.
- `picarones/core/runner.py` : refactor pour gérer le `execution_mode`
  des moteurs (IO-bound vs CPU-bound).

### Corrigé

- `python-multipart` durablement présent dans `[dev]` et `[web]`
  (FastAPI vérifie l'import au décorateur `@app.post`).
- Tests Windows SQLite `test_history_empty_db` (gc.collect avant
  unlink).
- `test_search_language_filter` (HuggingFace) — assertion corrigée.

---

## [1.0.0] — Sprint 9 — 2025-03

### Ajouté
- `README.md` complet bilingue (français + anglais) avec badges CI, description des fonctionnalités, tableau des moteurs, variables d'environnement
- `INSTALL.md` — guide d'installation détaillé pour Linux (Ubuntu/Debian), macOS et Windows, incluant Tesseract, Pero OCR, Ollama, configuration des clés API, Docker
- `CHANGELOG.md` — historique des sprints 1 à 9
- `CONTRIBUTING.md` — guide pour contribuer : ajouter un moteur OCR, un adaptateur LLM, soumettre une PR
- `Makefile` — commandes `make install`, `make test`, `make demo`, `make serve`, `make build`, `make build-exe`, `make docker-build`, `make lint`, `make clean`
- `Dockerfile` — image Docker multi-étape basée sur Python 3.11-slim, Tesseract pré-installé, `CMD ["picarones", "serve", "--host", "0.0.0.0"]`
- `docker-compose.yml` — service Picarones + service Ollama optionnel (profil `ollama`)
- `.github/workflows/ci.yml` — pipeline GitHub Actions : tests sur Python 3.11/3.12, Linux/macOS/Windows, rapport de couverture
- `picarones.spec` — configuration PyInstaller pour générer des exécutables standalone (Linux, macOS, Windows)
- `picarones/__main__.py` — permet l'exécution via `python -m picarones`
- Version bumped à `1.0.0` dans `pyproject.toml` et `__init__.py`
- Extras PyPI `[llm]`, `[ocr-cloud]`, `[all]` dans `pyproject.toml`
- Tests Sprint 9 : `tests/test_sprint9_packaging.py` (30 tests)

### Modifié
- `pyproject.toml` : version 1.0.0, nouveaux extras, classifiers mis à jour, URLs projet ajoutées

---

## [0.8.0] — Sprint 8 — 2025-03

### Ajouté
- **eScriptorium** (`picarones/importers/escriptorium.py`)
  - `EScriptoriumClient` : connexion par token API, listing projets/documents/pages, gestion de la pagination
  - `import_document()` : import d'un document avec ses transcriptions comme corpus Picarones
  - `export_benchmark_as_layer()` : export des résultats benchmark comme couche OCR nommée dans eScriptorium
  - `connect_escriptorium()` : connexion avec validation automatique
- **Gallica API** (`picarones/importers/gallica.py`)
  - `GallicaClient` : recherche SRU par cote/titre/auteur/date/langue/type
  - Récupération OCR Gallica texte brut (`f{n}.texteBrut`)
  - Import IIIF Gallica avec enrichissement OCR comme vérité terrain de référence
  - Métadonnées OAI-PMH (`/services/OAIRecord`)
  - `search_gallica()`, `import_gallica_document()` — fonctions de commodité
- **Suivi longitudinal** (`picarones/core/history.py`)
  - `BenchmarkHistory` : base SQLite horodatée par run, moteur, corpus, CER/WER
  - `record()` depuis `BenchmarkResult`, `record_single()` pour imports manuels
  - `query()` avec filtres engine/corpus/since/limit
  - `get_cer_curve()` : données prêtes pour Chart.js
  - `detect_regression()` / `detect_all_regressions()` : seuil configurable en points de CER
  - `export_json()` — export complet de l'historique
  - `generate_demo_history()` : 8 runs fictifs avec régression simulée au run 5
- **Analyse de robustesse** (`picarones/core/robustness.py`)
  - 5 types de dégradation : bruit gaussien, flou, rotation, réduction de résolution, binarisation
  - `degrade_image_bytes()` : Pillow (préféré) ou fallback pur Python
  - `RobustnessAnalyzer.analyze()` : CER par niveau, seuil critique automatique
  - `DegradationCurve`, `RobustnessReport`, `_build_summary()`
  - `generate_demo_robustness_report()` : rapport fictif réaliste sans moteur réel
- **CLI Sprint 8**
  - `picarones history` : historique avec filtres, détection de régression, export JSON, mode `--demo`
  - `picarones robustness` : analyse de robustesse, barres ASCII, export JSON, mode `--demo`
  - `picarones demo --with-history --with-robustness` : démonstration intégrée
- `picarones/importers/__init__.py` mis à jour pour exporter les nouveaux importeurs

### Tests
- `tests/test_sprint8_escriptorium_gallica.py` : 74 tests (eScriptorium, Gallica, CLI)
- `tests/test_sprint8_longitudinal_robustness.py` : 86 tests (history, robustesse, CLI)
- **Total** : 743 tests (anciennement 583)

---

## [0.7.0] — Sprint 7 — 2025-02

### Ajouté
- **Rapport HTML v2**
  - Intervalles de confiance Bootstrap à 95% (`bootstrap_ci()`)
  - Tests de Wilcoxon et matrices de tests par paires (`wilcoxon_test()`, `pairwise_stats()`)
  - Courbes de fiabilité (CER cumulatif par percentile de qualité)
  - Diagrammes de Venn des erreurs communes/exclusives entre concurrents (2 et 3 ensembles)
  - Clustering des patterns d'erreurs (k-means simplifié sur n-grammes d'erreur)
  - Matrice de corrélation entre métriques (Pearson)
  - Score de difficulté intrinsèque par document (`compute_difficulty()`, `compute_all_difficulties()`)
  - Scatter plots interactifs qualité image vs CER, colorés par type de script
  - Heatmaps de confusion unicode améliorées
- `picarones/core/statistics.py` : module dédié aux tests statistiques
- `picarones/core/difficulty.py` : score de difficulté intrinsèque

### Tests
- `tests/test_sprint7_advanced_report.py` : 100 tests (bootstrap, Wilcoxon, Venn, clustering, difficulté)
- **Total** : 583 tests (anciennement 483)

---

## [0.6.0] — Sprint 6 — 2025-02

### Ajouté
- **Interface web FastAPI** (`picarones/web/app.py`)
  - Endpoints REST pour lancer des benchmarks, consulter les résultats, lister les moteurs
  - Streaming des logs en temps réel (Server-Sent Events)
  - `picarones serve` — lancement du serveur uvicorn
- **Import HuggingFace Datasets** (`picarones/importers/huggingface.py`)
  - Recherche, filtrage et import partiel de datasets OCR/HTR
  - Datasets patrimoniaux pré-référencés : IAM, RIMES, READ-BAD, Esposalles…
  - Cache local avec gestion des versions
- **Import HTR-United** (`picarones/importers/htr_united.py`)
  - Listing et import depuis le catalogue HTR-United
  - Lecture des métadonnées : langue, script, institution, époque
- **Adaptateurs Ollama** (`picarones/llm/ollama_adapter.py`)
  - Support de Llama 3, Gemma, Phi et tout modèle Ollama local
  - Mode texte seul (LLMs non multimodaux)
- **Profils de normalisation pré-configurés**
  - Français médiéval, Français moderne, Latin médiéval, Imprimés anciens
  - Profil personnalisé exportable/importable

### Tests
- `tests/test_sprint6_web_interface.py` : 90 tests
- **Total** : 483 tests (anciennement 393)

---

## [0.5.0] — Sprint 5 — 2025-02

### Ajouté
- **Matrice de confusion unicode** (`picarones/core/confusion.py`)
  - `build_confusion_matrix()`, `aggregate_confusion_matrices()`
  - Affichage compact trié par fréquence d'erreur
- **Scores ligatures et diacritiques** (`picarones/core/char_scores.py`)
  - `compute_ligature_score()` : fi, fl, ff, ffi, ffl, st, ct, œ, æ, ꝑ, ꝓ…
  - `compute_diacritic_score()` : accents, cédilles, trémas, diacritiques combinants
- **Taxonomie des erreurs en 10 classes** (`picarones/core/taxonomy.py`)
  - Confusion visuelle, erreur diacritique, casse, ligature, abréviation, hapax, segmentation, hors-vocabulaire, lacune, sur-normalisation LLM
- **Analyse structurelle** (`picarones/core/structure.py`)
  - Score d'ordre de lecture, taux de segmentation des lignes, conservation des sauts de paragraphe
- **Métriques de qualité image** (`picarones/core/image_quality.py`)
  - Netteté (Laplacien), niveau de bruit, contraste (Michelson), détection rotation résiduelle
  - Corrélations image ↔ CER
- Intégration de toutes ces métriques dans le rapport HTML (vue Analyse, vue Caractères)
- Scatter plots qualité image vs CER

### Tests
- `tests/test_sprint5_advanced_metrics.py` : 100 tests
- **Total** : 393 tests (anciennement 293)

---

## [0.4.0] — Sprint 4 — 2025-01

### Ajouté
- **Adaptateurs APIs cloud OCR**
  - Mistral OCR (`picarones/engines/mistral_ocr.py`) — Mistral OCR 3, multimodal
  - Google Vision (`picarones/engines/google_vision.py`) — Document AI
  - Azure Document Intelligence (`picarones/engines/azure_doc_intel.py`)
- **Import IIIF v2/v3** (`picarones/importers/iiif.py`)
  - Sélecteur de pages (`"1-10"`, `"1,3,5"`, `"all"`)
  - Téléchargement images et extraction des annotations de transcription si disponibles
  - Compatibilité : Gallica, Bodleian, British Library, BSB, e-codices, Europeana
  - `picarones import iiif <url>` — commande CLI
- **Normalisation unicode** (`picarones/core/normalization.py`)
  - NFC, caseless, diplomatique (tables ſ=s, u=v, i=j, æ=ae, œ=oe…)
  - Profils configurables via YAML
  - CER diplomatique dans les métriques

### Tests
- `tests/test_sprint4_normalization_iiif.py` : 100 tests
- **Total** : 293 tests (anciennement 193)

---

## [0.3.0] — Sprint 3 — 2025-01

### Ajouté
- **Pipelines OCR+LLM** (`picarones/pipelines/base.py`)
  - Mode 1 — Post-correction texte brut (LLM reçoit la sortie OCR)
  - Mode 2 — Post-correction avec image (LLM reçoit image + OCR)
  - Mode 3 — Zero-shot LLM (LLM reçoit uniquement l'image)
  - Chaînes composables multi-étapes
- **Adaptateurs LLM**
  - OpenAI (`picarones/llm/openai_adapter.py`) — GPT-4o, GPT-4o mini
  - Anthropic (`picarones/llm/anthropic_adapter.py`) — Claude Sonnet, Haiku
  - Mistral (`picarones/llm/mistral_adapter.py`) — Mistral Large, Pixtral
- **Détection de sur-normalisation LLM** (`picarones/pipelines/over_normalization.py`)
  - Mesure du taux de modification sur des passages déjà corrects
  - Classe 10 dans la taxonomie des erreurs
- **Bibliothèque de prompts**
  - Prompts pour manuscrits médiévaux, imprimés anciens, latin
  - Versionning des prompts dans les métadonnées du rapport
- Vue spécifique OCR+LLM dans le rapport : diff triple GT / OCR brut / après correction

### Tests
- `tests/test_sprint3_llm_pipelines.py` : 100 tests
- **Total** : 193 tests (anciennement 93)

---

## [0.2.0] — Sprint 2 — 2025-01

### Ajouté
- **Rapport HTML interactif** (`picarones/report/generator.py`)
  - Fichier HTML auto-contenu, lisible hors-ligne
  - Tableau de classement des concurrents (CER, WER, scores), tri par colonne
  - Graphique radar (spider chart) : CER / WER / Précision diacritiques / Ligatures
  - Vue Galerie : toutes les images avec badges CER colorés (vert→rouge), filtres
  - Vue Document : image zoomable + diff coloré façon GitHub, scroll synchronisé N-way
  - Vue Analyse : histogrammes de distribution CER, scatter plots
  - Recommandation automatique de moteur
  - Exports CSV, JSON, ALTO XML depuis le rapport
- **Diff coloré** (`picarones/report/diff_utils.py`)
  - Diff au niveau caractère et mot
  - Insertions (vert), suppressions (rouge), substitutions (orange)
  - Bascule diplomatique / normalisé
- `picarones demo` — rapport de démonstration avec données fictives réalistes
- `picarones report --results results.json` — génère le HTML depuis un JSON existant
- `picarones/fixtures.py` — générateur de benchmarks fictifs (12 textes médiévaux, 4 concurrents)

### Tests
- `tests/test_report.py`, `tests/test_diff_utils.py` : 93 tests
- **Total** : 93 tests (anciennement 20)

---

## [0.1.0] — Sprint 1 — 2025-01

### Ajouté
- **Structure complète du projet** Python avec `pyproject.toml`, `setup`, packaging
- **Adaptateur Tesseract 5** (`picarones/engines/tesseract.py`) via `pytesseract`
  - Configuration lang, PSM, DPI
  - Récupération de la version
- **Adaptateur Pero OCR** (`picarones/engines/pero_ocr.py`)
  - Chargement de modèle, traitement d'image
- **Interface abstraite** `BaseOCREngine` avec `process_image()`, `get_version()`, propriétés
- **Calcul CER et WER** (`picarones/core/metrics.py`) via `jiwer`
  - CER brut, NFC, caseless
  - WER, WER normalisé, MER, WIL
  - Longueurs de référence et hypothèse
- **Chargement de corpus** (`picarones/core/corpus.py`)
  - Dossier local : paires image / `.gt.txt`
  - Détection automatique des extensions image (jpg, png, tif, bmp…)
  - Classe `Corpus`, `Document`
- **Export JSON** (`picarones/core/results.py`)
  - `BenchmarkResult`, `EngineReport`, `DocumentResult`
  - `ranking()` : classement par CER moyen
  - `to_json()` avec horodatage et métadonnées
- **Orchestrateur benchmark** (`picarones/core/runner.py`)
  - Traitement séquentiel des documents par moteur
  - Barre de progression `tqdm`
  - Cache des sorties par hash SHA-256
- **CLI Click** (`picarones/cli.py`)
  - `picarones run` — benchmark complet
  - `picarones metrics` — CER/WER entre deux fichiers
  - `picarones engines` — liste des moteurs avec statut
  - `picarones info` — version et dépendances
  - `--fail-if-cer-above` pour intégration CI/CD

### Tests
- `tests/test_metrics.py`, `test_corpus.py`, `test_engines.py`, `test_results.py` : 20 tests
