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

- **Sprint 72 — A.I.1 chantier 1 : vue « Worst lines globale »
  (clôture A.I.1).**  Suite directe Sprint 71 : la roadmap A.I.1
  comporte deux chantiers — la métrique rare-token recall (livrée)
  et la vue HTML qui expose les lignes individuelles les plus mal
  transcrites du corpus.  Ce sprint livre la vue.
  - Nouveau module `picarones/core/worst_lines.py` :
    - Dataclass ``WorstLineEntry(rank, cer, engine_name, doc_id,
      line_index, gt_line, hyp_line, script_type)``.
    - ``extract_worst_lines(benchmark, top_n=20, engine_filter,
      script_type_filter)`` collecte transversalement à tous les
      moteurs et documents, filtre par moteur et par strate
      (Sprint 45 ``doc_strata``), trie par CER décroissant, retourne
      les ``top_n`` premières avec rang 1-based.
    - Récupération des textes GT/hyp par re-split du
      ``DocumentResult.ground_truth`` / ``hypothesis`` à l'index de
      ligne (cf. limite : suppose un ``BenchmarkResult``
      non-compacté).
    - Lignes avec ``cer == 0.0`` ignorées (pas dans le worst).
  - Nouveau module `picarones/report/worst_lines_render.py` :
    - ``build_worst_lines_table_html(entries, labels)`` : tableau
      HTML server-side avec colonnes Rang / CER (cellule colorée
      gradient jaune→rouge) / Moteur / Document / Ligne # /
      [Strate] / Diff GT→OCR.  Colonne strate **adaptive**
      (omise si aucune entry n'a de ``script_type``).
    - Diff caractère par caractère via
      ``diff_utils.compute_char_diff`` (réutilisation Sprint 5),
      rendu inline avec rouge clair barré pour suppressions et vert
      clair pour insertions.
    - Anti-injection systématique sur engine_name, doc_id, GT/hyp
      lines, labels i18n.
    - Retourne ``""`` si la liste est vide (rapport adaptatif).
  - +25 tests dans `test_sprint72_worst_lines.py` :
    extraction (top_n, tri par CER décroissant, rang 1-based,
    top_n=0, lignes CER=0 ignorées) ; filtres (par moteur, par
    strate, valeurs inconnues) ; cas limites (pas de line_metrics,
    benchmark vide, sans doc_strata, hyp plus courte que GT) ;
    rendu (tableau, colonnes attendues, strate adaptive, cellule
    CER colorée, diff rendu, % affiché) ; anti-injection
    (engine_name, doc_id, GT line, label i18n).
  - **Verrou levé** : un chercheur qui voit *« 5 % de mes lignes
    ont un CER > 0,42 »* dans le rapport peut désormais voir
    **quelles** lignes — diff inline, document parent, ligne #,
    moteur — pour comprendre ce qui casse précisément.

- **Sprint 71 — A.I.1 chantier 2 : rare-token recall (couche
  de calcul, démarrage de la résolution des critiques
  structurelles A.I).**  Premier sprint du chantier A.I qui
  s'attaque à la critique « la granularité ne s'arrête plus à la
  page ».  Pour répondre à *« mon OCR a 5 % de CER, mais
  préserve-t-il les noms propres rares qui m'intéressent pour
  l'indexation prosopographique ? »*, le module mesure le **rappel
  sur les tokens rares** d'un corpus (hapax + dis legomena,
  défaut ``max_freq=2``).
  - Nouveau module `picarones/core/rare_tokens.py` :
    - ``tokenize(text)`` Unicode-aware : préserve les
      contractions (``L'an``, ``d’une``), composés
      (``peut-être``, ``c'est-à-dire``), apostrophe typographique
      ``’`` (U+2019).
    - ``frequency_distribution(documents, case_sensitive=False)``
      : ``{token: count}`` corpus-wide.
    - ``extract_rare_tokens(documents, max_freq=2)`` retourne le
      ``frozenset`` des tokens dont la fréquence corpus-wide est
      ``≤ max_freq``.  ``max_freq < 1`` → ``ValueError``.
    - ``compute_rare_token_recall(reference, hypothesis,
      rare_tokens)`` retourne ``{n_rare_tokens_in_reference,
      n_rare_tokens_recalled, recall, missed_tokens}``.
      Alignement **bag-of-tokens avec multiplicité** : un token
      rare présent 2× en GT et 1× en hyp donne recall = 0,5 (pas
      1,0).  ``missed_tokens`` liste les manqués avec
      multiplicité.
    - ``rare_token_recall`` raccourci.
  - **Pas d'enregistrement dans le registre typé Sprint 34** : la
    métrique exige un **3ᵉ argument** (le set des tokens rares,
    calculé corpus-wide en amont).  L'utilisateur appelle
    explicitement la fonction avec son set — pas de magie globale.
  - +28 tests dans `test_sprint71_rare_tokens.py` :
    tokenisation (8 cas — contractions ASCII et typographiques,
    composés, diacritiques, ponctuation, nombres, vide),
    frequency_distribution (4 cas — single/multi/casse), extraction
    (4 cas — hapax, hapax+dis legomena, max_freq invalide, vide),
    recall (10 cas — full/partiel/zéro, multiplicité, GT sans
    rare, hyp vide, None, casse), raccourci, et **2 cas réalistes
    « registre d'état civil »** dont un test de propriété qui
    démontre la conjecture du plan : un OCR qui rate les noms
    propres a un rare-token recall plus dégradé qu'un OCR qui
    rate un mot fréquent (« le »), même si leur CER caractère est
    similaire.
  - **Verrou levé** : un bench BnF qui veut savoir « ce moteur
    préserve-t-il bien les noms de famille de mes registres ? »
    a maintenant la métrique adaptée.  La vue HTML « Worst lines
    + tokens rares manqués » suit Sprint 72 (chantier 1 d'A.I.1).

- **Sprint 70 — CLI pour piloter les pipelines composées sans
  Python (axe B, suite Sprints 63-69).**  Permet de spécifier une
  pipeline ou une comparaison de N pipelines dans un fichier
  **YAML déclaratif** et de les exécuter via la CLI Picarones, sans
  écrire une ligne de Python.  Utile pour la reproductibilité
  (specs versionnées en git) et pour les non-développeurs.
  - Nouveau module `picarones/core/pipeline_spec_loader.py` :
    - ``load_pipeline_spec_from_yaml(path)`` /
      ``load_pipeline_spec_from_dict(data)`` : parse un YAML et
      construit une ``PipelineSpec``.  Format :
      ``name`` + liste de ``steps``, chaque step ayant ``name``,
      ``module`` (dotted path Python vers la classe ``BaseModule``
      tierce), ``args`` (kwargs constructeur, optionnel),
      ``inputs_from`` (DAG branchant Sprint 66, optionnel).
    - ``load_comparison_specs_from_yaml(path)`` : parse un YAML
      contenant ``pipelines: [...]`` et retourne ``(specs,
      extras)`` où ``extras`` est le dict YAML brut (pour
      récupérer ``rankings``, ``baseline``…).
    - Import dynamique via ``importlib.import_module`` ; la classe
      référencée doit hériter de ``BaseModule`` (validation
      stricte).
    - Exception dédiée ``PipelineSpecLoadError`` avec messages
      explicites pour 8 cas d'erreur (chemin invalide, module
      introuvable, classe absente, classe non-BaseModule,
      constructeur incompatible, ``args`` non dict, type
      d'artefact inconnu, champ requis absent).
    - **Aucun module métier n'est créé** : le YAML référence
      uniquement les classes que l'utilisateur a installées dans
      son environnement Python.  Picarones se contente de les
      importer et de les brancher.
  - Nouveau sous-groupe CLI `picarones pipeline` dans
    `picarones/cli.py` :
    - ``picarones pipeline run <spec.yaml> --corpus <dir>``
      [``--output-json``] [``--output-html``] [``--lang``] :
      exécute une pipeline composée sur un corpus.  Affiche le
      résumé par étape (succès, taux), exporte JSON brut et/ou
      HTML autonome (Sprint 67).
    - ``picarones pipeline compare <specs.yaml> --corpus <dir>``
      [``--output-html``] [``--baseline``] [``--lang``] : compare
      N pipelines sur le même corpus.  Affiche le ranking par
      CER, exporte le rapport comparatif HTML autonome (Sprint
      68) avec ``ranking_specs`` extraits du YAML
      (``rankings`` au top-level) ou par défaut CER seul.
  - +27 tests dans `test_sprint70_pipeline_cli.py` :
    ``_resolve_class`` (5 cas — valide, sans dot, module
    introuvable, classe absente, cible non-classe),
    ``load_pipeline_spec_from_dict`` (9 cas — valide minimal,
    avec args, name/steps/module manquants, args non dict, classe
    non BaseModule, constructeur invalide, inputs_from valide,
    inputs_from type inconnu), ``load_pipeline_spec_from_yaml``
    (3 cas — fichier introuvable, YAML invalide, round-trip
    valide), ``load_comparison_specs`` (2 cas), CLI ``pipeline
    run`` (2 cas — basic + avec output JSON+HTML), CLI ``pipeline
    compare`` (2 cas — basic + avec HTML et baseline), CLI help
    (3 cas — pipeline groupe listé, run et compare avec leurs
    options).
  - **Tous les modules utilisés sont des mocks** définis dans le
    fichier de test (``_CLIMockOCR``, ``_NotABaseModule``).
    Picarones n'expose volontairement aucun module métier.
  - **Verrou levé** : un workflow BnF type — décrire la pipeline
    dans ``my_pipeline.yaml``, versionner le fichier en git, lancer
    ``picarones pipeline run my_pipeline.yaml --corpus ./scans
    --output-html rapport.html`` — fonctionne sans qu'un
    ingénieur Python soit dans la boucle.

- **Sprint 69 — Documentation utilisateur « Écrire un module pour
  le banc d'essai de pipelines » (axe B, suite Sprints 63-68).**
  Premier guide pédagogique dédié à l'axe B : un chercheur ou
  ingénieur qui veut **brancher son propre module** dans Picarones
  (correcteur LLM, reconstructeur ALTO, classifieur d'entités,
  re-segmenteur…) trouve maintenant un guide complet bout-en-bout.
  - Nouveau document `docs/user/writing-a-pipeline-module.md` :
    - **TL;DR** avec un exemple `MyCorrector` minimal en 25 lignes.
    - Section **« Le contrat ``BaseModule`` »** : tableau des
      champs obligatoires (``input_types``, ``output_types``,
      ``execution_mode``, ``name``, ``process``) et liste des
      ``ArtifactType`` disponibles.
    - Section **« Exemples pédagogiques »** : trois mocks
      explicitement étiquetés « pédagogique » et marqués « à NE
      PAS copier en production » — correcteur LLM TEXT→TEXT,
      reconstructeur TEXT→ALTO, classifieur TEXT→ENTITIES.
    - Section **« Orchestrer une pipeline »** : mono-document
      (Sprint 63), corpus complet avec factory personnalisée
      (Sprint 64), comparaison de N pipelines (Sprint 65), DAG
      branchant via ``inputs_from`` (Sprint 66) — chaque
      sous-section avec snippet exécutable.
    - Section **« Générer un rapport HTML autonome »** : pipeline
      unique (Sprint 67) et comparaison (Sprint 68) avec snippets
      ``Path("rapport.html").write_text(...)``.
    - Section **« Bonnes pratiques »** : discipline des types,
      erreurs gracieuses, mesure du temps wall-clock, **pas de
      seuils éditoriaux dans votre module** (le chercheur juge,
      pas le module).
    - Section **« Anti-patterns »** : trois questions FAQ avec
      réponses explicites — « Pourquoi pas de correcteur LLM
      intégré ? », « Et si je veux juste tester un OCR seul ? »,
      « Mon module a besoin d'état mutable entre documents ? ».
    - **Tableau de référence rapide** des sprints axe B (32-34
      pour les fondations, 63-68 pour l'orchestration et le
      rapport).
  - +34 tests dans `test_sprint69_user_doc.py` :
    - **présence des 7 sections principales** (TL;DR, contrat,
      exemples, orchestration, rapport HTML, bonnes pratiques,
      anti-patterns) — anti-régression doc
    - **15 concepts API mentionnés** (``BaseModule``,
      ``ArtifactType``, ``input_types``, ``inputs_from``,
      ``RankingSpec``, ``compare_pipelines``,
      ``build_pipeline_comparison_report_html``, etc.)
    - **philosophie « banc d'essai pas atelier »** présente
      explicitement dans le doc, mention « aucun module métier »,
      exemples étiquetés « pédagogique »
    - **références aux 6 sprints axe B** (63-68) + au moins un
      sprint de la phase 0 (32-34)
    - **≥ 5 blocs de code Python** + imports valides depuis les
      vrais modules ``picarones.core.*`` et ``picarones.report.*``
  - **Verrou levé** : la barrière d'entrée pour un utilisateur
    tiers qui veut évaluer son module passe d'« il faut lire le
    code source des Sprints 63-68 pour comprendre comment ça
    marche » à « il y a un guide qui couvre le tout en une page,
    avec des snippets prêts à copier ».

- **Sprint 68 — Vue HTML de comparaison de N pipelines composées
  (axe B, suite Sprints 63-67).**  Suite directe Sprint 67 — la
  vue mono-pipeline est étendue avec un rendu **comparatif** entre
  N pipelines exécutées sur le même corpus (Sprint 65).  Pattern
  inchangé : server-side, pas de JS, anti-injection systématique.
  - Extension de ``picarones/report/pipeline_render.py`` :
    - ``RankingSpec(artifact_type, metric_name, higher_is_better=False,
      label=None)`` — dataclass légère qui décrit un classement à
      afficher.  ``display_label`` auto-généré (``"<at>.<metric>"``)
      ou explicite.
    - ``build_pipeline_ranking_table_html(comparison, ranking_spec)``
      — tableau rang × pipeline × valeur, classé selon
      ``ranking_by_final_metric`` (Sprint 65).  Cellule de rang
      colorée par gradient vert (1er) → rouge (dernier).  Pipelines
      sans valeur listés en queue avec tirets.  Vide si la
      comparaison ne contient aucune pipeline.
    - ``build_pipeline_gain_table_html(comparison, ranking_spec,
      baseline_pipeline)`` — tableau pipeline × {valeur, gain
      absolu, gain relatif} vs baseline.  Cellule de gain colorée
      en vert (favorable) ou rouge (défavorable) selon
      ``higher_is_better``.  Baseline marquée explicitement
      ``(référence)``.  Retourne ``""`` si la baseline est inconnue.
    - ``build_pipeline_comparison_summary_html(comparison)`` —
      encart résumé : corpus, n_docs, n_pipelines, durée totale,
      mini-résumé par pipeline (nom + ``n_succeeded/n_docs``).
    - ``build_pipeline_comparison_report_html(comparison,
      ranking_specs, baseline_pipeline, lang)`` — **document HTML
      autonome** (``<!doctype html>``) qui assemble le summary +
      les rankings + les gain tables.  Aucune auto-détection
      magique : si ``ranking_specs`` est vide, on n'affiche que
      le summary ; si ``baseline_pipeline`` est ``None``, pas de
      gain tables.  L'utilisateur déclare explicitement ce qu'il
      veut voir.
  - +14 clés i18n FR/EN (``pipeline_comparison_*``,
    ``pipeline_ranking_*``, ``pipeline_gain_*``,
    ``pipeline_baseline_marker``).  Les libellés
    ``pipeline_ranking_title`` et ``pipeline_gain_title`` sont des
    templates avec placeholders ``{label}`` et ``{baseline}``.
  - +26 tests dans
    `test_sprint68_pipeline_comparison_html.py` :
    ``RankingSpec`` (display_label auto/explicite, défaut
    ``higher_is_better``) ; ranking table (ordre ascendant/
    descendant, pipelines sans valeur en queue, cellule rang
    colorée, vide si comparison vide, label explicite dans titre) ;
    gain table (baseline marquée, valeurs absolues et relatives,
    cellules vert favorable / rouge défavorable selon
    ``higher_is_better``, baseline inconnue → vide) ; summary
    (corpus, comptes, mini-résumé par pipeline) ; document
    autonome (doctype, structure, lang FR/EN, rankings affichés
    si specs, pas de gain table sans baseline) ; anti-injection
    sur pipeline name / corpus / labels i18n ; complétude i18n
    sur les 14 nouvelles clés FR ET EN.
  - **Toujours pas de classification automatique** : on classe et
    on affiche les gains, mais on ne dit jamais « pipeline A est
    la meilleure ».  Le chercheur lit le ranking et décide selon
    ses critères.
  - **Verrou levé** : un chercheur peut désormais générer en une
    ligne le rapport HTML autonome d'une comparaison entre N
    pipelines :

    .. code-block:: python

       html = build_pipeline_comparison_report_html(
           comparison,
           ranking_specs=[
               RankingSpec(ArtifactType.TEXT, "cer", label="CER"),
               RankingSpec(ArtifactType.TEXT, "wer", label="WER"),
           ],
           baseline_pipeline="ocr_only",
       )
       Path("comparaison.html").write_text(html)

- **Sprint 67 — Vue HTML d'un benchmark de pipeline composée
  (axe B, suite Sprints 63-66).**  Pattern identique aux
  Sprints 41 (NER), 43 (calibration) et 62 (philologie) : rendu
  **server-side**, pas de JavaScript, déterministe, anti-injection
  systématique via ``html.escape``.
  - Nouveau module `picarones/report/pipeline_render.py` :
    - ``build_pipeline_summary_html(bench)`` — encart résumé
      global (pipeline, corpus, n_docs, n_pipelines_succeeded /
      failed avec cellule colorée par taux de succès, durée
      totale formatée).
    - ``build_pipeline_steps_table_html(bench)`` — tableau par
      étape avec colonnes : nom, ``n_succeeded`` / ``n_failed``,
      taux de succès (gradient rouge → vert), durée mean /
      median, métriques aux jonctions formatées
      (``<type>.<metric>: mean (n=N)``), error_breakdown
      catégorisé (``raised_exception`` / ``missing_input`` /
      ``missing_output`` / ``pipeline_aborted`` / ``other``).
      Adaptive : retourne ``""`` si aucune étape n'a été agrégée.
    - ``build_pipeline_report_html(bench, lang)`` — **document
      HTML autonome** (``<!doctype html>`` + ``<html>`` +
      ``<head>`` avec styles CSS inline + ``<body>``) que
      l'utilisateur peut écrire directement sur disque, **sans
      dépendre du générateur OCR historique** (rapport pipeline
      distinct du rapport ``BenchmarkResult``).  Helper
      ``_format_duration`` adaptatif (ms / s / mm:ss).
  - **Vue distincte du rapport OCR** : le rapport HTML existant
    (``ReportGenerator``) attend un ``BenchmarkResult`` (axe A) ;
    pour les pipelines composées on utilise
    ``PipelineBenchmarkResult`` (axe B).  Plutôt que de fusionner
    les deux, on livre un rapport autonome à part — clarté
    architecturale et pas de couplage.
  - +18 clés i18n FR/EN (``pipeline_report_*``,
    ``pipeline_summary_*``, ``pipeline_steps_*``,
    ``pipeline_*_label``).
  - +21 tests dans `test_sprint67_pipeline_html.py` :
    - summary : nom de pipeline / corpus / succeeded-failed
      affichés, durée formatée
    - steps table : noms, colonnes (8 attendues), métriques aux
      jonctions affichées (``text.cer 0.182 (n=10)``),
      error_breakdown affiché, vide sans agrégats, cellule de
      taux colorée
    - document autonome : doctype, structure html/head/body,
      styles inline, title contenant le pipeline name, attribut
      ``lang`` (FR + EN), summary et steps inclus
    - anti-injection : pipeline name / corpus name / step name /
      labels i18n contenant ``<script>`` tous correctement
      échappés
    - complétude i18n : 17 clés ``pipeline_*`` présentes en FR
      ET EN
  - **Pas de classification automatique** : le document affiche
    les chiffres bruts par étape ; aucun verdict « pipeline bonne
    ou mauvaise » imposé.
  - **Reporté Sprint 68** : rendu d'un
    ``PipelineComparisonResult`` (ranking + gain table entre N
    pipelines).
  - **Verrou levé** : un chercheur peut désormais générer un
    rapport HTML autonome après ``run_pipeline_benchmark`` —
    ``Path("rapport.html").write_text(build_pipeline_report_html(
    bench))`` — sans rien d'autre.

- **Sprint 66 — DAG branchant via ``inputs_from`` (axe B, suite
  Sprints 63-65).**  Les Sprints 63-65 traitaient des pipelines
  séquentielles : la sortie d'une étape alimente automatiquement
  la suivante via le bag d'artefacts (la dernière version d'un
  type écrase la précédente).  Ce sprint permet de **désigner
  explicitement la source d'un artefact** quand plusieurs étapes
  produisent le même type, débloquant des scénarios fork/merge
  dans une même pipeline (ex. comparer deux corrections LLM en
  parallèle d'un même OCR sans devoir basculer sur deux pipelines
  distinctes via Sprint 65).
  - ``PipelineStep.inputs_from: dict[ArtifactType, str]`` (vide
    par défaut) — pour chaque type d'entrée, l'étape peut désigner
    le nom de l'étape source dont consommer l'artefact.  La chaîne
    spéciale ``"__initial__"`` désigne les entrées initiales
    (utile pour les pipelines démarrant par un type fourni en
    entrée).
  - **Bag versionné** dans ``PipelineRunner.run`` : on stocke
    désormais ``versioned[(type, source_step_name)] = artifact``
    et on maintient un index ``latest[type] = step_name``.  En
    l'absence d'``inputs_from``, le runner prend la version la
    plus récente — comportement Sprint 63 strictement préservé.
  - **Validation étendue** dans ``PipelineSpec.validate`` :
    détecte les références ``inputs_from`` vers une étape inconnue,
    une étape qui ne produit pas le type demandé, ou un type que
    le module ne consomme pas.  Tous les problèmes sont remontés
    avec un message explicite indiquant l'étape concernée et la
    référence litigieuse.
  - **Référence vers étape qui a échoué** : si ``inputs_from``
    pointe vers un step qui a levé une exception, l'étape en aval
    rapporte une erreur ``entrée manquante : <type>@<step>`` —
    le marqueur ``@step`` permet au lecteur de comprendre
    immédiatement que la dépendance pointait vers un step en
    échec, pas un type absent.
  - **Rétrocompat stricte** : sans ``inputs_from``, le
    comportement Sprint 63 est intégralement préservé.  Les 42
    tests Sprints 63-65 passent sans modification.
  - +11 tests dans `test_sprint66_dag_branching.py` :
    - défaut ``inputs_from`` vide
    - validation : référence valide, ``"__initial__"``, étape
      inconnue, type non consommé
    - DAG fork explicite : 2 corrections en parallèle d'un même
      OCR avec métriques indépendantes
    - **fork vs chain divergent** : test propriété qui prouve que
      placer ``inputs_from={TEXT: "ocr"}`` change le résultat
      final (CER 0 en fork vs CER > 0 en chain) sur un même corpus
    - référence vers étape qui a échoué → erreur ``@step`` propre
    - rétrocompat sans ``inputs_from``
  - **Tous les modules utilisés sont des mocks** (``MockOCR``,
    ``TextFixer``, ``TextDoubler``, ``AlwaysFails``).  Picarones
    n'expose volontairement aucun module métier.
  - **Verrou levé** : un chercheur peut désormais composer une
    pipeline qui fork un même OCR vers plusieurs branches de
    correction et évaluer chacune indépendamment, dans une seule
    spec — sans devoir basculer sur ``compare_pipelines`` quand
    le besoin est de tracer le branchement dans un seul contexte
    d'exécution.

- **Sprint 65 — Comparaison de N pipelines composées sur le même
  corpus (axe B, suite Sprints 63-64).**  Réponse à la question
  typique BnF : « OCR seul vs OCR+correcteur A vs OCR+correcteur
  B : laquelle est la meilleure sur mon corpus, et de combien ? ».
  Philosophie inchangée — banc d'essai, pas atelier de production.
  - Nouveau module `picarones/core/pipeline_comparison.py` :
    - ``compare_pipelines(specs, corpus, factories=None)`` exécute
      séquentiellement N ``PipelineSpec`` sur le **même** corpus
      (même GT, comparaison apple-to-apple).  Garde-fou : noms
      uniques exigés (sinon ``ValueError`` explicite).
    - ``factories`` optionnel : dict ``{pipeline_name:
      InitialInputsFactory}`` pour personnaliser les entrées
      initiales par pipeline (utile pour comparer une pipeline qui
      démarre par ``IMAGE`` et une qui démarre par ``TEXT``).
    - ``PipelineComparisonResult`` : conteneur avec
      ``per_pipeline: dict[name → PipelineBenchmarkResult]``,
      ``pipeline_names()`` qui préserve l'ordre d'insertion,
      et deux utilitaires comparatifs.
    - ``ranking_by_final_metric(artifact_type, metric_name,
      higher_is_better=False)`` : trie les pipelines par la valeur
      **finale** de la métrique demandée à la jonction
      ``artifact_type`` (récupère le ``mean`` de la dernière étape
      qui a produit ce type, cohérent avec
      ``PipelineResult.junction_metrics_for`` Sprint 63).  Les
      pipelines sans métrique vont en queue.
    - ``gain_table(artifact_type, metric_name, baseline_pipeline)``
      : pour chaque pipeline, ``{value, absolute, relative}``
      vs baseline.  ``relative`` à ``None`` si baseline = 0
      (évite division par zéro silencieuse).  ``KeyError`` si
      la baseline est inconnue.
  - Approche purement infrastructure : aucun module métier, on se
    contente de réutiliser ``run_pipeline_benchmark`` (Sprint 64)
    pour chaque spec et d'ajouter la couche comparative au-dessus.
  - +13 tests dans `test_sprint65_pipeline_comparison.py`
    (single/multi pipelines, ordre préservé, noms en double,
    corpus vide, ranking ascendant/descendant, pipelines sans
    métrique en queue, ``gain_table`` avec baseline inconnue,
    baseline = 0 → relative None, baseline=self → absolute=0,
    cas réaliste OCR fautif vs OCR+correcteur, factories par
    pipeline, dataclass).
  - **Tous les modules utilisés sont des mocks** (``MockOCR``,
    ``TextFixer``, ``AlwaysFails``).  Picarones n'expose
    volontairement aucun module métier.
  - **Verrou levé** : un chercheur peut désormais comparer N
    pipelines tierces sur le **même** corpus en une commande,
    obtenir le ranking selon la métrique d'intérêt et la table
    de gain vs baseline.  Vue HTML dédiée et tests statistiques
    inter-pipelines arrivent dans les sprints suivants.

- **Sprint 64 — Orchestration corpus-wide d'une pipeline composée
  (axe B, suite directe Sprint 63).**  Le ``PipelineRunner`` du
  Sprint 63 exécute une pipeline sur un seul document ; ce sprint
  fournit l'orchestration sur un **corpus complet** et
  l'agrégation des résultats par étape.  Toujours dans la
  philosophie « banc d'essai, pas atelier de production » : aucun
  module métier n'est ajouté côté Picarones, l'utilisateur amène
  ses propres ``BaseModule`` (Sprint 33).
  - Nouveau module `picarones/core/pipeline_benchmark.py` :
    - ``InitialInputsFactory = Callable[[Document], dict[
      ArtifactType, Any]]`` : type pour la fonction qui produit
      les artefacts initiaux par document.
    - ``default_initial_inputs(doc)`` : factory par défaut qui
      retourne ``{IMAGE: doc.image_path}`` (cas standard d'une
      pipeline qui démarre par un OCR).  L'utilisateur peut
      fournir une factory personnalisée pour brancher d'autres
      sources (par exemple un ``ALTO`` pré-existant).
    - ``StepAggregate(step_name, n_docs, n_succeeded, n_failed,
      duration_seconds_total/mean/median, failing_doc_ids,
      junction_metrics, error_breakdown)`` : agrégat d'une étape
      sur tout le corpus.  Les métriques aux jonctions sont
      agrégées par type d'artefact, avec ``mean`` / ``median`` /
      ``n`` par métrique numérique (les non-numériques sont
      ignorées dans l'agrégat global mais restent visibles par
      doc).  ``error_breakdown`` catégorise les erreurs en
      ``missing_input`` / ``raised_exception`` /
      ``missing_output`` / ``pipeline_aborted`` / ``other`` via
      heuristique stable sur les messages produits par
      ``pipeline_runner._run_step``.
    - ``PipelineBenchmarkResult(pipeline_name, corpus_name,
      n_docs, per_doc_results, per_step_aggregates,
      total_duration_seconds)`` : conteneur global avec
      ``n_pipelines_succeeded`` / ``n_pipelines_failed`` calculés
      à la volée et ``aggregate_for_step(name)`` pour récupérer
      l'agrégat par nom.
    - ``run_pipeline_benchmark(spec, corpus,
      initial_inputs_factory)`` : itère séquentiellement sur les
      documents, appelle ``PipelineRunner.run`` sur chacun,
      capture gracieusement les erreurs de la factory, agrège
      par étape via ``_aggregate_step``.  Une spec invalide
      propage l'erreur à tous les documents (chacun a un
      ``PipelineResult`` avec ``error`` non vide et aucune étape
      exécutée).
  - **Périmètre Sprint 64** : séquentiel inter-documents.
    Comparaison de N pipelines sur le même corpus (Sprint 65),
    DAG branchant (Sprint 66), vue HTML pipelines (Sprint 67),
    parallélisation reportée à arbitrer.
  - +13 tests dans `test_sprint64_pipeline_benchmark.py` :
    factory par défaut, corpus vide, 1 doc OK, métriques agrégées
    sur 3 docs (CER mean/median/n), mix succès/échecs (1 doc
    crash → comptes corrects + failing_doc_ids + error_breakdown
    catégorisé en ``raised_exception``), 2 étapes avec rebond
    propre (étape 1 plante → étape 2 reçoit ``missing_input``
    avec le bon breakdown), spec invalide → tous les docs en
    pipeline_aborted, factory personnalisée, factory qui lève sur
    un doc → autres continuent, dataclasses (success_rate,
    aggregate_for_step retourne None pour nom inconnu).
  - **Tous les modules utilisés sont des mocks définis dans le
    test** (``MockOCR``, ``MockCrasherSometimes``,
    ``MockTextRewriter``).  Picarones n'expose volontairement
    aucun module métier.
  - **Verrou levé** : un utilisateur peut maintenant lancer une
    pipeline composée tierce sur **tout son corpus** en une
    commande, obtenir l'agrégat par étape (durée mean/median,
    métriques mean/median, taux d'erreur par catégorie) et les
    résultats détaillés par document.  La comparaison de
    plusieurs pipelines sur le même corpus arrive Sprint 65, la
    vue HTML dédiée Sprint 67.

- **Sprint 63 — Banc d'essai de pipelines composées : runner +
  évaluation aux jonctions (démarrage axe B du plan 2026).**
  Picarones est et reste un **banc d'essai**, pas un atelier de
  production : ce sprint livre l'infrastructure qui permet
  d'**évaluer des pipelines composées de modules tiers** que
  l'utilisateur amène (ses propres ``BaseModule`` du Sprint 33),
  **sans qu'aucun module métier ne soit fourni par Picarones**
  (pas de reconstructeur ALTO, pas de correcteur LLM, pas de
  re-segmenteur).
  - Nouveau module `picarones/core/pipeline_runner.py` :
    - ``PipelineStep(name, module)`` : une étape lit ses
      ``input_types`` / ``output_types`` directement depuis le
      ``BaseModule`` fourni par l'utilisateur.
    - ``PipelineSpec(name, steps)`` : DAG séquentiel de
      ``PipelineStep`` avec validation statique des types
      (``validate(initial_inputs)`` retourne la liste des
      problèmes ; ``is_valid`` raccourci booléen).
    - ``StepResult(step_name, duration_seconds, output_types,
      junction_metrics, error)`` : résultat d'une étape avec
      durée chronométrée, types effectivement produits, métriques
      aux jonctions et erreur éventuelle.
    - ``PipelineResult(pipeline_name, doc_id, steps,
      total_duration_seconds, error)`` : résultat complet pour un
      document, avec ``succeeded``, ``failing_steps``, et
      ``junction_metrics_for(artifact_type)`` qui retourne les
      métriques de la **dernière étape réussie** ayant produit le
      type demandé.
    - ``PipelineRunner.run(spec, document, initial_inputs)`` :
      exécute la pipeline sur **un seul document**.  À chaque
      étape : valide les entrées disponibles, exécute le module
      avec chronométrage wall-clock, capture gracieusement les
      exceptions (``RuntimeError``, etc.), valide que les sorties
      déclarées sont effectivement produites, met à jour le bag
      d'artefacts disponibles, et **évalue automatiquement chaque
      type produit contre la GT du même niveau** (Sprint 32) via
      ``compute_at_junction`` (Sprint 34) — sélectionnant les
      métriques pertinentes selon les types.
  - **Eager-load** des modules de métriques au top du
    ``pipeline_runner.py`` (``builtin_metrics``, les six modules
    philologiques, NER, reading_order, readability) pour garantir
    que le registre typé soit peuplé avant l'évaluation aux
    jonctions — sans ça, le runner trouverait un registre vide.
  - **Périmètre Sprint 63** : runner séquentiel mono-document.
    DAG branchant, parallélisation, agrégation corpus-wide et
    vue HTML dédiée aux pipelines sont reportés à des sprints
    dédiés.
  - +16 tests dans `test_sprint63_pipeline_runner.py` :
    validation de spec (vide, chaînée, manque d'entrée),
    exécution 1 étape (parfait + imparfait), exécution 2 étapes
    avec évaluation à chaque jonction et CER qui baisse après
    correction par le rewriter, erreurs gracieuses (module qui
    lève → RuntimeError capturé sans arrêter la chaîne ; module
    silencieux qui ne produit pas la sortie déclarée → erreur
    explicite ; spec invalide → erreur en amont, aucune étape
    exécutée), pas de GT → pas de métriques sans erreur, mesure
    du temps par étape, dataclasses (``StepResult`` /
    ``PipelineResult.succeeded`` / ``failing_steps`` /
    ``junction_metrics_for`` qui ignore les étapes en erreur).
  - **Tous les modules utilisés dans les tests sont des mocks
    définis dans le fichier de test** (``MockOCR``,
    ``MockTextRewriter``, ``MockCrasher``, ``MockSilentDropper``)
    — Picarones n'expose volontairement aucun module métier.
  - **Verrou levé** : l'utilisateur peut désormais brancher ses
    propres modules tiers (un correcteur LLM, un reconstructeur
    ALTO, un re-segmenteur, un classifieur d'entités), composer
    une pipeline et obtenir automatiquement les métriques à
    chaque étape contre la GT correspondante.  L'orchestration
    corpus-wide et la vue HTML dédiée arrivent dans les sprints
    suivants de l'axe B.

- **Sprint 62 — Vue HTML « Profil philologique » (clôture du
  câblage philologique bout-en-bout).**  Suite directe Sprint 61
  (câblage backend) — produit le bloc HTML qui remonte les six
  modules philologiques (Sprints 55-60) dans le rapport.  Pattern
  identique aux Sprints 41 (NER) et 43 (calibration) : rendu
  server-side, pas de JavaScript, déterministe.
  - Nouveau module `picarones/report/philological_render.py` :
    - 6 fonctions de rendu de section (une par module) :
      ``build_unicode_blocks_section``,
      ``build_abbreviations_section``, ``build_mufi_section``,
      ``build_early_modern_section``,
      ``build_modern_archives_section``,
      ``build_roman_numerals_section``.
    - Agrégateur ``build_philological_profile_html`` qui assemble
      les sections non vides en un bloc unique avec titre et note
      d'usage explicite (« L'outil ne classifie pas la convention
      adoptée par chaque moteur — c'est au chercheur de lire les
      chiffres et de conclure selon ses critères éditoriaux »).
    - **Adaptive masking complet** : chaque section n'apparaît que
      si au moins un moteur a du signal pour son module ; si aucun
      module n'a de signal, l'agrégateur retourne ``""`` et le
      bloc HTML complet est omis.
    - Cellules colorées par gradient rouge → jaune → vert
      proportionnel au score (identique au pattern Sprint 41
      ``ner_render``) ; pour le statut ``lost`` des numéraux
      romains, la coloration est inversée (un haut taux de perte
      est mauvais).
    - Affichage des effectifs ``n=…`` à côté de chaque score pour
      donner au chercheur le contexte (un score de 100 % sur n=1
      n'a pas la même valeur qu'un 80 % sur n=500).
  - Câblage dans ``ReportGenerator.generate`` : appel de
    ``build_philological_profile_html`` après les blocs
    NER/calibration/inter-moteurs/stratification, passage au
    template via la variable ``philological_profile_html``.
  - Câblage dans ``view_analyses.html`` : un nouveau
    ``chart-card`` pleine largeur conditionné au contenu
    (``{% if philological_profile_html %}``).
  - Anti-injection HTML systématique : tous les noms de moteurs,
    catégories, statuts, libellés i18n passent par
    ``html.escape`` avant insertion (testé : ``<script>`` dans le
    nom du moteur correctement échappé).
  - **Aucune classification automatique** : le mot
    « diplomatique » / « modernisant » n'apparaît que dans la
    note explicative en bas de section, jamais comme étiquette
    accolée à un moteur.
  - +25 clés i18n FR/EN (``philo_profile_*``,
    ``philo_unicode_*``, ``philo_abbreviations_*``,
    ``philo_mufi_*``, ``philo_early_modern_*``,
    ``philo_modern_archives_*``, ``philo_roman_numerals_*``,
    ``philo_roman_status_*``).
  - +18 tests dans `test_sprint62_philological_html.py` (sections
    individuelles ×6, adaptive masking complet, anti-injection sur
    nom de moteur et libellé i18n, valeurs en %, code couleur,
    pas de classification imposée, complétude i18n FR/EN sur les
    25 clés).
  - **Verrou levé** : les six modules philologiques sont désormais
    livrés bout-en-bout (calcul Sprints 55-60 + backend Sprint 61
    + HTML Sprint 62).  Un benchmark sur n'importe quel fonds
    patrimonial européen produit automatiquement, sans
    configuration, un profil philologique lisible dans le rapport
    HTML — donné par catégorie/bloc/statut, sans verdict.

- **Sprint 61 — Câblage backend des métriques philologiques au
  runner (Sprints 55-60).**  Suite directe Sprints 55-60 — les six
  modules philologiques (unicode_blocks, abbreviations, mufi,
  early_modern, modern_archives, roman_numerals) sont désormais
  calculés automatiquement par le runner pour chaque document et
  agrégés par moteur, **sans aucune option à activer**.
  - Nouveau module `picarones/core/philological_runner.py` :
    - ``compute_philological_metrics(reference, hypothesis)``
      calcule les six modules et retourne un dict avec une clé par
      module ayant du **signal exploitable** dans la GT
      (``n_markers_reference > 0``, ``n_mufi_chars_reference > 0``,
      au moins un caractère hors Basic Latin pour unicode_blocks,
      etc.).  Retourne ``None`` si aucun module n'a de signal.
    - ``aggregate_philological_metrics(per_doc_list)`` agrège les
      compteurs bruts par module (somme), recalcule les scores
      globaux à partir des sommes (accuracy, coverage, strict,
      expansion, value, preservation), et préserve les structures
      ``per_block`` / ``per_abbreviation`` / ``per_char`` /
      ``per_category`` / ``per_status`` agrégées.
    - **Adaptive masking** : un module n'apparaît dans le résultat
      que si au moins un document a eu du signal pour lui — les
      rapports restent lisibles sur les corpus sans marqueur
      philologique pertinent (typique des fonds XXIᵉ propres).
  - Nouveaux champs sur ``DocumentResult.philological_metrics`` et
    ``EngineReport.aggregated_philological`` (``Optional[dict]``,
    ``None`` par défaut, sérialisés conditionnellement par
    ``as_dict``, libérés par ``compact``).
  - Câblage dans ``runner._compute_document_result`` : le calcul
    est inconditionnel (coût O(N) sur le texte, négligeable face à
    l'OCR) et l'erreur d'un module individuel ne propage pas — on
    omet le module et on logue un warning explicite (jamais
    ``except: pass`` selon les règles CLAUDE.md).
  - Câblage dans ``run_benchmark`` : agrégation par moteur
    appelée juste après les autres agrégations Sprint 5/10/40/42.
  - **Rétrocompat stricte** : aucun paramètre ajouté, aucun
    comportement existant modifié ; un benchmark sans signal
    philologique voit ses ``philological_metrics`` à ``None`` (pas
    de champ dans le JSON de sortie).
  - +24 tests dans `test_sprint61_philological_runner.py` (champs
    par défaut, sérialisation conditionnelle, libération par
    compact, calcul adaptive sur 6 cas de figure — médiéval,
    imprimé ancien, moderne, numéral romain, diacritiques,
    ASCII pur —, agrégation : sommes correctes, recalcul des scores
    globaux, per_category modern_archives, intégration runner
    end-to-end avec mock ``EngineResult``).
  - **Verrou levé** : les six modules philologiques sont désormais
    visibles dans le pipeline standard de bench ; il manque
    uniquement la vue HTML dédiée (Sprint 62 à venir).

- **Sprint 60 — Numéraux romains transversaux : couche de calcul
  (clôture extension philologique par période).**  Suite directe
  Sprints 56-59.  Les numéraux romains traversent les trois
  périodes patrimoniales : médiéval (minuscules + ``j`` final
  ``mcclxxxij`` = 1282), imprimé ancien (``Tome IV``), moderne
  (``Louis XIV``, ``MCMXIV``).  Pour chaque numéral GT, l'OCR
  peut le **préserver strictement**, **changer la casse**,
  **supprimer le ``j`` médiéval**, **convertir en chiffres
  arabes**, ou le **perdre**.  Ce module classifie les 5 statuts
  pour que le chercheur juge lui-même la convention.
  - Nouveau module `picarones/core/roman_numerals.py` :
    - ``roman_to_int(s)`` : parsing tolérant casse + ``j`` médiéval
      final, validation stricte des paires soustractives canoniques
      (IV, IX, XL, XC, CD, CM uniquement) — rejette « ICI » (faux
      positif), « VV », « LL », « DD », « IIIII ».
    - Forme additive médiévale ``IIII``/``XXXX`` acceptée.
    - ``int_to_roman(n)`` : conversion vers la forme canonique
      majuscule.
    - ``detect_roman_numerals(text, min_length=1)`` : regex
      ``\b[IVXLCDMivxlcdmj]+\b`` + validation par parsing.
      Paramètre ``min_length`` pour filtrer les single-letter
      ambigus (« I » pronom anglais, « M » initiale).
    - ``compute_roman_numeral_metrics(ref, hyp)`` classifie chaque
      numéral GT selon 5 statuts ordonnés par priorité :
      ``strict_preserved``, ``case_changed``, ``j_dropped``,
      ``converted_to_arabic``, ``lost``.  Retourne
      ``per_status``, ``per_numeral``, ``lost_numerals``,
      ``global_strict_score`` (forme exacte uniquement) et
      ``global_value_score`` (toute forme préservant la valeur).
    - Le breakdown ``per_status`` discrimine la convention adoptée :
      - majoritaire ``strict_preserved`` → diplomatique ;
      - majoritaire ``case_changed`` → modernisation typo ;
      - majoritaire ``j_dropped`` → modernisation orthographique
        médiévale ;
      - majoritaire ``converted_to_arabic`` → modernisation
        profonde du système numéral ;
      - majoritaire ``lost`` → erreur OCR.
  - ``roman_numeral_strict_score`` et
    ``roman_numeral_value_score`` enregistrés dans le registre
    typé Sprint 34 pour ``(TEXT, TEXT)``.
  - **Choix éditorial assumé** : aucune classification
    automatique imposée — le chercheur lit ``per_status`` et
    conclut.
  - +93 tests dans `test_sprint60_roman_numerals.py`
    (parsing/conversion bidirectionnelle paramétrée standard +
    minuscules + ``j`` médiéval, formes invalides rejetées,
    aller-retour ``int_to_roman``, détection avec ``min_length``,
    frontière de mot anti-faux-positif (``VIVE`` ne match pas),
    rejet faux positif ``ICI``, **5 statuts discriminés
    individuellement**, priorité strict > arabic, **3 cas
    réalistes par période** — charte médiévale ``mcclxxxij``,
    imprimé ancien ``Tome IV``, moderne ``Louis XIV`` —,
    comptage exhaustif somme des per_status = total, dégénérés,
    raccourcis, intégration registre).
  - **Verrou levé** : l'extension philologique est livrée pour
    les trois périodes principales (médiéval Sprints 56-57,
    imprimé ancien Sprint 58, moderne Sprint 59) **plus** la
    dimension transversale numérale ; un benchmark sur n'importe
    quel fonds patrimonial européen peut classer les moteurs
    sur leur traitement des numéraux romains, indépendamment de
    la période.

- **Sprint 59 — Marqueurs et abréviations des archives modernes
  XIXᵉ-XXᵉ : couche de calcul (extension axe philologique aux
  périodes contemporaines).**  Suite des Sprints 56-58.  Sur les
  fonds modernes BnF (état civil, recensements, presse, monographies,
  archives militaires, annuaires), la typographie historique a
  disparu mais subsiste un riche système d'abréviations propres à
  l'archive contemporaine.  Le module les couvre en **9 catégories**
  pour qu'un chercheur puisse juger lui-même la convention adoptée
  par chaque moteur, **sans qu'aucune classification automatique
  ne soit imposée**.
  - Nouveau module `picarones/core/modern_archives.py` :
    - 9 catégories de marqueurs :
      1. ``civility_titles`` : Mme, Mlle, Mgr, Dr, Pr, Me, M.,
         R.P., S.M., S.A.R., S.E., S.S.
      2. ``ordinals`` : 1ᵉʳ, 1ʳᵉ, 2ᵉ, 2ᵈ, 2ᵈᵉ, 3ᵉ, Iᵉʳ, Vᵉ,
         XIᵉ-XXᵉ (avec exposants Unicode ᵉ ʳ ᵈ).
      3. ``currency`` : ₶ (livre tournois), ₣ ƒ (franc), £, l. s. d.
         (livre/sol/denier d'Ancien Régime).
      4. ``administrative`` : arr., dép., cant., com., reg., prov.
      5. ``civil_status`` : ° (né), † (mort), ✶, ⚭, ép., vve.
      6. ``typographic_punctuation`` : « », –, —, …, ’, ‘.
      7. ``latin_abbr_modern`` : e.g., i.e., etc., cf., ibid.,
         op. cit., ad lib., N.B.
      8. ``bibliographic`` : vol., t., p., pp., n°, fasc., éd.,
         ms., f., r°, v°.
      9. ``address`` : bd, av., r., pl., imp., fbg.
    - ``get_category(marker)`` classe un marqueur ou retourne
      ``None`` ; ``get_expansions(marker)`` retourne les formes
      développées connues.
    - ``detect_modern_markers(text)`` retourne la liste ordonnée
      ``[(index, marker, category)]`` avec **stratégie greedy
      « plus long gagne »** (S.A.R. avant S.A.) et **frontières
      de mot adaptées** (espace/ponctuation pour les abréviations
      à point comme « M. ", « arr. », « r° » ; ``\b`` standard
      pour les alphabétiques comme « Mme » ou « bd »).
    - ``compute_modern_archives_metrics(ref, hyp)`` retourne, par
      catégorie, **deux scores** dans la lignée du Sprint 56 :
      ``strict_score`` (forme abrégée préservée telle quelle) et
      ``expansion_score`` (forme abrégée OU forme développée
      présente, sensible à la casse seulement pour les
      abréviations alphanumériques).  Le ratio des deux par
      catégorie permet au chercheur de **juger lui-même la
      convention** : strict ≈ expansion ≈ 1 → diplomatique ;
      strict = 0, expansion = 1 → modernisant ; les deux faibles
      → erreur OCR.
    - ``missed_markers`` distingue les **pertes pures** (ni
      abrégé ni développé) des **modernisations** (abrégé absent
      mais développé présent) via le booléen
      ``expansion_preserved``.
  - ``modern_archives_strict_score`` et
    ``modern_archives_expansion_score`` enregistrés dans le
    registre typé Sprint 34 pour ``(TEXT, TEXT)``.
  - **Choix éditorial assumé** : aucun module ne classe un moteur
    « diplomatique » ou « modernisant ".  L'outil est conçu pour
    un usage de **recherche** (BnF, philologie, sciences
    historiques) — il fournit les chiffres bruts et laisse le
    chercheur conclure.
  - +75 tests dans `test_sprint59_modern_archives.py`
    (catégorisation paramétrée 33 marqueurs sur 9 catégories,
    ``get_expansions``, détection par catégorie ×9, **greedy
    plus-long-gagne** sur S.A.R., **frontière de mot** sur « bd »
    vs « abdomen ", scénarios standards diplomatique/modernisant/
    erreur, breakdown per_category, dégénérés, missed_markers
    distinguant pertes pures et modernisations, **5 cas réalistes**
    par catégorie clé — citation bibliographique, registre d'état
    civil, adresse de recensement, protocole royal, monnaie
    d'Ancien Régime, ponctuation typographique —, comptage
    exhaustif, sanité tables, raccourcis, intégration registre).
  - **Verrou levé** : un benchmark sur fonds modernes XIXᵉ-XXᵉ
    peut désormais classer les moteurs sur leur traitement des
    abréviations contemporaines — symétrique au Sprint 56
    (médiéval) et au Sprint 58 (imprimé ancien).  L'extension
    philologique couvre maintenant trois périodes principales
    des fonds patrimoniaux européens.

- **Sprint 58 — Marqueurs typographiques de l'imprimé ancien :
  couche de calcul (extension axe philologique aux périodes
  XVIᵉ-XVIIIᵉ).**  Les Sprints 56 (abréviations Capelli) et 57
  (couverture MUFI) sont orientés **médiéval scribal**.  Picarones
  doit aussi servir les éditeurs d'**imprimés anciens**, pour qui
  les marqueurs caractéristiques sont **typographiques** (et non
  scribaux) : ligatures composées, s long ſ, i sans point ı,
  esperluette &, tildes nasaux indiquant une abréviation.
  - Nouveau module `picarones/core/early_modern_typography.py` :
    - 5 catégories de marqueurs typographiques : ``ligatures``
      (ﬀ ﬁ ﬂ ﬃ ﬄ ﬅ ﬆ), ``long_s`` (ſ), ``dotless_i`` (ı),
      ``ampersand`` (&), ``nasal_tildes`` (ã Ã ñ Ñ õ Õ ũ Ũ ẽ Ẽ ĩ Ĩ
      pré-composés + séquences ``voyelle + U+0303``).
    - ``get_category(char)`` classe un caractère dans une catégorie
      ou retourne ``None``.
    - ``detect_markers(text)`` retourne la liste ordonnée des
      marqueurs avec leur index et leur catégorie ; reconnaît à la
      fois les caractères pré-composés et les séquences combinantes
      (``a`` + U+0303 → nasal_tildes).
    - ``compute_early_modern_metrics(ref, hyp)`` retourne le taux
      de préservation par catégorie + ``global_preservation`` +
      ``missed_markers`` (liste des marqueurs ratés avec index et
      catégorie).
    - Approche identique aux Sprints 56-57 (alignement caractère
      par caractère via ``difflib.SequenceMatcher``).
  - ``early_modern_preservation`` enregistré dans le registre typé
    Sprint 34 pour ``(TEXT, TEXT)``.
  - **Le breakdown par catégorie discrimine la convention** : un
    moteur diplomatique préserve toutes les catégories ; un moteur
    modernisant préserve typiquement l'esperluette mais pas les
    ligatures, le s long, le i sans point ni les tildes nasaux ;
    un moteur mixte panache.
  - +38 tests dans `test_sprint58_early_modern.py` (catégorisation
    paramétrée sur 18 caractères, détection des 5 catégories,
    séquence ``voyelle + U+0303``, ordre préservé, **trois
    scénarios standards** discriminés — diplomatique 1.0,
    modernisant 0.2, mixte 0.4 —, breakdown per_category, cas
    dégénérés, comptage exhaustif preserved+missed=total, sets
    disjoints, raccourci, intégration registre).
  - **Verrou levé** : un benchmark sur des imprimés anciens XVIᵉ-XVIIIᵉ
    peut désormais classer les moteurs sur leur convention typographique
    éditoriale (diplomatique vs modernisante) — symétrique à ce que
    le Sprint 56 fait pour les manuscrits médiévaux.

- **Sprint 57 — A.II.3.3 Couverture MUFI : couche de calcul
  (clôture A.II.3 philologique côté calcul).** Suite des Sprints
  55-56 dans l'axe philologique.  La **Medieval Unicode Font
  Initiative** (MUFI v4.0) standardise les caractères médiévaux
  attendus en transcription fidèle ; le module mesure le taux de
  caractères MUFI de la GT correctement restitués dans l'OCR.
  - Nouveau module `picarones/core/mufi.py` :
    - ``_MUFI_RANGES`` : 4 plages Unicode caractéristiques (PUA
      ``E000-F8FF``, Latin Extended-D ``A720-A7FF``, Combining
      Diacritical Marks Supplement ``1DC0-1DFF``, Alphabetic
      Presentation Forms ``FB00-FB4F``).
    - ``_MUFI_EXPLICIT_CHARS`` : liste raisonnée de lettres
      médiévales hors plages (þ, Þ, ð, Ð, ƿ, Ƿ, ſ, æ, Æ, œ, Œ,
      ø, Ø, ƀ, ŧ, đ, ħ, ȝ, Ȝ, ꜿ).
    - ``is_mufi_char(char, custom_chars=None)`` extensible via
      paramètre.
    - ``compute_mufi_coverage(reference, hypothesis, custom_chars)``
      aligne caractère par caractère via
      ``difflib.SequenceMatcher`` (même méthode que le bloc Unicode
      du Sprint 55), classe les positions GT MUFI, et compte les
      positions correctement restituées.
    - Retourne ``coverage`` global + ``per_char`` (total /
      preserved / coverage par caractère MUFI rencontré) + liste
      ``missed_chars`` (caractères MUFI ratés).
  - ``mufi_coverage`` enregistré dans le registre typé Sprint 34
    pour ``(TEXT, TEXT)``.
  - +41 tests dans `test_sprint57_mufi.py` (détection sur 28
    caractères clés, plage PUA, custom_chars extensible ; coverage
    diplomatique → 1, modernisante → 0, partielle avec breakdown
    per_char ; cas dégénérés ; comptage exhaustif ;
    intégration registre).

- **Sprint 56 — A.II.3.2 Score d'expansion d'abréviations
  médiévales : couche de calcul.** Suite du Sprint 55 dans l'axe
  A.II.3 (philologique).  Sur les manuscrits médiévaux, les
  scribes utilisent intensivement des signes d'abréviation
  (``ꝑ``=per, ``ꝓ``=pro, ``⁊``=et, etc.).  Un OCR/HTR a trois
  comportements possibles face à eux : préserver, développer, ou
  perdre.  Le module discrimine les trois.
  - Nouveau module `picarones/core/abbreviations.py` :
    - Table ``ABBREVIATION_EXPANSIONS`` des signes Capelli + MUFI
      les plus courants (10 entrées : ꝑ, ꝓ, ꝗ, ꝙ, ꝯ, ⁊, ꝝ, ꝫ,
      ꝭ, et séquences ``lettre + U+0303`` comme ``p̃``, ``q̃``).
    - ``detect_abbreviations(text)`` retourne la liste ordonnée
      des abréviations, avec doublons préservés et tolérance
      NFC/NFD.
    - ``compute_abbreviation_metrics(ref, hyp)`` produit deux
      scores complémentaires :
      - **strict_score** : taux d'abréviations Unicode dont la
        forme abrégée est préservée telle quelle (édition
        diplomatique).
      - **expansion_score** : taux d'abréviations dont SOIT la
        forme abrégée SOIT la forme développée attendue est
        présente (édition modernisée acceptée).
    - Le **ratio strict/expansion** dit la convention adoptée :
      si égal et proche de 1 → diplomatique ; si strict << expansion
      → modernisant ; les deux faibles → erreur OCR.
    - Frontière de mot exigée pour les expansions courtes (« et »,
      « us ») afin de limiter le bruit (« permettre » ne match
      pas « per »).
  - ``abbreviation_strict_score`` et ``abbreviation_expansion_score``
    enregistrés dans le registre typé Sprint 34 pour ``(TEXT, TEXT)``.
  - +23 tests dans `test_sprint56_abbreviations.py` couvrant la
    détection (Unicode + tilde combinant + duplications + NFD),
    les **trois scénarios standards** (diplomatique → strict=1,
    expansion=1 ; modernisant → strict=0, expansion=1 ;
    mauvais OCR → 0/0 ; mixte → strict=0.5, expansion=1), le
    breakdown per_abbreviation, les cas dégénérés, la frontière
    de mot pour les expansions courtes, l'intégration registre,
    et la sanité de la table d'expansions.

- **Sprint 55 — A.II.3.1 Précision par bloc Unicode : couche de
  calcul (démarrage A.II.3, métriques philologiques).** Pour un
  éditeur d'imprimés anciens ou un médiéviste, la question
  pertinente n'est pas seulement « quel CER global ? » mais
  « quels caractères historiques ce moteur restitue-t-il
  fidèlement ? ».  Le module produit la phrase actionnable du plan :
  *« GPT-4o restitue 100 % du Latin de Base mais 0 % des formes de
  présentation latine (ﬁ, ſ…) »*.
  - Nouveau module `picarones/core/unicode_blocks.py` :
    - Table de 22 blocs Unicode standard centrée sur les corpus
      patrimoniaux européens (Latin de Base, Latin Étendu A/B/C/D/E,
      Latin Extended Additional, Diacritiques combinants,
      Présentation latine, MUFI PUA, Greek, Cyrillic, etc.).
      Tout caractère hors table → ``"Other"`` (couverture
      exhaustive : ``sum(total) == len(GT)``).
    - ``get_block(char)`` retourne le nom du bloc Unicode.
    - ``compute_unicode_block_accuracy(reference, hypothesis)`` :
      aligne caractère par caractère via ``difflib.SequenceMatcher``,
      classe chaque caractère GT dans son bloc, et compte les
      positions correctement restituées (opcodes ``equal``).
      Retourne ``per_block`` (correct/total/accuracy par bloc) +
      ``global_accuracy``.
    - ``unicode_block_global_accuracy`` : raccourci enregistré dans
      le registre typé Sprint 34 pour ``(TEXT, TEXT)``.
  - +24 tests dans `test_sprint55_unicode_blocks.py` :
    - ``get_block`` sur 10 caractères clés (ASCII, é, ç, ƒ, ſ, ﬁ,
      diacritique combinant) + Other (vide, émoji)
    - calcul d'accuracy : identité, vide bilatéral / unilatéral,
      None, substitutions ciblées par bloc
    - **cas réaliste du plan** : OCR modernisant remplace ſ→s et
      ﬁ→fi → 100 % Latin de Base mais 0 % Présentation latine et
      0 % Latin Extended-A
    - insertions/suppressions, coverage exhaustive, raccourci,
      intégration registre

- **Sprint 54 — A.II.2.2 Layout F1 par type de région : couche de
  calcul (clôture A.II.2 côté calcul).** Dernière brique de l'axe
  A.II.2 (métriques structurelles).  Pour les manuscrits glosés
  (texte principal vs glose) ou les journaux multi-colonnes, c'est
  la métrique qui répond à *« le moteur sépare-t-il bien le texte
  principal de la glose ? »*.
  - Nouveau module `picarones/core/layout.py` :
    - dataclass `Region(id, type, bbox)` avec validation (bbox
      strictement positive)
    - `_iou_bbox` calcule l'IoU de deux rectangles (origine en haut
      à gauche, convention ALTO/PAGE)
    - `_align_regions` apparie GT ↔ hypothèse en greedy par IoU
      décroissant, **same type required** (case-insensitive),
      pattern identique au NER (Sprint 38)
    - `compute_layout_metrics(refs, hyps, iou_threshold=0.5)`
      retourne global F1 + per_type + listes
      ``missed_regions`` (FN) et ``hallucinated_regions`` (FP)
    - `layout_f1` : raccourci pour le F1 global
  - Conventions : seuil IoU par défaut à 0,5 (convention ICDAR),
    coercion automatique dict → ``Region``, comparaison de type
    insensible à la casse.
  - Pas d'enregistrement registre typé pour ce sprint — la métrique
    suppose un parsing préalable (extraction des régions avec types
    et bbox depuis l'ALTO/PAGE) qui ne s'inscrit pas directement
    dans le pattern `(ArtifactType, ArtifactType)`.  L'enregistrement
    suivra quand le parser ALTO standard sera livré.
  - +20 tests dans `test_sprint54_layout.py` (validation Region,
    IoU mathématique, cas standards : parfait, mauvais type,
    hallucination, FN, IoU sous/sur seuil, multi-type breakdown,
    alignement greedy avec best-IoU wins, dégénérés, type
    case-insensitive, shortcut).

- **Sprint 53 — A.II.2.1 Reading order F1 (ICDAR 2015) : couche de
  calcul.** Suite du Sprint 52 dans l'axe A.II.2 (métriques
  structurelles).  Sur un manuscrit glosé ou un journal multi-colonnes,
  un moteur peut avoir un excellent CER caractère et un ordre de
  lecture catastrophique — le CER seul ne capture pas cette
  dimension.
  - Nouveau module `picarones/core/reading_order.py` :
    - ``compute_reading_order_metrics(ref_order, hyp_order)`` :
      pour chaque paire ``(a, b)`` où ``a`` précède ``b`` dans la GT,
      vérifie si ``a`` précède aussi ``b`` dans l'hypothèse.  Retourne
      precision/recall/F1 + détails (TP/FP/FN, paires totales, régions
      communes vs disjointes).
    - ``reading_order_f1`` : raccourci qui retourne juste le F1.
  - Conventions : doublons traités à la première occurrence,
    séquences ``None``/vides → F1 = 0 (pas de récompense gratuite),
    séquence à 1 région → 0 paire émise → F1 = 0 (convention de bord).
  - Format compatible avec ``ReadingOrderGT.region_order`` du
    Sprint 32 — l'utilisateur fournit directement la liste d'IDs.
  - ``reading_order_f1`` enregistré dans le registre typé Sprint 34
    pour la jonction ``(READING_ORDER, READING_ORDER)``.
  - +16 tests dans `test_sprint53_reading_order.py` (cas canoniques :
    identique → F1=1, inversé → F1=0, permutation locale, insertion,
    suppression ; cas dégénérés : vide, single region, doublons,
    None ; comptages détaillés ; intégration registre typé).

- **Sprint 52 — A.II.2.3 Différence Flesch : couche de calcul
  (démarrage de l'Étape 3 / axe A — métriques structurelles).**
  Stratégie identique aux Sprints 35/38/39 (couche pure d'abord,
  câblage runner+HTML après).
  - Nouveau module `picarones/core/readability.py` :
    - ``count_syllables_word`` : heuristique groupes de voyelles
      consécutives (avec diacritiques FR/EN), fallback à 1 syllabe
      pour les mots sans voyelle (acronymes type « BNF »).
    - ``count_words`` (regex Unicode) et ``count_sentences``
      (découpe sur ``.!?…``, minimum 1 si le texte contient au
      moins un mot).
    - ``flesch_score(text, lang)`` avec coefficients FR
      (Kandel-Moles 1958, ``207 - 1.015·m/p - 73.6·s/m``) et EN
      (Flesch 1948, ``206.835 - 1.015·m/p - 84.6·s/m``).  Score
      borné dans ``[0, 100]``.
    - ``flesch_delta(reference, hypothesis, lang)`` retourne la
      différence ``Flesch(OCR) - Flesch(GT)``.  **Positif = signal
      d'over-normalisation LLM** (le LLM a lissé la langue
      historique).
  - **Aucun alignement caractère/mot requis** : la métrique reste
    calculable même quand l'OCR est très dégradé — c'est l'avantage
    clé pour repérer les VLM/LLM qui hallucinent du texte moderne
    plausible mais déconnecté de la GT.
  - `flesch_delta_fr` et `flesch_delta_en` enregistrés dans le
    registre typé Sprint 34 pour la jonction ``(TEXT, TEXT)``.
  - +25 tests dans `test_sprint52_readability.py` (compteurs de
    base avec cas limites, score Flesch borné, FR/EN cohérents,
    delta nul sur textes identiques, **cas réaliste de
    modernisation LLM** → delta > 10 pts, OCR dégradé borné,
    intégration registre typé).

- **Sprint 51 — Adapter Azure Document Intelligence : exposition de
  `Word.confidence` (clôture de l'adaptation engines).** Suite directe
  des Sprints 47-50. Azure DI expose ``analyzeResult.pages[].words[]``
  avec ``content`` et ``confidence`` ∈ [0, 1]. L'adapter parcourt cette
  hiérarchie et émet une entrée par mot au format Sprint 42.
  - Refactor : ``_run_ocr_with_result(image_path) → (text,
    analyze_result_dict)`` centralise les deux chemins (SDK
    ``azure-ai-documentintelligence`` et REST direct via ``urllib``
    avec polling Azure asynchrone).
  - ``_sdk_result_to_dict`` convertit l'objet SDK en dict normalisé
    identique à la réponse REST pour traitement uniforme.
  - ``_extract_token_confidences_from_result`` parcourt
    ``pages[].words[]``, extrait ``content`` et ``confidence``,
    filtre les confidences None / négatives et les contenus vides.
  - Le texte ``EngineResult.text`` est extrait depuis
    ``pages[].lines[]`` (préservation rétrocompat octet par octet).
  - Flag config ``expose_confidences: false``.
  - L'API est appelée une seule fois — aucun overhead.
  - +16 tests dans ``test_sprint51_azure_confidences.py`` (extraction
    multi-pages, filtrage 4 cas, cas dégénérés 4 cas, conversion SDK
    → dict, surcharge ``run()`` avec mock, échec API, intégration
    runner).

- **Sprint 50 — Adapter Google Vision : exposition de
  `Word.confidence`.** Suite directe des Sprints 47-49.
  ``DOCUMENT_TEXT_DETECTION`` expose ``Word.confidence`` au niveau mot
  sur ``page > block > paragraph > word`` ; l'adapter parcourt cette
  hiérarchie et émet une entrée par mot au format Sprint 42.
  - Refactor : ``_run_ocr_with_full_annotation(image_path) → (text,
    full_dict)`` centralise les deux chemins (SDK
    ``google-cloud-vision`` et REST direct via ``urllib``).
    ``_run_ocr`` reste rétrocompat (retourne juste le texte).
  - ``_sdk_full_text_to_dict`` convertit la réponse proto SDK en
    dict normalisé identique à la réponse REST, pour traitement
    uniforme.
  - ``_extract_token_confidences_from_full_text`` parcourt
    ``pages → blocks → paragraphs → words``, reconstruit chaque mot
    par concaténation des ``word.symbols[i].text``, et émet
    ``{"token": str, "confidence": float}`` (confidence ∈ [0, 1] —
    le runner Sprint 42 accepte directement ce format).
  - Filtrage cohérent avec les autres adapters : confidence None /
    négative → ignorée, mots vides → ignorés.
  - ``TEXT_DETECTION`` (mode "court") ne fournit pas de confidence
    par mot → ``token_confidences = None``.
  - Flag config ``expose_confidences: false``.
  - L'API est appelée une seule fois — aucun overhead.
  - +17 tests dans `test_sprint50_google_vision_confidences.py`
    (reconstruction depuis symbols, multi-pages/blocks, filtrage,
    flag, cas dégénérés, conversion SDK → dict, surcharge ``run()``
    avec mock du chemin réseau, REST avec urllib mocké, intégration
    runner).

- **Sprint 49 — Adapter Mistral OCR : exposition des
  `token_confidences` quand l'API les fournit.** Suite des Sprints
  47 (Tesseract) et 48 (Pero OCR). Mistral OCR a deux chemins :
  l'endpoint dédié `/v1/ocr` (modèle `mistral-ocr-latest`) qui peut
  exposer des champs `confidence` à différents niveaux, et l'API
  chat/vision (`pixtral-*`) qui ne fournit pas de confidences.
  - Refactor : nouvelle méthode `_run_ocr_with_response(image_path)`
    retourne `(text, raw_response)`. `_run_ocr_native_api` retourne
    désormais aussi le JSON brut. Le chemin chat/vision retourne
    `(text, None)` car aucune confidence n'est disponible.
  - `_extract_token_confidences_from_response` parse la réponse
    `/v1/ocr` en cascade :
    1. `pages[i].words[j]` avec `{"text", "confidence"}` →
       extraction directe
    2. `pages[i].lines[j]` avec `{"text", "confidence"}` →
       propagation de la confidence à chaque mot (pattern Pero
       Sprint 48)
    3. `pages[i].blocks[j]` → idem
  - Filtrage cohérent avec Tesseract/Pero : texte vide, confidence
    None, confidence négative → ignorés.
  - Si l'API ne retourne aucun champ `confidence` exploitable
    (cas courant si Mistral retourne uniquement du markdown), ou si
    on est sur le chemin chat/vision, `token_confidences = None`.
  - Nouveau paramètre config `expose_confidences: false` cohérent
    avec les autres adapters.
  - L'API est appelée **une seule fois** ; le coût est strictement
    identique à l'implémentation historique.
  - +17 tests dans `test_sprint49_mistral_confidences.py` couvrant
    l'extraction (words explicites, propagation lines/blocks,
    combinaison, filtrage texte vide / conf None / négative), les
    cas dégénérés (None, dict vide, pas de pages, markdown sans
    confidences, types invalides), le flag `expose_confidences=False`,
    la surcharge `run()` (mock du chemin réseau, chat/vision sans
    confidences, échec API), et l'intégration runner.

- **Sprint 48 — Adapter Pero OCR : exposition des `token_confidences`
  natifs.** Suite directe du Sprint 47 (Tesseract). Pero OCR fournit
  une confidence par ligne (``transcription_confidence``, probabilité
  CTC moyenne) ; l'adapter la propage à chaque mot de la ligne.
  - ``PeroOCREngine.run()`` surchargé : un seul appel
    ``parser.process_page`` produit le ``page_layout`` ; texte ET
    confidences en sont extraits sans coût supplémentaire (vs
    Tesseract qui doit faire deux appels distincts).
  - Refactor : ``_run_pero_pipeline(image_path) -> (text,
    page_layout)`` centralise l'appel au pipeline ; ``_run_ocr``
    devient un wrapper trivial pour rétrocompat.
  - ``_extract_token_confidences_from_layout`` parcourt
    regions/lines, applique ``transcription_confidence`` à chaque
    mot de la ligne, ignore les transcriptions vides / confidences
    None / confidences négatives, retourne ``None`` si aucune ligne
    n'avait de confidence exploitable.
  - Nouveau paramètre config ``expose_confidences: false`` (cohérent
    avec Tesseract Sprint 47).
  - Pipeline appelé une seule fois → **aucun overhead** par rapport
    à l'implémentation historique (vs un appel supplémentaire pour
    Tesseract).
  - +14 tests dans ``test_sprint48_pero_confidences.py`` couvrant :
    extraction depuis layout (tokens uniques, multi-lignes,
    transcription vide, confidence None / négative), flag
    ``expose_confidences=False``, cas dégénérés (None / regions
    vides / aucune confidence), surcharge ``run()`` (texte préservé
    octet par octet, échec du pipeline), intégration runner avec
    ``calibration_metrics`` correctement calculée, fallback gracieux
    quand pero-ocr est absent.

- **Sprint 47 — Adapter Tesseract : exposition des `token_confidences`
  natifs.** Premier des engines adaptés au câblage calibration
  (Sprint 42). L'utilisateur qui benchmarke avec Tesseract obtient
  désormais automatiquement ECE/MCE et reliability diagram dans le
  rapport, sans configuration supplémentaire.
  - `TesseractEngine.run()` est surchargé : appelle `image_to_string`
    pour le texte (rétrocompat octet par octet) **et** `image_to_data`
    pour les confidences mot par mot, retourne un `EngineResult` avec
    `token_confidences = [{"token": str, "confidence": float}, …]`
    (confidence ∈ [0, 100], le runner Sprint 42 normalise en [0, 1]).
  - Helper `_extract_token_confidences()` séparé du chemin OCR
    principal : si `image_to_data` lève, l'OCR continue normalement
    et `token_confidences = None` (warning explicite, pas
    `except: pass`).
  - Filtrage à la source : non-mots Tesseract (conf = -1), tokens
    vides, longueurs incompatibles → ignorés.
  - Nouveau paramètre config `expose_confidences: false` pour
    désactiver le second appel Tesseract (économie d'un appel par
    image en cas de besoin).
  - Coût additionnel : un appel `image_to_data` par image. Le texte
    de `image_to_string` n'est jamais reconstruit depuis
    `image_to_data` — préservation stricte du comportement
    historique.
  - +9 tests dans `test_sprint47_tesseract_confidences.py` couvrant
    l'exposition des confidences (avec mock pytesseract), la
    préservation octet par octet du texte, le flag
    `expose_confidences=False`, le fallback gracieux quand
    `image_to_data` lève (warning + `None`), le filtrage des
    non-mots/longueurs incompatibles, l'intégration bout-en-bout
    avec le runner (`calibration_metrics` calculé), et le cas
    pytesseract absent.

- **Sprint 46 — A.III stratification par `script_type` : vue HTML +
  détecteur narratif (clôture A.III)**. Suite directe du Sprint 45
  (couche backend). La vue stratifiée est désormais rendue dans le
  rapport et un détecteur signale automatiquement les corpus
  hétérogènes.
  - Nouveau module `picarones/report/stratification_render.py` :
    `build_stratified_ranking_html` rend un `<details>` natif
    (collapsible sans JS) par strate avec tableau moteur × (médiane,
    moyenne, docs). Cellule médiane colorée par gradient vert (faible
    CER) → rouge (élevé). Premier `<details>` ouvert par défaut pour
    donner le contexte. Bandeau d'avertissement en tête si
    `corpus_homogeneity` fourni (écart inter-strate du leader).
  - `_build_report_data` expose `available_strata`,
    `stratified_ranking`, `corpus_homogeneity` au top-level. Le bloc
    HTML est passé au template `view_ranking.html` qui l'insère après
    le tableau principal **uniquement si stratification disponible**
    (rapport adaptatif).
  - Nouveau `FactType.STRATIFICATION_RECOMMENDED` (priority 45,
    importance MEDIUM ou HIGH selon le gap) avec détecteur
    `detect_stratification_recommended` qui lit `corpus_homogeneity`
    et émet un Fact quand le gap inter-strate du leader dépasse
    5 points de CER (HIGH au-delà de 10 points). Templates FR/EN
    sans nombres en dur.
  - L'arbitre marque la paire `{GLOBAL_LEADER_CER,
    STRATIFICATION_RECOMMENDED}` comme **complémentaire** : la
    recommandation peut cohabiter avec la phrase du leader pour
    nuancer.
  - +8 clés i18n FR/EN pour la vue stratifiée
    (`stratification_caption`, `stratification_description`,
    `stratification_*_label`, `stratification_gap_summary`).
  - Anti-injection HTML via `html.escape` sur les noms de moteurs et
    les noms de strates.
  - +38 tests dans `test_sprint46_stratification_html.py` couvrant
    le rendu (un `<details>` par strate, métriques visibles, premier
    ouvert), le bandeau d'hétérogénéité, le masquage adaptatif (4
    cas), l'anti-injection (engine et stratum avec balises HTML),
    les seuils du détecteur (4 cas), la traçabilité
    anti-hallucination FR + EN, l'absence de chiffres en dur dans
    les templates, l'intégration `ReportGenerator` FR + EN, et la
    complétude i18n.

- **Sprint 45 — A.III stratification par `script_type` : couche
  d'agrégation backend.** Première brique de la « plus haute valeur
  ajoutée transversale » du plan d'évolution. Le rapport peut
  désormais classer les moteurs **par strate** (manuscrit gothique,
  cursive administrative, imprimé ancien, humanistique…) — la moyenne
  globale ment quand le corpus est hétérogène.
  - `BenchmarkResult.doc_strata: Optional[dict[str, str]]` :
    map ``{doc_id: script_type}`` capturée par le runner avant
    ``compact()`` (qui efface ``image_quality``).
  - `BenchmarkResult.available_strata()` : liste triée des strates
    distinctes, ignore les valeurs vides.
  - `BenchmarkResult.stratified_ranking()` : retourne
    ``{stratum: [ranking_entry]}`` avec mean/median CER **recalculés
    par strate**, tri par médiane (cohérent avec Sprint 44). Inclut
    les moteurs sans aucun doc dans une strate sous forme d'entrée
    dégénérée (mean/median = None).
  - `BenchmarkResult.corpus_homogeneity()` : pour le moteur leader
    global, retourne l'écart inter-strate de la médiane CER et
    identifie la paire de strates min/max — base du futur
    avertissement automatique « ce corpus est hétérogène,
    consultez la vue stratifiée ».
  - `as_dict()` expose `doc_strata`, `available_strata`,
    `stratified_ranking`, `corpus_homogeneity` quand renseignés
    (rétrocompat stricte sinon — clés absentes).
  - Le runner peuple `doc_strata` avant compact en lisant
    ``DocumentResult.image_quality["script_type"]``.
  - +16 tests dans `test_sprint45_stratification.py` couvrant les
    fields, available_strata, stratified_ranking (1 entrée/moteur/
    strate, métriques per-strate, tri par médiane, moteurs absents),
    corpus_homogeneity (None < 2 strates, calcul d'écart),
    sérialisation as_dict, et un **test propriété réaliste** : le
    leader global peut perdre sur une strate (Tesseract domine
    globalement mais perd sur le manuscrit où Pero gagne).

- **Sprint 44 — A.I.2 : tri par médiane CER par défaut + détecteur
  d'asymétrie.** Réponse à la critique structurelle 2 du plan
  d'évolution : sur les corpus patrimoniaux, la moyenne est facilement
  tirée par quelques documents catastrophiques et masque les
  performances réelles ; la médiane est plus représentative.
  - `EngineReport.median_cer` : nouvelle propriété qui lit
    `aggregated_metrics["cer"]["median"]`.
  - `BenchmarkResult.ranking()` :
    - inclut désormais `median_cer` dans chaque entrée (additif)
    - **trie par médiane CER croissante par défaut** (et non plus
      par moyenne)
    - retombe sur `mean_cer` quand `median_cer` est absent
      (rétrocompat pour le cas pathologique)
  - Nouveau `FactType.MEDIAN_MEAN_GAP_WARNING` et détecteur
    `detect_median_mean_gap_warning` (priority 140) : émet un Fact
    quand `|mean - median| / median > 30 %` pour le moteur leader.
    Importance MEDIUM par défaut, HIGH si gap relatif ≥ 100 %.
    Garde-fou : ne déclenche pas si la médiane est nulle.
  - Templates FR/EN — aucun nombre en dur, tout vient du payload
    (vérifié par test).
  - L'arbitre marque la paire `{GLOBAL_LEADER_CER,
    MEDIAN_MEAN_GAP_WARNING}` comme **complémentaire** : les deux
    phrases peuvent coexister dans la synthèse pour nuancer le
    leader.
  - +15 tests dans `test_sprint44_median_default.py` (propriété
    median_cer, tri par médiane sur cas asymétrique réaliste,
    fallback sur la moyenne, déclenchement du détecteur sur 4 cas
    dégénérés, importance MEDIUM/HIGH selon gap, traçabilité
    anti-hallucination FR + EN, intégration via build_synthesis).

- **Sprint 43 — A.II.1.b Calibration : vue HTML reliability diagram +
  tableau ECE/MCE (clôture A.II.1.b côté rapport).** Suite directe du
  Sprint 42 (câblage runner). Les chiffres de calibration sont
  désormais visibles dans le rapport HTML.
  - Nouveau module `picarones/report/calibration_render.py` :
    - `build_calibration_summary_html(engines_summary, labels)` :
      tableau résumé par moteur (ECE, MCE, Précision moyenne,
      Confiance moyenne, n_predictions, doc_count). Cellule ECE
      colorée par gradient vert (bien calibré) → rouge (mal
      calibré).
    - `build_reliability_diagram_svg(aggregated_calibration, labels,
      engine_name)` : SVG d'un reliability diagram avec barres
      d'accuracy par bin, ligne reliant les points
      `(avg_confidence, accuracy)`, diagonale de référence en
      pointillé, axes annotés (graduations 0/0.5/1).
    - `build_reliability_diagrams_grid_html(engines_summary,
      labels)` : grille auto-fit, un SVG par moteur ayant un
      `aggregated_calibration`.
    - Rendu strictement server-side, déterministe, **pas de
      JavaScript**, cohérent avec le SVG du CDD (Sprint 18) et les
      sections inter-moteurs (Sprint 37) et NER (Sprint 41).
  - `_build_report_data` expose `aggregated_calibration` par moteur
    dans `engines_summary`. `ReportGenerator.generate` calcule les
    deux blocs et les passe au template `view_analyses.html` qui les
    affiche dans une `chart-card` à largeur pleine **uniquement si
    au moins un moteur a un `aggregated_calibration`** (rapport
    adaptatif).
  - +13 clés i18n FR/EN (`h_calibration`, `calibration_note`,
    `calibration_summary_caption`, `calibration_engine_label`,
    `calibration_ece_label`, `calibration_mce_label`,
    `calibration_n_label`, `calibration_acc_label`,
    `calibration_conf_label`, `calibration_docs_label`,
    `reliability_diagram_title`, `reliability_x_axis`,
    `reliability_y_axis`).
  - +43 tests dans `test_sprint43_calibration_html.py` couvrant le
    rendu (résumé, SVG avec barres/points/diagonale, grille
    multi-moteurs), le masquage adaptatif (4 cas dégénérés),
    l'anti-injection (engine name `<script>` ou `<img>`),
    l'intégration rapport FR + EN, et la complétude i18n sur les
    13 clés × 2 langues.

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

- 1478 → 2086 tests (+17 Sprint 32, +23 Sprint 33, +21 Sprint 34,
  +27 Sprint 35, +22 Sprint 36, +42 Sprint 37, +19 Sprint 38,
  +32 Sprint 39, +16 Sprint 40, +38 Sprint 41, +17 Sprint 42,
  +43 Sprint 43, +15 Sprint 44, +16 Sprint 45, +38 Sprint 46,
  +9 Sprint 47, +14 Sprint 48, +17 Sprint 49, +17 Sprint 50,
  +16 Sprint 51, +25 Sprint 52, +16 Sprint 53, +20 Sprint 54,
  +24 Sprint 55, +23 Sprint 56, +41 Sprint 57). Aucune régression. **Phase 0 close ; Étape 2
  intégralement livrée ; Étape 3 / axe A.II.2 (métriques
  structurelles) couches de calcul intégralement livrées
  (Sprints 52-54) ; Étape 3 / axe A.II.3 (métriques philologiques)
  démarrée :** Précision par bloc Unicode (A.II.3.1, Sprint 55).

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
