"""Métriques officielles Picarones — Cercle 2.

Phase E du chantier de refonte en 3 cercles. Ce package contient
l'ensemble des **mesures et analyses au-delà du noyau** : tout ce qui
calcule, agrège ou interprète des métriques sur un corpus, mais qui
n'est pas une abstraction du domaine (Cercle 1, ``core/``) ni un
plugin niche (Cercle 3, ``extras/``).

Sous-modules
------------
Métriques scalaires et structurelles :

- :mod:`confusion`           matrice de confusion Unicode
- :mod:`char_scores`         scores ligatures/diacritiques
- :mod:`taxonomy`             taxonomie 9 classes d'erreurs
- :mod:`taxonomy_comparison`  comparaison taxonomique miroir
- :mod:`structure`            analyse structurelle (lignes/blocs)
- :mod:`line_metrics`         distribution CER par ligne (Gini, percentiles)
- :mod:`hallucination`        détection hallucinations VLM
- :mod:`reading_order`        F1 ordre de lecture (ICDAR 2015)
- :mod:`layout`               F1 layout par type de région
- :mod:`error_absorption`     correction vs introduction par jonction
- :mod:`searchability`        recherchabilité fuzzy (Levenshtein)
- :mod:`numerical_sequences`  préservation dates/cotes/numéraux
- :mod:`numerical_sequences_runner`
- :mod:`rare_tokens`          rappel sur tokens rares
- :mod:`readability`          Δ Flesch (sur-normalisation)
- :mod:`readability_runner`
- :mod:`searchability_runner`
- :mod:`specialization`       spécialisation inter-moteurs
- :mod:`worst_lines`          lignes pires globales
- :mod:`inter_engine`         divergence taxonomique + oracle gap
- :mod:`incremental_comparison` ANOVA-like par slot
- :mod:`baseline_comparison`  comparaison à l'historique
- :mod:`longitudinal`         régression linéaire + change-point

Fiabilité et calibration :

- :mod:`calibration`          ECE, MCE, reliability bins
- :mod:`reliability`          IAA Cohen κ + multirun stability
- :mod:`robustness`           courbes CER vs dégradation
- :mod:`robustness_projection` projection sur corpus réel

NER :

- :mod:`ner`, :mod:`ner_backends`

Économie et opération :

- :mod:`pricing`              table tarifaire
- :mod:`throughput`           pages/h effectif
- :mod:`cost_projection`      projection à volume cible
- :mod:`marginal_cost`        coût par erreur évitée

Contexte corpus :

- :mod:`history`              historique SQLite
- :mod:`difficulty`           score difficulté intrinsèque
- :mod:`image_quality`        contraste, bruit, flou…
- :mod:`normalization`        profils Unicode

Statistiques :

- :mod:`statistics`           Wilcoxon, Friedman, Nemenyi, Pareto, CDD

Aide à la décision :

- :mod:`levers`               leviers d'amélioration factuels
- :mod:`equivalence_profile`  curseur fin équivalences diplomatiques

Hooks et registres :

- :mod:`builtin_hooks`        12 hooks doc + 12 agrégateurs natifs

Moteur narratif :

- :mod:`narrative` (sous-package) : facts, registry, arbiter, renderer, 18 détecteurs

Rétrocompatibilité absolue
--------------------------
Tous les modules historiquement dans ``picarones.core.X`` restent
accessibles via des fichiers-shims qui les redirigent vers le nouvel
emplacement. Aucun import existant ne casse.

Voir :doc:`docs/architecture.md` et la phase E du plan de
refonte.
"""
