"""Métriques officielles Picarones — Cercle 2.

Ce package contient l'ensemble des mesures et analyses qui calculent,
agrègent ou interprètent des métriques sur un corpus. Il dépend du
cercle 1 (``picarones.core``) qui définit les abstractions, et est
consommé par le cercle 3 (``picarones.report``, ``picarones.cli``,
``picarones.web``) qui présente les résultats.

Sous-modules
------------
Coeur :

- :mod:`metrics`              compute_metrics (CER/WER/MER/WIL via jiwer)
- :mod:`statistics`           Wilcoxon, Friedman, Nemenyi, Pareto, CDD
- :mod:`runner`               run_benchmark — orchestration parallèle
- :mod:`builtin_hooks`        12 hooks doc + 12 agrégateurs natifs
- :mod:`builtin_metrics`      enregistrement métriques dans le registry
- :mod:`alto_metrics`         métriques jonction TEXT/ALTO
- :mod:`normalization`        profils Unicode

Erreurs et taxonomie :

- :mod:`confusion`            matrice de confusion Unicode
- :mod:`char_scores`          scores ligatures/diacritiques
- :mod:`taxonomy`             taxonomie 9 classes d'erreurs
- :mod:`taxonomy_comparison`  comparaison taxonomique miroir
- :mod:`taxonomy_cooccurrence` Jaccard inter-classes
- :mod:`taxonomy_intra_doc`   heatmap classes × position

Structure et lignes :

- :mod:`structure`            blocs/lignes/mots
- :mod:`line_metrics`         distribution CER par ligne (Gini, percentiles)
- :mod:`worst_lines`          lignes pires globales

Fiabilité et calibration :

- :mod:`calibration`          ECE, MCE, reliability bins
- :mod:`reliability`          IAA Cohen κ + multirun stability
- :mod:`hallucination`        détection hallucinations VLM
- :mod:`robustness`           courbes CER vs dégradation
- :mod:`robustness_projection` projection sur corpus réel

Image et difficulté :

- :mod:`image_quality`        contraste, bruit, flou…
- :mod:`image_predictive`     complexité paléographique
- :mod:`difficulty`           score difficulté intrinsèque

Contenu et lisibilité :

- :mod:`searchability`        recherchabilité fuzzy (Levenshtein)
- :mod:`numerical_sequences`  préservation dates/cotes/numéraux
- :mod:`rare_tokens`          rappel sur tokens rares
- :mod:`readability`          Δ Flesch (sur-normalisation)

Structure ALTO et entités :

- :mod:`layout`               F1 layout par type de région
- :mod:`reading_order`        F1 ordre de lecture (ICDAR 2015)
- :mod:`ner`, :mod:`ner_backends`
- :mod:`error_absorption`     correction vs introduction par jonction

Inter-moteurs et historique :

- :mod:`inter_engine`         divergence taxonomique + oracle gap
- :mod:`specialization`       spécialisation inter-moteurs
- :mod:`baseline_comparison`  comparaison à l'historique
- :mod:`longitudinal`         régression linéaire + change-point
- :mod:`incremental_comparison` ANOVA-like par slot
- :mod:`history`              historique SQLite

Économie et opération :

- :mod:`pricing`              table tarifaire
- :mod:`throughput`           pages/h effectif
- :mod:`cost_projection`      projection à volume cible
- :mod:`marginal_cost`        coût par erreur évitée

Philologie historique :

- :mod:`mufi`                 couverture MUFI (médiéval)
- :mod:`abbreviations`        signes d'abréviation Capelli
- :mod:`unicode_blocks`       précision par bloc Unicode
- :mod:`early_modern_typography` ligatures imprimées XVIᵉ-XVIIIᵉ
- :mod:`modern_archives`      marqueurs XIXᵉ-XXᵉ
- :mod:`roman_numerals`       numéraux romains
- :mod:`lexical_modernization` sur-normalisation lexicale

Pipelines composées (axe B) :

- :mod:`pipeline_benchmark`, :mod:`pipeline_comparison`,
  :mod:`pipeline_spec_loader`

Aide à la décision :

- :mod:`levers`               leviers d'amélioration factuels
- :mod:`equivalence_profile`  curseur fin équivalences diplomatiques
- :mod:`module_policy`        manifest + audit modules contribués

Câblages adaptifs (suffixe ``_hooks``) :

- :mod:`readability_hooks`, :mod:`searchability_hooks`,
  :mod:`numerical_sequences_hooks`, :mod:`philological_hooks` —
  adaptive masking document-par-document, consommés par
  :mod:`builtin_hooks`. Ces modules sont des couches d'adaptation
  entre le calcul pur (sans I/O) et le runner principal (avec
  agrégation par moteur).

Moteur narratif :

- :mod:`narrative` (sous-package) : arbiter, registry, renderer,
  18 détecteurs en 6 familles. Le modèle de données (``Fact``,
  ``FactType``, ``DetectorRegistry``) vit en cercle 1 dans
  :mod:`picarones.core.facts`.

Voir :doc:`docs/architecture.md` pour la cartographie complète et
la règle de dépendance des 3 cercles.
"""
