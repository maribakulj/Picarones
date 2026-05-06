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

Voir :doc:`docs/explanation/architecture.md` pour la cartographie complète et
la règle de dépendance des 3 cercles.
"""

# ──────────────────────────────────────────────────────────────────────────
# Sprint A3 (renforce le respect de la règle Cercle 2 → Cercle 1
# uniquement) — la cérémonie d'enregistrement des métriques typées dans
# le registre Sprint 34 a été déplacée ici depuis ``core/pipeline.py``
# qui violait la règle.
#
# Tout consommateur qui veut utiliser ``compute_at_junction``
# (``picarones.core.metric_registry``) doit avoir importé
# ``picarones.measurements`` au moins une fois pour que les décorateurs
# ``@register_metric`` aient été exécutés. C'est le cas par défaut dans
# le pipeline standard ; les notebooks isolés peuvent ajouter
# ``import picarones.measurements`` (suivi d'un commentaire d'exception
# ruff sur la ligne d'import si leur linter signale un import inutilisé).
#
# Sans ces imports, ``compute_at_junction`` trouverait un registre vide
# et ne calculerait rien aux jonctions.
# ──────────────────────────────────────────────────────────────────────────
# Sprint 34 : cer / wer / mer / wil + stub TEXT→ALTO
from picarones.measurements import builtin_metrics  # noqa: F401
# Sprints 55-60 : métriques philologiques.
from picarones.measurements import abbreviations  # noqa: F401
from picarones.measurements import early_modern_typography  # noqa: F401
from picarones.measurements import modern_archives  # noqa: F401
from picarones.measurements import mufi  # noqa: F401
from picarones.measurements import roman_numerals  # noqa: F401
from picarones.measurements import unicode_blocks  # noqa: F401
# Sprint 53 : reading order F1.  Sprints 38, 52 : NER, readability.
from picarones.measurements import ner  # noqa: F401
from picarones.measurements import readability  # noqa: F401
from picarones.measurements import reading_order  # noqa: F401
# Chantier 1 (post-Sprint 97) : métriques (ALTO, ALTO) pour évaluer
# les reconstructeurs ALTO contre une GT ALTO du document.
from picarones.measurements import alto_metrics  # noqa: F401

# ──────────────────────────────────────────────────────────────────────────
# Sprint « zéro dette actionnable » (mai 2026) — modules sans appel
# automatique par le runner OCR principal mais qui font partie de l'API
# publique de ``picarones.measurements``. L'import ici les rend
# accessibles en ``from picarones.measurements import X`` et garantit
# qu'aucun ne devient « test-only » silencieusement (cf.
# ``tests/architecture/test_module_coverage.py``).
#
# Distinction de scope :
# - Modules de calcul utilisés via les renderers HTML composables
#   (l'utilisateur les compose lui-même selon son use case) :
from picarones.measurements import baseline_comparison  # noqa: F401  # historique SQLite
from picarones.measurements import cost_projection  # noqa: F401  # volume cible utilisateur
from picarones.measurements import equivalence_profile  # noqa: F401  # curseur HTML
from picarones.measurements import error_absorption  # noqa: F401  # jonction pipeline composée
from picarones.measurements import layout  # noqa: F401  # GT ALTO requise (axe B)
from picarones.measurements import longitudinal  # noqa: F401  # historique SQLite
from picarones.measurements import marginal_cost  # noqa: F401  # paires de moteurs
from picarones.measurements import module_policy  # noqa: F401  # outil d'audit
from picarones.measurements import ner_backends  # noqa: F401  # factory backends NER
from picarones.measurements import rare_tokens  # noqa: F401  # corpus-wide
from picarones.measurements import reliability  # noqa: F401  # multi-runs
from picarones.measurements import taxonomy_cooccurrence  # noqa: F401  # depuis taxonomy
from picarones.measurements import taxonomy_intra_doc  # noqa: F401  # depuis taxonomy
