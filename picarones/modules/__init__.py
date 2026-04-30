"""Modules de référence (Chantier 1 du plan d'évolution post-Sprint 97).

Ce package contient des implémentations concrètes de
:class:`picarones.core.modules.BaseModule` livrées par Picarones lui-même
en complément des cinq adaptateurs OCR (``picarones.engines``) et des
quatre adaptateurs LLM (``picarones.llm``).

Philosophie
-----------
Picarones est un **banc d'essai** : l'utilisateur amène ses propres
modules tiers (correcteur LLM, reconstructeur ALTO, mappeur VLM…) et
Picarones les compare.  Pour autant, le banc d'essai a besoin d'au
moins un module de chaque catégorie pour servir de **baseline** —
sans cela, l'utilisateur ne peut comparer son module à rien.

Les modules de ce package sont donc explicitement étiquetés comme
**références** : ils sont volontairement primitifs, déterministes, et
sans dépendance externe.  Leur but est :

1. Valider l'infrastructure ``BaseModule`` / ``PipelineRunner`` /
   ``compute_at_junction`` en conditions réelles (pas seulement avec
   des ``MockModule`` de test).
2. Fournir un point de comparaison stable.  *« Mon reconstructeur ALTO
   fait-il mieux que la baseline mono-bloc ? »* est une question
   tractable.
3. Rendre exécutable le rapport de pipeline composée — sans un seul
   reconstructeur réel, les renderers ``pipeline_dag``,
   ``error_absorption``, ``incremental_comparison`` et
   ``module_audit`` n'ont pas de données à montrer.

Modules disponibles
-------------------
- :class:`TextToAltoMonoRegion` (``alto_text_to_mono_region``) :
  reconstructeur primitif TEXT (+ IMAGE) → ALTO XML, une seule
  ``TextRegion`` couvrant l'image entière, une ``TextLine`` par ligne
  du texte, une ``String`` par mot.

Conventions
-----------
- Tous les modules ici héritent de ``BaseModule`` et déclarent leurs
  ``input_types`` / ``output_types``.
- Tous les modules ont un ``name`` stable et déterministe.
- Tous les modules sont **purs** (zéro side-effect, zéro réseau, zéro
  écriture disque).  Ils peuvent donc être exécutés en
  ``ProcessPoolExecutor`` sans précaution particulière.
- Aucun module ne dépend des autres modules de ce package.

Pour contribuer un nouveau module de référence, suivez la même
discipline et ajoutez un test unitaire dans ``tests/`` couvrant au
moins :

1. Cas trivial (entrée vide).
2. Cas standard (entrée typique).
3. Validation des types via ``module.validate_inputs(...)``.
"""

from picarones.modules.alto_text_to_mono_region import TextToAltoMonoRegion

__all__ = ["TextToAltoMonoRegion"]
