"""Cercle 2 — Evaluation.

Vues d'évaluation, projecteurs et calculs de métriques.

Le cœur de la valeur ajoutée Picarones : **comparer librement des
pipelines hétérogènes en projetant leurs sorties vers une vue
d'évaluation explicite**.  L'utilisateur ne compare jamais directement
un OCR brut et une sortie ALTO reconstruite — il compare leur
projection dans une vue commune (texte, ALTO, recherchabilité, etc.)
et le rapport explicite ce que la vue ignore.

Sous-packages :

- ``views/`` — ``TextView``, ``AltoView``, ``SearchView``, ...
- ``projectors/`` — ``AltoToText``, ``PageToText``, ``CanonicalToText``,
  qui transforment un type d'artefact vers un autre avec un
  ``ProjectionReport`` listant les pertes (lossiness explicite).
- ``metrics/`` — calculs purs : CER/WER, MUFI, philological,
  statistics, NER, etc.  Une métrique = ``(input_types, output_types,
  callable)``.
- ``registry/`` — registre typé construit explicitement par un
  service au démarrage (pas par effet de bord d'import).

Règles d'import : ce cercle dépend de ``domain/`` uniquement.  Pas
de fastapi, pas de jinja, pas de moteur OCR.  Il peut utiliser
``numpy`` et ``scipy`` pour les calculs statistiques.

Voir ``docs/roadmap/rewrite-2026.md`` pour le rôle des vues dans le
rewrite ciblé (Sprints S13-S18).
"""

from __future__ import annotations

__all__: list[str] = []
