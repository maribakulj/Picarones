"""Cercle 2 — Pipeline execution.

Exécution séquentielle ou DAG-branchante d'une chaîne de modules
tiers (BaseModule).  Picarones ne fournit **aucun module métier** —
l'utilisateur amène ses propres adaptateurs OCR/LLM/VLM/correcteur/
reconstructeur ALTO ; le pipeline executor les compose, valide les
types aux jonctions et évalue automatiquement chaque artefact
produit contre la GT correspondante.

Modules cibles (à venir Sprints S6-S8) :

- ``spec.py`` — ``PipelineSpec``, ``PipelineStep``, ``inputs_from``
  (DAG branchant), validation statique des types aux jonctions.
- ``executor.py`` — ``PipelineExecutor.run(spec, document, inputs)``
  exécute mono-document avec capture gracieuse des erreurs.
- ``runner.py`` — ``CorpusRunner`` orchestre l'executor sur un
  corpus complet avec **backpressure**, **timeout depuis le début
  d'exécution réelle** (pas depuis la submission), **annulation
  propre** (signal aux workers en cours).
- ``cache.py`` — ``ArtifactCache`` indexé par
  ``hash(content + spec + code_version)`` pour reprise hashée
  (Sprint S7).
- ``protocols.py`` — protocole ``StepExecutor`` que doivent
  implémenter les adaptateurs.

Cible du Sprint S12 : équivalence numérique CER/WER avec l'ancien
``measurements.runner`` à 1e-9 près sur les fixtures.
"""

from __future__ import annotations

__all__: list[str] = []
