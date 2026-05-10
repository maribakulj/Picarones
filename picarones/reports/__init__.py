"""Cercle 3 — Reports.

Sortie en différents formats à partir d'un ``RunResult`` persisté.
Le rapport est une **vue** des artefacts et des résultats
d'évaluation, jamais une source de vérité.

Sous-packages :

- ``html/`` — rapport HTML interactif (cible Sprint S22).
  Consomme ``RunManifest`` + ``view_results.jsonl`` plutôt que
  l'ancien ``BenchmarkResult`` fourre-tout.
- ``json/`` — export JSON canonique pour intégration externe.
- ``csv/`` — exports tabulaires par vue d'évaluation.

Règles : un rapport ne doit jamais **recalculer** un score.  Tout
ce qu'il affiche provient des fichiers persistés par le run.

Note de migration : ce package s'appelle ``reports`` pendant le
rewrite pour cohabiter avec l'existant ``picarones.report`` (qui
sera supprimé au S22).  Renommé en ``reports`` à la fin du
rewrite.
"""

from __future__ import annotations

__all__: list[str] = []
