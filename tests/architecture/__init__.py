"""Invariants structurels du projet.

Ces tests ne vérifient pas un comportement métier mais la **forme**
du code lui-même : taille des fichiers, unicité des helpers de rendu,
cohérence des chemins documentés, couverture des modules par un
consommateur de production.

Ils existent pour casser le cycle « Claude dit que c'est propre ↔
audit suivant trouve une dérive ». Tant que ces invariants sont verts,
le projet est *structurellement* sain selon les seuils calibrés au
dernier release tag. Quand un invariant échoue, c'est un signal de
réveil : refactor, ou relèvement délibéré du seuil avec
justification dans le commit.

Re-calibrer à chaque release (``git tag vX.Y.Z``).
"""
