"""Registre typé de métriques — Sprint S5.

Construit **explicitement** par un service au démarrage de
l'application, pas par effet de bord d'import au top-level d'un
package.

Anti-pattern à éviter (présent dans l'existant et listé dans
``BACKLOG_POST_LIVRAISON.md`` §2.4) — un ``__init__.py`` qui
importe un sous-package "uniquement pour amorcer un registre",
chargeant des dizaines de modules et leurs dépendances optionnelles
au moment d'un simple ``import picarones``.

Pattern cible : un service ``build_default_registry()`` instancié
au démarrage de l'application qui ``register()`` chaque métrique
explicitement.  Le registre est ensuite injecté dans les services
qui en ont besoin.  Pas de singleton global, pas de side effect
d'import.
"""

from __future__ import annotations

__all__: list[str] = []
