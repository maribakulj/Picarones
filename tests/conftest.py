"""Configuration pytest globale.

Ce conftest racine ne fait **qu'une seule chose** : positionner les
variables d'environnement test-friendly **avant** tout import de
``picarones.web.*``. Sans ça, les singletons web (``JOBS_SEMAPHORE``,
``RATE_LIMITER``) seraient instanciés avec les valeurs de production
(2 jobs concurrents max, rate limit selon mode public) au moment du
premier import, et chaque test web verrait le bocal saturé.

L'isolation par-test des états globaux web (sémaphore, rate limiter,
browse roots) vit dans ``tests/web/conftest.py`` — fixture
``autouse=True`` qui ne s'applique qu'aux tests sous ``tests/web/``,
pour éviter qu'un test cercle 1 (``tests/core/``) ne paie le coût
de l'import de ``picarones.web.*`` à chaque exécution.
"""

from __future__ import annotations

import os

# Plafond très large pour ne jamais bloquer une suite de tests qui
# démarre rapidement plusieurs benchmarks daemon en parallèle.
os.environ.setdefault("PICARONES_MAX_CONCURRENT_JOBS", "32")

# Mode dev par défaut. Les tests qui valident le mode public le
# forcent eux-mêmes via ``monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")``.
os.environ.pop("PICARONES_PUBLIC_MODE", None)

# Rate limit désactivé en dev (déjà le défaut, explicité ici).
os.environ.setdefault("PICARONES_RATE_LIMIT_PER_HOUR", "0")
