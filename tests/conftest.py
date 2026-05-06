"""Configuration pytest globale.

Deux responsabilités, dans cet ordre :

1. **Ajouter le repo root à ``sys.path``** — garantit que
   ``tests.fixtures.*`` (mock adapters utilisés par les tests CLI
   E2E via dotted-path resolution ``importlib.import_module()``)
   sont importables de manière déterministe sur **tous les OS et
   versions Python**, indépendamment de la config ``pythonpath`` de
   pytest (qui peut diverger entre runners macOS/Windows/Linux et
   versions 3.11/3.12/3.13).

2. **Positionner les variables d'environnement test-friendly avant
   tout import de ``picarones.web.*``** — sinon les singletons web
   (``JOBS_SEMAPHORE``, ``RATE_LIMITER``) seraient instanciés avec
   les valeurs de production au premier import, et chaque test web
   verrait le bocal saturé.

L'isolation par-test des états globaux web (sémaphore, rate limiter,
browse roots) vit dans ``tests/web/conftest.py`` — fixture
``autouse=True`` qui ne s'applique qu'aux tests sous ``tests/web/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# (1) sys.path déterministe.  Le repo root contient le package
# ``picarones`` (déjà installable via ``pip install -e .``) ET le
# package ``tests`` (importable via ``tests.fixtures.X``).  On ajoute
# le repo root en tête pour garantir l'import déterministe sur tous
# les OS / versions Python.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# (2) Variables d'environnement.
# Plafond très large pour ne jamais bloquer une suite de tests qui
# démarre rapidement plusieurs benchmarks daemon en parallèle.
os.environ.setdefault("PICARONES_MAX_CONCURRENT_JOBS", "32")

# Mode dev par défaut. Les tests qui valident le mode public le
# forcent eux-mêmes via ``monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")``.
os.environ.pop("PICARONES_PUBLIC_MODE", None)

# Rate limit désactivé en dev (déjà le défaut, explicité ici).
os.environ.setdefault("PICARONES_RATE_LIMIT_PER_HOUR", "0")
