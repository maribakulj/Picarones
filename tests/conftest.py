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


# (3) Instrumentation temporaire — investigation du flake CI Python 3.12 ubuntu.
#
# Stack trace observée systématiquement à la fin de la suite :
#
#     File "concurrent/futures/process.py", line 587 in _join_executor_internals
#     File "concurrent/futures/process.py", line 106 in _python_exit
#     File "threading.py", line 1594 in _shutdown
#     Error: Process completed with exit code 124.
#
# Le ``_python_exit`` de ``concurrent.futures.process`` essaie de joindre des
# workers de ``ProcessPoolExecutor`` qui n'ont pas été shutdown.  Mais nous
# ne savons pas QUEL test (ou quelle dépendance tierce) instancie ce pool —
# aucun test du repo n'utilise explicitement ``execution_mode="cpu"``.
#
# Ce patch loggue chaque création de ``ProcessPoolExecutor`` avec sa stack
# trace pour identifier la source.  À retirer une fois la cause trouvée.
import sys as _sys
import traceback as _traceback
from concurrent.futures import process as _futures_process

_PROCESS_POOL_CREATIONS: list[str] = []


_original_pool_init = _futures_process.ProcessPoolExecutor.__init__


def _logged_pool_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    stack = _traceback.format_stack()
    record = (
        f"\n[conftest:investigate] ProcessPoolExecutor instancié "
        f"(args={args}, kwargs={kwargs}) — stack :\n"
        + "".join(stack[-12:])  # 12 derniers frames suffisent pour identifier
    )
    _PROCESS_POOL_CREATIONS.append(record)
    _sys.stderr.write(record)
    _sys.stderr.flush()
    return _original_pool_init(self, *args, **kwargs)


_futures_process.ProcessPoolExecutor.__init__ = _logged_pool_init


# (4) Désactivation préventive du thread daemon de tqdm.
# Sur Python 3.12+ (ubuntu-latest en CI), le combo
# ``tqdm._monitor`` + ``ProcessPoolExecutor`` (utilisé par
# ``picarones.measurements.runner.orchestration`` pour les moteurs
# CPU-bound : Tesseract, Pero OCR) provoque un hang du shutdown de
# l'interpréteur après ``=== passed ===``.  Le ``_python_exit`` de
# ``concurrent.futures.process`` essaie de joindre les workers du
# pool, mais le thread monitor de tqdm bloque la sortie globale —
# le hang dépasse le timeout GNU configuré dans ci.yml (9 min) et
# le job échoue avec exit code 124.
#
# ``monitor_interval=0`` désactive le polling thread de tqdm, qui
# n'est utile qu'à l'affichage interactif des progress bars (sans
# valeur ajoutée en CI où stdout est captured).  Fix idiomatique
# pour ce flake spécifique.
try:
    from tqdm import tqdm as _tqdm

    _tqdm.monitor_interval = 0
except ImportError:  # pragma: no cover
    # tqdm est une dep de prod (cf. pyproject.toml) ; cette branche
    # ne devrait jamais être atteinte en CI mais reste défensive.
    pass


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ARG001
    """Diagnostic du shutdown de l'interpréteur.

    Sur Python 3.12 ubuntu-latest, l'interpréteur restait jusqu'à 12
    minutes en hang après ``=== passed ===`` à cause de threads
    non-daemon ou de connexions sqlite non fermées que les tests
    avaient laissés.

    Ce hook :

    1. Liste les threads vivants à la fin de la session — si la
       liste contient autre chose que ``MainThread``, le développeur
       voit immédiatement quelle ressource fuit.
    2. Force le flush stdout/stderr pour que le diagnostic apparaisse
       même si l'interpréteur hang ensuite.
    3. Programme un ``faulthandler.dump_traceback_later(60)`` qui
       dumpera les stack traces de TOUS les threads après 60s
       d'inactivité — ce qu'on a besoin pour identifier la fuite si
       le hang persiste.
    """
    import faulthandler
    import sys
    import threading

    alive = [
        t for t in threading.enumerate()
        if t is not threading.main_thread() and t.is_alive()
    ]
    if alive:
        sys.stderr.write(
            "\n[conftest] threads encore vivants au sessionfinish "
            f"({len(alive)}) :\n",
        )
        for t in alive:
            sys.stderr.write(
                f"  - name={t.name!r} daemon={t.daemon} "
                f"alive={t.is_alive()}\n",
            )
        sys.stderr.flush()

    # Récap des créations de ProcessPoolExecutor capturées par le
    # patch (3) ci-dessus — utile en CI pour voir d'un coup d'œil
    # combien de pools ont été créés et qui est responsable.
    if _PROCESS_POOL_CREATIONS:
        sys.stderr.write(
            f"\n[conftest:investigate] {len(_PROCESS_POOL_CREATIONS)} "
            f"ProcessPoolExecutor créés pendant la session.\n"
            f"Stack traces complètes ci-dessus dans la sortie pytest.\n",
        )
        sys.stderr.flush()

    # Si le shutdown hang plus de 60s, on aura les stack traces.
    faulthandler.dump_traceback_later(
        timeout=60,
        repeat=False,
        file=sys.stderr,
    )
