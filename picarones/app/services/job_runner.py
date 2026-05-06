"""``JobRunner`` — pont entre l'API web et le ``RunOrchestrator``.

Le ``JobStore`` persiste l'état des jobs.  L'API web déclenche
l'exécution via ``POST /api/jobs``.  ``JobRunner`` orchestre le
cycle de vie complet :

1. Crée un ``JobRecord`` dans le ``JobStore`` (status ``pending``).
2. Lance un **thread daemon** qui exécute l'orchestrator de façon
   synchrone.
3. Met à jour le statut au fur et à mesure : ``running`` au démarrage,
   ``complete`` ou ``error`` à la fin.
4. Si le caller annule via ``DELETE /api/jobs/{id}`` (qui appelle
   ``store.mark_cancelled``), le thread l'observe au prochain check
   et abandonne — le résultat partiel est discardé.

Pourquoi un thread, pas asyncio
-------------------------------
``RunOrchestrator.execute`` est **synchrone** et utilise un
``ThreadPoolExecutor`` interne (``CorpusRunner``).  Le wrapper avec
asyncio créerait du complexité gratuite (mix sync/async, GIL).
Un ``threading.Thread(daemon=True)`` est l'outil correct ici.

Cancellation coopérative
------------------------
Pour S48, la cancellation est **best-effort** : le thread vérifie
``store.get(job_id).status == "cancelled"`` AVANT et APRÈS l'appel
à ``orchestrator.execute``.  Pendant l'exécution (potentiellement
plusieurs minutes), le thread ne peut pas interrompre l'orchestrator
sans support natif (cf. ``CorpusRunner.run(cancel_event=...)`` —
non encore propagé jusqu'à ``RunOrchestrator``).

Conséquence : ``DELETE /api/jobs/{id}`` pendant que le thread tourne
marque le statut comme ``cancelled``, mais le benchmark continue et
son résultat est discardé à la fin.  Une amélioration future
propagerait le ``cancel_event`` jusqu'au runner.

Anti-sur-ingénierie
-------------------
- Pas de queue de jobs avec backpressure : un thread par submit.
  Pour 100+ jobs simultanés, ajouter un ``ThreadPoolExecutor`` au
  niveau du runner.
- Pas de retry automatique sur échec.
- Pas de notification SSE des changements de statut (le caller
  poll ``GET /api/jobs/{id}``).
"""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from picarones.adapters.storage import JobStore

logger = logging.getLogger(__name__)


# Factory : un caller fournit un callable qui construit un
# ``RunOrchestrator`` lié à un ``output_dir`` donné.  L'inversion
# évite à ce module d'importer ``RunOrchestrator`` directement
# (cycles potentiels) et permet aux tests d'injecter un mock.
OrchestratorFactory = Callable[[Path], Any]
ReportRenderer = Callable[[Any, Path, str], Path]


class JobRunner:
    """Lance des jobs de benchmark en arrière-plan.

    Parameters
    ----------
    job_store:
        ``JobStore`` partagé avec les endpoints de lecture
        (``GET /api/jobs``, ``DELETE /api/jobs/{id}``).
    orchestrator_factory:
        Callable ``(output_dir: Path) -> RunOrchestrator`` qui
        construit un orchestrator par job.  Permet à chaque job
        d'avoir son propre output_dir isolé.
    report_renderer:
        Optionnel — passé à ``orchestrator.execute()`` pour rendre
        le rapport HTML.  Si ``None``, pas de rapport produit.

    Notes
    -----
    L'instance est thread-safe : ``submit`` est appelé depuis le
    thread FastAPI, le thread daemon écrit dans ``JobStore`` qui
    sérialise ses opérations SQLite.
    """

    def __init__(
        self,
        job_store: JobStore,
        orchestrator_factory: OrchestratorFactory,
        report_renderer: ReportRenderer | None = None,
    ) -> None:
        if not isinstance(job_store, JobStore):
            raise TypeError("job_store doit être un JobStore.")
        if not callable(orchestrator_factory):
            raise TypeError("orchestrator_factory doit être callable.")
        if report_renderer is not None and not callable(report_renderer):
            raise TypeError("report_renderer doit être callable ou None.")
        self._store = job_store
        self._factory = orchestrator_factory
        self._report_renderer = report_renderer
        # Tracking des threads actifs — utile pour les tests qui
        # attendent la fin d'un job soumis.
        self._threads: dict[str, threading.Thread] = {}

    # ──────────────────────────────────────────────────────────────────
    # API publique
    # ──────────────────────────────────────────────────────────────────

    def submit(
        self,
        run_spec: Any,
        output_dir: Path | str,
        *,
        job_id: str | None = None,
        payload: dict | None = None,
    ) -> str:
        """Crée un job et lance son exécution en thread arrière-plan.

        Returns
        -------
        str
            ``job_id`` (généré si non fourni).  Utilisable pour
            interroger ``GET /api/jobs/{job_id}``.

        Notes
        -----
        Idempotent uniquement si ``job_id`` est fourni explicitement
        (sinon UUID4 garantit l'unicité).  Si le ``job_id`` existe
        déjà, ``JobStore.create`` lève ``JobStoreError``.
        """
        job_id = job_id or uuid.uuid4().hex
        out_path = Path(output_dir)
        # ``payload`` est sérialisé en JSON dans le store — on stocke
        # la version du run_spec pour traçabilité.
        record_payload = dict(payload or {})
        record_payload.setdefault("output_dir", str(out_path))
        self._store.create(job_id, payload=record_payload)

        thread = threading.Thread(
            target=self._run,
            args=(job_id, run_spec, out_path),
            daemon=True,
            name=f"picarones-job-{job_id[:8]}",
        )
        self._threads[job_id] = thread
        thread.start()
        logger.info("[job_runner] job %s soumis (thread démarré).", job_id)
        return job_id

    def wait(self, job_id: str, timeout: float | None = None) -> bool:
        """Attend la fin du thread d'un job (utile aux tests).

        Returns
        -------
        bool
            ``True`` si le thread est terminé, ``False`` si timeout.
        """
        thread = self._threads.get(job_id)
        if thread is None:
            return True  # job inconnu = considéré fini
        thread.join(timeout=timeout)
        return not thread.is_alive()

    # ──────────────────────────────────────────────────────────────────
    # Worker thread
    # ──────────────────────────────────────────────────────────────────

    def _run(
        self,
        job_id: str,
        run_spec: Any,
        output_dir: Path,
    ) -> None:
        """Logique exécutée dans le thread daemon.  Capture toutes les
        exceptions et les transcrit en statut ``error`` du store.

        Hooks de cancellation coopérative :

        - **Avant** ``orchestrator.execute()`` : si le statut a été
          basculé en ``cancelled`` entre le ``submit`` et le démarrage
          du thread, on saute l'exécution.
        - **Après** ``orchestrator.execute()`` : si le statut a été
          basculé en ``cancelled`` pendant l'exécution, on discarde
          le résultat (le statut reste ``cancelled``).

        Sinon, statut final = ``complete`` ou ``error``.
        """
        # 1. Check pré-démarrage : annulé avant que le thread n'ait
        #    pris la main ?
        rec = self._store.get(job_id)
        if rec is None:
            logger.warning(
                "[job_runner] job %s introuvable au démarrage du "
                "thread — abandon.", job_id,
            )
            return
        if rec.status == "cancelled":
            logger.info(
                "[job_runner] job %s annulé avant démarrage — skip.",
                job_id,
            )
            return

        # 2. Marquer en cours.
        try:
            self._store.mark_running(job_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[job_runner] échec mark_running sur %s : %s — abandon.",
                job_id, exc,
            )
            return

        # 3. Exécution effective.
        try:
            orchestrator = self._factory(output_dir)
            result = orchestrator.execute(
                run_spec,
                report_renderer=self._report_renderer,
            )
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "[job_runner] job %s en échec : %s",
                job_id, error_msg,
            )
            self._store.mark_error(job_id, error_msg)
            return

        # 4. Check post-exécution : annulé pendant que le run tournait ?
        rec_after = self._store.get(job_id)
        if rec_after is not None and rec_after.status == "cancelled":
            logger.info(
                "[job_runner] job %s annulé pendant l'exécution — "
                "résultat discardé.", job_id,
            )
            return

        # 5. Succès — output_path = chemin du manifest persisté.
        manifest_path = result.persisted_files.get("manifest")
        output_path_str = str(manifest_path) if manifest_path else ""
        self._store.mark_complete(job_id, output_path=output_path_str)
        logger.info("[job_runner] job %s terminé avec succès.", job_id)


__all__ = ["JobRunner", "OrchestratorFactory", "ReportRenderer"]
