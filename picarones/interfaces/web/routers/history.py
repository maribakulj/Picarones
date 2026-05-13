"""Router des régressions détectées dans l'historique longitudinal.

Surface de l'infrastructure ``BenchmarkHistory`` qui était
limitée au CLI ``picarones history --regression``. Le rapport HTML
peut désormais consommer cet endpoint pour afficher un encart
*« ⚠ Tesseract a régressé de 0,8 pp depuis le 12 janvier »* en tête.

Sécurité — paramètre ``db_path``
---------------------------------
Le paramètre ``db_path`` est validé contre les racines workspace
autorisées via :func:`validated_path`. Sans ce garde-fou, l'endpoint
acceptait un chemin SQLite libre — vecteur de lecture filesystem
arbitraire (path traversal).  Pour pointer une base alternative à
l'extérieur des workspaces, exporter ``PICARONES_HISTORY_DB`` plutôt
que de passer ``db_path`` par query string.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()
_logger = logging.getLogger(__name__)


@router.get("/api/history/regressions")
async def api_history_regressions(
    engine: Optional[str] = Query(default=None, description="Filtre par moteur"),
    threshold: float = Query(default=0.01, description="Seuil régression CER absolu"),
    db_path: Optional[str] = Query(
        default=None,
        description=(
            "Chemin SQLite history (validé contre les workspace roots ; "
            "préférer la variable d'env PICARONES_HISTORY_DB)."
        ),
    ),
) -> dict:
    """Liste les régressions détectées dans l'historique longitudinal."""
    from picarones.evaluation.metrics.history import BenchmarkHistory

    if db_path:
        # Phase 7.2 audit code-quality : helper centralisé pour la
        # validation chemin → HTTPException 400.
        from picarones.interfaces.web._path_helpers import validated_user_path

        effective_db_path: Optional[str] = str(
            validated_user_path(db_path, must_exist=False),
        )
    else:
        env_db = os.environ.get("PICARONES_HISTORY_DB", "").strip()
        effective_db_path = env_db or None

    try:
        history = (
            BenchmarkHistory(effective_db_path)
            if effective_db_path
            else BenchmarkHistory()
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Ouverture historique échouée : {exc}",
        )

    # Si aucun moteur n'est passé, on liste tous les moteurs présents
    # dans l'historique et on tente une détection sur chacun.
    if engine:
        targets = [engine]
    else:
        try:
            entries = history.query(limit=10000)
            # Sprint S4 — fix : ``HistoryEntry`` expose
            # ``engine_name``, pas ``engine`` (typo masquée par
            # l'``except`` générique).  Avant ce fix, l'endpoint
            # sans param ``engine`` retournait toujours 0
            # régression — bug silencieux découvert par les tests
            # ``test_s4_history_router.py``.
            targets = sorted(
                {e.engine_name for e in entries if e.engine_name}
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "[regressions] énumération des moteurs échouée : %s", exc,
            )
            targets = []

    out: list[dict[str, Any]] = []
    for eng in targets:
        try:
            res = history.detect_regression(engine=eng, threshold=threshold)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "[regressions] detect_regression(%s) échoué : %s", eng, exc,
            )
            continue
        if res is None:
            continue
        d = {
            "engine": eng,
            "is_regression": getattr(res, "is_regression", False),
            "delta_cer": getattr(res, "delta_cer", None),
            "current_cer": getattr(res, "current_cer", None),
            "baseline_cer": getattr(res, "baseline_cer", None),
            "current_run_id": getattr(res, "current_run_id", None),
            "baseline_run_id": getattr(res, "baseline_run_id", None),
        }
        if d["is_regression"]:
            out.append(d)

    return {
        "threshold": float(threshold),
        "regressions": out,
        "count": len(out),
    }
