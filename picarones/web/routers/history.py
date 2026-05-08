"""Router des régressions détectées dans l'historique longitudinal.

Surface de l'infrastructure ``BenchmarkHistory`` qui était
limitée au CLI ``picarones history --regression``. Le rapport HTML
peut désormais consommer cet endpoint pour afficher un encart
*« ⚠ Tesseract a régressé de 0,8 pp depuis le 12 janvier »* en tête.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()
_logger = logging.getLogger(__name__)


@router.get("/api/history/regressions")
async def api_history_regressions(
    engine: Optional[str] = Query(default=None, description="Filtre par moteur"),
    threshold: float = Query(default=0.01, description="Seuil régression CER absolu"),
    db_path: Optional[str] = Query(default=None, description="Chemin SQLite history"),
) -> dict:
    """Liste les régressions détectées dans l'historique longitudinal."""
    from picarones.evaluation.metrics.history import BenchmarkHistory

    try:
        history = BenchmarkHistory(db_path) if db_path else BenchmarkHistory()
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
            targets = sorted({e.engine for e in entries if e.engine})
        except Exception:  # noqa: BLE001
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
