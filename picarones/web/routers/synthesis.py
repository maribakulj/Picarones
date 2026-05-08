"""Router de la synthèse narrative en preview.

Permet à un client d'obtenir la synthèse narrative d'un job terminé
sans devoir ouvrir le rapport HTML complet — utile pour un encart
*« le moteur narratif a-t-il quelque chose d'intéressant à dire ? »*
en tête de page.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from picarones.web import state

router = APIRouter()


@router.get("/api/benchmark/{job_id}/synthesis_preview")
async def api_benchmark_synthesis_preview(job_id: str, lang: str = "fr") -> dict:
    """Rend la synthèse narrative d'un job terminé sans rouvrir le HTML.

    Pipeline interne :

    1. Charge le ``BenchmarkJob`` (RAM ou DB).
    2. Lit le JSON de résultats associé via ``output_path``.
    3. Appelle ``build_synthesis()`` côté serveur.
    4. Retourne ``{sentences, facts, lang}``.

    Renvoie ``409 Conflict`` si le job n'est pas terminé, ``404`` si
    introuvable, ``422`` si le JSON associé est manquant ou cassé.

    La lecture du JSON et l'appel narrative sont délégués à un thread
    (``asyncio.to_thread``) pour ne pas bloquer l'event loop FastAPI
    sur l'I/O disque (rapports volumineux : plusieurs Mo).
    """
    if lang not in state.SUPPORTED_LANGS:
        lang = "fr"

    # Statut courant : RAM si dispo, sinon DB. Lookups rapides, pas de thread.
    ram_job = state.get_job_in_memory(job_id)
    db_job = state.JOB_STORE.get_job(job_id)
    if ram_job is None and db_job is None:
        raise HTTPException(status_code=404, detail=f"Job non trouvé : {job_id}")

    status = ram_job.status if ram_job is not None else db_job["status"]
    if status not in ("complete",):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Synthèse indisponible : statut courant = {status!r} "
                "(attendu 'complete')."
            ),
        )

    output_path = (
        ram_job.output_path if ram_job is not None
        else (db_job or {}).get("output_path", "")
    )
    if not output_path:
        raise HTTPException(status_code=422, detail="Aucun rapport produit pour ce job.")

    return await asyncio.to_thread(_build_synthesis_payload, job_id, lang, output_path)


def _build_synthesis_payload(job_id: str, lang: str, output_path: str) -> dict:
    """Localise le JSON associé au HTML, le lit, appelle ``build_synthesis``.

    Exécuté dans un thread — ``read_text`` peut bloquer plusieurs ms
    sur des rapports volumineux.
    """
    # Le HTML est à ``output_path`` ; le JSON associé est à côté
    # (convention ``picarones run -o results.json --output-html``).
    html_path = Path(output_path)
    json_candidates = [
        html_path.with_suffix(".json"),
        html_path.with_name(html_path.stem + "_results.json"),
        html_path.parent / "results.json",
    ]
    json_path: Optional[Path] = next(
        (p for p in json_candidates if p.exists()), None,
    )
    if json_path is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "JSON de résultats introuvable à côté du rapport HTML. "
                f"Cherché : {[str(p) for p in json_candidates]}"
            ),
        )

    try:
        report_json = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Lecture JSON échouée : {exc}")

    from picarones.reports_v2.narrative import build_synthesis

    synthesis = build_synthesis(report_json, lang=lang)
    return {
        "job_id": job_id,
        "lang": lang,
        "source_json": str(json_path),
        "sentences": synthesis.get("sentences", []),
        "facts": synthesis.get("facts", []),
    }
