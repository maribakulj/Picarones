"""Persistance des résultats partiels du benchmark (NDJSON).

Quand le runner traite un corpus, il écrit chaque ``DocumentResult``
dans un fichier ``{partial_dir}/picarones_{corpus}_{engine}.partial.json``
au format NDJSON. Si le benchmark est interrompu (Ctrl+C, crash, kill),
la prochaine exécution reprend depuis ce fichier sans perdre le travail
déjà fait.

Thread-safe : le module utilise un :class:`threading.Lock` partagé
entre toutes les écritures pour sérialiser les appends.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import threading
from pathlib import Path
from typing import Optional

from picarones.core.results import DocumentResult
from picarones.measurements.metrics import MetricsResult

logger = logging.getLogger(__name__)

# Lock pour la sérialisation des écritures de résultats partiels.
# Partagé entre tous les call sites (workers IO et CPU se relayent
# sur la même file).
_partial_write_lock = threading.Lock()


def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)[:64]


def _partial_path(
    corpus_name: str,
    engine_name: str,
    partial_dir: Optional[str | Path],
) -> Path:
    base = Path(partial_dir) if partial_dir else Path(tempfile.gettempdir())
    name = (
        f"picarones_{_sanitize_filename(corpus_name)}"
        f"_{_sanitize_filename(engine_name)}.partial.json"
    )
    return base / name


def _load_partial(
    corpus_name: str,
    engine_name: str,
    partial_dir: Optional[str | Path],
) -> tuple[Path, list[DocumentResult]]:
    """Charge les résultats partiels d'une exécution précédente interrompue.

    Returns
    -------
    (path, results) — chemin du fichier partiel et liste des
    DocumentResult déjà calculés.
    """
    path = _partial_path(corpus_name, engine_name, partial_dir)
    results: list[DocumentResult] = []
    if not path.exists():
        return path, results

    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                m = d.get("metrics", {})
                metrics = MetricsResult(
                    cer=m.get("cer", 1.0),
                    cer_nfc=m.get("cer_nfc", 1.0),
                    cer_caseless=m.get("cer_caseless", 1.0),
                    wer=m.get("wer", 1.0),
                    wer_normalized=m.get("wer_normalized", 1.0),
                    mer=m.get("mer", 1.0),
                    wil=m.get("wil", 1.0),
                    reference_length=m.get("reference_length", 0),
                    hypothesis_length=m.get("hypothesis_length", 0),
                    error=m.get("error"),
                )
                results.append(DocumentResult(
                    doc_id=d["doc_id"],
                    image_path=d.get("image_path", ""),
                    ground_truth=d.get("ground_truth", ""),
                    hypothesis=d.get("hypothesis", ""),
                    metrics=metrics,
                    duration_seconds=d.get("duration_seconds", 0.0),
                    engine_error=d.get("engine_error"),
                    ocr_intermediate=d.get("ocr_intermediate"),
                    pipeline_metadata=d.get("pipeline_metadata", {}),
                    confusion_matrix=d.get("confusion_matrix"),
                    char_scores=d.get("char_scores"),
                    taxonomy=d.get("taxonomy"),
                    structure=d.get("structure"),
                    image_quality=d.get("image_quality"),
                    line_metrics=d.get("line_metrics"),
                    hallucination_metrics=d.get("hallucination_metrics"),
                ))
    except Exception as e:
        logger.warning("Impossible de charger les résultats partiels '%s' : %s", path, e)
        results = []

    return path, results


def _save_partial_line(partial_path: Path, doc_result: DocumentResult) -> None:
    """Ajoute une entrée NDJSON au fichier de résultats partiels (thread-safe)."""
    try:
        line = json.dumps(doc_result.as_dict(), ensure_ascii=False) + "\n"
        with _partial_write_lock:
            with partial_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception as e:
        logger.warning("Impossible d'écrire dans le fichier partiel '%s' : %s", partial_path, e)


def _delete_partial(partial_path: Path) -> None:
    """Supprime le fichier de résultats partiels à la fin d'un moteur."""
    try:
        if partial_path.exists():
            partial_path.unlink()
    except Exception as e:
        logger.warning("Impossible de supprimer le fichier partiel '%s' : %s", partial_path, e)


__all__ = [
    "_delete_partial",
    "_load_partial",
    "_partial_path",
    "_partial_write_lock",
    "_sanitize_filename",
    "_save_partial_line",
]
