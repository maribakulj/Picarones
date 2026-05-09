"""Sprint D.2.b — reprise sur interruption pour ``run_benchmark_via_service``.

Persistance NDJSON des ``DocumentResult`` legacy au fil du
benchmark, pour permettre la reprise après crash / Ctrl+C / timeout
sans perdre le travail déjà fait.

Contrat
-------
Pour chaque couple ``(corpus_name, engine_name)``, un fichier
``{partial_dir}/picarones_{corpus}_{engine}.partial.jsonl`` accumule
une ligne JSON par ``DocumentResult`` au fur et à mesure de leur
calcul.  Au redémarrage, ``run_benchmark_via_service`` charge ce
fichier, identifie les ``doc_id`` déjà traités, et n'invoque le
``BenchmarkService`` que sur les documents restants.

Quand un engine a été traité en entier sans erreur, son fichier
partiel est supprimé.  Si un crash interrompt le run mid-engine,
le fichier persiste : la prochaine exécution reprendra exactement
où l'on s'est arrêté.

Trace de retrait
----------------
Module transitoire (Sprint D.2.b du plan v2.0).  Sera supprimé
en H.4 quand ``run_benchmark_via_service`` lui-même disparaîtra
au profit d'une consommation directe de ``BenchmarkService`` par
les callers (``cli``, ``web``).

Anti-sur-ingénierie
-------------------
- Format JSONL plat (une ligne = un ``DocumentResult.as_dict()``),
  pas de schéma versioné.  Si la structure du ``DocumentResult``
  legacy change, le fichier devient illisible — mais à ce stade
  on est déjà en post-rewrite v2.0+ et le legacy est mort.
- Lock thread-safe partagé module-level ; pas de tentative de
  partage inter-process (chaque process a son propre tempdir).
- Pas de checksum ni de validation de schéma — best-effort.  Une
  ligne corrompue = warning + ligne ignorée + on continue.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from picarones.evaluation.benchmark_result import DocumentResult

logger = logging.getLogger(__name__)

# Lock module-level pour sérialiser les appends NDJSON depuis
# plusieurs threads (workers IO/CPU du ``CorpusRunner``).  Un seul
# fichier sera écrit à la fois — c'est un goulot, mais l'écriture
# d'une ligne JSON est typiquement <1 ms, négligeable face au
# coût d'un OCR (100 ms - 5 s/doc).
_partial_write_lock = threading.Lock()


def _sanitize_filename(s: str) -> str:
    """Réduit ``s`` à ``[\\w\\-]`` et tronque à 64 chars.

    Cohérent avec le format historique du fichier partiel
    legacy ; permet à un opérateur de retrouver visuellement
    le fichier dans ``partial_dir``.
    """
    return re.sub(r"[^\w\-]", "_", s)[:64]


def _partial_path(
    corpus_name: str,
    engine_name: str,
    partial_dir: Optional[str | Path],
) -> Path:
    """Construit le chemin du fichier partiel pour ``(corpus, engine)``.

    Si ``partial_dir`` est ``None``, on tombe dans
    ``tempfile.gettempdir()`` — utile pour les tests qui ne veulent
    pas configurer un répertoire dédié mais bénéficient quand même
    de la reprise intra-process.
    """
    base = Path(partial_dir) if partial_dir else Path(tempfile.gettempdir())
    name = (
        f"picarones_{_sanitize_filename(corpus_name)}"
        f"_{_sanitize_filename(engine_name)}.partial.jsonl"
    )
    return base / name


def _load_partial(
    partial_path: Path,
) -> list[DocumentResult]:
    """Charge les ``DocumentResult`` déjà persistés à ``partial_path``.

    Retourne une liste vide si :
    - le fichier n'existe pas (premier run),
    - le fichier est illisible (warning loggué).

    Les lignes corrompues individuelles sont ignorées avec un
    warning ; les lignes valides sont conservées.  Cette
    tolérance évite qu'une ligne tronquée à la fin (typique
    d'un crash en cours d'écriture) ne fasse perdre tout le
    travail antérieur.
    """
    from picarones.evaluation.benchmark_result import DocumentResult
    from picarones.evaluation.metric_result import MetricsResult

    results: list[DocumentResult] = []
    if not partial_path.exists():
        return results

    try:
        with partial_path.open("r", encoding="utf-8") as fh:
            lines = list(fh)
    except OSError as exc:
        logger.warning(
            "[partial_dir] fichier '%s' illisible : %s — "
            "reprise désactivée pour cet engine.",
            partial_path, exc,
        )
        return results

    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning(
                "[partial_dir] ligne %d corrompue dans '%s' : %s "
                "— ignorée.", lineno, partial_path, exc,
            )
            continue
        try:
            metrics_dict = d.get("metrics", {}) or {}
            metrics = MetricsResult(
                cer=metrics_dict.get("cer"),
                cer_nfc=metrics_dict.get("cer_nfc"),
                cer_caseless=metrics_dict.get("cer_caseless"),
                wer=metrics_dict.get("wer"),
                wer_normalized=metrics_dict.get("wer_normalized"),
                mer=metrics_dict.get("mer"),
                wil=metrics_dict.get("wil"),
                reference_length=metrics_dict.get("reference_length", 0),
                hypothesis_length=metrics_dict.get("hypothesis_length", 0),
                error=metrics_dict.get("error"),
                cer_diplomatic=metrics_dict.get("cer_diplomatic"),
                diplomatic_profile_name=metrics_dict.get(
                    "diplomatic_profile_name",
                ),
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
                pipeline_metadata=d.get("pipeline_metadata", {}) or {},
                confusion_matrix=d.get("confusion_matrix"),
                char_scores=d.get("char_scores"),
                taxonomy=d.get("taxonomy"),
                structure=d.get("structure"),
                image_quality=d.get("image_quality"),
                line_metrics=d.get("line_metrics"),
                hallucination_metrics=d.get("hallucination_metrics"),
            ))
        except (KeyError, TypeError) as exc:
            logger.warning(
                "[partial_dir] ligne %d malformée dans '%s' : %s "
                "— ignorée.", lineno, partial_path, exc,
            )

    return results


def _save_partial_line(
    partial_path: Path, doc_result: Any,
) -> None:
    """Ajoute une ligne NDJSON pour ``doc_result`` (thread-safe).

    Crée ``partial_path.parent`` si nécessaire.  Toute erreur
    d'écriture est loggée mais non fatale : on ne veut pas qu'un
    problème de partial_dir (disque plein, permissions) fasse
    crasher un benchmark qui aurait sinon abouti.
    """
    try:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(doc_result.as_dict(), ensure_ascii=False) + "\n"
        with _partial_write_lock:
            with partial_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
    except OSError as exc:
        logger.warning(
            "[partial_dir] impossible d'écrire dans '%s' : %s",
            partial_path, exc,
        )


def _delete_partial(partial_path: Path) -> None:
    """Supprime ``partial_path`` à la fin d'un engine traité avec succès.

    L'absence de partial signale au prochain run qu'il n'y a pas
    de reprise à effectuer pour cet engine — le bench peut
    repartir de zéro proprement.
    """
    try:
        if partial_path.exists():
            partial_path.unlink()
    except OSError as exc:
        logger.warning(
            "[partial_dir] impossible de supprimer '%s' : %s",
            partial_path, exc,
        )


__all__ = [
    "_delete_partial",
    "_load_partial",
    "_partial_path",
    "_partial_write_lock",
    "_sanitize_filename",
    "_save_partial_line",
]
