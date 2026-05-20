"""Reprise sur interruption pour ``run_benchmark_via_service``.

Persistance NDJSON des ``DocumentResult`` au fil du benchmark, pour
permettre la reprise après crash / Ctrl+C / timeout sans perdre le
travail déjà fait.

Contrat
-------
Pour chaque couple ``(corpus_name, engine_name)``, un fichier
``{partial_dir}/picarones_{corpus}_{engine}_{fingerprint}.partial.jsonl``
accumule une ligne JSON par ``DocumentResult`` au fur et à mesure de
leur calcul.  Au redémarrage, ``run_benchmark_via_service`` charge ce
fichier, identifie les ``doc_id`` déjà traités, et n'invoque le
``BenchmarkService`` que sur les documents restants.

Quand un engine a été traité en entier sans erreur, son fichier
partiel est supprimé.  Si un crash interrompt le run mid-engine,
le fichier persiste : la prochaine exécution reprendra exactement
où l'on s'est arrêté.

Fingerprint anti-collision
---------------------------------------
Auparavant, la clé partial était ``(corpus.name, engine.name)`` —
insuffisant : deux runs successifs avec le même corpus et le même
engine **mais des configs différentes** (psm Tesseract, langue,
profil de normalisation, char_exclude, version code) réutilisaient
silencieusement les résultats du run précédent.  Reproductibilité
scientifique cassée.

Désormais :func:`compute_run_fingerprint` calcule un SHA-256 stable
de la config complète (engine_config, normalization_profile,
char_exclude, fichiers du corpus + mtime/size, version code).  Le
préfixe 16 hex est suffixé au nom du fichier partiel : un changement
de config = un fichier différent = pas de réutilisation illégale.

Anti-sur-ingénierie
-------------------
- Format JSONL plat (une ligne = un ``DocumentResult.as_dict()``),
  pas de schéma versioné.  Si la structure du ``DocumentResult``
  change, le fichier devient illisible — l'opérateur supprime
  ``partial_dir`` et relance.
- Lock thread-safe partagé module-level ; pas de tentative de
  partage inter-process (chaque process a son propre tempdir).
- Pas de checksum ni de validation de schéma — best-effort.  Une
  ligne corrompue = warning + ligne ignorée + on continue.
- Fingerprint basé sur ``(path, size, mtime)`` pour les fichiers
  corpus, pas sur le contenu lui-même : 100× plus rapide, suffisant
  pour détecter une modification.  Si un attaquant ``touch`` un
  fichier sans changer son contenu, le partial est invalidé (acceptable,
  conservative).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import threading
from collections.abc import Iterable, Mapping
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

    Permet à un opérateur de retrouver visuellement le fichier
    dans ``partial_dir``.
    """
    return re.sub(r"[^\w\-]", "_", s)[:64]


def _partial_path(
    corpus_name: str,
    engine_name: str,
    partial_dir: Optional[str | Path],
    *,
    fingerprint: Optional[str] = None,
) -> Path:
    """Construit le chemin du fichier partiel pour ``(corpus, engine)``.

    Si ``partial_dir`` est ``None``, on tombe dans
    ``tempfile.gettempdir()`` — utile pour les tests qui ne veulent
    pas configurer un répertoire dédié mais bénéficient quand même
    de la reprise intra-process.

    Si ``fingerprint`` est fourni, il est suffixé au nom :
    ``picarones_{corpus}_{engine}_{fingerprint}.partial.jsonl``.  Cela
    garantit que deux runs avec le même couple ``(corpus, engine)``
    mais des configs différentes ne partagent **jamais** leur fichier
    partiel.  Sans ``fingerprint``, le comportement legacy est
    préservé pour rétrocompatibilité tests.
    """
    base = Path(partial_dir) if partial_dir else Path(tempfile.gettempdir())
    fp_suffix = f"_{fingerprint}" if fingerprint else ""
    name = (
        f"picarones_{_sanitize_filename(corpus_name)}"
        f"_{_sanitize_filename(engine_name)}{fp_suffix}.partial.jsonl"
    )
    return base / name


def compute_run_fingerprint(
    *,
    engine_config: Mapping[str, Any] | None = None,
    normalization_profile: str | None = None,
    char_exclude: str | None = None,
    corpus_files: Iterable[str | Path] | None = None,
    code_version: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Calcule un fingerprint stable pour identifier un run.

    Composantes intégrées au hash :

    - ``engine_config`` — dict de paramètres moteur (lang, psm,
      model, etc.).  Encodé en JSON trié pour stabilité.
    - ``normalization_profile`` — identifiant du profil de
      normalisation Unicode.  Différents profils → métriques
      différentes → fingerprint différent.
    - ``char_exclude`` — caractères ignorés au calcul (CER/WER).
      Idem.
    - ``corpus_files`` — itérable de chemins.  Pour chaque, on
      hashe le chemin + ``stat.st_size`` + ``stat.st_mtime``.
      Détecte les modifs sans coût du hash de contenu.
    - ``code_version`` — version de Picarones courante.
    - ``extra`` — dict additionnel libre pour des éléments
      spécifiques à un pipeline (prompt_template, llm_params).

    Returns
    -------
    str
        Empreinte hexadécimale tronquée à 16 caractères — collision
        négligeable pour un usage par-utilisateur, lisible humainement.
    """
    hasher = hashlib.sha256()

    def _update(key: str, value: Any) -> None:
        hasher.update(b"\x00")
        hasher.update(key.encode("utf-8"))
        hasher.update(b"\x01")
        try:
            payload = json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            payload = repr(value)
        hasher.update(payload.encode("utf-8"))

    _update("engine_config", dict(engine_config or {}))
    _update("normalization_profile", normalization_profile or "")
    _update("char_exclude", char_exclude or "")
    _update("code_version", code_version or "")
    _update("extra", dict(extra or {}))

    if corpus_files is not None:
        # Tri pour stabilité indépendamment de l'ordre d'itération.
        for fpath in sorted(str(p) for p in corpus_files):
            hasher.update(b"\x02")
            hasher.update(fpath.encode("utf-8"))
            try:
                stat = Path(fpath).stat()
                hasher.update(
                    f":{stat.st_size}:{int(stat.st_mtime)}".encode("utf-8"),
                )
            except OSError:
                # Fichier disparu / inaccessible — ignoré au fingerprint.
                # Si le file disparait pendant la course, on prend ce
                # qu'on peut.
                continue

    return hasher.hexdigest()[:16]


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
            # Utilise ``DocumentResult.from_dict`` au lieu
            # de la reconstruction manuelle qui perdait
            # ``taxonomy``/``ner_metrics``/``calibration_metrics``/etc.
            # à la reprise — un partial chargé puis re-sérialisé
            # devait conserver l'intégralité du payload.
            results.append(DocumentResult.from_dict(d))
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


def partial_path_for_engine(
    *,
    corpus: Any,
    engine: Any,
    partial_dir: Optional[str | Path],
    engine_config: Mapping[str, Any] | None = None,
    normalization_profile: Any | None = None,
    char_exclude: Any | None = None,
    profile: str | None = None,
    code_version: str | None = None,
) -> Path:
    """Helper public qui calcule le ``Path`` du fichier partiel pour
    un couple ``(corpus, engine)`` en intégrant le fingerprint complet.

    Encapsule la combinaison ``_partial_path`` +
    :func:`compute_run_fingerprint` pour que le runner et les tests
    utilisent la **même** logique de nommage — sinon les tests ne
    peuvent pas pré-remplir un partial que le runner saura
    retrouver.

    Parameters
    ----------
    corpus:
        Doit exposer ``.name`` et ``.documents`` (chaque doc ayant
        ``.image_path``).
    engine:
        Doit exposer ``.name``.  ``engine_config`` peut être fourni
        séparément si la caller veut surcharger l'introspection.
    partial_dir:
        Dossier où vit le partial ; ``None`` → tempdir.
    engine_config:
        Si fourni, utilisé tel quel ; sinon l'appelant peut sonder
        l'engine via :func:`benchmark_runner._engine_config_for_fingerprint`
        avant d'appeler.
    normalization_profile, char_exclude, profile, code_version:
        Composantes incluses dans le fingerprint.  Passer ``None``
        pour ne pas contribuer (deux runs avec et sans normalisation
        auront alors des fingerprints différents seulement si l'un
        des deux est ``None``).
    """
    corpus_files = [
        doc.image_path for doc in getattr(corpus, "documents", [])
        if getattr(doc, "image_path", None)
    ]
    fp = compute_run_fingerprint(
        engine_config=engine_config or {"engine_name": getattr(engine, "name", "")},
        normalization_profile=(
            getattr(normalization_profile, "name", None)
            if normalization_profile is not None
            else None
        ),
        char_exclude=(
            "".join(sorted(char_exclude)) if char_exclude else None
        ),
        corpus_files=corpus_files,
        code_version=code_version,
        extra={"profile": profile} if profile else None,
    )
    return _partial_path(
        getattr(corpus, "name", ""), getattr(engine, "name", ""),
        partial_dir, fingerprint=fp,
    )


__all__ = [
    "_delete_partial",
    "_load_partial",
    "_partial_path",
    "_partial_write_lock",
    "_sanitize_filename",
    "_save_partial_line",
    "compute_run_fingerprint",
    "partial_path_for_engine",
]
