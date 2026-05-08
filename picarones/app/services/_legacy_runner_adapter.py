"""Sprint D.1 du plan v2.0 — adapter de compat ``run_benchmark`` legacy
→ ``BenchmarkService`` rewrite.

Ce module présente l'API mono-call historique de
``picarones.measurements.runner.run_benchmark`` mais s'appuie en
interne sur le rewrite (``BenchmarkService``,
``PipelineExecutor``, ``CorpusRunner``).  Il sert de pont
transitoire pour faciliter la migration des callers en plusieurs
étapes :

1. (cette session) Helpers de mapping ``Corpus`` ↔ ``CorpusSpec``
   et ``Document`` ↔ ``DocumentRef`` — testables indépendamment.
2. (sub-phase D.1.b) Mapping ``BaseOCREngine`` → ``PipelineSpec``
   + adapter resolver.
3. (sub-phase D.1.c) Conversion ``RunResult`` → ``BenchmarkResult``.
4. (sub-phase D.1.d) Fonction ``run_benchmark_via_service``
   complète avec progress callback, output_json, partial_dir.
5. (sub-phase D.1.e) Tests d'équivalence numérique (CER/WER) entre
   les deux runners sur les fixtures.

Trace de retrait
----------------
Ce module est **transitoire** (Sprint D du plan v2.0).  Il sera
supprimé en D.6 quand tous les callers (cli/_workflows,
web/benchmark_utils) consommeront ``BenchmarkService``
directement.

Cette première itération n'expose que les helpers de mapping
documents/corpus — la fonction publique
``run_benchmark_via_service`` arrive dans une session ultérieure
quand toutes les briques seront en place.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from picarones.domain.artifacts import ArtifactType
from picarones.domain.corpus import CorpusSpec
from picarones.domain.documents import DocumentRef, GroundTruthRef
from picarones.domain.errors import PicaronesError

if TYPE_CHECKING:
    from picarones.evaluation.corpus import Corpus, Document


# ──────────────────────────────────────────────────────────────────────
# Mapping Document (legacy) → DocumentRef (rewrite)
# ──────────────────────────────────────────────────────────────────────


def document_to_document_ref(
    document: "Document",
    *,
    workspace_dir: Path,
) -> DocumentRef:
    """Convertit un ``Document`` legacy en ``DocumentRef`` rewrite.

    Le ``Document`` legacy porte sa GT en mémoire (``ground_truth: str``
    et ``ground_truths: dict[ArtifactType, GTPayload]``).  Le
    ``DocumentRef`` rewrite porte des références filesystem
    (``GroundTruthRef.uri``).  La conversion écrit chaque GT
    in-memory dans ``workspace_dir`` et construit les références.

    Parameters
    ----------
    document:
        Document legacy.  ``image_path`` non-``None`` est requis ;
        ``ground_truth`` (TEXT) peut être vide.
    workspace_dir:
        Répertoire de travail où écrire les fichiers GT
        synthétisés.  Doit exister et être writable.

    Returns
    -------
    DocumentRef
        Référence canonique avec ``id``, ``image_uri`` et un tuple
        ordonné de ``GroundTruthRef`` (par niveau ArtifactType).

    Raises
    ------
    PicaronesError
        Si ``document.doc_id`` ne respecte pas le regex
        ``DocumentRef._DOC_ID_RE`` (fallback explicite si besoin).
    """
    if not workspace_dir.exists():
        raise PicaronesError(
            f"workspace_dir doit exister : {workspace_dir!r}"
        )

    doc_id = _safe_doc_id(document.doc_id)

    image_uri: str | None = None
    if document.image_path is not None:
        image_uri = str(document.image_path)

    ground_truths: list[GroundTruthRef] = []

    # Niveau TEXT : ``ground_truth: str`` → fichier .gt.txt dans
    # workspace.  On écrit toujours, même vide, pour préserver le
    # contrat (un caller qui lit le fichier obtient la chaîne vide
    # et le runner sait gérer ce cas en métriques).
    if document.ground_truth or _has_text_gt(document):
        text_content = document.ground_truth
        if not text_content and _has_text_gt(document):
            # Le payload est dans ``ground_truths[RAW_TEXT]``.
            from picarones.evaluation.corpus import TextGT

            payload = document.ground_truths.get(ArtifactType.RAW_TEXT)
            if isinstance(payload, TextGT):
                text_content = payload.text

        text_path = workspace_dir / f"{doc_id.replace('/', '_')}.gt.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text_content, encoding="utf-8")
        ground_truths.append(
            GroundTruthRef(type=ArtifactType.RAW_TEXT, uri=str(text_path)),
        )

    # Niveaux étendus (ALTO, PAGE, ENTITIES, READING_ORDER) :
    # déjà sérialisés via leur ``source_path`` quand disponibles.
    # On préfère le ``source_path`` original au lieu d'une copie
    # pour ne pas dupliquer.
    for level in (
        ArtifactType.ALTO_XML,
        ArtifactType.PAGE_XML,
        ArtifactType.ENTITIES,
        ArtifactType.READING_ORDER,
    ):
        payload = document.ground_truths.get(level)
        if payload is None:
            continue
        gt_uri = _resolve_gt_uri(
            level=level,
            payload=payload,
            doc_id=doc_id,
            workspace_dir=workspace_dir,
        )
        ground_truths.append(GroundTruthRef(type=level, uri=gt_uri))

    return DocumentRef(
        id=doc_id,
        image_uri=image_uri,
        ground_truths=tuple(ground_truths),
    )


def corpus_to_corpus_spec(
    corpus: "Corpus",
    *,
    workspace_dir: Path,
) -> CorpusSpec:
    """Convertit un ``Corpus`` legacy en ``CorpusSpec`` rewrite.

    Itère sur ``corpus.documents`` et applique
    ``document_to_document_ref`` pour chacun.

    Parameters
    ----------
    corpus:
        Corpus legacy.
    workspace_dir:
        Répertoire de travail où écrire les fichiers GT
        synthétisés (typiquement un ``tempfile.TemporaryDirectory``
        détenu par le caller).

    Returns
    -------
    CorpusSpec
        Spec immutable consommable par ``BenchmarkService.run``.
    """
    if not workspace_dir.exists():
        raise PicaronesError(
            f"workspace_dir doit exister : {workspace_dir!r}"
        )

    docs = tuple(
        document_to_document_ref(d, workspace_dir=workspace_dir)
        for d in corpus.documents
    )

    metadata: dict[str, str] = {}
    for k, v in (corpus.metadata or {}).items():
        # CorpusSpec.metadata accepte ``str`` only — sérialise les
        # valeurs scalaires en str ; les structures complexes sont
        # ignorées (le caller adapte si besoin).
        if isinstance(v, (str, int, float, bool)):
            metadata[str(k)] = str(v)

    if corpus.source_path:
        metadata.setdefault("source_path", str(corpus.source_path))

    return CorpusSpec(
        name=corpus.name,
        documents=docs,
        metadata=metadata,
    )


# ──────────────────────────────────────────────────────────────────────
# Helpers privés
# ──────────────────────────────────────────────────────────────────────


def _safe_doc_id(doc_id: str) -> str:
    """Coerce un ``Document.doc_id`` vers le regex de ``DocumentRef.id``.

    Le regex ``_DOC_ID_RE = r"^[A-Za-z0-9_.\\-/]+$"`` interdit les
    espaces, accents et caractères de contrôle.  Les doc_ids
    historiques issus de ``image_path.stem`` peuvent en contenir —
    on normalise NFD et on remplace tout ce qui n'est pas conforme.
    """
    if not doc_id:
        return "doc"
    import unicodedata

    # Normalise NFD pour décomposer les caractères accentués en
    # base + diacritique, puis filtre la base ASCII.
    normalized = unicodedata.normalize("NFD", doc_id)
    safe = []
    for ch in normalized:
        # Skip les diacritiques (Mn = Mark, Nonspacing).
        if unicodedata.category(ch) == "Mn":
            continue
        if ch.isalnum() or ch in "_.-/":
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_") or "doc"
    return out


def _has_text_gt(document: "Document") -> bool:
    """``True`` ssi le document a un payload TEXT (RAW_TEXT) renseigné."""
    return ArtifactType.RAW_TEXT in document.ground_truths


def _resolve_gt_uri(
    *,
    level: ArtifactType,
    payload: object,
    doc_id: str,
    workspace_dir: Path,
) -> str:
    """Retourne l'URI d'un payload GT.

    - Si ``payload.source_path`` existe sur disque → on l'utilise
      directement (pas de copie).
    - Sinon → on sérialise dans ``workspace_dir`` selon le niveau.
    """
    source_path = getattr(payload, "source_path", None)
    if source_path is not None and Path(source_path).exists():
        return str(source_path)

    # Sérialisation de secours pour les payloads in-memory
    suffix = _DEFAULT_SUFFIXES.get(level, ".gt.txt")
    safe_id = doc_id.replace("/", "_")
    out_path = workspace_dir / f"{safe_id}{suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content = _payload_to_text(level, payload)
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)


_DEFAULT_SUFFIXES: dict[ArtifactType, str] = {
    ArtifactType.ALTO_XML: ".gt.alto.xml",
    ArtifactType.PAGE_XML: ".gt.page.xml",
    ArtifactType.ENTITIES: ".gt.entities.json",
    ArtifactType.READING_ORDER: ".gt.reading_order.json",
}


def _payload_to_text(level: ArtifactType, payload: object) -> str:
    """Sérialise un payload GT (in-memory) vers une string fichier."""
    if level in (ArtifactType.ALTO_XML, ArtifactType.PAGE_XML):
        return getattr(payload, "xml_content", "")
    if level == ArtifactType.ENTITIES:
        import json
        return json.dumps(
            getattr(payload, "entities", []),
            ensure_ascii=False,
            indent=2,
        )
    if level == ArtifactType.READING_ORDER:
        import json
        return json.dumps(
            getattr(payload, "region_order", []),
            ensure_ascii=False,
        )
    # Niveau inconnu : on utilise le ``text`` si présent, sinon
    # une chaîne vide.
    return getattr(payload, "text", "") or ""


__all__ = [
    "document_to_document_ref",
    "corpus_to_corpus_spec",
]
