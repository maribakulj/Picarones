"""Chargement et gestion des corpus de documents (couche 3 — evaluation).

Coexistence avec ``domain.corpus.CorpusSpec``
---------------------------------------------
``evaluation.corpus`` (le présent module) porte les types **riches
en behavior** consommés par ``BenchmarkService`` (couche 6) :
``Document``, ``Corpus``, ``ArtifactType`` + payloads
``TextGT``/``AltoGT``/``PageGT``/``EntitiesGT``/``ReadingOrderGT``
chargés en mémoire, et la fonction ``load_corpus_from_directory``.

``domain.corpus.CorpusSpec`` + ``domain.documents.DocumentRef``
(Pydantic, immutable, déclaratif) sont une vue **structurelle**
utilisée par le pipeline executor canonique (``pipeline/``) et
les services d'orchestration (``app/services/``).  Les deux
modèles cohabitent intentionnellement — un convertisseur explicite
``CorpusSpec ↔ Corpus`` viendra quand un caller institutionnel
l'exigera.

Format supporté :
  - Paires classiques : image + .gt.txt
  - Triplets post-correction : image + .gt.txt + .ocr.txt
  - GT multi-niveaux (Sprint 32) : image + .gt.txt + .gt.alto.xml + ...

Convention :
  mon_document.jpg   ←→   mon_document.gt.txt              (paire texte)
  mon_document.jpg   ←→   mon_document.gt.txt + mon_document.ocr.txt (triplet)
  mon_document.jpg   ←→   mon_document.gt.txt + mon_document.gt.alto.xml (multi-niveaux)

Le fichier ``.ocr.txt`` contient le texte OCR bruité (sortie d'un moteur OCR)
qui sera utilisé comme entrée pour les benchmarks de post-correction LLM.
Il est optionnel — un corpus sans ``.ocr.txt`` reste un corpus classique.

GT multi-niveaux (Sprint 32 — Phase 0.1)
----------------------------------------
Chaque document peut porter une vérité terrain à plusieurs niveaux :
texte brut, ALTO XML, PAGE XML, entités nommées, ordre de lecture.
Le niveau ``TEXT`` reste la base (rétrocompatibilité stricte) ; les
autres niveaux sont optionnels et permettront aux modules futurs
(reconstructeurs ALTO, mappeurs VLM→ALTO, NER) d'évaluer leur sortie
contre la GT correspondante.

Suffixes détectés automatiquement par ``load_corpus_from_directory`` :
  - ``.gt.txt``                → ``ArtifactType.RAW_TEXT``         (TextGT)
  - ``.gt.alto.xml``           → ``ArtifactType.ALTO_XML``         (AltoGT)
  - ``.gt.page.xml``           → ``ArtifactType.PAGE_XML``         (PageGT)
  - ``.gt.entities.json``      → ``ArtifactType.ENTITIES``     (EntitiesGT)
  - ``.gt.reading_order.json`` → ``ArtifactType.READING_ORDER`` (ReadingOrderGT)

Extensions d'images acceptées : .jpg, .jpeg, .png, .tif, .tiff, .bmp, .webp
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from picarones.domain.artifacts import ArtifactType

logger = logging.getLogger(__name__)

# Extensions image reconnues
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ──────────────────────────────────────────────────────────────────────────
# Vérité terrain multi-niveaux (Sprint 32 — Phase 0.1)
# ──────────────────────────────────────────────────────────────────────────
#
# Phase 4 leftover (2026-05) — l'enum local ``GTLevel`` a été supprimé
# au profit de ``picarones.domain.artifacts.ArtifactType`` (canonique
# depuis Sprint A14).  Le mapping appliqué :
#
#     GTLevel.TEXT          → ArtifactType.RAW_TEXT
#     GTLevel.ALTO          → ArtifactType.ALTO_XML
#     GTLevel.PAGE          → ArtifactType.PAGE_XML
#     GTLevel.ENTITIES      → ArtifactType.ENTITIES
#     GTLevel.READING_ORDER → ArtifactType.READING_ORDER


@dataclass
class TextGT:
    """Texte brut transcrit (le niveau historique de Picarones)."""

    text: str
    source_path: Optional[Path] = None

    @property
    def level(self) -> ArtifactType:
        return ArtifactType.RAW_TEXT


@dataclass
class AltoGT:
    """ALTO XML brut.  Le contenu n'est pas parsé à la construction —
    chaque consommateur (métrique, exporteur) parse à la demande pour
    éviter de payer le coût quand inutile."""

    xml_content: str
    source_path: Optional[Path] = None

    @property
    def level(self) -> ArtifactType:
        return ArtifactType.ALTO_XML


@dataclass
class PageGT:
    """PAGE XML brut (PRImA)."""

    xml_content: str
    source_path: Optional[Path] = None

    @property
    def level(self) -> ArtifactType:
        return ArtifactType.PAGE_XML


@dataclass
class EntitiesGT:
    """Annotations d'entités nommées (NER).

    Format attendu : liste de dictionnaires
    ``{"label": str, "start": int, "end": int, "text": str}`` où
    ``start``/``end`` sont des offsets caractère sur le texte du niveau
    ``TEXT``.
    """

    entities: list[dict[str, Any]] = field(default_factory=list)
    source_path: Optional[Path] = None

    @property
    def level(self) -> ArtifactType:
        return ArtifactType.ENTITIES


@dataclass
class ReadingOrderGT:
    """Ordre de lecture des régions ALTO/PAGE.

    Liste ordonnée d'identifiants de région tels qu'ils apparaissent dans
    le ``.gt.alto.xml`` ou ``.gt.page.xml`` correspondant.
    """

    region_order: list[str] = field(default_factory=list)
    source_path: Optional[Path] = None

    @property
    def level(self) -> ArtifactType:
        return ArtifactType.READING_ORDER


# Union des payloads — utilisée pour le typage des métriques
GTPayload = Union[TextGT, AltoGT, PageGT, EntitiesGT, ReadingOrderGT]


# ──────────────────────────────────────────────────────────────────────────
# Suffixes reconnus par le loader pour chaque niveau
# ──────────────────────────────────────────────────────────────────────────


GT_SUFFIXES: dict[ArtifactType, str] = {
    ArtifactType.RAW_TEXT: ".gt.txt",
    ArtifactType.ALTO_XML: ".gt.alto.xml",
    ArtifactType.PAGE_XML: ".gt.page.xml",
    ArtifactType.ENTITIES: ".gt.entities.json",
    ArtifactType.READING_ORDER: ".gt.reading_order.json",
}


# ──────────────────────────────────────────────────────────────────────────
# Document et Corpus
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Document:
    """Un document du corpus : image + vérité terrain + (optionnel) OCR bruité.

    Quand ``ocr_text`` est renseigné (corpus triplet), le benchmark de
    post-correction LLM peut utiliser ce texte au lieu de lancer un moteur OCR.

    GT multi-niveaux (Sprint 32 — Phase 0.1)
    ----------------------------------------
    Le champ ``ground_truth: str`` reste le niveau ``TEXT`` historique et
    garantit la rétrocompatibilité stricte avec tout le code existant
    (runner, métriques, rapport, importers).  En complément, le champ
    ``ground_truths: dict[ArtifactType, GTPayload]`` peut porter des niveaux
    additionnels (ALTO, PAGE, ENTITIES, READING_ORDER).

    Les deux représentations restent synchronisées : si ``ground_truth``
    est passé sans entrée ``TEXT`` dans ``ground_truths``, le post-init
    crée automatiquement le ``TextGT`` correspondant.  Inversement, si un
    ``TextGT`` est présent dans ``ground_truths`` sans ``ground_truth``,
    le post-init renseigne le champ ``str``.
    """

    image_path: Path
    ground_truth: str = ""
    doc_id: str = ""
    ocr_text: Optional[str] = None
    """Texte OCR bruité pré-calculé (``None`` pour les corpus classiques sans ``.ocr.txt``)."""
    metadata: dict = field(default_factory=dict)
    ground_truths: dict[ArtifactType, GTPayload] = field(default_factory=dict)
    """GT multi-niveaux (Sprint 32).  Le niveau ``TEXT`` est synchronisé
    automatiquement avec le champ ``ground_truth`` ci-dessus."""

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = self.image_path.stem
        # Synchronise le niveau TEXT entre champ str et dict typé.
        if ArtifactType.RAW_TEXT in self.ground_truths:
            text_payload = self.ground_truths[ArtifactType.RAW_TEXT]
            if isinstance(text_payload, TextGT) and not self.ground_truth:
                self.ground_truth = text_payload.text
        elif self.ground_truth:
            self.ground_truths[ArtifactType.RAW_TEXT] = TextGT(text=self.ground_truth)

    def has_gt(self, level: ArtifactType) -> bool:
        """``True`` si le document possède une GT au niveau demandé."""
        return level in self.ground_truths

    def get_gt(self, level: ArtifactType) -> Optional[GTPayload]:
        """Retourne le payload GT au niveau demandé, ou ``None``."""
        return self.ground_truths.get(level)

    @property
    def gt_levels(self) -> set[ArtifactType]:
        """Niveaux de GT disponibles pour ce document."""
        return set(self.ground_truths.keys())


@dataclass
class Corpus:
    """Collection de documents avec leurs métadonnées."""

    name: str
    documents: list[Document]
    source_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    def __repr__(self) -> str:
        return f"Corpus(name={self.name!r}, documents={len(self.documents)})"

    @property
    def has_ocr_text(self) -> bool:
        """True si au moins un document possède un texte OCR pré-calculé."""
        return any(doc.ocr_text is not None for doc in self.documents)

    @property
    def ocr_text_count(self) -> int:
        """Nombre de documents avec un texte OCR pré-calculé."""
        return sum(1 for doc in self.documents if doc.ocr_text is not None)

    @property
    def available_gt_levels(self) -> set[ArtifactType]:
        """Union des niveaux de GT présents dans au moins un document."""
        levels: set[ArtifactType] = set()
        for doc in self.documents:
            levels.update(doc.gt_levels)
        return levels

    def gt_level_coverage(self) -> dict[ArtifactType, int]:
        """Nombre de documents possédant chaque niveau de GT."""
        coverage: dict[ArtifactType, int] = {}
        for doc in self.documents:
            for level in doc.gt_levels:
                coverage[level] = coverage.get(level, 0) + 1
        return coverage

    @property
    def stats(self) -> dict:
        gt_lengths = [len(doc.ground_truth) for doc in self.documents]
        if not gt_lengths:
            return {"document_count": 0}
        import statistics

        s = {
            "document_count": len(self.documents),
            "gt_length_mean": round(statistics.mean(gt_lengths), 1),
            "gt_length_median": round(statistics.median(gt_lengths), 1),
            "gt_length_min": min(gt_lengths),
            "gt_length_max": max(gt_lengths),
            "has_ocr_text": self.has_ocr_text,
            "ocr_text_count": self.ocr_text_count,
            # Sprint 32 — exposition de la couverture multi-niveaux
            "gt_level_coverage": {lvl.value: n for lvl, n in self.gt_level_coverage().items()},
        }
        return s


# ──────────────────────────────────────────────────────────────────────────
# Loader local
# ──────────────────────────────────────────────────────────────────────────


def _load_extra_gt_levels(image_path: Path, encoding: str) -> dict[ArtifactType, GTPayload]:
    """Détecte et charge les niveaux de GT additionnels présents à côté de l'image.

    Les erreurs de lecture/parse sont **dégradées en warning** (cf.
    CLAUDE.md : pas de ``except Exception: pass``) ; le document conserve
    alors les niveaux qui ont pu être chargés.
    """
    extras: dict[ArtifactType, GTPayload] = {}
    stem = image_path.stem

    # ALTO
    alto_path = image_path.with_name(stem + GT_SUFFIXES[ArtifactType.ALTO_XML])
    if alto_path.exists():
        try:
            extras[ArtifactType.ALTO_XML] = AltoGT(
                xml_content=alto_path.read_text(encoding=encoding),
                source_path=alto_path,
            )
        except OSError as exc:
            logger.warning("[corpus] ALTO ignoré pour %s : %s", image_path.name, exc)

    # PAGE
    page_path = image_path.with_name(stem + GT_SUFFIXES[ArtifactType.PAGE_XML])
    if page_path.exists():
        try:
            extras[ArtifactType.PAGE_XML] = PageGT(
                xml_content=page_path.read_text(encoding=encoding),
                source_path=page_path,
            )
        except OSError as exc:
            logger.warning("[corpus] PAGE XML ignoré pour %s : %s", image_path.name, exc)

    # ENTITIES (JSON)
    ent_path = image_path.with_name(stem + GT_SUFFIXES[ArtifactType.ENTITIES])
    if ent_path.exists():
        try:
            payload = json.loads(ent_path.read_text(encoding=encoding))
            if isinstance(payload, dict) and "entities" in payload:
                entities = payload["entities"]
            elif isinstance(payload, list):
                entities = payload
            else:
                logger.warning(
                    "[corpus] entités ignorées pour %s : format JSON inattendu",
                    image_path.name,
                )
                entities = None
            if entities is not None:
                extras[ArtifactType.ENTITIES] = EntitiesGT(
                    entities=entities,
                    source_path=ent_path,
                )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("[corpus] entités ignorées pour %s : %s", image_path.name, exc)

    # READING_ORDER (JSON)
    ro_path = image_path.with_name(stem + GT_SUFFIXES[ArtifactType.READING_ORDER])
    if ro_path.exists():
        try:
            payload = json.loads(ro_path.read_text(encoding=encoding))
            if isinstance(payload, dict) and "region_order" in payload:
                region_order = payload["region_order"]
            elif isinstance(payload, list):
                region_order = payload
            else:
                logger.warning(
                    "[corpus] reading_order ignoré pour %s : format JSON inattendu",
                    image_path.name,
                )
                region_order = None
            if region_order is not None:
                extras[ArtifactType.READING_ORDER] = ReadingOrderGT(
                    region_order=list(region_order),
                    source_path=ro_path,
                )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "[corpus] reading_order ignoré pour %s : %s", image_path.name, exc
            )

    return extras


def load_corpus_from_directory(
    directory: str | Path,
    name: Optional[str] = None,
    gt_suffix: str = ".gt.txt",
    ocr_suffix: str = ".ocr.txt",
    encoding: str = "utf-8",
) -> Corpus:
    """Charge un corpus depuis un dossier local.

    Supporte trois formats :

    - **Paires** : ``image + .gt.txt``
    - **Triplets** : ``image + .gt.txt + .ocr.txt`` (post-correction LLM)
    - **Multi-niveaux** (Sprint 32) : ``image + .gt.txt`` + un ou plusieurs
      des fichiers ``.gt.alto.xml``, ``.gt.page.xml``,
      ``.gt.entities.json``, ``.gt.reading_order.json``.

    Le fichier ``.ocr.txt`` et les fichiers GT additionnels sont tous
    **optionnels**.  Un corpus avec uniquement des paires se comporte
    exactement comme avant (rétrocompatibilité stricte).

    Parameters
    ----------
    directory:
        Chemin vers le dossier contenant les paires/triplets.
    name:
        Nom du corpus (par défaut : nom du dossier).
    gt_suffix:
        Suffixe des fichiers vérité terrain texte (par défaut : ``.gt.txt``).
    ocr_suffix:
        Suffixe des fichiers OCR bruité (par défaut : ``.ocr.txt``).
    encoding:
        Encodage des fichiers texte (par défaut : utf-8).

    Returns
    -------
    Corpus

    Raises
    ------
    FileNotFoundError
        Si le dossier n'existe pas.
    ValueError
        Si aucun document valide n'est trouvé.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {directory}")

    corpus_name = name or directory.name
    documents: list[Document] = []
    skipped = 0

    # Collecte de toutes les images (on exclut les fichiers cachés macOS ._* et .*)
    image_paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    )

    ocr_text_loaded = 0
    extra_levels_loaded: dict[ArtifactType, int] = {}

    for image_path in image_paths:
        gt_path = image_path.with_name(image_path.stem + gt_suffix)
        if not gt_path.exists():
            logger.debug("[corpus] Pas de fichier GT pour %s — ignoré.", image_path.name)
            skipped += 1
            continue

        try:
            ground_truth = gt_path.read_text(encoding=encoding).strip()
        except OSError as exc:
            logger.warning("[corpus] Impossible de lire %s : %s — ignoré.", gt_path, exc)
            skipped += 1
            continue

        # OCR bruité optionnel (.ocr.txt)
        ocr_text: Optional[str] = None
        ocr_path = image_path.with_name(image_path.stem + ocr_suffix)
        if ocr_path.exists():
            try:
                ocr_text = ocr_path.read_text(encoding=encoding).strip()
                ocr_text_loaded += 1
            except OSError as exc:
                logger.warning("[corpus] Impossible de lire %s : %s — OCR bruité ignoré.", ocr_path, exc)

        # GT multi-niveaux (Sprint 32) — détection automatique des fichiers additionnels
        ground_truths: dict[ArtifactType, GTPayload] = {
            ArtifactType.RAW_TEXT: TextGT(text=ground_truth, source_path=gt_path),
        }
        extras = _load_extra_gt_levels(image_path, encoding=encoding)
        ground_truths.update(extras)
        for lvl in extras:
            extra_levels_loaded[lvl] = extra_levels_loaded.get(lvl, 0) + 1

        documents.append(
            Document(
                image_path=image_path,
                ground_truth=ground_truth,
                ocr_text=ocr_text,
                ground_truths=ground_truths,
            )
        )

    if not documents:
        raise ValueError(
            f"Aucun document valide trouvé dans {directory}. "
            f"Vérifiez que les fichiers GT portent le suffixe '{gt_suffix}'."
        )

    if skipped:
        logger.info("[corpus] %d image(s) ignorée(s) faute de fichier GT.", skipped)

    if ocr_text_loaded:
        logger.info(
            "[corpus] Corpus '%s' chargé : %d documents (%d avec OCR bruité — post-correction disponible).",
            corpus_name, len(documents), ocr_text_loaded,
        )
    else:
        logger.info("[corpus] Corpus '%s' chargé : %d documents.", corpus_name, len(documents))

    if extra_levels_loaded:
        levels_summary = ", ".join(
            f"{lvl.value}={n}" for lvl, n in sorted(extra_levels_loaded.items(), key=lambda x: x[0].value)
        )
        logger.info(
            "[corpus] Corpus '%s' — niveaux de GT additionnels chargés : %s",
            corpus_name, levels_summary,
        )

    return Corpus(
        name=corpus_name,
        documents=documents,
        source_path=str(directory),
    )
