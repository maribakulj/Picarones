"""Chargement et gestion des corpus de documents.

Format supporté :
  - Paires classiques : image + .gt.txt
  - Triplets post-correction : image + .gt.txt + .ocr.txt

Convention :
  mon_document.jpg   ←→   mon_document.gt.txt              (paire)
  mon_document.jpg   ←→   mon_document.gt.txt + mon_document.ocr.txt  (triplet)

Le fichier ``.ocr.txt`` contient le texte OCR bruité (sortie d'un moteur OCR)
qui sera utilisé comme entrée pour les benchmarks de post-correction LLM.
Il est optionnel — un corpus sans ``.ocr.txt`` reste un corpus classique.

Extensions d'images acceptées : .jpg, .jpeg, .png, .tif, .tiff, .bmp, .webp
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# Extensions image reconnues
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class Document:
    """Un document du corpus : image + vérité terrain + (optionnel) OCR bruité.

    Quand ``ocr_text`` est renseigné (corpus triplet), le benchmark de
    post-correction LLM peut utiliser ce texte au lieu de lancer un moteur OCR.
    """

    image_path: Path
    ground_truth: str
    doc_id: str = ""
    ocr_text: Optional[str] = None
    """Texte OCR bruité pré-calculé (``None`` pour les corpus classiques sans ``.ocr.txt``)."""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = self.image_path.stem


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
        }
        return s


def load_corpus_from_directory(
    directory: str | Path,
    name: Optional[str] = None,
    gt_suffix: str = ".gt.txt",
    ocr_suffix: str = ".ocr.txt",
    encoding: str = "utf-8",
) -> Corpus:
    """Charge un corpus depuis un dossier local.

    Supporte deux formats :
    - **Paires** : ``image + .gt.txt``
    - **Triplets** : ``image + .gt.txt + .ocr.txt`` (post-correction LLM)

    Le fichier ``.ocr.txt`` est optionnel.  Quand il est présent, le champ
    ``Document.ocr_text`` est renseigné et le benchmark peut l'utiliser
    comme entrée OCR bruitée pour tester la post-correction LLM sans
    relancer un moteur OCR.

    Parameters
    ----------
    directory:
        Chemin vers le dossier contenant les paires/triplets.
    name:
        Nom du corpus (par défaut : nom du dossier).
    gt_suffix:
        Suffixe des fichiers vérité terrain (par défaut : ``.gt.txt``).
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

    for image_path in image_paths:
        gt_path = image_path.with_name(image_path.stem + gt_suffix)
        if not gt_path.exists():
            logger.debug("Pas de fichier GT pour %s — ignoré.", image_path.name)
            skipped += 1
            continue

        try:
            ground_truth = gt_path.read_text(encoding=encoding).strip()
        except OSError as exc:
            logger.warning("Impossible de lire %s : %s — ignoré.", gt_path, exc)
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
                logger.warning("Impossible de lire %s : %s — OCR bruité ignoré.", ocr_path, exc)

        documents.append(
            Document(
                image_path=image_path,
                ground_truth=ground_truth,
                ocr_text=ocr_text,
            )
        )

    if not documents:
        raise ValueError(
            f"Aucun document valide trouvé dans {directory}. "
            f"Vérifiez que les fichiers GT portent le suffixe '{gt_suffix}'."
        )

    if skipped:
        logger.info("%d image(s) ignorée(s) faute de fichier GT.", skipped)

    if ocr_text_loaded:
        logger.info(
            "Corpus '%s' chargé : %d documents (%d avec OCR bruité — post-correction disponible).",
            corpus_name, len(documents), ocr_text_loaded,
        )
    else:
        logger.info("Corpus '%s' chargé : %d documents.", corpus_name, len(documents))
    return Corpus(
        name=corpus_name,
        documents=documents,
        source_path=str(directory),
    )
