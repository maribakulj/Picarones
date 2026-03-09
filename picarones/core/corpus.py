"""Chargement et gestion des corpus de documents.

Format supporté (Sprint 1) : dossier local avec paires image / .gt.txt

Convention :
  mon_document.jpg   ←→   mon_document.gt.txt
  page_001.png       ←→   page_001.gt.txt

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
    """Une paire (image, texte de vérité terrain)."""

    image_path: Path
    ground_truth: str
    doc_id: str = ""
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
    def stats(self) -> dict:
        gt_lengths = [len(doc.ground_truth) for doc in self.documents]
        if not gt_lengths:
            return {"document_count": 0}
        import statistics

        return {
            "document_count": len(self.documents),
            "gt_length_mean": round(statistics.mean(gt_lengths), 1),
            "gt_length_median": round(statistics.median(gt_lengths), 1),
            "gt_length_min": min(gt_lengths),
            "gt_length_max": max(gt_lengths),
        }


def load_corpus_from_directory(
    directory: str | Path,
    name: Optional[str] = None,
    gt_suffix: str = ".gt.txt",
    encoding: str = "utf-8",
) -> Corpus:
    """Charge un corpus depuis un dossier local de paires image / GT.

    Parameters
    ----------
    directory:
        Chemin vers le dossier contenant les paires image + fichier GT.
    name:
        Nom du corpus (par défaut : nom du dossier).
    gt_suffix:
        Suffixe des fichiers vérité terrain (par défaut : ``.gt.txt``).
    encoding:
        Encodage des fichiers texte (par défaut : utf-8).

    Returns
    -------
    Corpus
        Objet Corpus prêt à être utilisé dans le pipeline.

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

        documents.append(
            Document(
                image_path=image_path,
                ground_truth=ground_truth,
            )
        )

    if not documents:
        raise ValueError(
            f"Aucun document valide trouvé dans {directory}. "
            f"Vérifiez que les fichiers GT portent le suffixe '{gt_suffix}'."
        )

    if skipped:
        logger.info("%d image(s) ignorée(s) faute de fichier GT.", skipped)

    logger.info("Corpus '%s' chargé : %d documents.", corpus_name, len(documents))
    return Corpus(
        name=corpus_name,
        documents=documents,
        source_path=str(directory),
    )
