"""Import de corpus depuis des manifestes IIIF v2 et v3.

Fonctionnement
--------------
1. Téléchargement et parsing du manifeste JSON (v2 ou v3 auto-détecté)
2. Extraction de la liste des canvases (pages) avec leurs URL d'image
3. Sélection optionnelle d'un sous-ensemble de pages (ex : ``--pages 1-10``)
4. Téléchargement des images dans un dossier local
5. Création de fichiers GT vides (``.gt.txt``) à remplir manuellement,
   OU chargement des annotations de transcription si présentes dans le manifeste
6. Construction et retour d'un objet ``Corpus``

Compatibilité
-------------
- IIIF Image API v2 et v3
- Manifestes Presentation API v2 et v3
- Instances : Gallica (BnF), Bodleian, British Library, BSB, e-codices,
  Europeana, et tout entrepôt IIIF-compliant

Utilisation
-----------
>>> from picarones.importers.iiif import IIIFImporter
>>> importer = IIIFImporter("https://gallica.bnf.fr/ark:/12148/xxx/manifest.json")
>>> corpus = importer.import_corpus(pages="1-10", output_dir="./corpus/")
>>> print(f"{len(corpus)} documents téléchargés")

Ou via la fonction de commodité :
>>> from picarones.importers.iiif import import_iiif_manifest
>>> corpus = import_iiif_manifest("https://...", pages="1-5", output_dir="./corpus/")
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from picarones.core.corpus import Corpus, Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing du sélecteur de pages
# ---------------------------------------------------------------------------

def parse_page_selector(pages: str, total: int) -> list[int]:
    """Parse un sélecteur de pages en liste d'indices 0-based.

    Formats acceptés :
    - ``"1-10"``        → pages 1 à 10 (1-based)
    - ``"1,3,5"``       → pages 1, 3 et 5
    - ``"1-5,10,15-20"`` → combinaison
    - ``"all"`` / ``""`` → toutes les pages

    Parameters
    ----------
    pages:
        Sélecteur de pages en chaîne de caractères.
    total:
        Nombre total de pages dans le manifeste.

    Returns
    -------
    list[int]
        Indices 0-based des pages sélectionnées, triés et dédoublonnés.

    Raises
    ------
    ValueError
        Si la syntaxe est invalide ou les numéros hors bornes.
    """
    if not pages or pages.strip().lower() == "all":
        return list(range(total))

    indices: set[int] = set()
    for part in pages.split(","):
        part = part.strip()
        if "-" in part:
            m = re.fullmatch(r"(\d+)-(\d+)", part)
            if not m:
                raise ValueError(f"Sélecteur de pages invalide : '{part}'")
            start, end = int(m.group(1)), int(m.group(2))
            if start < 1 or end > total or start > end:
                raise ValueError(
                    f"Plage {start}-{end} hors bornes (1–{total})"
                )
            indices.update(range(start - 1, end))
        else:
            n = int(part)
            if n < 1 or n > total:
                raise ValueError(f"Page {n} hors bornes (1–{total})")
            indices.add(n - 1)
    return sorted(indices)


# ---------------------------------------------------------------------------
# Données d'un canvas IIIF
# ---------------------------------------------------------------------------

@dataclass
class IIIFCanvas:
    """Représente un canvas (page) dans un manifeste IIIF."""

    index: int          # position 0-based dans le manifeste
    label: str          # étiquette lisible (ex : "f. 1r", "Page 1")
    image_url: str      # URL de l'image pleine résolution
    width: Optional[int] = None
    height: Optional[int] = None
    transcription: Optional[str] = None  # texte GT si annoté dans le manifeste


# ---------------------------------------------------------------------------
# Parseur de manifeste IIIF
# ---------------------------------------------------------------------------

class IIIFManifestParser:
    """Parse un manifeste IIIF Presentation API v2 ou v3."""

    def __init__(self, manifest: dict) -> None:
        self._manifest = manifest
        self._version = self._detect_version()

    def _detect_version(self) -> int:
        """Détecte la version du manifeste (2 ou 3)."""
        context = self._manifest.get("@context", "")
        if isinstance(context, list):
            context = " ".join(context)
        if "presentation/3" in context or self._manifest.get("type") == "Manifest":
            return 3
        return 2

    @property
    def version(self) -> int:
        return self._version

    @property
    def label(self) -> str:
        """Titre du manifeste."""
        raw = self._manifest.get("label", "")
        return _extract_label(raw)

    @property
    def attribution(self) -> str:
        raw = self._manifest.get("attribution", self._manifest.get("requiredStatement", ""))
        return _extract_label(raw)

    def canvases(self) -> list[IIIFCanvas]:
        """Retourne la liste des canvases du manifeste."""
        if self._version == 3:
            return self._parse_v3_canvases()
        return self._parse_v2_canvases()

    def _parse_v2_canvases(self) -> list[IIIFCanvas]:
        canvases: list[IIIFCanvas] = []
        sequences = self._manifest.get("sequences", [])
        if not sequences:
            return canvases
        raw_canvases = sequences[0].get("canvases", [])
        for i, canvas in enumerate(raw_canvases):
            label = _extract_label(canvas.get("label", f"canvas_{i+1}"))
            # Image principale : images[0].resource.@id ou service
            images = canvas.get("images", [])
            image_url = ""
            if images:
                resource = images[0].get("resource", {})
                image_url = _best_image_url_v2(resource, canvas)

            # Annotations de transcription (OA annotations)
            transcription = _extract_v2_transcription(canvas)

            canvases.append(IIIFCanvas(
                index=i,
                label=label,
                image_url=image_url,
                width=canvas.get("width"),
                height=canvas.get("height"),
                transcription=transcription,
            ))
        return canvases

    def _parse_v3_canvases(self) -> list[IIIFCanvas]:
        canvases: list[IIIFCanvas] = []
        items = self._manifest.get("items", [])
        for i, canvas in enumerate(items):
            label = _extract_label(canvas.get("label", f"canvas_{i+1}"))
            image_url = _best_image_url_v3(canvas)
            transcription = _extract_v3_transcription(canvas)
            canvases.append(IIIFCanvas(
                index=i,
                label=label,
                image_url=image_url,
                width=canvas.get("width"),
                height=canvas.get("height"),
                transcription=transcription,
            ))
        return canvases


# ---------------------------------------------------------------------------
# Helpers extraction URL et label
# ---------------------------------------------------------------------------

def _extract_label(raw: object) -> str:
    """Extrait une chaîne lisible depuis les différents formats de label IIIF."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list) and raw:
        return _extract_label(raw[0])
    if isinstance(raw, dict):
        # IIIF v3 : {"fr": ["titre"], "en": ["title"]}
        for lang in ("fr", "en", "none", "@value"):
            val = raw.get(lang, "")
            if val:
                if isinstance(val, list):
                    return val[0] if val else ""
                return str(val)
        # Fallback: première valeur
        for v in raw.values():
            return _extract_label(v)
    return str(raw) if raw else ""


def _best_image_url_v2(resource: dict, canvas: dict) -> str:
    """Construit l'URL d'image optimale depuis une ressource IIIF v2."""
    # 1. URL directe de la ressource
    direct = resource.get("@id", "")
    if direct and not direct.endswith("/info.json"):
        return direct

    # 2. Via le service IIIF Image API
    service = resource.get("service", {})
    if isinstance(service, list) and service:
        service = service[0]
    service_id = service.get("@id", service.get("id", ""))
    if service_id:
        return f"{service_id.rstrip('/')}/full/max/0/default.jpg"

    return direct


def _best_image_url_v3(canvas: dict) -> str:
    """Extrait l'URL d'image depuis un canvas IIIF v3."""
    items = canvas.get("items", [])
    for annotation_page in items:
        for annotation in annotation_page.get("items", []):
            body = annotation.get("body", {})
            if isinstance(body, list):
                body = body[0] if body else {}
            # URL directe
            url = body.get("id", body.get("@id", ""))
            if url and body.get("type", "") == "Image":
                return url
            # Via service IIIF Image API
            service = body.get("service", [])
            if isinstance(service, dict):
                service = [service]
            for svc in service:
                svc_id = svc.get("id", svc.get("@id", ""))
                if svc_id:
                    return f"{svc_id.rstrip('/')}/full/max/0/default.jpg"
            if url:
                return url
    return ""


def _extract_v2_transcription(canvas: dict) -> Optional[str]:
    """Tente d'extraire le texte GT depuis les annotations OA d'un canvas v2."""
    other_content = canvas.get("otherContent", [])
    for oc in other_content:
        if not isinstance(oc, dict):
            continue
        motivation = oc.get("motivation", "")
        if "transcrib" in motivation.lower() or "supplementing" in motivation.lower():
            resources = oc.get("resources", [])
            texts = []
            for res in resources:
                body = res.get("resource", {})
                if body.get("@type") == "cnt:ContentAsText":
                    texts.append(body.get("chars", ""))
            if texts:
                return "\n".join(texts)
    return None


def _extract_v3_transcription(canvas: dict) -> Optional[str]:
    """Tente d'extraire le texte GT depuis les annotations d'un canvas v3."""
    annotations = canvas.get("annotations", [])
    for ann_page in annotations:
        items = ann_page.get("items", [])
        for ann in items:
            motivation = ann.get("motivation", "")
            if "transcrib" in motivation.lower() or "supplementing" in motivation.lower():
                body = ann.get("body", {})
                if isinstance(body, dict) and body.get("type") == "TextualBody":
                    return body.get("value", "")
    return None


# ---------------------------------------------------------------------------
# Téléchargement avec retry
# ---------------------------------------------------------------------------

def _download_url(
    url: str,
    retries: int = 4,
    backoff: float = 2.0,
    timeout: int = 60,
) -> bytes:
    """Télécharge une URL avec retry exponentiel."""
    headers = {
        "User-Agent": "Picarones/1.0 (OCR benchmark platform; https://github.com/maribakulj/Picarones)"
    }
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        if attempt > 0:
            wait = backoff ** attempt
            logger.debug("Retry %d/%d dans %.1fs — %s", attempt, retries - 1, wait, url)
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_exc = exc
            logger.warning("Erreur téléchargement %s : %s", url, exc)
    raise RuntimeError(f"Impossible de télécharger {url} après {retries} tentatives") from last_exc


def _fetch_manifest(url: str) -> dict:
    """Télécharge et parse un manifeste IIIF JSON."""
    data = _download_url(url)
    try:
        return json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifeste IIIF invalide (JSON mal formé) : {url}") from exc


# ---------------------------------------------------------------------------
# Importeur principal
# ---------------------------------------------------------------------------

class IIIFImporter:
    """Importe un corpus depuis un manifeste IIIF.

    Parameters
    ----------
    manifest_url:
        URL du manifeste IIIF (Presentation API v2 ou v3).
    max_resolution:
        Résolution maximale des images téléchargées (largeur en pixels).
        0 = résolution maximale disponible.
    """

    def __init__(
        self,
        manifest_url: str,
        max_resolution: int = 0,
    ) -> None:
        self.manifest_url = manifest_url
        self.max_resolution = max_resolution
        self._manifest: Optional[dict] = None
        self._parser: Optional[IIIFManifestParser] = None

    def load(self) -> "IIIFImporter":
        """Télécharge et parse le manifeste."""
        logger.info("Téléchargement du manifeste IIIF : %s", self.manifest_url)
        self._manifest = _fetch_manifest(self.manifest_url)
        self._parser = IIIFManifestParser(self._manifest)
        logger.info(
            "Manifeste chargé — version IIIF %d — titre : %s — %d canvas",
            self._parser.version,
            self._parser.label,
            len(self._parser.canvases()),
        )
        return self

    @property
    def parser(self) -> IIIFManifestParser:
        if self._parser is None:
            self.load()
        return self._parser  # type: ignore[return-value]

    def list_canvases(self, pages: str = "all") -> list[IIIFCanvas]:
        """Retourne la liste des canvases sélectionnés."""
        all_canvases = self.parser.canvases()
        indices = parse_page_selector(pages, len(all_canvases))
        return [all_canvases[i] for i in indices]

    def import_corpus(
        self,
        pages: str = "all",
        output_dir: Optional[str | Path] = None,
        show_progress: bool = True,
    ) -> Corpus:
        """Télécharge les images et construit un corpus Picarones.

        Si les canvases contiennent des annotations de transcription (GT),
        elles sont automatiquement sauvegardées dans les fichiers ``.gt.txt``.
        Sinon, des fichiers ``.gt.txt`` vides sont créés.

        Parameters
        ----------
        pages:
            Sélecteur de pages (ex : ``"1-10"``, ``"1,3,5"``).
        output_dir:
            Dossier de destination pour les images et les GT.
            Si None, le corpus est retourné en mémoire sans écriture disque.
        show_progress:
            Affiche une barre de progression tqdm.

        Returns
        -------
        Corpus
            Corpus prêt à être utilisé dans ``run_benchmark``.
        """
        canvases = self.list_canvases(pages)
        if not canvases:
            raise ValueError("Aucun canvas sélectionné.")

        out_dir: Optional[Path] = Path(output_dir) if output_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        # Nom du corpus depuis le titre du manifeste
        corpus_name = self.parser.label or "iiif_corpus"

        documents: list[Document] = []
        iterator: Iterator[IIIFCanvas] = iter(canvases)

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(canvases, desc="Import IIIF", unit="page")
            except ImportError:
                pass

        for canvas in iterator:
            doc_id = f"{_slugify(canvas.label) or f'canvas_{canvas.index+1:04d}'}"

            if not canvas.image_url:
                logger.warning("Canvas %s : pas d'URL d'image — ignoré.", canvas.label)
                continue

            # Ajuster la résolution si max_resolution est défini
            image_url = self._adjust_resolution(canvas.image_url, canvas.width)

            # Téléchargement de l'image
            try:
                image_bytes = _download_url(image_url)
            except RuntimeError as exc:
                logger.error("Canvas %s : erreur téléchargement : %s", canvas.label, exc)
                continue

            # Déterminer l'extension de l'image
            ext = _guess_extension(image_url)

            if out_dir:
                # Sauvegarde sur disque
                image_path = out_dir / f"{doc_id}{ext}"
                image_path.write_bytes(image_bytes)

                gt_path = out_dir / f"{doc_id}.gt.txt"
                gt_text = canvas.transcription or ""
                gt_path.write_text(gt_text, encoding="utf-8")

                documents.append(Document(
                    image_path=image_path,
                    ground_truth=gt_text,
                    doc_id=doc_id,
                    metadata={"iiif_label": canvas.label, "canvas_index": canvas.index},
                ))
            else:
                # Corpus en mémoire (image stockée comme chemin temporaire virtuel)
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                tmp.write(image_bytes)
                tmp.close()
                documents.append(Document(
                    image_path=Path(tmp.name),
                    ground_truth=canvas.transcription or "",
                    doc_id=doc_id,
                    metadata={"iiif_label": canvas.label, "canvas_index": canvas.index},
                ))

        if not documents:
            raise ValueError("Aucun document importé depuis le manifeste IIIF.")

        logger.info("Import IIIF terminé : %d documents.", len(documents))

        return Corpus(
            name=corpus_name,
            documents=documents,
            source_path=self.manifest_url,
            metadata={
                "iiif_manifest_url": self.manifest_url,
                "iiif_version": self.parser.version,
                "iiif_attribution": self.parser.attribution,
                "pages_selected": pages,
            },
        )

    def _adjust_resolution(self, image_url: str, canvas_width: Optional[int]) -> str:
        """Ajuste l'URL IIIF Image API pour respecter max_resolution."""
        if not self.max_resolution or not canvas_width:
            return image_url
        if canvas_width <= self.max_resolution:
            return image_url
        # Remplacer /full/max/ ou /full/full/ par /full/{w},/
        url = re.sub(
            r"/full/(max|full)/",
            f"/full/{self.max_resolution},/",
            image_url,
        )
        return url


# ---------------------------------------------------------------------------
# Helpers utilitaires
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convertit un label IIIF en identifiant de fichier sûr."""
    text = re.sub(r"[^\w\s-]", "", text.strip())
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:60]


def _guess_extension(url: str) -> str:
    """Détermine l'extension de l'image depuis l'URL."""
    url_lower = url.lower().split("?")[0]
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"):
        if url_lower.endswith(ext):
            return ext
    # Par défaut pour les URLs IIIF Image API
    if "/default." in url_lower or "/native." in url_lower:
        return ".jpg"
    return ".jpg"


# ---------------------------------------------------------------------------
# Fonction de commodité
# ---------------------------------------------------------------------------

def import_iiif_manifest(
    manifest_url: str,
    pages: str = "all",
    output_dir: Optional[str | Path] = None,
    max_resolution: int = 0,
    show_progress: bool = True,
) -> Corpus:
    """Importe un corpus depuis un manifeste IIIF en une seule ligne.

    Parameters
    ----------
    manifest_url:
        URL du manifeste IIIF (v2 ou v3).
    pages:
        Sélecteur de pages (ex : ``"1-10"``, ``"1,3,5"``). ``"all"`` par défaut.
    output_dir:
        Dossier de destination. Si None, corpus en mémoire.
    max_resolution:
        Résolution maximale (px). 0 = pas de limite.
    show_progress:
        Affiche une barre de progression.

    Returns
    -------
    Corpus
    """
    importer = IIIFImporter(manifest_url, max_resolution=max_resolution)
    importer.load()
    return importer.import_corpus(
        pages=pages,
        output_dir=output_dir,
        show_progress=show_progress,
    )
