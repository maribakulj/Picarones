"""Import de corpus depuis Gallica (BnF) via l'API SRU et IIIF.

Fonctionnement
--------------
1. Recherche dans Gallica par cote (ark), titre, auteur ou date via l'API SRU BnF
2. Récupération des images via l'API IIIF Gallica
3. Récupération de l'OCR Gallica existant (texte brut ou ALTO) comme concurrent de référence

API utilisées
-------------
- SRU BnF : https://gallica.bnf.fr/SRU?operation=searchRetrieve&query=...
- IIIF Gallica : https://gallica.bnf.fr/ark:/12148/{ark}/manifest.json
- OCR texte brut : https://gallica.bnf.fr/ark:/12148/{ark}/f{n}.texteBrut
- Métadonnées OAI-PMH : https://gallica.bnf.fr/services/OAIRecord?ark={ark}

Usage
-----
>>> from picarones.importers.gallica import GallicaClient
>>> client = GallicaClient()
>>> results = client.search(title="Froissart", date_from=1380, date_to=1420, max_results=10)
>>> corpus = client.import_document(results[0].ark, pages="1-5", include_gallica_ocr=True)
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from picarones.core.corpus import Corpus, Document

logger = logging.getLogger(__name__)

# Namespaces SRU/OAI
_NS_SRU = "http://www.loc.gov/zing/srw/"
_NS_DC = "http://purl.org/dc/elements/1.1/"
_NS_OAI = "http://www.openarchives.org/OAI/2.0/"

_GALLICA_BASE = "https://gallica.bnf.fr"
_SRU_URL = f"{_GALLICA_BASE}/SRU"
_IIIF_MANIFEST_TPL = f"{_GALLICA_BASE}/ark:/{{ark}}/manifest.json"
_OCR_BRUT_TPL = f"{_GALLICA_BASE}/ark:/{{ark}}/f{{page}}.texteBrut"


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class GallicaRecord:
    """Un résultat de recherche Gallica."""
    ark: str
    """Identifiant ARK sans préfixe (ex: ``'12148/btv1b8453561w'``)."""
    title: str
    creator: str = ""
    date: str = ""
    description: str = ""
    type_doc: str = ""
    language: str = ""
    rights: str = ""
    has_ocr: bool = False
    """True si Gallica fournit un OCR pour ce document."""

    @property
    def url(self) -> str:
        return f"{_GALLICA_BASE}/ark:/12148/{self.ark}"

    @property
    def manifest_url(self) -> str:
        return f"{_GALLICA_BASE}/ark:/12148/{self.ark}/manifest.json"

    def as_dict(self) -> dict:
        return {
            "ark": self.ark,
            "title": self.title,
            "creator": self.creator,
            "date": self.date,
            "description": self.description,
            "type_doc": self.type_doc,
            "language": self.language,
            "has_ocr": self.has_ocr,
            "url": self.url,
            "manifest_url": self.manifest_url,
        }


# ---------------------------------------------------------------------------
# Client Gallica
# ---------------------------------------------------------------------------

class GallicaClient:
    """Client pour les APIs Gallica (SRU, IIIF, OCR texte brut).

    Parameters
    ----------
    timeout:
        Timeout HTTP en secondes.
    delay_between_requests:
        Délai en secondes entre chaque requête (pour respecter les conditions
        d'utilisation Gallica).

    Examples
    --------
    >>> client = GallicaClient()
    >>> results = client.search(author="Froissart", max_results=5)
    >>> for r in results:
    ...     print(r.title, r.date)
    >>> corpus = client.import_document(results[0].ark, pages="1-3")
    """

    def __init__(
        self,
        timeout: int = 30,
        delay_between_requests: float = 0.5,
    ) -> None:
        self.timeout = timeout
        self.delay = delay_between_requests

    def _fetch_url(self, url: str) -> bytes:
        """Télécharge le contenu d'une URL."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Picarones/1.0 (BnF; research tool)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"HTTP {exc.code} sur {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Impossible de joindre {url}: {exc.reason}"
            ) from exc
        finally:
            if self.delay > 0:
                time.sleep(self.delay)

    def _build_sru_query(
        self,
        ark: Optional[str] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
        date_from: Optional[int] = None,
        date_to: Optional[int] = None,
        doc_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """Construit une requête CQL pour l'API SRU BnF."""
        clauses: list[str] = []

        if ark:
            # Recherche par identifiant ARK
            clauses.append(f'dc.identifier any "{ark}"')
        if title:
            clauses.append(f'dc.title all "{title}"')
        if author:
            clauses.append(f'dc.creator all "{author}"')
        if date_from and date_to:
            clauses.append(f'dc.date >= "{date_from}" and dc.date <= "{date_to}"')
        elif date_from:
            clauses.append(f'dc.date >= "{date_from}"')
        elif date_to:
            clauses.append(f'dc.date <= "{date_to}"')
        if doc_type:
            clauses.append(f'dc.type all "{doc_type}"')
        if language:
            clauses.append(f'dc.language all "{language}"')

        if not clauses:
            return 'gallica all "document"'
        return " and ".join(clauses)

    def search(
        self,
        ark: Optional[str] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
        date_from: Optional[int] = None,
        date_to: Optional[int] = None,
        doc_type: Optional[str] = None,
        language: Optional[str] = None,
        max_results: int = 20,
    ) -> list[GallicaRecord]:
        """Recherche dans Gallica via l'API SRU BnF.

        Parameters
        ----------
        ark:
            Identifiant ARK (ex : ``'12148/btv1b8453561w'``).
        title:
            Mots-clés dans le titre.
        author:
            Mots-clés dans l'auteur/créateur.
        date_from:
            Borne inférieure de date (année).
        date_to:
            Borne supérieure de date (année).
        doc_type:
            Type de document (``'monographie'``, ``'périodique'``, ``'manuscrit'``…).
        language:
            Code langue ISO 639 (``'fre'``, ``'lat'``, ``'ger'``…).
        max_results:
            Nombre maximum de résultats à retourner.

        Returns
        -------
        list[GallicaRecord]
            Liste des documents trouvés.
        """
        query = self._build_sru_query(
            ark=ark,
            title=title,
            author=author,
            date_from=date_from,
            date_to=date_to,
            doc_type=doc_type,
            language=language,
        )

        params = urllib.parse.urlencode({
            "operation": "searchRetrieve",
            "version": "1.2",
            "query": query,
            "maximumRecords": min(max_results, 50),
            "startRecord": 1,
            "recordSchema": "unimarcXchange",
        })
        url = f"{_SRU_URL}?{params}"

        try:
            raw = self._fetch_url(url)
        except RuntimeError as exc:
            logger.error("Erreur recherche SRU Gallica: %s", exc)
            return []

        return self._parse_sru_response(raw, max_results)

    def _parse_sru_response(self, xml_bytes: bytes, max_results: int) -> list[GallicaRecord]:
        """Parse la réponse SRU XML de Gallica."""
        records: list[GallicaRecord] = []
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            logger.error("Impossible de parser la réponse SRU: %s", exc)
            return records

        # Les enregistrements sont dans srw:records/srw:record/srw:recordData
        for rec_elem in root.iter():
            if rec_elem.tag.endswith("}record") or rec_elem.tag == "record":
                record = self._parse_record_element(rec_elem)
                if record:
                    records.append(record)
                if len(records) >= max_results:
                    break

        return records

    def _parse_record_element(self, elem: ET.Element) -> Optional[GallicaRecord]:
        """Extrait les métadonnées d'un enregistrement SRU."""
        # Chercher les champs Dublin Core dans l'enregistrement
        def find_text(tag_suffix: str) -> str:
            for child in elem.iter():
                if child.tag.endswith(tag_suffix) and child.text:
                    return child.text.strip()
            return ""

        def find_all_text(tag_suffix: str) -> list[str]:
            return [
                child.text.strip()
                for child in elem.iter()
                if child.tag.endswith(tag_suffix) and child.text
            ]

        # Chercher l'ARK dans l'identifiant
        identifiers = find_all_text("identifier")
        ark = ""
        for ident in identifiers:
            # Format typique : "https://gallica.bnf.fr/ark:/12148/btv1b8453561w"
            m = re.search(r"ark:/(\d+/\w+)", ident)
            if m:
                ark = m.group(1)
                break

        if not ark:
            return None

        title = find_text("title") or "Sans titre"
        creator = find_text("creator")
        date = find_text("date")

        # Vérifier si OCR disponible (heuristique : type monographie/périodique généralement)
        doc_types = find_all_text("type")
        has_ocr = any(
            t.lower() in ("monographie", "fascicule", "texte", "text")
            for t in doc_types
        )

        return GallicaRecord(
            ark=ark,
            title=title,
            creator=creator,
            date=date,
            description=find_text("description"),
            type_doc=", ".join(doc_types),
            language=find_text("language"),
            has_ocr=has_ocr,
        )

    def get_ocr_text(self, ark: str, page: int) -> str:
        """Récupère l'OCR Gallica d'une page spécifique (texte brut).

        Parameters
        ----------
        ark:
            Identifiant ARK (ex : ``'12148/btv1b8453561w'``).
        page:
            Numéro de page 1-based.

        Returns
        -------
        str
            Texte OCR Gallica pour cette page (peut être vide si non disponible).
        """
        url = _OCR_BRUT_TPL.format(ark=ark, page=page)
        try:
            raw = self._fetch_url(url)
            text = raw.decode("utf-8", errors="replace").strip()
            # Gallica retourne parfois du HTML pour les pages sans OCR
            if text.startswith("<!") or "<html" in text[:100].lower():
                return ""
            return text
        except RuntimeError as exc:
            logger.debug("OCR non disponible pour %s f%d: %s", ark, page, exc)
            return ""

    def import_document(
        self,
        ark: str,
        pages: str = "all",
        output_dir: Optional[str] = None,
        include_gallica_ocr: bool = True,
        max_resolution: int = 0,
        show_progress: bool = True,
    ) -> Corpus:
        """Importe un document Gallica comme corpus Picarones.

        Utilise le manifeste IIIF Gallica pour lister les pages et télécharger
        les images. L'OCR Gallica est optionnellement récupéré comme GT ou comme
        transcription de référence.

        Parameters
        ----------
        ark:
            Identifiant ARK (ex : ``'12148/btv1b8453561w'``).
        pages:
            Sélecteur de pages (``'all'``, ``'1-10'``, ``'1,3,5'``…).
        output_dir:
            Dossier local pour stocker images et GT.
        include_gallica_ocr:
            Si True, récupère l'OCR Gallica comme texte de référence.
        max_resolution:
            Largeur maximale des images téléchargées (0 = maximum disponible).
        show_progress:
            Affiche une barre de progression.

        Returns
        -------
        Corpus
            Corpus avec images et OCR Gallica comme GT (si disponible).
        """
        from picarones.importers.iiif import IIIFImporter

        manifest_url = f"{_GALLICA_BASE}/ark:/12148/{ark}/manifest.json"
        logger.info("Import Gallica ARK %s via IIIF : %s", ark, manifest_url)

        # Utiliser l'importeur IIIF existant pour les images
        importer = IIIFImporter(manifest_url, max_resolution=max_resolution)
        importer.load()

        corpus = importer.import_corpus(
            pages=pages,
            output_dir=output_dir or f"./corpus_gallica_{ark.split('/')[-1]}/",
            show_progress=show_progress,
        )

        # Enrichir avec l'OCR Gallica si demandé
        if include_gallica_ocr:
            selected_indices = importer.list_canvases(pages)
            for i, doc in enumerate(corpus.documents):
                page_num = selected_indices[i] + 1 if i < len(selected_indices) else i + 1
                gallica_ocr = self.get_ocr_text(ark, page_num)
                if gallica_ocr:
                    doc.metadata["gallica_ocr"] = gallica_ocr
                    # Si pas de GT manuscrite, utiliser l'OCR Gallica comme référence
                    if not doc.ground_truth.strip():
                        doc.ground_truth = gallica_ocr
                        doc.metadata["gt_source"] = "gallica_ocr"

        # Ajouter métadonnées Gallica
        corpus.metadata.update({
            "source": "gallica",
            "ark": ark,
            "manifest_url": manifest_url,
            "gallica_url": f"{_GALLICA_BASE}/ark:/12148/{ark}",
            "include_gallica_ocr": include_gallica_ocr,
        })

        return corpus

    def get_metadata(self, ark: str) -> dict:
        """Récupère les métadonnées OAI-PMH d'un document Gallica.

        Parameters
        ----------
        ark:
            Identifiant ARK.

        Returns
        -------
        dict
            Métadonnées Dublin Core du document.
        """
        url = f"{_GALLICA_BASE}/services/OAIRecord?ark=ark:/12148/{ark}"
        try:
            raw = self._fetch_url(url)
            root = ET.fromstring(raw)
        except (RuntimeError, ET.ParseError) as exc:
            logger.error("Erreur métadonnées OAI %s: %s", ark, exc)
            return {"ark": ark}

        def find_text(tag_suffix: str) -> str:
            for elem in root.iter():
                if elem.tag.endswith(tag_suffix) and elem.text:
                    return elem.text.strip()
            return ""

        return {
            "ark": ark,
            "title": find_text("title"),
            "creator": find_text("creator"),
            "date": find_text("date"),
            "description": find_text("description"),
            "subject": find_text("subject"),
            "language": find_text("language"),
            "type": find_text("type"),
            "format": find_text("format"),
            "source": find_text("source"),
            "url": f"{_GALLICA_BASE}/ark:/12148/{ark}",
        }


# ---------------------------------------------------------------------------
# Fonctions de commodité
# ---------------------------------------------------------------------------

def search_gallica(
    title: Optional[str] = None,
    author: Optional[str] = None,
    ark: Optional[str] = None,
    date_from: Optional[int] = None,
    date_to: Optional[int] = None,
    max_results: int = 20,
) -> list[GallicaRecord]:
    """Recherche rapide dans Gallica.

    Crée un client temporaire et effectue une recherche.

    Parameters
    ----------
    title, author, ark, date_from, date_to:
        Critères de recherche.
    max_results:
        Nombre maximum de résultats.

    Returns
    -------
    list[GallicaRecord]

    Examples
    --------
    >>> results = search_gallica(title="Froissart", date_from=1380, date_to=1430)
    >>> for r in results[:3]:
    ...     print(r.title, r.ark)
    """
    client = GallicaClient()
    return client.search(
        ark=ark,
        title=title,
        author=author,
        date_from=date_from,
        date_to=date_to,
        max_results=max_results,
    )


def import_gallica_document(
    ark: str,
    pages: str = "all",
    output_dir: Optional[str] = None,
    include_gallica_ocr: bool = True,
) -> Corpus:
    """Importe un document Gallica en une ligne.

    Parameters
    ----------
    ark:
        Identifiant ARK (``'12148/btv1b8453561w'`` ou URL complète).
    pages:
        Sélecteur de pages (``'all'``, ``'1-10'``…).
    output_dir:
        Dossier de sortie.
    include_gallica_ocr:
        Inclure l'OCR Gallica comme GT.

    Returns
    -------
    Corpus
    """
    # Normaliser l'ARK (extraire depuis URL complète si besoin)
    m = re.search(r"ark:/(\d+/\w+)", ark)
    if m:
        ark = m.group(1)

    client = GallicaClient()
    return client.import_document(
        ark=ark,
        pages=pages,
        output_dir=output_dir,
        include_gallica_ocr=include_gallica_ocr,
    )
