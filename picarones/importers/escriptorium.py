"""Intégration eScriptorium — import et export via API REST.

Fonctionnement
--------------
1. Authentification par token (settings → API key dans eScriptorium)
2. Listing et import de projets, documents et transcriptions
3. Export des résultats de benchmark Picarones comme couche OCR dans eScriptorium

API eScriptorium
----------------
eScriptorium expose une API REST documentée à /api/.
Les endpoints principaux utilisés ici :
- GET  /api/projects/                → liste des projets
- GET  /api/documents/               → liste des documents (filtrables par projet)
- GET  /api/documents/{pk}/parts/    → liste des pages d'un document
- GET  /api/documents/{pk}/parts/{pk}/transcriptions/  → transcriptions d'une page
- POST /api/documents/{pk}/parts/{pk}/transcriptions/  → créer une couche OCR

Usage
-----
>>> from picarones.importers.escriptorium import EScriptoriumClient
>>> client = EScriptoriumClient("https://escriptorium.example.org", token="abc123")
>>> projects = client.list_projects()
>>> corpus = client.import_document(doc_id=42, transcription_layer="manual")
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from picarones.core.corpus import Corpus, Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structures de données eScriptorium
# ---------------------------------------------------------------------------

@dataclass
class EScriptoriumProject:
    """Représentation d'un projet eScriptorium."""
    pk: int
    name: str
    slug: str
    owner: str = ""
    document_count: int = 0

    def as_dict(self) -> dict:
        return {
            "pk": self.pk,
            "name": self.name,
            "slug": self.slug,
            "owner": self.owner,
            "document_count": self.document_count,
        }


@dataclass
class EScriptoriumDocument:
    """Représentation d'un document eScriptorium."""
    pk: int
    name: str
    project: str = ""
    part_count: int = 0
    transcription_layers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "pk": self.pk,
            "name": self.name,
            "project": self.project,
            "part_count": self.part_count,
            "transcription_layers": self.transcription_layers,
        }


@dataclass
class EScriptoriumPart:
    """Une page (part) d'un document eScriptorium."""
    pk: int
    title: str
    image_url: str
    order: int = 0
    transcriptions: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Client API eScriptorium
# ---------------------------------------------------------------------------

class EScriptoriumClient:
    """Client pour l'API REST d'eScriptorium.

    Parameters
    ----------
    base_url:
        URL racine de l'instance (ex : ``"https://escriptorium.example.org"``).
    token:
        Token d'authentification API (depuis Settings > API dans eScriptorium).
    timeout:
        Timeout HTTP en secondes.

    Examples
    --------
    >>> client = EScriptoriumClient("https://escriptorium.example.org", token="abc123")
    >>> projects = client.list_projects()
    >>> corpus = client.import_document(42, transcription_layer="manual")
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Token {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """Effectue une requête GET et retourne le JSON."""
        url = f"{self.base_url}/api/{path.lstrip('/')}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"eScriptorium API erreur {exc.code} sur {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Impossible de joindre {self.base_url}: {exc.reason}"
            ) from exc

    def _post(self, path: str, payload: dict) -> dict:
        """Effectue une requête POST avec payload JSON."""
        url = f"{self.base_url}/api/{path.lstrip('/')}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers=self._headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"eScriptorium API erreur {exc.code} sur {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Impossible de joindre {self.base_url}: {exc.reason}"
            ) from exc

    def _paginate(self, path: str, params: Optional[dict] = None) -> list[dict]:
        """Parcourt toutes les pages de résultats paginés."""
        results: list[dict] = []
        current_params = dict(params or {})
        current_params.setdefault("page_size", 100)
        page_num = 1
        while True:
            current_params["page"] = page_num
            data = self._get(path, current_params)
            if isinstance(data, list):
                results.extend(data)
                break
            results.extend(data.get("results", []))
            if not data.get("next"):
                break
            page_num += 1
        return results

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Vérifie que l'URL et le token sont valides.

        Returns
        -------
        bool
            True si l'authentification réussit.
        """
        try:
            self._get("projects/", {"page_size": 1})
            return True
        except RuntimeError:
            return False

    def list_projects(self) -> list[EScriptoriumProject]:
        """Retourne la liste des projets accessibles.

        Returns
        -------
        list[EScriptoriumProject]
        """
        raw = self._paginate("projects/")
        projects = []
        for item in raw:
            projects.append(EScriptoriumProject(
                pk=item["pk"],
                name=item.get("name", ""),
                slug=item.get("slug", ""),
                owner=item.get("owner", {}).get("username", "") if isinstance(item.get("owner"), dict) else str(item.get("owner", "")),
                document_count=item.get("documents_count", 0),
            ))
        return projects

    def list_documents(
        self,
        project_pk: Optional[int] = None,
    ) -> list[EScriptoriumDocument]:
        """Retourne la liste des documents, filtrés par projet si fourni.

        Parameters
        ----------
        project_pk:
            PK du projet eScriptorium (optionnel).

        Returns
        -------
        list[EScriptoriumDocument]
        """
        params: dict = {}
        if project_pk is not None:
            params["project"] = project_pk
        raw = self._paginate("documents/", params)
        docs = []
        for item in raw:
            layers = [
                t.get("name", "") if isinstance(t, dict) else str(t)
                for t in item.get("transcriptions", [])
            ]
            docs.append(EScriptoriumDocument(
                pk=item["pk"],
                name=item.get("name", ""),
                project=str(item.get("project", "")),
                part_count=item.get("parts_count", 0),
                transcription_layers=layers,
            ))
        return docs

    def list_parts(self, doc_pk: int) -> list[EScriptoriumPart]:
        """Retourne les pages (parts) d'un document.

        Parameters
        ----------
        doc_pk:
            PK du document eScriptorium.

        Returns
        -------
        list[EScriptoriumPart]
        """
        raw = self._paginate(f"documents/{doc_pk}/parts/")
        parts = []
        for item in raw:
            parts.append(EScriptoriumPart(
                pk=item["pk"],
                title=item.get("title", "") or f"Part {item.get('order', 0) + 1}",
                image_url=item.get("image", "") or "",
                order=item.get("order", 0),
            ))
        return parts

    def get_transcriptions(self, doc_pk: int, part_pk: int) -> list[dict]:
        """Retourne les transcriptions disponibles pour une page.

        Parameters
        ----------
        doc_pk:
            PK du document.
        part_pk:
            PK de la page.

        Returns
        -------
        list[dict]
            Chaque dict contient ``{"name": str, "content": str}``.
        """
        raw = self._get(f"documents/{doc_pk}/parts/{part_pk}/transcriptions/")
        if isinstance(raw, list):
            return raw
        return raw.get("results", [])

    def import_document(
        self,
        doc_pk: int,
        transcription_layer: str = "manual",
        output_dir: Optional[str] = None,
        download_images: bool = True,
        show_progress: bool = True,
    ) -> Corpus:
        """Importe un document eScriptorium comme corpus Picarones.

        Télécharge les images et récupère les transcriptions de la couche
        spécifiée comme vérité terrain.

        Parameters
        ----------
        doc_pk:
            PK du document dans eScriptorium.
        transcription_layer:
            Nom de la couche de transcription à utiliser comme GT.
        output_dir:
            Dossier local pour les images téléchargées. Si None, les images
            sont stockées en mémoire (pas de sauvegarde sur disque).
        download_images:
            Si True, télécharge les images dans output_dir.
        show_progress:
            Affiche une barre de progression tqdm.

        Returns
        -------
        Corpus
            Corpus Picarones avec documents et GT.
        """
        # Récupérer les métadonnées du document
        doc_info = self._get(f"documents/{doc_pk}/")
        doc_name = doc_info.get("name", f"document_{doc_pk}")

        parts = self.list_parts(doc_pk)
        if not parts:
            raise ValueError(f"Aucune page trouvée dans le document {doc_pk}")

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(parts, desc=f"Import {doc_name}")
            except ImportError:
                iterator = iter(parts)
        else:
            iterator = iter(parts)

        out_path: Optional[Path] = None
        if output_dir and download_images:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

        documents: list[Document] = []
        for part in iterator:
            # Récupérer les transcriptions
            transcriptions = self.get_transcriptions(doc_pk, part.pk)
            gt_text = ""
            for t in transcriptions:
                layer_name = t.get("transcription", {}).get("name", "") if isinstance(t.get("transcription"), dict) else t.get("name", "")
                if layer_name == transcription_layer or not transcription_layer:
                    # Le contenu est dans "content" ou dans les lignes
                    lines = t.get("lines", []) or []
                    if lines:
                        gt_text = "\n".join(
                            line.get("content", "") or ""
                            for line in lines
                            if line.get("content")
                        )
                    else:
                        gt_text = t.get("content", "") or ""
                    break

            # Image
            image_path = part.image_url or f"escriptorium://doc{doc_pk}/part{part.pk}"
            if out_path and part.image_url and download_images:
                ext = Path(urllib.parse.urlparse(part.image_url).path).suffix or ".jpg"
                local_img = out_path / f"part_{part.pk:05d}{ext}"
                try:
                    urllib.request.urlretrieve(part.image_url, local_img)
                    image_path = str(local_img)
                except Exception as exc:
                    logger.warning("Impossible de télécharger l'image %s: %s", part.image_url, exc)

                # Sauvegarder la GT
                gt_path = out_path / f"part_{part.pk:05d}.gt.txt"
                gt_path.write_text(gt_text, encoding="utf-8")

            documents.append(Document(
                doc_id=f"part_{part.pk:05d}",
                image_path=image_path,
                ground_truth=gt_text,
                metadata={
                    "source": "escriptorium",
                    "doc_pk": doc_pk,
                    "part_pk": part.pk,
                    "part_title": part.title,
                    "transcription_layer": transcription_layer,
                },
            ))

        return Corpus(
            name=doc_name,
            source=f"{self.base_url}/document/{doc_pk}/",
            documents=documents,
            metadata={
                "escriptorium_url": self.base_url,
                "doc_pk": doc_pk,
                "transcription_layer": transcription_layer,
            },
        )

    def export_benchmark_as_layer(
        self,
        benchmark_result: "BenchmarkResult",
        doc_pk: int,
        engine_name: str,
        layer_name: Optional[str] = None,
        part_mapping: Optional[dict[str, int]] = None,
    ) -> int:
        """Exporte les résultats Picarones comme couche OCR dans eScriptorium.

        Parameters
        ----------
        benchmark_result:
            Résultats du benchmark Picarones.
        doc_pk:
            PK du document cible dans eScriptorium.
        engine_name:
            Nom du moteur dont on exporte les transcriptions.
        layer_name:
            Nom de la couche à créer (défaut : ``"picarones_{engine_name}"``).
        part_mapping:
            Correspondance ``doc_id → part_pk`` eScriptorium. Si None,
            la correspondance est inférée depuis les métadonnées des documents.

        Returns
        -------
        int
            Nombre de pages exportées avec succès.
        """
        if layer_name is None:
            layer_name = f"picarones_{engine_name}"

        # Trouver le rapport du moteur
        engine_report = None
        for report in benchmark_result.engine_reports:
            if report.engine_name == engine_name:
                engine_report = report
                break
        if engine_report is None:
            raise ValueError(f"Moteur '{engine_name}' introuvable dans les résultats.")

        exported = 0
        for doc_result in engine_report.document_results:
            if doc_result.engine_error:
                continue

            # Déterminer le part_pk
            part_pk: Optional[int] = None
            if part_mapping and doc_result.doc_id in part_mapping:
                part_pk = part_mapping[doc_result.doc_id]
            else:
                # Essayer d'extraire depuis doc_id (ex: "part_00042")
                try:
                    part_pk = int(doc_result.doc_id.replace("part_", "").lstrip("0") or "0")
                except ValueError:
                    logger.warning("Impossible de déterminer part_pk pour %s", doc_result.doc_id)
                    continue

            try:
                self._post(
                    f"documents/{doc_pk}/parts/{part_pk}/transcriptions/",
                    {
                        "name": layer_name,
                        "content": doc_result.hypothesis,
                        "source": "picarones",
                    },
                )
                exported += 1
                logger.debug("Exporté part %d → couche '%s'", part_pk, layer_name)
            except RuntimeError as exc:
                logger.warning("Erreur export part %d: %s", part_pk, exc)

        return exported


# ---------------------------------------------------------------------------
# Interface de niveau module
# ---------------------------------------------------------------------------

def connect_escriptorium(
    base_url: str,
    token: str,
    timeout: int = 30,
) -> EScriptoriumClient:
    """Crée et retourne un client eScriptorium authentifié.

    Parameters
    ----------
    base_url:
        URL de l'instance eScriptorium.
    token:
        Token API.
    timeout:
        Timeout HTTP.

    Returns
    -------
    EScriptoriumClient

    Raises
    ------
    RuntimeError
        Si la connexion échoue (URL invalide, token incorrect, serveur inaccessible).
    """
    client = EScriptoriumClient(base_url, token, timeout)
    if not client.test_connection():
        raise RuntimeError(
            f"Impossible de se connecter à {base_url}. "
            "Vérifiez l'URL et le token API."
        )
    return client
