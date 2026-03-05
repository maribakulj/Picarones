"""Import de datasets OCR/HTR depuis HuggingFace Hub.

Ce module fournit :
- :class:`HuggingFaceDataset` — métadonnées d'un dataset HuggingFace
- :class:`HuggingFaceImporter` — recherche et import de datasets
- :func:`search_hf_datasets` — recherche par tags dans l'API HuggingFace
- :func:`import_hf_dataset` — téléchargement d'un dataset vers un dossier local

Les datasets patrimoniaux de référence sont pré-référencés pour une découverte
rapide sans requête réseau.

Exemple
-------
    importer = HuggingFaceImporter()
    results = importer.search("medieval OCR", tags=["ocr"])
    corpus = importer.import_dataset(results[0].dataset_id, output_dir="./corpus/")
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Datasets de référence pré-référencés
# ---------------------------------------------------------------------------

_REFERENCE_DATASETS: list[dict] = [
    {
        "dataset_id": "Teklia/RIMES",
        "title": "RIMES — Reconnaissance et Indexation de données Manuscrites et de fac-similEs",
        "description": "Corpus de courriers manuscrits français modernes. Standard de référence pour la reconnaissance d'écriture manuscrite.",
        "language": ["French"],
        "tags": ["htr", "ocr", "handwritten", "french", "modern"],
        "license": "cc-by-4.0",
        "size_category": "1K<n<10K",
        "task": "image-to-text",
        "institution": "IRISA / A2iA",
        "downloads": 1200,
    },
    {
        "dataset_id": "Teklia/IAM",
        "title": "IAM Handwriting Database",
        "description": "Corpus de référence anglais pour la reconnaissance d'écriture manuscrite.",
        "language": ["English"],
        "tags": ["htr", "ocr", "handwritten", "english"],
        "license": "other",
        "size_category": "10K<n<100K",
        "task": "image-to-text",
        "institution": "University of Bern",
        "downloads": 8400,
    },
    {
        "dataset_id": "CATMuS/medieval",
        "title": "CATMuS Medieval — Consistent Approaches to Transcribing ManuScripts",
        "description": "Dataset multilingue de manuscrits médiévaux (latin, français, occitan, espagnol) pour l'entraînement de modèles HTR.",
        "language": ["Latin", "French", "Occitan", "Spanish"],
        "tags": ["htr", "medieval", "manuscripts", "latin", "french", "historical"],
        "license": "cc-by-4.0",
        "size_category": "100K<n<1M",
        "task": "image-to-text",
        "institution": "Inria / EPHE",
        "downloads": 3100,
    },
    {
        "dataset_id": "htr-united/cremma-medieval",
        "title": "CREMMA Medieval",
        "description": "Corpus de manuscrits médiévaux français XIIe-XVe siècles.",
        "language": ["French", "Latin"],
        "tags": ["htr", "medieval", "french", "manuscripts", "htr-united"],
        "license": "cc-by-4.0",
        "size_category": "1K<n<10K",
        "task": "image-to-text",
        "institution": "Inria",
        "downloads": 520,
    },
    {
        "dataset_id": "biglam/europeana_newspapers",
        "title": "Europeana Newspapers",
        "description": "Journaux numérisés européens du XIXe siècle (OCR + images).",
        "language": ["French", "German", "Dutch", "Finnish"],
        "tags": ["ocr", "newspapers", "historical", "19th-century", "europeana"],
        "license": "cc0-1.0",
        "size_category": "1M<n<10M",
        "task": "image-to-text",
        "institution": "Europeana Foundation",
        "downloads": 15200,
    },
    {
        "dataset_id": "stefanklut/esposalles",
        "title": "Esposalles Dataset",
        "description": "Registres de mariage catalans du XVIIe siècle pour la reconnaissance d'écriture historique.",
        "language": ["Catalan", "Latin"],
        "tags": ["htr", "historical", "registers", "catalan", "17th-century"],
        "license": "cc-by-4.0",
        "size_category": "1K<n<10K",
        "task": "image-to-text",
        "institution": "Universitat Autònoma de Barcelona",
        "downloads": 340,
    },
    {
        "dataset_id": "bnf-gallica/gallica-ocr",
        "title": "Gallica OCR — BnF",
        "description": "Extraits d'imprimés anciens numérisés depuis Gallica avec vérité terrain.",
        "language": ["French", "Latin"],
        "tags": ["ocr", "historical", "printed", "gallica", "bnf", "french"],
        "license": "etalab-2.0",
        "size_category": "10K<n<100K",
        "task": "image-to-text",
        "institution": "Bibliothèque nationale de France",
        "downloads": 2800,
    },
    {
        "dataset_id": "Bozen-Baptism/baptism-records",
        "title": "Bozen Baptism Records",
        "description": "Registres de baptêmes de Bozen (Italie/Autriche) du XVIIIe siècle.",
        "language": ["German", "Latin"],
        "tags": ["htr", "historical", "registers", "german", "latin", "18th-century"],
        "license": "cc-by-4.0",
        "size_category": "1K<n<10K",
        "task": "image-to-text",
        "institution": "University of Innsbruck",
        "downloads": 190,
    },
    {
        "dataset_id": "read-bad/readbad",
        "title": "READ-BAD — Recognition and Enrichment of Archival Documents",
        "description": "Corpus multilingue de documents d'archives pour l'OCR historique (Latin, Allemand, Anglais).",
        "language": ["German", "English", "Latin"],
        "tags": ["ocr", "htr", "historical", "archives", "read"],
        "license": "cc-by-4.0",
        "size_category": "10K<n<100K",
        "task": "image-to-text",
        "institution": "University of Graz",
        "downloads": 1050,
    },
]

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class HuggingFaceDataset:
    """Métadonnées d'un dataset HuggingFace."""

    dataset_id: str
    title: str
    description: str = ""
    language: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    license: str = ""
    size_category: str = ""
    task: str = "image-to-text"
    institution: str = ""
    downloads: int = 0
    source: str = "reference"  # "reference" | "api"

    def as_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "title": self.title,
            "description": self.description,
            "language": self.language,
            "tags": self.tags,
            "license": self.license,
            "size_category": self.size_category,
            "task": self.task,
            "institution": self.institution,
            "downloads": self.downloads,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HuggingFaceDataset":
        return cls(
            dataset_id=d.get("dataset_id", d.get("id", "")),
            title=d.get("title", d.get("dataset_id", "")),
            description=d.get("description", ""),
            language=d.get("language", []),
            tags=d.get("tags", []),
            license=d.get("license", ""),
            size_category=d.get("size_category", d.get("cardData", {}).get("size_categories", [""])[0] if isinstance(d.get("cardData"), dict) else ""),
            task=d.get("task", "image-to-text"),
            institution=d.get("institution", ""),
            downloads=d.get("downloads", d.get("downloadsAllTime", 0)),
            source=d.get("source", "api"),
        )

    @property
    def hf_url(self) -> str:
        return f"https://huggingface.co/datasets/{self.dataset_id}"


# ---------------------------------------------------------------------------
# Importer principal
# ---------------------------------------------------------------------------

class HuggingFaceImporter:
    """Recherche et importe des datasets depuis HuggingFace Hub."""

    _API_BASE = "https://huggingface.co/api"

    def __init__(self, token: Optional[str] = None) -> None:
        self._token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    def _headers(self) -> dict:
        h = {"User-Agent": "picarones-hf-importer/1.0"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def search(
        self,
        query: str = "",
        tags: Optional[list[str]] = None,
        language: Optional[str] = None,
        limit: int = 20,
        use_reference: bool = True,
    ) -> list[HuggingFaceDataset]:
        """Recherche des datasets avec filtres.

        Interroge d'abord les datasets de référence pré-intégrés, puis
        l'API HuggingFace si disponible.
        """
        results: list[HuggingFaceDataset] = []

        # Datasets de référence
        if use_reference:
            ref_results = self._search_reference(query, tags, language)
            results.extend(ref_results)

        # API HuggingFace (optionnel, peut échouer silencieusement)
        try:
            api_results = self._search_api(query, tags, language, limit)
            # Déduplique (priorité aux références)
            existing_ids = {r.dataset_id for r in results}
            for ds in api_results:
                if ds.dataset_id not in existing_ids:
                    results.append(ds)
                    existing_ids.add(ds.dataset_id)
        except Exception:
            pass

        return results[:limit]

    def _search_reference(
        self,
        query: str,
        tags: Optional[list[str]],
        language: Optional[str],
    ) -> list[HuggingFaceDataset]:
        datasets = [HuggingFaceDataset.from_dict(d) for d in _REFERENCE_DATASETS]
        datasets = [ds._replace_source("reference") for ds in datasets]

        if query:
            q = query.lower()
            datasets = [
                ds for ds in datasets
                if (q in ds.title.lower()
                    or q in ds.description.lower()
                    or q in ds.dataset_id.lower()
                    or any(q in t.lower() for t in ds.tags)
                    or any(q in l.lower() for l in ds.language))
            ]

        if tags:
            for tag in tags:
                t_lower = tag.lower()
                datasets = [
                    ds for ds in datasets
                    if any(t_lower in dt.lower() for dt in ds.tags)
                ]

        if language:
            lang_lower = language.lower()
            datasets = [
                ds for ds in datasets
                if any(lang_lower in l.lower() for l in ds.language)
            ]

        return datasets

    def _search_api(
        self,
        query: str,
        tags: Optional[list[str]],
        language: Optional[str],
        limit: int,
    ) -> list[HuggingFaceDataset]:
        params: dict[str, str] = {
            "task_categories": "image-to-text",
            "limit": str(min(limit, 50)),
            "full": "False",
        }
        if query:
            params["search"] = query
        if language:
            params["language"] = language
        if tags:
            params["tags"] = ",".join(tags)

        url = f"{self._API_BASE}/datasets?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []
        for item in data if isinstance(data, list) else []:
            ds = HuggingFaceDataset(
                dataset_id=item.get("id", ""),
                title=item.get("id", ""),
                description=item.get("description", ""),
                language=item.get("language", []),
                tags=item.get("tags", []),
                license=item.get("license", ""),
                size_category=(
                    item.get("cardData", {}).get("size_categories", [""])[0]
                    if isinstance(item.get("cardData"), dict)
                    else ""
                ),
                task="image-to-text",
                downloads=item.get("downloadsAllTime", 0),
                source="api",
            )
            if ds.dataset_id:
                results.append(ds)
        return results

    def import_dataset(
        self,
        dataset_id: str,
        output_dir: str | Path,
        split: str = "train",
        max_samples: int = 100,
        show_progress: bool = True,
    ) -> dict:
        """Importe un dataset depuis HuggingFace vers un dossier local.

        Retourne les métadonnées de l'import.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        meta = {
            "source": "huggingface",
            "dataset_id": dataset_id,
            "split": split,
            "max_samples": max_samples,
            "imported_at": _iso_now(),
        }
        meta_file = output_path / "huggingface_meta.json"
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # Tentative d'import via datasets library si disponible
        files_imported = _try_import_with_datasets_lib(
            dataset_id, output_path, split, max_samples, show_progress
        )

        return {
            "dataset_id": dataset_id,
            "output_dir": str(output_path),
            "files_imported": files_imported,
            "metadata_file": str(meta_file),
        }


def _try_import_with_datasets_lib(
    dataset_id: str,
    output_path: Path,
    split: str,
    max_samples: int,
    show_progress: bool,
) -> int:
    """Essaie d'importer avec la librairie `datasets` de HuggingFace."""
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(dataset_id, split=split, streaming=True)
        count = 0
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            # Cherche champ image et texte
            image = item.get("image") or item.get("img")
            text = item.get("text") or item.get("transcription") or item.get("ground_truth", "")

            if image is not None:
                img_file = output_path / f"doc_{i:04d}.jpg"
                try:
                    image.save(str(img_file))
                except Exception:
                    pass

            gt_file = output_path / f"doc_{i:04d}.gt.txt"
            gt_file.write_text(str(text), encoding="utf-8")
            count += 1

        return count
    except (ImportError, Exception):
        return 0


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Extension de HuggingFaceDataset (helper privé)
# ---------------------------------------------------------------------------

def _patch_dataset_replace_source() -> None:
    """Ajoute un helper _replace_source à HuggingFaceDataset."""
    def _replace_source(self, source: str) -> "HuggingFaceDataset":
        from dataclasses import replace
        return replace(self, source=source)
    HuggingFaceDataset._replace_source = _replace_source


_patch_dataset_replace_source()
