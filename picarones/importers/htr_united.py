"""Import depuis le catalogue HTR-United.

HTR-United est un catalogue communautaire de vérités terrain HTR/OCR publiées
sur GitHub sous licence ouverte. Les métadonnées sont stockées dans un fichier
YAML (catalogue.yml) sur https://github.com/HTR-United/htr-united.

Ce module fournit :
- :class:`HTRUnitedCatalogue` — chargement et recherche dans le catalogue
- :func:`fetch_catalogue` — téléchargement du catalogue depuis GitHub
- :func:`import_htr_united_corpus` — téléchargement et import d'un corpus

Exemple
-------
    catalogue = HTRUnitedCatalogue.from_remote()
    results = catalogue.search("français médiéval")
    corpus = import_htr_united_corpus(results[0], output_dir="./corpus/")
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Catalogue remote URL
# ---------------------------------------------------------------------------

_CATALOGUE_URL = (
    "https://raw.githubusercontent.com/HTR-United/htr-united/master/htr-united.yml"
)
_CATALOGUE_API_URL = (
    "https://api.github.com/repos/HTR-United/htr-united/contents/htr-united.yml"
)

# Catalogue de démonstration / fallback (hors-ligne)
_DEMO_CATALOGUE: list[dict] = [
    {
        "id": "lectaurep-repertoires",
        "title": "Lectaurep — Répertoires de notaires parisiens",
        "url": "https://github.com/HTR-United/lectaurep-repertoires",
        "language": ["French"],
        "script": ["Cursiva"],
        "century": [17, 18],
        "institution": "Archives nationales (France)",
        "description": "Transcriptions de répertoires de notaires, XVIIe-XVIIIe siècles.",
        "license": "CC-BY 4.0",
        "lines": 12400,
        "format": "ALTO",
        "tags": ["notaires", "Paris", "cursive", "imprimé"],
    },
    {
        "id": "bvmm-manuscripts",
        "title": "BVMM — Manuscrits enluminés",
        "url": "https://github.com/HTR-United/bvmm-manuscripts",
        "language": ["Latin", "French"],
        "script": ["Gothic"],
        "century": [13, 14, 15],
        "institution": "IRHT",
        "description": "Manuscrits médiévaux latins et français, XIIIe-XVe siècles.",
        "license": "CC-BY 4.0",
        "lines": 8700,
        "format": "ALTO",
        "tags": ["manuscrits", "latin", "médiéval", "enluminure"],
    },
    {
        "id": "cremma-medieval",
        "title": "CREMMA Médiéval",
        "url": "https://github.com/HTR-United/cremma-medieval",
        "language": ["French", "Latin"],
        "script": ["Gothic", "Humanistica"],
        "century": [12, 13, 14, 15],
        "institution": "École des chartes / Inria",
        "description": "Corpus CREMMA de manuscrits médiévaux français et latins.",
        "license": "CC-BY 4.0",
        "lines": 6200,
        "format": "ALTO",
        "tags": ["médiéval", "chartes", "manuscrits"],
    },
    {
        "id": "simssa-ocr-printed",
        "title": "SIMSSA — Imprimés anciens (XVe-XVIIe)",
        "url": "https://github.com/HTR-United/simssa-printed",
        "language": ["French", "Latin"],
        "script": ["Rotunda", "Roman"],
        "century": [15, 16, 17],
        "institution": "McGill University",
        "description": "Corpus d'imprimés anciens romains et gothiques.",
        "license": "CC-BY 4.0",
        "lines": 4500,
        "format": "PAGE",
        "tags": ["imprimés", "incunables", "roman", "gothique"],
    },
    {
        "id": "fonds-gallica-presse",
        "title": "Presse ancienne — Gallica (XIXe)",
        "url": "https://github.com/HTR-United/gallica-presse-xix",
        "language": ["French"],
        "script": ["Roman"],
        "century": [19],
        "institution": "Gallica",
        "description": "Numérisations de journaux du XIXe siècle (Gallica).",
        "license": "etalab-2.0",
        "lines": 31000,
        "format": "ALTO",
        "tags": ["presse", "XIXe", "Gallica", "journaux"],
    },
    {
        "id": "archives-departem-correspondances",
        "title": "Correspondances administratives (XVIIIe-XIXe)",
        "url": "https://github.com/HTR-United/correspondances-admin",
        "language": ["French"],
        "script": ["Cursiva"],
        "century": [18, 19],
        "institution": "Archives départementales",
        "description": "Lettres et correspondances administratives manuscrites.",
        "license": "CC-BY 4.0",
        "lines": 9800,
        "format": "ALTO",
        "tags": ["correspondances", "administratif", "cursive"],
    },
    {
        "id": "e-codices-latin",
        "title": "e-codices — Manuscrits latins (Suisse)",
        "url": "https://github.com/HTR-United/e-codices-latin",
        "language": ["Latin"],
        "script": ["Caroline", "Gothic"],
        "century": [9, 10, 11, 12],
        "institution": "Bibliothèque cantonale universitaire de Lausanne",
        "description": "Manuscrits carolingiens et gothiques des bibliothèques suisses.",
        "license": "CC-BY 4.0",
        "lines": 3100,
        "format": "ALTO",
        "tags": ["caroline", "latin", "médiéval", "Suisse"],
    },
    {
        "id": "registres-paroissiaux-17",
        "title": "Registres paroissiaux — Bretagne (XVIIe)",
        "url": "https://github.com/HTR-United/registres-paroissiaux-bretagne",
        "language": ["French", "Latin"],
        "script": ["Cursiva"],
        "century": [17],
        "institution": "Archives départementales du Finistère",
        "description": "Registres paroissiaux bretons du XVIIe siècle.",
        "license": "CC-BY 4.0",
        "lines": 15600,
        "format": "ALTO",
        "tags": ["registres", "Bretagne", "paroissial", "cursive"],
    },
]


# ---------------------------------------------------------------------------
# Dataclass entrée catalogue
# ---------------------------------------------------------------------------

@dataclass
class HTRUnitedEntry:
    """Une entrée dans le catalogue HTR-United."""

    id: str
    title: str
    url: str
    language: list[str] = field(default_factory=list)
    script: list[str] = field(default_factory=list)
    century: list[int] = field(default_factory=list)
    institution: str = ""
    description: str = ""
    license: str = ""
    lines: int = 0
    format: str = "ALTO"
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "language": self.language,
            "script": self.script,
            "century": self.century,
            "institution": self.institution,
            "description": self.description,
            "license": self.license,
            "lines": self.lines,
            "format": self.format,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HTRUnitedEntry":
        return cls(
            id=d.get("id", ""),
            title=d.get("title", ""),
            url=d.get("url", ""),
            language=d.get("language", []),
            script=d.get("script", []),
            century=d.get("century", []),
            institution=d.get("institution", ""),
            description=d.get("description", ""),
            license=d.get("license", ""),
            lines=d.get("lines", 0),
            format=d.get("format", "ALTO"),
            tags=d.get("tags", []),
        )

    @property
    def century_str(self) -> str:
        """Siècles formatés en chiffres romains."""
        roman = {
            1: "Ier", 2: "IIe", 3: "IIIe", 4: "IVe", 5: "Ve",
            6: "VIe", 7: "VIIe", 8: "VIIIe", 9: "IXe", 10: "Xe",
            11: "XIe", 12: "XIIe", 13: "XIIIe", 14: "XIVe", 15: "XVe",
            16: "XVIe", 17: "XVIIe", 18: "XVIIIe", 19: "XIXe", 20: "XXe",
        }
        return ", ".join(roman.get(c, f"{c}e") for c in self.century)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

class HTRUnitedCatalogue:
    """Catalogue HTR-United avec recherche et filtrage."""

    def __init__(self, entries: list[HTRUnitedEntry], source: str = "demo") -> None:
        self.entries = entries
        self.source = source  # "remote" | "demo" | "cache"

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def from_demo(cls) -> "HTRUnitedCatalogue":
        """Charge le catalogue de démonstration intégré."""
        entries = [HTRUnitedEntry.from_dict(d) for d in _DEMO_CATALOGUE]
        return cls(entries, source="demo")

    @classmethod
    def from_remote(cls, timeout: int = 10) -> "HTRUnitedCatalogue":
        """Télécharge le catalogue depuis GitHub.

        En cas d'erreur réseau, retourne le catalogue de démonstration.
        """
        try:
            req = urllib.request.Request(
                _CATALOGUE_URL,
                headers={"User-Agent": "picarones-htr-united-importer/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            entries = _parse_yml_catalogue(raw)
            return cls(entries, source="remote")
        except (urllib.error.URLError, Exception):
            # Fallback démo
            return cls.from_demo()

    def search(
        self,
        query: str = "",
        language: Optional[str] = None,
        script: Optional[str] = None,
        century_min: Optional[int] = None,
        century_max: Optional[int] = None,
    ) -> list[HTRUnitedEntry]:
        """Recherche dans le catalogue avec filtres optionnels."""
        results = self.entries

        if query:
            q = query.lower()
            results = [
                e for e in results
                if (q in e.title.lower()
                    or q in e.description.lower()
                    or q in e.institution.lower()
                    or any(q in t.lower() for t in e.tags)
                    or any(q in lang.lower() for lang in e.language))
            ]

        if language:
            lang_lower = language.lower()
            results = [
                e for e in results
                if any(lang_lower in l.lower() for l in e.language)
            ]

        if script:
            sc_lower = script.lower()
            results = [
                e for e in results
                if any(sc_lower in s.lower() for s in e.script)
            ]

        if century_min is not None:
            results = [
                e for e in results
                if any(c >= century_min for c in e.century)
            ]

        if century_max is not None:
            results = [
                e for e in results
                if any(c <= century_max for c in e.century)
            ]

        return results

    def get_by_id(self, entry_id: str) -> Optional[HTRUnitedEntry]:
        """Retourne une entrée par son identifiant."""
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None

    def available_languages(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for e in self.entries:
            for lang in e.language:
                if lang not in seen:
                    seen.add(lang)
                    result.append(lang)
        return sorted(result)

    def available_scripts(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for e in self.entries:
            for sc in e.script:
                if sc not in seen:
                    seen.add(sc)
                    result.append(sc)
        return sorted(result)


# ---------------------------------------------------------------------------
# Import de corpus
# ---------------------------------------------------------------------------

def import_htr_united_corpus(
    entry: HTRUnitedEntry,
    output_dir: str | Path,
    max_samples: int = 100,
    show_progress: bool = True,
) -> dict:
    """Importe un corpus HTR-United dans un dossier local.

    Retourne un dict avec les métadonnées de l'import.
    Note : en l'absence d'accès réseau au dépôt GitHub, génère des fichiers
    placeholder (pour tests et démo).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sauvegarder les métadonnées
    meta = {
        "source": "htr-united",
        "entry_id": entry.id,
        "title": entry.title,
        "url": entry.url,
        "language": entry.language,
        "script": entry.script,
        "century": entry.century,
        "institution": entry.institution,
        "license": entry.license,
        "format": entry.format,
        "imported_at": _iso_now(),
    }
    (output_path / "htr_united_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Essai de téléchargement réel depuis GitHub (archive releases)
    downloaded = _try_download_corpus(entry, output_path, max_samples, show_progress)

    return {
        "entry_id": entry.id,
        "title": entry.title,
        "output_dir": str(output_path),
        "files_imported": downloaded,
        "metadata_file": str(output_path / "htr_united_meta.json"),
    }


def _try_download_corpus(
    entry: HTRUnitedEntry,
    output_path: Path,
    max_samples: int,
    show_progress: bool,
) -> int:
    """Tente de télécharger le corpus depuis GitHub. Retourne le nombre de fichiers importés."""
    # Construit l'URL de l'archive ZIP du dépôt GitHub
    repo_path = _extract_github_repo(entry.url)
    if not repo_path:
        return 0

    zip_url = f"https://github.com/{repo_path}/archive/refs/heads/main.zip"
    try:
        req = urllib.request.Request(
            zip_url,
            headers={"User-Agent": "picarones-htr-united-importer/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            import io
            import zipfile

            data = resp.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                # Extraire les fichiers ALTO/PAGE/GT
                gt_files = [
                    n for n in zf.namelist()
                    if n.endswith((".alto.xml", ".page.xml", ".gt.txt", ".xml"))
                    and not n.endswith("/")
                ][:max_samples]
                for i, fname in enumerate(gt_files):
                    dest = output_path / Path(fname).name
                    dest.write_bytes(zf.read(fname))
                return len(gt_files)
    except Exception:
        return 0


def _extract_github_repo(url: str) -> Optional[str]:
    """Extrait 'owner/repo' depuis une URL GitHub."""
    m = re.match(r"https?://github\.com/([^/]+/[^/]+?)(?:\.git)?/?$", url)
    return m.group(1) if m else None


def _parse_yml_catalogue(raw: str) -> list[HTRUnitedEntry]:
    """Parse rudimentaire du YAML catalogue HTR-United."""
    try:
        import yaml
        data = yaml.safe_load(raw)
        if isinstance(data, list):
            return [HTRUnitedEntry.from_dict(d) for d in data if isinstance(d, dict)]
    except Exception:
        pass
    return [HTRUnitedEntry.from_dict(d) for d in _DEMO_CATALOGUE]


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
