"""Chargement et préparation des assets du rapport HTML.

Ce module concentre tout ce qui touche aux ressources binaires
embarquées ou référencées par le rapport :

- ``load_vendor_js`` lit un fichier JS vendorisé (Chart.js, etc.).
- ``encode_image_b64`` redimensionne et encode une image en data-URI.
- ``encode_images_b64_from_result`` itère sur un BenchmarkResult.
- ``externalize_images_to_dir`` écrit les images sur disque à côté
  du HTML (mode ``--lazy-images`` du Sprint A5).

Extrait de ``picarones/report/generator.py`` lors du sprint de
découpage : isole l'I/O image et vendor du reste de l'orchestration.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from picarones.core.results import BenchmarkResult

logger = logging.getLogger(__name__)

#: Dossier où sont stockées les ressources JS embarquées.
_VENDOR_DIR = Path(__file__).parent / "vendor"


def load_vendor_js(name: str) -> str:
    """Lit un fichier JS vendorisé et retourne son contenu.

    Si le fichier n'existe pas, retourne un commentaire JS qui
    garde le rapport valide (pas de SyntaxError côté navigateur).
    """
    p = _VENDOR_DIR / name
    if p.exists():
        return p.read_text(encoding="utf-8")
    return f"/* vendor/{name} non trouvé */"


def encode_image_b64(image_path: str, max_width: int = 1200) -> str:
    """Lit une image, la redimensionne si besoin, et retourne un data-URI base64.

    Retourne ``""`` si l'image est introuvable ou si l'encodage
    échoue (Pillow indisponible, format non géré, fichier corrompu).
    Logue un avertissement dans ce dernier cas — le rapport reste
    fonctionnel mais l'image manquera dans la galerie.
    """
    p = Path(image_path)
    if not p.exists():
        return ""
    try:
        from PIL import Image

        with Image.open(p) as img:
            if img.width > max_width:
                ratio = max_width / img.width
                new_h = max(1, int(img.height * ratio))
                img = img.resize((max_width, new_h), Image.LANCZOS)
            # Convertir en RGB pour éviter les problèmes de mode (RGBA, palette…)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            fmt = "JPEG" if p.suffix.lower() in (".jpg", ".jpeg") else "PNG"
            img.save(buf, format=fmt, optimize=True, quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            mime = "image/jpeg" if fmt == "JPEG" else "image/png"
            return f"data:{mime};base64,{b64}"
    except Exception as exc:  # noqa: BLE001 — fallback gracieux + warning
        logger.warning(
            "[report] échec d'encodage base64 de l'image %s : %s — "
            "le rapport ignorera cette image",
            image_path,
            exc,
        )
        return ""


def encode_images_b64_from_result(
    benchmark: "BenchmarkResult", max_width: int = 1200,
) -> dict[str, str]:
    """Encode toutes les images d'un BenchmarkResult en base64.

    Returns
    -------
    dict
        ``{doc_id: data_uri}``
    """
    images: dict[str, str] = {}
    if not benchmark.engine_reports:
        return images
    for dr in benchmark.engine_reports[0].document_results:
        if dr.image_path and dr.doc_id not in images:
            uri = encode_image_b64(dr.image_path, max_width=max_width)
            if uri:
                images[dr.doc_id] = uri
    return images


def externalize_images_to_dir(
    benchmark: "BenchmarkResult",
    output_dir: Path,
    max_width: int = 1200,
    asset_subdir: str = "report-assets",
) -> dict[str, str]:
    """Sprint A5 (item M-16) — écrit les images sur disque dans un
    sous-dossier à côté du HTML, et retourne ``{doc_id: url_relative}``.

    Mode « lazy loading » : au lieu d'embarquer chaque image en
    base64 dans le HTML (50 MB+ pour un corpus de 100 documents,
    ~200 MB+ pour 1 000 documents), on les externalise en fichiers
    PNG/JPEG locaux. Le HTML les référence via
    ``<img src="report-assets/…">`` avec ``loading="lazy"`` côté
    navigateur.

    Le rapport reste auto-portant si l'utilisateur copie le dossier
    ``report-assets/`` à côté du HTML (cf. CLI ``--lazy-images``).

    Parameters
    ----------
    benchmark:
        Résultat de benchmark (lit ``image_path`` de chaque DocumentResult).
    output_dir:
        Dossier où le HTML sera écrit ; le sous-dossier d'assets sera
        créé à côté.
    max_width:
        Largeur max du redimensionnement (cohérent avec
        ``encode_image_b64``).
    asset_subdir:
        Nom du sous-dossier d'assets (défaut ``"report-assets"``).

    Returns
    -------
    dict[str, str]
        ``{doc_id: "report-assets/<doc_id>.png"}`` (URL relative
        consommable directement dans un attribut HTML ``src``).
    """
    from PIL import Image

    assets_dir = output_dir / asset_subdir
    assets_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}

    seen_ids: set[str] = set()
    for engine_report in benchmark.engine_reports:
        for dr in engine_report.document_results:
            doc_id = dr.doc_id
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            try:
                src = Path(dr.image_path)
                if not src.exists():
                    continue
                # Nom de fichier dérivé du doc_id, normalisé sans
                # caractères dangereux pour le filesystem.
                safe_id = "".join(
                    c if c.isalnum() or c in "._-" else "_" for c in doc_id
                )
                dest = assets_dir / f"{safe_id}{src.suffix.lower() or '.png'}"
                with Image.open(src) as img:
                    if img.width > max_width:
                        ratio = max_width / img.width
                        new_h = max(1, int(img.height * ratio))
                        img = img.resize((max_width, new_h), Image.LANCZOS)
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    fmt = "JPEG" if dest.suffix in (".jpg", ".jpeg") else "PNG"
                    img.save(dest, format=fmt, optimize=True, quality=85)
                # URL relative (POSIX style même sur Windows pour HTML).
                out[doc_id] = f"{asset_subdir}/{dest.name}"
            except Exception as exc:  # noqa: BLE001 — fallback silencieux + warning
                logger.warning(
                    "[report] échec d'externalisation de l'image %s : %s — "
                    "le rapport ignorera cette image",
                    dr.image_path,
                    exc,
                )
    return out


__all__ = [
    "load_vendor_js",
    "encode_image_b64",
    "encode_images_b64_from_result",
    "externalize_images_to_dir",
]
