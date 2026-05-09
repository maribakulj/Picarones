"""Utilitaires de manipulation de corpus côté web.

Détection ALTO/PAGE, extraction de texte GT, analyse de la structure
d'un dossier corpus, extraction de ZIP avec garde-fous (taille
décompressée, nombre de fichiers). Le parsing XML sécurisé délègue
à :func:`picarones.formats._xml_utils.safe_parse_xml`.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from picarones.formats._xml_utils import safe_parse_xml
from picarones.interfaces.web.state import IMAGE_EXTS

# Garde-fous ZIP-bomb pour l'upload
MAX_ZIP_TOTAL_SIZE = 500 * 1024 * 1024
"""500 Mo décompressé maximum."""

MAX_ZIP_FILES = 2000
"""Nombre maximum de fichiers extraits."""


# ──────────────────────────────────────────────────────────────────────────
# Détection ALTO / PAGE depuis bytes XML
# ──────────────────────────────────────────────────────────────────────────

def detect_xml_gt(xml_bytes: bytes) -> tuple[str, str] | None:
    """Détecte si ``xml_bytes`` est un fichier ALTO ou PAGE XML.

    Retourne ``(format_label, texte_gt)`` ou ``None`` si le format
    n'est pas reconnu.
    """
    root = safe_parse_xml(xml_bytes)
    if root is None:
        return None

    tag = root.tag

    # ALTO XML : namespace contient loc.gov/standards/alto ou balise racine "alto"
    ns_alto = "http://www.loc.gov/standards/alto"
    is_alto = (
        ns_alto in tag
        or tag.lower() == "alto"
        or (tag.startswith("{") and tag.split("}")[1].lower() in ("alto",))
    )
    if is_alto:
        return ("ALTO XML", extract_alto_text(root))

    # PAGE XML : balise racine PcGts (avec ou sans namespace)
    local = tag.split("}")[-1] if "}" in tag else tag
    if local == "PcGts":
        return ("PAGE XML", extract_page_text(root))

    return None


def extract_alto_text(root: ET.Element) -> str:
    """Extrait le texte plein d'un arbre ALTO XML.

    Concatène les attributs ``CONTENT`` des balises ``<String>`` dans
    l'ordre de lecture (bloc → ligne → mot), avec un espace entre mots
    et un saut de ligne entre lignes.
    """
    lines: list[str] = []
    for elem in root.iter():
        local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if local == "TextLine":
            words: list[str] = []
            for child in elem.iter():
                child_local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if child_local == "String":
                    content = child.get("CONTENT", "")
                    if content:
                        words.append(content)
            if words:
                lines.append(" ".join(words))
    return "\n".join(lines)


def extract_page_text(root: ET.Element) -> str:
    """Extrait le texte plein d'un arbre PAGE XML.

    Concatène le contenu des balises ``<Unicode>`` dans l'ordre de
    lecture.
    """
    texts: list[str] = []
    for elem in root.iter():
        local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if local == "Unicode" and elem.text:
            texts.append(elem.text.strip())
    return "\n".join(t for t in texts if t)


# ──────────────────────────────────────────────────────────────────────────
# Analyse d'un dossier corpus
# ──────────────────────────────────────────────────────────────────────────

def analyze_corpus_dir(path: Path) -> dict:
    """Analyse un dossier et retourne un résumé des paires image/GT détectées.

    - Détecte les paires ``image.{jpg,png,...}`` + ``image.gt.txt``.
    - Détecte les paires ``image.{jpg,...}`` + ``image.xml`` (ALTO ou
      PAGE) et matérialise le ``image.gt.txt`` correspondant pour le
      chargeur de corpus.
    - Identifie le format dominant et le nombre de fichiers
      ``image.ocr.txt`` (corpus triplets pré-OCRisés).
    """
    # Exclure les fichiers cachés macOS (._* AppleDouble) et tout fichier
    # débutant par un point.
    images = sorted(
        f.name for f in path.iterdir()
        if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith(".")
    )
    pairs: list[dict] = []
    missing_gt: list[str] = []
    for img in images:
        stem = Path(img).stem
        gt_txt = path / (stem + ".gt.txt")
        gt_xml = path / (stem + ".xml")
        if gt_txt.exists():
            pairs.append({"image": img, "gt": stem + ".gt.txt", "gt_format": "texte brut"})
        elif gt_xml.exists():
            result = detect_xml_gt(gt_xml.read_bytes())
            if result is not None:
                fmt, text = result
                gt_txt.write_text(text, encoding="utf-8")
                pairs.append({"image": img, "gt": stem + ".gt.txt", "gt_format": fmt})
            else:
                missing_gt.append(img)
        else:
            missing_gt.append(img)

    # Format dominant
    formats = {p["gt_format"] for p in pairs}
    if len(formats) == 1:
        dominant_format: str = formats.pop()
    elif formats:
        dominant_format = "mixte"
    else:
        dominant_format = "texte brut"

    ocr_text_count = sum(
        1 for p in pairs
        if (path / (Path(p["image"]).stem + ".ocr.txt")).exists()
    )

    return {
        "doc_count": len(pairs),
        "pairs": pairs[:20],
        "total_pairs": len(pairs),
        "missing_gt": missing_gt[:10],
        "has_missing_gt": len(missing_gt) > 0,
        "warnings": [f"GT manquant : {img}" for img in missing_gt[:5]],
        "usable": len(pairs) > 0,
        "gt_format": dominant_format,
        "has_ocr_text": ocr_text_count > 0,
        "ocr_text_count": ocr_text_count,
    }


# ──────────────────────────────────────────────────────────────────────────
# Extraction ZIP sécurisée
# ──────────────────────────────────────────────────────────────────────────

def flatten_zip_to_dir(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extrait un ZIP en aplatissant les paires image/.gt.txt/.xml dans ``dest``.

    Garde-fous :
    - Ignore les fichiers cachés macOS (préfixe ``.`` ou ``__MACOSX``).
    - Refuse si la taille décompressée totale dépasse ``MAX_ZIP_TOTAL_SIZE``.
    - Refuse si le nombre de fichiers extraits dépasse ``MAX_ZIP_FILES``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    total_size = 0
    file_count = 0
    for member in zf.infolist():
        if member.is_dir():
            continue
        p = Path(member.filename)
        name = p.name
        if name.startswith("."):
            continue
        if (
            p.suffix.lower() in IMAGE_EXTS
            or name.endswith(".gt.txt")
            or name.endswith(".ocr.txt")
            or p.suffix.lower() == ".xml"
        ):
            total_size += member.file_size
            if total_size > MAX_ZIP_TOTAL_SIZE:
                raise ValueError(
                    f"ZIP trop volumineux : taille décompressée > "
                    f"{MAX_ZIP_TOTAL_SIZE // (1024*1024)} Mo"
                )
            file_count += 1
            if file_count > MAX_ZIP_FILES:
                raise ValueError(f"ZIP contient trop de fichiers (> {MAX_ZIP_FILES})")
            data = zf.read(member.filename)
            (dest / name).write_bytes(data)


__all__ = [
    "MAX_ZIP_TOTAL_SIZE",
    "MAX_ZIP_FILES",
    "safe_parse_xml",
    "detect_xml_gt",
    "extract_alto_text",
    "extract_page_text",
    "analyze_corpus_dir",
    "flatten_zip_to_dir",
]
