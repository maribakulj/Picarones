"""Utilitaires de manipulation de corpus côté web.

Détection ALTO/PAGE, extraction de texte GT, analyse de la structure
d'un dossier corpus, extraction de ZIP avec garde-fous (taille
décompressée, nombre de fichiers, validation image extraite,
détection de collision de basename). Le parsing XML sécurisé délègue
à :func:`picarones.formats._xml_utils.safe_parse_xml`.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from picarones.formats._xml_utils import safe_parse_xml
from picarones.interfaces.web.state import IMAGE_EXTS

logger = logging.getLogger(__name__)

# Garde-fous ZIP-bomb pour l'upload
MAX_ZIP_TOTAL_SIZE = 500 * 1024 * 1024
"""500 Mo décompressé maximum."""

MAX_ZIP_FILES = 2000
"""Nombre maximum de fichiers extraits."""

MAX_ZIP_MEMBER_SIZE = 100 * 1024 * 1024
"""Taille décompressée max d'UN membre (aligné sur l'upload image)."""

MAX_ZIP_COMPRESSION_RATIO = 200
"""Ratio décompressé/compressé max par membre.  Une image ou un
texte patrimonial légitime reste sous ~200:1 ; au-delà = zip bomb
(les bombes classiques sont à 1000:1+).  Garde-fou complémentaire
du plafond absolu : attrape une bombe AVANT de la décompresser."""

_ZIP_CHUNK_SIZE = 1024 * 1024
"""Bloc de lecture pour l'extraction ZIP en flux (jamais le membre
entier en RAM)."""


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

def _slug_dirname(source_path: Path) -> str:
    """Slugifie le ``dirname`` d'une entrée ZIP pour préfixer en cas de collision.

    ``a/b/img.png`` → ``a_b``.  Caractères non sûrs (``..``, séparateurs)
    sont normalisés en ``_``.  Vide si l'entrée est à la racine du ZIP.
    """
    parent = source_path.parent
    if parent == Path() or str(parent) == ".":
        return ""
    parts = [
        part.replace("..", "_").replace("/", "_").replace("\\", "_")
        for part in parent.parts
        if part not in ("", "/", "\\")
    ]
    return "_".join(p for p in parts if p)


def _resolve_collision(
    name: str, source_path: Path, taken: set[str],
) -> str:
    """Renomme ``name`` pour éviter une collision avec ``taken``.

    Stratégie :
    1. Préfixe avec le slug du dirname source (traçabilité).  Si pas de
       dirname ou si déjà pris, ajoute un suffixe numérique.
    2. Lève ``ValueError`` après 1000 tentatives (corpus pathologique).
    """
    slug = _slug_dirname(source_path)
    if slug:
        candidate = f"{slug}__{name}"
        if candidate not in taken:
            return candidate
    stem = Path(name).stem
    suffix = "".join(Path(name).suffixes)
    for n in range(2, 1001):
        candidate = f"{stem}_{n}{suffix}"
        if candidate not in taken:
            return candidate
    raise ValueError(
        f"Impossible de résoudre la collision de basename pour {name!r} "
        f"après 1000 tentatives — corpus pathologique ?",
    )


def flatten_zip_to_dir(
    zf: zipfile.ZipFile,
    dest: Path,
    *,
    validate_images: bool = True,
) -> None:
    """Extrait un ZIP en aplatissant les paires image/.gt.txt/.xml dans ``dest``.

    Garde-fous :

    - Ignore les fichiers cachés macOS (préfixe ``.`` ou ``__MACOSX``).
    - Refuse si la taille décompressée totale dépasse ``MAX_ZIP_TOTAL_SIZE``.
    - Refuse si le nombre de fichiers extraits dépasse ``MAX_ZIP_FILES``.
    - **Détection de collision de basename** : ``a/img.png`` et
      ``b/img.png`` ne s'écrasent plus silencieusement — le second est
      renommé avec un préfixe dérivé de son dossier source (ex.
      ``b__img.png``) et un warning est loggué.  Sans ce garde-fou,
      l'utilisateur pouvait associer silencieusement une image à une
      GT incorrecte.
    - **Validation image** : chaque image extraite passe par
      :func:`validate_image_safe` (Pillow.verify, anti-bombe), de la
      même manière que les uploads directs.  Désactivable via
      ``validate_images=False`` (utile aux tests qui ne fournissent
      pas de PNG complets).
    """
    # Import retardé : ``security`` dépend de ``state`` qui dépend de
    # ``corpus_utils`` → circulaire si import toplevel.
    from picarones.interfaces.web.security import validate_image_file_safe

    dest.mkdir(parents=True, exist_ok=True)
    total_size = 0
    file_count = 0
    written_names: set[str] = set()
    for member in zf.infolist():
        if member.is_dir():
            continue
        p = Path(member.filename)
        name = p.name
        if name.startswith("."):
            continue
        suffix_lower = p.suffix.lower()
        is_image = suffix_lower in IMAGE_EXTS
        if not (
            is_image
            or name.endswith(".gt.txt")
            or name.endswith(".ocr.txt")
            or suffix_lower == ".xml"
        ):
            continue

        file_count += 1
        if file_count > MAX_ZIP_FILES:
            raise ValueError(f"ZIP contient trop de fichiers (> {MAX_ZIP_FILES})")

        # Garde-fous AVANT décompression (header ZIP).  Audit P0.5 :
        # rejet précoce d'un membre trop gros ou au ratio de bombe,
        # sans avoir à le décompresser.
        declared = member.file_size
        if declared > MAX_ZIP_MEMBER_SIZE:
            raise ValueError(
                f"Membre ZIP '{name}' trop volumineux : "
                f"{declared // (1024*1024)} Mo décompressé > "
                f"{MAX_ZIP_MEMBER_SIZE // (1024*1024)} Mo"
            )
        comp = member.compress_size
        if comp > 0 and declared / comp > MAX_ZIP_COMPRESSION_RATIO:
            raise ValueError(
                f"Membre ZIP '{name}' : ratio de compression "
                f"{declared // max(comp, 1)}:1 > "
                f"{MAX_ZIP_COMPRESSION_RATIO}:1 — bombe de "
                "décompression suspectée."
            )

        # Détection de collision AVANT écriture : ``a/img.png`` et
        # ``b/img.png`` ne doivent pas s'écraser silencieusement
        # (vecteur de mauvaise association image/GT après aplatissement).
        if name in written_names:
            new_name = _resolve_collision(name, p, written_names)
            logger.warning(
                "[flatten_zip] collision de basename %r — renommé en %r "
                "(source ZIP : %r)",
                name, new_name, member.filename,
            )
            name = new_name
        written_names.add(name)

        # Extraction EN FLUX (audit P0.5) : jamais le membre entier en
        # RAM.  On compte les octets RÉELLEMENT décompressés (un header
        # ZIP menteur ne contourne donc pas les plafonds).
        dest_path = dest / name
        written = 0
        try:
            with zf.open(member) as src, dest_path.open("wb") as dst:
                while True:
                    chunk = src.read(_ZIP_CHUNK_SIZE)
                    if not chunk:
                        break
                    written += len(chunk)
                    total_size += len(chunk)
                    if written > MAX_ZIP_MEMBER_SIZE:
                        raise ValueError(
                            f"Membre ZIP '{name}' dépasse "
                            f"{MAX_ZIP_MEMBER_SIZE // (1024*1024)} Mo "
                            "à la décompression (header menteur ?)."
                        )
                    if total_size > MAX_ZIP_TOTAL_SIZE:
                        raise ValueError(
                            f"ZIP trop volumineux : taille décompressée > "
                            f"{MAX_ZIP_TOTAL_SIZE // (1024*1024)} Mo"
                        )
                    dst.write(chunk)
            # Validation image depuis le FICHIER (pas de read_bytes
            # intégral) — les images extraites d'un ZIP doivent passer
            # la même vérification que les uploads directs.
            if is_image and validate_images:
                validate_image_file_safe(dest_path, filename=name)
        except Exception:
            # Nettoyage du fichier partiel/invalide ; le caller purge
            # de toute façon corpus_dir, mais on ne laisse pas de
            # résidu si flatten est utilisé hors api_corpus_upload.
            dest_path.unlink(missing_ok=True)
            raise


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
