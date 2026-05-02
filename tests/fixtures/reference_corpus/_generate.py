"""Génère les images PNG du corpus de référence (Sprint A5, M-14).

Idempotent : produit les mêmes octets à chaque exécution grâce à la
police par défaut Pillow (police bitmap interne, ne dépend pas du
système). Les fichiers sont écrits à côté de ce script.

Exécution :

    python tests/fixtures/reference_corpus/_generate.py

Le workflow CI ``perf_regression.yml`` régénère les fichiers en début
de run pour s'assurer qu'ils sont à jour vis-à-vis du code de
génération.
"""

from __future__ import annotations

from pathlib import Path

# Chaque entrée = (id, ligne_1, ligne_2_optionnelle, ...).
# Les textes sont en français pour exercer Tesseract `fra`.
_DOCUMENTS: list[tuple[str, list[str]]] = [
    (
        "doc_01_imprime_moderne",
        [
            "Picarones est une plateforme de banc d'essai",
            "pour des moteurs OCR sur documents",
            "patrimoniaux. Cette image est synthetique.",
        ],
    ),
    (
        "doc_02_chiffres_dates",
        [
            "Charte du 14 mars 1789, signee par",
            "le notaire Jean Dupont. Folio 23 verso.",
            "Tarif: 5 livres 12 sols 6 deniers.",
        ],
    ),
    (
        "doc_03_noms_propres",
        [
            "Liste des temoins :",
            "Marie Lefevre, Pierre Bernard,",
            "Antoine Rousseau, Catherine Moreau.",
        ],
    ),
    (
        "doc_04_courte_phrase",
        [
            "L'ancien Regime se termine en 1789.",
        ],
    ),
    (
        "doc_05_paragraphe_long",
        [
            "Au commencement de l'an mille sept cent",
            "quatre vingt neuf, le royaume de France",
            "comptait environ vingt huit millions",
            "d'habitants. Paris seule en hebergeait",
            "six cent cinquante mille.",
        ],
    ),
]


def _render_one(out_dir: Path, doc_id: str, lines: list[str]) -> None:
    """Rend une image PNG + son fichier .gt.txt à côté.

    Police : police bitmap interne de Pillow (``ImageFont.load_default``)
    pour que l'image soit identique sur tous les systèmes (pas de
    dépendance à des polices installées).
    """
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    # On rend large pour que Tesseract ait de quoi mâcher.
    line_height = 30
    margin = 20
    width = 800
    height = margin * 2 + line_height * len(lines)

    img = Image.new("RGB", (width, height), color=(255, 255, 245))
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        # Échelle x4 par redimensionnement : on rend petit puis on
        # upscale pour obtenir un texte ~24 px de haut, lisible par
        # Tesseract sans nécessiter une vraie police TrueType.
        small = Image.new("RGB", (width // 4, line_height // 4 * len(lines)), color=(255, 255, 245))
        small_draw = ImageDraw.Draw(small)
        small_draw.text((5, 5 + i * line_height // 4), line, fill=(20, 20, 20), font=font)
        # Composite en upscale dans le canvas final.
        # (On garde la version brute pour rester déterministe.)
        del small_draw, small
        draw.text((margin, margin + i * line_height), line, fill=(20, 20, 20), font=font)

    png_path = out_dir / f"{doc_id}.png"
    img.save(png_path, format="PNG", optimize=True)

    gt_path = out_dir / f"{doc_id}.gt.txt"
    gt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(out_dir: Path | None = None) -> Path:
    """Régénère le corpus dans ``out_dir`` (défaut : à côté de ce script).

    Retourne le chemin du dossier."""
    if out_dir is None:
        out_dir = Path(__file__).parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for doc_id, lines in _DOCUMENTS:
        _render_one(out_dir, doc_id, lines)
    return out_dir


if __name__ == "__main__":
    p = generate()
    print(f"Corpus de référence (re)généré dans {p}")
    print(f"  {len(_DOCUMENTS)} documents, "
          f"~{sum(len(lines) for _, lines in _DOCUMENTS)} lignes au total.")
