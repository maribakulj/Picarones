"""Données de test réalistes pour valider le rapport HTML sans moteurs OCR installés.

Usage :
    from picarones.fixtures import generate_sample_benchmark
    bm = generate_sample_benchmark()
    bm.to_json("sample_results.json")
"""

from __future__ import annotations

import base64
import random
import struct
import zlib
from pathlib import Path
from typing import Optional

from picarones.core.metrics import MetricsResult, aggregate_metrics
from picarones.core.results import BenchmarkResult, DocumentResult, EngineReport

# ---------------------------------------------------------------------------
# Textes GT réalistes (documents patrimoniaux BnF)
# ---------------------------------------------------------------------------

_GT_TEXTS = [
    "Icy commence le prologue de maistre Jehan Froissart sus les croniques de France & d'Angleterre.",
    "En l'an de grace mil trois cens soixante, regnoit en France le noble roy Jehan, filz du roy Phelippe de Valois.",
    "Item ledit jour furent menez en ladicte ville de Paris plusieurs prisonniers sarasins & mahommetans.",
    "Le chancellier du roy manda à tous les baillifs & seneschaulx que on feist crier & publier par tous les carrefours.",
    "Cy après sensuyt la copie des lettres patentes données par nostre seigneur le roy à ses très chiers & feaulx.",
    "Nous Charles, par la grace de Dieu roy de France, à tous ceulx qui ces presentes lettres verront, salut.",
    "Savoir faisons que pour considéracion des bons & aggreables services que nostre amé & feal conseillier.",
    "Donné à Paris, le vingt & deuxième jour du mois de juillet, l'an de grace mil quatre cens & troys.",
    "Les dessus ditz ambassadeurs respondirent que leur seigneur & maistre estoit très joyeulx de ceste aliance.",
    "Après lesquelles choses ainsi faictes & passées, le dit traictié fut ratiffié & confirmé de toutes parties.",
    "Item, en ladicte année, fut faicte grant assemblée de gens d'armes tant à cheval que à pied.",
    "Et pour ce que la chose est notoire & manifeste, nous avons fait mettre nostre scel à ces presentes.",
]

# ---------------------------------------------------------------------------
# Erreurs OCR typiques par moteur (transformations appliquées au GT)
# ---------------------------------------------------------------------------

def _tesseract_errors(text: str, rng: random.Random) -> str:
    """Simule les erreurs typiques de Tesseract sur documents médiévaux."""
    replacements = [
        ("ſ", "f"), ("œ", "oe"), ("æ", "ae"),
        ("&", "8"), ("é", "e"), ("è", "e"),
        ("nostre", "noltre"), ("maistre", "inaistre"),
        ("faictes", "faictcs"), ("ledit", "Ledit"),
        ("regnoit", "regnoit"), ("Froissart", "Froiflart"),
        ("conseillie", "conlcillier"), ("consideracion", "confideration"),
        ("ny", "uy"), ("lx", "le"),
    ]
    for src, tgt in rng.sample(replacements, k=min(rng.randint(2, 5), len(replacements))):
        text = text.replace(src, tgt, 1)
    if rng.random() < 0.3:
        words = text.split()
        if len(words) > 5:
            idx = rng.randint(1, len(words) - 2)
            words.pop(idx)
            text = " ".join(words)
    return text


def _pero_errors(text: str, rng: random.Random) -> str:
    """Pero OCR : moins d'erreurs, mais confusions diacritiques persistantes."""
    replacements = [
        ("é", "é"), ("è", "e"), ("ê", "e"),
        ("œ", "oe"), ("&", "&"),
        ("uy", "ny"), ("rr", "ri"),
        ("nostre", "noſtre"), ("maistre", "maistre"),
    ]
    for src, tgt in rng.sample(replacements, k=rng.randint(0, 3)):
        text = text.replace(src, tgt, 1)
    return text


def _bad_engine_errors(text: str, rng: random.Random) -> str:
    """Moteur de mauvaise qualité : nombreuses erreurs."""
    words = text.split()
    result = []
    for word in words:
        r = rng.random()
        if r < 0.15:
            pass  # mot supprimé
        elif r < 0.30:
            # substitution partielle
            chars = list(word)
            if len(chars) > 2:
                i = rng.randint(0, len(chars) - 1)
                chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
            result.append("".join(chars))
        else:
            result.append(word)
    if rng.random() < 0.2:
        result.insert(rng.randint(0, len(result)), rng.choice(["|||", "---", "###"]))
    return " ".join(result)


# ---------------------------------------------------------------------------
# Génération d'une image PNG placeholder (pur Python, sans Pillow)
# ---------------------------------------------------------------------------

def _make_placeholder_png(width: int = 300, height: int = 200, text_hint: str = "") -> bytes:
    """Génère un PNG minimal représentant une page de document (gris clair).

    Le PNG est valide et affichable dans tous les navigateurs.
    On dessine une zone blanche avec une bordure et quelques lignes simulant du texte.
    """
    # Créer les données de pixels RGB
    pixels = []
    for y in range(height):
        row = []
        for x in range(width):
            # Fond légèrement crème (#f5f0e8)
            if x < 3 or x >= width - 3 or y < 3 or y >= height - 3:
                row.extend([180, 160, 140])  # bordure grise
            elif 20 < y < 24 or 35 < y < 39:
                # Lignes de titre simulées
                if 30 < x < width - 30:
                    row.extend([80, 80, 80])  # texte gris foncé
                else:
                    row.extend([245, 240, 232])
            elif y > 50 and (y - 50) % 18 < 2 and 20 < x < width - 20:
                row.extend([120, 120, 120])  # lignes de texte simulées
            else:
                row.extend([245, 240, 232])
        pixels.append(bytes(row))

    def make_png(w: int, h: int, rows: list[bytes]) -> bytes:
        def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
            c = chunk_type + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        raw = b"".join(b"\x00" + row for row in rows)
        idat = png_chunk(b"IDAT", zlib.compress(raw))
        iend = png_chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    return make_png(width, height, pixels)


def _png_to_data_uri(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Génération du benchmark de test
# ---------------------------------------------------------------------------

def _make_metrics(reference: str, hypothesis: str) -> MetricsResult:
    from picarones.core.metrics import compute_metrics
    return compute_metrics(reference, hypothesis)


def generate_sample_benchmark(
    n_docs: int = 12,
    seed: int = 42,
    include_images: bool = True,
) -> BenchmarkResult:
    """Génère un BenchmarkResult fictif mais réaliste.

    Parameters
    ----------
    n_docs:
        Nombre de documents dans le corpus de test (max = len(_GT_TEXTS)).
    seed:
        Graine aléatoire pour la reproductibilité.
    include_images:
        Si True, génère des images PNG placeholder encodées en base64.

    Returns
    -------
    BenchmarkResult
        Prêt pour le rapport HTML ou l'export JSON.
    """
    rng = random.Random(seed)
    n_docs = min(n_docs, len(_GT_TEXTS))
    gt_texts = _GT_TEXTS[:n_docs]

    engines_config = [
        ("pero_ocr", "0.7.2", {"config": "/models/pero_printed.ini"}, _pero_errors),
        ("tesseract", "5.3.3", {"lang": "fra", "psm": 6}, _tesseract_errors),
        ("ancien_moteur", "2.1.0", {"lang": "fra"}, _bad_engine_errors),
    ]

    engine_reports: list[EngineReport] = []
    image_b64_cache: dict[str, str] = {}

    for engine_name, engine_version, engine_cfg, error_fn in engines_config:
        doc_results: list[DocumentResult] = []

        for i, gt in enumerate(gt_texts):
            doc_id = f"folio_{i+1:03d}"
            image_path = f"/corpus/images/{doc_id}.jpg"

            # Générer l'image placeholder une fois
            if include_images and doc_id not in image_b64_cache:
                png = _make_placeholder_png(320, 220, gt[:20])
                image_b64_cache[doc_id] = _png_to_data_uri(png)

            # Générer la sortie OCR avec erreurs
            hypothesis = error_fn(gt, rng)

            metrics = _make_metrics(gt, hypothesis)

            doc_results.append(
                DocumentResult(
                    doc_id=doc_id,
                    image_path=image_path,
                    ground_truth=gt,
                    hypothesis=hypothesis,
                    metrics=metrics,
                    duration_seconds=round(rng.uniform(0.3, 4.5), 3),
                )
            )

        report = EngineReport(
            engine_name=engine_name,
            engine_version=engine_version,
            engine_config=engine_cfg,
            document_results=doc_results,
        )
        engine_reports.append(report)

    bm = BenchmarkResult(
        corpus_name="Corpus de test — Chroniques médiévales BnF",
        corpus_source="/corpus/chroniques/",
        document_count=n_docs,
        engine_reports=engine_reports,
        metadata={
            "description": "Données de démonstration générées par picarones.fixtures",
            "script": "gothique textura",
            "langue": "Français médiéval (XIVe-XVe siècle)",
            "institution": "BnF — Département des manuscrits",
        },
    )

    # Attacher les images base64 au benchmark (hors du schéma standard,
    # le générateur HTML les récupérera depuis ce champ supplémentaire)
    bm.metadata["_images_b64"] = image_b64_cache  # type: ignore[assignment]

    return bm
