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
from picarones.pipelines.over_normalization import detect_over_normalization
# Sprint 5 — métriques avancées
from picarones.core.confusion import build_confusion_matrix
from picarones.core.char_scores import compute_ligature_score, compute_diacritic_score
from picarones.core.taxonomy import classify_errors, aggregate_taxonomy
from picarones.core.structure import analyze_structure, aggregate_structure
from picarones.core.image_quality import generate_mock_quality_scores, aggregate_image_quality
from picarones.core.char_scores import aggregate_ligature_scores, aggregate_diacritic_scores

# ---------------------------------------------------------------------------
# Textes GT réalistes (documents patrimoniaux BnF)
# ---------------------------------------------------------------------------

_GT_TEXTS = [
    # Textes avec graphies médiévales incluant ſ, &, u/v — pour démontrer le CER diplomatique
    "Icy commence le prologue de maiſtre Jehan Froiſſart ſus les croniques de France & d'Angleterre.",
    "En l'an de grace mil trois cens ſoixante, regnoit en France le noble roy Jehan, filz du roy Phelippe de Valois.",
    "Item ledit iour furent menez en ladicte ville de Paris pluſieurs priſonniers ſaraſins & mahommetans.",
    "Le chancellier du roy manda à tous les baillifs & ſeneſchaulx que on feiſt crier & publier par tous les carrefours.",
    "Cy après ſenſuyt la copie des lettres patentes données par noſtre ſeigneur le roy à ſes très chiers & feaulx.",
    "Nous Charles, par la grace de Dieu roy de France, à tous ceulx qui ces preſentes lettres verront, ſalut.",
    "Sauoir faiſons que pour conſidéracion des bons & aggreables ſeruices que noſtre amé & feal conſeillier.",
    "Donné à Paris, le vingt & deuxième iour du mois de iuillet, l'an de grace mil quatre cens & troys.",
    "Les deſſus ditz ambaſſadeurs reſpondirent que leur ſeigneur & maiſtre eſtoit très ioyeulx de ceſte aliance.",
    "Après lesquelles choſes ainſi faictes & paſſées, le dit traictié fut ratiffié & confirmé de toutes parties.",
    "Item, en ladicte année, fut faicte grant aſſemblée de gens d'armes tant à cheual que à pied.",
    "Et pour ce que la choſe eſt notoire & manifeſte, nous auons fait mettre noſtre ſcel à ces preſentes.",
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


def _llm_correction(text: str, rng: random.Random) -> str:
    """Simule la correction GPT-4o sur la sortie Tesseract.

    Le LLM corrige la majorité des erreurs OCR mais introduit parfois
    de la sur-normalisation (classe 10) : il modernise des graphies médiévales
    légitimes (nostre → notre, maistre → maître, faict → fait).
    """
    # Corrections typiques que le LLM réussit (erreurs OCR fréquentes)
    good_corrections = [
        ("noltre", "nostre"), ("inaistre", "maistre"),
        ("faictcs", "faictes"), ("conlcillier", "conseillie"),
        ("confideration", "consideracion"), ("Froiflart", "Froissart"),
        ("8", "&"), ("oe", "œ"),
    ]
    for src, tgt in good_corrections:
        text = text.replace(src, tgt)

    # Sur-normalisation : le LLM modernise parfois à tort (classe 10)
    # Ces remplacements s'appliquent sur le texte (partiellement corrigé ci-dessus)
    over_normalizations = [
        ("nostre", "notre"), ("maistre", "maître"),
        ("faictes", "faites"), ("Donné", "donné"),
        ("conseillier", "conseiller"), ("consideracion", "considération"),
    ]
    # ~45% de chance de sur-normaliser sur chaque document
    if rng.random() < 0.45:
        for src, tgt in rng.sample(over_normalizations, k=rng.randint(1, 2)):
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

    # (name, version, config, error_fn, is_pipeline, pipeline_info)
    engines_config = [
        ("pero_ocr", "0.7.2", {"config": "/models/pero_printed.ini"}, _pero_errors, False, {}),
        ("tesseract", "5.3.3", {"lang": "fra", "psm": 6}, _tesseract_errors, False, {}),
        ("ancien_moteur", "2.1.0", {"lang": "fra"}, _bad_engine_errors, False, {}),
        # Pipeline fictif : tesseract → gpt-4o (post-correction image+texte)
        (
            "tesseract → gpt-4o",
            "ocr=5.3.3; llm=gpt-4o",
            {"lang": "fra", "psm": 6},
            _llm_correction,  # appliqué sur la sortie tesseract
            True,
            {
                "pipeline_mode": "text_and_image",
                "prompt_file": "correction_medieval_french.txt",
                "llm_model": "gpt-4o",
                "llm_provider": "openai",
                "pipeline_steps": [
                    {"type": "ocr", "engine": "tesseract", "version": "5.3.3"},
                    {
                        "type": "llm",
                        "model": "gpt-4o",
                        "provider": "openai",
                        "mode": "text_and_image",
                        "prompt_file": "correction_medieval_french.txt",
                    },
                ],
            },
        ),
    ]

    engine_reports: list[EngineReport] = []
    image_b64_cache: dict[str, str] = {}

    # Pré-calculer les sorties tesseract pour le pipeline
    tess_outputs: dict[str, str] = {}

    for engine_name, engine_version, engine_cfg, error_fn, is_pipeline, pipeline_info in engines_config:
        doc_results: list[DocumentResult] = []

        for i, gt in enumerate(gt_texts):
            doc_id = f"folio_{i+1:03d}"
            image_path = f"/corpus/images/{doc_id}.jpg"

            # Générer l'image placeholder une fois
            if include_images and doc_id not in image_b64_cache:
                png = _make_placeholder_png(320, 220, gt[:20])
                image_b64_cache[doc_id] = _png_to_data_uri(png)

            if is_pipeline:
                # Pour le pipeline : appliquer tesseract d'abord, puis LLM correction
                ocr_intermediate = tess_outputs.get(doc_id) or _tesseract_errors(gt, random.Random(rng.randint(0, 9999)))
                hypothesis = _llm_correction(ocr_intermediate, rng)
                # Calcul de la sur-normalisation (classe 10)
                over_norm = detect_over_normalization(gt, ocr_intermediate, hypothesis)
                pipeline_meta = {
                    "pipeline_mode": pipeline_info.get("pipeline_mode"),
                    "prompt_file": pipeline_info.get("prompt_file"),
                    "llm_model": pipeline_info.get("llm_model"),
                    "llm_provider": pipeline_info.get("llm_provider"),
                    "over_normalization": over_norm.as_dict(),
                }
                duration = round(rng.uniform(2.5, 12.0), 3)  # plus lent qu'un OCR seul
            else:
                ocr_intermediate = None
                hypothesis = error_fn(gt, rng)
                pipeline_meta = {}
                duration = round(rng.uniform(0.3, 4.5), 3)
                # Mémoriser la sortie tesseract pour le pipeline
                if engine_name == "tesseract":
                    tess_outputs[doc_id] = hypothesis

            metrics = _make_metrics(gt, hypothesis)

            # Sprint 5 — métriques avancées patrimoniales
            cm = build_confusion_matrix(gt, hypothesis)
            lig_score = compute_ligature_score(gt, hypothesis)
            diac_score = compute_diacritic_score(gt, hypothesis)
            taxonomy_result = classify_errors(gt, hypothesis)
            struct_result = analyze_structure(gt, hypothesis)
            iq_result = generate_mock_quality_scores(doc_id, seed=rng.randint(0, 999999))

            doc_results.append(
                DocumentResult(
                    doc_id=doc_id,
                    image_path=image_path,
                    ground_truth=gt,
                    hypothesis=hypothesis,
                    metrics=metrics,
                    duration_seconds=duration,
                    ocr_intermediate=ocr_intermediate,
                    pipeline_metadata=pipeline_meta,
                    confusion_matrix=cm.as_dict(),
                    char_scores={
                        "ligature": lig_score.as_dict(),
                        "diacritic": diac_score.as_dict(),
                    },
                    taxonomy=taxonomy_result.as_dict(),
                    structure=struct_result.as_dict(),
                    image_quality=iq_result.as_dict(),
                )
            )

        # Agréger les stats de sur-normalisation pour le pipeline
        effective_pipeline_info = dict(pipeline_info)
        if is_pipeline:
            over_norms = [
                dr.pipeline_metadata.get("over_normalization")
                for dr in doc_results
                if dr.pipeline_metadata.get("over_normalization")
            ]
            if over_norms:
                total_correct = sum(r["total_correct_ocr_words"] for r in over_norms)
                total_over = sum(r["over_normalized_count"] for r in over_norms)
                effective_pipeline_info["over_normalization"] = {
                    "score": round(total_over / total_correct, 4) if total_correct > 0 else 0.0,
                    "total_correct_ocr_words": total_correct,
                    "over_normalized_count": total_over,
                    "document_count": len(over_norms),
                }

        # Agrégation Sprint 5
        from picarones.core.confusion import aggregate_confusion_matrices, ConfusionMatrix
        from picarones.core.char_scores import LigatureScore, DiacriticScore
        from picarones.core.taxonomy import TaxonomyResult
        from picarones.core.structure import StructureResult
        from picarones.core.image_quality import ImageQualityResult

        agg_confusion = aggregate_confusion_matrices([
            ConfusionMatrix(**dr.confusion_matrix)
            for dr in doc_results if dr.confusion_matrix
        ]).as_compact_dict(min_count=1)

        agg_lig = aggregate_ligature_scores([
            LigatureScore(**dr.char_scores["ligature"])
            for dr in doc_results if dr.char_scores
        ])
        agg_diac = aggregate_diacritic_scores([
            DiacriticScore(**dr.char_scores["diacritic"])
            for dr in doc_results if dr.char_scores
        ])
        agg_char_scores = {"ligature": agg_lig, "diacritic": agg_diac}

        agg_taxonomy = aggregate_taxonomy([
            TaxonomyResult.from_dict(dr.taxonomy)
            for dr in doc_results if dr.taxonomy
        ])

        agg_structure = aggregate_structure([
            StructureResult.from_dict(dr.structure)
            for dr in doc_results if dr.structure
        ])

        agg_iq = aggregate_image_quality([
            ImageQualityResult.from_dict(dr.image_quality)
            for dr in doc_results if dr.image_quality
        ])

        report = EngineReport(
            engine_name=engine_name,
            engine_version=engine_version,
            engine_config=engine_cfg,
            document_results=doc_results,
            pipeline_info=effective_pipeline_info,
            aggregated_confusion=agg_confusion,
            aggregated_char_scores=agg_char_scores,
            aggregated_taxonomy=agg_taxonomy,
            aggregated_structure=agg_structure,
            aggregated_image_quality=agg_iq,
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
