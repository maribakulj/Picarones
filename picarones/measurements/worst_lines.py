"""Extraction transversale des « Worst lines » du corpus — Sprint 72.

Sprint 72 — A.I.1 chantier 1 du plan d'évolution 2026.

Pourquoi ce module
------------------
Le percentile p95 du CER ligne (calculé par ``line_metrics.py``,
Sprint 10) est un nombre abstrait : *« 5 % de mes lignes ont un
CER > 0,42 »*.  Le chercheur veut **voir** ces lignes : leur
texte, leur diff, leur document parent, pour comprendre ce qui
casse.

Ce module fournit la requête transversale qui collecte, depuis un
``BenchmarkResult``, les **N lignes les plus mal transcrites de
tout le corpus**, classées par CER ligne.  Filtrable par moteur
et par strate.

Limite documentée
-----------------
``DocumentResult.line_metrics`` ne stocke que les CER par ligne,
**pas le texte des lignes**.  Pour récupérer les textes GT/hyp
on resplitte ``ground_truth`` et ``hypothesis`` du
``DocumentResult`` à l'index de la ligne.  Cette logique
**suppose un BenchmarkResult non-compacté** — après ``compact()``
les textes sont tronqués à 200 caractères et les lignes au-delà
de cette troncature ne sont plus accessibles.  En pratique on
extrait les worst lines **avant** la sérialisation/compactage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WorstLineEntry:
    """Une ligne du corpus identifiée comme mal transcrite.

    Champs
    ------
    rank:
        Position dans le classement (1-based, 1 = pire CER).
    cer:
        CER de la ligne ∈ [0, 1].
    engine_name:
        Nom du moteur ayant produit cette hypothèse.
    doc_id:
        Identifiant du document parent.
    line_index:
        Index 0-based de la ligne dans le document GT.
    gt_line:
        Texte de la ligne dans la GT.
    hyp_line:
        Texte correspondant dans l'hypothèse (peut être ``""``
        si l'OCR a sauté la ligne).
    script_type:
        Strate du document si disponible (``script_type``
        capturé par le runner pour la stratification A.III).
    """

    rank: int
    cer: float
    engine_name: str
    doc_id: str
    line_index: int
    gt_line: str
    hyp_line: str
    script_type: Optional[str] = None


def _split_lines(text: Optional[str]) -> list[str]:
    """Splitte un texte en lignes (cohérent avec ``line_metrics``).

    Supporte les fins de ligne ``\\n``, ``\\r\\n``, ``\\r``.  Les
    lignes vides sont préservées.  Retourne une liste vide si le
    texte est None ou vide.
    """
    if not text:
        return []
    # ``splitlines`` gère \r\n et \r correctement
    return text.splitlines()


def _line_at(text: Optional[str], index: int) -> str:
    """Retourne la ligne à l'index demandé, ou ``""`` si l'index
    est hors borne (cas où l'OCR a moins de lignes que la GT)."""
    lines = _split_lines(text)
    if 0 <= index < len(lines):
        return lines[index]
    return ""


def extract_worst_lines(
    benchmark,
    *,
    top_n: int = 20,
    engine_filter: Optional[str] = None,
    script_type_filter: Optional[str] = None,
) -> list[WorstLineEntry]:
    """Extrait les ``top_n`` lignes les plus mal transcrites du
    corpus, transversalement à tous les moteurs et documents.

    Parameters
    ----------
    benchmark:
        ``BenchmarkResult`` non-compacté (cf. limite ci-dessus).
        L'objet doit exposer ``engine_reports`` (liste de
        ``EngineReport``) et optionnellement ``doc_strata``
        (map ``{doc_id: script_type}``, Sprint 45).
    top_n:
        Nombre de lignes à retourner.  Défaut : 20.
    engine_filter:
        Si fourni, n'inclut que les lignes produites par ce moteur
        (match exact sur ``engine_name``).
    script_type_filter:
        Si fourni, n'inclut que les lignes des documents de cette
        strate (nécessite ``benchmark.doc_strata``).

    Returns
    -------
    list[WorstLineEntry]
        Liste triée par CER décroissant (pire en premier),
        rang 1-based attribué après tri.  Vide si aucune ligne
        exploitable.
    """
    if top_n <= 0:
        return []

    doc_strata = getattr(benchmark, "doc_strata", None) or {}
    candidates: list[tuple[float, str, str, int, str, str, Optional[str]]] = []

    for engine_report in getattr(benchmark, "engine_reports", []):
        engine_name = engine_report.engine_name
        if engine_filter is not None and engine_name != engine_filter:
            continue
        for dr in engine_report.document_results:
            line_metrics = getattr(dr, "line_metrics", None)
            if not line_metrics:
                continue
            cer_per_line = line_metrics.get("cer_per_line") if isinstance(
                line_metrics, dict,
            ) else getattr(line_metrics, "cer_per_line", None)
            if not cer_per_line:
                continue
            doc_id = dr.doc_id
            doc_strata_value = doc_strata.get(doc_id)
            if (
                script_type_filter is not None
                and doc_strata_value != script_type_filter
            ):
                continue
            for idx, cer in enumerate(cer_per_line):
                if cer <= 0.0:
                    continue
                gt_line = _line_at(dr.ground_truth, idx)
                hyp_line = _line_at(dr.hypothesis, idx)
                if not gt_line and not hyp_line:
                    continue
                candidates.append((
                    float(cer), engine_name, doc_id, idx,
                    gt_line, hyp_line, doc_strata_value,
                ))

    if not candidates:
        return []

    # Tri par CER décroissant ; en cas d'égalité, ordre stable
    # (engine, doc_id, line_index) pour reproductibilité.
    candidates.sort(
        key=lambda c: (-c[0], c[1], c[2], c[3]),
    )
    selected = candidates[:top_n]

    return [
        WorstLineEntry(
            rank=i + 1,
            cer=cer,
            engine_name=engine,
            doc_id=doc_id,
            line_index=line_index,
            gt_line=gt_line,
            hyp_line=hyp_line,
            script_type=script_type,
        )
        for i, (
            cer, engine, doc_id, line_index,
            gt_line, hyp_line, script_type,
        ) in enumerate(selected)
    ]


__all__ = [
    "WorstLineEntry",
    "extract_worst_lines",
]
