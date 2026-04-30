"""Layout F1 par type de région — Sprint 54.

Sprint 54 — A.II.2.2 du plan d'évolution 2026.

Pourquoi ce module
------------------
Un médiéviste qui édite un manuscrit glosé veut savoir : *« le moteur
sépare-t-il bien le texte principal de la glose ? »*.  Le score de
structure global de Picarones (Sprint 5) agrège fusion/fragmentation
de lignes en un seul nombre — utile mais non typé.  Ce module
discrimine par **type de région** ALTO/PAGE (``TextRegion``,
``MarginNote``, ``Header``, ``Footer``, ``Drop-Cap``...) en
appliquant le pattern ICDAR layout standard :

- **TP** : région GT et région hypothèse de **même type** avec
  chevauchement IoU ≥ seuil (alignement greedy par IoU décroissant),
- **FN** : région GT non matchée,
- **FP** : région hypothèse non matchée,
- F1 calculé global et par type.

Le pattern d'alignement est le même que pour le NER (Sprint 38) — on
réutilise une approche éprouvée plutôt que d'en inventer une nouvelle.

Stratégie de découpage
----------------------
Cohérente avec NER (Sprint 38), Flesch (Sprint 52), Reading order F1
(Sprint 53) : couche de calcul pure d'abord.  L'utilisateur fournit
deux listes de ``Region`` (typiquement extraites de ALTO/PAGE par un
parser amont — le parser ALTO/PAGE standard de Picarones suivra
dans un sprint dédié).  Pas de câblage runner ni de vue HTML ici.

Convention de coordonnées
-------------------------
Une bbox est un tuple ``(x, y, width, height)`` en pixels (origine
en haut à gauche, axe y vers le bas — convention ALTO et PAGE
standard).  L'IoU est calculée sur l'aire d'intersection / union des
rectangles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Modèle de données
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Region:
    """Une région ALTO/PAGE alignable sur sa GT.

    Attributs
    ---------
    id:
        Identifiant unique au sein de la séquence (ex. ``"r_1"``,
        ``"region_main"``).  Informatif — l'alignement se fait par IoU,
        pas par ID.
    type:
        Catégorie de la région (``"TextRegion"``, ``"MarginNote"``,
        ``"Header"``, etc.).  Comparaison **case-insensitive**.
    bbox:
        Rectangle ``(x, y, width, height)`` en pixels, origine en haut
        à gauche.  Doit avoir width > 0 et height > 0.
    """

    id: str
    type: str
    bbox: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        x, y, w, h = self.bbox
        if w <= 0 or h <= 0:
            raise ValueError(
                f"Region {self.id!r} : bbox invalide (w={w}, h={h}). "
                "width et height doivent être strictement positifs."
            )

    @property
    def area(self) -> int:
        _, _, w, h = self.bbox
        return w * h


def _to_region(obj: Region | dict) -> Region:
    """Coerce un dict en ``Region`` (clés ``id``, ``type``, ``bbox``)."""
    if isinstance(obj, Region):
        return obj
    return Region(
        id=str(obj["id"]),
        type=str(obj["type"]),
        bbox=tuple(obj["bbox"]),  # type: ignore[arg-type]
    )


# ──────────────────────────────────────────────────────────────────────────
# IoU + alignement greedy
# ──────────────────────────────────────────────────────────────────────────


def _iou_bbox(a: Region, b: Region) -> float:
    """Intersection-over-Union de deux bboxes ``(x, y, w, h)``."""
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    inter_x = max(ax, bx)
    inter_y = max(ay, by)
    inter_x_end = min(ax + aw, bx + bw)
    inter_y_end = min(ay + ah, by + bh)
    inter_w = max(0, inter_x_end - inter_x)
    inter_h = max(0, inter_y_end - inter_y)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _align_regions(
    references: list[Region],
    hypotheses: list[Region],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Appareillage greedy par IoU décroissant ; same type requis.

    Renvoie ``(matches, unmatched_refs, unmatched_hyps)`` —
    ``matches`` est une liste de ``(idx_ref, idx_hyp, iou)``.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, r in enumerate(references):
        for j, h in enumerate(hypotheses):
            if r.type.casefold() != h.type.casefold():
                continue
            iou = _iou_bbox(r, h)
            if iou >= iou_threshold:
                candidates.append((iou, i, j))

    # Tri stable : IoU décroissant, puis indices croissants pour
    # déterminisme sur égalités.
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    matched_refs: set[int] = set()
    matched_hyps: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, i, j in candidates:
        if i in matched_refs or j in matched_hyps:
            continue
        matched_refs.add(i)
        matched_hyps.add(j)
        matches.append((i, j, iou))

    unmatched_refs = set(range(len(references))) - matched_refs
    unmatched_hyps = set(range(len(hypotheses))) - matched_hyps
    return matches, unmatched_refs, unmatched_hyps


# ──────────────────────────────────────────────────────────────────────────
# Métrique principale
# ──────────────────────────────────────────────────────────────────────────


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "support": tp + fn}


def compute_layout_metrics(
    reference_regions: Iterable[Region | dict] | None,
    hypothesis_regions: Iterable[Region | dict] | None,
    iou_threshold: float = 0.5,
) -> dict:
    """Calcule precision/recall/F1 sur le layout par type de région.

    Parameters
    ----------
    reference_regions:
        Liste de régions GT (``Region`` ou dict ``{id, type, bbox}``).
    hypothesis_regions:
        Liste de régions produites par le moteur OCR/HTR ou un
        layout-detector.
    iou_threshold:
        Seuil de chevauchement minimal pour déclarer un appariement
        (défaut : 0,5 — convention ICDAR).

    Returns
    -------
    dict
        ``{
            "global": {"precision", "recall", "f1", "support"},
            "per_type": {type_name: {"precision", ...}},
            "true_positives": int,
            "false_positives": int,
            "false_negatives": int,
            "missed_regions": list[dict],          # GT non matchées
            "hallucinated_regions": list[dict],    # hyp non matchées
            "iou_threshold": float,
        }``

    Cas dégénérés
    -------------
    - Deux listes vides → F1 = 0 et tous compteurs à 0.
    - GT vide + hyp non-vide → F1 = 0 (toutes hyp = FP).
    - hyp vide + GT non-vide → F1 = 0 (toutes GT = FN).
    """
    refs = [_to_region(r) for r in (reference_regions or [])]
    hyps = [_to_region(h) for h in (hypothesis_regions or [])]

    matches, unmatched_refs, unmatched_hyps = _align_regions(
        refs, hyps, iou_threshold,
    )

    tp = len(matches)
    fn = len(unmatched_refs)
    fp = len(unmatched_hyps)

    cat_tp: dict[str, int] = {}
    cat_fn: dict[str, int] = {}
    cat_fp: dict[str, int] = {}
    for i, _j, _iou in matches:
        cat = refs[i].type
        cat_tp[cat] = cat_tp.get(cat, 0) + 1
    for i in unmatched_refs:
        cat = refs[i].type
        cat_fn[cat] = cat_fn.get(cat, 0) + 1
    for j in unmatched_hyps:
        cat = hyps[j].type
        cat_fp[cat] = cat_fp.get(cat, 0) + 1

    all_categories = sorted(set(cat_tp) | set(cat_fn) | set(cat_fp))
    per_type = {
        cat: _prf(
            cat_tp.get(cat, 0),
            cat_fp.get(cat, 0),
            cat_fn.get(cat, 0),
        )
        for cat in all_categories
    }

    return {
        "global": _prf(tp, fp, fn),
        "per_type": per_type,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "missed_regions": [
            {"id": refs[i].id, "type": refs[i].type, "bbox": list(refs[i].bbox)}
            for i in sorted(unmatched_refs)
        ],
        "hallucinated_regions": [
            {"id": hyps[j].id, "type": hyps[j].type, "bbox": list(hyps[j].bbox)}
            for j in sorted(unmatched_hyps)
        ],
        "iou_threshold": iou_threshold,
    }


def layout_f1(
    reference_regions: Iterable[Region | dict] | None,
    hypothesis_regions: Iterable[Region | dict] | None,
    iou_threshold: float = 0.5,
) -> float:
    """Raccourci : F1 global du layout."""
    return compute_layout_metrics(
        reference_regions, hypothesis_regions, iou_threshold,
    )["global"]["f1"]


__all__ = [
    "Region",
    "compute_layout_metrics",
    "layout_f1",
]
