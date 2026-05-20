"""Calcul des métriques de précision sur entités nommées (NER).

A.II.1.a du plan d'évolution 2026 : couche de calcul pure.

Pourquoi ce module
------------------
Pour un médiéviste, un archiviste ou un économiste-historien,
l'utilité aval d'un OCR ne se mesure pas seulement au CER ; ce qui
compte c'est de savoir si les **entités nommées** (personnes, lieux,
dates, organisations) ont survécu à la transcription.  Un CER de 5 %
qui rate 80 % des noms propres est inutilisable pour l'indexation
prosopographique.

Stratégie de découpage en sprints
---------------------------------
Comme pour la divergence taxonomique (Sprints 35-37), on découpe :

- **Sprint 38** (ici) — couche de calcul pure : alignement IoU entre
  deux listes d'entités, calcul de Precision/Recall/F1 par catégorie
  et global, détection des hallucinations d'entité.  Aucune dépendance
  externe (pas de spaCy, pas de Stanza) ; les listes d'entités sont
  fournies en entrée.  Un test de l'enregistrement dans le registre
  typé Sprint 34 garantit l'intégration.
- **Sprint à venir** — backend extracteur (spaCy / Stanza / HIPE) et
  câblage runner+narratif+HTML.

Format des entités
------------------
Compatible avec ``EntitiesGT`` du Sprint 32 — chaque entité est un
dictionnaire ``{"label": str, "start": int, "end": int, "text": str}``
où ``start``/``end`` sont des offsets caractère.

Convention d'alignement
-----------------------
Une entité hypothèse "matche" une entité de référence si :

1. les **labels sont identiques** (case-insensitive),
2. le ratio d'**Intersection-over-Union** (IoU) sur leurs spans
   caractère est ``≥ iou_threshold`` (défaut : 0,5).

Une entité de référence non matchée → faux négatif (recall pénalisé).
Une entité hypothèse non matchée → faux positif (précision pénalisée).
Un faux positif est aussi compté comme **hallucination d'entité**, ce
qui est utile pour les VLM/LLM qui inventent.

Limites
-------
- L'alignement bag-of-spans : une entité peut être matchée par au plus
  une entité de l'autre côté (sinon double-comptage).
- Les modèles NER (spaCy, etc.) hallucinent eux-mêmes.  La métrique
  mesure conjointement OCR + NER.  Documenter explicitement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from picarones.evaluation.metric_registry import register_metric
from picarones.domain.artifacts import ArtifactType

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Modèle de données
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Entity:
    """Entité nommée alignée sur un texte.

    Attributs
    ---------
    label:
        Catégorie de l'entité (ex. ``"PER"``, ``"LOC"``, ``"DATE"``).
        La comparaison se fait en *case-insensitive*.
    start, end:
        Offsets caractère (inclus, exclu) sur le texte de référence.
    text:
        Forme de surface — informative, **non utilisée pour
        l'alignement** (deux entités peuvent matcher même si leur
        forme de surface diffère, du moment que leurs spans
        chevauchent suffisamment).
    """

    label: str
    start: int
    end: int
    text: str = ""

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(
                f"Entity span invalide : start={self.start} > end={self.end}"
            )

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


def _to_entity(obj: Entity | dict) -> Entity:
    """Coerce un dict (format EntitiesGT) en ``Entity``."""
    if isinstance(obj, Entity):
        return obj
    return Entity(
        label=str(obj["label"]),
        start=int(obj["start"]),
        end=int(obj["end"]),
        text=str(obj.get("text", "")),
    )


# ──────────────────────────────────────────────────────────────────────────
# Alignement par IoU
# ──────────────────────────────────────────────────────────────────────────


def _iou(a: Entity, b: Entity) -> float:
    """Intersection-over-Union sur les spans caractère."""
    inter_start = max(a.start, b.start)
    inter_end = min(a.end, b.end)
    inter = max(0, inter_end - inter_start)
    union = a.length + b.length - inter
    if union <= 0:
        return 0.0
    return inter / union


def _align(
    references: list[Entity],
    hypotheses: list[Entity],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Aligne deux listes d'entités par IoU décroissant (greedy).

    Returns
    -------
    matches:
        Liste de triplets ``(idx_ref, idx_hyp, iou)`` triés par IoU
        décroissant — chaque entité n'apparaît qu'une fois.
    unmatched_refs:
        Indices des entités GT non matchées (faux négatifs).
    unmatched_hyps:
        Indices des entités hypothèse non matchées (faux positifs).
    """
    candidates: list[tuple[float, int, int]] = []
    for i, r in enumerate(references):
        for j, h in enumerate(hypotheses):
            if r.label.casefold() != h.label.casefold():
                continue
            score = _iou(r, h)
            if score >= iou_threshold:
                candidates.append((score, i, j))

    # Tri par IoU décroissant ; à IoU égale, on prend l'ordre des paires
    # pour garantir un tri stable et déterministe.
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    matched_refs: set[int] = set()
    matched_hyps: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, i, j in candidates:
        if i in matched_refs or j in matched_hyps:
            continue
        matched_refs.add(i)
        matched_hyps.add(j)
        matches.append((i, j, score))

    unmatched_refs = set(range(len(references))) - matched_refs
    unmatched_hyps = set(range(len(hypotheses))) - matched_hyps
    return matches, unmatched_refs, unmatched_hyps


# ──────────────────────────────────────────────────────────────────────────
# Calcul des métriques
# ──────────────────────────────────────────────────────────────────────────


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Précision / rappel / F1 à partir des comptes."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": tp + fn,
    }


def compute_ner_metrics(
    reference_entities: Iterable[Entity | dict],
    hypothesis_entities: Iterable[Entity | dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Calcule la précision/rappel/F1 sur entités nommées.

    Parameters
    ----------
    reference_entities:
        Liste d'entités GT (format ``Entity`` ou dict de
        ``EntitiesGT``).
    hypothesis_entities:
        Liste d'entités produites par le NER sur la sortie OCR.
    iou_threshold:
        Seuil de chevauchement caractère pour qu'un appariement
        soit valide (défaut : 0,5 — convention CoNLL/HIPE).

    Returns
    -------
    dict
        ``{
            "global": {"precision", "recall", "f1", "support"},
            "per_category": {label: {"precision", ...}},
            "true_positives": int,
            "false_positives": int,
            "false_negatives": int,
            "hallucinated_entities": list[dict],   # entités OCR sans GT
            "missed_entities":       list[dict],   # entités GT non détectées
            "iou_threshold": float,
        }``
    """
    refs = [_to_entity(e) for e in reference_entities]
    hyps = [_to_entity(e) for e in hypothesis_entities]

    matches, unmatched_refs, unmatched_hyps = _align(refs, hyps, iou_threshold)

    tp = len(matches)
    fn = len(unmatched_refs)
    fp = len(unmatched_hyps)

    # Comptes par catégorie
    cat_tp: dict[str, int] = {}
    cat_fn: dict[str, int] = {}
    cat_fp: dict[str, int] = {}
    for i, _j, _score in matches:
        cat = refs[i].label
        cat_tp[cat] = cat_tp.get(cat, 0) + 1
    for i in unmatched_refs:
        cat = refs[i].label
        cat_fn[cat] = cat_fn.get(cat, 0) + 1
    for j in unmatched_hyps:
        cat = hyps[j].label
        cat_fp[cat] = cat_fp.get(cat, 0) + 1

    all_categories = sorted(set(cat_tp) | set(cat_fn) | set(cat_fp))
    per_category = {
        cat: _prf(cat_tp.get(cat, 0), cat_fp.get(cat, 0), cat_fn.get(cat, 0))
        for cat in all_categories
    }

    return {
        "global": _prf(tp, fp, fn),
        "per_category": per_category,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "hallucinated_entities": [
            {"label": hyps[j].label, "start": hyps[j].start,
             "end": hyps[j].end, "text": hyps[j].text}
            for j in sorted(unmatched_hyps)
        ],
        "missed_entities": [
            {"label": refs[i].label, "start": refs[i].start,
             "end": refs[i].end, "text": refs[i].text}
            for i in sorted(unmatched_refs)
        ],
        "iou_threshold": iou_threshold,
    }


# ──────────────────────────────────────────────────────────────────────────
# Enregistrement dans le registre typé
# ──────────────────────────────────────────────────────────────────────────


@register_metric(
    name="ner_f1",
    input_types=(ArtifactType.ENTITIES, ArtifactType.ENTITIES),
    description=(
        "F1 global sur les entités nommées (alignement IoU ≥ 0,5, "
        "labels case-insensitive). Pour le détail par catégorie, "
        "utiliser compute_ner_metrics directement."
    ),
    higher_is_better=True,
    tags={"downstream", "ner", "structure"},
)
def ner_f1(
    reference_entities: Iterable[Entity | dict],
    hypothesis_entities: Iterable[Entity | dict],
) -> float:
    """F1 global ; raccourci enregistré pour les jonctions ``(ENTITIES, ENTITIES)``."""
    return compute_ner_metrics(reference_entities, hypothesis_entities)["global"]["f1"]


__all__ = [
    "Entity",
    "compute_ner_metrics",
    "ner_f1",
]
