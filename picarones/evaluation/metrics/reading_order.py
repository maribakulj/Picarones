"""Reading order F1 (ICDAR 2015, Antonacopoulos)

A.II.2.1 du plan d'évolution 2026.

Pourquoi ce module
------------------
Sur un manuscrit glosé, un journal multi-colonnes ou un registre
paroissial complexe, le **classement des moteurs en CER** peut être
trompeur : un moteur peut avoir un excellent CER caractère et un
**ordre de lecture catastrophique**.  Le résultat est inutilisable
pour la recherche plein texte (Elastic, Solr) ou pour reconstituer
une narration linéaire.

La métrique standard est définie par Antonacopoulos et al. dans
ICDAR 2015 — F1 sur les **paires d'ordre relatif** entre régions
ALTO/PAGE.  Pour chaque paire ``(a, b)`` telle que ``a`` précède
``b`` dans la GT :

- **TP** si ``a`` précède aussi ``b`` dans l'hypothèse,
- **FN** si la paire est manquante (régions absentes ou ordre
  inversé) côté hypothèse,
- **FP** si une paire ``(a, b)`` apparaît dans l'hypothèse alors que
  la GT n'a pas cet ordre (régions hallucinées ou inversion).

Le F1 est la moyenne harmonique des deux.

Stratégie de découpage
----------------------
Cohérent avec NER (Sprint 38), calibration (Sprint 39), Flesch
couche de calcul pure d'abord.  L'utilisateur fournit
deux listes ordonnées d'IDs de régions (typiquement extraites de
ALTO/PAGE par un parser amont).  Le câblage runner et la vue HTML
suivent dans des sprints dédiés.

Compatible directement avec ``ReadingOrderGT`` du Sprint 32 :
``ReadingOrderGT.region_order`` est exactement le format attendu.

Convention sur les régions
--------------------------
- Les IDs sont des chaînes (``"r_1"``, ``"region_main"``, etc.).
- Les **doublons** sont ignorés au calcul des paires ordonnées
  (chaque ID compte une fois par séquence).
- Une région présente dans la GT mais absente de l'hypothèse
  contribue aux paires FN.
- Une région présente dans l'hypothèse mais absente de la GT
  contribue aux paires FP.
- Si une séquence a < 2 régions distinctes, aucune paire n'est
  émise — le F1 retourne ``0.0`` ou ``1.0`` selon que les deux
  séquences soient identiques.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Iterable

from picarones.evaluation.metric_registry import register_metric
from picarones.domain.artifacts import ArtifactType

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _ordered_pairs(sequence: list[str]) -> set[tuple[str, str]]:
    """Retourne l'ensemble des paires ``(a, b)`` telles que ``a``
    précède strictement ``b`` dans ``sequence``.

    Doublons : chaque ID est traité une seule fois (première occurrence
    dans la séquence).  Cohérent avec ICDAR 2015 où les régions ont
    des IDs uniques.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for r in sequence:
        if r not in seen_set:
            seen.append(r)
            seen_set.add(r)
    return set(combinations(seen, 2))


def _normalize_input(value: Iterable[str] | None) -> list[str]:
    """Coerce une entrée en list[str], en filtrant les valeurs vides."""
    if value is None:
        return []
    return [str(v) for v in value if v is not None and str(v).strip()]


# ──────────────────────────────────────────────────────────────────────────
# Métrique principale
# ──────────────────────────────────────────────────────────────────────────


def compute_reading_order_metrics(
    reference_order: Iterable[str] | None,
    hypothesis_order: Iterable[str] | None,
) -> dict:
    """Calcule precision / recall / F1 sur l'ordre relatif des régions.

    Parameters
    ----------
    reference_order:
        Séquence ordonnée d'IDs de régions issue de la GT (typiquement
        ``ReadingOrderGT.region_order`` du Sprint 32).
    hypothesis_order:
        Séquence ordonnée d'IDs de régions produite par un moteur
        OCR/HTR ou un reconstructeur ALTO.

    Returns
    -------
    dict
        ``{"precision", "recall", "f1", "true_positives",
        "false_positives", "false_negatives", "n_ref_pairs",
        "n_hyp_pairs", "common_regions", "ref_only_regions",
        "hyp_only_regions"}``.

    Comportements aux bornes
    ------------------------
    - Deux séquences identiques (mêmes régions, même ordre) → F1 = 1.0.
    - Ordre strictement inversé → F1 = 0.0 (toutes les paires
      relatives sont fausses).
    - Une séquence vide vs une séquence non vide → F1 = 0.0.
    - Deux séquences vides → F1 = 0.0 et tous les compteurs à 0
      (convention : on ne récompense pas l'absence).
    """
    ref = _normalize_input(reference_order)
    hyp = _normalize_input(hypothesis_order)

    ref_pairs = _ordered_pairs(ref)
    hyp_pairs = _ordered_pairs(hyp)

    # Audit Classe B : la GT ne fournit aucune paire ordonnée (< 2
    # régions distinctes) ⇒ l'ordre de lecture n'est **pas évaluable**
    # ⇒ scores ``None`` (omis en agrégation), au lieu de 0.0 qui
    # comptait à tort comme un échec un document sans ordre à juger.
    if not ref_pairs:
        ref_set0 = set(ref)
        hyp_set0 = set(hyp)
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "true_positives": 0,
            "false_positives": len(hyp_pairs),
            "false_negatives": 0,
            "n_ref_pairs": 0,
            "n_hyp_pairs": len(hyp_pairs),
            "common_regions": sorted(ref_set0 & hyp_set0),
            "ref_only_regions": sorted(ref_set0 - hyp_set0),
            "hyp_only_regions": sorted(hyp_set0 - ref_set0),
        }

    tp = len(ref_pairs & hyp_pairs)
    fn = len(ref_pairs - hyp_pairs)
    fp = len(hyp_pairs - ref_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    ref_set = set(ref)
    hyp_set = set(hyp)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "n_ref_pairs": len(ref_pairs),
        "n_hyp_pairs": len(hyp_pairs),
        "common_regions": sorted(ref_set & hyp_set),
        "ref_only_regions": sorted(ref_set - hyp_set),
        "hyp_only_regions": sorted(hyp_set - ref_set),
    }


# ──────────────────────────────────────────────────────────────────────────
# Enregistrement dans le registre typé
# ──────────────────────────────────────────────────────────────────────────


@register_metric(
    name="reading_order_f1",
    input_types=(ArtifactType.READING_ORDER, ArtifactType.READING_ORDER),
    description=(
        "F1 sur l'ordre relatif des régions ALTO/PAGE (ICDAR 2015, "
        "Antonacopoulos). Pour chaque paire (a,b) où a précède b dans "
        "la GT, vérifie que a précède aussi b dans l'hypothèse."
    ),
    higher_is_better=True,
    tags={"structure", "icdar", "alto", "page"},
)
def reading_order_f1(
    reference: Iterable[str] | None,
    hypothesis: Iterable[str] | None,
) -> float:
    """Raccourci : retourne uniquement le F1 global.

    Pour les détails par paire (TP/FP/FN, régions communes, etc.),
    appeler ``compute_reading_order_metrics`` directement.
    """
    return compute_reading_order_metrics(reference, hypothesis)["f1"]


__all__ = [
    "compute_reading_order_metrics",
    "reading_order_f1",
]
