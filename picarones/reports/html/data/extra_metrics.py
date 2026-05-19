"""Métriques additionnelles consommées par le rapport HTML.

Sprint « câblage des modules test-only » (mai 2026) : intègre dans le
flux de génération du rapport des modules de mesure qui jusque-là
n'étaient appelés par aucun consommateur en production. Concrètement :

- :func:`compute_rare_token_recall_per_engine` — Sprint 71 (A.I.1) :
  recall sur tokens rares (hapax + dis legomena) corpus-wide. Discrimine
  un OCR qui rate les noms propres rares (critique pour l'indexation
  prosopographique).
- :func:`compute_taxonomy_cooccurrence_section` — Sprint 75 (A.I.4
  chantier 1) : indice de Jaccard inter-classes au niveau document.
- :func:`compute_taxonomy_intra_doc_section` — Sprint 76 (A.I.4
  chantier 2) : heatmap class × position pour repérer les zones
  concentrées d'erreur.
- :func:`compute_marginal_cost_section` — Sprint 91 (A.II.6) : coût
  marginal d'un moteur B vs A par erreur évitée.

Toutes les fonctions sont **pures** (pas de mutation in-place) et
retournent ``None`` ou un dict vide quand les pré-requis ne sont pas
réunis (corpus vide, taxonomy absente, etc.) — pattern adaptive masking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from picarones.evaluation.metrics.marginal_cost import compute_marginal_cost_matrix
from picarones.evaluation.metrics.rare_tokens import (
    compute_rare_token_recall,
    extract_rare_tokens,
)
from picarones.evaluation.metrics.taxonomy_cooccurrence import (
    compute_taxonomy_cooccurrence,
)
from picarones.evaluation.metrics.taxonomy_intra_doc import (
    compute_taxonomy_position_heatmap,
)

if TYPE_CHECKING:
    from picarones.evaluation.benchmark_result import BenchmarkResult


# ──────────────────────────────────────────────────────────────────
# Rare-token recall
# ──────────────────────────────────────────────────────────────────


def compute_rare_token_recall_per_engine(
    benchmark: "BenchmarkResult",
    max_freq: int = 2,
) -> dict[str, dict]:
    """Recall corpus-wide sur les tokens rares pour chaque moteur.

    Étapes :
    1. Extraire les tokens rares du corpus (apparaissent ≤ ``max_freq``
       fois dans toutes les GT).
    2. Pour chaque moteur, calculer le recall moyen pondéré par doc.

    Retour : ``{engine_name: {n_rare_tokens, n_recalled, recall, n_docs}}``,
    vide si aucun moteur ou aucun token rare détecté.
    """
    if not benchmark.engine_reports:
        return {}
    # Liste des GT du corpus (premier moteur fait foi).
    gts = [
        dr.ground_truth
        for dr in benchmark.engine_reports[0].document_results
        if dr.ground_truth
    ]
    if not gts:
        return {}
    rare_tokens = extract_rare_tokens(gts, max_freq=max_freq)
    if not rare_tokens:
        return {}

    out: dict[str, dict] = {}
    for report in benchmark.engine_reports:
        n_total_rare = 0
        n_total_recalled = 0
        n_docs = 0
        for dr in report.document_results:
            if dr.metrics.error is not None:
                continue
            metrics = compute_rare_token_recall(
                dr.ground_truth, dr.hypothesis, rare_tokens,
            )
            n_total_rare += metrics["n_rare_tokens_in_reference"]
            n_total_recalled += metrics["n_rare_tokens_recalled"]
            n_docs += 1
        recall = (
            n_total_recalled / n_total_rare if n_total_rare > 0 else None
        )
        out[report.engine_name] = {
            "n_rare_tokens": n_total_rare,
            "n_recalled": n_total_recalled,
            "recall": recall,
            "n_docs": n_docs,
            "max_freq": max_freq,
        }
    return out


# ──────────────────────────────────────────────────────────────────
# Co-occurrence taxonomique
# ──────────────────────────────────────────────────────────────────


def compute_taxonomy_cooccurrence_section(
    benchmark: "BenchmarkResult",
) -> Optional[dict]:
    """Calcule la matrice de co-occurrence taxonomique corpus-wide.

    Pour chaque document, on collecte l'union des classes d'erreur
    apparues sur ce document tous moteurs confondus, puis on calcule
    l'indice de Jaccard entre paires de classes au niveau corpus.

    Retour : sortie de
    :func:`picarones.evaluation.metrics.taxonomy_cooccurrence.compute_taxonomy_cooccurrence`,
    ou ``None`` si aucune classification taxonomique n'est disponible.
    """
    # Map doc_id → index dans per_doc_classes pour merger correctement
    # les classes des moteurs additionnels qui évaluent le même doc.
    # **Bug évité** : ne PAS utiliser un set pour retrouver l'index — un
    # set n'a pas d'ordre garanti, ``list(set).index(x)`` retourne un
    # index qui ne correspond pas à la position dans la liste parallèle.
    doc_id_to_idx: dict[str, int] = {}
    per_doc_classes: list[set[str]] = []

    for report in benchmark.engine_reports:
        for dr in report.document_results:
            if dr.taxonomy is None:
                continue
            classes = {
                cls
                for cls, count in (dr.taxonomy.get("counts") or {}).items()
                if count > 0
            }
            if not classes:
                continue
            idx = doc_id_to_idx.get(dr.doc_id)
            if idx is None:
                doc_id_to_idx[dr.doc_id] = len(per_doc_classes)
                per_doc_classes.append(classes)
            else:
                # Doc déjà vu (autre moteur) : merger les classes.
                per_doc_classes[idx] |= classes

    if not per_doc_classes:
        return None
    return compute_taxonomy_cooccurrence(per_doc_classes)


# ──────────────────────────────────────────────────────────────────
# Heatmap intra-document class × position
# ──────────────────────────────────────────────────────────────────


def compute_taxonomy_intra_doc_section(
    benchmark: "BenchmarkResult",
    n_bins: int = 10,
) -> Optional[dict]:
    """Heatmap agrégée class × position binnée sur l'ensemble du corpus.

    Pour chaque doc unique on garde le heatmap calculé par le **premier**
    moteur (déduplication : un même doc évalué par N moteurs ne compte
    qu'une fois). Puis on somme par classe et bin de position.

    Retourne un dict compatible avec
    :func:`picarones.reports.html.renderers.taxonomy_intra_doc.build_taxonomy_intra_doc_html`
    (clés ``n_bins``, ``per_class``, ``total_errors``, ``n_words_gt``).
    Retourne ``None`` si aucun document n'a de signal exploitable.
    """
    aggregated: dict[str, list[int]] = {}
    seen_doc_ids: set[str] = set()
    total_errors = 0
    n_words_gt = 0

    for report in benchmark.engine_reports:
        for dr in report.document_results:
            if dr.doc_id in seen_doc_ids:
                continue  # déduplication : ne pas compter un doc 2 fois
            if dr.metrics.error is not None or not dr.ground_truth:
                continue
            heatmap = compute_taxonomy_position_heatmap(
                dr.ground_truth, dr.hypothesis, n_bins=n_bins,
            )
            if heatmap is None:
                continue
            seen_doc_ids.add(dr.doc_id)
            n_words_gt += len(dr.ground_truth.split())
            per_class = heatmap.get("per_class", {})
            for cls, counts in per_class.items():
                cls_total = sum(counts)
                if cls_total == 0:
                    continue
                total_errors += cls_total
                if cls not in aggregated:
                    aggregated[cls] = [0] * n_bins
                for i in range(n_bins):
                    aggregated[cls][i] += counts[i] if i < len(counts) else 0

    if not aggregated:
        return None
    return {
        "n_bins": n_bins,
        "n_docs_with_data": len(seen_doc_ids),
        "total_errors": total_errors,
        "n_words_gt": n_words_gt,
        "per_class": aggregated,
    }


# ──────────────────────────────────────────────────────────────────
# Coût marginal inter-moteurs
# ──────────────────────────────────────────────────────────────────


def compute_marginal_cost_section(
    engines_summary: list[dict],
) -> Optional[list[dict]]:
    """Matrice de coût marginal entre paires de moteurs.

    Lit ``cost`` (attaché par :func:`attach_engine_costs`) et estime
    le nombre d'erreurs. Pour chaque paire ``A → B``, calcule le coût
    additionnel par erreur évitée.

    **Note d'estimation** : le nombre d'erreurs est dérivé de
    ``cer × n_caractères_corpus`` quand la longueur moyenne de doc
    est disponible, sinon repli sur ``cer × 1000`` (proxy pour
    1000 caractères standardisés). Les coûts marginaux affichés sont
    des estimations pessimistes — pour un benchmark de corpus
    homogène, l'ordonnancement est fiable ; pour un mix de
    types de documents, à interpréter avec prudence.

    Retour : liste de dicts (sortie ``["pairs"]`` de
    :func:`compute_marginal_cost_matrix`) triée par coût marginal
    croissant, ou ``None`` si moins de 2 moteurs ont des données
    coût + erreur exploitables.
    """
    per_engine: dict[str, dict] = {}
    for entry in engines_summary:
        cost = entry.get("cost") or {}
        cost_per_1k = cost.get("cost_per_1k_pages_eur")
        cer = entry.get("cer")
        doc_count = entry.get("doc_count") or 0
        if cost_per_1k is None or cer is None or doc_count == 0:
            continue
        # Proxy : cer × 1000 caractères / page (échelle stable cohérente
        # avec ``cost_per_1k_pages_eur``).
        estimated_errors = cer * 1000.0
        per_engine[entry["name"]] = {
            "cost": cost_per_1k,
            "errors": estimated_errors,
        }
    if len(per_engine) < 2:
        return None
    result = compute_marginal_cost_matrix(per_engine)
    if not result:
        return None
    # ``compute_marginal_cost_matrix`` retourne ``{"pairs": [...]}``.
    # On expose la liste ``pairs`` pour que le renderer reçoive un
    # itérable de dicts (pas un wrapper).
    return result.get("pairs") or None


__all__ = [
    "compute_rare_token_recall_per_engine",
    "compute_taxonomy_cooccurrence_section",
    "compute_taxonomy_intra_doc_section",
    "compute_marginal_cost_section",
]
