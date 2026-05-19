"""Calculs de classement / stratification d'un ``BenchmarkResult``.

Extrait de ``benchmark_result.py`` (god-module, audit prod P1).
Logique **purement computationnelle** : fonctions libres sans état,
sans effet de bord, ne dépendant que de ``engine_reports`` et
``doc_strata``.  ``BenchmarkResult`` conserve des méthodes minces
qui délèguent ici — l'API publique (``result.ranking()``,
``result.stratified_ranking()``, …) est inchangée.

Duck-typing volontaire (annotations ``Any``/string) : importer
``EngineReport`` créerait un cycle ``benchmark_result`` ↔ ce module.
Seules les libs stdlib sont utilisées (``statistics``) — conforme à
la whitelist externe de la couche ``evaluation``.
"""

from __future__ import annotations

import statistics as _stats
from typing import Any, Mapping, Optional, Protocol, Sequence


# Protocols structurels (audit prod P1.3) — remplacent le duck-typing
# ``Any`` sans réintroduire de cycle ``benchmark_result`` ↔ ce module
# (un Protocol est structurel : aucun import de ``EngineReport``).
# Annotations seules (``from __future__ import annotations`` ⇒ lazy,
# zéro impact runtime) : documentent le contrat consommé ici.
class _MetricsLike(Protocol):
    error: object | None
    cer: float | None
    wer: float | None
    cer_errors: int | None
    cer_ref_chars: int | None
    wer_errors: int | None
    wer_ref_words: int | None


class _DocResultLike(Protocol):
    doc_id: str
    metrics: _MetricsLike | None


class _EngineReportLike(Protocol):
    engine_name: str
    micro_cer: float | None
    micro_wer: float | None
    mean_cer: float | None
    median_cer: float | None
    mean_wer: float | None
    document_results: Sequence[_DocResultLike]
    aggregated_metrics: Mapping[str, Any]


def _sort_key(entry: dict) -> tuple:
    """Priorité scientifique : micro-CER ; repli médiane puis moyenne ;
    +∞ si rien (moteur sans document exploitable)."""
    primary = entry.get("micro_cer")
    if primary is None:
        primary = entry.get("median_cer")
    if primary is None:
        primary = entry.get("mean_cer")
    return (primary is None, primary if primary is not None else float("inf"))


def compute_ranking(engine_reports: Sequence[_EngineReportLike]) -> list[dict]:
    """Classement des moteurs trié par **CER micro-moyenné** croissant.

    Audit scientifique F1 (mai 2026) — le tri par défaut bascule vers
    le **micro-CER** (Σ distance_édition / Σ caractères_référence),
    métrique d'agrégation standard du domaine OCR/HTR (ICDAR, OCR-D,
    HTR-United, Transkribus, eScriptorium).  C'est la seule agrégation
    défendable scientifiquement comme chiffre d'en-tête : elle
    pondère chaque document par sa longueur, là où une moyenne ou une
    médiane de taux par document donne le même poids à une légende de
    10 caractères et à une page de 5 000 et peut inverser le
    classement réel des moteurs.

    Historique : Sprint 44 avait basculé moyenne → médiane pour la
    robustesse à l'asymétrie des corpus patrimoniaux.  Le diagnostic
    de fond (la *moyenne* est tirée par quelques documents
    catastrophiques) est exact, mais la *réponse* correcte n'est pas
    la médiane de taux (toujours aveugle à la longueur) : c'est le
    micro-CER.  ``mean_cer`` et ``median_cer`` restent exposés dans
    chaque entrée comme **diagnostics de dispersion** (un grand écart
    micro↔médiane signale une distribution très hétérogène — cf.
    détecteur ``median_mean_gap_warning``), pas comme critère de
    classement.

    Le tri prend ``micro_cer`` quand disponible et retombe sur
    ``median_cer`` puis ``mean_cer`` (corpus sans comptes bruts :
    jiwer absent, références vides).
    """
    ranked = []
    for report in engine_reports:
        ranked.append(
            {
                "engine": report.engine_name,
                "micro_cer": report.micro_cer,
                "micro_wer": report.micro_wer,
                "mean_cer": report.mean_cer,
                "median_cer": report.median_cer,
                "mean_wer": report.mean_wer,
                "documents": len(report.document_results),
                "failed": report.aggregated_metrics.get("failed_count", 0),
            }
        )

    return sorted(ranked, key=_sort_key)


def available_strata(doc_strata: Optional[dict[str, str]]) -> list[str]:
    """Liste triée des strates ``script_type`` distinctes du corpus.

    Vide si ``doc_strata`` est ``None`` ou si aucun document n'a de
    valeur non vide.  Garantit un ordre stable (tri lexical).
    """
    if not doc_strata:
        return []
    return sorted({s for s in doc_strata.values() if s})


def doc_ids_in_stratum(
    doc_strata: Optional[dict[str, str]], stratum: str,
) -> set[str]:
    """Ensemble des ``doc_id`` dont la strate est ``stratum``."""
    if not doc_strata:
        return set()
    return {
        doc_id for doc_id, st in doc_strata.items()
        if st == stratum
    }


def compute_stratified_ranking(
    engine_reports: Sequence[_EngineReportLike],
    doc_strata: Optional[dict[str, str]],
) -> dict[str, list[dict]]:
    """Retourne un classement séparé par strate ``script_type``.

    Pour chaque strate, recalcule mean/median/micro CER **uniquement
    sur les documents de la strate** et trie comme ``compute_ranking``
    (micro-CER, repli médiane/moyenne).

    Returns
    -------
    dict[str, list[dict]]
        ``{stratum_name: [ranking_entry, ...]}``.  Vide si pas de
        stratification disponible (``doc_strata`` non renseigné).
        Chaque ``ranking_entry`` a la même structure que
        ``compute_ranking`` : ``engine``, ``micro_cer``, ``micro_wer``,
        ``mean_cer``, ``median_cer``, ``mean_wer``, ``documents``,
        ``failed``.
    """
    strata = available_strata(doc_strata)
    if not strata:
        return {}

    result: dict[str, list[dict]] = {}
    for stratum in strata:
        doc_ids = doc_ids_in_stratum(doc_strata, stratum)
        if not doc_ids:
            continue

        entries: list[dict] = []
        for report in engine_reports:
            # ``Sprint A14-S1`` : ``MetricsResult.cer`` / ``.wer`` sont
            # ``Optional[float]`` ; le double filtre ``error is None``
            # garantit ``cer/wer is not None`` par convention, mais on
            # le filtre explicitement aussi pour que mypy le voie.
            stratum_metrics = [
                dr.metrics
                for dr in report.document_results
                if dr.doc_id in doc_ids
                and dr.metrics is not None
                and dr.metrics.error is None
            ]
            cers: list[float] = [
                m.cer for m in stratum_metrics if m.cer is not None
            ]
            wers: list[float] = [
                m.wer for m in stratum_metrics if m.wer is not None
            ]
            # Micro-CER/WER de la strate (audit F1) — recalcul depuis
            # les comptes bruts, cohérent avec ``compute_ranking``.
            tot_ce = sum(
                m.cer_errors for m in stratum_metrics
                if m.cer_errors is not None and m.cer_ref_chars is not None
            )
            tot_cr = sum(
                m.cer_ref_chars for m in stratum_metrics
                if m.cer_errors is not None and m.cer_ref_chars is not None
            )
            tot_we = sum(
                m.wer_errors for m in stratum_metrics
                if m.wer_errors is not None and m.wer_ref_words is not None
            )
            tot_wr = sum(
                m.wer_ref_words for m in stratum_metrics
                if m.wer_errors is not None and m.wer_ref_words is not None
            )
            micro_cer = round(tot_ce / tot_cr, 6) if tot_cr > 0 else None
            micro_wer = round(tot_we / tot_wr, 6) if tot_wr > 0 else None
            failed = sum(
                1 for dr in report.document_results
                if dr.doc_id in doc_ids
                and dr.metrics is not None
                and dr.metrics.error is not None
            )
            if not cers:
                entries.append({
                    "engine": report.engine_name,
                    "micro_cer": None,
                    "micro_wer": None,
                    "mean_cer": None,
                    "median_cer": None,
                    "mean_wer": None,
                    "documents": 0,
                    "failed": failed,
                })
                continue
            entries.append({
                "engine": report.engine_name,
                "micro_cer": micro_cer,
                "micro_wer": micro_wer,
                "mean_cer": _stats.mean(cers),
                "median_cer": _stats.median(cers),
                "mean_wer": _stats.mean(wers) if wers else None,
                "documents": len(cers),
                "failed": failed,
            })

        result[stratum] = sorted(entries, key=_sort_key)
    return result


def compute_corpus_homogeneity(
    engine_reports: Sequence[_EngineReportLike],
    doc_strata: Optional[dict[str, str]],
) -> Optional[dict]:
    """Mesure d'hétérogénéité du corpus du point de vue NER/OCR.

    Pour chaque moteur, calcule la variance des CER médians par
    strate.  Une variance élevée signale que le moteur se comporte
    très différemment selon le type de document — la moyenne globale
    est alors trompeuse et l'utilisateur doit consulter la vue
    stratifiée (cf. plan d'évolution A.III).

    Returns
    -------
    dict | None
        ``{
            "n_strata": int,
            "max_inter_strata_gap": float,        # plus grand écart sur le top moteur
            "leader": str,                         # moteur top global
            "leader_per_stratum_median": {strate: median_cer},
            "leader_max_gap_strata": [str, str],   # paire de strates qui maximise l'écart
        }``
        ``None`` si moins de 2 strates ou pas de leader.
    """
    strata_rankings = compute_stratified_ranking(engine_reports, doc_strata)
    if len(strata_rankings) < 2:
        return None

    global_ranking = compute_ranking(engine_reports)

    def _repr_cer(entry: dict) -> Optional[float]:
        # CER représentatif cohérent avec ``compute_ranking`` : micro
        # (audit F1) puis repli médiane / moyenne.
        for key in ("micro_cer", "median_cer", "mean_cer"):
            v = entry.get(key)
            if v is not None:
                return float(v)
        return None

    valid = [r for r in global_ranking if _repr_cer(r) is not None]
    if not valid:
        return None
    leader = valid[0]["engine"]

    # CER représentatif (micro, repli médiane) du leader sur chaque
    # strate où il a au moins 1 document.
    per_stratum: dict[str, float] = {}
    for stratum, entries in strata_rankings.items():
        for entry in entries:
            if entry["engine"] != leader:
                continue
            rc = _repr_cer(entry)
            if rc is None:
                continue
            per_stratum[stratum] = rc
            break

    if len(per_stratum) < 2:
        return None

    items = sorted(per_stratum.items(), key=lambda kv: kv[1])
    min_strata, min_med = items[0]
    max_strata, max_med = items[-1]
    max_gap = max_med - min_med

    return {
        "n_strata": len(strata_rankings),
        "max_inter_strata_gap": max_gap,
        "leader": leader,
        "leader_per_stratum_median": per_stratum,
        "leader_max_gap_strata": [min_strata, max_strata],
    }


__all__ = [
    "available_strata",
    "compute_corpus_homogeneity",
    "compute_ranking",
    "compute_stratified_ranking",
    "doc_ids_in_stratum",
]
