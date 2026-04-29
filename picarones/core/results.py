"""Modèle de données des résultats et export JSON.

Hiérarchie
----------
BenchmarkResult
  └── EngineReport          (un par moteur)
        └── DocumentResult  (un par document)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from picarones import __version__
from picarones.core.metrics import MetricsResult, aggregate_metrics


@dataclass
class DocumentResult:
    """Résultat d'un moteur sur un seul document."""

    doc_id: str
    image_path: str
    ground_truth: str
    hypothesis: str
    metrics: MetricsResult
    duration_seconds: float
    engine_error: Optional[str] = None
    # Champs spécifiques aux pipelines OCR+LLM
    ocr_intermediate: Optional[str] = None
    """Sortie OCR brute avant correction LLM (None pour les moteurs OCR seuls)."""
    pipeline_metadata: dict = field(default_factory=dict)
    """Métadonnées du pipeline : mode, prompt, over-normalization…"""
    # Champs Sprint 5 — métriques avancées patrimoniales
    confusion_matrix: Optional[dict] = None
    """Matrice de confusion unicode sérialisée."""
    char_scores: Optional[dict] = None
    """Scores ligatures et diacritiques."""
    taxonomy: Optional[dict] = None
    """Classification taxonomique des erreurs (classes 1-9)."""
    structure: Optional[dict] = None
    """Analyse structurelle (segmentation lignes, ordre lecture)."""
    image_quality: Optional[dict] = None
    """Métriques de qualité image."""
    # Champs Sprint 10 — distribution des erreurs + hallucinations VLM
    line_metrics: Optional[dict] = None
    """Distribution CER par ligne (percentiles, Gini, heatmap de position)."""
    hallucination_metrics: Optional[dict] = None
    """Métriques de détection des hallucinations VLM (ancrage, ratio longueur, blocs)."""
    # Champ Sprint 40 — métriques NER calculées si la GT a un EntitiesGT
    # ET qu'un EntityExtractor a été passé au runner.  ``None`` sinon.
    ner_metrics: Optional[dict] = None
    """Précision/rappel/F1 sur entités nommées (Sprint 38-40).

    Format : retour de ``compute_ner_metrics`` (global, per_category,
    hallucinated_entities, missed_entities, etc.).  Présent uniquement si
    le document a un niveau de GT ``ENTITIES`` ET que le runner a reçu
    un ``EntityExtractor``.
    """
    # Sprint 42 — calibration des confidences moteur (ECE, MCE, bins)
    calibration_metrics: Optional[dict] = None
    """Métriques de calibration (Sprint 39+42).

    Format : retour de ``compute_calibration_metrics`` (ece, mce,
    n_bins, n_predictions, overall_accuracy, overall_confidence, bins).
    Présent uniquement si le moteur a fourni des ``token_confidences``
    sur l'``EngineResult``.
    """
    # Sprint 61 — métriques philologiques (Sprints 55-60) calculées
    # automatiquement.  Présent uniquement si au moins un module a
    # détecté du signal dans la GT.
    philological_metrics: Optional[dict] = None
    """Métriques philologiques (Sprints 55-60).

    Dict avec une clé par module en présence de signal :

    - ``unicode_blocks``    : Sprint 55, retour de ``compute_unicode_block_accuracy``
    - ``abbreviations``     : Sprint 56, retour de ``compute_abbreviation_metrics``
    - ``mufi``              : Sprint 57, retour de ``compute_mufi_coverage``
    - ``early_modern``      : Sprint 58, retour de ``compute_early_modern_metrics``
    - ``modern_archives``   : Sprint 59, retour de ``compute_modern_archives_metrics``
    - ``roman_numerals``    : Sprint 60, retour de ``compute_roman_numeral_metrics``

    Un module n'est inclus que si la GT contient du signal exploitable
    (n_markers_reference > 0, n_mufi_chars_reference > 0, etc.).
    Cette logique adaptative permet de garder les rapports lisibles
    sur les corpus sans marqueurs philologiques.
    """
    # Sprint 86 — recherchabilité fuzzy (Sprint 84) calculée
    # automatiquement avec adaptive masking.
    searchability_metrics: Optional[dict] = None
    """Recherchabilité fuzzy (Sprint 84+86).

    Format : retour de ``compute_searchability`` ({n_gt_tokens,
    n_searchable, recall, missed_tokens, max_distance}). Présent
    uniquement si la GT contient au moins un token.
    """
    # Sprint 86 — précision sur séquences numériques (Sprint 85)
    # calculée automatiquement avec adaptive masking.
    numerical_sequence_metrics: Optional[dict] = None
    """Précision sur séquences numériques (Sprint 85+86).

    Format : retour de ``compute_numerical_sequence_metrics``
    (global_strict_score, global_value_score, n_total,
    per_category). Présent uniquement si la GT contient au
    moins une séquence détectée.
    """

    def as_dict(self) -> dict:
        d = {
            "doc_id": self.doc_id,
            "image_path": self.image_path,
            "ground_truth": self.ground_truth,
            "hypothesis": self.hypothesis,
            "metrics": self.metrics.as_dict(),
            "duration_seconds": self.duration_seconds,
            "engine_error": self.engine_error,
        }
        if self.ocr_intermediate is not None:
            d["ocr_intermediate"] = self.ocr_intermediate
        if self.pipeline_metadata:
            d["pipeline_metadata"] = self.pipeline_metadata
        if self.confusion_matrix is not None:
            d["confusion_matrix"] = self.confusion_matrix
        if self.char_scores is not None:
            d["char_scores"] = self.char_scores
        if self.taxonomy is not None:
            d["taxonomy"] = self.taxonomy
        if self.structure is not None:
            d["structure"] = self.structure
        if self.image_quality is not None:
            d["image_quality"] = self.image_quality
        if self.line_metrics is not None:
            d["line_metrics"] = self.line_metrics
        if self.hallucination_metrics is not None:
            d["hallucination_metrics"] = self.hallucination_metrics
        if self.ner_metrics is not None:
            d["ner_metrics"] = self.ner_metrics
        if self.calibration_metrics is not None:
            d["calibration_metrics"] = self.calibration_metrics
        if self.philological_metrics is not None:
            d["philological_metrics"] = self.philological_metrics
        if self.searchability_metrics is not None:
            d["searchability_metrics"] = self.searchability_metrics
        if self.numerical_sequence_metrics is not None:
            d["numerical_sequence_metrics"] = self.numerical_sequence_metrics
        return d

    def compact(self) -> None:
        """Libère les champs lourds pour réduire l'empreinte mémoire.

        Appelé après que les données ont été sérialisées dans le fichier
        partiel et que les agrégations ont été calculées.  Les champs
        ``ground_truth`` et ``hypothesis`` sont tronqués et les analyses
        détaillées (confusion, taxonomy…) sont supprimées.
        """
        # Garder un extrait pour le rapport, libérer le texte complet
        if len(self.ground_truth) > 200:
            self.ground_truth = self.ground_truth[:200] + "…"
        if len(self.hypothesis) > 200:
            self.hypothesis = self.hypothesis[:200] + "…"
        if self.ocr_intermediate and len(self.ocr_intermediate) > 200:
            self.ocr_intermediate = self.ocr_intermediate[:200] + "…"
        # Les analyses per-document ne sont plus nécessaires après agrégation
        self.confusion_matrix = None
        self.char_scores = None
        self.taxonomy = None
        self.structure = None
        self.image_quality = None
        self.line_metrics = None
        self.hallucination_metrics = None
        self.ner_metrics = None
        self.calibration_metrics = None
        self.philological_metrics = None
        self.searchability_metrics = None
        self.numerical_sequence_metrics = None


@dataclass
class EngineReport:
    """Rapport complet d'un moteur (ou pipeline) sur l'ensemble du corpus."""

    engine_name: str
    engine_version: str
    engine_config: dict
    document_results: list[DocumentResult]
    aggregated_metrics: dict = field(default_factory=dict)
    pipeline_info: dict = field(default_factory=dict)
    """Métadonnées du pipeline OCR+LLM (vide pour les moteurs OCR seuls).
    Clés typiques : mode, prompt_file, llm_model, llm_provider, pipeline_steps,
    over_normalization (score agrégé, classe 10 de la taxonomie).
    """
    # Métriques agrégées Sprint 5
    aggregated_confusion: Optional[dict] = None
    """Matrice de confusion unicode agrégée sur le corpus."""
    aggregated_char_scores: Optional[dict] = None
    """Scores ligatures/diacritiques agrégés."""
    aggregated_taxonomy: Optional[dict] = None
    """Distribution taxonomique des erreurs agrégée."""
    aggregated_structure: Optional[dict] = None
    """Métriques structurelles agrégées."""
    aggregated_image_quality: Optional[dict] = None
    """Métriques de qualité image agrégées."""
    # Sprint 10
    aggregated_line_metrics: Optional[dict] = None
    """Distribution CER par ligne agrégée (Gini moyen, percentiles, heatmap, taux catastrophiques)."""
    aggregated_hallucination: Optional[dict] = None
    """Métriques d'hallucination VLM agrégées (ancrage moyen, taux de docs hallucinés…)."""
    # Sprint 40
    aggregated_ner: Optional[dict] = None
    """Métriques NER agrégées sur le corpus : F1 micro/macro globaux et
    par catégorie, total hallucinations/missed.  ``None`` si aucun
    document n'a porté de calcul NER."""
    # Sprint 42
    aggregated_calibration: Optional[dict] = None
    """Calibration agrégée sur le corpus : ECE, MCE, reliability diagram
    micro recalculé à partir des sommes par bin.  ``None`` si aucun
    document n'avait de ``calibration_metrics`` (cas par défaut tant que
    les engines n'exposent pas ``token_confidences``)."""
    # Sprint 61
    aggregated_philological: Optional[dict] = None
    """Métriques philologiques agrégées sur le corpus (Sprints 55-60).

    Dict avec une clé par module ayant du signal sur au moins un
    document.  Pour chaque module, l'agrégation somme les compteurs
    bruts (n_total, n_preserved, etc.) et recalcule les scores
    globaux ; les structures per_category/per_block/per_status sont
    également agrégées.  ``None`` si aucun document n'a porté de
    ``philological_metrics``."""
    # Sprint 86
    aggregated_searchability: Optional[dict] = None
    """Recherchabilité fuzzy agrégée corpus-wide (Sprint 84+86).

    Format ``{n_docs, n_gt_tokens, n_searchable, recall,
    missed_tokens_sample, max_distance}``. ``None`` si aucun
    document n'a porté de ``searchability_metrics``."""
    aggregated_numerical_sequences: Optional[dict] = None
    """Précision sur séquences numériques agrégée (Sprint 85+86).

    Format identique à ``compute_numerical_sequence_metrics`` :
    global_strict_score, global_value_score, n_total,
    per_category{n_total, strict, value, strict_score,
    value_score, lost_items}. ``None`` si aucun document n'avait
    de séquence numérique exploitable."""

    def __post_init__(self) -> None:
        if not self.aggregated_metrics and self.document_results:
            self.aggregated_metrics = aggregate_metrics(
                [dr.metrics for dr in self.document_results]
            )

    @property
    def mean_cer(self) -> Optional[float]:
        cer_stats = self.aggregated_metrics.get("cer", {})
        return cer_stats.get("mean")

    @property
    def median_cer(self) -> Optional[float]:
        """CER médian sur le corpus.

        Sprint 44 — devient le critère de tri par défaut du ``ranking()``
        car la moyenne est facilement tirée par quelques documents
        catastrophiques sur une distribution asymétrique (typique des
        corpus patrimoniaux).
        """
        cer_stats = self.aggregated_metrics.get("cer", {})
        return cer_stats.get("median")

    @property
    def mean_wer(self) -> Optional[float]:
        wer_stats = self.aggregated_metrics.get("wer", {})
        return wer_stats.get("mean")

    @property
    def ligature_score(self) -> Optional[float]:
        """Score de ligatures agrégé (None si non calculé)."""
        if self.aggregated_char_scores:
            return self.aggregated_char_scores.get("ligature", {}).get("score")
        return None

    @property
    def diacritic_score(self) -> Optional[float]:
        """Score diacritique agrégé (None si non calculé)."""
        if self.aggregated_char_scores:
            return self.aggregated_char_scores.get("diacritic", {}).get("score")
        return None

    @property
    def is_pipeline(self) -> bool:
        """Vrai si ce rapport correspond à un pipeline OCR+LLM."""
        return bool(self.pipeline_info)

    def as_dict(self) -> dict:
        d = {
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "engine_config": self.engine_config,
            "aggregated_metrics": self.aggregated_metrics,
            "document_results": [dr.as_dict() for dr in self.document_results],
        }
        if self.pipeline_info:
            d["pipeline_info"] = self.pipeline_info
        if self.aggregated_confusion is not None:
            d["aggregated_confusion"] = self.aggregated_confusion
        if self.aggregated_char_scores is not None:
            d["aggregated_char_scores"] = self.aggregated_char_scores
        if self.aggregated_taxonomy is not None:
            d["aggregated_taxonomy"] = self.aggregated_taxonomy
        if self.aggregated_structure is not None:
            d["aggregated_structure"] = self.aggregated_structure
        if self.aggregated_image_quality is not None:
            d["aggregated_image_quality"] = self.aggregated_image_quality
        if self.aggregated_line_metrics is not None:
            d["aggregated_line_metrics"] = self.aggregated_line_metrics
        if self.aggregated_hallucination is not None:
            d["aggregated_hallucination"] = self.aggregated_hallucination
        if self.aggregated_ner is not None:
            d["aggregated_ner"] = self.aggregated_ner
        if self.aggregated_calibration is not None:
            d["aggregated_calibration"] = self.aggregated_calibration
        if self.aggregated_philological is not None:
            d["aggregated_philological"] = self.aggregated_philological
        if self.aggregated_searchability is not None:
            d["aggregated_searchability"] = self.aggregated_searchability
        if self.aggregated_numerical_sequences is not None:
            d["aggregated_numerical_sequences"] = (
                self.aggregated_numerical_sequences
            )
        return d


@dataclass
class BenchmarkResult:
    """Résultat complet d'un benchmark multi-moteurs sur un corpus."""

    corpus_name: str
    corpus_source: Optional[str]
    document_count: int
    engine_reports: list[EngineReport]
    run_date: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    picarones_version: str = __version__
    metadata: dict = field(default_factory=dict)
    # Sprint 36 — analyse inter-moteurs (divergence taxonomique +
    # complémentarité / oracle).  Calculée par le runner avant compact()
    # afin d'avoir accès aux hypothèses brutes.  ``None`` si moins de
    # 2 moteurs ou si le calcul a été désactivé.
    inter_engine_analysis: Optional[dict] = None
    # Sprint 45 — A.III stratification : map ``{doc_id: script_type}``
    # capturée avant ``compact()`` (qui efface ``image_quality``).
    # ``None`` si aucun document n'expose de ``script_type`` dans son
    # ``image_quality.script_type`` ou ``metadata.script_type``.
    doc_strata: Optional[dict[str, str]] = None

    def ranking(self) -> list[dict]:
        """Retourne le classement des moteurs trié par **médiane CER** croissante.

        Sprint 44 — A.I.2 du plan d'évolution : le tri par défaut bascule
        de la moyenne vers la médiane.  Sur des distributions
        asymétriques (typique des corpus patrimoniaux : 80 % des docs
        à 3 % de CER, 20 % à 40 %), la moyenne est tirée par quelques
        documents catastrophiques et masque les performances réelles.
        La médiane est plus représentative ; cohérente aussi avec le
        test de Friedman qui travaille déjà sur les rangs (Sprint 18).

        Le champ ``mean_cer`` est conservé dans chaque entrée pour
        rétrocompatibilité — les consommateurs (CLI, détecteurs
        narratifs, vue HTML) continuent à pouvoir l'afficher en colonne
        secondaire.  Le tri prend ``median_cer`` quand disponible et
        retombe sur ``mean_cer`` sinon.
        """
        ranked = []
        for report in self.engine_reports:
            ranked.append(
                {
                    "engine": report.engine_name,
                    "mean_cer": report.mean_cer,
                    "median_cer": report.median_cer,
                    "mean_wer": report.mean_wer,
                    "documents": len(report.document_results),
                    "failed": report.aggregated_metrics.get("failed_count", 0),
                }
            )

        def _sort_key(entry: dict) -> tuple:
            # Priorité : médiane si disponible, sinon moyenne, sinon +∞
            primary = entry.get("median_cer")
            if primary is None:
                primary = entry.get("mean_cer")
            return (primary is None, primary if primary is not None else float("inf"))

        return sorted(ranked, key=_sort_key)

    # ──────────────────────────────────────────────────────────────────
    # Sprint 45 — A.III stratification par script_type
    # ──────────────────────────────────────────────────────────────────

    def available_strata(self) -> list[str]:
        """Liste triée des strates ``script_type`` distinctes du corpus.

        Vide si ``doc_strata`` est ``None`` ou si aucun document n'a de
        valeur non vide.  Garantit un ordre stable (tri lexical).
        """
        if not self.doc_strata:
            return []
        return sorted({s for s in self.doc_strata.values() if s})

    def _doc_ids_in_stratum(self, stratum: str) -> set[str]:
        """Ensemble des ``doc_id`` dont la strate est ``stratum``."""
        if not self.doc_strata:
            return set()
        return {
            doc_id for doc_id, st in self.doc_strata.items()
            if st == stratum
        }

    def stratified_ranking(self) -> dict[str, list[dict]]:
        """Retourne un classement séparé par strate ``script_type``.

        Pour chaque strate, recalcule mean/median CER **uniquement sur
        les documents de la strate** et trie par médiane (cohérent avec
        ``ranking()`` Sprint 44).

        Returns
        -------
        dict[str, list[dict]]
            ``{stratum_name: [ranking_entry, ...]}``.  Vide si pas de
            stratification disponible (``doc_strata`` non renseigné).
            Chaque ``ranking_entry`` a la même structure que
            ``ranking()`` : ``engine``, ``mean_cer``, ``median_cer``,
            ``mean_wer``, ``documents``, ``failed``.
        """
        strata = self.available_strata()
        if not strata:
            return {}

        import statistics as _stats

        result: dict[str, list[dict]] = {}
        for stratum in strata:
            doc_ids = self._doc_ids_in_stratum(stratum)
            if not doc_ids:
                continue

            entries: list[dict] = []
            for report in self.engine_reports:
                cers = [
                    dr.metrics.cer
                    for dr in report.document_results
                    if dr.doc_id in doc_ids
                    and dr.metrics is not None
                    and dr.metrics.error is None
                ]
                wers = [
                    dr.metrics.wer
                    for dr in report.document_results
                    if dr.doc_id in doc_ids
                    and dr.metrics is not None
                    and dr.metrics.error is None
                ]
                failed = sum(
                    1 for dr in report.document_results
                    if dr.doc_id in doc_ids
                    and dr.metrics is not None
                    and dr.metrics.error is not None
                )
                if not cers:
                    entries.append({
                        "engine": report.engine_name,
                        "mean_cer": None,
                        "median_cer": None,
                        "mean_wer": None,
                        "documents": 0,
                        "failed": failed,
                    })
                    continue
                entries.append({
                    "engine": report.engine_name,
                    "mean_cer": _stats.mean(cers),
                    "median_cer": _stats.median(cers),
                    "mean_wer": _stats.mean(wers) if wers else None,
                    "documents": len(cers),
                    "failed": failed,
                })

            def _sort_key(entry: dict) -> tuple:
                primary = entry.get("median_cer")
                if primary is None:
                    primary = entry.get("mean_cer")
                return (primary is None, primary if primary is not None else float("inf"))

            result[stratum] = sorted(entries, key=_sort_key)
        return result

    def corpus_homogeneity(self) -> Optional[dict]:
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
        strata_rankings = self.stratified_ranking()
        if len(strata_rankings) < 2:
            return None

        global_ranking = self.ranking()
        valid = [
            r for r in global_ranking
            if r.get("median_cer") is not None
        ]
        if not valid:
            return None
        leader = valid[0]["engine"]

        # CER médian du leader sur chaque strate (où il a au moins 1 doc)
        per_stratum: dict[str, float] = {}
        for stratum, entries in strata_rankings.items():
            for entry in entries:
                if entry["engine"] != leader:
                    continue
                med = entry.get("median_cer")
                if med is None:
                    continue
                per_stratum[stratum] = float(med)
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

    def as_dict(self) -> dict:
        d = {
            "picarones_version": self.picarones_version,
            "run_date": self.run_date,
            "corpus": {
                "name": self.corpus_name,
                "source": self.corpus_source,
                "document_count": self.document_count,
            },
            "ranking": self.ranking(),
            "engine_reports": [r.as_dict() for r in self.engine_reports],
            "metadata": self.metadata,
        }
        if self.inter_engine_analysis is not None:
            d["inter_engine_analysis"] = self.inter_engine_analysis
        if self.doc_strata:
            d["doc_strata"] = self.doc_strata
            d["available_strata"] = self.available_strata()
            stratified = self.stratified_ranking()
            if stratified:
                d["stratified_ranking"] = stratified
            homogeneity = self.corpus_homogeneity()
            if homogeneity:
                d["corpus_homogeneity"] = homogeneity
        return d

    def to_json(self, path: str | Path, indent: int = 2) -> Path:
        """Sérialise le benchmark en JSON et l'écrit sur disque.

        Parameters
        ----------
        path:
            Chemin du fichier JSON de sortie.
        indent:
            Indentation JSON (défaut : 2 espaces).

        Returns
        -------
        Path
            Chemin absolu du fichier écrit.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, ensure_ascii=False, indent=indent)
        return output_path.resolve()

    @classmethod
    def from_json(cls, path: str | Path) -> dict:
        """Charge un résultat JSON brut depuis le disque (pour le rapport HTML).

        Retourne le dict Python — la reconstruction complète en objets
        est réservée aux sprints suivants.
        """
        with Path(path).open(encoding="utf-8") as fh:
            return json.load(fh)
