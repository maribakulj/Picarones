"""Modèle de données des résultats et export JSON (Cercle 2).

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

from picarones.evaluation.metric_result import MetricsResult, aggregate_metrics

# Calculs de classement/stratification extraits (audit prod P1 —
# dégonflage god-module).  Réimportés sous alias : les méthodes
# ``ranking()`` / ``stratified_ranking()`` / … de ``BenchmarkResult``
# délèguent ici, l'API publique reste identique.
from picarones.evaluation.benchmark_result_ranking import (
    available_strata as _available_strata,
    compute_corpus_homogeneity as _compute_corpus_homogeneity,
    compute_ranking as _compute_ranking,
    compute_stratified_ranking as _compute_stratified_ranking,
    doc_ids_in_stratum as _doc_ids_in_stratum,
)


def _resolve_picarones_version() -> str:
    """Récupère la version courante de Picarones sans dépendance vers
    le package racine.

    Raison : la couche ``evaluation`` ne peut pas importer
    ``picarones`` (le package racine, qui importe ``measurements``
    et déclencherait un cycle).  On lit la version via
    ``importlib.metadata`` (chemin de production : wheel installé)
    avec un fallback ``"1.0.0"`` cohérent avec
    ``picarones/__init__.py``.
    """
    try:
        from importlib.metadata import version as _get_version
        return _get_version("picarones")
    except Exception:  # noqa: BLE001
        return "1.0.0"


__version__ = _resolve_picarones_version()


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
    """Précision/rappel/F1 sur entités nommées

    Format : retour de ``compute_ner_metrics`` (global, per_category,
    hallucinated_entities, missed_entities, etc.).  Présent uniquement si
    le document a un niveau de GT ``ENTITIES`` ET que le runner a reçu
    un ``EntityExtractor``.
    """
    # calibration des confidences moteur (ECE, MCE, bins)
    calibration_metrics: Optional[dict] = None
    """Métriques de calibration (Sprint 39+42).

    Format : retour de ``compute_calibration_metrics`` (ece, mce,
    n_bins, n_predictions, overall_accuracy, overall_confidence, bins).
    Présent uniquement si le moteur a fourni des ``token_confidences``
    sur l'``EngineResult``.
    """
    # métriques philologiques (Sprints 55-60) calculées
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
    # Sprint 87 — delta Flesch (Sprint 52) calculé automatiquement
    # avec adaptive masking (≥ 5 mots dans la GT).
    readability_metrics: Optional[dict] = None
    """Métriques de lisibilité (Sprint 52+87).

    Format ``{lang, flesch_reference, flesch_hypothesis,
    flesch_delta, n_words_reference}``.  Présent uniquement si
    la GT contient au moins 5 mots."""
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
        if self.readability_metrics is not None:
            d["readability_metrics"] = self.readability_metrics
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentResult":
        """Reconstruit un :class:`DocumentResult` depuis ``as_dict()``.

        Phase 2.2 du chantier post-rewrite : restauration fidèle de
        tous les champs avancés (confusion_matrix, taxonomy, structure,
        hallucination_metrics, ner_metrics, calibration_metrics,
        philological_metrics, searchability_metrics,
        numerical_sequence_metrics, readability_metrics,
        pipeline_metadata, ocr_intermediate).

        Avant ce durcissement, ``ReportGenerator.from_json`` faisait sa
        propre reconstruction qui ne couvrait que CER/WER/MER/WIL +
        doc_id/image_path/ground_truth/hypothesis — toutes les
        analyses détaillées étaient perdues, donc le rapport régénéré
        depuis JSON n'avait plus accès aux vues taxonomy, NER,
        calibration, etc.  La reproductibilité scientifique était
        cassée.
        """
        return cls(
            doc_id=data["doc_id"],
            image_path=data["image_path"],
            ground_truth=data["ground_truth"],
            hypothesis=data["hypothesis"],
            metrics=MetricsResult.from_dict(data["metrics"]),
            duration_seconds=data.get("duration_seconds", 0.0),
            engine_error=data.get("engine_error"),
            ocr_intermediate=data.get("ocr_intermediate"),
            pipeline_metadata=data.get("pipeline_metadata", {}) or {},
            confusion_matrix=data.get("confusion_matrix"),
            char_scores=data.get("char_scores"),
            taxonomy=data.get("taxonomy"),
            structure=data.get("structure"),
            image_quality=data.get("image_quality"),
            line_metrics=data.get("line_metrics"),
            hallucination_metrics=data.get("hallucination_metrics"),
            ner_metrics=data.get("ner_metrics"),
            calibration_metrics=data.get("calibration_metrics"),
            philological_metrics=data.get("philological_metrics"),
            searchability_metrics=data.get("searchability_metrics"),
            numerical_sequence_metrics=data.get("numerical_sequence_metrics"),
            readability_metrics=data.get("readability_metrics"),
        )

    def compact(
        self,
        text_limit: Optional[int] = None,
        drop_analyses: bool = False,
    ) -> None:
        """Libère les champs lourds pour réduire l'empreinte mémoire.

        A.I.0 P0 : compaction désormais opt-in.
        Auparavant, le runner appelait ``compact()`` sans paramètres
        avant de sérialiser le JSON, ce qui amputait silencieusement
        toutes les analyses per-document (confusion, taxonomy,
        philological, searchability, etc.) et tronquait
        ``ground_truth``/``hypothesis``/``ocr_intermediate`` à 200
        caractères.  Le rapport HTML — qui consomme ce JSON — recevait
        des données déjà mutilées, contredisant directement la
        promesse "self-contained HTML report" du README.

        Désormais, l'appel par défaut ``compact()`` est un **no-op**.
        Le caller doit explicitement demander la troncature et/ou la
        suppression des analyses :

        - ``compact(text_limit=200)`` : tronque les textes à 200 chars.
        - ``compact(drop_analyses=True)`` : supprime les dicts d'analyse.
        - ``compact(text_limit=200, drop_analyses=True)`` : ancien
          comportement, à utiliser en pipeline web pour un rendu
          interactif léger uniquement.

        Le runner (``runner/orchestration.py``) ne compacte plus par
        défaut ; le JSON exporté contient désormais toutes les
        analyses détaillées.

        Parameters
        ----------
        text_limit:
            Si fourni (int > 0), tronque ``ground_truth``,
            ``hypothesis`` et ``ocr_intermediate`` à cette longueur en
            ajoutant "…".  ``None`` (défaut) = pas de troncature.
        drop_analyses:
            Si ``True``, met à ``None`` toutes les analyses
            per-document (confusion, taxonomy, philological…).  Défaut :
            ``False`` = on conserve toutes les analyses.
        """
        if text_limit is not None and text_limit > 0:
            if len(self.ground_truth) > text_limit:
                self.ground_truth = self.ground_truth[:text_limit] + "…"
            if len(self.hypothesis) > text_limit:
                self.hypothesis = self.hypothesis[:text_limit] + "…"
            if self.ocr_intermediate and len(self.ocr_intermediate) > text_limit:
                self.ocr_intermediate = self.ocr_intermediate[:text_limit] + "…"

        if drop_analyses:
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
            self.readability_metrics = None


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
    aggregated_line_metrics: Optional[dict] = None
    """Distribution CER par ligne agrégée (Gini moyen, percentiles, heatmap, taux catastrophiques)."""
    aggregated_hallucination: Optional[dict] = None
    """Métriques d'hallucination VLM agrégées (ancrage moyen, taux de docs hallucinés…)."""
    aggregated_ner: Optional[dict] = None
    """Métriques NER agrégées sur le corpus : F1 micro/macro globaux et
    par catégorie, total hallucinations/missed.  ``None`` si aucun
    document n'a porté de calcul NER."""
    aggregated_calibration: Optional[dict] = None
    """Calibration agrégée sur le corpus : ECE, MCE, reliability diagram
    micro recalculé à partir des sommes par bin.  ``None`` si aucun
    document n'avait de ``calibration_metrics`` (cas par défaut tant que
    les engines n'exposent pas ``token_confidences``)."""
    aggregated_philological: Optional[dict] = None
    """Métriques philologiques agrégées sur le corpus (Sprints 55-60).

    Dict avec une clé par module ayant du signal sur au moins un
    document.  Pour chaque module, l'agrégation somme les compteurs
    bruts (n_total, n_preserved, etc.) et recalcule les scores
    globaux ; les structures per_category/per_block/per_status sont
    également agrégées.  ``None`` si aucun document n'a porté de
    ``philological_metrics``."""
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
    # A.II.2 (delta Flesch agrégé)
    aggregated_readability: Optional[dict] = None
    """Delta Flesch agrégé corpus-wide (Sprint 52+87).

    Format ``{lang, n_docs, n_docs_with_delta, delta_mean,
    delta_median, delta_min, delta_max, n_over_normalized,
    n_under_normalized, over_normalized_rate}``.  ``None`` si
    aucun document n'avait de ``readability_metrics``."""
    # Phase 3.4 audit code-quality (2026-05) — câblage de
    # ``aggregate_over_normalization`` (classe 10 de la taxonomie).
    aggregated_over_normalization: Optional[dict] = None
    """Sur-normalisation LLM agrégée corpus-wide.

    Format ``{score, total_correct_ocr_words, over_normalized_count,
    document_count}`` produit par
    :func:`picarones.evaluation.metrics.over_normalization.aggregate_over_normalization`.
    ``None`` si aucun document n'a porté de
    ``pipeline_metadata["over_normalization"]`` (cas d'un benchmark
    OCR seul, sans étape LLM)."""

    def __post_init__(self) -> None:
        if not self.aggregated_metrics and self.document_results:
            self.aggregated_metrics = aggregate_metrics(
                [dr.metrics for dr in self.document_results]
            )

    @property
    def micro_cer(self) -> Optional[float]:
        """CER **micro-moyenné** corpus = Σ distance_édition / Σ car_référence.

        Audit scientifique F1 — métrique d'agrégation standard du domaine
        OCR/HTR (ICDAR, OCR-D, HTR-United, Transkribus, eScriptorium).
        Contrairement à ``mean_cer`` / ``median_cer`` (macro, aveugles à
        la longueur), elle pondère chaque document par son nombre de
        caractères : une page de 5 000 caractères pèse 500× une légende
        de 10.  C'est le critère de tri par défaut de ``ranking()``.
        ``None`` si aucun document n'a de comptes bruts (jiwer absent,
        références vides).
        """
        return self.aggregated_metrics.get("cer_micro", {}).get("value")

    @property
    def micro_wer(self) -> Optional[float]:
        """WER micro-moyenné corpus = Σ erreurs_mot / Σ mots_référence."""
        return self.aggregated_metrics.get("wer_micro", {}).get("value")

    @property
    def mean_cer(self) -> Optional[float]:
        cer_stats = self.aggregated_metrics.get("cer", {})
        return cer_stats.get("mean")

    @property
    def median_cer(self) -> Optional[float]:
        """CER médian sur le corpus.

        devient le critère de tri par défaut du ``ranking()``
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
        if self.aggregated_readability is not None:
            d["aggregated_readability"] = self.aggregated_readability
        if self.aggregated_over_normalization is not None:
            d["aggregated_over_normalization"] = self.aggregated_over_normalization
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "EngineReport":
        """Reconstruit un :class:`EngineReport` depuis ``as_dict()``.

        Phase 2.2 du chantier post-rewrite : restauration fidèle des
        ``aggregated_*`` (confusion, char_scores, taxonomy, structure,
        image_quality, line_metrics, hallucination, ner, calibration,
        philological, searchability, numerical_sequences, readability)
        et de ``pipeline_info``.
        """
        return cls(
            engine_name=data["engine_name"],
            engine_version=data.get("engine_version", "unknown"),
            engine_config=data.get("engine_config", {}),
            document_results=[
                DocumentResult.from_dict(dr)
                for dr in data.get("document_results", [])
            ],
            aggregated_metrics=data.get("aggregated_metrics", {}) or {},
            pipeline_info=data.get("pipeline_info", {}) or {},
            aggregated_confusion=data.get("aggregated_confusion"),
            aggregated_char_scores=data.get("aggregated_char_scores"),
            aggregated_taxonomy=data.get("aggregated_taxonomy"),
            aggregated_structure=data.get("aggregated_structure"),
            aggregated_image_quality=data.get("aggregated_image_quality"),
            aggregated_line_metrics=data.get("aggregated_line_metrics"),
            aggregated_hallucination=data.get("aggregated_hallucination"),
            aggregated_ner=data.get("aggregated_ner"),
            aggregated_calibration=data.get("aggregated_calibration"),
            aggregated_philological=data.get("aggregated_philological"),
            aggregated_searchability=data.get("aggregated_searchability"),
            aggregated_numerical_sequences=data.get(
                "aggregated_numerical_sequences",
            ),
            aggregated_readability=data.get("aggregated_readability"),
            aggregated_over_normalization=data.get(
                "aggregated_over_normalization",
            ),
        )


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
    # Audit scientifique F3 — intégrité des données.  ``True`` quand le
    # benchmark provient de ``generate_sample_benchmark`` (commande
    # ``picarones demo``) : résultats **fabriqués** par des fonctions de
    # transformation, pas de vrais moteurs OCR.  Sérialisé dans le JSON
    # et propagé au rapport HTML (bandeau d'avertissement inamovible) et
    # à l'export CSV, pour qu'un rapport de démonstration ne puisse
    # jamais être diffusé comme un résultat scientifique réel.
    is_demo: bool = False
    # analyse inter-moteurs (divergence taxonomique +
    # complémentarité / oracle).  Calculée par le runner avant compact()
    # afin d'avoir accès aux hypothèses brutes.  ``None`` si moins de
    # 2 moteurs ou si le calcul a été désactivé.
    inter_engine_analysis: Optional[dict] = None
    # A.III stratification : map ``{doc_id: script_type}``
    # capturée avant ``compact()`` (qui efface ``image_quality``).
    # ``None`` si aucun document n'expose de ``script_type`` dans son
    # ``image_quality.script_type`` ou ``metadata.script_type``.
    doc_strata: Optional[dict[str, str]] = None
    # Phase B6 (mai 2026) — résultats des EvaluationView du
    # RunOrchestrator (text_final, alto_documentary, searchability).
    # Structure : ``{view_name: {engine_name: {doc_id: {metric: value}}}}``.
    # Vide si le run a été lancé sans vues (cas legacy
    # ``run_benchmark_via_service`` sans RunOrchestrator).
    # Consommé par le rapport HTML (sections multi-vues) et par le
    # narrative engine pour mettre en avant les pipelines qui
    # produisent un ALTO valide vs ceux qui restent en RAW_TEXT seul.
    view_results: dict[str, dict[str, dict[str, dict[str, float]]]] = field(
        default_factory=dict,
    )

    def ranking(self) -> list[dict]:
        """Classement des moteurs trié par CER micro-moyenné croissant.

        Délègue à
        :func:`picarones.evaluation.benchmark_result_ranking.compute_ranking`
        (logique computationnelle extraite — audit prod P1).  Voir
        cette fonction pour le rationale scientifique du tri micro-CER.
        """
        return _compute_ranking(self.engine_reports)

    def available_strata(self) -> list[str]:
        """Liste triée des strates ``script_type`` distinctes du corpus."""
        return _available_strata(self.doc_strata)

    def _doc_ids_in_stratum(self, stratum: str) -> set[str]:
        """Ensemble des ``doc_id`` dont la strate est ``stratum``."""
        return _doc_ids_in_stratum(self.doc_strata, stratum)

    def stratified_ranking(self) -> dict[str, list[dict]]:
        """Classement séparé par strate ``script_type``.

        Délègue à
        :func:`picarones.evaluation.benchmark_result_ranking.compute_stratified_ranking`.
        """
        return _compute_stratified_ranking(
            self.engine_reports, self.doc_strata,
        )

    def corpus_homogeneity(self) -> Optional[dict]:
        """Mesure d'hétérogénéité du corpus (variance inter-strates).

        Délègue à
        :func:`picarones.evaluation.benchmark_result_ranking.compute_corpus_homogeneity`.
        """
        return _compute_corpus_homogeneity(
            self.engine_reports, self.doc_strata,
        )

    def as_dict(self) -> dict:
        d = {
            "picarones_version": self.picarones_version,
            "run_date": self.run_date,
            # F3 — drapeau d'intégrité au niveau racine ET corpus pour
            # qu'aucun consommateur (HTML, CSV, scripts tiers) ne puisse
            # l'ignorer par accident.
            "is_demo": self.is_demo,
            "corpus": {
                "name": self.corpus_name,
                "source": self.corpus_source,
                "document_count": self.document_count,
                "is_demo": self.is_demo,
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
    def from_dict(cls, data: dict) -> "BenchmarkResult":
        """Reconstruit un :class:`BenchmarkResult` complet depuis
        ``as_dict()``.

        Phase 2.2 du chantier post-rewrite : fidélité du round-trip
        ``to_json → from_dict``.  Auparavant, ``from_json`` retournait
        le dict brut et l'appelant devait reconstruire à la main —
        d'où la dérive entre ``ReportGenerator.__init__`` (objets) et
        ``ReportGenerator.from_json`` (dicts appauvris).  Désormais, un
        seul chemin canonique : ``BenchmarkResult.from_dict(dict)`` →
        objet complet, indistinguable d'un benchmark fraîchement
        exécuté.
        """
        corpus_info = data.get("corpus", {}) or {}
        return cls(
            corpus_name=corpus_info.get("name", "Corpus"),
            corpus_source=corpus_info.get("source"),
            document_count=corpus_info.get("document_count", 0),
            engine_reports=[
                EngineReport.from_dict(er)
                for er in data.get("engine_reports", [])
            ],
            run_date=data.get("run_date", ""),
            picarones_version=data.get("picarones_version", ""),
            metadata=data.get("metadata", {}) or {},
            # F3 — accepte le drapeau à la racine ou dans ``corpus``
            # (round-trip fidèle, y compris depuis un JSON de démo).
            is_demo=bool(
                data.get("is_demo", corpus_info.get("is_demo", False))
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> dict:
        """Charge le JSON brut (dict Python) — rétrocompatibilité.

        Pour reconstruire un :class:`BenchmarkResult` complet (objets),
        utiliser :meth:`from_dict` après :meth:`from_json`, ou
        directement :meth:`from_json_object` ci-dessous.

        Cette méthode est conservée parce que de nombreux consommateurs
        (tests, ``ReportGenerator.from_json`` legacy, scripts CLI ad
        hoc) attendent encore un dict.  Le rewrite v2.0 préfère les
        objets reconstruits ; les nouveaux callers doivent utiliser
        :meth:`from_json_object`.
        """
        with Path(path).open(encoding="utf-8") as fh:
            return json.load(fh)

    @classmethod
    def from_json_object(cls, path: str | Path) -> "BenchmarkResult":
        """Charge un JSON et reconstruit un :class:`BenchmarkResult`
        complet (objets), avec toutes les analyses avancées préservées.

        Round-trip garanti : ``BenchmarkResult.from_json_object(
        bm.to_json(p)) == bm`` au sens structurel (les champs
        ``aggregated_metrics`` peuvent être recalculés par
        ``__post_init__`` si absents, sinon préservés).
        """
        with Path(path).open(encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
