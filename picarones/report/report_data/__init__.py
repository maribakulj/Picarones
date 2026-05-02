"""Construction du dict de données consommé par le template Jinja.

Avant le découpage, ``picarones.report.generator._build_report_data``
faisait 463 lignes pour transformer un :class:`BenchmarkResult` en
dict prêt pour Jinja. Cette fonction empilait par sprint des blocs
indépendants — engines, documents, statistiques, scatter plots,
front Pareto, etc.

Ce sous-package éclate la construction en modules thématiques :

- :mod:`engines` — résumé par moteur (``engines_summary``).
- :mod:`documents` — vue galerie + détail + difficulté Sprint 7.
- :mod:`statistics` — Wilcoxon, Friedman, Nemenyi, bootstrap CIs,
  reliability curves, Venn, error clusters, corrélations.
- :mod:`scatter` — Sprint 10 : Gini vs CER, ratio vs anchor.
- :mod:`pareto` — Sprint 19 : 3 fronts Pareto + métadonnées pricing.

L'API publique :func:`build_report_data` orchestre ces modules dans
le bon ordre (les coûts du module Pareto enrichissent en place le
``engines_summary`` produit par :mod:`engines`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from picarones.core.results import BenchmarkResult

from picarones.report.report_data.documents import (
    annotate_documents_with_difficulty,
    build_documents,
)
from picarones.report.report_data.engines import build_engines_summary
from picarones.report.report_data.pareto import build_pareto_section
from picarones.report.report_data.scatter import (
    build_gini_vs_cer,
    build_ratio_vs_anchor,
)
from picarones.report.report_data.statistics import (
    build_bootstrap_cis,
    build_correlation_per_engine,
    build_error_clusters,
    build_friedman_and_nemenyi,
    build_pairwise_wilcoxon,
    build_reliability_curves,
    build_venn_data,
)


def build_report_data(
    benchmark: "BenchmarkResult", images_b64: dict[str, str],
) -> dict:
    """Transforme un :class:`BenchmarkResult` en dict pour le rapport HTML.

    L'ordre est important : :mod:`pareto` lit et enrichit en place
    le ``engines_summary`` produit par :mod:`engines`.
    """
    engines_summary = build_engines_summary(benchmark)
    documents = build_documents(benchmark, images_b64)
    annotate_documents_with_difficulty(benchmark, documents)

    pareto_data = build_pareto_section(engines_summary, benchmark)

    return {
        "meta": {
            "corpus_name": benchmark.corpus_name,
            "corpus_source": benchmark.corpus_source,
            "document_count": benchmark.document_count,
            "run_date": benchmark.run_date,
            "picarones_version": benchmark.picarones_version,
            "metadata": benchmark.metadata,
        },
        "ranking": benchmark.ranking(),
        "engines": engines_summary,
        "documents": documents,
        # Sprint 7
        "statistics": {
            "pairwise_wilcoxon": build_pairwise_wilcoxon(benchmark),
            "bootstrap_cis": build_bootstrap_cis(benchmark),
            **build_friedman_and_nemenyi(benchmark),
        },
        "reliability_curves": build_reliability_curves(benchmark),
        "venn_data": build_venn_data(benchmark),
        "error_clusters": build_error_clusters(benchmark),
        "correlation_per_engine": build_correlation_per_engine(benchmark),
        # Sprint 10
        "gini_vs_cer": build_gini_vs_cer(benchmark),
        "ratio_vs_anchor": build_ratio_vs_anchor(benchmark),
        # Sprint 19 — vue Pareto coût/qualité avec variantes d'axe
        "pareto": pareto_data,
        # Sprint 36 — analyse inter-moteurs (divergence taxonomique +
        # complémentarité / oracle).  ``None`` si moins de 2 moteurs.
        "inter_engine_analysis": benchmark.inter_engine_analysis,
        # Sprint 45-46 — stratification par script_type
        "available_strata": benchmark.available_strata(),
        "stratified_ranking": benchmark.stratified_ranking() or None,
        "corpus_homogeneity": benchmark.corpus_homogeneity(),
    }


__all__ = ["build_report_data"]
