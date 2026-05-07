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
  Expose deux fonctions séparées : :func:`attach_engine_costs`
  (mute) et :func:`build_pareto_section` (pure).

L'API publique :func:`build_report_data` orchestre ces modules dans
le bon ordre. La séquence Pareto en deux temps
(``attach_engine_costs`` → ``build_pareto_section``) rend la
mutation explicite — les fonctions ``build_*`` du sous-package
sont pures sauf ``attach_engine_costs`` dont le nom le dit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from picarones.evaluation.benchmark_result import BenchmarkResult

from picarones.reports_v2.html.data.documents import (
    annotate_documents_with_difficulty,
    build_documents,
)
from picarones.reports_v2.html.data.engines import build_engines_summary
from picarones.reports_v2.html.data.extra_metrics import (
    compute_marginal_cost_section,
    compute_rare_token_recall_per_engine,
    compute_taxonomy_cooccurrence_section,
    compute_taxonomy_intra_doc_section,
)
from picarones.reports_v2.html.data.pareto import (
    attach_engine_costs,
    build_pareto_section,
)
from picarones.reports_v2.html.data.scatter import (
    build_gini_vs_cer,
    build_ratio_vs_anchor,
)
from picarones.reports_v2.html.data.statistics import (
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

    Ordre critique :

    1. Construire ``engines_summary`` (pur).
    2. Construire ``documents`` puis annoter avec la difficulté (mute
       ``documents``).
    3. **Attacher** les coûts à ``engines_summary`` (mute, nom
       explicite).
    4. **Construire** le bloc Pareto (pure, lit les coûts attachés).
    """
    engines_summary = build_engines_summary(benchmark)
    documents = build_documents(benchmark, images_b64)
    annotate_documents_with_difficulty(benchmark, documents)

    attach_engine_costs(engines_summary, benchmark)
    pareto_data = build_pareto_section(engines_summary)

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
        # Sprint « câblage des modules test-only » (mai 2026) — métriques
        # corpus-wide qui jusque-là n'étaient pas remontées dans le rapport.
        # Sprint 71 (A.I.1) : recall sur tokens rares (hapax + dis legomena).
        "rare_token_recall": compute_rare_token_recall_per_engine(benchmark),
        # Sprint 75 (A.I.4) : co-occurrence taxonomique inter-classes.
        "taxonomy_cooccurrence": compute_taxonomy_cooccurrence_section(benchmark),
        # Sprint 76 (A.I.4) : heatmap class × position (intra-document).
        "taxonomy_intra_doc": compute_taxonomy_intra_doc_section(benchmark),
        # Sprint 91 (A.II.6) : matrice de coût marginal entre paires de moteurs.
        "marginal_cost": compute_marginal_cost_section(engines_summary),
    }


__all__ = ["build_report_data"]
