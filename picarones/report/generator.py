"""Générateur du rapport HTML interactif auto-contenu.

Le rapport produit est un fichier HTML unique embarquant :
- Toutes les données (JSON inline)
- Chart.js et diff2html (depuis cdnjs)
- CSS et JavaScript de l'application

Vues disponibles
----------------
1. Classement  — tableau triable par colonne (CER, WER, MER, WIL)
2. Galerie     — grille d'images avec badge CER coloré
3. Document    — image zoomable + diff coloré GT / OCR par moteur
4. Analyses    — histogramme CER + graphique radar

Architecture
------------
Ce module est l'**orchestrateur**. Les responsabilités lourdes sont
découpées en sous-modules :

- :mod:`picarones.report.assets` — chargement vendor.js, encodage
  base64 d'images, externalisation lazy.
- :mod:`picarones.report.report_data` — construction du dict JSON
  passé au template (engines, documents, statistiques, Pareto, etc.).
- :mod:`picarones.report.render_helpers` — couleurs / SVG mutualisés.

Rétrocompat
-----------
Deux noms historiques sont **encore importés par des tests** sous
leur préfixe ``_`` et doivent être préservés :

- ``_build_report_data`` (importé par 14 fichiers de tests).
- ``_cer_color`` (importé par ``tests/report/test_report.py``).

Les autres noms ``_pct``, ``_safe``, ``_cer_bg``, ``_encode_image_b64``,
``_encode_images_b64_from_result``, ``_externalize_images_to_dir``,
``_load_vendor_js`` sont soit utilisés en interne (les 3 derniers,
voir :meth:`ReportGenerator.generate`), soit accessibles via leur
nom canonique dans :mod:`picarones.report.assets` ou
:mod:`picarones.report.render_helpers`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from picarones.evaluation.benchmark_result import BenchmarkResult
from picarones.measurements.statistics import build_critical_difference_svg
from picarones.reports_v2._helpers.assets import (
    encode_images_b64_from_result as _encode_images_b64_from_result,
    externalize_images_to_dir as _externalize_images_to_dir,
    load_vendor_js as _load_vendor_js,
)

# Ré-exports rétrocompat consommés par les tests externes (cf. docstring
# de module). La directive de fin de ligne documente l'intention de
# ré-export et empêche ruff de marquer l'import comme inutilisé.
from picarones.reports_v2._helpers.render_helpers import cer_step_color as _cer_color  # noqa: F401
from picarones.report.report_data import build_report_data as _build_report_data  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rendu Jinja2
# ---------------------------------------------------------------------------

# Depuis le Sprint 16, le template monolithique ~3100 lignes a été découpé en
# fichiers externes dans ``picarones/report/templates/`` (CSS, JS, vues HTML).
# ``base.html.j2`` assemble le tout via ``{% include %}``.

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _build_jinja_env():
    """Construit l'Environment Jinja2 pour le rapport.

    Autoescape désactivé : le comportement est équivalent à celui du
    ``_HTML_TEMPLATE.format()`` historique. Les variables injectées
    (JSON embarqué, SVG généré, synthèse narrative issue de templates
    internes) sont toutes produites par le code Picarones et ne
    nécessitent pas d'échappement HTML.
    """
    from jinja2 import Environment, FileSystemLoader
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    return env


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Génère un rapport HTML interactif depuis un BenchmarkResult.

    Usage
    -----
    >>> from picarones.report import ReportGenerator
    >>> gen = ReportGenerator(benchmark_result)
    >>> path = gen.generate("rapport.html")
    >>> # Rapport en anglais :
    >>> gen_en = ReportGenerator(benchmark_result, lang="en")
    >>> path_en = gen_en.generate("report.html")
    """

    def __init__(
        self,
        benchmark: BenchmarkResult,
        images_b64: Optional[dict[str, str]] = None,
        lang: str = "fr",
        normalization_profile: Any = None,
        lazy_images: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        benchmark:
            Résultat de benchmark à visualiser.
        images_b64:
            Dictionnaire {doc_id: data-URI base64 OU url relative} des images.
            Si None, le générateur cherche dans ``benchmark.metadata["_images_b64"]``.
            Si ``lazy_images=True``, la valeur attendue est une URL relative
            comme ``"report-assets/<doc>.png"``.
        lang:
            Code langue du rapport : ``"fr"`` (défaut) ou ``"en"``.
        normalization_profile:
            Profil de normalisation effectivement utilisé (Sprint 27 — pour
            le snapshot de reproductibilité). ``None`` retombe sur le
            profil mentionné dans ``benchmark.metadata["normalization_profile"]``
            s'il est présent, sinon snapshot indisponible.
        lazy_images:
            Sprint A5 (M-16) — si ``True``, les images sont écrites en
            fichiers PNG/JPEG dans ``<output_dir>/report-assets/`` à côté
            du HTML, et référencées via ``<img loading="lazy">``.
            Le rapport reste auto-portant si on copie aussi le dossier
            d'assets. Utile pour les corpus > 50 documents (un rapport
            base64 monolithique de 1 000 docs dépasse 200 MB et fait
            ramer le navigateur). En mode mono-doc ou démo : laisser
            ``False`` pour un fichier HTML unique transportable.
        """
        self.benchmark = benchmark
        self.images_b64: dict[str, str] = images_b64 or {}
        self.lang = lang
        self.normalization_profile = normalization_profile
        self.lazy_images = lazy_images

        # Récupérer les images embarquées dans les metadata (fixtures)
        if not self.images_b64:
            self.images_b64 = benchmark.metadata.get("_images_b64", {})  # type: ignore[assignment]

        # Sprint 27 — fallback : profil de normalisation depuis les metadata
        if self.normalization_profile is None:
            self.normalization_profile = benchmark.metadata.get("normalization_profile")

    def generate(self, output_path: str | Path) -> Path:
        """Génère le fichier HTML et le sauvegarde sur disque.

        Parameters
        ----------
        output_path:
            Chemin du fichier HTML à écrire.

        Returns
        -------
        Path
            Chemin absolu du fichier généré.
        """
        from picarones.i18n import get_labels

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Sprint A5 (M-16) — externalisation des images si lazy_images=True
        # ou auto-encodage base64 sinon. Les deux modes alimentent la même
        # variable ``images_b64`` (le nom est conservé pour rétrocompat ;
        # en mode lazy la valeur est une URL relative au lieu d'un data-URI).
        # En mode lazy, on **force** l'externalisation même si self.images_b64
        # est pré-rempli (par les fixtures, par metadata, etc.) — sinon le
        # rapport contiendrait quand même des data-URI géants.
        if self.lazy_images:
            images_b64 = _externalize_images_to_dir(
                self.benchmark, output_path.parent,
            )
        else:
            images_b64 = self.images_b64
            if not images_b64:
                images_b64 = _encode_images_b64_from_result(self.benchmark)

        labels = get_labels(self.lang)
        report_data = _build_report_data(self.benchmark, images_b64)

        # Sprint 27 — snapshots de reproductibilité (pricing, glossaire,
        # profil de normalisation, environnement). Embarqués dans le JSON
        # du rapport pour qu'un lecteur puisse régénérer la synthèse, le
        # Pareto et le glossaire sans accès au code source.
        from picarones.report.snapshot import snapshot_all
        report_data["snapshots"] = snapshot_all(
            lang=self.lang,
            normalization_profile=self.normalization_profile,
        )

        report_json = json.dumps(report_data, ensure_ascii=False, separators=(",", ":"))
        i18n_json = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
        chartjs_js = _load_vendor_js("chart.umd.min.js")

        # Sprint 17 — rendu SVG du CDD côté serveur (statique, pas de JS)
        cdd_svg = build_critical_difference_svg(
            report_data.get("statistics", {}).get("nemenyi", {}),
        )

        # Sprint 18 — synthèse factuelle narrative (déterministe, sans LLM)
        from picarones.measurements.narrative import build_synthesis
        synthesis = build_synthesis(report_data, lang=self.lang)

        # Sprint 20 — glossaire contextuel chargé depuis YAML
        from picarones.reports_v2.glossary import load_glossary
        glossary = load_glossary(self.lang)
        glossary_json = json.dumps(glossary, ensure_ascii=False, separators=(",", ":"))

        section_html = self._build_section_html(report_data, labels)

        env = _build_jinja_env()
        template = env.get_template("base.html.j2")
        html = template.render(
            corpus_name=self.benchmark.corpus_name,
            picarones_version=self.benchmark.picarones_version,
            report_data_json=report_json,
            i18n_json=i18n_json,
            html_lang=labels.get("html_lang", "fr"),
            chartjs_inline=chartjs_js,
            critical_difference_svg=cdd_svg,
            friedman=report_data.get("statistics", {}).get("friedman", {}),
            synthesis=synthesis,
            glossary_json=glossary_json,
            **section_html,
        )

        output_path.write_text(html, encoding="utf-8")
        return output_path.resolve()

    def _build_section_html(
        self, report_data: dict, labels: dict[str, str],
    ) -> dict[str, str]:
        """Construit toutes les sections HTML conditionnelles du rapport.

        Chaque renderer (NER, calibration, philologie, etc.) est appelé
        de manière indépendante. Une section retourne ``""`` si aucun
        moteur n'a de signal pour elle — le template gère l'affichage
        conditionnel.

        Returns
        -------
        dict[str, str]
            Map ``{nom_de_section: html}`` à splatter dans
            ``template.render(**section_html)``.
        """
        engines = report_data.get("engines", [])

        # Sprint 37 — section inter-moteurs (matrice de divergence + oracle).
        from picarones.report.inter_engine_render import (
            build_divergence_matrix_html,
            build_oracle_gap_html,
        )
        # Sprint 41 — section NER (résumé F1 par moteur + heatmap par catégorie).
        from picarones.report.ner_render import (
            build_ner_per_category_html,
            build_ner_summary_html,
        )
        # Sprint 43 — section calibration (tableau ECE/MCE + grille de
        # reliability diagrams par moteur).
        from picarones.report.calibration_render import (
            build_calibration_summary_html,
            build_reliability_diagrams_grid_html,
        )
        # Sprint 46 — section stratifiée (tableau par strate).
        from picarones.report.stratification_render import (
            build_stratified_ranking_html,
        )
        # Sprint 62 — profil philologique (6 sections adaptive).
        from picarones.report.philological_render import (
            build_philological_profile_html,
        )
        # Sprint 86 — A.II.5 : recherchabilité fuzzy + séquences numériques.
        from picarones.report.searchability_render import (
            build_searchability_summary_html,
        )
        from picarones.report.numerical_sequences_render import (
            build_numerical_sequences_html,
        )
        # Sprint 87 — A.II.2 : lisibilité (delta Flesch).
        from picarones.report.readability_render import (
            build_readability_summary_html,
        )
        # Sprint 89 — A.II.8b : spécialisation inter-moteurs.
        from picarones.report.specialization_render import (
            build_specialization_html,
        )
        # Chantier 3 (post-Sprint 97) — 3 vues thématiques composées.
        from picarones.report.views import (
            build_advanced_taxonomy_view_html,
            build_diagnostics_view_html,
            build_economics_view_html,
        )
        # Sprint « câblage des modules test-only » (mai 2026) — sections
        # qui consomment les nouvelles métriques calculées dans
        # ``report_data.extra_metrics``.
        from picarones.report.marginal_cost_render import (
            build_marginal_cost_html,
        )
        from picarones.report.rare_token_recall_render import (
            build_rare_token_recall_html,
        )
        from picarones.report.taxonomy_cooccurrence_render import (
            build_taxonomy_cooccurrence_html,
        )
        from picarones.report.taxonomy_intra_doc_render import (
            build_taxonomy_intra_doc_html,
        )

        # Spécialisation : construit une map {engine: counts} depuis les
        # ``aggregated_taxonomy`` ; un moteur sans taxonomie est exclu.
        taxos: dict = {}
        for eng in engines:
            tax = eng.get("aggregated_taxonomy")
            if isinstance(tax, dict):
                counts = tax.get("counts") if "counts" in tax else tax
                if isinstance(counts, dict) and counts:
                    taxos[eng.get("name", "?")] = {
                        k: float(v) for k, v in counts.items()
                        if isinstance(v, (int, float))
                    }

        return {
            # Sprint 37
            "divergence_matrix_html": build_divergence_matrix_html(
                report_data.get("inter_engine_analysis"), labels=labels,
            ),
            "oracle_gap_html": build_oracle_gap_html(
                report_data.get("inter_engine_analysis"), labels=labels,
            ),
            # Sprint 41
            "ner_summary_html": build_ner_summary_html(engines, labels=labels),
            "ner_per_category_html": build_ner_per_category_html(engines, labels=labels),
            # Sprint 43
            "calibration_summary_html": build_calibration_summary_html(
                engines, labels=labels,
            ),
            "reliability_diagrams_html": build_reliability_diagrams_grid_html(
                engines, labels=labels,
            ),
            # Sprint 46
            "stratified_ranking_html": build_stratified_ranking_html(
                report_data.get("stratified_ranking"),
                report_data.get("available_strata"),
                report_data.get("corpus_homogeneity"),
                labels=labels,
            ),
            # Sprint 62
            "philological_profile_html": build_philological_profile_html(
                engines, labels=labels,
            ),
            # Sprint 86
            "searchability_html": build_searchability_summary_html(
                engines, labels=labels,
            ),
            "numerical_sequences_html": build_numerical_sequences_html(
                engines, labels=labels,
            ),
            # Sprint 87
            "readability_html": build_readability_summary_html(
                engines, labels=labels,
            ),
            # Sprint 89
            "specialization_html": build_specialization_html(taxos, labels=labels),
            # Chantier 3 — vues thématiques composées
            "economics_view_html": build_economics_view_html(
                report_data, labels=labels,
                engine_reports=self.benchmark.engine_reports,
            ),
            "advanced_taxonomy_view_html": build_advanced_taxonomy_view_html(
                report_data, labels=labels,
            ),
            "diagnostics_view_html": build_diagnostics_view_html(
                report_data, labels=labels,
            ),
            # Sprint « câblage des modules test-only » (mai 2026) :
            # 4 nouvelles sections pour les modules câblés en
            # ``report_data.extra_metrics``. Adaptive : "" si pas de signal.
            "taxonomy_cooccurrence_html": build_taxonomy_cooccurrence_html(
                report_data.get("taxonomy_cooccurrence"), labels=labels,
            ),
            "taxonomy_intra_doc_html": build_taxonomy_intra_doc_html(
                report_data.get("taxonomy_intra_doc"), labels=labels,
            ),
            "rare_token_recall_html": build_rare_token_recall_html(
                report_data.get("rare_token_recall"), labels=labels,
            ),
            "marginal_cost_html": build_marginal_cost_html(
                report_data.get("marginal_cost"), labels=labels,
            ),
        }

    @classmethod
    def from_json(cls, json_path: str | Path, **kwargs) -> "ReportGenerator":
        """Crée un générateur depuis un fichier JSON de résultats.

        Compatible avec les fichiers produits par ``BenchmarkResult.to_json()``.
        Les images base64 doivent être passées via ``kwargs["images_b64"]``
        si elles ne sont pas dans le JSON.
        """
        import json as _json

        data = _json.loads(Path(json_path).read_text(encoding="utf-8"))

        # Reconstruction minimale d'un BenchmarkResult depuis le dict
        from picarones.measurements.metrics import MetricsResult
        from picarones.evaluation.benchmark_result import DocumentResult, EngineReport

        engine_reports = []
        for er_data in data.get("engine_reports", []):
            doc_results = []
            for dr_data in er_data.get("document_results", []):
                m = dr_data["metrics"]
                metrics = MetricsResult(
                    cer=m["cer"], cer_nfc=m["cer_nfc"], cer_caseless=m["cer_caseless"],
                    wer=m["wer"], wer_normalized=m["wer_normalized"],
                    mer=m["mer"], wil=m["wil"],
                    reference_length=m["reference_length"],
                    hypothesis_length=m["hypothesis_length"],
                    error=m.get("error"),
                )
                doc_results.append(DocumentResult(
                    doc_id=dr_data["doc_id"],
                    image_path=dr_data["image_path"],
                    ground_truth=dr_data["ground_truth"],
                    hypothesis=dr_data["hypothesis"],
                    metrics=metrics,
                    duration_seconds=dr_data.get("duration_seconds", 0.0),
                    engine_error=dr_data.get("engine_error"),
                ))
            engine_reports.append(EngineReport(
                engine_name=er_data["engine_name"],
                engine_version=er_data.get("engine_version", "unknown"),
                engine_config=er_data.get("engine_config", {}),
                document_results=doc_results,
            ))

        corpus_info = data.get("corpus", {})
        bm = BenchmarkResult(
            corpus_name=corpus_info.get("name", "Corpus"),
            corpus_source=corpus_info.get("source"),
            document_count=corpus_info.get("document_count", 0),
            engine_reports=engine_reports,
            run_date=data.get("run_date", ""),
            picarones_version=data.get("picarones_version", ""),
            metadata=data.get("metadata", {}),
        )

        images_b64 = kwargs.pop("images_b64", {})
        return cls(bm, images_b64=images_b64, **kwargs)
