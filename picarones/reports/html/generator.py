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

- :mod:`picarones.reports._helpers.assets` — chargement vendor.js, encodage
  base64 d'images, externalisation lazy.
- :mod:`picarones.reports.html.data` — construction du dict JSON
  passé au template (engines, documents, statistiques, Pareto, etc.).
- :mod:`picarones.reports._helpers.render_helpers` — couleurs / SVG mutualisés.

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
nom canonique dans :mod:`picarones.reports._helpers.assets` ou
:mod:`picarones.reports._helpers.render_helpers`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from picarones.evaluation.benchmark_result import BenchmarkResult
from picarones.evaluation.statistics import build_critical_difference_svg
from picarones.reports._helpers.assets import (
    encode_images_b64_from_result as _encode_images_b64_from_result,
    externalize_images_to_dir as _externalize_images_to_dir,
    load_vendor_js as _load_vendor_js,
)

# Ré-exports rétrocompat consommés par les tests externes (cf. docstring
# de module). La directive de fin de ligne documente l'intention de
# ré-export et empêche ruff de marquer l'import comme inutilisé.
from picarones.reports._helpers.render_helpers import cer_step_color as _cer_color  # noqa: F401
from picarones.reports.html.data import build_report_data as _build_report_data  # noqa: F401

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

    Sprint S1 (Bandit B701, CWE-94) : autoescape activé via
    ``select_autoescape``.  Les variables qui contiennent du HTML
    pré-construit (renderers thématiques, SVG, JSON) sont marquées
    avec ``| safe`` dans les templates ; les variables d'origine
    utilisateur sont auto-échappées.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2", "xml"]),
        keep_trailing_newline=True,
    )
    return env


def _safe_json_for_script_tag(data: object) -> str:
    """Sérialise data en JSON safe pour injection dans <script type="application/json">.

    Sprint S1 — protection XSS : un fragment ``</script>`` dans une
    chaîne JSON termine le tag <script> parent, même si la chaîne
    est syntaxiquement bien formée côté JSON.

    Solution standard : remplacer ``<``, ``>``, ``&`` par leurs
    séquences d'échappement Unicode JSON.  JavaScript décode au
    parse — plus de tag-break possible.
    """
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # On remplace ``<`` par ``<`` (séquence JSON, JavaScript la
    # décode au parse en ``<``).  Idem ``>`` et ``&``.  Le ``\\`` du
    # Python source produit un seul ``\`` dans la sortie.
    return (
        raw.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Génère un rapport HTML interactif depuis un BenchmarkResult.

    Usage
    -----
    >>> from picarones.reports.html import ReportGenerator
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
        from picarones.reports.i18n import get_labels

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
        from picarones.reports.html.snapshot import snapshot_all
        report_data["snapshots"] = snapshot_all(
            lang=self.lang,
            normalization_profile=self.normalization_profile,
        )

        report_json = _safe_json_for_script_tag(report_data)
        i18n_json = _safe_json_for_script_tag(labels)
        chartjs_js = _load_vendor_js("chart.umd.min.js")

        # Sprint 17 — rendu SVG du CDD côté serveur (statique, pas de JS)
        cdd_svg = build_critical_difference_svg(
            report_data.get("statistics", {}).get("nemenyi", {}),
        )

        # Sprint 18 — synthèse factuelle narrative (déterministe, sans LLM)
        from picarones.reports.narrative import build_synthesis
        synthesis = build_synthesis(report_data, lang=self.lang)

        # Sprint 20 — glossaire contextuel chargé depuis YAML
        from picarones.reports.glossary import load_glossary
        glossary = load_glossary(self.lang)
        glossary_json = _safe_json_for_script_tag(glossary)

        section_html = self._build_section_html(report_data, labels)

        env = _build_jinja_env()
        template = env.get_template("base.html.j2")
        html = template.render(
            corpus_name=self.benchmark.corpus_name,
            picarones_version=self.benchmark.picarones_version,
            # Audit scientifique F3 — bandeau d'intégrité rendu
            # côté serveur (visible même JavaScript désactivé).
            is_demo=getattr(self.benchmark, "is_demo", False),
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
        from picarones.reports.html.renderers.inter_engine import (
            build_divergence_matrix_html,
            build_oracle_gap_html,
        )
        # Sprint 41 — section NER (résumé F1 par moteur + heatmap par catégorie).
        from picarones.reports.html.renderers.ner import (
            build_ner_per_category_html,
            build_ner_summary_html,
        )
        # Sprint 43 — section calibration (tableau ECE/MCE + grille de
        # reliability diagrams par moteur).
        from picarones.reports.html.renderers.calibration import (
            build_calibration_summary_html,
            build_reliability_diagrams_grid_html,
        )
        # Sprint 46 — section stratifiée (tableau par strate).
        from picarones.reports.html.renderers.stratification import (
            build_stratified_ranking_html,
        )
        # Sprint 62 — profil philologique (6 sections adaptive).
        from picarones.reports.html.renderers.philological import (
            build_philological_profile_html,
        )
        # Sprint 86 — A.II.5 : recherchabilité fuzzy + séquences numériques.
        from picarones.reports.html.renderers.searchability import (
            build_searchability_summary_html,
        )
        from picarones.reports.html.renderers.numerical_sequences import (
            build_numerical_sequences_html,
        )
        # Sprint 87 — A.II.2 : lisibilité (delta Flesch).
        from picarones.reports.html.renderers.readability import (
            build_readability_summary_html,
        )
        # Sprint 89 — A.II.8b : spécialisation inter-moteurs.
        from picarones.reports.html.renderers.specialization import (
            build_specialization_html,
        )
        # Chantier 3 (post-Sprint 97) — 3 vues thématiques composées.
        from picarones.reports.html.views import (
            build_advanced_taxonomy_view_html,
            build_diagnostics_view_html,
            build_economics_view_html,
        )
        # Sprint « câblage des modules test-only » (mai 2026) — sections
        # qui consomment les nouvelles métriques calculées dans
        # ``report_data.extra_metrics``.
        from picarones.reports.html.renderers.marginal_cost import (
            build_marginal_cost_html,
        )
        from picarones.reports.html.renderers.rare_token_recall import (
            build_rare_token_recall_html,
        )
        from picarones.reports.html.renderers.taxonomy_cooccurrence import (
            build_taxonomy_cooccurrence_html,
        )
        from picarones.reports.html.renderers.taxonomy_intra_doc import (
            build_taxonomy_intra_doc_html,
        )
        # Phase B6 (mai 2026) — sections par vue d'évaluation
        # (text_final, alto_documentary, searchability) issues du
        # RunOrchestrator.  Adaptive : "" si benchmark.view_results
        # est vide (chemin legacy sans vues).
        from picarones.reports.html.renderers.view_results import (
            build_view_results_html,
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
            # Phase B6 (mai 2026) — sections par vue d'évaluation.
            "view_results_html": build_view_results_html(
                self.benchmark.view_results,
                all_engine_names=[
                    r.engine_name for r in self.benchmark.engine_reports
                ],
                lang=self.lang,
            ),
        }

    @classmethod
    def from_json(cls, json_path: str | Path, **kwargs) -> "ReportGenerator":
        """Crée un générateur depuis un fichier JSON de résultats.

        Compatible avec les fichiers produits par ``BenchmarkResult.to_json()``.
        Les images base64 doivent être passées via ``kwargs["images_b64"]``
        si elles ne sont pas dans le JSON.

        Phase 2.2 du chantier post-rewrite : délégué à
        :meth:`BenchmarkResult.from_json_object` qui reconstruit tous
        les champs avancés (confusion_matrix, taxonomy, structure,
        hallucination_metrics, ner_metrics, calibration_metrics,
        philological_metrics, searchability_metrics,
        numerical_sequence_metrics, readability_metrics,
        pipeline_metadata, ocr_intermediate + leurs équivalents
        ``aggregated_*`` au niveau EngineReport).  Le rapport régénéré
        depuis JSON est désormais indistinguable du rapport in-memory.
        """
        bm = BenchmarkResult.from_json_object(json_path)
        images_b64 = kwargs.pop("images_b64", {})
        return cls(bm, images_b64=images_b64, **kwargs)
