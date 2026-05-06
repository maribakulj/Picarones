"""``HtmlReportRenderer`` — produit un rapport HTML depuis un ``RunResult``.

Cible documentée du rewrite : la génération HTML vit dans la couche
``reports_v2/html/`` (cf. ``picarones/reports_v2/__init__.py``).
Un rapport est un **format de sortie** consommant un ``RunResult``
persisté — pas un service métier.  ``app/services/`` orchestre la
génération via ``RunOrchestrator``, mais le rendu lui-même est ici.

Premier rapport HTML du nouveau monde.  Volontairement minimal : ce
service répond à *« je veux ouvrir un fichier ``.html`` et voir mon
benchmark »*, pas à *« je veux les 22 vues legacy avec Chart.js, CDD,
narrative engine, glossaire, mode avancé »* — ces vues vivent toujours
dans ``picarones.report.*`` (legacy) et seront ré-intégrées au cas par
cas dans une phase ultérieure du rewrite.

Caractéristiques
----------------
- Rendu **server-side, HTML autonome** : pas de JS, pas de CSS
  externe (les styles sont inlinés).  Un fichier qui s'ouvre
  partout, conservable en archive.
- **Pattern d'omission visible** : pour chaque vue × pipeline, si le
  pipeline ne produit pas d'artefact éligible, la cellule affiche
  ``OMIS`` au lieu d'un score factice ``0`` qui mentirait.
- **Anti-injection** : tout texte d'origine utilisateur ou métier
  (``corpus_name``, ``run_id``, ``pipeline_name``, ``view.name``,
  ``view.description``, etc.) passe par :func:`html.escape`.
- **Bilingue light** : ``lang="fr"`` ou ``lang="en"`` via paramètre
  constructeur — labels traduits, valeurs intactes.

Anti-sur-ingénierie
-------------------
- Pas de coloration par gradient.  Les valeurs sont affichées en
  toutes lettres ; le caller qui veut un rendu visuel sophistiqué
  utilise le legacy.
- Pas d'arrow ↑/↓ par métrique : ``EvaluationView`` ne porte pas
  cette info (elle vit dans ``MetricSpec``, qui n'est pas dans le
  ``RunResult``).  À ajouter quand un caller a vraiment besoin.
- Pas de tri automatique des pipelines par classement : on respecte
  l'ordre du manifest (déterminisme byte-à-byte sur deux runs
  identiques).
- Pas de rendu Markdown ou Jinja2.  Construction str pure
  (``f"…"``) — facile à debugger, byte-déterministe.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from picarones.domain.evaluation_spec import EvaluationView
from picarones.domain.run_manifest import RunManifest
from picarones.app.results import RunDocumentResult, RunResult
from picarones.evaluation.views.base import ViewResult
from picarones.pipeline.types import PipelineResult


#: Marqueur affiché quand un pipeline est OMIS d'une vue.
_OMITTED_MARKER = "OMIS"


# ──────────────────────────────────────────────────────────────────────
# Labels bilingues (FR + EN)
# ──────────────────────────────────────────────────────────────────────


_LABELS: dict[str, dict[str, str]] = {
    "fr": {
        "title": "Rapport Picarones",
        "corpus": "Corpus",
        "run_id": "Identifiant du run",
        "code_version": "Version du code",
        "started_at": "Démarré",
        "completed_at": "Terminé",
        "duration_seconds": "Durée (secondes)",
        "n_documents": "Nombre de documents",
        "pipelines_overview": "Pipelines exécutées",
        "pipeline": "Pipeline",
        "n_succeeded": "Succès",
        "n_failed": "Échecs",
        "duration_total": "Durée totale (s)",
        "view": "Vue",
        "description": "Description",
        "warnings": "Avertissements",
        "ignored_dimensions": "Dimensions explicitement non évaluées",
        "results_per_pipeline": "Résultats par pipeline (moyenne)",
        "n_observations": "n",
        "omitted_explanation": (
            "Pipeline ne produisant pas d'artefact éligible à cette vue. "
            "Pas de score factice — l'omission est l'information."
        ),
        "footer": "Généré par Picarones (rewrite ciblé S21)",
        "no_data_for_view": (
            "Aucun pipeline n'a produit d'artefact éligible à cette vue."
        ),
    },
    "en": {
        "title": "Picarones Report",
        "corpus": "Corpus",
        "run_id": "Run identifier",
        "code_version": "Code version",
        "started_at": "Started",
        "completed_at": "Completed",
        "duration_seconds": "Duration (seconds)",
        "n_documents": "Document count",
        "pipelines_overview": "Pipelines executed",
        "pipeline": "Pipeline",
        "n_succeeded": "Succeeded",
        "n_failed": "Failed",
        "duration_total": "Total duration (s)",
        "view": "View",
        "description": "Description",
        "warnings": "Warnings",
        "ignored_dimensions": "Dimensions explicitly not evaluated",
        "results_per_pipeline": "Per-pipeline results (mean)",
        "n_observations": "n",
        "omitted_explanation": (
            "Pipeline did not produce any artifact eligible for this view. "
            "No fake score — omission is the information."
        ),
        "footer": "Generated by Picarones (targeted rewrite S21)",
        "no_data_for_view": (
            "No pipeline produced an artifact eligible for this view."
        ),
    },
}


# ──────────────────────────────────────────────────────────────────────
# CSS minimal inliné
# ──────────────────────────────────────────────────────────────────────


_INLINE_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
       Helvetica, Arial, sans-serif; margin: 2em; line-height: 1.4;
       color: #222; }
header { border-bottom: 2px solid #444; padding-bottom: 0.8em;
         margin-bottom: 1.5em; }
h1 { margin: 0 0 0.4em 0; }
h2 { margin-top: 2em; padding-top: 0.6em; border-top: 1px solid #ccc; }
h3 { margin-top: 1.4em; }
table { border-collapse: collapse; margin: 0.8em 0; min-width: 60%; }
th, td { border: 1px solid #ccc; padding: 0.4em 0.8em; text-align: left;
         vertical-align: top; }
th { background: #f4f4f4; font-weight: 600; }
td.numeric { text-align: right; font-variant-numeric: tabular-nums; }
td.omitted { color: #888; font-style: italic; background: #fafafa;
             text-align: center; }
ul.warnings { background: #fff8e1; border-left: 4px solid #f9a825;
              padding: 0.6em 1em; margin: 0.8em 0; }
.description { color: #555; font-style: italic; margin: 0.3em 0 1em 0; }
.ignored { color: #777; font-size: 0.9em; margin-top: 0.6em; }
code { background: #f4f4f4; padding: 0.1em 0.3em; border-radius: 3px; }
footer { margin-top: 3em; padding-top: 0.8em; border-top: 1px solid #ccc;
         color: #888; font-size: 0.85em; }
.empty-view { color: #888; font-style: italic; padding: 0.8em;
              border: 1px dashed #ccc; }
""".strip()


# ──────────────────────────────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Aggregate:
    """Moyenne d'une métrique pour un pipeline donné dans une vue."""

    mean: float
    n: int


class HtmlReportRenderer:
    """Génère un rapport HTML à partir d'un ``RunResult``.

    Parameters
    ----------
    lang:
        Langue des labels.  ``"fr"`` (défaut) ou ``"en"``.  Une langue
        non supportée fait fallback sur ``"fr"`` avec un caractère
        diacritique préservé pour signaler qu'un fallback a eu lieu.
    """

    def __init__(self, *, lang: str = "fr") -> None:
        if lang not in _LABELS:
            lang = "fr"
        self._lang = lang
        self._labels = _LABELS[lang]

    # ──────────────────────────────────────────────────────────────────
    # API publique
    # ──────────────────────────────────────────────────────────────────

    def render(self, result: RunResult) -> str:
        """Produit le HTML d'un ``RunResult`` (chargé en mémoire)."""
        manifest = result.manifest
        artifact_to_pipeline = _build_artifact_to_pipeline_map(
            result.document_results,
        )
        pipeline_summaries = _summarize_pipelines(result.document_results)

        sections = [
            self._render_head(manifest),
            self._render_header_block(manifest),
            self._render_pipelines_overview(
                manifest.pipeline_names, pipeline_summaries,
            ),
        ]
        for view in manifest.view_specs:
            view_results = result.view_results_for(view.name)
            sections.append(
                self._render_view(
                    view=view,
                    view_results=view_results,
                    pipeline_names=manifest.pipeline_names,
                    artifact_to_pipeline=artifact_to_pipeline,
                ),
            )
        sections.append(self._render_footer(manifest))
        return "\n".join(sections) + "\n"

    def render_from_dir(self, run_dir: Path | str) -> str:
        """Lit les 3 fichiers persistés et produit le HTML.

        Pendant inverse de ``BenchmarkService.persist`` : permet de
        re-générer un rapport sans avoir le ``RunResult`` en mémoire
        (cas de la CLI ``picarones report <run_dir>``).
        """
        result = self.load_run_result(run_dir)
        return self.render(result)

    # ──────────────────────────────────────────────────────────────────
    # Loader (statique, utilisable hors instance)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def load_run_result(run_dir: Path | str) -> RunResult:
        """Reconstruit un ``RunResult`` depuis les 4 fichiers persistés
        par ``BenchmarkService.persist`` (S41).

        Raises
        ------
        FileNotFoundError
            Si l'un des fichiers obligatoires (manifest,
            pipeline_results, view_results) est manquant.
            ``artifacts_index.jsonl`` est optionnel pour rester
            compatible avec d'anciens runs persistés avant S41.
        """
        d = Path(run_dir)
        manifest_path = d / "run_manifest.json"
        pipelines_path = d / "pipeline_results.jsonl"
        artifacts_index_path = d / "artifacts_index.jsonl"
        views_path = d / "view_results.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"run_manifest.json absent du dossier : {d!r}",
            )
        if not pipelines_path.exists():
            raise FileNotFoundError(
                f"pipeline_results.jsonl absent du dossier : {d!r}",
            )
        if not views_path.exists():
            raise FileNotFoundError(
                f"view_results.jsonl absent du dossier : {d!r}",
            )
        manifest = RunManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8"),
        )

        # S41 — l'index d'artefacts est désormais séparé des
        # pipeline_results.jsonl.  On le lit AVANT pour pouvoir
        # ré-attacher les artefacts à chaque pipeline_result lors de
        # la reconstruction.
        artifacts_by_pipeline: dict[
            tuple[str, str], list[dict],
        ] = {}
        if artifacts_index_path.exists():
            with artifacts_index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    # `pipeline_name` est uniquement un champ d'index
                    # (groupement) — on le retire avant de re-valider
                    # un Artifact (qui a `extra="forbid"`).  En revanche
                    # `document_id` fait partie de l'Artifact lui-même
                    # et doit être préservé pour la validation pydantic.
                    pipe_name = rec.pop("pipeline_name")
                    doc_id = rec["document_id"]
                    artifacts_by_pipeline.setdefault(
                        (doc_id, pipe_name), [],
                    ).append(rec)

        # Reconstruire les pipeline_results et view_results par doc.
        pipeline_results_by_doc: dict[str, list[PipelineResult]] = {}
        with pipelines_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                doc_id = payload["document_id"]
                # Ré-attache les artefacts depuis l'index S41 si présent.
                key = (doc_id, payload.get("pipeline_name", ""))
                if key in artifacts_by_pipeline and "artifacts" not in payload:
                    payload["artifacts"] = artifacts_by_pipeline[key]
                pipeline_results_by_doc.setdefault(doc_id, []).append(
                    PipelineResult.model_validate(payload),
                )

        view_results_by_doc: dict[str, list[ViewResult]] = {}
        with views_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                doc_id = payload.pop("document_id")
                view_results_by_doc.setdefault(doc_id, []).append(
                    ViewResult.model_validate(payload),
                )

        all_doc_ids = sorted(
            set(pipeline_results_by_doc) | set(view_results_by_doc),
        )
        document_results = tuple(
            RunDocumentResult(
                document_id=doc_id,
                pipeline_results=tuple(
                    pipeline_results_by_doc.get(doc_id, []),
                ),
                view_results=tuple(view_results_by_doc.get(doc_id, [])),
            )
            for doc_id in all_doc_ids
        )
        return RunResult(manifest=manifest, document_results=document_results)

    # ──────────────────────────────────────────────────────────────────
    # Helpers de rendu
    # ──────────────────────────────────────────────────────────────────

    def _render_head(self, manifest: RunManifest) -> str:
        title = html.escape(
            f"{self._labels['title']} — {manifest.corpus_name}",
        )
        return (
            f'<!DOCTYPE html>\n'
            f'<html lang="{self._lang}">\n'
            f'<head>\n'
            f'<meta charset="utf-8">\n'
            f'<title>{title}</title>\n'
            f'<style>\n{_INLINE_CSS}\n</style>\n'
            f'</head>\n'
            f'<body>'
        )

    def _render_header_block(self, manifest: RunManifest) -> str:
        L = self._labels
        return (
            f'<header>\n'
            f'<h1>{html.escape(L["title"])}</h1>\n'
            f'<p>{html.escape(L["corpus"])} : '
            f'<strong>{html.escape(manifest.corpus_name)}</strong></p>\n'
            f'<p>{html.escape(L["run_id"])} : '
            f'<code>{html.escape(manifest.run_id)}</code></p>\n'
            f'<p>{html.escape(L["code_version"])} : '
            f'<code>{html.escape(manifest.code_version)}</code></p>\n'
            f'<p>{html.escape(L["started_at"])} : '
            f'{html.escape(manifest.started_at.isoformat())} • '
            f'{html.escape(L["completed_at"])} : '
            f'{html.escape(manifest.completed_at.isoformat())} • '
            f'{html.escape(L["duration_seconds"])} : '
            f'{manifest.duration_seconds:.3f}</p>\n'
            f'<p>{html.escape(L["n_documents"])} : '
            f'{manifest.n_documents}</p>\n'
            f'</header>'
        )

    def _render_pipelines_overview(
        self,
        pipeline_names: tuple[str, ...],
        summaries: dict[str, "_PipelineSummary"],
    ) -> str:
        L = self._labels
        rows = []
        for name in pipeline_names:
            s = summaries.get(name)
            if s is None:
                # Pipeline du manifest sans aucun résultat (cas dégénéré).
                rows.append(
                    f'<tr><td>{html.escape(name)}</td>'
                    f'<td class="numeric">0</td>'
                    f'<td class="numeric">0</td>'
                    f'<td class="numeric">—</td></tr>',
                )
                continue
            rows.append(
                f'<tr>'
                f'<td>{html.escape(name)}</td>'
                f'<td class="numeric">{s.n_succeeded}</td>'
                f'<td class="numeric">{s.n_failed}</td>'
                f'<td class="numeric">{s.duration_total:.3f}</td>'
                f'</tr>',
            )
        rows_html = "\n".join(rows) if rows else (
            '<tr><td colspan="4" class="omitted">—</td></tr>'
        )
        return (
            f'<section id="pipelines-overview">\n'
            f'<h2>{html.escape(L["pipelines_overview"])}</h2>\n'
            f'<table>\n'
            f'<thead><tr>'
            f'<th>{html.escape(L["pipeline"])}</th>'
            f'<th>{html.escape(L["n_succeeded"])}</th>'
            f'<th>{html.escape(L["n_failed"])}</th>'
            f'<th>{html.escape(L["duration_total"])}</th>'
            f'</tr></thead>\n'
            f'<tbody>\n{rows_html}\n</tbody>\n'
            f'</table>\n'
            f'</section>'
        )

    def _render_view(
        self,
        *,
        view: EvaluationView,
        view_results: tuple[ViewResult, ...],
        pipeline_names: tuple[str, ...],
        artifact_to_pipeline: dict[str, str],
    ) -> str:
        L = self._labels
        view_id = html.escape(view.name)
        per_pipeline = _aggregate_view_by_pipeline(
            view_results=view_results,
            artifact_to_pipeline=artifact_to_pipeline,
            metric_names=view.metric_names,
        )

        warnings_html = ""
        if view.warnings:
            items = "\n".join(
                f'<li>{html.escape(w)}</li>' for w in view.warnings
            )
            warnings_html = (
                f'<ul class="warnings">\n{items}\n</ul>'
            )

        # En-tête : Pipeline | metric_a | metric_b | ... | n
        header_cells = [
            f'<th>{html.escape(L["pipeline"])}</th>',
        ]
        for m in view.metric_names:
            header_cells.append(f'<th>{html.escape(m)}</th>')
        header_cells.append(
            f'<th>{html.escape(L["n_observations"])}</th>',
        )

        # Lignes : un par pipeline du manifest.
        body_rows: list[str] = []
        any_data = bool(per_pipeline)
        for pipeline_name in pipeline_names:
            cells = [f'<td>{html.escape(pipeline_name)}</td>']
            agg = per_pipeline.get(pipeline_name)
            if agg is None:
                # OMIS — rendu fusionné sur toutes les colonnes métriques + n.
                cells.append(
                    f'<td colspan="{len(view.metric_names) + 1}" '
                    f'class="omitted" '
                    f'title="{html.escape(L["omitted_explanation"])}">'
                    f'{_OMITTED_MARKER}'
                    f'</td>',
                )
            else:
                # Une cellule par métrique + colonne n.
                # n = max(n_observations) parmi les métriques calculées
                # (typiquement identique pour toutes les métriques d'une
                # même vue).
                for m in view.metric_names:
                    metric_agg = agg.get(m)
                    if metric_agg is None:
                        cells.append('<td class="numeric">—</td>')
                    else:
                        cells.append(
                            f'<td class="numeric">{metric_agg.mean:.4f}</td>',
                        )
                ns = [a.n for a in agg.values() if a is not None]
                n = max(ns) if ns else 0
                cells.append(f'<td class="numeric">{n}</td>')
            body_rows.append(f'<tr>{"".join(cells)}</tr>')

        if any_data:
            table_html = (
                f'<h3>{html.escape(L["results_per_pipeline"])}</h3>\n'
                f'<table>\n'
                f'<thead><tr>{"".join(header_cells)}</tr></thead>\n'
                f'<tbody>\n' + "\n".join(body_rows) + '\n</tbody>\n'
                '</table>'
            )
        else:
            table_html = (
                f'<p class="empty-view">'
                f'{html.escape(L["no_data_for_view"])}</p>'
            )

        ignored_html = ""
        if view.ignored_dimensions:
            ignored_html = (
                f'<p class="ignored">'
                f'{html.escape(L["ignored_dimensions"])} : '
                f'{html.escape(", ".join(view.ignored_dimensions))}'
                f'</p>'
            )

        return (
            f'<section class="view" id="view-{view_id}">\n'
            f'<h2>{html.escape(L["view"])} : '
            f'{html.escape(view.name)}</h2>\n'
            f'<p class="description">'
            f'{html.escape(view.description or "")}</p>\n'
            f'{warnings_html}\n'
            f'{table_html}\n'
            f'{ignored_html}\n'
            f'</section>'
        )

    def _render_footer(self, manifest: RunManifest) -> str:
        return (
            f'<footer>\n'
            f'<p>{html.escape(self._labels["footer"])} • '
            f'{html.escape(manifest.code_version)}</p>\n'
            f'</footer>\n'
            f'</body>\n'
            f'</html>'
        )


# ──────────────────────────────────────────────────────────────────────
# Helpers d'agrégation (purs, testables sans rendu)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _PipelineSummary:
    n_succeeded: int
    n_failed: int
    duration_total: float


def _summarize_pipelines(
    document_results: Iterable[RunDocumentResult],
) -> dict[str, _PipelineSummary]:
    """Agrège succès/échecs/durée par pipeline_name."""
    n_ok: dict[str, int] = {}
    n_fail: dict[str, int] = {}
    duration: dict[str, float] = {}
    for doc_result in document_results:
        for pr in doc_result.pipeline_results:
            name = pr.pipeline_name
            if pr.succeeded:
                n_ok[name] = n_ok.get(name, 0) + 1
            else:
                n_fail[name] = n_fail.get(name, 0) + 1
            duration[name] = duration.get(name, 0.0) + pr.duration_seconds
    all_names = set(n_ok) | set(n_fail) | set(duration)
    return {
        name: _PipelineSummary(
            n_succeeded=n_ok.get(name, 0),
            n_failed=n_fail.get(name, 0),
            duration_total=duration.get(name, 0.0),
        )
        for name in all_names
    }


def _build_artifact_to_pipeline_map(
    document_results: Iterable[RunDocumentResult],
) -> dict[str, str]:
    """Construit ``{artifact_id: pipeline_name}`` à partir des
    ``PipelineResult.artifacts`` de chaque doc.

    Permet de retrouver à quelle pipeline appartient un
    ``ViewResult.candidate_artifact_id``.
    """
    out: dict[str, str] = {}
    for doc_result in document_results:
        for pr in doc_result.pipeline_results:
            for art in pr.artifacts:
                out[art.id] = pr.pipeline_name
    return out


def _aggregate_view_by_pipeline(
    *,
    view_results: tuple[ViewResult, ...],
    artifact_to_pipeline: dict[str, str],
    metric_names: tuple[str, ...],
) -> dict[str, dict[str, _Aggregate]]:
    """Agrège les ``ViewResult`` en moyenne par (pipeline, métrique).

    Returns
    -------
    dict
        ``{pipeline_name: {metric_name: _Aggregate(mean, n)}}``.
        Pipelines absents = aucun ViewResult ne leur correspond
        (omis explicitement de la vue).
    """
    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for vr in view_results:
        pipeline_name = artifact_to_pipeline.get(vr.candidate_artifact_id)
        if pipeline_name is None:
            # Artefact orphelin : on l'ignore silencieusement (cas
            # bizarre, ne devrait pas arriver depuis BenchmarkService).
            continue
        for metric_name, value in vr.metric_values.items():
            if metric_name not in metric_names:
                continue
            if value is None:
                continue
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            sums.setdefault(pipeline_name, {}).setdefault(metric_name, 0.0)
            counts.setdefault(pipeline_name, {}).setdefault(metric_name, 0)
            sums[pipeline_name][metric_name] += fv
            counts[pipeline_name][metric_name] += 1
    out: dict[str, dict[str, _Aggregate]] = {}
    for pipeline_name, metric_sums in sums.items():
        out[pipeline_name] = {
            m: _Aggregate(
                mean=metric_sums[m] / counts[pipeline_name][m],
                n=counts[pipeline_name][m],
            )
            for m in metric_sums
        }
    return out


__all__ = [
    "HtmlReportRenderer",
]
