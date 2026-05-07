"""Comparaison de deux runs de benchmark (Sprint 28).

Phase 5.E — module relocalisé depuis ``picarones.report.comparison``
vers ``picarones.reports_v2.html.comparison``.  Le chemin legacy
reste disponible via un shim avec ``DeprecationWarning`` ;
suppression prévue en 2.0.

Le Sprint 8 a livré la persistance longitudinale via SQLite
(``picarones.measurements.history``) et un détecteur de régression CLI. Mais
aucun outil n'exposait la **comparaison** de deux runs côté rapport :
un chercheur qui itère sur 8 prompts ne pouvait pas voir d'un coup
*« Tesseract → GPT-4o version V2 a régressé de 0,8 pp en CER moyen
sur la strate paroissiaux par rapport à V1 »*.

Ce module fournit :

- ``load_benchmark_json(path)`` — charge le JSON produit par
  ``BenchmarkResult.as_dict()`` ou ``picarones run -o results.json``.
- ``compare_benchmarks(a, b)`` — calcule les deltas par moteur
  (CER mean, WER mean, comptes de documents traités/échoués) et
  par strate quand la métadonnée est présente.
- ``detect_regressions(diff, threshold)`` — liste les moteurs en
  régression (delta CER > threshold) et en amélioration
  (delta CER < -threshold).
- ``render_comparison_html(diff, output_path)`` — rendu HTML
  auto-contenu minimal via Jinja2 pour partage.

Conventions
-----------
- Les deltas sont calculés ``b - a`` (donc positif = ``b`` est pire).
- Un moteur présent dans un seul run apparaît dans ``only_in_a`` /
  ``only_in_b``, jamais dans ``deltas``.
- Un moteur dont le ``mean_cer`` est ``None`` (échec total) est
  signalé mais ne génère pas de delta numérique.
- ``threshold`` est en absolu (CER en fraction, pas en %). Défaut
  0.005 = 0,5 pp.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------

@dataclass
class EngineDelta:
    """Différence ``b - a`` pour un moteur donné."""
    engine: str
    cer_a: Optional[float]
    cer_b: Optional[float]
    delta_cer: Optional[float]
    wer_a: Optional[float]
    wer_b: Optional[float]
    delta_wer: Optional[float]
    docs_a: int
    docs_b: int
    failed_a: int
    failed_b: int
    is_regression: bool = False
    is_improvement: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "cer_a": self.cer_a,
            "cer_b": self.cer_b,
            "delta_cer": self.delta_cer,
            "wer_a": self.wer_a,
            "wer_b": self.wer_b,
            "delta_wer": self.delta_wer,
            "docs_a": self.docs_a,
            "docs_b": self.docs_b,
            "failed_a": self.failed_a,
            "failed_b": self.failed_b,
            "is_regression": self.is_regression,
            "is_improvement": self.is_improvement,
        }


@dataclass
class ComparisonResult:
    """Résultat d'une comparaison ``b - a`` entre deux runs."""
    label_a: str
    label_b: str
    run_date_a: Optional[str]
    run_date_b: Optional[str]
    corpus_a: Optional[str]
    corpus_b: Optional[str]
    deltas: list[EngineDelta] = field(default_factory=list)
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    threshold: float = 0.005

    def as_dict(self) -> dict[str, Any]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "run_date_a": self.run_date_a,
            "run_date_b": self.run_date_b,
            "corpus_a": self.corpus_a,
            "corpus_b": self.corpus_b,
            "threshold": self.threshold,
            "deltas": [d.as_dict() for d in self.deltas],
            "only_in_a": list(self.only_in_a),
            "only_in_b": list(self.only_in_b),
            "regressions": [d.as_dict() for d in self.deltas if d.is_regression],
            "improvements": [d.as_dict() for d in self.deltas if d.is_improvement],
        }


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_benchmark_json(path: str | Path) -> dict[str, Any]:
    """Charge un JSON de benchmark depuis disque.

    Accepte :
      - le format ``BenchmarkResult.as_dict()`` (clé ``ranking``,
        ``engine_reports`` ou ``engines``) ;
      - un dict déjà parsé ; dans ce cas, ``path`` peut être un dict.
    """
    if isinstance(path, dict):
        return path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier benchmark introuvable : {p}")
    with p.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Le JSON {p} doit être un dict.")
    return data


# ---------------------------------------------------------------------------
# Comparaison
# ---------------------------------------------------------------------------

def _ranking_index(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Indexe ``ranking`` par nom de moteur — robuste aux deux formats.

    Un ``BenchmarkResult.as_dict()`` expose ``ranking`` directement
    (clés ``engine``, ``mean_cer``, …). Le format alternatif ``engines``
    expose le même contenu sous des clés légèrement différentes —
    on normalise vers le format ``ranking``.
    """
    ranking = data.get("ranking")
    if isinstance(ranking, list) and ranking:
        return {
            r["engine"]: {
                "engine": r["engine"],
                "mean_cer": r.get("mean_cer"),
                "mean_wer": r.get("mean_wer"),
                "documents": int(r.get("documents") or 0),
                "failed": int(r.get("failed") or 0),
            }
            for r in ranking
            if isinstance(r, dict) and r.get("engine")
        }
    # Fallback : ``engines`` (format report_data)
    engines = data.get("engines") or []
    out: dict[str, dict[str, Any]] = {}
    if isinstance(engines, list):
        for e in engines:
            if not isinstance(e, dict):
                continue
            name = e.get("name") or e.get("engine")
            if not name:
                continue
            out[name] = {
                "engine": name,
                "mean_cer": e.get("cer"),
                "mean_wer": e.get("wer"),
                "documents": int(e.get("documents") or 0),
                "failed": int(e.get("failed") or 0),
            }
    return out


def _label_of(data: dict[str, Any], default: str) -> str:
    meta = data.get("meta") or {}
    return (
        meta.get("corpus_name")
        or (data.get("corpus") or {}).get("name")
        or default
    )


def _run_date_of(data: dict[str, Any]) -> Optional[str]:
    return (
        data.get("run_date")
        or (data.get("meta") or {}).get("run_date")
    )


def _corpus_of(data: dict[str, Any]) -> Optional[str]:
    meta = data.get("meta") or {}
    return (
        meta.get("corpus_source")
        or (data.get("corpus") or {}).get("source")
        or meta.get("corpus_name")
    )


def _safe_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(b) - float(a)


def compare_benchmarks(
    a: str | Path | dict[str, Any],
    b: str | Path | dict[str, Any],
    *,
    threshold: float = 0.005,
    label_a: str = "A",
    label_b: str = "B",
) -> ComparisonResult:
    """Compare deux runs et retourne les deltas par moteur.

    Convention : un delta CER positif signifie que ``b`` est *moins bon*
    que ``a`` (régression). Un seuil ``threshold`` strictement positif
    (en fraction, ex. 0,005 = 0,5 pp) discrimine régression / bruit.
    """
    da = load_benchmark_json(a) if not isinstance(a, dict) else a
    db = load_benchmark_json(b) if not isinstance(b, dict) else b

    idx_a = _ranking_index(da)
    idx_b = _ranking_index(db)

    common = sorted(set(idx_a) & set(idx_b))
    only_a = sorted(set(idx_a) - set(idx_b))
    only_b = sorted(set(idx_b) - set(idx_a))

    deltas: list[EngineDelta] = []
    for name in common:
        ea = idx_a[name]
        eb = idx_b[name]
        delta_cer = _safe_delta(ea["mean_cer"], eb["mean_cer"])
        delta_wer = _safe_delta(ea["mean_wer"], eb["mean_wer"])
        regression = bool(delta_cer is not None and delta_cer > threshold)
        improvement = bool(delta_cer is not None and delta_cer < -threshold)
        deltas.append(
            EngineDelta(
                engine=name,
                cer_a=ea["mean_cer"],
                cer_b=eb["mean_cer"],
                delta_cer=delta_cer,
                wer_a=ea["mean_wer"],
                wer_b=eb["mean_wer"],
                delta_wer=delta_wer,
                docs_a=int(ea["documents"]),
                docs_b=int(eb["documents"]),
                failed_a=int(ea["failed"]),
                failed_b=int(eb["failed"]),
                is_regression=regression,
                is_improvement=improvement,
            )
        )

    # Tri : régressions (delta décroissant) puis améliorations (delta croissant).
    deltas.sort(key=lambda d: (
        not d.is_regression,
        -(d.delta_cer if d.delta_cer is not None else 0.0),
    ))

    return ComparisonResult(
        label_a=label_a,
        label_b=label_b,
        run_date_a=_run_date_of(da),
        run_date_b=_run_date_of(db),
        corpus_a=_corpus_of(da),
        corpus_b=_corpus_of(db),
        deltas=deltas,
        only_in_a=only_a,
        only_in_b=only_b,
        threshold=float(threshold),
    )


def detect_regressions(
    diff: ComparisonResult,
) -> list[EngineDelta]:
    """Retourne uniquement les moteurs en régression dans ``diff``."""
    return [d for d in diff.deltas if d.is_regression]


# ---------------------------------------------------------------------------
# Rendu HTML
# ---------------------------------------------------------------------------

_COMPARISON_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Picarones — Comparaison de runs</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 980px; margin: 2em auto; padding: 0 1em; color: #111; }
  h1 { border-bottom: 2px solid #333; padding-bottom: .4em; }
  h2 { margin-top: 1.6em; color: #333; }
  table { width: 100%; border-collapse: collapse; margin: 1em 0; }
  th, td { padding: .5em .8em; text-align: left; border-bottom: 1px solid #ddd; }
  th { background: #f3f3f3; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  tr.regression td { background: #fef0f0; }
  tr.improvement td { background: #f0fef2; }
  .delta-pos { color: #b0322a; font-weight: 600; }
  .delta-neg { color: #1b8a3a; font-weight: 600; }
  .badge { display: inline-block; padding: .15em .55em; border-radius: 4px;
           font-size: .8em; font-weight: 600; }
  .badge.reg { background: #fde2e0; color: #8a1c14; }
  .badge.imp { background: #e0f8e6; color: #0a5e22; }
  .meta { color: #666; font-size: .9em; }
  .empty { color: #999; font-style: italic; }
</style>
</head>
<body>
<h1>Comparaison : {{ diff.label_a }} → {{ diff.label_b }}</h1>
<p class="meta">
  Run A : {{ diff.run_date_a or "?" }} · corpus {{ diff.corpus_a or "?" }}<br>
  Run B : {{ diff.run_date_b or "?" }} · corpus {{ diff.corpus_b or "?" }}<br>
  Seuil régression / amélioration : {{ "%.3f"|format(diff.threshold) }}
  ({{ "%.1f"|format(diff.threshold * 100) }} pp de CER absolu).
</p>

<h2>Moteurs comparés ({{ diff.deltas|length }})</h2>
{% if not diff.deltas %}
  <p class="empty">Aucun moteur commun aux deux runs.</p>
{% else %}
<table>
  <thead>
    <tr>
      <th scope=\"col\">Moteur</th>
      <th scope=\"col\" class="num">CER A</th>
      <th scope=\"col\" class="num">CER B</th>
      <th scope=\"col\" class="num">Δ CER</th>
      <th scope=\"col\" class="num">Docs A → B</th>
      <th scope=\"col\">État</th>
    </tr>
  </thead>
  <tbody>
  {% for d in diff.deltas %}
    <tr class="{% if d.is_regression %}regression{% elif d.is_improvement %}improvement{% endif %}">
      <td>{{ d.engine }}</td>
      <td class="num">{{ "%.3f"|format(d.cer_a) if d.cer_a is not none else "—" }}</td>
      <td class="num">{{ "%.3f"|format(d.cer_b) if d.cer_b is not none else "—" }}</td>
      <td class="num">
        {% if d.delta_cer is none %}—
        {% elif d.delta_cer > 0 %}<span class="delta-pos">+{{ "%.3f"|format(d.delta_cer) }}</span>
        {% else %}<span class="delta-neg">{{ "%.3f"|format(d.delta_cer) }}</span>
        {% endif %}
      </td>
      <td class="num">{{ d.docs_a }} → {{ d.docs_b }}</td>
      <td>
        {% if d.is_regression %}<span class="badge reg">régression</span>
        {% elif d.is_improvement %}<span class="badge imp">amélioration</span>
        {% else %}<span class="meta">stable</span>{% endif %}
      </td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

{% if diff.only_in_a %}
<h2>Présents uniquement dans A</h2>
<ul>{% for n in diff.only_in_a %}<li>{{ n }}</li>{% endfor %}</ul>
{% endif %}

{% if diff.only_in_b %}
<h2>Présents uniquement dans B</h2>
<ul>{% for n in diff.only_in_b %}<li>{{ n }}</li>{% endfor %}</ul>
{% endif %}

<p class="meta">Picarones — Sprint 28 · rapport de comparaison de runs.</p>
</body>
</html>
"""


def render_comparison_html(
    diff: ComparisonResult,
    output_path: str | Path,
) -> Path:
    """Sérialise un ``ComparisonResult`` en rapport HTML auto-contenu."""
    from jinja2 import Environment, select_autoescape

    env = Environment(autoescape=select_autoescape(["html", "j2"]))
    template = env.from_string(_COMPARISON_TEMPLATE)
    html = template.render(diff=diff)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


__all__ = [
    "EngineDelta",
    "ComparisonResult",
    "load_benchmark_json",
    "compare_benchmarks",
    "detect_regressions",
    "render_comparison_html",
]
