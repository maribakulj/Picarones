"""Sprint S1.3 — XSS dans le rapport HTML généré.

Vérifie que tout contenu utilisateur (corpus_name, sentences narratives,
nom de moteur) est échappé HTML dans le rapport produit par
``picarones.reports.html.ReportGenerator``.

Bandit B701 (CWE-94) avait flaggé ``Environment(autoescape=False)`` —
ce test concrétise l'attaque que l'audit décrivait : un corpus nommé
``</title><script>alert(1)</script>`` doit être inerte dans le HTML.
"""

from __future__ import annotations

from pathlib import Path

from picarones.evaluation.benchmark_result import (
    BenchmarkResult,
    DocumentResult,
    EngineReport,
)


def _make_minimal_benchmark(
    corpus_name: str,
    engine_name: str = "tesseract",
) -> BenchmarkResult:
    """Construit un BenchmarkResult minimal valide avec le ``corpus_name`` fourni."""
    from picarones.evaluation.metric_result import MetricsResult

    metrics = MetricsResult(cer=0.0, wer=0.0, mer=0.0, wil=0.0)
    doc = DocumentResult(
        doc_id="doc01",
        image_path="doc01.png",
        ground_truth="Bonjour",
        hypothesis="Bonjour",
        metrics=metrics,
        duration_seconds=0.1,
    )
    engine = EngineReport(
        engine_name=engine_name,
        engine_version="5.3.0",
        engine_config={},
        document_results=[doc],
    )
    return BenchmarkResult(
        corpus_name=corpus_name,
        corpus_source=None,
        document_count=1,
        engine_reports=[engine],
    )


# ──────────────────────────────────────────────────────────────────────
# 1. corpus_name avec script tag → doit être échappé dans <title>
# ──────────────────────────────────────────────────────────────────────


class TestCorpusNameXSS:
    """Le ``corpus_name`` est injecté dans ``<title>`` de
    ``base.html.j2``.  Sans échappement, un nom malicieux compromet
    tout rapport partagé."""

    def test_script_tag_in_corpus_name_is_escaped(self, tmp_path: Path) -> None:
        from picarones.reports.html import ReportGenerator

        bench = _make_minimal_benchmark(
            corpus_name="</title><script>alert('xss')</script>",
        )
        gen = ReportGenerator(bench)
        out = tmp_path / "report.html"
        gen.generate(out)

        html = out.read_text()

        # Le tag </title> ne doit PAS apparaître au milieu du HTML
        # (autre que celui légitime à la fin du <head>) au point
        # d'attacher un <script>.
        assert "<script>alert('xss')</script>" not in html, (
            "XSS confirmé : le script malicieux est exécutable dans le rapport.\n"
            "Causes possibles : autoescape=False dans Jinja2 + corpus_name "
            "non filtré."
        )

        # Forme correcte : caractères dangereux échappés.
        assert "&lt;script&gt;alert(" in html or "&#x3c;script&#x3e;" in html.lower(), (
            "Le ``<`` dans corpus_name doit être échappé en ``&lt;``."
        )

    def test_html_attribute_injection_in_corpus_name(self, tmp_path: Path) -> None:
        """Cas d'attaque attribute-based : ``" onerror=alert(1)``
        peut casser une balise si corpus_name est utilisée dans
        un attribut."""
        from picarones.reports.html import ReportGenerator

        bench = _make_minimal_benchmark(
            corpus_name='" onerror="alert(1)" foo="',
        )
        gen = ReportGenerator(bench)
        out = tmp_path / "report.html"
        gen.generate(out)

        html = out.read_text()

        # Le caractère ``"`` doit être échappé en ``&quot;`` ou
        # ``&#34;`` partout où corpus_name est rendu.  Aucune
        # attribute injection ne doit subsister.
        assert ' onerror="alert(' not in html, (
            "Attribute injection : un guillemet non échappé permet "
            "d'injecter onerror=."
        )

    def test_corpus_name_with_unicode_renders_correctly(
        self, tmp_path: Path,
    ) -> None:
        """Corollaire — vérifie qu'un nom Unicode légitime
        (``Manuscrit médiéval — chartes XIII°``) reste lisible
        après échappement."""
        from picarones.reports.html import ReportGenerator

        bench = _make_minimal_benchmark(
            corpus_name="Manuscrit médiéval — chartes XIII°",
        )
        gen = ReportGenerator(bench)
        out = tmp_path / "report.html"
        gen.generate(out)

        html = out.read_text()
        # Caractères Unicode safe doivent rester intacts.
        assert "Manuscrit médiéval" in html
        assert "XIII°" in html


# ──────────────────────────────────────────────────────────────────────
# 2. Engine name (moteur OCR) avec injection
# ──────────────────────────────────────────────────────────────────────


class TestEngineNameXSS:
    """Le nom de moteur peut venir d'un import HuggingFace ou d'une
    config utilisateur.  Doit être échappé dans tous les renderers
    qui l'affichent."""

    def test_engine_name_with_script_is_escaped(self, tmp_path: Path) -> None:
        from picarones.reports.html import ReportGenerator

        bench = _make_minimal_benchmark(
            corpus_name="test",
            engine_name="<script>alert('engine')</script>",
        )
        gen = ReportGenerator(bench)
        out = tmp_path / "report.html"
        gen.generate(out)

        html = out.read_text()
        assert "<script>alert('engine')</script>" not in html, (
            "Engine name XSS : un nom de moteur malicieux est exécuté."
        )


# ──────────────────────────────────────────────────────────────────────
# 3. Bandit B701 ne doit plus signaler autoescape=False
# ──────────────────────────────────────────────────────────────────────


class TestJinja2EnvIsAutoescaped:
    """Garde-fou contre la régression : ``_build_jinja_env`` doit
    retourner un Environment avec autoescape activé pour les
    extensions HTML/J2."""

    def test_env_has_autoescape_enabled_for_html(self) -> None:
        from picarones.reports.html.generator import _build_jinja_env

        env = _build_jinja_env()
        # autoescape doit être un Callable (select_autoescape) ou True,
        # pas False ni None.
        autoescape = env.autoescape
        assert autoescape, (
            f"Jinja2 Environment.autoescape={autoescape!r} — XSS exposé."
        )
        # Si c'est une fonction (select_autoescape), elle doit
        # retourner True pour les HTML/J2.
        if callable(autoescape):
            assert autoescape("base.html.j2"), (
                "select_autoescape doit activer l'échappement pour .j2"
            )
            assert autoescape("any.html"), (
                "select_autoescape doit activer l'échappement pour .html"
            )
