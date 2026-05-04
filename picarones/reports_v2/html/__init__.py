"""Rendu HTML du rewrite ciblé.

API publique :

- :class:`HtmlReportRenderer` — produit un fichier HTML autonome
  depuis un ``RunResult`` (ou les 3 fichiers persistés par
  ``BenchmarkService.persist``).

Usage
-----

::

    from pathlib import Path
    from picarones.reports_v2.html import HtmlReportRenderer

    renderer = HtmlReportRenderer(lang="fr")
    html = renderer.render(run_result)
    Path("rapport.html").write_text(html, encoding="utf-8")
"""

from __future__ import annotations

from picarones.reports_v2.html.render import HtmlReportRenderer

__all__ = ["HtmlReportRenderer"]
