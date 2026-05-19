"""Rendu JSON canonique des résultats de benchmark

API publique :

- ``JsonReportRenderer.render(run_result) -> str`` : document JSON
  consolidé, sérialisation déterministe (clés triées, indent=2,
  Unicode préservé).
"""

from __future__ import annotations

from picarones.reports.json.render import JsonReportRenderer

__all__ = ["JsonReportRenderer"]
