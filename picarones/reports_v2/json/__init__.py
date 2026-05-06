"""Rendu JSON canonique des résultats de benchmark — Sprint A14-S43.

API publique :

- ``JsonReportRenderer.render(run_result) -> str`` : document JSON
  consolidé, sérialisation déterministe (clés triées, indent=2,
  Unicode préservé).
"""

from __future__ import annotations

from picarones.reports_v2.json.render import JsonReportRenderer

__all__ = ["JsonReportRenderer"]
