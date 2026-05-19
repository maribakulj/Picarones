"""Rendu CSV des résultats de benchmark

API publique :

- ``CsvReportRenderer.render(run_result) -> str`` : produit un CSV
  prêt à écrire sur disque.

Format : une ligne par (document × pipeline × view × metric).
``OMITTED`` est explicite — pas de score factice 0.
"""

from __future__ import annotations

from picarones.reports.csv.render import CsvReportRenderer

__all__ = ["CsvReportRenderer"]
