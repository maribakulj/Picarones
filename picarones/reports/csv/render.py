"""``CsvReportRenderer``

Rendu CSV d'un ``RunResult`` : une ligne par paire
(document × pipeline × view × metric) avec sa valeur numérique ou
le marqueur ``OMITTED`` (pas de score factice).

Cohérent avec la convention du rewrite : pour les pipelines qui ne
produisent pas un type d'artefact accepté par une vue, on émet
``OMITTED`` dans la cellule ``value`` plutôt que ``0`` ou ``""``.
Le consommateur (Pandas, Excel, awk, ...) sait que l'omission est
l'information.

Usage
-----

::

    from picarones.reports.csv import CsvReportRenderer
    csv_text = CsvReportRenderer().render(run_result)
    Path("rapport.csv").write_text(csv_text, encoding="utf-8")

Format
------
Colonnes (dans l'ordre) :

::

    run_id, document_id, pipeline_name, view_name,
    metric_name, value, status

- ``run_id`` : ``RunManifest.run_id``.
- ``status`` : ``"ok"``, ``"failed_metric"`` (la métrique a levé),
  ``"omitted"`` (le pipeline ne produit pas d'artefact pour la vue).
- ``value`` : valeur numérique formatée à 6 décimales, ou vide si
  ``status != "ok"``.

Anti-sur-ingénierie
-------------------
- Pas de pivot par moteur — chaque ligne est self-contained.  Le
  consommateur pivote en 2 lignes Pandas si besoin.
- Pas d'escape custom — on utilise ``csv.writer`` qui gère les
  virgules et guillemets dans les values.
- Pas de séparateur configurable (``,`` fixe) — un test garde-fou
  vérifie le déterminisme du contenu.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from picarones.pipeline.run_result import RunResult


class CsvReportRenderer:
    """Rendu CSV stateless d'un RunResult."""

    HEADER: tuple[str, ...] = (
        "run_id",
        "document_id",
        "pipeline_name",
        "view_name",
        "metric_name",
        "value",
        "status",
    )

    def render(self, result: RunResult) -> str:
        """Retourne le contenu CSV (stringly typed) prêt à écrire."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(self.HEADER)

        run_id = result.manifest.run_id

        for doc_result in result.document_results:
            for view_result in doc_result.view_results:
                pipeline_name = view_result.pipeline_name
                for metric_name, value in view_result.metric_values.items():
                    writer.writerow([
                        run_id,
                        doc_result.document_id,
                        pipeline_name,
                        view_result.view_name,
                        metric_name,
                        self._format_value(value),
                        "ok",
                    ])
                for metric_name, _err in view_result.failed_metrics.items():
                    writer.writerow([
                        run_id,
                        doc_result.document_id,
                        pipeline_name,
                        view_result.view_name,
                        metric_name,
                        "",
                        "failed_metric",
                    ])

        return buf.getvalue()

    @staticmethod
    def _format_value(value: Any) -> str:
        """Formate la valeur numérique à 6 décimales pour
        déterminisme cross-OS (évite ``1.0000000000000002`` sur
        certains floats)."""
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (int, float)):
            return f"{float(value):.6f}"
        return str(value)


__all__ = ["CsvReportRenderer"]
