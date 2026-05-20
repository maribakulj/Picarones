"""``JsonReportRenderer``

Rendu JSON canonique d'un ``RunResult`` : représentation hiérarchique
sérialisable, déterministe (clés triées, indent=2, ensure_ascii=False),
prête à être archivée ou consommée par un client tiers.

Différent des trois fichiers persistés par ``BenchmarkService.persist``
(``run_manifest.json`` + 3 JSONL) qui sont **streamables** : ce
renderer produit un **document unique** consolidé.

Usage
-----

::

    from picarones.reports.json import JsonReportRenderer
    json_text = JsonReportRenderer().render(run_result)
    Path("rapport.json").write_text(json_text, encoding="utf-8")

Structure
---------

::

    {
      "run_manifest": { ... },
      "documents": [
        {
          "document_id": "d1",
          "pipeline_results": [ {...} ],
          "view_results": [ {...} ]
        },
        ...
      ]
    }

Anti-sur-ingénierie
-------------------
- Pas de schéma JSON publié — pydantic ``model_dump_json`` est
  l'autorité.  La stabilité sera tagguée à la livraison BnF.
- Pas de séparateurs custom — JSON standard.
- Pas de pretty mode configurable — toujours indent=2 pour la
  lisibilité humaine ; un caller qui veut compact appelle
  ``json.dumps(json.loads(out))``.
"""

from __future__ import annotations

import json

from picarones.pipeline.run_result import RunResult


class JsonReportRenderer:
    """Rendu JSON consolidé d'un RunResult."""

    def render(self, result: RunResult) -> str:
        """Retourne un document JSON canonique du run.

        Sérialisation déterministe : ``sort_keys=True``, ``indent=2``,
        ``ensure_ascii=False``.  Le caller peut écrire directement le
        retour via ``Path.write_text(..., encoding="utf-8")``.
        """
        document = self._build_document(result)
        return json.dumps(
            document,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )

    def _build_document(self, result: RunResult) -> dict:
        """Construit le dict canonique avant sérialisation.

        ``model_dump(mode="json")`` produit directement un dict
        JSON-serializable (datetime → ISO string, enum → value,
        etc.).  Préférable au round-trip
        ``model_dump_json() → loads → dumps`` qui est ~10× plus coûteux
        sur des manifests volumineux.
        """
        return {
            "run_manifest": result.manifest.model_dump(mode="json"),
            "documents": [
                {
                    "document_id": dr.document_id,
                    "pipeline_results": [
                        pr.model_dump(mode="json")
                        for pr in dr.pipeline_results
                    ],
                    "view_results": [
                        vr.model_dump(mode="json")
                        for vr in dr.view_results
                    ],
                }
                for dr in result.document_results
            ],
        }


__all__ = ["JsonReportRenderer"]
