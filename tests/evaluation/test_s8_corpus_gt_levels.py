"""Sprint S8.7 — couverture des propriétés ``level`` des
payloads GT (Ground Truth) dans
``picarones/evaluation/corpus.py``.

Cible (avant) : lignes 92, 106, 118, 136, 152 — les 5 propriétés
``level`` (une par type de GT : Text, Alto, Page, Entities,
ReadingOrder) qui mappent vers la valeur ``ArtifactType``
correspondante.

Ces propriétés sont triviales mais critiques : elles servent
de discriminant pour le routing des métriques par type de GT
(``compute_at_junction(level=ArtifactType.X)``).  Une régression
silencieuse (par ex. ``PageGT.level`` retourne accidentellement
``ALTO_XML``) ferait passer les calculs de PAGE par les
projecteurs ALTO sans erreur visible — c'est exactement le
genre de bug qu'un type tag explicite est censé empêcher.
"""

from __future__ import annotations

from picarones.domain.artifacts import ArtifactType
from picarones.evaluation.corpus import (
    AltoGT,
    EntitiesGT,
    PageGT,
    ReadingOrderGT,
    TextGT,
)


class TestGTLevelProperties:
    def test_text_gt_level(self) -> None:
        gt = TextGT(text="hello")
        assert gt.level == ArtifactType.RAW_TEXT

    def test_alto_gt_level(self) -> None:
        gt = AltoGT(xml_content="<alto/>")
        assert gt.level == ArtifactType.ALTO_XML

    def test_page_gt_level(self) -> None:
        gt = PageGT(xml_content="<PcGts/>")
        assert gt.level == ArtifactType.PAGE_XML

    def test_entities_gt_level(self) -> None:
        gt = EntitiesGT(entities=[])
        assert gt.level == ArtifactType.ENTITIES

    def test_reading_order_gt_level(self) -> None:
        gt = ReadingOrderGT(region_order=[])
        assert gt.level == ArtifactType.READING_ORDER

    def test_levels_are_distinct(self) -> None:
        """Garde-fou méta : les 5 levels doivent être 5 valeurs
        distinctes.  Si deux GT renvoient le même level, c'est un
        bug de routing (un payload sera traité comme l'autre)."""
        levels = {
            TextGT(text="").level,
            AltoGT(xml_content="").level,
            PageGT(xml_content="").level,
            EntitiesGT(entities=[]).level,
            ReadingOrderGT(region_order=[]).level,
        }
        assert len(levels) == 5
