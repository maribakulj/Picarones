"""Tests Sprint 32 — GT multi-niveaux (Phase 0.1 du plan d'évolution).

Vérifie :

1. Rétrocompatibilité stricte : un corpus historique (image + .gt.txt
   uniquement) se charge exactement comme avant et expose la même API
   (``doc.ground_truth: str``).
2. Détection automatique des niveaux additionnels : ``.gt.alto.xml``,
   ``.gt.page.xml``, ``.gt.entities.json``, ``.gt.reading_order.json``.
3. Couverture partielle : un corpus mixte où seuls certains documents
   ont l'ALTO doit refléter cette couverture dans
   ``Corpus.gt_level_coverage()``.
4. Synchronisation TEXT entre champ ``ground_truth`` et
   ``ground_truths[GTLevel.TEXT]`` dans les deux sens.
5. Robustesse : un fichier JSON cassé est dégradé en warning, le
   document reste chargé avec les niveaux qui ont fonctionné.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picarones.evaluation.corpus import (
    AltoGT,
    Document,
    EntitiesGT,
    GT_SUFFIXES,
    GTLevel,
    PageGT,
    ReadingOrderGT,
    TextGT,
    load_corpus_from_directory,
)


# Mini-PNG 1×1 valide réutilisé dans les tests
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _write_pair(directory: Path, stem: str, gt_text: str) -> Path:
    """Écrit une paire image + .gt.txt classique."""
    image = directory / f"{stem}.png"
    image.write_bytes(_TINY_PNG)
    (directory / f"{stem}.gt.txt").write_text(gt_text, encoding="utf-8")
    return image


# ──────────────────────────────────────────────────────────────────────────
# 1. Rétrocompatibilité stricte
# ──────────────────────────────────────────────────────────────────────────


class TestBackwardCompat:
    def test_text_only_corpus_loads_unchanged(self, tmp_path: Path) -> None:
        _write_pair(tmp_path, "doc_001", "Première page.")
        _write_pair(tmp_path, "doc_002", "Deuxième page.")

        corpus = load_corpus_from_directory(tmp_path)

        assert len(corpus) == 2
        for doc in corpus:
            # API historique : ground_truth: str
            assert isinstance(doc.ground_truth, str)
            assert doc.ground_truth  # non vide
            # Le niveau TEXT est automatiquement peuplé
            assert doc.has_gt(GTLevel.TEXT)
            assert not doc.has_gt(GTLevel.ALTO)
            assert not doc.has_gt(GTLevel.PAGE)

    def test_document_dataclass_default_is_text_only(self) -> None:
        doc = Document(image_path=Path("/tmp/x.png"), ground_truth="abc")

        assert doc.ground_truth == "abc"
        assert doc.gt_levels == {GTLevel.TEXT}
        text_payload = doc.get_gt(GTLevel.TEXT)
        assert isinstance(text_payload, TextGT)
        assert text_payload.text == "abc"

    def test_document_construction_via_ground_truths_dict(self) -> None:
        """Construction par le nouveau format : le champ str est synchronisé."""
        doc = Document(
            image_path=Path("/tmp/x.png"),
            ground_truths={GTLevel.TEXT: TextGT(text="hello")},
        )
        # Le post-init renseigne ground_truth depuis le dict
        assert doc.ground_truth == "hello"

    def test_no_extra_levels_means_no_change_in_api(self, tmp_path: Path) -> None:
        """Un corpus sans fichier ALTO/PAGE/JSON ne doit jamais lever."""
        _write_pair(tmp_path, "x", "y")
        corpus = load_corpus_from_directory(tmp_path)
        assert corpus.available_gt_levels == {GTLevel.TEXT}


# ──────────────────────────────────────────────────────────────────────────
# 2. Détection automatique des niveaux additionnels
# ──────────────────────────────────────────────────────────────────────────


_ALTO_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Layout><Page><PrintSpace>
    <TextBlock ID="block_1"><TextLine ID="line_1">
      <String CONTENT="Bonjour"/>
    </TextLine></TextBlock>
  </PrintSpace></Page></Layout>
</alto>
"""

_PAGE_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page><TextRegion id="r1"><TextLine id="l1">
    <TextEquiv><Unicode>Salut</Unicode></TextEquiv>
  </TextLine></TextRegion></Page>
</PcGts>
"""


class TestExtraLevelsDetection:
    def test_alto_detected(self, tmp_path: Path) -> None:
        _write_pair(tmp_path, "doc", "Bonjour")
        (tmp_path / f"doc{GT_SUFFIXES[GTLevel.ALTO]}").write_text(_ALTO_SAMPLE, encoding="utf-8")

        corpus = load_corpus_from_directory(tmp_path)
        doc = corpus.documents[0]

        assert doc.has_gt(GTLevel.ALTO)
        alto = doc.get_gt(GTLevel.ALTO)
        assert isinstance(alto, AltoGT)
        assert "TextBlock" in alto.xml_content
        assert alto.source_path is not None

    def test_page_detected(self, tmp_path: Path) -> None:
        _write_pair(tmp_path, "doc", "Salut")
        (tmp_path / f"doc{GT_SUFFIXES[GTLevel.PAGE]}").write_text(_PAGE_SAMPLE, encoding="utf-8")

        corpus = load_corpus_from_directory(tmp_path)
        doc = corpus.documents[0]

        page = doc.get_gt(GTLevel.PAGE)
        assert isinstance(page, PageGT)
        assert "TextRegion" in page.xml_content

    def test_entities_detected_object_form(self, tmp_path: Path) -> None:
        _write_pair(tmp_path, "doc", "Marie de Bourgogne en 1477.")
        entities = {
            "entities": [
                {"label": "PER", "start": 0, "end": 17, "text": "Marie de Bourgogne"},
                {"label": "DATE", "start": 21, "end": 25, "text": "1477"},
            ]
        }
        (tmp_path / f"doc{GT_SUFFIXES[GTLevel.ENTITIES]}").write_text(
            json.dumps(entities), encoding="utf-8"
        )

        corpus = load_corpus_from_directory(tmp_path)
        doc = corpus.documents[0]

        ent = doc.get_gt(GTLevel.ENTITIES)
        assert isinstance(ent, EntitiesGT)
        assert len(ent.entities) == 2
        assert ent.entities[0]["label"] == "PER"

    def test_entities_detected_array_form(self, tmp_path: Path) -> None:
        """Le loader accepte aussi un tableau JSON brut."""
        _write_pair(tmp_path, "doc", "Texte.")
        ent_data = [{"label": "MISC", "start": 0, "end": 5, "text": "Texte"}]
        (tmp_path / f"doc{GT_SUFFIXES[GTLevel.ENTITIES]}").write_text(
            json.dumps(ent_data), encoding="utf-8"
        )

        corpus = load_corpus_from_directory(tmp_path)
        ent = corpus.documents[0].get_gt(GTLevel.ENTITIES)
        assert isinstance(ent, EntitiesGT)
        assert ent.entities[0]["label"] == "MISC"

    def test_reading_order_detected(self, tmp_path: Path) -> None:
        _write_pair(tmp_path, "doc", "Multi-colonnes.")
        ro = {"region_order": ["r_main", "r_marginalia", "r_footer"]}
        (tmp_path / f"doc{GT_SUFFIXES[GTLevel.READING_ORDER]}").write_text(
            json.dumps(ro), encoding="utf-8"
        )

        corpus = load_corpus_from_directory(tmp_path)
        ro_payload = corpus.documents[0].get_gt(GTLevel.READING_ORDER)
        assert isinstance(ro_payload, ReadingOrderGT)
        assert ro_payload.region_order == ["r_main", "r_marginalia", "r_footer"]

    def test_all_four_extra_levels_simultaneously(self, tmp_path: Path) -> None:
        _write_pair(tmp_path, "doc", "Texte complet.")
        (tmp_path / f"doc{GT_SUFFIXES[GTLevel.ALTO]}").write_text(_ALTO_SAMPLE, encoding="utf-8")
        (tmp_path / f"doc{GT_SUFFIXES[GTLevel.PAGE]}").write_text(_PAGE_SAMPLE, encoding="utf-8")
        (tmp_path / f"doc{GT_SUFFIXES[GTLevel.ENTITIES]}").write_text(
            json.dumps([{"label": "X", "start": 0, "end": 1, "text": "T"}]), encoding="utf-8"
        )
        (tmp_path / f"doc{GT_SUFFIXES[GTLevel.READING_ORDER]}").write_text(
            json.dumps(["r1"]), encoding="utf-8"
        )

        doc = load_corpus_from_directory(tmp_path).documents[0]
        assert doc.gt_levels == {
            GTLevel.TEXT,
            GTLevel.ALTO,
            GTLevel.PAGE,
            GTLevel.ENTITIES,
            GTLevel.READING_ORDER,
        }


# ──────────────────────────────────────────────────────────────────────────
# 3. Couverture partielle (corpus mixte)
# ──────────────────────────────────────────────────────────────────────────


class TestPartialCoverage:
    def test_partial_alto_coverage(self, tmp_path: Path) -> None:
        """3 documents, seul le premier porte un ALTO."""
        _write_pair(tmp_path, "doc_001", "Premier")
        _write_pair(tmp_path, "doc_002", "Deuxième")
        _write_pair(tmp_path, "doc_003", "Troisième")
        (tmp_path / f"doc_001{GT_SUFFIXES[GTLevel.ALTO]}").write_text(
            _ALTO_SAMPLE, encoding="utf-8"
        )

        corpus = load_corpus_from_directory(tmp_path)

        coverage = corpus.gt_level_coverage()
        assert coverage[GTLevel.TEXT] == 3
        assert coverage[GTLevel.ALTO] == 1
        # available_gt_levels = union sur tout le corpus
        assert corpus.available_gt_levels == {GTLevel.TEXT, GTLevel.ALTO}
        # Mais seul doc_001 expose ALTO
        doc_001 = next(d for d in corpus if d.doc_id == "doc_001")
        doc_002 = next(d for d in corpus if d.doc_id == "doc_002")
        assert doc_001.has_gt(GTLevel.ALTO)
        assert not doc_002.has_gt(GTLevel.ALTO)

    def test_stats_exposes_coverage(self, tmp_path: Path) -> None:
        _write_pair(tmp_path, "a", "x")
        _write_pair(tmp_path, "b", "y")
        (tmp_path / f"a{GT_SUFFIXES[GTLevel.ALTO]}").write_text(_ALTO_SAMPLE, encoding="utf-8")

        stats = load_corpus_from_directory(tmp_path).stats
        assert stats["gt_level_coverage"]["text"] == 2
        assert stats["gt_level_coverage"]["alto"] == 1


# ──────────────────────────────────────────────────────────────────────────
# 4. Synchronisation bidirectionnelle TEXT
# ──────────────────────────────────────────────────────────────────────────


class TestTextSync:
    def test_str_to_dict_sync(self) -> None:
        doc = Document(image_path=Path("/tmp/x.png"), ground_truth="aaa")
        text_gt = doc.get_gt(GTLevel.TEXT)
        assert isinstance(text_gt, TextGT)
        assert text_gt.text == "aaa"

    def test_dict_to_str_sync(self) -> None:
        doc = Document(
            image_path=Path("/tmp/x.png"),
            ground_truths={GTLevel.TEXT: TextGT(text="bbb")},
        )
        assert doc.ground_truth == "bbb"

    def test_both_provided_keeps_str(self) -> None:
        """Si les deux sont fournis, le champ str est préservé tel quel —
        le dict reste la source pour les autres niveaux."""
        doc = Document(
            image_path=Path("/tmp/x.png"),
            ground_truth="canon",
            ground_truths={GTLevel.TEXT: TextGT(text="autre")},
        )
        # Le champ str fourni explicitement n'est pas écrasé
        assert doc.ground_truth == "canon"


# ──────────────────────────────────────────────────────────────────────────
# 5. Robustesse — JSON cassé
# ──────────────────────────────────────────────────────────────────────────


class TestRobustness:
    def test_broken_entities_json_is_warning_not_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_pair(tmp_path, "doc", "Texte.")
        (tmp_path / f"doc{GT_SUFFIXES[GTLevel.ENTITIES]}").write_text(
            "{ ceci n'est pas du JSON", encoding="utf-8"
        )

        with caplog.at_level("WARNING", logger="picarones.evaluation.corpus"):
            corpus = load_corpus_from_directory(tmp_path)

        # Le document reste chargé avec son niveau TEXT
        doc = corpus.documents[0]
        assert doc.has_gt(GTLevel.TEXT)
        assert not doc.has_gt(GTLevel.ENTITIES)
        # Et un warning explicite a été émis (cf. règle CLAUDE.md)
        assert any("entités" in rec.message.lower() for rec in caplog.records)

    def test_unexpected_json_format_is_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_pair(tmp_path, "doc", "Texte.")
        # JSON valide mais format inattendu (pas dict avec "entities", pas liste)
        (tmp_path / f"doc{GT_SUFFIXES[GTLevel.ENTITIES]}").write_text(
            json.dumps({"foo": "bar"}), encoding="utf-8"
        )

        with caplog.at_level("WARNING", logger="picarones.evaluation.corpus"):
            corpus = load_corpus_from_directory(tmp_path)

        assert not corpus.documents[0].has_gt(GTLevel.ENTITIES)
        assert any("format" in rec.message.lower() for rec in caplog.records)
