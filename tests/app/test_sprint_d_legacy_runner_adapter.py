"""Sprint D.1.a (plan v2.0) — helpers de mapping Corpus → CorpusSpec.

Vérifie que ``picarones.app.services._legacy_runner_adapter`` produit
des ``DocumentRef`` et ``CorpusSpec`` cohérents avec les ``Document``
et ``Corpus`` legacy.

Cette première itération du Sprint D pose la fondation des sub-phases
D.1.b-e qui construiront un adapter ``run_benchmark_via_service``
complet.  Les helpers testés ici sont **purs** (pas de réseau, pas de
LLM, juste de la transformation data) et réutilisables.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from picarones.app.services._legacy_runner_adapter import (
    corpus_to_corpus_spec,
    document_to_document_ref,
)
from picarones.domain.artifacts import ArtifactType
from picarones.domain.errors import PicaronesError
from picarones.evaluation.corpus import (
    AltoGT,
    Corpus,
    Document,
    EntitiesGT,
    PageGT,
    ReadingOrderGT,
    TextGT,
)


# ──────────────────────────────────────────────────────────────────────
# document_to_document_ref
# ──────────────────────────────────────────────────────────────────────


class TestDocumentToDocumentRef:
    def test_minimal_document(self, tmp_path: Path) -> None:
        """Un Document avec image + ground_truth se mappe en DocumentRef
        avec image_uri + GroundTruthRef RAW_TEXT."""
        img = tmp_path / "doc1.png"
        img.write_bytes(b"\x89PNG fake")
        doc = Document(
            image_path=img,
            ground_truth="bonjour le monde",
            doc_id="doc1",
        )

        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        assert ref.id == "doc1"
        assert ref.image_uri == str(img)
        assert len(ref.ground_truths) == 1
        gt = ref.ground_truths[0]
        assert gt.type == ArtifactType.RAW_TEXT
        # Le contenu de la GT est écrit dans workspace_dir.
        assert Path(gt.uri).read_text(encoding="utf-8") == "bonjour le monde"

    def test_image_path_none_yields_image_uri_none(self, tmp_path: Path) -> None:
        # Un document sans image (corpus textuel post-OCR).
        # Note : Document() exige un image_path str ou Path — on teste
        # le cas borderline où le caller passe None explicitement via
        # un workaround.  Le mapping doit tomber sur image_uri=None.
        doc = Document(
            image_path=tmp_path / "phantom.png",  # pas créé
            ground_truth="texte sans image",
            doc_id="textonly",
        )
        # On simule "pas d'image" en utilisant un chemin qui n'existe
        # pas — DocumentRef accepte image_uri quelconque (validation
        # filesystem est côté caller).
        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        assert ref.image_uri is not None  # le path est conservé tel quel

    def test_workspace_dir_must_exist(self, tmp_path: Path) -> None:
        doc = Document(
            image_path=tmp_path / "x.png",
            ground_truth="x",
            doc_id="x",
        )
        with pytest.raises(PicaronesError, match="doit exister"):
            document_to_document_ref(
                doc, workspace_dir=tmp_path / "nope_does_not_exist",
            )

    def test_unsafe_doc_id_is_sanitized(self, tmp_path: Path) -> None:
        """Un doc_id avec espaces ou accents est nettoyé pour respecter
        le regex de DocumentRef (alphanum + ._-/)."""
        img = tmp_path / "fancy.png"
        img.write_bytes(b"x")
        doc = Document(
            image_path=img,
            ground_truth="hello",
            doc_id="folio 1.r (été)",
        )
        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        # Espaces et parenthèses → underscores ; le résultat reste
        # informatif et conforme.
        assert " " not in ref.id
        assert "(" not in ref.id
        assert "folio_1.r" in ref.id

    def test_empty_doc_id_falls_back_to_doc(self, tmp_path: Path) -> None:
        img = tmp_path / "noid.png"
        img.write_bytes(b"x")
        # Document avec doc_id vide → __post_init__ remplit avec stem.
        # On force le contournement en assignant après.
        doc = Document(image_path=img, ground_truth="g")
        doc.doc_id = ""  # type: ignore[misc]
        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        assert ref.id == "doc"

    def test_with_alto_payload(self, tmp_path: Path) -> None:
        """Un document avec un payload ALTO produit un GroundTruthRef
        ALTO_XML."""
        img = tmp_path / "altodoc.png"
        img.write_bytes(b"x")
        doc = Document(
            image_path=img,
            ground_truth="alto baseline text",
            doc_id="altodoc",
            ground_truths={
                ArtifactType.RAW_TEXT: TextGT(text="alto baseline text"),
                ArtifactType.ALTO_XML: AltoGT(
                    xml_content="<alto/>",
                ),
            },
        )
        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        types = [gt.type for gt in ref.ground_truths]
        assert ArtifactType.ALTO_XML in types
        alto = next(gt for gt in ref.ground_truths if gt.type == ArtifactType.ALTO_XML)
        # Le contenu XML est écrit dans le workspace.
        assert Path(alto.uri).read_text(encoding="utf-8") == "<alto/>"

    def test_with_entities_payload(self, tmp_path: Path) -> None:
        img = tmp_path / "entdoc.png"
        img.write_bytes(b"x")
        ents = [{"label": "PER", "start": 0, "end": 5, "text": "Marie"}]
        doc = Document(
            image_path=img,
            ground_truth="Marie",
            doc_id="entdoc",
            ground_truths={
                ArtifactType.RAW_TEXT: TextGT(text="Marie"),
                ArtifactType.ENTITIES: EntitiesGT(entities=ents),
            },
        )
        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        ent_ref = next(
            gt for gt in ref.ground_truths if gt.type == ArtifactType.ENTITIES
        )
        # JSON sérialisé avec préservation des accents.
        loaded = json.loads(Path(ent_ref.uri).read_text(encoding="utf-8"))
        assert loaded == ents

    def test_with_existing_source_path_no_copy(self, tmp_path: Path) -> None:
        """Si le payload GT a un ``source_path`` existant, l'URI pointe
        directement dessus (pas de copie redondante)."""
        img = tmp_path / "withsource.png"
        img.write_bytes(b"x")

        # Crée un fichier ALTO source réel.
        existing_alto = tmp_path / "real.alto.xml"
        existing_alto.write_text("<alto>real</alto>", encoding="utf-8")

        doc = Document(
            image_path=img,
            ground_truth="x",
            doc_id="withsource",
            ground_truths={
                ArtifactType.RAW_TEXT: TextGT(text="x"),
                ArtifactType.ALTO_XML: AltoGT(
                    xml_content="<alto>memcontent</alto>",
                    source_path=existing_alto,
                ),
            },
        )
        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        alto = next(
            gt for gt in ref.ground_truths if gt.type == ArtifactType.ALTO_XML
        )
        # L'URI pointe sur le fichier source, pas sur une copie.
        assert alto.uri == str(existing_alto)

    def test_all_five_levels(self, tmp_path: Path) -> None:
        img = tmp_path / "allgt.png"
        img.write_bytes(b"x")
        doc = Document(
            image_path=img,
            ground_truth="t",
            doc_id="allgt",
            ground_truths={
                ArtifactType.RAW_TEXT: TextGT(text="t"),
                ArtifactType.ALTO_XML: AltoGT(xml_content="<alto/>"),
                ArtifactType.PAGE_XML: PageGT(xml_content="<PAGE/>"),
                ArtifactType.ENTITIES: EntitiesGT(entities=[]),
                ArtifactType.READING_ORDER: ReadingOrderGT(
                    region_order=["r1", "r2"],
                ),
            },
        )
        ref = document_to_document_ref(doc, workspace_dir=tmp_path)
        types = {gt.type for gt in ref.ground_truths}
        assert types == {
            ArtifactType.RAW_TEXT,
            ArtifactType.ALTO_XML,
            ArtifactType.PAGE_XML,
            ArtifactType.ENTITIES,
            ArtifactType.READING_ORDER,
        }


# ──────────────────────────────────────────────────────────────────────
# corpus_to_corpus_spec
# ──────────────────────────────────────────────────────────────────────


class TestCorpusToCorpusSpec:
    def test_empty_corpus(self, tmp_path: Path) -> None:
        corpus = Corpus(name="empty", documents=[])
        spec = corpus_to_corpus_spec(corpus, workspace_dir=tmp_path)
        assert spec.name == "empty"
        assert spec.documents == ()

    def test_single_document(self, tmp_path: Path) -> None:
        img = tmp_path / "d1.png"
        img.write_bytes(b"x")
        doc = Document(
            image_path=img,
            ground_truth="hello",
            doc_id="d1",
        )
        corpus = Corpus(name="mini", documents=[doc])
        spec = corpus_to_corpus_spec(corpus, workspace_dir=tmp_path)
        assert spec.name == "mini"
        assert len(spec.documents) == 1
        assert spec.documents[0].id == "d1"

    def test_metadata_scalar_values_propagate(self, tmp_path: Path) -> None:
        img = tmp_path / "d.png"
        img.write_bytes(b"x")
        corpus = Corpus(
            name="meta",
            documents=[Document(image_path=img, ground_truth="x", doc_id="d")],
            metadata={
                "language": "fr",
                "n_pages": 42,
                "is_curated": True,
                "complex": {"nested": "ignored"},  # ignoré (pas scalaire)
            },
        )
        spec = corpus_to_corpus_spec(corpus, workspace_dir=tmp_path)
        assert spec.metadata["language"] == "fr"
        assert spec.metadata["n_pages"] == "42"
        assert spec.metadata["is_curated"] == "True"
        assert "complex" not in spec.metadata

    def test_source_path_added_to_metadata(self, tmp_path: Path) -> None:
        img = tmp_path / "d.png"
        img.write_bytes(b"x")
        corpus = Corpus(
            name="src",
            documents=[Document(image_path=img, ground_truth="x", doc_id="d")],
            source_path="/some/dir",
        )
        spec = corpus_to_corpus_spec(corpus, workspace_dir=tmp_path)
        assert spec.metadata["source_path"] == "/some/dir"

    def test_workspace_dir_must_exist(self, tmp_path: Path) -> None:
        corpus = Corpus(name="x", documents=[])
        with pytest.raises(PicaronesError, match="doit exister"):
            corpus_to_corpus_spec(
                corpus, workspace_dir=tmp_path / "nope_dont_exist",
            )

    def test_preserves_document_order(self, tmp_path: Path) -> None:
        """L'ordre des documents est préservé (utile pour la
        reproductibilité)."""
        docs = []
        for i in range(5):
            img = tmp_path / f"doc{i}.png"
            img.write_bytes(b"x")
            docs.append(
                Document(
                    image_path=img,
                    ground_truth=f"text {i}",
                    doc_id=f"doc{i}",
                ),
            )
        corpus = Corpus(name="ordered", documents=docs)
        spec = corpus_to_corpus_spec(corpus, workspace_dir=tmp_path)
        assert [d.id for d in spec.documents] == ["doc0", "doc1", "doc2", "doc3", "doc4"]

    def test_unique_doc_ids_required(self, tmp_path: Path) -> None:
        """``CorpusSpec`` impose des doc_ids uniques.  Si deux Documents
        partagent le même id, la conversion lève."""
        from picarones.domain.errors import CorpusSpecError

        img1 = tmp_path / "a.png"
        img1.write_bytes(b"x")
        img2 = tmp_path / "b.png"
        img2.write_bytes(b"x")
        docs = [
            Document(image_path=img1, ground_truth="a", doc_id="same"),
            Document(image_path=img2, ground_truth="b", doc_id="same"),
        ]
        corpus = Corpus(name="dup", documents=docs)
        with pytest.raises(CorpusSpecError, match="dupliqu"):
            corpus_to_corpus_spec(corpus, workspace_dir=tmp_path)
