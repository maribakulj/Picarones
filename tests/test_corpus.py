"""Tests unitaires pour picarones.core.corpus."""

import pytest
from pathlib import Path

from picarones.core.corpus import load_corpus_from_directory, Corpus, Document


@pytest.fixture
def sample_corpus_dir(tmp_path: Path) -> Path:
    """Crée un mini-corpus temporaire avec 3 paires image/GT."""
    images = [
        ("page_001.png", "La première page du document médiéval."),
        ("page_002.png", "Deuxième folio avec des abréviations."),
        ("page_003.png", "Fin du manuscrit avec colophon."),
    ]
    for filename, gt_text in images:
        # Image factice (1×1 PNG valide)
        image_path = tmp_path / filename
        image_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        gt_path = tmp_path / (Path(filename).stem + ".gt.txt")
        gt_path.write_text(gt_text, encoding="utf-8")
    return tmp_path


class TestLoadCorpusFromDirectory:
    def test_loads_correct_count(self, sample_corpus_dir):
        corpus = load_corpus_from_directory(sample_corpus_dir)
        assert len(corpus) == 3

    def test_corpus_name_defaults_to_dir_name(self, sample_corpus_dir):
        corpus = load_corpus_from_directory(sample_corpus_dir)
        assert corpus.name == sample_corpus_dir.name

    def test_corpus_name_can_be_set(self, sample_corpus_dir):
        corpus = load_corpus_from_directory(sample_corpus_dir, name="Mon corpus test")
        assert corpus.name == "Mon corpus test"

    def test_document_ids(self, sample_corpus_dir):
        corpus = load_corpus_from_directory(sample_corpus_dir)
        ids = {doc.doc_id for doc in corpus}
        assert "page_001" in ids
        assert "page_002" in ids
        assert "page_003" in ids

    def test_ground_truth_content(self, sample_corpus_dir):
        corpus = load_corpus_from_directory(sample_corpus_dir)
        doc = next(d for d in corpus if d.doc_id == "page_001")
        assert "médiéval" in doc.ground_truth

    def test_source_path_set(self, sample_corpus_dir):
        corpus = load_corpus_from_directory(sample_corpus_dir)
        assert corpus.source_path == str(sample_corpus_dir)

    def test_nonexistent_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_corpus_from_directory(tmp_path / "inexistant")

    def test_directory_without_gt_raises(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"fake")
        with pytest.raises(ValueError):
            load_corpus_from_directory(tmp_path)

    def test_ignores_images_without_gt(self, sample_corpus_dir, tmp_path):
        # Copie le corpus et ajoute une image sans GT
        import shutil
        dest = tmp_path / "corpus2"
        shutil.copytree(sample_corpus_dir, dest)
        (dest / "orphan.png").write_bytes(b"fake")
        corpus = load_corpus_from_directory(dest)
        assert len(corpus) == 3  # L'image orpheline est ignorée

    def test_stats_computed(self, sample_corpus_dir):
        corpus = load_corpus_from_directory(sample_corpus_dir)
        stats = corpus.stats
        assert stats["document_count"] == 3
        assert stats["gt_length_min"] > 0


class TestCorpusIteration:
    def test_iterable(self, sample_corpus_dir):
        corpus = load_corpus_from_directory(sample_corpus_dir)
        docs = list(corpus)
        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)

    def test_repr(self, sample_corpus_dir):
        corpus = load_corpus_from_directory(sample_corpus_dir)
        r = repr(corpus)
        assert "Corpus" in r
        assert "3" in r
