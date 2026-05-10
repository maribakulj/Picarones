"""Sprint D.2.b — reprise sur interruption (``partial_dir``) dans
``run_benchmark_via_service``.

Couvre :

- Helpers ``picarones.app.services.partial_store`` (chemin,
  sérialisation NDJSON, tolérance aux lignes corrompues).
- Comportement bout-en-bout de ``run_benchmark_via_service`` quand
  ``partial_dir`` est fourni :
  reprise depuis un partial existant, suppression à la fin d'un
  engine traité avec succès, isolation per-engine.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from picarones.adapters.ocr.base import BaseOCRAdapter
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.app.services.partial_store import (
    _delete_partial,
    _load_partial,
    _partial_path,
    _sanitize_filename,
    _save_partial_line,
)
from picarones.app.services.benchmark_runner import (
    run_benchmark_via_service,
)
from picarones.evaluation.benchmark_result import DocumentResult
from picarones.evaluation.corpus import Corpus, Document
from picarones.evaluation.metric_result import MetricsResult


# ──────────────────────────────────────────────────────────────────────
# Mocks
# ──────────────────────────────────────────────────────────────────────


class _MockOCR(BaseOCRAdapter):
    """Adapter canonique minimal pour les tests.

    Compat ergonomique avec le pattern legacy : un test peut faire
    ``ocr._run_ocr = lambda p: "..."`` après construction pour
    customiser la sortie ; le mock l'invoque depuis ``execute()``.
    Sans override, retourne ``"ocr text"`` par défaut.
    """

    def __init__(self, name: str = "mock_ocr") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def execute(self, inputs, params, context):
        from pathlib import Path

        out_dir = Path(context.workspace_uri)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{context.document_id}_mock.txt"
        runtime_override = getattr(self, "_run_ocr", None)
        if callable(runtime_override):
            text = runtime_override(out_path)
        else:
            text = "ocr text"
        out_path.write_text(text, encoding="utf-8")
        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:{self._name}:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
                uri=str(out_path),
            ),
        }


def _make_doc_result(doc_id: str, hyp: str = "h", cer: float = 0.1) -> DocumentResult:
    return DocumentResult(
        doc_id=doc_id,
        image_path=f"/tmp/{doc_id}.png",
        ground_truth="g",
        hypothesis=hyp,
        metrics=MetricsResult(
            cer=cer,
            cer_nfc=cer,
            cer_caseless=cer,
            wer=cer,
            wer_normalized=cer,
            mer=cer,
            wil=cer,
            reference_length=1,
            hypothesis_length=1,
        ),
        duration_seconds=0.5,
    )


# ──────────────────────────────────────────────────────────────────────
# 1. Helpers partial_store
# ──────────────────────────────────────────────────────────────────────


class TestSanitizeFilename:
    def test_keeps_word_chars_and_dash(self) -> None:
        assert _sanitize_filename("abc-123_def") == "abc-123_def"

    def test_replaces_special_chars(self) -> None:
        assert _sanitize_filename("a/b:c d") == "a_b_c_d"

    def test_truncates_to_64_chars(self) -> None:
        result = _sanitize_filename("a" * 100)
        assert len(result) == 64
        assert result == "a" * 64


class TestPartialPath:
    def test_uses_partial_dir(self, tmp_path: Path) -> None:
        path = _partial_path("corpus_x", "engine_y", tmp_path)
        assert path.parent == tmp_path
        assert "corpus_x" in path.name
        assert "engine_y" in path.name
        assert path.suffix == ".jsonl"

    def test_sanitizes_names_in_path(self, tmp_path: Path) -> None:
        path = _partial_path("c/orpus", "engine:a", tmp_path)
        # Pas de slash résiduel dans le filename — uniquement dans
        # le dirname (tmp_path).
        assert "/" not in path.name
        assert ":" not in path.name

    def test_none_partial_dir_falls_back_to_tempdir(self) -> None:
        import tempfile
        path = _partial_path("c", "e", None)
        assert path.parent == Path(tempfile.gettempdir())


class TestSaveAndLoad:
    def test_round_trip_single_result(self, tmp_path: Path) -> None:
        path = tmp_path / "r.jsonl"
        dr = _make_doc_result("doc1", hyp="hello", cer=0.05)

        _save_partial_line(path, dr)
        loaded = _load_partial(path)

        assert len(loaded) == 1
        assert loaded[0].doc_id == "doc1"
        assert loaded[0].hypothesis == "hello"
        assert loaded[0].metrics.cer == pytest.approx(0.05)

    def test_round_trip_preserves_optional_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "r.jsonl"
        dr = _make_doc_result("doc1")
        dr.ocr_intermediate = "intermediate"
        dr.pipeline_metadata = {"mode": "post_correction_texte"}

        _save_partial_line(path, dr)
        loaded = _load_partial(path)

        assert loaded[0].ocr_intermediate == "intermediate"
        assert loaded[0].pipeline_metadata == {"mode": "post_correction_texte"}

    def test_appends_multiple_results(self, tmp_path: Path) -> None:
        path = tmp_path / "r.jsonl"
        for i in range(3):
            _save_partial_line(path, _make_doc_result(f"doc{i}"))

        loaded = _load_partial(path)
        assert [d.doc_id for d in loaded] == ["doc0", "doc1", "doc2"]

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert _load_partial(path) == []

    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "nope.jsonl"
        assert _load_partial(path) == []

    def test_corrupted_line_is_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = tmp_path / "r.jsonl"
        # Une ligne valide + une corrompue + une valide.
        _save_partial_line(path, _make_doc_result("doc0"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write("not valid json\n")
        _save_partial_line(path, _make_doc_result("doc2"))

        with caplog.at_level("WARNING"):
            loaded = _load_partial(path)

        assert [d.doc_id for d in loaded] == ["doc0", "doc2"]

    def test_save_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "r.jsonl"
        _save_partial_line(path, _make_doc_result("doc0"))
        assert path.exists()

    def test_concurrent_writes_are_safe(self, tmp_path: Path) -> None:
        """Le lock module-level sérialise les appends — le fichier ne
        contient jamais une ligne tronquée même avec N threads."""
        path = tmp_path / "concurrent.jsonl"
        n_threads = 8
        per_thread = 10

        def writer(tid: int) -> None:
            for i in range(per_thread):
                _save_partial_line(path, _make_doc_result(f"t{tid}_d{i}"))

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        loaded = _load_partial(path)
        assert len(loaded) == n_threads * per_thread
        # Tous les doc_ids sont uniques et bien formés.
        assert len({d.doc_id for d in loaded}) == n_threads * per_thread


class TestDelete:
    def test_delete_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "r.jsonl"
        path.write_text("x\n", encoding="utf-8")
        _delete_partial(path)
        assert not path.exists()

    def test_delete_missing_file_is_noop(self, tmp_path: Path) -> None:
        path = tmp_path / "nope.jsonl"
        # Ne lève pas.
        _delete_partial(path)


# ──────────────────────────────────────────────────────────────────────
# 2. Resume bout-en-bout dans run_benchmark_via_service
# ──────────────────────────────────────────────────────────────────────


class TestResumeViaPartialDir:
    """Sprint D.2.b — quand ``partial_dir`` est fourni,
    ``run_benchmark_via_service`` reprend depuis l'éventuel partial
    existant et persiste chaque ``DocumentResult`` au fil de l'eau."""

    def _make_corpus(self, tmp_path: Path, n: int = 3) -> Corpus:
        docs = []
        for i in range(n):
            img = tmp_path / f"doc{i}.png"
            img.write_bytes(b"x")
            docs.append(Document(
                image_path=img,
                ground_truth=f"gt {i}",
                doc_id=f"doc{i}",
            ))
        return Corpus(name="resume_test", documents=docs)

    def test_fresh_run_deletes_partial_on_success(self, tmp_path: Path) -> None:
        partial_dir = tmp_path / "partials"
        corpus = self._make_corpus(tmp_path, n=2)
        ocr = _MockOCR(name="resumable")
        ocr._run_ocr = lambda p: "match"

        bm = run_benchmark_via_service(
            corpus, [ocr], partial_dir=partial_dir,
        )

        assert bm.document_count == 2
        # Plus aucun fichier partial pour cet engine après succès.
        partial_path = _partial_path(corpus.name, ocr.name, partial_dir)
        assert not partial_path.exists()

    def test_resume_skips_already_done_docs(self, tmp_path: Path) -> None:
        """Si un partial existe avec doc0 déjà calculé, le run ne
        ré-invoque pas l'engine pour doc0 — il prend le résultat
        partiel tel quel."""
        partial_dir = tmp_path / "partials"
        partial_dir.mkdir()
        corpus = self._make_corpus(tmp_path, n=3)

        ocr = _MockOCR(name="resumable2")
        # On compte combien de fois l'engine est appelé.
        call_count = {"n": 0}

        def counting_ocr(p):
            call_count["n"] += 1
            return "match"

        ocr._run_ocr = counting_ocr

        # Pré-écrire un partial pour doc0 avec une CER fictive de 0.99
        # pour vérifier qu'on prend la valeur du partial, pas une
        # nouvelle exécution.
        partial_path = _partial_path(corpus.name, ocr.name, partial_dir)
        pre_existing = _make_doc_result("doc0", hyp="from_partial", cer=0.99)
        _save_partial_line(partial_path, pre_existing)

        bm = run_benchmark_via_service(
            corpus, [ocr], partial_dir=partial_dir,
        )

        # L'engine n'a été appelé que pour doc1 + doc2 (pas doc0).
        assert call_count["n"] == 2

        # Le résultat final contient bien les 3 docs, doc0 venant
        # du partial (CER 0.99).
        report = bm.engine_reports[0]
        assert len(report.document_results) == 3
        doc0_result = next(d for d in report.document_results if d.doc_id == "doc0")
        assert doc0_result.hypothesis == "from_partial"
        assert doc0_result.metrics.cer == pytest.approx(0.99)

    def test_all_docs_already_done_skips_engine_entirely(
        self, tmp_path: Path,
    ) -> None:
        partial_dir = tmp_path / "partials"
        partial_dir.mkdir()
        corpus = self._make_corpus(tmp_path, n=2)

        ocr = _MockOCR(name="alldone")
        ocr._run_ocr = lambda p: pytest.fail(
            "Engine ne devrait pas être appelé — tout est dans le partial.",
        )

        partial_path = _partial_path(corpus.name, ocr.name, partial_dir)
        for i in range(2):
            _save_partial_line(
                partial_path, _make_doc_result(f"doc{i}", hyp=f"prefilled{i}"),
            )

        bm = run_benchmark_via_service(
            corpus, [ocr], partial_dir=partial_dir,
        )

        report = bm.engine_reports[0]
        assert len(report.document_results) == 2
        # Ordre du corpus original préservé.
        assert [d.doc_id for d in report.document_results] == ["doc0", "doc1"]
        assert [d.hypothesis for d in report.document_results] == [
            "prefilled0", "prefilled1",
        ]

    def test_per_engine_isolation(self, tmp_path: Path) -> None:
        """Deux engines ont chacun leur propre fichier partial — un
        partial pour engine_a ne pollue pas engine_b."""
        partial_dir = tmp_path / "partials"
        partial_dir.mkdir()
        corpus = self._make_corpus(tmp_path, n=2)

        ocr_a = _MockOCR(name="engine_a")
        ocr_a._run_ocr = lambda p: "from_a"
        ocr_b = _MockOCR(name="engine_b")
        ocr_b._run_ocr = lambda p: "from_b"

        # Pré-remplir uniquement le partial de engine_a pour doc0.
        partial_a = _partial_path(corpus.name, ocr_a.name, partial_dir)
        _save_partial_line(
            partial_a, _make_doc_result("doc0", hyp="A_pre"),
        )

        bm = run_benchmark_via_service(
            corpus, [ocr_a, ocr_b], partial_dir=partial_dir,
        )

        report_a = next(r for r in bm.engine_reports if r.engine_name == "engine_a")
        report_b = next(r for r in bm.engine_reports if r.engine_name == "engine_b")

        # engine_a : doc0 vient du partial, doc1 calculé.
        a_doc0 = next(d for d in report_a.document_results if d.doc_id == "doc0")
        assert a_doc0.hypothesis == "A_pre"

        # engine_b : doc0 calculé from_b (pas de partial pour B).
        b_doc0 = next(d for d in report_b.document_results if d.doc_id == "doc0")
        assert b_doc0.hypothesis == "from_b"

    def test_partial_files_removed_on_success(self, tmp_path: Path) -> None:
        partial_dir = tmp_path / "partials"
        corpus = self._make_corpus(tmp_path, n=2)

        engines = [_MockOCR(name=f"e{i}") for i in range(3)]
        for e in engines:
            e._run_ocr = lambda p: "match"

        run_benchmark_via_service(
            corpus, engines, partial_dir=partial_dir,
        )

        # Aucun fichier partial ne survit après un run réussi.
        leftovers = list(partial_dir.glob("*.partial.jsonl"))
        assert leftovers == [], f"partials résiduels : {leftovers}"

    def test_no_partial_dir_keeps_unified_path(self, tmp_path: Path) -> None:
        """Sans ``partial_dir``, le code garde le chemin rapide
        unifié (pas de fichiers partiels créés)."""
        corpus = self._make_corpus(tmp_path, n=2)
        ocr = _MockOCR(name="no_partial")
        ocr._run_ocr = lambda p: "match"

        bm = run_benchmark_via_service(corpus, [ocr])
        assert bm.document_count == 2

        # Aucun .partial.jsonl créé dans tmp_path car le chemin
        # unifié n'écrit pas de partials.
        leftovers = list(tmp_path.rglob("*.partial.jsonl"))
        assert leftovers == []

    def test_partial_persists_when_engine_was_not_finished(
        self, tmp_path: Path,
    ) -> None:
        """Si le run a réussi pour engine_a (partial supprimé) mais
        seuls 1/2 docs sont dans le partial de engine_b avant
        cancel, le partial de engine_b doit survivre pour reprise."""
        partial_dir = tmp_path / "partials"
        partial_dir.mkdir()
        corpus = self._make_corpus(tmp_path, n=2)

        # Simulation d'un état post-crash : engine_b a un partial
        # avec doc0 mais pas doc1.  cancel_event signalé avant
        # l'engine suivant.
        ocr_b = _MockOCR(name="incomplete_b")
        partial_b = _partial_path(corpus.name, ocr_b.name, partial_dir)
        _save_partial_line(
            partial_b, _make_doc_result("doc0", hyp="B0_pre"),
        )

        # cancel_event signalé → on n'entre pas dans la boucle
        # engine.  Pas de docs traités pendant ce run.
        cancel = threading.Event()
        cancel.set()

        bm = run_benchmark_via_service(
            corpus, [ocr_b],
            partial_dir=partial_dir,
            cancel_event=cancel,
        )

        # Aucun engine traité (cancel pré-engine).
        assert bm.engine_reports == []
        # Le partial de engine_b est préservé pour la prochaine
        # exécution.
        assert partial_b.exists()


# ──────────────────────────────────────────────────────────────────────
# 3. Sérialisation NDJSON cross-process
# ──────────────────────────────────────────────────────────────────────


class TestNDJSONFormat:
    """Le format NDJSON (une ligne JSON par document) est ce qui
    rend la reprise robuste : un crash mid-write tronque au pire
    une ligne ; toutes les lignes complètes restent lisibles."""

    def test_one_json_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "r.jsonl"
        _save_partial_line(path, _make_doc_result("doc0"))
        _save_partial_line(path, _make_doc_result("doc1"))

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        for line in lines:
            payload = json.loads(line)
            assert "doc_id" in payload
            assert "metrics" in payload

    def test_unicode_preserved_in_hypothesis(self, tmp_path: Path) -> None:
        path = tmp_path / "r.jsonl"
        dr = _make_doc_result("doc1")
        dr.hypothesis = "Église — œ ç à é"

        _save_partial_line(path, dr)
        loaded = _load_partial(path)

        assert loaded[0].hypothesis == "Église — œ ç à é"
