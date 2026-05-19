"""Sprint S8.7 — couverture des branches résilience de
``picarones/app/services/partial_store.py``.

Cible : lignes 110-116 (OSError sur read), 121 (ligne vide
ignorée), 166-167 (KeyError/TypeError sur entrée malformée).

Ces branches sont la garantie de tolérance aux fichiers partiels
dégradés (crash, disque plein, schéma changé entre versions) :
sans elles, une seule ligne corrompue ferait perdre tout le
travail du benchmark précédent.
"""

from __future__ import annotations

import json

from picarones.app.services.partial_store import (
    _load_partial,
    _save_partial_line,
    _delete_partial,
)


def _valid_doc_dict() -> dict:
    """Dict minimal qui instancie un ``DocumentResult`` valide."""
    return {
        "doc_id": "doc1",
        "image_path": "/tmp/img.png",
        "ground_truth": "ref",
        "hypothesis": "hyp",
        "metrics": {
            "cer": 0.1,
            "wer": 0.2,
            "reference_length": 3,
            "hypothesis_length": 3,
        },
        "duration_seconds": 0.5,
    }


class TestLoadPartialDegraded:
    def test_nonexistent_file_returns_empty(self, tmp_path) -> None:
        result = _load_partial(tmp_path / "absent.jsonl")
        assert result == []

    def test_unreadable_file_returns_empty_with_warning(
        self, tmp_path, monkeypatch, caplog,
    ) -> None:
        """``OSError`` à l'ouverture (disque cassé, permission, etc.)
        → log warning, retour liste vide.  Mock direct de
        ``Path.open`` car ``chmod 0o000`` ne bloque pas root."""
        from pathlib import Path

        partial = tmp_path / "blocked.jsonl"
        partial.write_text(json.dumps(_valid_doc_dict()) + "\n")

        original_open = Path.open

        def raising_open(self, *args, **kwargs):
            if self == partial:
                raise OSError("simulated disk failure")
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", raising_open)

        with caplog.at_level("WARNING"):
            result = _load_partial(partial)
        assert result == []
        assert any(
            "illisible" in rec.message for rec in caplog.records
        )

    def test_empty_lines_skipped(self, tmp_path) -> None:
        """Lignes vides ne doivent pas être traitées comme JSON
        invalide — branche ``if not line: continue``."""
        partial = tmp_path / "with_empty.jsonl"
        partial.write_text(
            json.dumps(_valid_doc_dict()) + "\n"
            "\n"  # ligne vide
            "   \n"  # whitespace-only
            + json.dumps(_valid_doc_dict() | {"doc_id": "doc2"}) + "\n",
        )
        result = _load_partial(partial)
        assert len(result) == 2
        assert {r.doc_id for r in result} == {"doc1", "doc2"}

    def test_corrupt_json_line_skipped_with_warning(
        self, tmp_path, caplog,
    ) -> None:
        partial = tmp_path / "corrupt.jsonl"
        partial.write_text(
            json.dumps(_valid_doc_dict()) + "\n"
            "{not valid json\n"  # ligne corrompue
            + json.dumps(_valid_doc_dict() | {"doc_id": "doc2"}) + "\n",
        )
        with caplog.at_level("WARNING"):
            result = _load_partial(partial)
        assert len(result) == 2, (
            "les lignes valides doivent être chargées malgré la "
            "ligne corrompue"
        )
        assert any(
            "corrompue" in rec.message for rec in caplog.records
        )

    def test_malformed_entry_missing_required_field(
        self, tmp_path, caplog,
    ) -> None:
        """Entrée JSON valide mais sans ``doc_id`` (champ requis du
        DocumentResult) → ``KeyError`` capturé, log + skip."""
        partial = tmp_path / "malformed.jsonl"
        bad = _valid_doc_dict()
        del bad["doc_id"]  # supprime un champ requis
        partial.write_text(
            json.dumps(_valid_doc_dict()) + "\n"
            + json.dumps(bad) + "\n",
        )
        with caplog.at_level("WARNING"):
            result = _load_partial(partial)
        assert len(result) == 1
        assert any(
            "malformée" in rec.message for rec in caplog.records
        )


class TestSavePartialLineFailure:
    def test_writes_line_and_is_appendable(self, tmp_path) -> None:
        """Test smoke positif : ``_save_partial_line`` écrit + le
        fichier est lisible par ``_load_partial``."""
        from picarones.evaluation.benchmark_result import DocumentResult
        from picarones.evaluation.metric_result import MetricsResult

        partial = tmp_path / "out.jsonl"
        doc = DocumentResult(
            doc_id="d1", image_path="", ground_truth="ref",
            hypothesis="hyp",
            metrics=MetricsResult(
                cer=0.0, wer=0.0,
                reference_length=3, hypothesis_length=3,
            ),
            duration_seconds=0.0,
        )
        _save_partial_line(partial, doc)
        _save_partial_line(partial, doc)  # 2 lignes pour test append

        loaded = _load_partial(partial)
        assert len(loaded) == 2
        assert all(r.doc_id == "d1" for r in loaded)


class TestDeletePartial:
    def test_existing_file_deleted(self, tmp_path) -> None:
        partial = tmp_path / "to_delete.jsonl"
        partial.write_text("{}\n")
        _delete_partial(partial)
        assert not partial.exists()

    def test_nonexistent_file_is_noop(self, tmp_path) -> None:
        """Pas d'erreur si le fichier n'existe pas."""
        _delete_partial(tmp_path / "never.jsonl")  # no raise
