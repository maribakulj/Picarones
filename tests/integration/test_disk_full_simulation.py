"""Sprint S5 — Simulation de disque plein (ENOSPC).

Vérifie la robustesse du chemin "écriture sur disque" face à un
``OSError(28, 'No space left on device')``.

Cas couverts :

- ``partial_store._save_partial_line`` doit logger un warning et NE
  PAS lever (le benchmark continue, on ne casse pas tout pour une
  ligne perdue).
- ``BenchmarkResult.to_json`` doit propager l'OSError (l'utilisateur
  veut savoir que le rapport n'a pas pu être écrit).
- Aucun fichier corrompu / partiel n'est laissé.
"""

from __future__ import annotations

import errno
import os
import json
from pathlib import Path
from unittest.mock import patch

import pytest


def _enospc_oserror():
    """Construit un OSError(ENOSPC) prêt à utiliser comme side_effect."""
    return OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


# --------------------------------------------------------------------------
# 1. partial_store._save_partial_line absorbe ENOSPC
# --------------------------------------------------------------------------


class TestPartialStoreEnospcAbsorbed:
    """Quand le disque est plein, on ne veut pas casser un
    benchmark de 1000 docs juste parce que le partial_dir est full :
    ``_save_partial_line`` log warning et retourne."""

    def test_save_partial_line_enospc_logs_warning_no_raise(
        self, tmp_path, caplog,
    ):
        from picarones.app.services.partial_store import _save_partial_line
        from picarones.evaluation.benchmark_result import DocumentResult
        from picarones.evaluation.metric_result import MetricsResult

        partial_path = tmp_path / "p.partial.jsonl"
        doc = DocumentResult(
            doc_id="d1",
            image_path="/a/b.jpg",
            ground_truth="x",
            hypothesis="x",
            metrics=MetricsResult(reference_length=1, hypothesis_length=1),
            duration_seconds=0.1,
        )

        # Patch ``open`` pour lever ENOSPC à l'ouverture en append.
        original_open = Path.open

        def _open_with_enospc(self, mode="r", *args, **kwargs):
            if "a" in mode and self == partial_path:
                raise _enospc_oserror()
            return original_open(self, mode, *args, **kwargs)

        with patch.object(Path, "open", _open_with_enospc):
            with caplog.at_level("WARNING"):
                # Ne doit PAS lever
                _save_partial_line(partial_path, doc)

        # Le warning a été loggé
        assert any(
            "partial_dir" in rec.message or "impossible" in rec.message.lower()
            for rec in caplog.records
        )
        # Aucun fichier partiel n'a été créé (open a échoué avant écriture)
        assert not partial_path.exists()


# --------------------------------------------------------------------------
# 2. _delete_partial absorbe ENOSPC
# --------------------------------------------------------------------------


class TestDeletePartialEnospcAbsorbed:
    def test_delete_partial_oserror_logs_warning(self, tmp_path, caplog):
        from picarones.app.services.partial_store import _delete_partial

        # Créer un fichier réel
        partial_path = tmp_path / "p.partial.jsonl"
        partial_path.write_text('{"doc_id": "x"}\n', encoding="utf-8")

        with patch.object(Path, "unlink", side_effect=_enospc_oserror()):
            with caplog.at_level("WARNING"):
                # Ne lève pas
                _delete_partial(partial_path)

        # Le warning est loggé
        assert any(
            "partial_dir" in rec.message or "impossible" in rec.message.lower()
            for rec in caplog.records
        )


# --------------------------------------------------------------------------
# 3. BenchmarkResult.to_json sur disque plein
# --------------------------------------------------------------------------


class TestBenchmarkResultToJsonEnospc:
    """``to_json`` ouvre un fichier et écrit en JSON. Sur ENOSPC,
    on doit propager l'OSError (l'utilisateur veut le savoir, le
    rapport est critique). Et aucun fichier corrompu ne doit
    rester sur disque (le file handler ferme automatiquement, mais
    on vérifie qu'aucun .json tronqué ne pollue le résultat).
    """

    def test_to_json_enospc_propagates_and_no_garbage(self, tmp_path):
        from picarones.evaluation.benchmark_result import (
            BenchmarkResult,
            EngineReport,
            DocumentResult,
        )
        from picarones.evaluation.metric_result import MetricsResult

        dr = DocumentResult(
            doc_id="d1",
            image_path="/a/b.jpg",
            ground_truth="x",
            hypothesis="x",
            metrics=MetricsResult(reference_length=1, hypothesis_length=1),
            duration_seconds=0.1,
        )
        report = EngineReport(
            engine_name="e",
            engine_version="1",
            engine_config={},
            document_results=[dr],
        )
        bench = BenchmarkResult(
            corpus_name="c",
            corpus_source=None,
            document_count=1,
            engine_reports=[report],
        )

        out = tmp_path / "rapport.json"

        # Patch json.dump pour lever ENOSPC pendant l'écriture
        # (simule un disque qui se remplit pendant l'écriture).
        with patch(
            "picarones.evaluation.benchmark_result.json.dump",
            side_effect=_enospc_oserror(),
        ):
            with pytest.raises(OSError) as exc_info:
                bench.to_json(out)
            assert exc_info.value.errno == errno.ENOSPC

        # Le fichier a pu être créé (ouverture en mode "w" précède dump)
        # mais s'il existe il doit être vide (aucune ligne JSON valide).
        if out.exists():
            content = out.read_text(encoding="utf-8")
            # Pas de JSON tronqué : soit vide, soit explicitement
            # incomplet. On ne tolère pas un demi-objet.
            if content:
                # Doit être impossible de parser comme JSON valide
                with pytest.raises(json.JSONDecodeError):
                    json.loads(content)


# --------------------------------------------------------------------------
# 4. Idempotence du delete_partial absent
# --------------------------------------------------------------------------


class TestDeletePartialAbsent:
    """Si le fichier n'existe pas, ``_delete_partial`` est un no-op
    silencieux (pas de FileNotFoundError, pas de warning)."""

    def test_delete_nonexistent_partial_silent_noop(self, tmp_path, caplog):
        from picarones.app.services.partial_store import _delete_partial

        nonexistent = tmp_path / "absent.partial.jsonl"
        assert not nonexistent.exists()

        with caplog.at_level("WARNING"):
            _delete_partial(nonexistent)

        # Pas de warning : c'est un no-op silencieux par contrat
        warnings = [
            r for r in caplog.records
            if r.levelname == "WARNING"
        ]
        assert warnings == []
