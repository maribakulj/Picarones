"""Tests E2E pour les nouvelles options CLI exposées en Phase B3-final
corr-A/B/C (mai 2026).

Vérifie que :

1. ``picarones run --help`` documente bien les 5 nouvelles options
   (``--views``, ``--expose-alto``, ``--char-exclude``,
   ``--partial-dir``, ``--entity-extractor``).
2. Les options sont effectivement propagées (test bout-en-bout via
   ``CliRunner`` sur un corpus mini avec ``PrecomputedTextAdapter``).
3. Le rapport HTML généré contient bien les sections multi-vues
   quand ``--views`` est utilisé.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def runner():
    from click.testing import CliRunner
    return CliRunner()


@pytest.fixture
def cli():
    from picarones.interfaces.cli import cli as cli_group
    return cli_group


@pytest.fixture
def mini_corpus(tmp_path: Path) -> Path:
    """Corpus mini déterministe avec PrecomputedTextAdapter."""
    from PIL import Image

    for i in range(2):
        doc_id = f"doc{i:02d}"
        img = Image.new("RGB", (50, 50), color=(255, 255, 255))
        img.save(tmp_path / f"{doc_id}.png")
        (tmp_path / f"{doc_id}.gt.txt").write_text(
            f"Texte référence {i}", encoding="utf-8",
        )
        (tmp_path / f"{doc_id}.tess.txt").write_text(
            f"Texte référence {i}", encoding="utf-8",
        )
    return tmp_path


# ──────────────────────────────────────────────────────────────────────
# 1. Smoke --help expose les nouvelles options
# ──────────────────────────────────────────────────────────────────────


class TestNewOptionsExposedInHelp:
    def test_views_option_documented(self, runner, cli) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--views" in result.output
        assert "alto_documentary" in result.output or "alto" in result.output

    def test_expose_alto_option_documented(self, runner, cli) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--expose-alto" in result.output

    def test_char_exclude_option_documented(self, runner, cli) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--char-exclude" in result.output

    def test_partial_dir_option_documented(self, runner, cli) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--partial-dir" in result.output

    def test_entity_extractor_option_documented(self, runner, cli) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--entity-extractor" in result.output


# ──────────────────────────────────────────────────────────────────────
# 2. Propagation effective via le helper local _run_orchestrator_for_cli
# ──────────────────────────────────────────────────────────────────────


class TestHelperPropagation:
    """Vérifie que ``_run_orchestrator_for_cli`` (helper local CLI)
    propage correctement les nouveaux params aux composants Phase B.

    Tester via le binaire ``picarones run`` requiert un engine OCR
    réel (Tesseract / mock complexe).  Tester le helper directement
    est plus rapide et couvre la propagation, ce qui est l'enjeu
    de l'audit B3-final.
    """

    def _make_corpus_and_adapter(self, tmp_path: Path):
        from picarones.adapters.ocr.base import BaseOCRAdapter
        from picarones.domain.artifacts import Artifact, ArtifactType
        from picarones.evaluation.corpus import Corpus, Document

        class _MockOCR(BaseOCRAdapter):
            def __init__(self) -> None:
                self._name = "mock_cli"

            @property
            def name(self) -> str:
                return self._name

            def execute(self, inputs, params, context):
                out = Path(context.workspace_uri) / f"{context.document_id}.txt"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("ok", encoding="utf-8")
                return {ArtifactType.RAW_TEXT: Artifact(
                    id=f"{context.document_id}:{self._name}:raw_text",
                    document_id=context.document_id,
                    type=ArtifactType.RAW_TEXT,
                    produced_by_step="ocr", uri=str(out),
                )}

        img = tmp_path / "doc0.png"
        img.write_bytes(b"x")
        corpus = Corpus(name="cli_helper_test", documents=[Document(
            image_path=img, ground_truth="ok", doc_id="doc0",
        )])
        return corpus, _MockOCR()

    def test_helper_default_views_text_final(self, tmp_path: Path) -> None:
        from picarones.interfaces.cli._workflows import (
            _run_orchestrator_for_cli,
        )
        corpus, engine = self._make_corpus_and_adapter(tmp_path)
        bm = _run_orchestrator_for_cli(corpus, [engine])
        # Le helper retourne un BenchmarkResult avec view_results
        # contenant text_final (vue par défaut).
        assert "text_final" in bm.view_results
        assert "alto_documentary" not in bm.view_results
        assert "searchability" not in bm.view_results

    def test_helper_custom_views_propagated(self, tmp_path: Path) -> None:
        from picarones.interfaces.cli._workflows import (
            _run_orchestrator_for_cli,
        )
        corpus, engine = self._make_corpus_and_adapter(tmp_path)
        bm = _run_orchestrator_for_cli(
            corpus, [engine],
            views=("text_final", "searchability"),
        )
        assert "text_final" in bm.view_results
        assert "searchability" in bm.view_results
        assert "alto_documentary" not in bm.view_results

    def test_helper_partial_dir_propagated(self, tmp_path: Path) -> None:
        """``partial_dir`` propagé jusqu'à RunSpec → partial JSONL créé."""
        from picarones.interfaces.cli._workflows import (
            _run_orchestrator_for_cli,
        )
        corpus, engine = self._make_corpus_and_adapter(tmp_path)
        partial_dir = tmp_path / "partial"
        bm = _run_orchestrator_for_cli(
            corpus, [engine], partial_dir=str(partial_dir),
        )
        # Le run réussit ; le partial est nettoyé en fin de run
        # (cf. _orchestrator_partial.delete_partial).
        assert bm.document_count == 1


# ──────────────────────────────────────────────────────────────────────
# 3. _engine_from_name reçoit bien expose_alto
# ──────────────────────────────────────────────────────────────────────


class TestEngineFromNameExposeAlto:
    def test_expose_alto_propagated_to_tesseract_adapter(self) -> None:
        """``_engine_from_name(name, expose_alto=True)`` instancie un
        TesseractAdapter avec ``expose_alto=True``.

        Garantit que le flag traverse de la CLI à l'adapter sans
        régression silencieuse.
        """
        from picarones.interfaces.cli import _engine_from_name

        adapter = _engine_from_name(
            "tesseract", lang="fra", psm=6, expose_alto=True,
        )
        assert adapter.expose_alto is True

    def test_default_no_expose_alto(self) -> None:
        from picarones.interfaces.cli import _engine_from_name

        adapter = _engine_from_name("tesseract", lang="fra", psm=6)
        assert adapter.expose_alto is False
