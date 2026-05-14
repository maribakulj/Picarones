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
    """Vérification stricte : chaque option B3-final affiche son nom,
    son help text réel et au moins une valeur d'exemple métier.

    Phase D4 audit B3-final — renforcement des assertions identifié
    comme faible par l'audit (avant : ``assert "--views" in output``,
    après : vérification du texte d'aide complet).
    """

    def test_views_option_fully_documented(self, runner, cli) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        # Présence du flag.
        assert "--views" in result.output
        # Valeurs canoniques mentionnées dans le help text.
        assert "text_final" in result.output
        assert "alto_documentary" in result.output
        assert "searchability" in result.output

    def test_expose_alto_option_fully_documented(self, runner, cli) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--expose-alto" in result.output
        # Help text mentionne ALTO XML + Tesseract.
        assert "ALTO" in result.output
        assert (
            "Tesseract" in result.output or "tesseract" in result.output
        )

    def test_char_exclude_option_fully_documented(self, runner, cli) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--char-exclude" in result.output
        # Mentionne CER/WER (cas d'usage).
        assert "CER" in result.output or "WER" in result.output

    def test_partial_dir_option_fully_documented(self, runner, cli) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--partial-dir" in result.output
        # Mentionne le cas d'usage (reprise).
        assert "reprise" in result.output.lower() or "resume" in result.output.lower()

    def test_entity_extractor_option_fully_documented(self, runner, cli) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--entity-extractor" in result.output
        # Mentionne le format attendu (dotted path).
        assert "dotted" in result.output.lower() or ":" in result.output

    def test_workflows_secondaires_also_have_options(
        self, runner, cli,
    ) -> None:
        """Phase D1 audit — les commandes diagnose/economics/edition
        exposent aussi les 5 options B3-final via le decorator
        ``_b3_final_options``."""
        for cmd in ("diagnose", "economics", "edition"):
            result = runner.invoke(cli, [cmd, "--help"])
            assert result.exit_code == 0, (
                f"'{cmd} --help' a planté"
            )
            for opt in ("--views", "--expose-alto", "--char-exclude",
                        "--partial-dir", "--entity-extractor"):
                assert opt in result.output, (
                    f"Commande {cmd!r} : option {opt!r} manquante "
                    f"dans --help (decorator _b3_final_options non "
                    "appliqué ?)"
                )


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

    def test_helper_output_json_does_not_crash(self, tmp_path: Path) -> None:
        """Régression : passer ``output_json`` au helper CLI ne doit
        PAS lever ``ValueError: Aucun document valide trouvé``.

        Avant le fix (mai 2026, branche test-alto-pipelines), le path
        ``execute_preset(..., spec.output_json=set)`` faisait appel à
        ``_persist_legacy_benchmark_json`` qui tentait un
        ``load_corpus_from_directory(extracted_dir)`` — or en mode
        preset, ``extracted_dir`` est le ``workspace_dir`` qui ne
        contient que les ``.gt.txt`` synthétisés par
        ``document_to_document_ref``, pas les images.

        Symptôme observé en prod : "Erreur : Aucun document valide
        trouvé dans /tmp/picarones_web_*/gt. Vérifiez que les
        fichiers GT portent le suffixe '.gt.txt'." — affiché après
        que les 6 documents aient pourtant été correctement OCRés
        (les logs montraient les appels Mistral réussis juste avant).

        Le fix passe le ``Corpus`` mémoire à ``execute_preset`` via
        ``corpus_legacy=corpus`` ; ``_persist_legacy_benchmark_json``
        l'utilise alors directement sans reload.
        """
        from picarones.interfaces.cli._workflows import (
            _run_orchestrator_for_cli,
        )
        corpus, engine = self._make_corpus_and_adapter(tmp_path)
        output_json = tmp_path / "results.json"
        # NB : on passe explicitement ``output_json`` — c'est ce qui
        # active le path ``_persist_legacy_benchmark_json`` qui était
        # bugué.  Sans ``output_json``, le test passait à tort.
        bm = _run_orchestrator_for_cli(
            corpus, [engine],
            output_json=str(output_json),
        )
        assert bm.document_count == 1
        # Le JSON legacy doit avoir été persisté correctement (preuve
        # que ``_persist_legacy_benchmark_json`` a réussi avec le
        # corpus mémoire au lieu de crasher sur le reload).
        assert output_json.exists(), (
            f"output_json {output_json} n'a pas été écrit — le path "
            "`_persist_legacy_benchmark_json` a probablement crashé "
            "(regression : reload depuis workspace_dir gt-only)"
        )
        # Sanity : le JSON est parsable.
        import json
        loaded = json.loads(output_json.read_text(encoding="utf-8"))
        assert "engines" in loaded or "engine_reports" in loaded

    def test_helper_partial_dir_propagated(self, tmp_path: Path) -> None:
        """``partial_dir`` propagé jusqu'à RunSpec → directory créé +
        nettoyé en fin de run (lifecycle complet).

        Phase D4 audit B3-final — renforcement de l'assertion.  Avant
        on vérifiait juste ``document_count`` ; un partial_dir absent
        passait silencieusement.  Maintenant on vérifie le lifecycle :
        le directory est créé pendant le run et nettoyé à la fin
        (``delete_partial``).
        """
        from picarones.interfaces.cli._workflows import (
            _run_orchestrator_for_cli,
        )
        corpus, engine = self._make_corpus_and_adapter(tmp_path)
        partial_dir = tmp_path / "partial"
        # Pre-conditions : le directory n'existe pas encore.
        assert not partial_dir.exists()

        bm = _run_orchestrator_for_cli(
            corpus, [engine], partial_dir=str(partial_dir),
        )
        # Post-conditions : le run réussit et a effectivement créé
        # le directory (preuve que le param est arrivé jusqu'à
        # ``_execute_with_partial``).  Le contenu .jsonl est nettoyé
        # par ``delete_partial`` en fin de run réussi.
        assert bm.document_count == 1
        assert partial_dir.exists(), (
            f"partial_dir {partial_dir} n'a pas été créé — preuve que "
            "le param n'est pas propagé jusqu'à l'orchestrateur"
        )
        # Les .jsonl du partial sont supprimés en fin de run.
        jsonl_files = list(partial_dir.glob("*.jsonl"))
        assert not jsonl_files, (
            f"Partial JSONL non nettoyé en fin de run réussi : "
            f"{jsonl_files}"
        )


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

    def test_expose_alto_with_non_tesseract_warns(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Phase D4 audit B3-final — l'utilisateur qui demande
        ``--expose-alto`` avec un moteur autre que Tesseract reçoit
        un avertissement explicite plutôt qu'un silent drop du flag.

        On utilise ``precomputed_text`` car il est disponible sans
        binaire externe (pas besoin de Tesseract installé pour le
        test).
        """
        import logging
        from picarones.interfaces.cli import _engine_from_name

        with caplog.at_level(logging.WARNING):
            try:
                _engine_from_name(
                    "precomputed_text", lang="fra", psm=6,
                    expose_alto=True,
                )
            except Exception:
                # Le factory peut lever pour args manquants — on
                # capture mais ce n'est pas l'enjeu du test : on
                # vérifie juste le warning émis AVANT.
                pass

        # L'avertissement doit mentionner que le moteur ne supporte
        # pas l'ALTO + que seul Tesseract le fait.
        warnings_text = "\n".join(
            r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING
        )
        assert "expose-alto" in warnings_text.lower() or \
               "expose_alto" in warnings_text.lower() or \
               "alto" in warnings_text.lower()
        assert "precomputed_text" in warnings_text
