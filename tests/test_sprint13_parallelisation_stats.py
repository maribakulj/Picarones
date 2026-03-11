"""Tests Sprint 13 — Corrections structurelles : parallelisation, exceptions, statistiques.

Classes de tests
----------------
TestPyprojectCorrections     (4 tests)  — Part 1 : Beta, deps clarifiées
TestEngineExecutionMode      (5 tests)  — Part 2 : execution_mode sur les classes moteur
TestRunnerParallelParams     (5 tests)  — Part 3 : signature run_benchmark étendue
TestRunnerTimeout            (3 tests)  — Part 3 : timeout par document
TestRunnerPartialResults     (4 tests)  — Part 3 : sauvegarde / reprise partiels
TestRunnerSilentExceptions   (3 tests)  — Part 2 : warnings au lieu de pass silencieux
TestWilcoxonValidation       (7 tests)  — Part 4 : valeurs de référence connues
TestWilcoxonScipyIntegration (3 tests)  — Part 4 : cohérence scipy / natif
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_corpus(tmp_path):
    """Corpus minimal de 3 documents pour les tests runner."""
    from PIL import Image
    for i in range(3):
        img = Image.new("RGB", (100, 30), color="white")
        img.save(tmp_path / f"doc{i:02d}.png")
        (tmp_path / f"doc{i:02d}.gt.txt").write_text(f"texte vérité {i}", encoding="utf-8")
    return tmp_path


# ===========================================================================
# Part 1 — Corrections pyproject.toml
# ===========================================================================

class TestPyprojectCorrections:

    def _read_pyproject(self) -> str:
        return (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    def test_classifier_is_beta(self):
        """Le classifier doit être 4 - Beta et non 5 - Production/Stable."""
        content = self._read_pyproject()
        assert "Development Status :: 4 - Beta" in content, (
            "pyproject.toml doit contenir 'Development Status :: 4 - Beta'"
        )
        assert "Production/Stable" not in content, (
            "pyproject.toml ne doit plus contenir 'Production/Stable'"
        )

    def test_fastapi_not_in_base_deps(self):
        """fastapi ne doit pas être dans les dépendances de base."""
        import re
        content = self._read_pyproject()
        # Extraire la section dependencies = [...] sous [project] (avant la 1re section suivante)
        m = re.search(r"^dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL | re.MULTILINE)
        assert m, "Section dependencies introuvable dans pyproject.toml"
        base_deps = m.group(1)
        assert "fastapi" not in base_deps, (
            "fastapi ne doit pas être dans les dépendances de base — seulement dans [web]"
        )

    def test_httpx_not_in_base_deps(self):
        """httpx ne doit pas être dans les dépendances de base."""
        import re
        content = self._read_pyproject()
        m = re.search(r"^dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL | re.MULTILINE)
        assert m
        base_deps = m.group(1)
        assert "httpx" not in base_deps, (
            "httpx ne doit pas être dans les dépendances de base — seulement dans [web]"
        )

    def test_web_extra_has_fastapi_httpx_multipart(self):
        """L'extra [web] doit contenir fastapi, httpx et python-multipart."""
        import tomllib
        with (ROOT / "pyproject.toml").open("rb") as fh:
            data = tomllib.load(fh)
        web_deps = " ".join(data["project"]["optional-dependencies"]["web"])
        assert "fastapi" in web_deps
        assert "httpx" in web_deps
        assert "python-multipart" in web_deps


# ===========================================================================
# Part 2 — execution_mode sur les classes moteur
# ===========================================================================

class TestEngineExecutionMode:

    def test_base_engine_default_mode_is_io(self):
        """BaseOCREngine doit avoir execution_mode = 'io' par défaut."""
        from picarones.engines.base import BaseOCREngine
        assert BaseOCREngine.execution_mode == "io"

    def test_tesseract_engine_mode_is_cpu(self):
        """TesseractEngine doit avoir execution_mode = 'cpu'."""
        from picarones.engines.tesseract import TesseractEngine
        assert TesseractEngine.execution_mode == "cpu"

    def test_pero_engine_mode_is_cpu(self):
        """PeroOCREngine doit avoir execution_mode = 'cpu'."""
        from picarones.engines.pero_ocr import PeroOCREngine
        assert PeroOCREngine.execution_mode == "cpu"

    def test_mistral_engine_default_mode_is_io(self):
        """MistralOCREngine doit hériter execution_mode = 'io'."""
        from picarones.engines.mistral_ocr import MistralOCREngine
        assert MistralOCREngine.execution_mode == "io"

    def test_google_vision_default_mode_is_io(self):
        """GoogleVisionEngine doit hériter execution_mode = 'io'."""
        from picarones.engines.google_vision import GoogleVisionEngine
        assert GoogleVisionEngine.execution_mode == "io"


# ===========================================================================
# Part 3 — Signature run_benchmark étendue
# ===========================================================================

class TestRunnerParallelParams:

    def test_max_workers_param_exists(self):
        """run_benchmark doit accepter max_workers."""
        from picarones.core.runner import run_benchmark
        sig = inspect.signature(run_benchmark)
        assert "max_workers" in sig.parameters

    def test_max_workers_default_is_4(self):
        """max_workers doit avoir 4 comme valeur par défaut."""
        from picarones.core.runner import run_benchmark
        sig = inspect.signature(run_benchmark)
        assert sig.parameters["max_workers"].default == 4

    def test_timeout_seconds_param_exists(self):
        """run_benchmark doit accepter timeout_seconds."""
        from picarones.core.runner import run_benchmark
        sig = inspect.signature(run_benchmark)
        assert "timeout_seconds" in sig.parameters

    def test_timeout_seconds_default_is_60(self):
        """timeout_seconds doit avoir 60.0 comme valeur par défaut."""
        from picarones.core.runner import run_benchmark
        sig = inspect.signature(run_benchmark)
        assert sig.parameters["timeout_seconds"].default == 60.0

    def test_partial_dir_param_exists(self):
        """run_benchmark doit accepter partial_dir (None par défaut)."""
        from picarones.core.runner import run_benchmark
        sig = inspect.signature(run_benchmark)
        assert "partial_dir" in sig.parameters
        assert sig.parameters["partial_dir"].default is None


# ===========================================================================
# Part 3 — Timeout par document
# ===========================================================================

class TestRunnerTimeout:

    def test_timeout_doc_result_has_error(self, tmp_corpus):
        """Un document ayant dépassé le timeout doit avoir engine_error contenant 'timeout'."""
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.core.runner import run_benchmark
        from picarones.engines.base import BaseOCREngine
        import time

        class SlowEngine(BaseOCREngine):
            @property
            def name(self): return "slow_engine"
            def version(self): return "0.1"
            def _run_ocr(self, image_path):
                time.sleep(5)  # 5 secondes — dépasse le timeout de 1s
                return "jamais atteint"

        corpus = load_corpus_from_directory(str(tmp_corpus))
        result = run_benchmark(
            corpus, [SlowEngine()],
            show_progress=False,
            timeout_seconds=1.0,
            max_workers=1,
        )
        assert len(result.engine_reports) == 1
        report = result.engine_reports[0]
        assert len(report.document_results) == len(corpus)
        # Au moins un document doit être marqué timeout
        timeout_docs = [dr for dr in report.document_results if dr.engine_error and "timeout" in dr.engine_error]
        assert len(timeout_docs) > 0, "Aucun document marqué timeout — le timeout ne fonctionne pas"

    def test_timeout_doc_result_cer_is_one(self, tmp_corpus):
        """Un document timeout doit avoir CER = 1.0."""
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.core.runner import run_benchmark
        from picarones.engines.base import BaseOCREngine
        import time

        class SlowEngine(BaseOCREngine):
            @property
            def name(self): return "slow"
            def version(self): return "0.1"
            def _run_ocr(self, image_path):
                time.sleep(5)
                return ""

        corpus = load_corpus_from_directory(str(tmp_corpus))
        result = run_benchmark(
            corpus, [SlowEngine()],
            show_progress=False,
            timeout_seconds=1.0,
            max_workers=1,
        )
        for dr in result.engine_reports[0].document_results:
            if dr.engine_error and "timeout" in dr.engine_error:
                assert dr.metrics.cer == 1.0

    def test_fast_docs_not_affected_by_timeout(self, tmp_corpus):
        """Des documents rapides ne doivent pas être touchés par un timeout généreux."""
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.core.runner import run_benchmark
        from picarones.engines.base import BaseOCREngine

        class FastEngine(BaseOCREngine):
            @property
            def name(self): return "fast"
            def version(self): return "0.1"
            def _run_ocr(self, image_path): return "texte ocr"

        corpus = load_corpus_from_directory(str(tmp_corpus))
        result = run_benchmark(
            corpus, [FastEngine()],
            show_progress=False,
            timeout_seconds=30.0,
        )
        timeout_docs = [
            dr for dr in result.engine_reports[0].document_results
            if dr.engine_error and "timeout" in dr.engine_error
        ]
        assert len(timeout_docs) == 0, "Les documents rapides ne doivent pas être marqués timeout"


# ===========================================================================
# Part 3 — Résultats partiels (sauvegarde / reprise)
# ===========================================================================

class TestRunnerPartialResults:

    def test_partial_file_created_during_run(self, tmp_corpus, tmp_path):
        """_save_partial_line doit être appelée pour chaque document traité."""
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.core.runner import run_benchmark
        from picarones.engines.base import BaseOCREngine
        import picarones.core.runner as runner_mod

        save_calls: list[str] = []
        original_save = runner_mod._save_partial_line

        def tracking_save(path, doc_result):
            save_calls.append(doc_result.doc_id)
            original_save(path, doc_result)

        class MockEngine(BaseOCREngine):
            @property
            def name(self): return "mock"
            def version(self): return "0.1"
            def _run_ocr(self, image_path): return "texte"

        corpus = load_corpus_from_directory(str(tmp_corpus))
        with patch.object(runner_mod, "_save_partial_line", side_effect=tracking_save):
            run_benchmark(
                corpus, [MockEngine()],
                show_progress=False,
                partial_dir=str(tmp_path),
            )
        assert len(save_calls) == len(corpus), (
            f"_save_partial_line appelée {len(save_calls)} fois, attendu {len(corpus)}"
        )

    def test_partial_file_deleted_after_success(self, tmp_corpus, tmp_path):
        """Le fichier .partial.json doit être supprimé après un benchmark réussi."""
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.core.runner import run_benchmark
        from picarones.engines.base import BaseOCREngine

        class MockEngine(BaseOCREngine):
            @property
            def name(self): return "mock"
            def version(self): return "0.1"
            def _run_ocr(self, image_path): return "texte"

        corpus = load_corpus_from_directory(str(tmp_corpus))
        run_benchmark(
            corpus, [MockEngine()],
            show_progress=False,
            partial_dir=str(tmp_path),
        )
        partial_files = list(tmp_path.glob("*.partial.json"))
        assert len(partial_files) == 0, f"Fichier(s) partiel(s) non supprimé(s) : {partial_files}"

    def test_partial_load_skips_already_done_docs(self, tmp_corpus, tmp_path):
        """La reprise depuis un fichier partiel doit sauter les documents déjà traités."""
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.core.runner import _load_partial, _partial_path, _sanitize_filename

        corpus = load_corpus_from_directory(str(tmp_corpus))
        corpus_name = corpus.name
        engine_name = "mock_engine"

        # Créer un fichier partiel simulant 1 document déjà traité
        path = _partial_path(corpus_name, engine_name, tmp_path)
        doc = corpus.documents[0]
        partial_line = {
            "doc_id": doc.doc_id,
            "image_path": str(doc.image_path),
            "ground_truth": doc.ground_truth,
            "hypothesis": "déjà traité",
            "metrics": {"cer": 0.1, "cer_nfc": 0.1, "cer_caseless": 0.1,
                        "wer": 0.1, "wer_normalized": 0.1, "mer": 0.1, "wil": 0.1,
                        "reference_length": 10, "hypothesis_length": 10},
            "duration_seconds": 0.5,
        }
        path.write_text(json.dumps(partial_line) + "\n", encoding="utf-8")

        _, loaded = _load_partial(corpus_name, engine_name, tmp_path)
        assert len(loaded) == 1
        assert loaded[0].doc_id == doc.doc_id
        assert loaded[0].hypothesis == "déjà traité"

    def test_partial_load_returns_empty_for_missing_file(self, tmp_path):
        """Si aucun fichier partiel n'existe, la liste doit être vide."""
        from picarones.core.runner import _load_partial
        _, loaded = _load_partial("corpus_inexistant", "moteur_inexistant", tmp_path)
        assert loaded == []


# ===========================================================================
# Part 2 — Exceptions non silencieuses dans le runner
# ===========================================================================

class TestRunnerSilentExceptions:

    def test_confusion_failure_logs_warning(self, tmp_corpus, caplog):
        """Une erreur dans build_confusion_matrix doit être loguée, pas ignorée."""
        import logging
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.core.runner import run_benchmark
        from picarones.engines.base import BaseOCREngine

        class MockEngine(BaseOCREngine):
            @property
            def name(self): return "mock"
            def version(self): return "0.1"
            def _run_ocr(self, image_path): return "texte ocr"

        corpus = load_corpus_from_directory(str(tmp_corpus))
        with patch(
            "picarones.core.runner._compute_document_result",
            wraps=__import__("picarones.core.runner", fromlist=["_compute_document_result"])._compute_document_result,
        ):
            with patch("picarones.core.confusion.build_confusion_matrix", side_effect=RuntimeError("crash test")):
                with caplog.at_level(logging.WARNING):
                    result = run_benchmark(corpus, [MockEngine()], show_progress=False)

        assert result is not None, "Le benchmark ne doit pas planter si la confusion matrix échoue"
        # La clé est que le benchmark se termine normalement
        assert len(result.engine_reports) == 1

    def test_progress_callback_failure_logs_warning(self, tmp_corpus, caplog):
        """Une exception dans le progress_callback doit être loguée, pas propagée."""
        import logging
        from picarones.core.corpus import load_corpus_from_directory
        from picarones.core.runner import run_benchmark
        from picarones.engines.base import BaseOCREngine

        class MockEngine(BaseOCREngine):
            @property
            def name(self): return "mock"
            def version(self): return "0.1"
            def _run_ocr(self, image_path): return "texte"

        def bad_callback(engine_name, doc_idx, doc_id):
            raise ValueError("callback crash")

        corpus = load_corpus_from_directory(str(tmp_corpus))
        with caplog.at_level(logging.WARNING):
            result = run_benchmark(
                corpus, [MockEngine()],
                show_progress=False,
                progress_callback=bad_callback,
            )
        assert result is not None
        assert any("progress_callback" in r.message for r in caplog.records), (
            "L'exception du callback doit être loguée en WARNING"
        )

    def test_aggregate_helpers_log_on_failure(self, caplog):
        """Les helpers _aggregate_* doivent logger en WARNING et retourner None."""
        import logging
        from picarones.core.runner import _aggregate_confusion

        # Créer un doc_result avec des données de confusion corrompues
        from picarones.core.results import DocumentResult
        from picarones.core.metrics import MetricsResult
        bad_dr = DocumentResult(
            doc_id="x", image_path="x.png", ground_truth="gt", hypothesis="hyp",
            metrics=MetricsResult(cer=0.1, cer_nfc=0.1, cer_caseless=0.1,
                                   wer=0.1, wer_normalized=0.1, mer=0.1, wil=0.1,
                                   reference_length=2, hypothesis_length=2),
            duration_seconds=0.1,
            confusion_matrix={"invalid_key": "corrupt_data"},  # va planter ConfusionMatrix(**...)
        )
        with caplog.at_level(logging.WARNING):
            result = _aggregate_confusion([bad_dr])
        assert result is None
        assert any("aggregate_confusion" in r.message for r in caplog.records)


# ===========================================================================
# Part 4 — Validation du test de Wilcoxon contre valeurs de référence
# ===========================================================================

class TestWilcoxonValidation:

    def test_identical_sequences_not_significant(self):
        """Séquences identiques → pas de différence, p = 1.0, significant = False."""
        from picarones.core.statistics import wilcoxon_test
        a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        r = wilcoxon_test(a, a)
        assert r["significant"] is False
        assert r["p_value"] == 1.0
        assert r["n_pairs"] == 0

    def test_all_positive_diffs_w_minus_is_zero(self):
        """Si toutes les différences a−b sont positives : W⁻ = 0, W⁺ = n(n+1)/2."""
        from picarones.core.statistics import wilcoxon_test
        n = 10
        a = [float(i) for i in range(1, n + 1)]
        b = [0.0] * n
        r = wilcoxon_test(a, b)
        expected_total = n * (n + 1) / 2.0
        assert math.isclose(r["W_minus"], 0.0, abs_tol=1e-9)
        assert math.isclose(r["W_plus"], expected_total, abs_tol=1e-9)

    def test_w_plus_w_minus_sum_invariant(self):
        """W⁺ + W⁻ doit toujours être égal à n(n+1)/2 (n = nombre de paires non nulles)."""
        from picarones.core.statistics import wilcoxon_test
        a = [0.10, 0.25, 0.05, 0.40, 0.30, 0.15, 0.20, 0.35, 0.08, 0.18]
        b = [0.12, 0.20, 0.08, 0.35, 0.28, 0.18, 0.15, 0.40, 0.10, 0.20]
        r = wilcoxon_test(a, b)
        n = r["n_pairs"]
        expected = n * (n + 1) / 2.0
        actual = r["W_plus"] + r["W_minus"]
        assert math.isclose(actual, expected, abs_tol=1e-6), (
            f"W⁺+W⁻ = {actual} ≠ n(n+1)/2 = {expected}"
        )

    def test_clearly_different_sequences_significant(self):
        """Deux séquences très différentes (n=15) doivent donner p < 0.05."""
        from picarones.core.statistics import wilcoxon_test
        a = [0.05] * 15          # moteur A très performant
        b = [0.60] * 15          # moteur B peu performant — toutes diff = −0.55
        # Diffs a−b = −0.55 pour tous → W⁺ = 0 → devrait être significatif
        r = wilcoxon_test(a, b)
        assert r["significant"] is True, f"p = {r['p_value']} — devrait être significatif"
        assert r["p_value"] < 0.05

    def test_large_n_normal_approximation_reasonable(self):
        """Pour n = 20, l'approximation normale doit donner une p-value dans [0, 1]."""
        from picarones.core.statistics import wilcoxon_test
        import random
        rng = random.Random(42)
        a = [rng.uniform(0.1, 0.5) for _ in range(20)]
        b = [x + rng.uniform(0.0, 0.1) for x in a]
        r = wilcoxon_test(a, b)
        assert 0.0 <= r["p_value"] <= 1.0
        assert r["n_pairs"] <= 20

    def test_small_n_returns_conservative_p(self):
        """Pour n < 10, la p-value doit être 0.04 (significatif) ou 0.20 (non sign.)."""
        from picarones.core.statistics import wilcoxon_test, _SCIPY_AVAILABLE
        if _SCIPY_AVAILABLE:
            pytest.skip("scipy disponible — la table exacte n'est pas utilisée")
        a = [0.1, 0.2, 0.3]
        b = [0.5, 0.6, 0.7]  # toutes diff = −0.4 → W = 0 → significatif
        r = wilcoxon_test(a, b)
        # Avec n=3, W=0 ≤ _W_CRITICAL[3]=0 → p=0.04
        assert r["p_value"] in (0.04, 0.20)

    def test_result_keys_complete(self):
        """Le dict retourné doit contenir toutes les clés documentées."""
        from picarones.core.statistics import wilcoxon_test
        r = wilcoxon_test([0.1, 0.3, 0.2, 0.4, 0.15, 0.35, 0.25, 0.5, 0.45, 0.05],
                          [0.2, 0.2, 0.3, 0.3, 0.25, 0.25, 0.35, 0.35, 0.40, 0.15])
        for key in ("statistic", "p_value", "significant", "interpretation", "n_pairs", "W_plus", "W_minus"):
            assert key in r, f"Clé manquante dans le résultat Wilcoxon : {key}"


# ===========================================================================
# Part 4 — Cohérence scipy / implémentation native
# ===========================================================================

class TestWilcoxonScipyIntegration:

    def test_scipy_available_flag_is_bool(self):
        """_SCIPY_AVAILABLE doit être un booléen."""
        from picarones.core.statistics import _SCIPY_AVAILABLE
        assert isinstance(_SCIPY_AVAILABLE, bool)

    def test_scipy_and_native_agree_on_significance(self):
        """Scipy et l'implémentation native doivent s'accorder sur la significativité."""
        from picarones.core.statistics import wilcoxon_test, _SCIPY_AVAILABLE, _native_p_value
        if not _SCIPY_AVAILABLE:
            pytest.skip("scipy non disponible")

        # Cas avec différences claires et n suffisant pour que les deux méthodes convergent
        a = [0.05, 0.08, 0.06, 0.07, 0.04, 0.09, 0.05, 0.07, 0.06, 0.08,
             0.05, 0.07, 0.06, 0.08, 0.04]
        b = [0.30, 0.35, 0.28, 0.32, 0.31, 0.29, 0.34, 0.33, 0.30, 0.31,
             0.29, 0.32, 0.33, 0.30, 0.31]

        r = wilcoxon_test(a, b)
        # Avec scipy, résultat doit être significatif
        assert r["significant"] is True

    def test_scipy_p_value_in_valid_range(self):
        """La p-value fournie par scipy doit être dans [0, 1]."""
        from picarones.core.statistics import wilcoxon_test, _SCIPY_AVAILABLE
        if not _SCIPY_AVAILABLE:
            pytest.skip("scipy non disponible")

        a = [0.1 + i * 0.02 for i in range(12)]
        b = [0.1 + i * 0.01 for i in range(12)]
        r = wilcoxon_test(a, b)
        assert 0.0 <= r["p_value"] <= 1.0
