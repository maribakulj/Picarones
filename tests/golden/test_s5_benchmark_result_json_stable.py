"""Sprint S5 — Tests de stabilité du JSON ``BenchmarkResult``.

Garantit que la sérialisation JSON de ``BenchmarkResult.as_dict``/
``to_json`` est :

- **Stable** : deux sérialisations successives produisent les mêmes
  bytes (modulo la clé ``run_date`` qui est forcée déterministe).
- **Conforme au snapshot** : le JSON correspond à un golden file
  versionné dans ``tests/golden/fixtures/benchmark_result_v2.json``.

Si le snapshot n'existe pas au premier run, il est créé et le test
échoue avec un message demandant de commit le fichier.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


GOLDEN_PATH = (
    Path(__file__).parent / "fixtures" / "benchmark_result_v2.json"
)


def _build_deterministic_benchmark_result():
    """Construit un BenchmarkResult totalement déterministe pour le snapshot.

    - Date fixée
    - Version fixée
    - 2 documents, 2 moteurs
    - Pas de valeurs aléatoires
    """
    from picarones.evaluation.benchmark_result import (
        BenchmarkResult,
        DocumentResult,
        EngineReport,
    )
    from picarones.evaluation.metric_result import MetricsResult

    # Document 1, moteur A
    dr_a_1 = DocumentResult(
        doc_id="doc1",
        image_path="/fixtures/doc1.jpg",
        ground_truth="Bonjour le monde",
        hypothesis="Bonjour le monde",
        metrics=MetricsResult(
            cer=0.0,
            cer_nfc=0.0,
            cer_caseless=0.0,
            wer=0.0,
            wer_normalized=0.0,
            mer=0.0,
            wil=0.0,
            reference_length=16,
            hypothesis_length=16,
        ),
        duration_seconds=1.5,
    )
    dr_a_2 = DocumentResult(
        doc_id="doc2",
        image_path="/fixtures/doc2.jpg",
        ground_truth="Au revoir",
        hypothesis="Au revoir!",
        metrics=MetricsResult(
            cer=0.05,
            cer_nfc=0.05,
            cer_caseless=0.05,
            wer=0.1,
            wer_normalized=0.1,
            mer=0.05,
            wil=0.1,
            reference_length=9,
            hypothesis_length=10,
        ),
        duration_seconds=2.0,
    )

    # Document 1, moteur B
    dr_b_1 = DocumentResult(
        doc_id="doc1",
        image_path="/fixtures/doc1.jpg",
        ground_truth="Bonjour le monde",
        hypothesis="Bonjour Ie monde",  # I capital au lieu de l minuscule
        metrics=MetricsResult(
            cer=0.0625,
            cer_nfc=0.0625,
            cer_caseless=0.0,
            wer=0.333333,
            wer_normalized=0.333333,
            mer=0.0625,
            wil=0.111111,
            reference_length=16,
            hypothesis_length=16,
        ),
        duration_seconds=2.5,
    )
    dr_b_2 = DocumentResult(
        doc_id="doc2",
        image_path="/fixtures/doc2.jpg",
        ground_truth="Au revoir",
        hypothesis="Au revoir",
        metrics=MetricsResult(
            cer=0.0,
            cer_nfc=0.0,
            cer_caseless=0.0,
            wer=0.0,
            wer_normalized=0.0,
            mer=0.0,
            wil=0.0,
            reference_length=9,
            hypothesis_length=9,
        ),
        duration_seconds=1.8,
    )

    report_a = EngineReport(
        engine_name="engine_alpha",
        engine_version="1.0.0",
        engine_config={"lang": "fra"},
        document_results=[dr_a_1, dr_a_2],
    )
    report_b = EngineReport(
        engine_name="engine_beta",
        engine_version="2.1.3",
        engine_config={"lang": "fra"},
        document_results=[dr_b_1, dr_b_2],
    )

    bench = BenchmarkResult(
        corpus_name="test_corpus_s5",
        corpus_source="/fixtures/corpus.zip",
        document_count=2,
        engine_reports=[report_a, report_b],
        run_date="2026-05-09T00:00:00+00:00",  # forcée déterministe
        picarones_version="2.0.0-test",
        metadata={"sprint": "S5", "deterministic": True},
    )
    return bench


# --------------------------------------------------------------------------
# 1. Stabilité : sérialiser 2 fois doit produire les mêmes bytes
# --------------------------------------------------------------------------


class TestBenchmarkResultSerializationStability:
    def test_two_serializations_same_bytes(self):
        bench = _build_deterministic_benchmark_result()
        # JSON sérialisation déterministe : ensure_ascii + sort_keys
        # via json.dumps explicite.
        s1 = json.dumps(
            bench.as_dict(), ensure_ascii=False, sort_keys=True, indent=2,
        )
        s2 = json.dumps(
            bench.as_dict(), ensure_ascii=False, sort_keys=True, indent=2,
        )
        assert s1 == s2, "BenchmarkResult.as_dict instable entre 2 appels"

    def test_serialization_via_to_json_stable(self, tmp_path):
        bench = _build_deterministic_benchmark_result()
        path1 = bench.to_json(tmp_path / "bench1.json")
        path2 = bench.to_json(tmp_path / "bench2.json")
        # Les deux fichiers doivent avoir le même contenu byte-pour-byte
        b1 = path1.read_bytes()
        b2 = path2.read_bytes()
        assert b1 == b2, "to_json non déterministe entre 2 écritures"


# --------------------------------------------------------------------------
# 2. Snapshot golden
# --------------------------------------------------------------------------


class TestBenchmarkResultGoldenSnapshot:
    def test_matches_golden_fixture(self):
        bench = _build_deterministic_benchmark_result()
        # Sérialisation canonique avec sort_keys pour stabilité
        actual = json.dumps(
            bench.as_dict(), ensure_ascii=False, sort_keys=True, indent=2,
        )

        if not GOLDEN_PATH.exists():
            # Premier run : on crée le snapshot et on échoue
            # explicitement pour forcer l'opérateur à commit.
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(actual + "\n", encoding="utf-8")
            pytest.fail(
                f"Snapshot golden créé dans {GOLDEN_PATH} — "
                "vérifier le contenu et commit le fichier."
            )

        expected = GOLDEN_PATH.read_text(encoding="utf-8").rstrip("\n")
        assert actual == expected, (
            f"Snapshot divergeant. Golden: {GOLDEN_PATH}.\n"
            "Si le changement est intentionnel, supprimer le golden et "
            "relancer le test pour le régénérer."
        )


# --------------------------------------------------------------------------
# 3. Structure invariante : les clés de premier niveau ne changent pas
# --------------------------------------------------------------------------


class TestBenchmarkResultTopLevelKeys:
    """Les clés top-level du JSON font partie de l'API publique
    (consommée par les rapports HTML, l'export CSV…). Les changer
    sans préavis casse les consommateurs."""

    def test_top_level_keys_preserved(self):
        bench = _build_deterministic_benchmark_result()
        d = bench.as_dict()

        expected_keys = {
            "picarones_version",
            "run_date",
            "corpus",
            "ranking",
            "engine_reports",
            "metadata",
        }
        actual_keys = set(d.keys())
        # Toutes les clés requises présentes
        missing = expected_keys - actual_keys
        assert not missing, (
            f"Clés top-level manquantes dans BenchmarkResult.as_dict: {missing}"
        )

    def test_corpus_substructure_keys(self):
        bench = _build_deterministic_benchmark_result()
        d = bench.as_dict()
        corpus = d["corpus"]
        assert "name" in corpus
        assert "source" in corpus
        assert "document_count" in corpus
