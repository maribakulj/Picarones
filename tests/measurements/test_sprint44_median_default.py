"""Tests Sprint 44 — médiane par défaut + détecteur d'asymétrie.

Couvre :

1. ``EngineReport.median_cer`` lit ``aggregated_metrics["cer"]["median"]``.
2. ``BenchmarkResult.ranking()`` :
   - inclut ``median_cer`` dans chaque entrée
   - trie sur la médiane par défaut (et non plus la moyenne)
   - retombe sur la moyenne si la médiane est absente
3. Détecteur ``MEDIAN_MEAN_GAP_WARNING`` :
   - se déclenche quand le ratio ``|moyenne - médiane| / médiane > 30%``
   - ne se déclenche pas quand symétrique
   - ne se déclenche pas si la médiane est nulle (corpus parfait)
   - importance HIGH si gap relatif ≥ 100 %
4. Anti-hallucination : chaque nombre rendu est dans le payload.
5. Rétrocompat : les consommateurs qui lisent ``mean_cer`` continuent
   à fonctionner.
"""

from __future__ import annotations

import re

import pytest

from picarones.measurements.metrics import MetricsResult
from picarones.measurements.narrative.detectors import detect_median_mean_gap_warning
from picarones.domain.facts import FactImportance, FactType
from picarones.measurements.narrative.renderer import extract_numbers, render_fact
from picarones.core.results import BenchmarkResult, DocumentResult, EngineReport


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _make_dr(cer: float, doc_id: str = "d") -> DocumentResult:
    return DocumentResult(
        doc_id=doc_id, image_path="/tmp/x.png",
        ground_truth="x", hypothesis="x",
        metrics=MetricsResult(
            cer=cer, cer_nfc=cer, cer_caseless=cer,
            wer=cer, wer_normalized=cer, mer=cer, wil=cer,
            reference_length=1, hypothesis_length=1,
        ),
        duration_seconds=0.1,
    )


def _make_engine_report(name: str, cers: list[float]) -> EngineReport:
    drs = [_make_dr(c, doc_id=f"d{i}") for i, c in enumerate(cers)]
    return EngineReport(
        engine_name=name, engine_version="1", engine_config={},
        document_results=drs,
    )


# ──────────────────────────────────────────────────────────────────────────
# 1. EngineReport.median_cer
# ──────────────────────────────────────────────────────────────────────────


class TestMedianCerProperty:
    def test_returns_median_from_aggregated(self) -> None:
        rep = _make_engine_report("e", [0.0, 0.0, 0.0, 1.0, 1.0])
        # Médiane de [0,0,0,1,1] = 0
        assert rep.median_cer == pytest.approx(0.0)

    def test_returns_none_when_no_docs(self) -> None:
        rep = EngineReport(
            engine_name="e", engine_version="1", engine_config={},
            document_results=[],
        )
        # Pas de docs → aggregated_metrics vide → mean/median = None
        assert rep.median_cer is None


# ──────────────────────────────────────────────────────────────────────────
# 2. ranking() — tri par médiane
# ──────────────────────────────────────────────────────────────────────────


class TestRankingByMedian:
    def test_includes_median_cer(self) -> None:
        bench = BenchmarkResult(
            corpus_name="c", corpus_source=None, document_count=3,
            engine_reports=[_make_engine_report("a", [0.1, 0.2, 0.3])],
        )
        ranking = bench.ranking()
        assert "median_cer" in ranking[0]
        assert ranking[0]["median_cer"] == pytest.approx(0.2)

    def test_sorts_by_median_not_mean(self) -> None:
        # Moteur A : 80 % à 0,03 + 20 % à 0,40 → moyenne ≈ 0,11, médiane = 0,03
        # Moteur B : 100 % à 0,05                 → moyenne = 0,05, médiane = 0,05
        # Tri par moyenne :   B (0.05) < A (0.11) → A est 2e
        # Tri par médiane :   A (0.03) < B (0.05) → A est 1er
        ers = [
            _make_engine_report(
                "A_asymmetric",
                [0.03] * 8 + [0.40] * 2,
            ),
            _make_engine_report(
                "B_steady",
                [0.05] * 10,
            ),
        ]
        bench = BenchmarkResult(
            corpus_name="c", corpus_source=None, document_count=10,
            engine_reports=ers,
        )
        ranking = bench.ranking()
        # Le moteur A doit gagner sur la médiane même si sa moyenne est pire
        assert ranking[0]["engine"] == "A_asymmetric"
        assert ranking[0]["mean_cer"] > ranking[1]["mean_cer"]
        assert ranking[0]["median_cer"] < ranking[1]["median_cer"]

    def test_falls_back_to_mean_when_median_missing(self) -> None:
        """Si median_cer est None, le tri retombe sur mean_cer.

        On reproduit ici la clé de tri utilisée par
        ``BenchmarkResult.ranking()`` pour valider sa logique sur des
        entrées synthétiques (impossible à produire via vrais
        ``EngineReport`` car ``aggregate_metrics`` calcule toujours
        une médiane quand il y a au moins un doc).
        """
        ranked = [
            {"engine": "x", "mean_cer": 0.10, "median_cer": None,
             "mean_wer": 0.0, "documents": 1, "failed": 0},
            {"engine": "y", "mean_cer": 0.05, "median_cer": None,
             "mean_wer": 0.0, "documents": 1, "failed": 0},
        ]

        def _key(e: dict) -> tuple:
            p = e.get("median_cer") if e.get("median_cer") is not None else e.get("mean_cer")
            return (p is None, p if p is not None else float("inf"))

        ranking = sorted(ranked, key=_key)
        # y (mean=0.05) doit passer avant x (mean=0.10)
        assert ranking[0]["engine"] == "y"


# ──────────────────────────────────────────────────────────────────────────
# 3. Détecteur MEDIAN_MEAN_GAP_WARNING
# ──────────────────────────────────────────────────────────────────────────


class TestMedianMeanGapDetector:
    def test_no_fact_when_distribution_symmetric(self) -> None:
        data = {"ranking": [{
            "engine": "tess", "median_cer": 0.05, "mean_cer": 0.055,
            "documents": 100,
        }]}
        # Gap relatif = 10% → en dessous du seuil 30%
        assert detect_median_mean_gap_warning(data) == []

    def test_emits_fact_when_asymmetric(self) -> None:
        data = {"ranking": [{
            "engine": "tess", "median_cer": 0.03, "mean_cer": 0.07,
            "documents": 100,
        }]}
        # Gap relatif = 133% → au-dessus du seuil
        facts = detect_median_mean_gap_warning(data)
        assert len(facts) == 1
        assert facts[0].type is FactType.MEDIAN_MEAN_GAP_WARNING
        assert facts[0].importance is FactImportance.HIGH  # >= 100 %
        assert facts[0].payload["engine"] == "tess"

    def test_medium_importance_when_moderate_gap(self) -> None:
        data = {"ranking": [{
            "engine": "tess", "median_cer": 0.05, "mean_cer": 0.075,
            "documents": 100,
        }]}
        # Gap relatif = 50% → au-dessus du seuil mais < 100 %
        facts = detect_median_mean_gap_warning(data)
        assert facts[0].importance is FactImportance.MEDIUM

    def test_no_fact_when_median_zero(self) -> None:
        """Médiane nulle → ratio non calculable → on s'abstient."""
        data = {"ranking": [{
            "engine": "tess", "median_cer": 0.0, "mean_cer": 0.05,
            "documents": 100,
        }]}
        assert detect_median_mean_gap_warning(data) == []

    def test_no_fact_when_no_ranking(self) -> None:
        assert detect_median_mean_gap_warning({}) == []
        assert detect_median_mean_gap_warning({"ranking": []}) == []
        assert detect_median_mean_gap_warning({"ranking": [{
            "engine": "x", "mean_cer": None, "median_cer": None,
        }]}) == []


# ──────────────────────────────────────────────────────────────────────────
# 4. Traçabilité anti-hallucination
# ──────────────────────────────────────────────────────────────────────────


class TestTraceability:
    @pytest.mark.parametrize("lang", ["fr", "en"])
    def test_every_rendered_number_is_in_payload(self, lang: str) -> None:
        data = {"ranking": [{
            "engine": "tess", "median_cer": 0.03, "mean_cer": 0.07,
            "documents": 100,
        }]}
        facts = detect_median_mean_gap_warning(data)
        sentence = render_fact(facts[0], lang)

        # Whitelist : aucune constante de template n'est attendue ici
        whitelist: set[str] = set()
        # Recompute payload representations
        payload_nums: set[str] = set()
        for v in facts[0].payload.values():
            if isinstance(v, (int, float)):
                payload_nums.add(str(v))
                if isinstance(v, float) and v.is_integer():
                    payload_nums.add(str(int(v)))

        for num in extract_numbers(sentence):
            normalized = num.replace(",", ".")
            assert normalized in payload_nums | whitelist, (
                f"Nombre {normalized!r} dans la phrase rendue n'est pas "
                f"traçable au payload {facts[0].payload!r}"
            )

    def test_template_has_no_hardcoded_numbers(self) -> None:
        from picarones.measurements.narrative.renderer import _load_templates
        for lang in ("fr", "en"):
            tpl = _load_templates(lang).get("median_mean_gap_warning", "")
            assert tpl, f"Template absent pour {lang}"
            # Enlever les placeholders {x} avant de chercher des chiffres
            cleaned = re.sub(r"\{[^}]+\}", "", tpl)
            digits = re.findall(r"\d", cleaned)
            assert not digits, f"Template {lang} contient des chiffres en dur : {digits}"


# ──────────────────────────────────────────────────────────────────────────
# 5. Intégration via build_synthesis
# ──────────────────────────────────────────────────────────────────────────


class TestSynthesisIntegration:
    def test_detector_registered_by_default(self) -> None:
        from picarones.measurements.narrative.registry import iter_detectors
        types = {entry.fact_type for entry in iter_detectors()}
        assert FactType.MEDIAN_MEAN_GAP_WARNING in types

    def test_synthesis_includes_warning_when_asymmetric(self) -> None:
        from picarones.measurements.narrative import build_synthesis
        data = {"ranking": [{
            "engine": "tess", "median_cer": 0.03, "mean_cer": 0.07,
            "documents": 100,
        }]}
        out = build_synthesis(data, lang="fr", max_facts=5)
        sentences = out["sentences"]
        # Au moins une phrase doit mentionner l'asymétrie
        assert any(
            "asymétrique" in s.lower() or "médiane" in s.lower()
            for s in sentences
        )
