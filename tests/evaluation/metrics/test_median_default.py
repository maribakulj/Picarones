"""Tests Sprint 44 (médiane) — révisés par l'audit scientifique F1.

Historique : le Sprint 44 avait fait du **CER médian** le critère de
tri par défaut.  L'audit scientifique (mai 2026, F1) a montré que la
médiane de taux par document reste aveugle à la longueur ; le critère
de tri par défaut est désormais le **CER micro-moyenné**
(Σ distance_édition / Σ caractères_référence), standard du domaine
OCR/HTR.  La médiane redevient un **repli** (corpus sans comptes
bruts) et un **diagnostic de dispersion** (détecteur
``median_mean_gap_warning``), plus un critère de classement.

Couvre :

1. ``EngineReport.median_cer`` lit ``aggregated_metrics["cer"]["median"]``.
2. ``BenchmarkResult.ranking()`` :
   - inclut ``micro_cer`` et ``median_cer`` dans chaque entrée
   - trie sur le **micro-CER** par défaut quand les comptes bruts
     sont disponibles
   - retombe sur la médiane puis la moyenne si le micro est absent
3. Détecteur ``MEDIAN_MEAN_GAP_WARNING`` (inchangé) :
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

from picarones.evaluation.metric_result import MetricsResult
from picarones.reports.narrative.detectors import detect_median_mean_gap_warning
from picarones.domain.facts import FactImportance, FactType
from picarones.reports.narrative.renderer import extract_numbers, render_fact
from picarones.evaluation.benchmark_result import BenchmarkResult, DocumentResult, EngineReport


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _make_dr(
    cer: float,
    doc_id: str = "d",
    ref_chars: int | None = None,
) -> DocumentResult:
    """DocumentResult synthétique.

    Si ``ref_chars`` est fourni, on renseigne les comptes bruts
    (``cer_errors``/``cer_ref_chars``) cohérents avec ``cer`` pour
    activer le micro-CER ; sinon ils restent ``None`` et le tri
    retombe sur la médiane (chemin de repli historique Sprint 44).
    """
    cer_errors = None
    cer_ref_chars = None
    wer_errors = None
    wer_ref_words = None
    if ref_chars is not None:
        cer_ref_chars = ref_chars
        cer_errors = round(cer * ref_chars)
        wer_ref_words = max(1, ref_chars // 5)
        wer_errors = round(cer * wer_ref_words)
    return DocumentResult(
        doc_id=doc_id, image_path="/tmp/x.png",
        ground_truth="x", hypothesis="x",
        metrics=MetricsResult(
            cer=cer, cer_nfc=cer, cer_caseless=cer,
            wer=cer, wer_normalized=cer, mer=cer, wil=cer,
            reference_length=ref_chars or 1, hypothesis_length=ref_chars or 1,
            cer_errors=cer_errors, cer_ref_chars=cer_ref_chars,
            wer_errors=wer_errors, wer_ref_words=wer_ref_words,
        ),
        duration_seconds=0.1,
    )


def _make_engine_report(
    name: str,
    cers: list[float],
    ref_chars: list[int] | None = None,
) -> EngineReport:
    if ref_chars is None:
        drs = [_make_dr(c, doc_id=f"d{i}") for i, c in enumerate(cers)]
    else:
        drs = [
            _make_dr(c, doc_id=f"d{i}", ref_chars=rc)
            for i, (c, rc) in enumerate(zip(cers, ref_chars))
        ]
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


class TestRankingByMicro:
    def test_includes_micro_and_median_cer(self) -> None:
        bench = BenchmarkResult(
            corpus_name="c", corpus_source=None, document_count=3,
            engine_reports=[_make_engine_report(
                "a", [0.1, 0.2, 0.3], ref_chars=[100, 100, 100],
            )],
        )
        ranking = bench.ranking()
        assert "median_cer" in ranking[0]
        assert "micro_cer" in ranking[0]
        assert ranking[0]["median_cer"] == pytest.approx(0.2)
        # micro = (10+20+30)/300 = 0.2 (longueurs égales → micro == mean)
        assert ranking[0]["micro_cer"] == pytest.approx(0.2)

    def test_micro_is_default_sort_key_and_can_beat_median(self) -> None:
        """Cas scientifiquement décisif (F1) : micro ≠ médiane.

        Moteur A : excellent sur 9 courts documents (10 car, CER 0,02)
        mais catastrophique sur 1 page longue (5 000 car, CER 0,50).
          - médiane CER = 0,02  (tirée par les courts)
          - micro CER   = (9·10·0,02 + 5000·0,50) / (9·10 + 5000)
                        ≈ 2502 / 5090 ≈ 0,4916
        Moteur B : régulier partout (CER 0,10).
          - médiane = 0,10 ; micro ≈ 0,10
        Tri médiane : A (0,02) < B (0,10) → A gagnerait à tort.
        Tri micro   : B (0,10) < A (0,49) → B gagne, ce qui reflète
        la réalité (A rate la moitié d'une page de 5 000 caractères).
        """
        a = _make_engine_report(
            "A_short_specialist",
            [0.02] * 9 + [0.50],
            ref_chars=[10] * 9 + [5000],
        )
        b = _make_engine_report(
            "B_steady",
            [0.10] * 10,
            ref_chars=[500] * 10,
        )
        bench = BenchmarkResult(
            corpus_name="c", corpus_source=None, document_count=10,
            engine_reports=[a, b],
        )
        ranking = bench.ranking()
        # Le tri micro doit placer B premier, contredisant la médiane.
        assert ranking[0]["engine"] == "B_steady"
        assert ranking[0]["micro_cer"] < ranking[1]["micro_cer"]
        # ... alors que la médiane aurait (à tort) favorisé A.
        a_entry = next(r for r in ranking if r["engine"] == "A_short_specialist")
        assert a_entry["median_cer"] < ranking[0]["median_cer"]
        assert a_entry["micro_cer"] == pytest.approx(0.4916, abs=2e-3)

    def test_falls_back_to_median_when_micro_missing(self) -> None:
        """Sans comptes bruts (jiwer absent / fixture legacy), le tri
        retombe sur la médiane — comportement Sprint 44 préservé."""
        ers = [
            _make_engine_report("A_asymmetric", [0.03] * 8 + [0.40] * 2),
            _make_engine_report("B_steady", [0.05] * 10),
        ]
        bench = BenchmarkResult(
            corpus_name="c", corpus_source=None, document_count=10,
            engine_reports=ers,
        )
        ranking = bench.ranking()
        assert ranking[0]["micro_cer"] is None  # pas de comptes bruts
        assert ranking[0]["engine"] == "A_asymmetric"
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
            {"engine": "x", "micro_cer": None, "mean_cer": 0.10,
             "median_cer": None, "mean_wer": 0.0, "documents": 1, "failed": 0},
            {"engine": "y", "micro_cer": None, "mean_cer": 0.05,
             "median_cer": None, "mean_wer": 0.0, "documents": 1, "failed": 0},
        ]

        def _key(e: dict) -> tuple:
            p = e.get("micro_cer")
            if p is None:
                p = e.get("median_cer")
            if p is None:
                p = e.get("mean_cer")
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
        from picarones.reports.narrative.renderer import _load_templates
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
        from picarones.reports.narrative.registry import iter_detectors
        types = {entry.fact_type for entry in iter_detectors()}
        assert FactType.MEDIAN_MEAN_GAP_WARNING in types

    def test_synthesis_includes_warning_when_asymmetric(self) -> None:
        from picarones.reports.narrative import build_synthesis
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
