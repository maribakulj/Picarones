"""Tests Sprint 45 — couche backend de stratification par script_type.

Couvre :

1. ``BenchmarkResult.doc_strata`` accepte ``None`` (rétrocompat) ou
   un dict ``{doc_id: script_type}``.
2. ``available_strata()`` retourne la liste triée des strates
   distinctes (vide si pas de doc_strata, ignore les valeurs vides).
3. ``stratified_ranking()`` :
   - Recalcule mean/median par moteur sur les docs de la strate
   - Trie par médiane (cohérent avec ``ranking()`` Sprint 44)
   - Inclut les moteurs sans aucun doc dans la strate (entrée
     dégénérée avec mean/median = None)
4. ``corpus_homogeneity()`` :
   - Retourne ``None`` quand < 2 strates
   - Calcule l'écart inter-strate du leader (en CER médian)
   - Identifie la paire de strates min/max
5. ``as_dict()`` expose ``doc_strata``, ``available_strata`` et
   ``stratified_ranking`` quand renseignés (rétrocompat sinon).
6. **Test propriété** : sur un corpus asymétrique réaliste où le
   leader global change selon la strate, ``stratified_ranking``
   doit refléter ce changement.
"""

from __future__ import annotations

import pytest

from picarones.measurements.metrics import MetricsResult
from picarones.core.results import BenchmarkResult, DocumentResult, EngineReport


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _make_dr(doc_id: str, cer: float, error: str | None = None) -> DocumentResult:
    return DocumentResult(
        doc_id=doc_id, image_path=f"/tmp/{doc_id}.png",
        ground_truth="x", hypothesis="x",
        metrics=MetricsResult(
            cer=cer, cer_nfc=cer, cer_caseless=cer,
            wer=cer, wer_normalized=cer, mer=cer, wil=cer,
            reference_length=1, hypothesis_length=1,
            error=error,
        ),
        duration_seconds=0.1,
    )


def _make_engine(name: str, cers_by_doc: dict[str, float]) -> EngineReport:
    drs = [_make_dr(d, c) for d, c in cers_by_doc.items()]
    return EngineReport(
        engine_name=name, engine_version="1", engine_config={},
        document_results=drs,
    )


def _make_benchmark(
    engines: list[EngineReport],
    doc_strata: dict[str, str] | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        corpus_name="test",
        corpus_source=None,
        document_count=0,
        engine_reports=engines,
        doc_strata=doc_strata,
    )


# ──────────────────────────────────────────────────────────────────────────
# 1. doc_strata field
# ──────────────────────────────────────────────────────────────────────────


class TestDocStrataField:
    def test_default_is_none(self) -> None:
        b = _make_benchmark([_make_engine("a", {"d1": 0.1})])
        assert b.doc_strata is None

    def test_accepts_dict(self) -> None:
        b = _make_benchmark(
            [_make_engine("a", {"d1": 0.1})],
            doc_strata={"d1": "gothic"},
        )
        assert b.doc_strata == {"d1": "gothic"}


# ──────────────────────────────────────────────────────────────────────────
# 2. available_strata
# ──────────────────────────────────────────────────────────────────────────


class TestAvailableStrata:
    def test_empty_when_no_doc_strata(self) -> None:
        b = _make_benchmark([_make_engine("a", {"d1": 0.1})])
        assert b.available_strata() == []

    def test_returns_sorted_unique(self) -> None:
        b = _make_benchmark(
            [_make_engine("a", {"d1": 0.1, "d2": 0.2, "d3": 0.3})],
            doc_strata={
                "d1": "gothic", "d2": "humanistic", "d3": "gothic",
            },
        )
        assert b.available_strata() == ["gothic", "humanistic"]

    def test_ignores_empty_strings(self) -> None:
        b = _make_benchmark(
            [_make_engine("a", {"d1": 0.1, "d2": 0.2})],
            doc_strata={"d1": "gothic", "d2": ""},
        )
        assert b.available_strata() == ["gothic"]


# ──────────────────────────────────────────────────────────────────────────
# 3. stratified_ranking
# ──────────────────────────────────────────────────────────────────────────


class TestStratifiedRanking:
    def test_empty_when_no_strata(self) -> None:
        b = _make_benchmark([_make_engine("a", {"d1": 0.1})])
        assert b.stratified_ranking() == {}

    def test_one_entry_per_engine_per_stratum(self) -> None:
        b = _make_benchmark(
            [
                _make_engine("a", {"d1": 0.1, "d2": 0.2, "d3": 0.3}),
                _make_engine("b", {"d1": 0.5, "d2": 0.6, "d3": 0.7}),
            ],
            doc_strata={"d1": "S1", "d2": "S1", "d3": "S2"},
        )
        out = b.stratified_ranking()
        assert set(out.keys()) == {"S1", "S2"}
        for stratum, entries in out.items():
            assert len(entries) == 2

    def test_metrics_are_per_stratum(self) -> None:
        b = _make_benchmark(
            [_make_engine("a", {"d1": 0.0, "d2": 0.0, "d3": 0.5})],
            doc_strata={"d1": "S1", "d2": "S1", "d3": "S2"},
        )
        out = b.stratified_ranking()
        s1 = out["S1"][0]
        s2 = out["S2"][0]
        assert s1["mean_cer"] == pytest.approx(0.0)
        assert s1["median_cer"] == pytest.approx(0.0)
        assert s1["documents"] == 2
        assert s2["mean_cer"] == pytest.approx(0.5)
        assert s2["documents"] == 1

    def test_sorts_by_median_within_each_stratum(self) -> None:
        # Sur S1, A médiane=0.0, B médiane=0.1 → A 1er
        # Sur S2, A médiane=0.5, B médiane=0.0 → B 1er (changement de leader)
        b = _make_benchmark(
            [
                _make_engine("a", {"d1": 0.0, "d2": 0.0, "d3": 0.5, "d4": 0.5}),
                _make_engine("b", {"d1": 0.1, "d2": 0.1, "d3": 0.0, "d4": 0.0}),
            ],
            doc_strata={"d1": "S1", "d2": "S1", "d3": "S2", "d4": "S2"},
        )
        out = b.stratified_ranking()
        assert out["S1"][0]["engine"] == "a"
        assert out["S2"][0]["engine"] == "b"

    def test_engine_with_no_docs_in_stratum_appears_with_none(self) -> None:
        # B n'a aucun doc dans S2
        b = _make_benchmark(
            [
                _make_engine("a", {"d1": 0.0, "d2": 0.0, "d3": 0.5}),
                _make_engine("b", {"d1": 0.1, "d2": 0.1}),
            ],
            doc_strata={"d1": "S1", "d2": "S1", "d3": "S2"},
        )
        out = b.stratified_ranking()
        s2_b = next(e for e in out["S2"] if e["engine"] == "b")
        assert s2_b["mean_cer"] is None
        assert s2_b["median_cer"] is None
        assert s2_b["documents"] == 0


# ──────────────────────────────────────────────────────────────────────────
# 4. corpus_homogeneity
# ──────────────────────────────────────────────────────────────────────────


class TestCorpusHomogeneity:
    def test_returns_none_when_no_strata(self) -> None:
        b = _make_benchmark([_make_engine("a", {"d1": 0.1})])
        assert b.corpus_homogeneity() is None

    def test_returns_none_when_single_stratum(self) -> None:
        b = _make_benchmark(
            [_make_engine("a", {"d1": 0.1, "d2": 0.2})],
            doc_strata={"d1": "S1", "d2": "S1"},
        )
        assert b.corpus_homogeneity() is None

    def test_detects_inter_stratum_gap(self) -> None:
        # Le leader A : médiane = 0.0 sur S1, 0.5 sur S2 → gap = 0.5
        b = _make_benchmark(
            [_make_engine("a", {"d1": 0.0, "d2": 0.0, "d3": 0.5, "d4": 0.5})],
            doc_strata={"d1": "S1", "d2": "S1", "d3": "S2", "d4": "S2"},
        )
        h = b.corpus_homogeneity()
        assert h is not None
        assert h["leader"] == "a"
        assert h["n_strata"] == 2
        assert h["max_inter_strata_gap"] == pytest.approx(0.5)
        assert set(h["leader_max_gap_strata"]) == {"S1", "S2"}
        # Min en premier (S1 = 0.0), max en deuxième (S2 = 0.5)
        assert h["leader_max_gap_strata"][0] == "S1"
        assert h["leader_max_gap_strata"][1] == "S2"


# ──────────────────────────────────────────────────────────────────────────
# 5. as_dict expose les strates
# ──────────────────────────────────────────────────────────────────────────


class TestAsDictSerialization:
    def test_no_strata_keys_when_doc_strata_is_none(self) -> None:
        b = _make_benchmark([_make_engine("a", {"d1": 0.1})])
        d = b.as_dict()
        assert "doc_strata" not in d
        assert "stratified_ranking" not in d
        assert "corpus_homogeneity" not in d

    def test_strata_keys_present_when_doc_strata_is_set(self) -> None:
        b = _make_benchmark(
            [_make_engine("a", {"d1": 0.0, "d2": 0.5})],
            doc_strata={"d1": "S1", "d2": "S2"},
        )
        d = b.as_dict()
        assert d["doc_strata"] == {"d1": "S1", "d2": "S2"}
        assert d["available_strata"] == ["S1", "S2"]
        assert "S1" in d["stratified_ranking"]


# ──────────────────────────────────────────────────────────────────────────
# 6. Test propriété — leader change selon la strate (cas réaliste)
# ──────────────────────────────────────────────────────────────────────────


class TestRealisticAsymmetry:
    def test_global_leader_can_lose_on_a_stratum(self) -> None:
        """Cas patrimonial typique : Tesseract domine globalement
        (médiocre sur le manuscrit, excellent sur l'imprimé), Pero
        domine spécifiquement sur le manuscrit."""
        b = _make_benchmark(
            [
                # Tesseract : 10 docs imprimés à 0.02, 5 docs manuscrit à 0.30
                _make_engine("tesseract", {
                    **{f"print_{i}": 0.02 for i in range(10)},
                    **{f"ms_{i}": 0.30 for i in range(5)},
                }),
                # Pero : 10 docs imprimés à 0.05, 5 docs manuscrit à 0.10
                _make_engine("pero", {
                    **{f"print_{i}": 0.05 for i in range(10)},
                    **{f"ms_{i}": 0.10 for i in range(5)},
                }),
            ],
            doc_strata={
                **{f"print_{i}": "imprimé" for i in range(10)},
                **{f"ms_{i}": "manuscrit" for i in range(5)},
            },
        )

        # Globalement, Tesseract gagne sur la médiane (0.02 sur la
        # majorité des docs vs 0.05 pour Pero)
        global_leader = b.ranking()[0]["engine"]
        assert global_leader == "tesseract"

        # Mais sur la strate manuscrit, Pero gagne (0.10 < 0.30)
        strat = b.stratified_ranking()
        assert strat["manuscrit"][0]["engine"] == "pero"
        assert strat["imprimé"][0]["engine"] == "tesseract"

        # Le score d'homogénéité doit refléter le fort écart de
        # Tesseract entre strates (0.02 vs 0.30 = 0.28)
        h = b.corpus_homogeneity()
        assert h["leader"] == "tesseract"
        assert h["max_inter_strata_gap"] == pytest.approx(0.28, abs=1e-9)
