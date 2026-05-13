"""Phase 3.2 audit code-quality — end-to-end du journal de fallback.

Vérifie que la chaîne complète fonctionne :

1. Un importer (HTR-United) dégrade en mode démo →
   ``record_fallback`` côté importer.
2. Le runner consomme via ``consume_fallback_log()`` et stocke dans
   ``BenchmarkResult.metadata["importer_fallbacks"]``.
3. ``build_report_data`` propage la liste dans
   ``report_data["importer_fallbacks"]``.
4. Le détecteur narratif ``detect_importer_fallback`` (history.py:280)
   produit un ``Fact(FactType.IMPORTER_FALLBACK_TRIGGERED, ...)``.
5. ``build_synthesis`` rend une phrase qui mentionne l'incident.

Avant la Phase 3.2 : étapes 2-3 manquaient — le détecteur ne
recevait jamais de données malgré l'API ``_fallback_log`` câblée
côté importer.
"""

from __future__ import annotations

import pytest

from picarones.adapters.corpus._fallback_log import (
    consume_fallback_log,
    peek_fallback_log,
    record_fallback,
    reset_fallback_log,
)
from picarones.domain.facts import FactType
from picarones.evaluation.benchmark_result import BenchmarkResult
from picarones.reports.html.data import build_report_data
from picarones.reports.narrative import build_synthesis
from picarones.reports.narrative.detectors.history import detect_importer_fallback


@pytest.fixture(autouse=True)
def _clean_fallback_log() -> None:
    """Le journal est un singleton thread-safe — on le vide avant
    et après chaque test pour éviter les contaminations croisées."""
    reset_fallback_log()
    yield
    reset_fallback_log()


# --------------------------------------------------------------------------
# Étape 1 : record_fallback est appelable + sérialise correctement
# --------------------------------------------------------------------------


def test_record_fallback_appends_entry() -> None:
    record_fallback(
        importer="htr_united",
        operation="catalogue_remote_fetch",
        error=RuntimeError("DNS timeout"),
        extra={"url": "https://example.org/cat.yml"},
    )
    entries = peek_fallback_log()
    assert len(entries) == 1
    assert entries[0]["importer"] == "htr_united"
    assert entries[0]["operation"] == "catalogue_remote_fetch"
    assert "DNS timeout" in entries[0]["error"]
    assert entries[0]["extra"]["url"] == "https://example.org/cat.yml"


def test_htr_united_fallback_records_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """``HTRUnitedCatalogue.from_remote`` doit appeler ``record_fallback``
    quand le réseau échoue (régression : avant Phase 3.2 le warning
    log était là, le record manquait)."""
    import urllib.error

    from picarones.adapters.corpus.htr_united import HTRUnitedCatalogue

    def _boom(*_a, **_kw):
        raise urllib.error.URLError("simulated DNS failure")

    monkeypatch.setattr(
        "picarones.adapters.corpus.htr_united.urllib.request.urlopen",
        _boom,
    )
    cat = HTRUnitedCatalogue.from_remote(timeout=1)
    assert cat.source == "demo"  # fallback effectif

    entries = peek_fallback_log()
    assert len(entries) == 1
    assert entries[0]["importer"] == "htr_united"
    assert entries[0]["operation"] == "catalogue_remote_fetch"
    assert entries[0]["extra"]["fallback_used"] == "demo"


# --------------------------------------------------------------------------
# Étape 4 : le détecteur narratif émet un Fact à partir de la liste
# --------------------------------------------------------------------------


def test_detector_emits_fact_from_benchmark_data() -> None:
    benchmark_data = {
        "importer_fallbacks": [
            {
                "importer": "htr_united",
                "operation": "catalogue_remote_fetch",
                "error": "URLError(...)",
                "extra": {"fallback_used": "demo"},
            },
        ],
    }
    facts = detect_importer_fallback(benchmark_data)
    assert len(facts) == 1
    assert facts[0].type is FactType.IMPORTER_FALLBACK_TRIGGERED
    assert facts[0].payload["importer"] == "htr_united"


def test_detector_silent_when_no_fallback() -> None:
    """Pas de clé → pas de Fact."""
    assert detect_importer_fallback({}) == []
    assert detect_importer_fallback({"importer_fallbacks": []}) == []


# --------------------------------------------------------------------------
# Étape 3 : build_report_data propage metadata.importer_fallbacks
# --------------------------------------------------------------------------


def _empty_benchmark_with_metadata(metadata: dict) -> BenchmarkResult:
    """Benchmark sans engine (suffisant pour tester la propagation
    de ``metadata.importer_fallbacks`` vers ``report_data``)."""
    return BenchmarkResult(
        corpus_name="t",
        corpus_source=None,
        document_count=0,
        engine_reports=[],
        metadata=metadata,
    )


def test_build_report_data_propagates_fallbacks() -> None:
    bench = _empty_benchmark_with_metadata({
        "importer_fallbacks": [
            {"importer": "htr_united", "operation": "catalogue_remote_fetch",
             "error": "URLError(timeout)"},
        ],
    })
    data = build_report_data(bench, images_b64={})
    assert "importer_fallbacks" in data
    assert len(data["importer_fallbacks"]) == 1
    assert data["importer_fallbacks"][0]["importer"] == "htr_united"


def test_build_report_data_empty_when_no_fallback() -> None:
    bench = _empty_benchmark_with_metadata({})
    data = build_report_data(bench, images_b64={})
    assert data["importer_fallbacks"] == []


# --------------------------------------------------------------------------
# Étape 5 : build_synthesis fait remonter l'incident dans la prose
# --------------------------------------------------------------------------


def test_build_synthesis_mentions_fallback_in_french() -> None:
    """La synthèse française doit produire au moins un fragment
    textuel qui mentionne l'importer en mode dégradé."""
    data = {
        "engines": [],
        "ranking": [],
        "importer_fallbacks": [
            {
                "importer": "htr_united",
                "operation": "catalogue_remote_fetch",
                "error": "URLError(timeout)",
                "extra": {"fallback_used": "demo"},
            },
        ],
    }
    out = build_synthesis(data, lang="fr", max_facts=5)
    # Le texte rendu doit contenir au moins le nom de l'importer.
    rendered = " ".join(out.get("paragraphs", []) or []) + " " + str(out)
    assert "htr_united" in rendered.lower() or "htr-united" in rendered.lower(), (
        f"La synthèse FR ne mentionne pas l'importer HTR-United malgré "
        f"un fallback enregistré.  Sortie : {out!r}"
    )


# --------------------------------------------------------------------------
# Étape 2 : consume vide bien la liste (anti-contamination cross-run)
# --------------------------------------------------------------------------


def test_consume_clears_the_log() -> None:
    record_fallback(importer="a", operation="x")
    record_fallback(importer="b", operation="y")
    first = consume_fallback_log()
    assert len(first) == 2

    second = consume_fallback_log()
    assert second == []  # vidé par le premier consume
