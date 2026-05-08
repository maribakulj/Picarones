"""Tests Sprint 28 — UX : save/load config, compare runs, synthesis preview.

Le Sprint 28 réduit la friction du chercheur qui itère sur 8 prompts :

1. ``/api/config/save`` + ``/api/config/load`` — sérialisation/import
   d'une configuration de benchmark en JSON.
2. ``picarones/report/comparison.py`` — comparaison de deux runs JSON
   avec deltas par moteur et détection de régressions.
3. ``picarones compare`` (CLI) — équivalent ligne de commande.
4. ``/api/benchmark/{job_id}/synthesis_preview`` — synthèse narrative
   d'un job terminé sans rouvrir le HTML.
5. ``/api/history/regressions`` — surface de l'infrastructure Sprint 8.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# 1. Module comparison — compare_benchmarks
# ---------------------------------------------------------------------------

def _benchmark_json(engines_to_cer: dict[str, float], **extra) -> dict:
    """Fabrique un dict ``BenchmarkResult.as_dict()``-like."""
    ranking = [
        {
            "engine": name,
            "mean_cer": cer,
            "mean_wer": cer * 1.5 if cer is not None else None,
            "documents": 10,
            "failed": 0,
        }
        for name, cer in engines_to_cer.items()
    ]
    return {
        "ranking": ranking,
        "run_date": extra.get("run_date", "2026-04-01T00:00:00+00:00"),
        "corpus": {"name": extra.get("corpus", "test_corpus"), "source": "fixture"},
    }


class TestCompareBenchmarks:
    def test_identical_runs_no_regression(self):
        from picarones.reports_v2.html.comparison import compare_benchmarks
        a = _benchmark_json({"tesseract": 0.05, "pero": 0.07})
        b = _benchmark_json({"tesseract": 0.05, "pero": 0.07})
        diff = compare_benchmarks(a, b, threshold=0.005)
        assert all(d.delta_cer == 0.0 for d in diff.deltas)
        assert all(not d.is_regression and not d.is_improvement for d in diff.deltas)

    def test_regression_detected_above_threshold(self):
        from picarones.reports_v2.html.comparison import compare_benchmarks, detect_regressions
        a = _benchmark_json({"tesseract": 0.05})
        b = _benchmark_json({"tesseract": 0.06})  # +1 pp
        diff = compare_benchmarks(a, b, threshold=0.005)
        regs = detect_regressions(diff)
        assert len(regs) == 1
        assert regs[0].engine == "tesseract"
        assert regs[0].delta_cer == pytest.approx(0.01, abs=1e-9)

    def test_improvement_detected_below_threshold(self):
        from picarones.reports_v2.html.comparison import compare_benchmarks
        a = _benchmark_json({"tesseract": 0.05})
        b = _benchmark_json({"tesseract": 0.04})  # -1 pp
        diff = compare_benchmarks(a, b, threshold=0.005)
        assert diff.deltas[0].is_improvement
        assert not diff.deltas[0].is_regression

    def test_below_threshold_is_stable(self):
        from picarones.reports_v2.html.comparison import compare_benchmarks
        a = _benchmark_json({"tesseract": 0.05})
        b = _benchmark_json({"tesseract": 0.052})  # +0.2 pp, sous le seuil 0.5 pp
        diff = compare_benchmarks(a, b, threshold=0.005)
        assert not diff.deltas[0].is_regression

    def test_engines_only_in_one_side(self):
        from picarones.reports_v2.html.comparison import compare_benchmarks
        a = _benchmark_json({"tesseract": 0.05, "pero": 0.07})
        b = _benchmark_json({"tesseract": 0.05, "kraken": 0.06})
        diff = compare_benchmarks(a, b, threshold=0.005)
        assert diff.only_in_a == ["pero"]
        assert diff.only_in_b == ["kraken"]
        assert {d.engine for d in diff.deltas} == {"tesseract"}

    def test_none_cer_does_not_raise(self):
        from picarones.reports_v2.html.comparison import compare_benchmarks
        a = _benchmark_json({"tesseract": None})
        b = _benchmark_json({"tesseract": 0.05})
        diff = compare_benchmarks(a, b)
        assert diff.deltas[0].delta_cer is None
        assert not diff.deltas[0].is_regression

    def test_regressions_sorted_by_severity(self):
        from picarones.reports_v2.html.comparison import compare_benchmarks
        a = _benchmark_json({"a": 0.05, "b": 0.05, "c": 0.05})
        b = _benchmark_json({"a": 0.07, "b": 0.10, "c": 0.06})  # b plus grave
        diff = compare_benchmarks(a, b, threshold=0.005)
        # Régressions en tête, plus grosse d'abord
        engines_in_order = [d.engine for d in diff.deltas]
        assert engines_in_order.index("b") < engines_in_order.index("a")

    def test_loads_from_file_path(self, tmp_path):
        from picarones.reports_v2.html.comparison import compare_benchmarks
        a_path = tmp_path / "a.json"
        b_path = tmp_path / "b.json"
        a_path.write_text(json.dumps(_benchmark_json({"tesseract": 0.05})))
        b_path.write_text(json.dumps(_benchmark_json({"tesseract": 0.06})))
        diff = compare_benchmarks(a_path, b_path, threshold=0.005)
        assert len(diff.deltas) == 1


class TestRenderComparisonHTML:
    def test_html_is_self_contained_and_named(self, tmp_path):
        from picarones.reports_v2.html.comparison import compare_benchmarks, render_comparison_html
        a = _benchmark_json({"tesseract": 0.05})
        b = _benchmark_json({"tesseract": 0.07})
        diff = compare_benchmarks(a, b, label_a="V1", label_b="V2")
        out = tmp_path / "diff.html"
        render_comparison_html(diff, out)
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        # Étiquettes et delta visible
        assert "V1" in html and "V2" in html
        assert "+0.020" in html  # delta CER affiché
        assert "régression" in html.lower()


# ---------------------------------------------------------------------------
# 2. CLI picarones compare
# ---------------------------------------------------------------------------

class TestCompareCLI:
    def test_basic_compare_writes_html(self, tmp_path):
        from picarones.interfaces.cli._legacy import cli
        a_path = tmp_path / "a.json"
        b_path = tmp_path / "b.json"
        a_path.write_text(json.dumps(_benchmark_json({"tesseract": 0.05})))
        b_path.write_text(json.dumps(_benchmark_json({"tesseract": 0.05})))
        out = tmp_path / "out.html"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "compare", str(a_path), str(b_path), "-o", str(out),
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_regression_exit_code_is_2(self, tmp_path):
        from picarones.interfaces.cli._legacy import cli
        a_path = tmp_path / "a.json"
        b_path = tmp_path / "b.json"
        a_path.write_text(json.dumps(_benchmark_json({"tesseract": 0.05})))
        b_path.write_text(json.dumps(_benchmark_json({"tesseract": 0.10})))
        runner = CliRunner()
        result = runner.invoke(cli, [
            "compare", str(a_path), str(b_path),
            "-o", str(tmp_path / "out.html"),
        ])
        # exit 2 = régression détectée (utile en CI)
        assert result.exit_code == 2, result.output
        assert "régression" in result.output.lower() or "tesseract" in result.output.lower()

    def test_json_mode_outputs_dict(self, tmp_path):
        from picarones.interfaces.cli._legacy import cli
        a_path = tmp_path / "a.json"
        b_path = tmp_path / "b.json"
        a_path.write_text(json.dumps(_benchmark_json({"tesseract": 0.05})))
        b_path.write_text(json.dumps(_benchmark_json({"tesseract": 0.05})))
        runner = CliRunner()
        result = runner.invoke(cli, [
            "compare", str(a_path), str(b_path), "--json",
        ])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert "deltas" in parsed
        assert "regressions" in parsed


# ---------------------------------------------------------------------------
# 3. /api/config/save + /api/config/load
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from picarones.interfaces.web._legacy.app import app
    return TestClient(app)


class TestConfigSaveLoad:
    def test_save_returns_attachment(self, client):
        r = client.post("/api/config/save", json={
            "engines": ["tesseract"],
            "normalization_profile": "medieval_french",
            "label": "test-1",
        })
        assert r.status_code == 200
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "test-1" in cd
        body = r.json()
        assert body["schema_version"] == 1
        assert body["engines"] == ["tesseract"]
        assert "saved_at" in body

    def test_save_filters_unknown_fields(self, client):
        r = client.post("/api/config/save", json={
            "engines": ["tesseract"],
            "secret_token": "ne-doit-pas-apparaitre",
            "OPENAI_API_KEY": "sk-...",
        })
        body = r.json()
        assert "secret_token" not in body
        assert "OPENAI_API_KEY" not in body

    def test_save_sanitizes_label_for_filename(self, client):
        r = client.post("/api/config/save", json={
            "label": "../../etc/passwd",
        })
        cd = r.headers.get("content-disposition", "")
        assert ".." not in cd
        assert "etc" in cd or "passwd" in cd  # caractères alnum gardés

    def test_load_round_trip(self, client):
        original = {
            "engines": ["tesseract", "pero"],
            "normalization_profile": "medieval_french",
            "char_exclude": "',-",
            "lang": "fra",
        }
        # 1. save
        r1 = client.post("/api/config/save", json=original)
        saved = r1.json()
        # 2. load
        r2 = client.post("/api/config/load", json=saved)
        assert r2.status_code == 200
        loaded = r2.json()["config"]
        # Les champs originaux survivent au round-trip
        for k, v in original.items():
            assert loaded[k] == v

    def test_load_rejects_bad_schema_version(self, client):
        r = client.post("/api/config/load", json={"schema_version": 99})
        assert r.status_code == 400
        assert "schema" in r.json()["detail"].lower()

    def test_load_rejects_missing_schema(self, client):
        r = client.post("/api/config/load", json={"engines": ["tesseract"]})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# 4. /api/benchmark/{job_id}/synthesis_preview
# ---------------------------------------------------------------------------

class TestSynthesisPreviewEndpoint:
    @pytest.fixture
    def job_with_results(self, monkeypatch, tmp_path):
        """Crée un job 'complete' + JSON résultat sur disque."""
        from picarones import fixtures
        from picarones.interfaces.web._legacy.jobs import get_default_store, reset_default_store
        from picarones.interfaces.web._legacy import app as web_app
        from picarones.interfaces.web._legacy import state as web_state
        # Isolate store
        monkeypatch.setenv("PICARONES_JOBS_DB", str(tmp_path / "jobs.db"))
        reset_default_store()
        web_state.JOB_STORE = get_default_store()
        web_state.JOBS.clear()
        # Génère un benchmark + écrit son JSON
        b = fixtures.generate_sample_benchmark(n_docs=4)
        out_dir = tmp_path / "rep"
        out_dir.mkdir()
        html_path = out_dir / "report.html"
        json_path = html_path.with_suffix(".json")
        json_path.write_text(json.dumps(b.as_dict(), ensure_ascii=False))
        # Crée le job en base
        jid = web_state.JOB_STORE.create_job(job_id="job-prev-1")
        web_state.JOB_STORE.set_status(jid, "complete")
        web_state.JOB_STORE.update_progress(jid, output_path=str(html_path))
        return TestClient(web_app.app), jid

    def test_returns_synthesis_for_complete_job(self, job_with_results):
        client, jid = job_with_results
        r = client.get(f"/api/benchmark/{jid}/synthesis_preview")
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["job_id"] == jid
        assert d["lang"] == "fr"
        assert "sentences" in d and isinstance(d["sentences"], list)

    def test_404_for_unknown_job(self, client):
        r = client.get("/api/benchmark/never-existed/synthesis_preview")
        assert r.status_code == 404

    def test_409_when_job_not_complete(self, monkeypatch, tmp_path):
        from picarones.interfaces.web._legacy.jobs import get_default_store, reset_default_store
        from picarones.interfaces.web._legacy import app as web_app
        from picarones.interfaces.web._legacy import state as web_state
        monkeypatch.setenv("PICARONES_JOBS_DB", str(tmp_path / "jobs.db"))
        reset_default_store()
        web_state.JOB_STORE = get_default_store()
        web_state.JOBS.clear()
        web_state.JOB_STORE.create_job(job_id="running-1")
        web_state.JOB_STORE.set_status("running-1", "running")
        client = TestClient(web_app.app)
        r = client.get("/api/benchmark/running-1/synthesis_preview")
        assert r.status_code == 409


# ---------------------------------------------------------------------------
# 5. /api/history/regressions
# ---------------------------------------------------------------------------

class TestHistoryRegressionsEndpoint:
    def test_empty_history_returns_zero(self, client, tmp_path):
        # Pas d'historique → 0 régression
        db = tmp_path / "history.db"
        r = client.get(f"/api/history/regressions?db_path={db}")
        assert r.status_code == 200
        d = r.json()
        assert d["count"] == 0
        assert d["regressions"] == []

    def test_threshold_param_is_propagated(self, client, tmp_path):
        db = tmp_path / "history.db"
        r = client.get(f"/api/history/regressions?threshold=0.05&db_path={db}")
        assert r.json()["threshold"] == pytest.approx(0.05)
