"""Tests Sprint 46 — vue HTML stratifiée + détecteur narratif.

Couvre :

1. ``build_stratified_ranking_html`` rend un ``<details>`` par strate
   avec tableau moteur × (médiane, moyenne, docs).
2. Bandeau d'hétérogénéité affiché si ``corpus_homogeneity`` fourni.
3. **Masquage adaptatif** : retourne ``""`` si pas de strates.
4. **Anti-injection** : noms de strates et de moteurs avec balises
   HTML sont échappés.
5. **Détecteur ``STRATIFICATION_RECOMMENDED``** :
   - se déclenche au-delà de 5 points d'écart inter-strate
   - importance HIGH au-delà de 10 points, MEDIUM sinon
   - ne se déclenche pas sans corpus_homogeneity
6. **Anti-hallucination** : chaque nombre rendu est dans le payload.
7. **Intégration ReportGenerator** : la section apparaît dans
   ``view_ranking`` quand ``doc_strata`` est peuplé.
8. **i18n FR/EN** : clés présentes pour la vue + le template narratif.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from picarones.core.metrics import MetricsResult
from picarones.core.narrative.detectors import detect_stratification_recommended
from picarones.core.narrative.facts import FactImportance, FactType
from picarones.core.narrative.renderer import extract_numbers, render_fact
from picarones.core.results import DocumentResult
from picarones.report.generator import ReportGenerator
from picarones.report.stratification_render import build_stratified_ranking_html


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


_SAMPLE_STRAT = {
    "gothique": [
        {"engine": "pero", "median_cer": 0.05, "mean_cer": 0.07, "documents": 10},
        {"engine": "tess", "median_cer": 0.20, "mean_cer": 0.22, "documents": 10},
    ],
    "imprimé": [
        {"engine": "tess", "median_cer": 0.02, "mean_cer": 0.03, "documents": 10},
        {"engine": "pero", "median_cer": 0.05, "mean_cer": 0.06, "documents": 10},
    ],
}
_SAMPLE_STRATA = ["gothique", "imprimé"]
_SAMPLE_HOMOG = {
    "leader": "tess",
    "n_strata": 2,
    "max_inter_strata_gap": 0.18,
    "leader_max_gap_strata": ["imprimé", "gothique"],
    "leader_per_stratum_median": {"imprimé": 0.02, "gothique": 0.20},
}


def _make_dr(doc_id: str, cer: float) -> DocumentResult:
    return DocumentResult(
        doc_id=doc_id, image_path=f"/tmp/{doc_id}.png",
        ground_truth="x", hypothesis="x",
        metrics=MetricsResult(
            cer=cer, cer_nfc=cer, cer_caseless=cer,
            wer=cer, wer_normalized=cer, mer=cer, wil=cer,
            reference_length=1, hypothesis_length=1,
        ),
        duration_seconds=0.1,
    )




# ──────────────────────────────────────────────────────────────────────────
# 1-2. build_stratified_ranking_html
# ──────────────────────────────────────────────────────────────────────────


class TestRendering:
    def test_renders_one_details_per_stratum(self) -> None:
        html = build_stratified_ranking_html(
            _SAMPLE_STRAT, _SAMPLE_STRATA, _SAMPLE_HOMOG,
        )
        assert html.count("<details") == 2
        # Premier ouvert
        assert "<details" in html and " open" in html

    def test_includes_engine_metrics(self) -> None:
        html = build_stratified_ranking_html(
            _SAMPLE_STRAT, _SAMPLE_STRATA, _SAMPLE_HOMOG,
        )
        # Médianes en pourcentage
        assert "5.00 %" in html   # pero gothique
        assert "20.00 %" in html  # tess gothique
        assert "2.00 %" in html   # tess imprimé

    def test_homogeneity_banner_present(self) -> None:
        html = build_stratified_ranking_html(
            _SAMPLE_STRAT, _SAMPLE_STRATA, _SAMPLE_HOMOG,
        )
        # Le bandeau d'avertissement doit apparaître
        assert "tess" in html
        assert "18.0" in html

    def test_no_homogeneity_no_banner(self) -> None:
        html = build_stratified_ranking_html(
            _SAMPLE_STRAT, _SAMPLE_STRATA, homogeneity=None,
        )
        # Pas de bandeau jaune
        assert "#fff8e1" not in html

    def test_uses_i18n_labels(self) -> None:
        labels = {
            "stratification_caption": "CUSTOM_CAPTION",
            "stratification_median_label": "MED",
            "stratification_mean_label": "MEAN",
        }
        html = build_stratified_ranking_html(
            _SAMPLE_STRAT, _SAMPLE_STRATA, None, labels=labels,
        )
        assert "CUSTOM_CAPTION" in html
        assert "MED" in html
        assert "MEAN" in html


# ──────────────────────────────────────────────────────────────────────────
# 3. Masquage adaptatif
# ──────────────────────────────────────────────────────────────────────────


class TestAdaptiveMasking:
    def test_empty_when_no_stratified_ranking(self) -> None:
        assert build_stratified_ranking_html(None, ["S1"]) == ""
        assert build_stratified_ranking_html({}, ["S1"]) == ""

    def test_empty_when_no_available_strata(self) -> None:
        assert build_stratified_ranking_html(_SAMPLE_STRAT, None) == ""
        assert build_stratified_ranking_html(_SAMPLE_STRAT, []) == ""


# ──────────────────────────────────────────────────────────────────────────
# 4. Anti-injection
# ──────────────────────────────────────────────────────────────────────────


class TestAntiInjection:
    def test_engine_name_escaped(self) -> None:
        bad_strat = {
            "S1": [
                {"engine": "<script>alert(1)</script>",
                 "median_cer": 0.1, "mean_cer": 0.1, "documents": 1},
            ],
        }
        html = build_stratified_ranking_html(bad_strat, ["S1"])
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_stratum_name_escaped(self) -> None:
        bad_strat = {
            "<img src=x>": [
                {"engine": "a", "median_cer": 0.1,
                 "mean_cer": 0.1, "documents": 1},
            ],
        }
        html = build_stratified_ranking_html(bad_strat, ["<img src=x>"])
        assert "<img src=x>" not in html
        assert "&lt;img" in html


# ──────────────────────────────────────────────────────────────────────────
# 5. Détecteur STRATIFICATION_RECOMMENDED
# ──────────────────────────────────────────────────────────────────────────


def _data(gap: float, **overrides) -> dict:
    homog = {
        "leader": "tess", "n_strata": 2,
        "max_inter_strata_gap": gap,
        "leader_max_gap_strata": ["S1", "S2"],
        "leader_per_stratum_median": {"S1": 0.02, "S2": 0.02 + gap},
    }
    homog.update(overrides)
    return {"corpus_homogeneity": homog}


class TestStratificationDetector:
    def test_no_fact_below_threshold(self) -> None:
        # 4 points → en dessous du seuil 5 points
        assert detect_stratification_recommended(_data(0.04)) == []

    def test_emits_fact_above_threshold(self) -> None:
        facts = detect_stratification_recommended(_data(0.07))
        assert len(facts) == 1
        assert facts[0].type is FactType.STRATIFICATION_RECOMMENDED

    def test_medium_below_10pts(self) -> None:
        facts = detect_stratification_recommended(_data(0.07))
        assert facts[0].importance is FactImportance.MEDIUM

    def test_high_above_10pts(self) -> None:
        facts = detect_stratification_recommended(_data(0.18))
        assert facts[0].importance is FactImportance.HIGH

    def test_no_homogeneity_no_fact(self) -> None:
        assert detect_stratification_recommended({}) == []
        assert detect_stratification_recommended({"corpus_homogeneity": None}) == []

    def test_payload_carries_strata_and_cers(self) -> None:
        facts = detect_stratification_recommended(_data(0.18))
        p = facts[0].payload
        assert p["leader"] == "tess"
        assert p["n_strata"] == 2
        assert p["min_stratum"] == "S1"
        assert p["max_stratum"] == "S2"
        assert p["gap_pct"] == 18.0


# ──────────────────────────────────────────────────────────────────────────
# 6. Anti-hallucination
# ──────────────────────────────────────────────────────────────────────────


class TestTraceability:
    @pytest.mark.parametrize("lang", ["fr", "en"])
    def test_every_rendered_number_is_in_payload(self, lang: str) -> None:
        # On utilise des noms de strates sans chiffres (la traçabilité
        # exige que tout chiffre rendu vienne du payload, mais les
        # noms de strates côté GT peuvent légitimement contenir des
        # chiffres ; pour le test on isole les nombres "métriques").
        data = {"corpus_homogeneity": {
            "leader": "tess", "n_strata": 2,
            "max_inter_strata_gap": 0.18,
            "leader_max_gap_strata": ["impr", "goth"],
            "leader_per_stratum_median": {"impr": 0.02, "goth": 0.20},
        }}
        facts = detect_stratification_recommended(data)
        sentence = render_fact(facts[0], lang)

        payload_nums: set[str] = set()
        for v in facts[0].payload.values():
            if isinstance(v, (int, float)):
                payload_nums.add(str(v))
                if isinstance(v, float) and v.is_integer():
                    payload_nums.add(str(int(v)))
            elif isinstance(v, str):
                # Capture aussi les chiffres présents dans les chaînes
                # du payload (ex. noms de strates contenant un nombre)
                for match in re.findall(r"\d+(?:[.,]\d+)?", v):
                    payload_nums.add(match.replace(",", "."))

        for num in extract_numbers(sentence):
            normalized = num.replace(",", ".")
            assert normalized in payload_nums, (
                f"Nombre {normalized!r} non traçable au payload "
                f"{facts[0].payload!r}"
            )

    def test_template_has_no_hardcoded_numbers(self) -> None:
        from picarones.core.narrative.renderer import _load_templates
        for lang in ("fr", "en"):
            tpl = _load_templates(lang).get("stratification_recommended", "")
            assert tpl, f"Template absent pour {lang}"
            cleaned = re.sub(r"\{[^}]+\}", "", tpl)
            digits = re.findall(r"\d", cleaned)
            assert not digits, f"Template {lang} contient des chiffres en dur : {digits}"


# ──────────────────────────────────────────────────────────────────────────
# 7. Intégration ReportGenerator
# ──────────────────────────────────────────────────────────────────────────


class TestReportIntegration:
    def test_section_absent_without_strata(self, tmp_path: Path) -> None:
        from picarones.fixtures import generate_sample_benchmark
        bench = generate_sample_benchmark()
        bench.doc_strata = None  # force absence
        out = tmp_path / "report.html"
        ReportGenerator(bench).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "stratified-ranking" not in html

    def test_section_present_with_strata(self, tmp_path: Path) -> None:
        from picarones.fixtures import generate_sample_benchmark
        bench = generate_sample_benchmark()
        # La fixture peuple image_quality.script_type ; on extrait
        # manuellement comme le ferait le runner.
        strata_map: dict[str, str] = {}
        for r in bench.engine_reports:
            for dr in r.document_results:
                if dr.image_quality and dr.image_quality.get("script_type"):
                    strata_map.setdefault(dr.doc_id, dr.image_quality["script_type"])
        bench.doc_strata = strata_map

        out = tmp_path / "report.html"
        ReportGenerator(bench).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "stratified-ranking" in html
        # Au moins un <details> rendu
        assert "<details" in html

    def test_french_locale_uses_french_labels(self, tmp_path: Path) -> None:
        from picarones.fixtures import generate_sample_benchmark
        bench = generate_sample_benchmark()
        strata_map = {}
        for r in bench.engine_reports:
            for dr in r.document_results:
                if dr.image_quality and dr.image_quality.get("script_type"):
                    strata_map.setdefault(dr.doc_id, dr.image_quality["script_type"])
        bench.doc_strata = strata_map

        out = tmp_path / "report_fr.html"
        ReportGenerator(bench, lang="fr").generate(out)
        html = out.read_text(encoding="utf-8")
        assert "Classement par strate" in html
        assert "Médiane CER" in html

    def test_english_locale_uses_english_labels(self, tmp_path: Path) -> None:
        from picarones.fixtures import generate_sample_benchmark
        bench = generate_sample_benchmark()
        strata_map = {}
        for r in bench.engine_reports:
            for dr in r.document_results:
                if dr.image_quality and dr.image_quality.get("script_type"):
                    strata_map.setdefault(dr.doc_id, dr.image_quality["script_type"])
        bench.doc_strata = strata_map

        out = tmp_path / "report_en.html"
        ReportGenerator(bench, lang="en").generate(out)
        html = out.read_text(encoding="utf-8")
        assert "Ranking by stratum" in html
        assert "Median CER" in html


# ──────────────────────────────────────────────────────────────────────────
# 8. i18n FR/EN
# ──────────────────────────────────────────────────────────────────────────


REQUIRED_KEYS = (
    "stratification_caption",
    "stratification_description",
    "stratification_median_label",
    "stratification_mean_label",
    "stratification_docs_label",
    "stratification_no_data_label",
    "stratification_n_docs_label",
    "stratification_gap_summary",
)


class TestI18NCompleteness:
    @pytest.mark.parametrize("lang", ["fr", "en"])
    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_key_present(self, lang: str, key: str) -> None:
        path = (
            Path(__file__).parent.parent
            / "picarones" / "report" / "i18n" / f"{lang}.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert key in data, f"Clé {key!r} manquante dans {lang}.json"
        assert data[key].strip(), f"Clé {key!r} vide dans {lang}.json"
