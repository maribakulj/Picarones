"""Tests Sprint 65 — comparaison de N pipelines sur un corpus.

Couvre :

1. ``compare_pipelines`` :
   - 1 pipeline → équivalent à ``run_pipeline_benchmark`` mais
     emballé dans un ``PipelineComparisonResult``
   - 2+ pipelines → résultats indexés par nom dans l'ordre
     d'insertion
   - Noms en double → ``ValueError`` explicite
   - ``factories`` par pipeline respecté
   - Corpus vide → résultats vides cohérents
2. ``ranking_by_final_metric`` :
   - Tri ascendant pour métriques de type CER (par défaut)
   - Tri descendant si ``higher_is_better=True``
   - Pipelines sans métrique → en queue, ordre préservé
3. ``gain_table`` :
   - ``baseline_pipeline`` inconnue → ``KeyError``
   - Baseline elle-même : absolute=0, relative=0
   - ``relative`` à ``None`` si baseline = 0
   - ``absolute`` et ``relative`` à ``None`` si valeur absente
4. Cas réaliste : OCR fautif vs OCR+correcteur → le correcteur
   gagne au ranking et au gain_table.
5. Philosophie inchangée : tous les modules sont des **mocks**
   définis dans le test.
"""

from __future__ import annotations

from typing import Any

import pytest

from picarones.core.corpus import Corpus, Document, GTLevel, TextGT
from picarones.core.modules import ArtifactType, BaseModule
from picarones.measurements.pipeline_comparison import (
    PipelineComparisonResult,
    compare_pipelines,
)
from picarones.core.pipeline import PipelineSpec, PipelineStep


# ──────────────────────────────────────────────────────────────────────────
# Mocks
# ──────────────────────────────────────────────────────────────────────────


class MockOCR(BaseModule):
    input_types = (ArtifactType.IMAGE,)
    output_types = (ArtifactType.TEXT,)
    execution_mode: Any = "io"

    def __init__(self, fn) -> None:
        self._fn = fn

    @property
    def name(self) -> str:
        return "mock-ocr"

    def process(self, inputs):
        return {ArtifactType.TEXT: self._fn(inputs[ArtifactType.IMAGE])}


class TextFixer(BaseModule):
    """Rewriter mock qui applique un dict de remplacements."""

    input_types = (ArtifactType.TEXT,)
    output_types = (ArtifactType.TEXT,)
    execution_mode: Any = "cpu"

    def __init__(self, replacements: dict[str, str]) -> None:
        self._replacements = replacements

    @property
    def name(self) -> str:
        return "fixer"

    def process(self, inputs):
        text = inputs[ArtifactType.TEXT]
        for src, dst in self._replacements.items():
            text = text.replace(src, dst)
        return {ArtifactType.TEXT: text}


def _make_corpus(n: int = 2, name: str = "demo") -> Corpus:
    docs = []
    for i in range(n):
        gt = f"texte {i}"
        docs.append(Document(
            image_path=f"/tmp/d{i}.png",
            ground_truth=gt,
            doc_id=f"d{i}",
            ground_truths={GTLevel.TEXT: TextGT(text=gt)},
        ))
    return Corpus(name=name, documents=docs)


def _ocr_perfect(path: str) -> str:
    idx = path.replace("/tmp/d", "").replace(".png", "")
    return f"texte {idx}"


def _ocr_with_typo(path: str) -> str:
    idx = path.replace("/tmp/d", "").replace(".png", "")
    return f"txete {idx}"


# ──────────────────────────────────────────────────────────────────────────
# 1. compare_pipelines — chemins nominaux
# ──────────────────────────────────────────────────────────────────────────


class TestCompareBasic:
    def test_single_pipeline(self) -> None:
        corpus = _make_corpus(2)
        spec = PipelineSpec(
            name="ocr_only",
            steps=[PipelineStep("ocr", MockOCR(_ocr_perfect))],
        )
        result = compare_pipelines([spec], corpus)
        assert result.corpus_name == "demo"
        assert result.n_docs == 2
        assert result.pipeline_names() == ["ocr_only"]
        assert "ocr_only" in result.per_pipeline

    def test_multiple_pipelines_preserved_order(self) -> None:
        corpus = _make_corpus(1)
        specs = [
            PipelineSpec("alpha", [PipelineStep("ocr", MockOCR(_ocr_perfect))]),
            PipelineSpec("beta", [PipelineStep("ocr", MockOCR(_ocr_perfect))]),
            PipelineSpec("gamma", [PipelineStep("ocr", MockOCR(_ocr_perfect))]),
        ]
        result = compare_pipelines(specs, corpus)
        assert result.pipeline_names() == ["alpha", "beta", "gamma"]

    def test_duplicate_names_raises(self) -> None:
        corpus = _make_corpus(1)
        specs = [
            PipelineSpec("dup", [PipelineStep("ocr", MockOCR(_ocr_perfect))]),
            PipelineSpec("dup", [PipelineStep("ocr", MockOCR(_ocr_perfect))]),
        ]
        with pytest.raises(ValueError, match="non uniques"):
            compare_pipelines(specs, corpus)

    def test_empty_corpus(self) -> None:
        corpus = Corpus(name="empty", documents=[])
        spec = PipelineSpec(
            name="ocr",
            steps=[PipelineStep("ocr", MockOCR(_ocr_perfect))],
        )
        result = compare_pipelines([spec], corpus)
        assert result.n_docs == 0
        assert "ocr" in result.per_pipeline


# ──────────────────────────────────────────────────────────────────────────
# 2. ranking_by_final_metric
# ──────────────────────────────────────────────────────────────────────────


class TestRanking:
    def test_lower_is_better_default(self) -> None:
        corpus = _make_corpus(2)
        specs = [
            # OCR parfait → CER=0
            PipelineSpec("perfect", [
                PipelineStep("ocr", MockOCR(_ocr_perfect)),
            ]),
            # OCR fautif → CER>0
            PipelineSpec("typo", [
                PipelineStep("ocr", MockOCR(_ocr_with_typo)),
            ]),
        ]
        result = compare_pipelines(specs, corpus)
        ranked = result.ranking_by_final_metric(
            ArtifactType.TEXT, "cer",
        )
        # Le parfait arrive en premier (CER 0 < typo CER > 0)
        assert ranked[0][0] == "perfect"
        assert ranked[0][1] == 0.0
        assert ranked[1][0] == "typo"
        assert ranked[1][1] > 0.0

    def test_higher_is_better(self) -> None:
        corpus = _make_corpus(1)
        # On utilise la métrique unicode_block_global_accuracy
        # (plus haut = meilleur)
        specs = [
            PipelineSpec("perfect", [
                PipelineStep("ocr", MockOCR(_ocr_perfect)),
            ]),
            PipelineSpec("typo", [
                PipelineStep("ocr", MockOCR(_ocr_with_typo)),
            ]),
        ]
        result = compare_pipelines(specs, corpus)
        # On bascule sur cer + higher_is_better=True : on vérifie
        # que le tri s'inverse
        ranked_lower = result.ranking_by_final_metric(
            ArtifactType.TEXT, "cer", higher_is_better=False,
        )
        ranked_higher = result.ranking_by_final_metric(
            ArtifactType.TEXT, "cer", higher_is_better=True,
        )
        # Si les deux pipelines ont des valeurs différentes, l'ordre
        # est inversé
        if ranked_lower[0][1] != ranked_lower[1][1]:
            assert ranked_lower[0][0] != ranked_higher[0][0]

    def test_pipelines_without_metric_in_queue(self) -> None:
        # Pipeline qui ne produit pas de TEXT (ex. crash de tous
        # les docs) : pas de métrique → en queue
        corpus = _make_corpus(1)

        class AlwaysFails(BaseModule):
            input_types = (ArtifactType.IMAGE,)
            output_types = (ArtifactType.TEXT,)
            execution_mode: Any = "io"

            @property
            def name(self) -> str:
                return "fail"

            def process(self, inputs):
                raise RuntimeError("boom")

        specs = [
            PipelineSpec("ok", [
                PipelineStep("ocr", MockOCR(_ocr_perfect)),
            ]),
            PipelineSpec("ko", [
                PipelineStep("ocr", AlwaysFails()),
            ]),
        ]
        result = compare_pipelines(specs, corpus)
        ranked = result.ranking_by_final_metric(
            ArtifactType.TEXT, "cer",
        )
        # ok est en tête, ko en queue avec valeur None
        assert ranked[0][0] == "ok"
        assert ranked[0][1] == 0.0
        assert ranked[-1][0] == "ko"
        assert ranked[-1][1] is None


# ──────────────────────────────────────────────────────────────────────────
# 3. gain_table
# ──────────────────────────────────────────────────────────────────────────


class TestGainTable:
    def test_baseline_unknown_raises(self) -> None:
        corpus = _make_corpus(1)
        spec = PipelineSpec("a", [PipelineStep("ocr", MockOCR(_ocr_perfect))])
        result = compare_pipelines([spec], corpus)
        with pytest.raises(KeyError, match="baseline"):
            result.gain_table(
                ArtifactType.TEXT, "cer", baseline_pipeline="inconnue",
            )

    def test_baseline_self_zero_gain(self) -> None:
        corpus = _make_corpus(1)
        spec = PipelineSpec("a", [PipelineStep("ocr", MockOCR(_ocr_perfect))])
        result = compare_pipelines([spec], corpus)
        gains = result.gain_table(ArtifactType.TEXT, "cer", "a")
        assert gains["a"]["absolute"] == 0.0
        # CER vaut 0 pour les deux ; relative = None car baseline = 0
        assert gains["a"]["relative"] is None

    def test_relative_none_when_baseline_zero(self) -> None:
        corpus = _make_corpus(1)
        specs = [
            PipelineSpec("perfect", [
                PipelineStep("ocr", MockOCR(_ocr_perfect)),
            ]),
            PipelineSpec("typo", [
                PipelineStep("ocr", MockOCR(_ocr_with_typo)),
            ]),
        ]
        result = compare_pipelines(specs, corpus)
        gains = result.gain_table(ArtifactType.TEXT, "cer", "perfect")
        # baseline = 0 → relative = None
        assert gains["typo"]["relative"] is None
        assert gains["typo"]["absolute"] is not None
        assert gains["typo"]["absolute"] > 0

    def test_realistic_fixer_outperforms_baseline(self) -> None:
        # OCR avec fautes corrigeables, fixer ramène à perfection
        corpus = _make_corpus(2)

        def ocr_typo(path: str) -> str:
            idx = path.replace("/tmp/d", "").replace(".png", "")
            return f"txete {idx}"  # 'texte' → 'txete'

        specs = [
            PipelineSpec("ocr_only", [
                PipelineStep("ocr", MockOCR(ocr_typo)),
            ]),
            PipelineSpec("ocr_with_fixer", [
                PipelineStep("ocr", MockOCR(ocr_typo)),
                PipelineStep("fix", TextFixer({"txete": "texte"})),
            ]),
        ]
        result = compare_pipelines(specs, corpus)
        gains = result.gain_table(
            ArtifactType.TEXT, "cer", "ocr_only",
        )
        # ocr_only : CER > 0 ; ocr_with_fixer : CER = 0
        assert gains["ocr_only"]["value"] > 0
        assert gains["ocr_with_fixer"]["value"] == 0.0
        # absolute négatif (CER baisse → mieux)
        assert gains["ocr_with_fixer"]["absolute"] < 0


# ──────────────────────────────────────────────────────────────────────────
# 4. factories par pipeline
# ──────────────────────────────────────────────────────────────────────────


class TestCustomFactoriesPerPipeline:
    def test_factories_routed_per_pipeline(self) -> None:
        corpus = _make_corpus(1)
        # Pipeline A : démarre par IMAGE (factory par défaut)
        # Pipeline B : démarre par TEXT (factory custom)
        specs = [
            PipelineSpec("from_image", [
                PipelineStep("ocr", MockOCR(_ocr_perfect)),
            ]),
            PipelineSpec("from_text", [
                PipelineStep("fix", TextFixer({"texte": "TEXTE"})),
            ]),
        ]
        factories = {
            "from_text": lambda doc: {ArtifactType.TEXT: doc.ground_truth},
        }
        result = compare_pipelines(specs, corpus, factories)
        # Les deux pipelines ont tourné sans erreur
        assert result.per_pipeline["from_image"].n_pipelines_succeeded == 1
        assert result.per_pipeline["from_text"].n_pipelines_succeeded == 1


# ──────────────────────────────────────────────────────────────────────────
# 5. Dataclass directe
# ──────────────────────────────────────────────────────────────────────────


class TestDataclass:
    def test_default(self) -> None:
        r = PipelineComparisonResult(corpus_name="c")
        assert r.n_docs == 0
        assert r.per_pipeline == {}
        assert r.pipeline_names() == []
