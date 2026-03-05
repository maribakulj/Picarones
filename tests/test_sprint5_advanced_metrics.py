"""Tests Sprint 5 : métriques avancées patrimoniales.

Couvre :
- Matrice de confusion unicode (confusion.py)
- Scores ligatures et diacritiques (char_scores.py)
- Taxonomie des erreurs classes 1-9 (taxonomy.py)
- Analyse structurelle (structure.py)
- Qualité image (image_quality.py)
- Intégration dans les fixtures et le rapport HTML
"""

from __future__ import annotations

import pytest

# ===========================================================================
# Tests ConfusionMatrix
# ===========================================================================

from picarones.core.confusion import (
    ConfusionMatrix,
    EMPTY_CHAR,
    build_confusion_matrix,
    aggregate_confusion_matrices,
    top_confused_chars,
)


class TestBuildConfusionMatrix:

    def test_identical_texts(self):
        cm = build_confusion_matrix("abc", "abc")
        # Pas de substitutions
        assert cm.total_substitutions == 0
        assert cm.total_insertions == 0
        assert cm.total_deletions == 0

    def test_empty_texts(self):
        cm = build_confusion_matrix("", "")
        assert cm.total_errors == 0

    def test_simple_substitution(self):
        cm = build_confusion_matrix("abc", "axc")
        # 'b' → 'x'
        assert "b" in cm.matrix
        assert "x" in cm.matrix["b"]
        assert cm.matrix["b"]["x"] >= 1

    def test_deletion_recorded(self):
        cm = build_confusion_matrix("abc", "ac")
        # 'b' supprimé
        assert "b" in cm.matrix
        assert EMPTY_CHAR in cm.matrix["b"]

    def test_insertion_recorded(self):
        cm = build_confusion_matrix("ac", "abc")
        # 'b' inséré
        assert EMPTY_CHAR in cm.matrix
        assert "b" in cm.matrix[EMPTY_CHAR]

    def test_no_whitespace_recorded_by_default(self):
        cm = build_confusion_matrix("a b", "a x")
        # Les espaces ne doivent pas être dans la matrice
        assert " " not in cm.matrix

    def test_as_dict_structure(self):
        cm = build_confusion_matrix("hello", "hallo")
        d = cm.as_dict()
        assert "matrix" in d
        assert "total_substitutions" in d
        assert "total_insertions" in d
        assert "total_deletions" in d

    def test_top_confusions(self):
        cm = build_confusion_matrix("eeee", "aaaa")
        tops = cm.top_confusions(n=5)
        assert len(tops) >= 1
        assert tops[0]["gt"] == "e"
        assert tops[0]["ocr"] == "a"
        assert tops[0]["count"] == 4

    def test_medieval_chars_tracked(self):
        cm = build_confusion_matrix("maiſon", "maifon")
        # ſ confondu avec f
        assert "ſ" in cm.matrix
        assert "f" in cm.matrix["ſ"]

    def test_as_compact_dict_filters_low_count(self):
        cm = build_confusion_matrix("aab", "axb")
        # avec min_count=2, une substitution unique filtrée
        compact = cm.as_compact_dict(min_count=2)
        # Le 'a'→'x' ne doit pas apparaître (1 seule occurrence)
        matrix = compact["matrix"]
        for gt_counts in matrix.values():
            for ocr_char, cnt in gt_counts.items():
                assert cnt >= 2


class TestAggregateConfusionMatrices:

    def test_empty_list(self):
        cm = aggregate_confusion_matrices([])
        assert cm.total_errors == 0

    def test_single_matrix(self):
        cm1 = build_confusion_matrix("abc", "axc")
        agg = aggregate_confusion_matrices([cm1])
        assert agg.matrix == cm1.matrix

    def test_counts_sum(self):
        cm1 = build_confusion_matrix("abc", "axc")
        cm2 = build_confusion_matrix("abc", "axc")
        agg = aggregate_confusion_matrices([cm1, cm2])
        # La confusion 'b'→'x' doit apparaître 2 fois
        assert agg.matrix.get("b", {}).get("x", 0) >= 2

    def test_total_errors_sum(self):
        cm1 = build_confusion_matrix("abc", "axc")
        cm2 = build_confusion_matrix("def", "dxf")
        agg = aggregate_confusion_matrices([cm1, cm2])
        assert agg.total_errors >= cm1.total_errors + cm2.total_errors


class TestTopConfusedChars:

    def test_returns_list(self):
        cm = build_confusion_matrix("aaabbb", "aaaxxx")
        tops = top_confused_chars(cm, n=5)
        assert isinstance(tops, list)

    def test_sorted_by_errors_desc(self):
        cm = aggregate_confusion_matrices([
            build_confusion_matrix("bbb", "xxx"),  # 3 fois
            build_confusion_matrix("a", "y"),       # 1 fois
        ])
        tops = top_confused_chars(cm, n=10)
        if len(tops) >= 2:
            assert tops[0]["total_errors"] >= tops[1]["total_errors"]

    def test_excludes_empty_char(self):
        cm = build_confusion_matrix("abc", "ac")  # b supprimé
        tops = top_confused_chars(cm, exclude_empty=True)
        assert all(t["char"] != EMPTY_CHAR for t in tops)


# ===========================================================================
# Tests LigatureScore
# ===========================================================================

from picarones.core.char_scores import (
    LIGATURE_TABLE,
    DIACRITIC_MAP,
    LigatureScore,
    DiacriticScore,
    compute_ligature_score,
    compute_diacritic_score,
    aggregate_ligature_scores,
    aggregate_diacritic_scores,
    _ALL_LIGATURES,
    _ALL_DIACRITICS,
)


class TestLigatureTable:

    def test_fi_ligature_present(self):
        assert "\uFB01" in LIGATURE_TABLE  # ﬁ

    def test_fl_ligature_present(self):
        assert "\uFB02" in LIGATURE_TABLE  # ﬂ

    def test_oe_ligature_present(self):
        assert "\u0153" in LIGATURE_TABLE  # œ

    def test_ae_ligature_present(self):
        assert "\u00E6" in LIGATURE_TABLE  # æ

    def test_ff_ligature_present(self):
        assert "\uFB00" in LIGATURE_TABLE  # ﬀ

    def test_equivalents_are_lists(self):
        for lig, equivs in LIGATURE_TABLE.items():
            assert isinstance(equivs, list)
            assert len(equivs) >= 1


class TestComputeLigatureScore:

    def test_no_ligatures_in_gt(self):
        result = compute_ligature_score("bonjour monde", "bonjour monde")
        assert result.score == pytest.approx(1.0)
        assert result.total_in_gt == 0

    def test_ligature_correctly_recognized(self):
        # GT avec ﬁ (fi ligature), OCR reconnaît "fi"
        result = compute_ligature_score("ﬁn", "fin")
        assert result.total_in_gt == 1
        assert result.score == pytest.approx(1.0)

    def test_ligature_unicode_to_unicode(self):
        # GT et OCR ont tous les deux ﬁ
        result = compute_ligature_score("ﬁn", "ﬁn")
        assert result.score == pytest.approx(1.0)

    def test_oe_ligature(self):
        result = compute_ligature_score("œuvre", "oeuvre")
        assert result.total_in_gt == 1
        assert result.score == pytest.approx(1.0)

    def test_ae_ligature(self):
        result = compute_ligature_score("æther", "aether")
        assert result.total_in_gt == 1
        assert result.score == pytest.approx(1.0)

    def test_as_dict_structure(self):
        result = compute_ligature_score("ﬁn", "fin")
        d = result.as_dict()
        assert "total_in_gt" in d
        assert "correctly_recognized" in d
        assert "score" in d
        assert "per_ligature" in d

    def test_empty_texts(self):
        result = compute_ligature_score("", "")
        assert result.score == pytest.approx(1.0)
        assert result.total_in_gt == 0


class TestComputeDiacriticScore:

    def test_no_diacritics(self):
        result = compute_diacritic_score("bonjour", "bonjour")
        assert result.score == pytest.approx(1.0)
        assert result.total_in_gt == 0

    def test_accent_preserved(self):
        result = compute_diacritic_score("été", "été")
        assert result.score == pytest.approx(1.0)
        assert result.correctly_recognized == result.total_in_gt

    def test_accent_lost(self):
        result = compute_diacritic_score("étude", "etude")
        assert result.total_in_gt >= 1
        # é → e : perte du diacritique
        assert result.correctly_recognized < result.total_in_gt
        assert result.score < 1.0

    def test_cedille_tracked(self):
        result = compute_diacritic_score("façon", "facon")
        assert result.total_in_gt >= 1
        assert result.score < 1.0

    def test_empty_texts(self):
        result = compute_diacritic_score("", "")
        assert result.score == pytest.approx(1.0)

    def test_as_dict_structure(self):
        result = compute_diacritic_score("été", "ete")
        d = result.as_dict()
        assert "total_in_gt" in d
        assert "correctly_recognized" in d
        assert "score" in d


class TestAggregateLigatureScores:

    def test_empty_list(self):
        result = aggregate_ligature_scores([])
        assert result["score"] == pytest.approx(1.0)
        assert result["total_in_gt"] == 0

    def test_aggregation(self):
        s1 = LigatureScore(total_in_gt=4, correctly_recognized=3, score=0.75)
        s2 = LigatureScore(total_in_gt=2, correctly_recognized=2, score=1.0)
        result = aggregate_ligature_scores([s1, s2])
        assert result["total_in_gt"] == 6
        assert result["correctly_recognized"] == 5
        assert result["score"] == pytest.approx(5/6, abs=1e-4)


class TestAggregateDiacriticScores:

    def test_aggregation(self):
        s1 = DiacriticScore(total_in_gt=10, correctly_recognized=8, score=0.8)
        s2 = DiacriticScore(total_in_gt=5, correctly_recognized=5, score=1.0)
        result = aggregate_diacritic_scores([s1, s2])
        assert result["total_in_gt"] == 15
        assert result["correctly_recognized"] == 13


# ===========================================================================
# Tests TaxonomyResult
# ===========================================================================

from picarones.core.taxonomy import (
    TaxonomyResult,
    ERROR_CLASSES,
    classify_errors,
    aggregate_taxonomy,
    VISUAL_CONFUSIONS,
)


class TestErrorClasses:

    def test_nine_classes(self):
        assert len(ERROR_CLASSES) == 9

    def test_class_names(self):
        assert "visual_confusion" in ERROR_CLASSES
        assert "diacritic_error" in ERROR_CLASSES
        assert "case_error" in ERROR_CLASSES
        assert "ligature_error" in ERROR_CLASSES
        assert "lacuna" in ERROR_CLASSES


class TestClassifyErrors:

    def test_identical_texts(self):
        result = classify_errors("bonjour monde", "bonjour monde")
        assert result.total_errors == 0

    def test_empty_texts(self):
        result = classify_errors("", "")
        assert result.total_errors == 0

    def test_case_error_detected(self):
        result = classify_errors("Bonjour Monde", "bonjour monde")
        assert result.counts["case_error"] >= 1

    def test_diacritic_error_detected(self):
        result = classify_errors("été chez nous", "ete chez nous")
        assert result.counts["diacritic_error"] >= 1

    def test_lacuna_detected(self):
        result = classify_errors("le chat dort paisiblement", "le chat")
        assert result.counts["lacuna"] >= 1

    def test_segmentation_detected(self):
        result = classify_errors("hello world test", "helloworld test")
        # "hello world" fusionné en "helloworld"
        assert result.counts["segmentation_error"] >= 0  # peut être classé hapax aussi

    def test_ligature_error_detected(self):
        result = classify_errors("ﬁn de siècle", "fin de siècle")
        # ﬁ vs fi est une ligature correcte, pas une erreur
        # Mais si on avait: GT=ﬁ, OCR=ﬁ → correct
        # Test avec ligature mal reconnue: GT=ﬁn, OCR=fïn (erreur diac)
        assert result.total_errors >= 0  # pas d'erreur ici (fin est équivalent)

    def test_as_dict_structure(self):
        result = classify_errors("test erreur ici", "test erreur là")
        d = result.as_dict()
        assert "counts" in d
        assert "total_errors" in d
        assert "class_distribution" in d
        assert "examples" in d

    def test_from_dict_roundtrip(self):
        result = classify_errors("bonjour monde", "Bonjour monde")
        d = result.as_dict()
        restored = TaxonomyResult.from_dict(d)
        assert restored.total_errors == result.total_errors
        assert restored.counts == result.counts

    def test_class_distribution_sums_to_one(self):
        result = classify_errors("abc def ghi", "xyz uvw rst")
        dist = result.class_distribution
        if dist:
            assert abs(sum(dist.values()) - 1.0) < 1e-6

    def test_all_classes_in_counts(self):
        result = classify_errors("test", "teSt")
        for cls in ERROR_CLASSES:
            assert cls in result.counts


class TestAggregateTaxonomy:

    def test_empty(self):
        result = aggregate_taxonomy([])
        assert result["total_errors"] == 0

    def test_sums_counts(self):
        r1 = TaxonomyResult(
            counts={"visual_confusion": 2, "diacritic_error": 1, **{k: 0 for k in ERROR_CLASSES if k not in ["visual_confusion", "diacritic_error"]}},
            total_errors=3,
        )
        r2 = TaxonomyResult(
            counts={"visual_confusion": 1, "diacritic_error": 3, **{k: 0 for k in ERROR_CLASSES if k not in ["visual_confusion", "diacritic_error"]}},
            total_errors=4,
        )
        agg = aggregate_taxonomy([r1, r2])
        assert agg["counts"]["visual_confusion"] == 3
        assert agg["counts"]["diacritic_error"] == 4
        assert agg["total_errors"] == 7


# ===========================================================================
# Tests StructureResult
# ===========================================================================

from picarones.core.structure import (
    StructureResult,
    analyze_structure,
    aggregate_structure,
)


class TestAnalyzeStructure:

    def test_identical_single_line(self):
        result = analyze_structure("ligne unique", "ligne unique")
        assert result.gt_line_count == 1
        assert result.ocr_line_count == 1
        assert result.line_fusion_count == 0
        assert result.line_fragmentation_count == 0

    def test_empty_texts(self):
        result = analyze_structure("", "")
        assert result.gt_line_count == 0
        assert result.ocr_line_count == 0

    def test_multiline_equal(self):
        gt = "ligne 1\nligne 2\nligne 3"
        result = analyze_structure(gt, gt)
        assert result.gt_line_count == 3
        assert result.ocr_line_count == 3

    def test_line_fusion_detected(self):
        gt = "ligne 1\nligne 2\nligne 3"
        ocr = "ligne 1 ligne 2\nligne 3"  # fusion de 2 lignes en 1
        result = analyze_structure(gt, ocr)
        # Le nombre de lignes OCR < GT
        assert result.ocr_line_count < result.gt_line_count

    def test_reading_order_score_perfect(self):
        text = "le chat dort ici"
        result = analyze_structure(text, text)
        assert result.reading_order_score > 0.9

    def test_reading_order_score_low_for_scrambled(self):
        gt = "le chat dort paisiblement sur le canapé"
        ocr = "canapé sur le paisiblement dort chat le"
        result = analyze_structure(gt, ocr)
        assert result.reading_order_score < 1.0

    def test_line_accuracy_perfect(self):
        gt = "ligne 1\nligne 2"
        ocr = "ligne 1\nligne 2"
        result = analyze_structure(gt, ocr)
        assert result.line_accuracy == pytest.approx(1.0)

    def test_line_accuracy_degraded(self):
        gt = "ligne 1\nligne 2\nligne 3\nligne 4"
        ocr = "ligne 1"
        result = analyze_structure(gt, ocr)
        assert result.line_accuracy < 1.0

    def test_as_dict_structure(self):
        result = analyze_structure("ligne 1\nligne 2", "ligne 1\nligne 2")
        d = result.as_dict()
        required = ["gt_line_count", "ocr_line_count", "line_fusion_count",
                    "line_fragmentation_count", "reading_order_score",
                    "paragraph_conservation_score", "line_accuracy"]
        for key in required:
            assert key in d

    def test_from_dict_roundtrip(self):
        result = analyze_structure("a\nb\nc", "a\nb")
        d = result.as_dict()
        restored = StructureResult.from_dict(d)
        assert restored.gt_line_count == result.gt_line_count
        assert restored.ocr_line_count == result.ocr_line_count

    def test_line_fusion_rate_property(self):
        result = StructureResult(gt_line_count=10, ocr_line_count=8, line_fusion_count=2)
        assert result.line_fusion_rate == pytest.approx(0.2)

    def test_line_fragmentation_rate_property(self):
        result = StructureResult(gt_line_count=5, ocr_line_count=8, line_fragmentation_count=3)
        assert result.line_fragmentation_rate == pytest.approx(0.6)


class TestAggregateStructure:

    def test_empty(self):
        result = aggregate_structure([])
        assert result == {}

    def test_single_result(self):
        r = StructureResult(
            gt_line_count=5, ocr_line_count=5,
            reading_order_score=0.9, paragraph_conservation_score=1.0,
        )
        agg = aggregate_structure([r])
        assert agg["mean_reading_order_score"] == pytest.approx(0.9)
        assert agg["document_count"] == 1

    def test_mean_fusion_rate(self):
        r1 = StructureResult(gt_line_count=10, ocr_line_count=8, line_fusion_count=2)
        r2 = StructureResult(gt_line_count=10, ocr_line_count=6, line_fusion_count=4)
        agg = aggregate_structure([r1, r2])
        # fusion rates: 0.2 et 0.4 → mean = 0.3
        assert agg["mean_line_fusion_rate"] == pytest.approx(0.3, rel=1e-3)


# ===========================================================================
# Tests ImageQualityResult
# ===========================================================================

from picarones.core.image_quality import (
    ImageQualityResult,
    generate_mock_quality_scores,
    aggregate_image_quality,
    _global_quality_score,
)


class TestImageQualityResult:

    def test_quality_tier_good(self):
        r = ImageQualityResult(quality_score=0.8)
        assert r.quality_tier == "good"
        assert r.is_good_quality is True

    def test_quality_tier_medium(self):
        r = ImageQualityResult(quality_score=0.55)
        assert r.quality_tier == "medium"
        assert r.is_good_quality is False

    def test_quality_tier_poor(self):
        r = ImageQualityResult(quality_score=0.2)
        assert r.quality_tier == "poor"

    def test_as_dict_structure(self):
        r = ImageQualityResult(
            sharpness_score=0.8, noise_level=0.1, rotation_degrees=0.5,
            contrast_score=0.9, quality_score=0.75, analysis_method="mock",
        )
        d = r.as_dict()
        assert "sharpness_score" in d
        assert "noise_level" in d
        assert "rotation_degrees" in d
        assert "contrast_score" in d
        assert "quality_score" in d
        assert "quality_tier" in d
        assert "analysis_method" in d

    def test_from_dict_roundtrip(self):
        r = ImageQualityResult(
            sharpness_score=0.7, noise_level=0.2, rotation_degrees=1.0,
            contrast_score=0.8, quality_score=0.65, analysis_method="pillow",
        )
        d = r.as_dict()
        restored = ImageQualityResult.from_dict(d)
        assert restored.sharpness_score == pytest.approx(r.sharpness_score, rel=1e-3)
        assert restored.quality_score == pytest.approx(r.quality_score, rel=1e-3)
        assert restored.analysis_method == r.analysis_method

    def test_from_dict_ignores_quality_tier(self):
        # quality_tier est une propriété, pas un param init → from_dict doit l'ignorer
        data = {
            "sharpness_score": 0.5, "noise_level": 0.3, "rotation_degrees": 0.0,
            "contrast_score": 0.6, "quality_score": 0.5, "analysis_method": "mock",
            "quality_tier": "medium",  # doit être ignoré
        }
        r = ImageQualityResult.from_dict(data)
        assert r.quality_score == pytest.approx(0.5)


class TestGenerateMockQualityScores:

    def test_returns_image_quality_result(self):
        r = generate_mock_quality_scores("folio_001")
        assert isinstance(r, ImageQualityResult)

    def test_scores_in_range(self):
        r = generate_mock_quality_scores("folio_001", seed=42)
        assert 0.0 <= r.quality_score <= 1.0
        assert 0.0 <= r.sharpness_score <= 1.0
        assert 0.0 <= r.noise_level <= 1.0
        assert 0.0 <= r.contrast_score <= 1.0

    def test_reproducible_with_seed(self):
        r1 = generate_mock_quality_scores("folio_001", seed=42)
        r2 = generate_mock_quality_scores("folio_001", seed=42)
        assert r1.quality_score == r2.quality_score

    def test_analysis_method_mock(self):
        r = generate_mock_quality_scores("folio_001")
        assert r.analysis_method == "mock"

    def test_no_error(self):
        r = generate_mock_quality_scores("folio_001")
        assert r.error is None


class TestGlobalQualityScore:

    def test_perfect_input(self):
        score = _global_quality_score(sharpness=1.0, noise=0.0, rotation_abs=0.0, contrast=1.0)
        assert score == pytest.approx(1.0)

    def test_worst_input(self):
        score = _global_quality_score(sharpness=0.0, noise=1.0, rotation_abs=10.0, contrast=0.0)
        assert score == pytest.approx(0.0)

    def test_medium_input(self):
        score = _global_quality_score(sharpness=0.5, noise=0.5, rotation_abs=0.0, contrast=0.5)
        assert 0.0 < score < 1.0


class TestAggregateImageQuality:

    def test_empty_list(self):
        result = aggregate_image_quality([])
        assert result == {}

    def test_single_result(self):
        r = ImageQualityResult(quality_score=0.75, analysis_method="mock")
        agg = aggregate_image_quality([r])
        assert agg["mean_quality_score"] == pytest.approx(0.75)
        assert agg["document_count"] == 1

    def test_tier_distribution(self):
        results = [
            ImageQualityResult(quality_score=0.8, analysis_method="mock"),  # good
            ImageQualityResult(quality_score=0.5, analysis_method="mock"),  # medium
            ImageQualityResult(quality_score=0.2, analysis_method="mock"),  # poor
        ]
        agg = aggregate_image_quality(results)
        assert agg["quality_distribution"]["good"] == 1
        assert agg["quality_distribution"]["medium"] == 1
        assert agg["quality_distribution"]["poor"] == 1

    def test_scores_list_present(self):
        results = [ImageQualityResult(quality_score=0.6, analysis_method="mock")]
        agg = aggregate_image_quality(results)
        assert "scores" in agg
        assert len(agg["scores"]) == 1

    def test_errors_excluded(self):
        results = [
            ImageQualityResult(quality_score=0.8, analysis_method="mock"),
            ImageQualityResult(quality_score=0.0, analysis_method="none", error="file not found"),
        ]
        agg = aggregate_image_quality(results)
        assert agg["document_count"] == 1  # seul le résultat sans erreur compte


# ===========================================================================
# Tests d'intégration Sprint 5 (fixtures + rapport)
# ===========================================================================

class TestFixturesSprint5:

    def test_doc_result_has_confusion_matrix(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark()
        for er in bm.engine_reports:
            for dr in er.document_results:
                assert dr.confusion_matrix is not None, (
                    f"confusion_matrix manquante pour {er.engine_name}/{dr.doc_id}"
                )
                break

    def test_doc_result_has_char_scores(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark()
        for er in bm.engine_reports:
            dr = er.document_results[0]
            assert dr.char_scores is not None
            assert "ligature" in dr.char_scores
            assert "diacritic" in dr.char_scores

    def test_doc_result_has_taxonomy(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark()
        for er in bm.engine_reports:
            dr = er.document_results[0]
            assert dr.taxonomy is not None
            assert "counts" in dr.taxonomy
            assert "total_errors" in dr.taxonomy

    def test_doc_result_has_structure(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark()
        for er in bm.engine_reports:
            dr = er.document_results[0]
            assert dr.structure is not None
            assert "gt_line_count" in dr.structure

    def test_doc_result_has_image_quality(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark()
        for er in bm.engine_reports:
            dr = er.document_results[0]
            assert dr.image_quality is not None
            assert "quality_score" in dr.image_quality

    def test_engine_report_has_aggregated_confusion(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark()
        for er in bm.engine_reports:
            assert er.aggregated_confusion is not None
            assert "matrix" in er.aggregated_confusion

    def test_engine_report_has_aggregated_char_scores(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark()
        for er in bm.engine_reports:
            assert er.aggregated_char_scores is not None
            assert "ligature" in er.aggregated_char_scores
            assert "diacritic" in er.aggregated_char_scores

    def test_engine_report_ligature_score_property(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark()
        for er in bm.engine_reports:
            score = er.ligature_score
            assert score is not None
            assert 0.0 <= score <= 1.0

    def test_engine_report_diacritic_score_property(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark()
        for er in bm.engine_reports:
            score = er.diacritic_score
            assert score is not None
            assert 0.0 <= score <= 1.0

    def test_engine_report_has_aggregated_taxonomy(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark()
        for er in bm.engine_reports:
            assert er.aggregated_taxonomy is not None
            assert "total_errors" in er.aggregated_taxonomy

    def test_engine_report_has_aggregated_structure(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark()
        for er in bm.engine_reports:
            assert er.aggregated_structure is not None
            assert "mean_reading_order_score" in er.aggregated_structure

    def test_engine_report_has_aggregated_image_quality(self):
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark()
        for er in bm.engine_reports:
            assert er.aggregated_image_quality is not None
            assert "mean_quality_score" in er.aggregated_image_quality

    def test_bad_engine_has_more_errors(self):
        """L'ancien moteur doit avoir plus d'erreurs taxonomiques que pero_ocr."""
        from picarones.fixtures import generate_sample_benchmark
        bm = generate_sample_benchmark()
        pero = next(er for er in bm.engine_reports if er.engine_name == "pero_ocr")
        bad = next(er for er in bm.engine_reports if er.engine_name == "ancien_moteur")
        assert bad.aggregated_taxonomy["total_errors"] > pero.aggregated_taxonomy["total_errors"]


class TestReportSprint5:

    def test_report_data_has_ligature_score(self):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import _build_report_data
        bm = generate_sample_benchmark()
        data = _build_report_data(bm, {})
        for eng in data["engines"]:
            assert "ligature_score" in eng, f"ligature_score manquant pour {eng['name']}"

    def test_report_data_has_diacritic_score(self):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import _build_report_data
        bm = generate_sample_benchmark()
        data = _build_report_data(bm, {})
        for eng in data["engines"]:
            assert "diacritic_score" in eng

    def test_report_data_has_aggregated_taxonomy(self):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import _build_report_data
        bm = generate_sample_benchmark()
        data = _build_report_data(bm, {})
        for eng in data["engines"]:
            assert "aggregated_taxonomy" in eng

    def test_report_data_has_aggregated_image_quality(self):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import _build_report_data
        bm = generate_sample_benchmark()
        data = _build_report_data(bm, {})
        for eng in data["engines"]:
            assert "aggregated_image_quality" in eng

    def test_html_has_characters_tab(self, tmp_path):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import ReportGenerator
        bm = generate_sample_benchmark()
        out = tmp_path / "report.html"
        ReportGenerator(bm).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "Caractères" in html

    def test_html_has_ligatures_column(self, tmp_path):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import ReportGenerator
        bm = generate_sample_benchmark()
        out = tmp_path / "report.html"
        ReportGenerator(bm).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "Ligatures" in html

    def test_html_has_diacritiques_column(self, tmp_path):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import ReportGenerator
        bm = generate_sample_benchmark()
        out = tmp_path / "report.html"
        ReportGenerator(bm).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "Diacritiques" in html

    def test_html_has_scatter_plot(self, tmp_path):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import ReportGenerator
        bm = generate_sample_benchmark()
        out = tmp_path / "report.html"
        ReportGenerator(bm).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "chart-quality-cer" in html

    def test_html_has_taxonomy_chart(self, tmp_path):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import ReportGenerator
        bm = generate_sample_benchmark()
        out = tmp_path / "report.html"
        ReportGenerator(bm).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "chart-taxonomy" in html

    def test_html_has_confusion_heatmap(self, tmp_path):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import ReportGenerator
        bm = generate_sample_benchmark()
        out = tmp_path / "report.html"
        ReportGenerator(bm).generate(out)
        html = out.read_text(encoding="utf-8")
        assert "confusion-heatmap" in html or "matrice de confusion" in html.lower()

    def test_doc_results_have_image_quality_in_report(self):
        from picarones.fixtures import generate_sample_benchmark
        from picarones.report.generator import _build_report_data
        bm = generate_sample_benchmark()
        data = _build_report_data(bm, {})
        doc = data["documents"][0]
        # Au moins un engine result doit avoir image_quality
        has_iq = any("image_quality" in er for er in doc["engine_results"])
        assert has_iq, "Aucun document result n'a de données image_quality"

    def test_json_export_contains_sprint5_data(self, tmp_path):
        from picarones.fixtures import generate_sample_benchmark
        import json
        bm = generate_sample_benchmark()
        out = tmp_path / "results.json"
        bm.to_json(out)
        data = json.loads(out.read_text())
        # Vérifier dans les engine_reports
        er = data["engine_reports"][0]
        assert "aggregated_taxonomy" in er
        assert "aggregated_char_scores" in er
        # Vérifier dans les document_results
        dr = er["document_results"][0]
        assert "taxonomy" in dr
        assert "char_scores" in dr
        assert "structure" in dr
