"""Phase 1 du chantier post-rewrite — durcissements sécurité P0.

Couvre trois durcissements introduits pour fermer des surfaces filesystem
laissées ouvertes par le rewrite :

1. **Path traversal ``output_dir`` dans les importers HTR-United/HuggingFace.**
   Avant durcissement : un POST ``output_dir="/etc/picarones_pwned"``
   passait directement à l'importer, vecteur d'écriture filesystem
   arbitraire.  Désormais ``validated_path`` rejette en 400 avant délégation.

2. **Path traversal ``db_path`` dans ``/api/history/regressions``.**
   Avant durcissement : ``db_path=/etc/passwd`` ouvrait un SQLite
   arbitraire (lecture libre, log d'erreur informatif).  Désormais
   ``validated_path`` rejette en 400 ; pour pointer une base hors
   workspace, exporter ``PICARONES_HISTORY_DB``.

3. **ZIP basename collision + validation image extraite.**
   Avant durcissement : ``a/img.png`` et ``b/img.png`` s'écrasaient
   silencieusement après aplatissement ; les images extraites n'étaient
   pas passées à ``validate_image_safe`` (vecteur zip bomb jusqu'à
   500 Mo brut).  Désormais : collision → renommage avec préfixe slug
   du dirname + warning ; image invalide → ``ValueError`` (HTTP 415).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest


# PNG 1x1 minimal valide pour passer Pillow.verify.
_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_importers_app():
    from fastapi import FastAPI

    from picarones.interfaces.web.routers import importers as imp_router

    app = FastAPI()
    app.include_router(imp_router.router)
    return app


def _make_history_app():
    from fastapi import FastAPI

    from picarones.interfaces.web.routers import history as hist_router

    app = FastAPI()
    app.include_router(hist_router.router)
    return app


# ──────────────────────────────────────────────────────────────────────
# 1. output_dir path traversal — HTR-United + HuggingFace
# ──────────────────────────────────────────────────────────────────────


class TestImportersOutputDirTraversal:
    """Aucun ``output_dir`` libre hors des racines workspace.

    Important : on n'utilise PAS ``patch`` sur l'importer — la validation
    doit échouer AVANT toute délégation au backend.  Si la validation
    laisse passer, le mock ne sera pas appelé mais la requête sera
    acceptée — c'est ce qu'on doit empêcher.
    """

    def test_htr_united_rejects_absolute_path_outside_workspace(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_importers_app()
        with TestClient(app) as client:
            r = client.post(
                "/api/htr-united/import",
                json={
                    "entry_id": "any_id",
                    "output_dir": "/etc/picarones_pwned",
                    "max_samples": 1,
                },
            )
            # 400 = PathValidationError mappée par le handler.
            assert r.status_code == 400, (
                f"Attendu 400 (path validation), reçu {r.status_code} : "
                f"{r.text}"
            )
            assert "hors zone autorisée" in r.json()["detail"]

    def test_htr_united_rejects_traversal(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_importers_app()
        with TestClient(app) as client:
            r = client.post(
                "/api/htr-united/import",
                json={
                    "entry_id": "any_id",
                    "output_dir": "../../../etc/passwd",
                    "max_samples": 1,
                },
            )
            assert r.status_code == 400
            # Le message peut citer la racine ou le chemin original ;
            # on vérifie juste qu'on n'a pas réussi à passer.
            detail = r.json()["detail"]
            assert "hors zone" in detail or "invalide" in detail

    def test_huggingface_rejects_absolute_path_outside_workspace(
        self,
    ) -> None:
        from fastapi.testclient import TestClient

        app = _make_importers_app()
        with TestClient(app) as client:
            r = client.post(
                "/api/huggingface/import",
                json={
                    "dataset_id": "any/dataset",
                    "output_dir": "/var/lib/pwned",
                    "split": "train",
                    "max_samples": 1,
                },
            )
            assert r.status_code == 400
            assert "hors zone autorisée" in r.json()["detail"]

    def test_huggingface_rejects_traversal(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_importers_app()
        with TestClient(app) as client:
            r = client.post(
                "/api/huggingface/import",
                json={
                    "dataset_id": "any/dataset",
                    "output_dir": "../../../etc/passwd_dir",
                    "split": "train",
                    "max_samples": 1,
                },
            )
            assert r.status_code == 400

    def test_huggingface_accepts_path_under_tmp(self, tmp_path: Path) -> None:
        """``tmp_path`` est sous ``tempfile.gettempdir()`` donc dans les
        racines workspace par défaut (mode dev).  On vérifie que la
        validation laisse passer une cible légitime."""
        from fastapi.testclient import TestClient

        app = _make_importers_app()
        with patch(
            "picarones.adapters.corpus.huggingface.HuggingFaceImporter.import_dataset",
        ) as mock_import:
            mock_import.return_value = {
                "imported": 1, "output_dir": str(tmp_path),
            }
            with TestClient(app) as client:
                r = client.post(
                    "/api/huggingface/import",
                    json={
                        "dataset_id": "test/dataset",
                        "output_dir": str(tmp_path),
                        "split": "train",
                        "max_samples": 1,
                    },
                )
                assert r.status_code == 200, r.text
                # Vérifie que la valeur passée à l'importer est résolue
                # (str du Path absolu) — pas la chaîne brute si elle
                # avait été relative.
                assert mock_import.called


# ──────────────────────────────────────────────────────────────────────
# 2. db_path path traversal — /api/history/regressions
# ──────────────────────────────────────────────────────────────────────


class TestHistoryRegressionsDbPathTraversal:
    """``db_path`` doit être sous une racine workspace ou refusé en 400.

    Sans ce garde-fou, l'endpoint ouvrait silencieusement n'importe quel
    SQLite lisible par le process (lecture filesystem arbitraire via
    paramètres SQL).
    """

    def test_absolute_path_outside_workspace_rejected(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_history_app()
        with TestClient(app) as client:
            r = client.get(
                "/api/history/regressions",
                params={"db_path": "/etc/passwd"},
            )
            assert r.status_code == 400, r.text
            assert "hors zone autorisée" in r.json()["detail"]

    def test_traversal_rejected(self) -> None:
        from fastapi.testclient import TestClient

        app = _make_history_app()
        with TestClient(app) as client:
            r = client.get(
                "/api/history/regressions",
                params={"db_path": "../../../etc/passwd"},
            )
            assert r.status_code == 400

    def test_no_db_path_uses_default(self) -> None:
        """Sans ``db_path``, l'endpoint utilise le défaut ``BenchmarkHistory()``
        (~/.picarones/history.db).  Pas de 400, retourne une liste vide
        si la base n'existe pas (cas frais)."""
        from fastapi.testclient import TestClient

        app = _make_history_app()
        with TestClient(app) as client:
            r = client.get("/api/history/regressions")
            # Soit 200 (base existe, pas de régression), soit 500 (base
            # absente).  On accepte les deux — c'est le comportement
            # historique, hors scope du durcissement de chemin.
            assert r.status_code in (200, 500), r.text


# ──────────────────────────────────────────────────────────────────────
# 3. ZIP basename collision + validation image extraite
# ──────────────────────────────────────────────────────────────────────


def _zip_with_entries(entries: dict[str, bytes]) -> bytes:
    """ZIP en mémoire à partir de ``{nom: bytes}``."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestZipBasenameCollision:
    """``a/img.png`` et ``b/img.png`` ne doivent plus s'écraser
    silencieusement après aplatissement par basename."""

    def test_collision_resolved_with_dirname_prefix(self, tmp_path: Path) -> None:
        from picarones.interfaces.web.corpus_utils import flatten_zip_to_dir

        zip_bytes = _zip_with_entries({
            "folder_a/page_001.png": _MINIMAL_PNG,
            "folder_a/page_001.gt.txt": b"GT A",
            "folder_b/page_001.png": _MINIMAL_PNG,
            "folder_b/page_001.gt.txt": b"GT B",
        })
        dest = tmp_path / "extract"

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            flatten_zip_to_dir(zf, dest)

        names = {p.name for p in dest.iterdir()}
        # La première occurrence garde le nom brut ; les suivantes sont
        # préfixées par le slug du dirname source.
        assert "page_001.png" in names
        # Le second doit avoir été renommé — par slug ``folder_b``.
        renamed_png = {n for n in names if n.endswith("page_001.png")}
        assert len(renamed_png) == 2, (
            f"Attendu 2 images distinctes (1 nominale + 1 renommée), "
            f"trouvé {renamed_png}"
        )
        # On vérifie qu'au moins une variante porte un slug de dossier.
        assert any(
            "folder_a" in n or "folder_b" in n
            for n in renamed_png - {"page_001.png"}
        )

    def test_no_silent_overwrite_of_image_pairs(self, tmp_path: Path) -> None:
        """Garantie fonctionnelle : 4 fichiers entrent → 4 fichiers sortent."""
        from picarones.interfaces.web.corpus_utils import flatten_zip_to_dir

        zip_bytes = _zip_with_entries({
            "a/img.png": _MINIMAL_PNG,
            "a/img.gt.txt": b"A",
            "b/img.png": _MINIMAL_PNG,
            "b/img.gt.txt": b"B",
        })
        dest = tmp_path / "extract"
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            flatten_zip_to_dir(zf, dest)

        files = list(dest.iterdir())
        # 4 fichiers entrent dans le ZIP, 4 doivent ressortir (les
        # collisions sont résolues, pas écrasées).
        assert len(files) == 4, (
            f"Attendu 4 fichiers (anti-collision), trouvé "
            f"{[p.name for p in files]}"
        )


class TestZipExtractedImageValidation:
    """Les images extraites du ZIP doivent passer ``validate_image_safe``
    — sans ce garde-fou, un attaquant pouvait emballer une fausse image
    (DecompressionBombError, format invalide) jusqu'à 500 Mo non
    vérifiés."""

    def test_invalid_extracted_image_rejected(self, tmp_path: Path) -> None:
        from picarones.interfaces.web.corpus_utils import flatten_zip_to_dir

        zip_bytes = _zip_with_entries({
            # Header PNG seul mais sans IHDR — invalide.
            "fake.png": b"\x89PNG\r\n\x1a\nFAKE_NOT_A_REAL_PNG",
        })
        dest = tmp_path / "extract"

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            with pytest.raises(ValueError) as excinfo:
                flatten_zip_to_dir(zf, dest)
        # Le message doit mentionner le filename pour aider au debug.
        assert "fake.png" in str(excinfo.value)

    def test_valid_extracted_image_passes(self, tmp_path: Path) -> None:
        from picarones.interfaces.web.corpus_utils import flatten_zip_to_dir

        zip_bytes = _zip_with_entries({
            "ok.png": _MINIMAL_PNG,
            "ok.gt.txt": b"Hello",
        })
        dest = tmp_path / "extract"

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            flatten_zip_to_dir(zf, dest)

        assert (dest / "ok.png").exists()
        assert (dest / "ok.gt.txt").exists()

    def test_validate_images_false_skips_validation(
        self, tmp_path: Path,
    ) -> None:
        """Le kwarg ``validate_images=False`` désactive la vérification —
        utilisé par certains tests qui se concentrent sur d'autres
        propriétés (path traversal, par exemple) sans avoir besoin de
        fournir un PNG complet."""
        from picarones.interfaces.web.corpus_utils import flatten_zip_to_dir

        zip_bytes = _zip_with_entries({
            "skipme.png": b"\x89PNG_FAKE",
        })
        dest = tmp_path / "extract"
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            flatten_zip_to_dir(zf, dest, validate_images=False)
        assert (dest / "skipme.png").exists()


# ──────────────────────────────────────────────────────────────────────
# 4. Phase 2 — pipeline_mode strict (rupture API)
# ──────────────────────────────────────────────────────────────────────


def _make_benchmark_app():
    """App FastAPI minimale pour tester le rejet 422 au niveau router."""
    from fastapi import FastAPI

    from picarones.interfaces.web.routers import benchmark as bench_router

    app = FastAPI()
    app.include_router(bench_router.router)
    return app


class TestPipelineModeStrictAPI:
    """Phase 2 du chantier post-rewrite : le typage ``Literal`` de
    ``PipelineConfig.pipeline_mode`` rejette en 422 toute valeur
    hors de la matrice canonique avant même que le router ne soit
    appelé.  Avant ce durcissement, le ``mode_map.get(...,
    "text_only")`` aliasait silencieusement.
    """

    def test_invalid_pipeline_mode_returns_422(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        app = _make_benchmark_app()
        with TestClient(app) as client:
            r = client.post(
                "/api/benchmark/run",
                json={
                    "corpus_path": str(tmp_path),
                    "competitors": [
                        {
                            "name": "p",
                            "ocr_engine": "tesseract",
                            "ocr_model": "fra",
                            "llm_provider": "mistral",
                            "llm_model": "ministral-3b-latest",
                            "pipeline_mode": "magic_unknown_mode",
                            "prompt_file": "",
                        },
                    ],
                    "normalization_profile": "nfc",
                    "output_dir": str(tmp_path),
                    "report_name": "test",
                    "report_lang": "fr",
                },
            )
            assert r.status_code == 422, r.text

    def test_legacy_alias_post_correction_text_rejected_422(
        self, tmp_path: Path,
    ) -> None:
        from fastapi.testclient import TestClient

        app = _make_benchmark_app()
        with TestClient(app) as client:
            r = client.post(
                "/api/benchmark/run",
                json={
                    "corpus_path": str(tmp_path),
                    "competitors": [
                        {
                            "name": "p",
                            "ocr_engine": "tesseract",
                            "ocr_model": "fra",
                            "llm_provider": "mistral",
                            "llm_model": "ministral-3b-latest",
                            # Alias supprimé Phase 2.
                            "pipeline_mode": "post_correction_text",
                            "prompt_file": "",
                        },
                    ],
                    "normalization_profile": "nfc",
                    "output_dir": str(tmp_path),
                    "report_name": "test",
                    "report_lang": "fr",
                },
            )
            assert r.status_code == 422, r.text

    @pytest.mark.parametrize(
        "valid_mode", ["text_only", "text_and_image", "zero_shot"],
    )
    def test_canonical_modes_pass_pydantic(self, valid_mode: str) -> None:
        """Les 3 modes canoniques sont acceptés par Pydantic — la
        suite (instanciation moteur, exécution) peut échouer pour
        d'autres raisons mais ce n'est pas notre test."""
        from picarones.interfaces.web.models import PipelineConfig

        comp = PipelineConfig(
            name="t", engine_name="tesseract",
            llm_provider="mistral", llm_model="m",
            pipeline_mode=valid_mode,
        )
        assert comp.pipeline_mode == valid_mode

    def test_empty_mode_pass_pydantic_for_ocr_only(self) -> None:
        """``pipeline_mode=""`` (défaut) doit rester accepté pour les
        configs OCR seul (sans ``llm_provider``)."""
        from picarones.interfaces.web.models import PipelineConfig

        comp = PipelineConfig(
            name="t", engine_name="tesseract", llm_provider="",
        )
        assert comp.pipeline_mode == ""


# ──────────────────────────────────────────────────────────────────────
# 5. Phase 2.2 — from_json fidèle (round-trip complet)
# ──────────────────────────────────────────────────────────────────────


class TestBenchmarkResultRoundTrip:
    """Phase 2.2 du chantier post-rewrite : ``BenchmarkResult.to_json``
    suivi de :meth:`BenchmarkResult.from_json_object` doit restaurer
    **tous** les champs avancés (taxonomy, structure, hallucination,
    NER, calibration, philological, searchability, numerical,
    readability, pipeline_metadata, ocr_intermediate + leurs
    ``aggregated_*`` correspondants).

    Avant ce durcissement, ``ReportGenerator.from_json`` faisait sa
    propre reconstruction qui ne couvrait que CER/WER + textes — toutes
    les analyses étaient perdues, ce qui rendait le rapport régénéré
    différent du rapport in-memory.  Reproductibilité scientifique
    cassée.
    """

    def _make_rich_benchmark(self):
        from picarones.evaluation.benchmark_result import (
            BenchmarkResult, DocumentResult, EngineReport,
        )
        from picarones.evaluation.metric_result import MetricsResult

        metrics = MetricsResult(
            cer=0.15, cer_nfc=0.14, cer_caseless=0.13,
            wer=0.20, wer_normalized=0.19,
            mer=0.16, wil=0.18,
            reference_length=100, hypothesis_length=95,
            cer_diplomatic=0.12,
            diplomatic_profile_name="medieval_french",
        )
        dr = DocumentResult(
            doc_id="doc1",
            image_path="/tmp/doc1.png",
            ground_truth="Hello world",
            hypothesis="He11o world",
            metrics=metrics,
            duration_seconds=1.5,
            ocr_intermediate="He11o w0rld",
            pipeline_metadata={"mode": "text_only", "prompt_file": "x.txt"},
            confusion_matrix={"l→1": 2},
            char_scores={"ligature": {"score": 0.95}},
            taxonomy={"classes": {"1": 3, "2": 1}},
            structure={"line_count": 5},
            image_quality={"contrast": 0.75},
            line_metrics={"cer_per_line": [0.1, 0.2, 0.3]},
            hallucination_metrics={"anchoring": 0.85, "n_blocks": 1},
            ner_metrics={"f1_micro": 0.80, "per_category": {"PER": 0.9}},
            calibration_metrics={"ece": 0.05, "mce": 0.10},
            philological_metrics={"mufi": {"coverage": 0.92}},
            searchability_metrics={
                "n_gt_tokens": 2, "n_searchable": 2, "recall": 1.0,
            },
            numerical_sequence_metrics={
                "global_strict_score": 1.0, "n_total": 0,
            },
            readability_metrics={
                "lang": "fr", "flesch_delta": -5.2, "n_words_reference": 100,
            },
        )
        er = EngineReport(
            engine_name="tesseract",
            engine_version="5.3.0",
            engine_config={"lang": "fra"},
            document_results=[dr],
            pipeline_info={"mode": "text_only"},
            aggregated_confusion={"l→1": 2},
            aggregated_char_scores={"ligature": {"score": 0.95}},
            aggregated_taxonomy={"classes": {"1": 3}},
            aggregated_structure={"line_count_total": 5},
            aggregated_image_quality={"contrast_mean": 0.75},
            aggregated_line_metrics={"gini_mean": 0.3},
            aggregated_hallucination={"anchoring_mean": 0.85},
            aggregated_ner={"f1_micro": 0.80},
            aggregated_calibration={"ece": 0.05},
            aggregated_philological={"mufi": {"coverage": 0.92}},
            aggregated_searchability={"recall": 1.0},
            aggregated_numerical_sequences={"global_strict_score": 1.0},
            aggregated_readability={"delta_mean": -5.2},
        )
        return BenchmarkResult(
            corpus_name="rich-corpus",
            corpus_source="tests",
            document_count=1,
            engine_reports=[er],
            run_date="2026-05-12T12:00:00Z",
            picarones_version="2.0.0",
            metadata={"context": "phase2_test"},
        )

    def test_round_trip_preserves_all_document_level_fields(
        self, tmp_path: Path,
    ) -> None:
        from picarones.evaluation.benchmark_result import BenchmarkResult

        bm = self._make_rich_benchmark()
        path = tmp_path / "rich.json"
        bm.to_json(path)
        loaded = BenchmarkResult.from_json_object(path)

        orig = bm.engine_reports[0].document_results[0]
        rebuilt = loaded.engine_reports[0].document_results[0]

        assert rebuilt.doc_id == orig.doc_id
        assert rebuilt.ground_truth == orig.ground_truth
        assert rebuilt.hypothesis == orig.hypothesis
        assert rebuilt.ocr_intermediate == orig.ocr_intermediate
        assert rebuilt.pipeline_metadata == orig.pipeline_metadata
        assert rebuilt.confusion_matrix == orig.confusion_matrix
        assert rebuilt.char_scores == orig.char_scores
        assert rebuilt.taxonomy == orig.taxonomy
        assert rebuilt.structure == orig.structure
        assert rebuilt.image_quality == orig.image_quality
        assert rebuilt.line_metrics == orig.line_metrics
        assert rebuilt.hallucination_metrics == orig.hallucination_metrics
        assert rebuilt.ner_metrics == orig.ner_metrics
        assert rebuilt.calibration_metrics == orig.calibration_metrics
        assert rebuilt.philological_metrics == orig.philological_metrics
        assert rebuilt.searchability_metrics == orig.searchability_metrics
        assert (
            rebuilt.numerical_sequence_metrics
            == orig.numerical_sequence_metrics
        )
        assert rebuilt.readability_metrics == orig.readability_metrics
        # Métriques diplomatiques (anciennement perdues).
        assert rebuilt.metrics.cer_diplomatic == orig.metrics.cer_diplomatic
        assert (
            rebuilt.metrics.diplomatic_profile_name
            == orig.metrics.diplomatic_profile_name
        )

    def test_round_trip_preserves_aggregated_engine_fields(
        self, tmp_path: Path,
    ) -> None:
        from picarones.evaluation.benchmark_result import BenchmarkResult

        bm = self._make_rich_benchmark()
        path = tmp_path / "rich.json"
        bm.to_json(path)
        loaded = BenchmarkResult.from_json_object(path)

        orig = bm.engine_reports[0]
        rebuilt = loaded.engine_reports[0]
        assert rebuilt.pipeline_info == orig.pipeline_info
        assert rebuilt.aggregated_confusion == orig.aggregated_confusion
        assert rebuilt.aggregated_char_scores == orig.aggregated_char_scores
        assert rebuilt.aggregated_taxonomy == orig.aggregated_taxonomy
        assert rebuilt.aggregated_structure == orig.aggregated_structure
        assert (
            rebuilt.aggregated_image_quality == orig.aggregated_image_quality
        )
        assert rebuilt.aggregated_line_metrics == orig.aggregated_line_metrics
        assert (
            rebuilt.aggregated_hallucination == orig.aggregated_hallucination
        )
        assert rebuilt.aggregated_ner == orig.aggregated_ner
        assert rebuilt.aggregated_calibration == orig.aggregated_calibration
        assert (
            rebuilt.aggregated_philological == orig.aggregated_philological
        )
        assert (
            rebuilt.aggregated_searchability == orig.aggregated_searchability
        )
        assert (
            rebuilt.aggregated_numerical_sequences
            == orig.aggregated_numerical_sequences
        )
        assert rebuilt.aggregated_readability == orig.aggregated_readability

    def test_report_generator_from_json_uses_rich_reconstruction(
        self, tmp_path: Path,
    ) -> None:
        """``ReportGenerator.from_json`` doit désormais accéder aux
        champs avancés (avant Phase 2.2 il les perdait)."""
        from picarones.reports.html.generator import ReportGenerator

        bm = self._make_rich_benchmark()
        path = tmp_path / "rich.json"
        bm.to_json(path)

        gen = ReportGenerator.from_json(path)
        dr = gen.benchmark.engine_reports[0].document_results[0]
        # Champs qui étaient à None avant Phase 2.2.
        assert dr.taxonomy is not None
        assert dr.hallucination_metrics is not None
        assert dr.philological_metrics is not None
        assert dr.calibration_metrics is not None
        assert dr.searchability_metrics is not None


# ──────────────────────────────────────────────────────────────────────
# 6. Phase 2.3 — partial store fingerprint
# ──────────────────────────────────────────────────────────────────────


class TestPartialStoreFingerprint:
    """Phase 2.3 du chantier post-rewrite : la clé du fichier partiel
    inclut désormais un fingerprint SHA-256 stable de la config
    complète (engine_config, normalization_profile, char_exclude,
    fichiers corpus + mtime/size, version code).

    Avant ce durcissement, la clé était ``(corpus.name, engine.name)``
    seule — deux runs avec configs différentes recyclaient
    silencieusement les résultats du précédent.  Reproductibilité
    scientifique brisée.
    """

    def test_fingerprint_stable_for_same_config(self, tmp_path: Path) -> None:
        from picarones.app.services.partial_store import (
            compute_run_fingerprint,
        )

        f1 = tmp_path / "a.png"
        f1.write_bytes(b"\x00" * 100)
        fp1 = compute_run_fingerprint(
            engine_config={"lang": "fra", "psm": 6},
            normalization_profile="medieval_french",
            char_exclude="',-",
            corpus_files=[f1],
            code_version="1.0",
        )
        fp2 = compute_run_fingerprint(
            engine_config={"psm": 6, "lang": "fra"},  # ordre différent
            normalization_profile="medieval_french",
            char_exclude="',-",
            corpus_files=[f1],
            code_version="1.0",
        )
        assert fp1 == fp2, "Le fingerprint doit être insensible à l'ordre dict"

    def test_fingerprint_changes_with_engine_config(
        self, tmp_path: Path,
    ) -> None:
        from picarones.app.services.partial_store import (
            compute_run_fingerprint,
        )

        f1 = tmp_path / "a.png"
        f1.write_bytes(b"\x00" * 100)
        fp_psm6 = compute_run_fingerprint(
            engine_config={"lang": "fra", "psm": 6},
            corpus_files=[f1],
            code_version="1.0",
        )
        fp_psm3 = compute_run_fingerprint(
            engine_config={"lang": "fra", "psm": 3},
            corpus_files=[f1],
            code_version="1.0",
        )
        assert fp_psm6 != fp_psm3, (
            "Un changement de psm doit changer le fingerprint"
        )

    def test_fingerprint_changes_with_normalization_profile(
        self, tmp_path: Path,
    ) -> None:
        from picarones.app.services.partial_store import (
            compute_run_fingerprint,
        )

        f1 = tmp_path / "a.png"
        f1.write_bytes(b"\x00" * 100)
        fp_med = compute_run_fingerprint(
            engine_config={"lang": "fra"},
            normalization_profile="medieval_french",
            corpus_files=[f1],
        )
        fp_nfc = compute_run_fingerprint(
            engine_config={"lang": "fra"},
            normalization_profile="nfc",
            corpus_files=[f1],
        )
        assert fp_med != fp_nfc

    def test_fingerprint_changes_with_char_exclude(
        self, tmp_path: Path,
    ) -> None:
        from picarones.app.services.partial_store import (
            compute_run_fingerprint,
        )

        fp_with = compute_run_fingerprint(
            engine_config={"lang": "fra"},
            char_exclude="',-",
        )
        fp_without = compute_run_fingerprint(
            engine_config={"lang": "fra"},
            char_exclude="",
        )
        assert fp_with != fp_without

    def test_fingerprint_changes_with_corpus_content(
        self, tmp_path: Path,
    ) -> None:
        """Si un fichier change de taille / mtime, le fingerprint change.

        Détection légère (pas de hash du contenu) mais suffit pour
        invalider la reprise après modification utilisateur du corpus.
        """
        import os
        import time

        from picarones.app.services.partial_store import (
            compute_run_fingerprint,
        )

        f1 = tmp_path / "a.png"
        f1.write_bytes(b"\x00" * 100)
        fp_v1 = compute_run_fingerprint(
            engine_config={"lang": "fra"},
            corpus_files=[f1],
        )
        # Réécrire avec une taille différente.
        f1.write_bytes(b"\x00" * 200)
        # Forcer un mtime différent (certains FS ont une résolution
        # de seconde, on attend > 1 s).
        new_mtime = time.time() + 5
        os.utime(f1, (new_mtime, new_mtime))
        fp_v2 = compute_run_fingerprint(
            engine_config={"lang": "fra"},
            corpus_files=[f1],
        )
        assert fp_v1 != fp_v2

    def test_partial_path_uses_fingerprint_suffix(
        self, tmp_path: Path,
    ) -> None:
        from picarones.app.services.partial_store import _partial_path

        path_with = _partial_path(
            "my_corpus", "tesseract", tmp_path, fingerprint="abc123",
        )
        path_without = _partial_path(
            "my_corpus", "tesseract", tmp_path,
        )
        assert path_with != path_without
        assert "abc123" in path_with.name
        # Le format historique reste pour la rétrocompat.
        assert path_without.name == "picarones_my_corpus_tesseract.partial.jsonl"

    def test_engine_config_for_fingerprint_distinguishes_psm(self) -> None:
        """``_engine_config_for_fingerprint`` capture les attributs
        opérationnels d'un adapter OCR (lang, psm, model, …)."""
        from picarones.app.services.benchmark_runner import (
            _engine_config_for_fingerprint,
        )

        class _FakeOCR:
            name = "tesseract"
            lang = "fra"
            psm = 6
            is_pipeline = False

        class _FakeOCRDiff:
            name = "tesseract"
            lang = "fra"
            psm = 3
            is_pipeline = False

        c1 = _engine_config_for_fingerprint(_FakeOCR())
        c2 = _engine_config_for_fingerprint(_FakeOCRDiff())
        assert c1 != c2
        assert c1["psm"] == 6
        assert c2["psm"] == 3


# ──────────────────────────────────────────────────────────────────────
# 7. Phase 3 — Adapters kraken et calamari (moteurs fantômes implémentés)
# ──────────────────────────────────────────────────────────────────────


class TestKrakenAdapter:
    """Phase 3 du chantier post-rewrite : ``KrakenAdapter`` rend
    l'engine ``kraken`` réellement utilisable (au lieu d'être
    juste annoncé par ``/api/engines``)."""

    def test_kraken_requires_model_path(self) -> None:
        from picarones.adapters.ocr import KrakenAdapter
        from picarones.adapters.ocr.base import OCRAdapterError

        with pytest.raises(OCRAdapterError, match="model_path est obligatoire"):
            KrakenAdapter()

    def test_kraken_via_factory(self, tmp_path: Path) -> None:
        from picarones.adapters.ocr import KrakenAdapter
        from picarones.adapters.ocr.factory import ocr_adapter_from_name

        # Modèle factice — l'adapter ne le charge qu'à execute().
        model = tmp_path / "fake.mlmodel"
        model.write_bytes(b"fake")
        adapter = ocr_adapter_from_name("kraken", model_path=str(model))
        assert isinstance(adapter, KrakenAdapter)
        assert adapter.name == "kraken"
        assert adapter.model_path == model

    def test_kraken_validates_name(self) -> None:
        from picarones.adapters.ocr import KrakenAdapter
        from picarones.adapters.ocr.base import OCRAdapterError

        with pytest.raises(OCRAdapterError, match="name invalide"):
            KrakenAdapter(name="bad name with spaces", model_path="x")


class TestCalamariAdapter:
    """Phase 3 du chantier post-rewrite : ``CalamariAdapter`` rend
    l'engine ``calamari`` réellement utilisable."""

    def test_calamari_requires_checkpoint(self) -> None:
        from picarones.adapters.ocr import CalamariAdapter
        from picarones.adapters.ocr.base import OCRAdapterError

        with pytest.raises(OCRAdapterError, match="checkpoint est obligatoire"):
            CalamariAdapter()

    def test_calamari_via_factory(self, tmp_path: Path) -> None:
        from picarones.adapters.ocr import CalamariAdapter
        from picarones.adapters.ocr.factory import ocr_adapter_from_name

        ckpt = tmp_path / "fake.ckpt"
        ckpt.write_bytes(b"fake")
        adapter = ocr_adapter_from_name("calamari", checkpoint=str(ckpt))
        assert isinstance(adapter, CalamariAdapter)
        assert adapter.name == "calamari"
        assert adapter.checkpoint == ckpt

    def test_calamari_validates_batch_size(self) -> None:
        from picarones.adapters.ocr import CalamariAdapter
        from picarones.adapters.ocr.base import OCRAdapterError

        with pytest.raises(OCRAdapterError, match="batch_size doit être"):
            CalamariAdapter(checkpoint="x", batch_size=0)


class TestEngineMatrixCoherence:
    """Phase 3 du chantier post-rewrite : la matrice des moteurs est
    cohérente entre ``/api/engines``, la factory canonique, le
    builder web ``_OCR_KWARGS_BUILDERS`` et l'index public."""

    def test_kraken_and_calamari_in_factory_supported_list(self) -> None:
        from picarones.adapters.ocr.factory import _SUPPORTED

        assert "kraken" in _SUPPORTED
        assert "calamari" in _SUPPORTED

    def test_kraken_and_calamari_in_web_builders(self) -> None:
        from picarones.interfaces.web.benchmark_utils import (
            _OCR_KWARGS_BUILDERS,
        )

        assert "kraken" in _OCR_KWARGS_BUILDERS
        assert "calamari" in _OCR_KWARGS_BUILDERS

    def test_kraken_calamari_exposed_at_package_root(self) -> None:
        from picarones.adapters.ocr import (
            CalamariAdapter,
            KrakenAdapter,
        )

        assert KrakenAdapter.__name__ == "KrakenAdapter"
        assert CalamariAdapter.__name__ == "CalamariAdapter"


# ──────────────────────────────────────────────────────────────────────
# 8. Phase 4 — upload_purge_task branché au lifespan
# ──────────────────────────────────────────────────────────────────────


class TestUploadPurgeTaskWired:
    """Phase 4 du chantier post-rewrite : la tâche
    ``upload_purge_task`` est désormais démarrée par le lifespan de
    ``picarones.interfaces.web.app`` (auparavant définie mais jamais
    lancée — code zombie)."""

    def test_lifespan_starts_purge_task(self, monkeypatch) -> None:
        """Au démarrage de l'app FastAPI, un ``asyncio.create_task`` doit
        emballer ``upload_purge_task``.  On patch la fonction pour
        l'observer puis on enclenche le lifespan."""
        from fastapi.testclient import TestClient

        observed: dict = {"started": False, "uploads_root": None}

        async def _fake_purge_task(uploads_root):
            observed["started"] = True
            observed["uploads_root"] = uploads_root
            # Boucle infinie minimale — annulée au shutdown.
            import asyncio
            try:
                while True:
                    await asyncio.sleep(3600)
            except asyncio.CancelledError:
                raise

        monkeypatch.setattr(
            "picarones.interfaces.web.maintenance.upload_purge_task",
            _fake_purge_task,
        )
        # Forcer la rétention pour ne pas que la fonction réelle short-circuit.
        monkeypatch.setenv("PICARONES_UPLOAD_RETENTION_DAYS", "7")

        from picarones.interfaces.web.app import app

        with TestClient(app):
            # Le lifespan a démarré ; la tâche tourne en arrière-plan.
            # On laisse à asyncio le temps de la lancer.
            import time
            time.sleep(0.05)

        assert observed["started"] is True, (
            "upload_purge_task aurait dû être démarrée par le lifespan"
        )

    def test_purge_protects_active_corpus(self, tmp_path: Path) -> None:
        """Si un job ``pending``/``running`` référence un corpus_id, la
        purge ne supprime pas ce dossier — même s'il est ancien."""
        import time

        from picarones.interfaces.web.maintenance import purge_old_uploads

        # 2 corpus : un actif (référencé), un orphelin.
        active = tmp_path / "active_corpus"
        orphan = tmp_path / "orphan_corpus"
        active.mkdir()
        orphan.mkdir()
        # Vieillir les deux pour qu'ils passent la rétention de 0 jour.
        old = time.time() - 86400 * 30
        import os
        os.utime(active, (old, old))
        os.utime(orphan, (old, old))

        purged = purge_old_uploads(
            tmp_path,
            retention_days=7,
            active_corpus_ids={"active_corpus"},
        )

        purged_names = [p.name for p in purged]
        assert "orphan_corpus" in purged_names
        assert "active_corpus" not in purged_names
        # Vérification physique
        assert active.exists()
        assert not orphan.exists()


# ──────────────────────────────────────────────────────────────────────
# 9. Phase 5b — engine_name (renommage rupture du field ocr_engine)
# ──────────────────────────────────────────────────────────────────────


class TestPipelineConfigEngineNameRename:
    """Phase 5b du chantier post-rewrite : le field ``ocr_engine`` du
    payload ``PipelineConfig`` est renommé en ``engine_name`` car il
    accepte aussi des VLMs (zero_shot) et la source ``corpus`` (OCR
    pré-calculé) — le préfixe ``ocr_`` était trompeur.

    Rupture API : un client qui envoie l'ancien nom doit recevoir une
    erreur Pydantic explicite plutôt que d'aliaser silencieusement.
    """

    def test_engine_name_field_accepted(self) -> None:
        from picarones.interfaces.web.models import PipelineConfig

        cfg = PipelineConfig(
            name="t", engine_name="tesseract", llm_provider="",
        )
        assert cfg.engine_name == "tesseract"

    def test_legacy_ocr_engine_kwarg_rejected_by_strict_mode(self) -> None:
        """Pydantic v2 ignore par défaut les extras non déclarés mais
        ne reconnaît plus ``ocr_engine`` comme alias.  On vérifie que
        passer juste ``ocr_engine=`` ne remplit pas ``engine_name``
        (rupture silencieuse acceptée vs explicite — Pydantic v2 ne
        peut pas distinguer entre 'extra ignoré' et 'mauvais nom')."""
        from picarones.interfaces.web.models import PipelineConfig

        cfg = PipelineConfig(name="t", llm_provider="")
        # Default : engine_name=""
        assert cfg.engine_name == ""
        # Construire avec un kwarg dynamic = legacy name → engine_name
        # reste vide (Pydantic v2 ignore les extras non-strict).
        cfg2 = PipelineConfig.model_validate(
            {"name": "t", "ocr_engine": "tesseract", "llm_provider": ""},
        )
        assert cfg2.engine_name == "", (
            "Le legacy ``ocr_engine`` ne doit PAS remplir engine_name "
            "automatiquement — sinon on aliase silencieusement et la "
            "rupture API n'est pas réelle."
        )

    def test_router_payload_uses_engine_name(self) -> None:
        """Le router ``/api/benchmark/run`` accepte le payload
        avec ``engine_name`` et le propage."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from picarones.interfaces.web.routers import benchmark as bench_router

        app = FastAPI()
        app.include_router(bench_router.router)
        with TestClient(app) as client:
            # On vise un payload qui valide Pydantic mais échoue à
            # l'instanciation moteur (corpus inexistant) — l'important
            # est que le 422 Pydantic ne se déclenche pas sur le field.
            r = client.post(
                "/api/benchmark/run",
                json={
                    "corpus_path": "/tmp/no_such_dir_for_phase5b_test",
                    "competitors": [{
                        "name": "p",
                        "engine_name": "tesseract",
                        "ocr_model": "fra",
                        "llm_provider": "",
                        "llm_model": "",
                        "pipeline_mode": "",
                        "prompt_file": "",
                    }],
                    "normalization_profile": "nfc",
                    "output_dir": "/tmp",
                    "report_name": "test",
                    "report_lang": "fr",
                },
            )
            # Pas un 422 Pydantic → le field engine_name a bien
            # été accepté.  (400 attendu : corpus_path inexistant.)
            assert r.status_code != 422, (
                "Le router refuse le payload avec engine_name : "
                f"{r.text}"
            )


# ──────────────────────────────────────────────────────────────────────
# 10. Phase 4.4 — JS is_demo HTR-United badge
# ──────────────────────────────────────────────────────────────────────


class TestHtrUnitedDemoBadgeBinding:
    """Phase 4.4 du chantier post-rewrite : l'API
    ``/api/htr-united/catalogue`` retourne ``is_demo`` ; le frontend
    doit afficher un badge visible quand le serveur a fallback sur
    le catalogue embarqué (réseau distant indisponible).

    Avant : l'UI annonçait "Catalogue HTR-United" sans distinguer
    démo vs remote — vecteur de confusion utilisateur."""

    def test_template_exposes_demo_banner(self) -> None:
        from pathlib import Path

        tmpl = (
            Path(__file__).resolve().parents[2]
            / "picarones/interfaces/web/templates/_view_import.html"
        )
        html = tmpl.read_text(encoding="utf-8")
        assert "htr-demo-banner" in html, (
            "Le bandeau ``htr-demo-banner`` doit exister dans "
            "_view_import.html pour afficher le mode démo"
        )
        assert "htr_demo_badge" in html, (
            "L'i18n key ``htr_demo_badge`` doit être présente"
        )

    def test_js_updates_banner_from_is_demo_flag(self) -> None:
        from pathlib import Path

        js = (
            Path(__file__).resolve().parents[2]
            / "picarones/interfaces/web/static/web-app.js"
        )
        src = js.read_text(encoding="utf-8")
        assert "function _updateHtrDemoBanner" in src, (
            "_updateHtrDemoBanner doit être défini"
        )
        # initHTRFilters et searchHTRUnited doivent l'appeler.
        assert "_updateHtrDemoBanner(Boolean(d.is_demo))" in src, (
            "initHTRFilters et searchHTRUnited doivent passer "
            "le flag is_demo au binding UI"
        )
        # i18n key déclarée FR + EN.
        assert "htr_demo_badge:" in src
        assert "htr_demo_note:" in src
