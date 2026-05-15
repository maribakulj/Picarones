"""Régression pour les trous architecturaux identifiés par l'audit
B3-final (mai 2026, suite au fix du bug "Aucun document valide trouvé"
commit 7f313a9).

Trous adressés ici :

- **Trou 1** : ``attach_ner_metrics_to_benchmark`` faisait une lookup
  silencieusement vide entre les doc_id normalisés
  (``DocumentResult.doc_id`` ← ``_safe_doc_id`` côté CorpusSpec) et les
  doc_id legacy bruts (``Document.doc_id``).  Symptôme : NER attaché à
  zéro doc pour un corpus avec des doc_id "sales" (espaces, accents).

- **Trou 9** : ``execute_preset(..., output_json=set)`` SANS
  ``corpus_legacy`` levait ``ValueError: Aucun document valide
  trouvé`` cryptique au lieu d'un message qui pointe vers le contrat
  cassé (mode preset = workspace gt-only sans images).
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────
# Trou 1 — NER doc_id mismatch (silent lookup miss)
# ──────────────────────────────────────────────────────────────────────


class TestTrou1NerDocIdNormalization:
    """``attach_ner_metrics_to_benchmark`` doit indexer le corpus avec
    la même normalisation que ``CorpusSpec`` pour matcher les
    ``DocumentResult.doc_id`` du benchmark_result.

    Avant le fix, un ``Document(doc_id="Image 01")`` se retrouvait avec
    un ``DocumentResult.doc_id="Image_01"`` côté benchmark_result, mais
    le ``docs_by_id = {d.doc_id: d for d in corpus.documents}`` indexait
    sur ``"Image 01"`` → lookup miss → NER skip silencieux.
    """

    def _make_corpus_with_dirty_doc_id(self, tmp_path: Path):
        """Corpus avec un doc_id qui contient espace + accent."""
        from picarones.domain.artifacts import ArtifactType
        from picarones.evaluation.corpus import Corpus, Document, EntitiesGT

        img = tmp_path / "image01.png"
        img.write_bytes(b"x")
        # doc_id "sale" : espace, accent, et caractère non-alphanum.
        dirty_doc_id = "Médiéval 01"
        doc = Document(
            image_path=img,
            ground_truth="Jean habite Paris",
            doc_id=dirty_doc_id,
            ground_truths={
                ArtifactType.ENTITIES: EntitiesGT(
                    entities=[
                        {"label": "PER", "start": 0, "end": 4, "text": "Jean"},
                        {"label": "LOC", "start": 12, "end": 17, "text": "Paris"},
                    ],
                ),
            },
        )
        corpus = Corpus(name="trou1", documents=[doc])
        return corpus, dirty_doc_id

    def _make_benchmark_result_with_normalized_id(self, normalized_id: str):
        """BenchmarkResult mock où le DocumentResult porte le doc_id
        DÉJÀ normalisé (= ce qu'aurait produit le converter via
        DocumentRef.id)."""
        from picarones.evaluation.benchmark_result import (
            BenchmarkResult, DocumentResult, EngineReport,
        )
        from picarones.evaluation.metric_result import MetricsResult

        dr = DocumentResult(
            doc_id=normalized_id,
            image_path="image01.png",
            ground_truth="Jean habite Paris",
            hypothesis="Jean habite Paris",
            metrics=MetricsResult(
                cer=0.0, wer=0.0,
                reference_length=18, hypothesis_length=18,
            ),
            duration_seconds=0.1,
        )
        report = EngineReport(
            engine_name="mock",
            engine_version="0.0.0-test",
            engine_config={},
            document_results=[dr],
        )
        return BenchmarkResult(
            corpus_name="trou1",
            corpus_source=None,
            document_count=1,
            engine_reports=[report],
        )

    def test_ner_attached_despite_dirty_doc_id(self, tmp_path: Path) -> None:
        """Bug Trou 1 : un doc_id "Médiéval 01" (espace + accent) doit
        quand même recevoir ses NER metrics."""
        from picarones.app.services._benchmark_conversions import _safe_doc_id
        from picarones.app.services._benchmark_ner import (
            attach_ner_metrics_to_benchmark,
        )

        corpus, dirty_doc_id = self._make_corpus_with_dirty_doc_id(tmp_path)
        # ID normalisé tel que le converter le produirait.
        normalized = _safe_doc_id(dirty_doc_id)
        assert normalized != dirty_doc_id, (
            f"Le test n'a de sens que si _safe_doc_id modifie l'id : "
            f"{dirty_doc_id!r} → {normalized!r}"
        )

        bm = self._make_benchmark_result_with_normalized_id(normalized)

        # Extractor mock qui retourne les 2 entités attendues.
        def fake_extractor(text: str) -> list[dict]:
            return [
                {"label": "PER", "start": 0, "end": 4, "text": "Jean"},
                {"label": "LOC", "start": 12, "end": 17, "text": "Paris"},
            ]

        attach_ner_metrics_to_benchmark(bm, corpus, fake_extractor)

        # Sans le fix : ner_metrics restait None car lookup
        # ``docs_by_id.get("Médiéval_01")`` retournait None
        # (clés indexées sur dirty_doc_id).  Avec le fix, lookup
        # réussit et ner_metrics est attaché.
        dr = bm.engine_reports[0].document_results[0]
        assert dr.ner_metrics is not None, (
            "NER non-attaché : la lookup docs_by_id a probablement raté "
            "à cause d'une normalisation de doc_id non appliquée côté "
            "attach_ner_metrics_to_benchmark."
        )
        # Et l'agrégat engine-level doit aussi être set.
        assert bm.engine_reports[0].aggregated_ner is not None


# ──────────────────────────────────────────────────────────────────────
# Trou 9 — corpus_legacy oublié dans execute_preset (regression piège)
# ──────────────────────────────────────────────────────────────────────


class TestTrou9CorpusLegacyMissingMessage:
    """``execute_preset(..., output_json=set)`` SANS ``corpus_legacy``
    doit lever une ``ValueError`` avec un message **explicite** qui
    pointe vers le contrat cassé.

    Sans cette validation, on retombe sur le bug 7f313a9 avec un
    message cryptique ("Aucun document valide trouvé dans /tmp/.../gt")
    qui suggérait à tort que c'est le corpus qui est mal formé.
    """

    def test_missing_corpus_legacy_raises_explicit_error(
        self, tmp_path: Path,
    ) -> None:
        """Reproduit le contexte du bug : helper preset complet, mais
        sans passer ``corpus_legacy`` à ``execute_preset``."""
        from picarones.adapters.ocr.base import BaseOCRAdapter
        from picarones.app.services import (
            RunOrchestrator, prepare_preset_args,
        )
        from picarones.domain.artifacts import Artifact, ArtifactType
        from picarones.evaluation.corpus import Corpus, Document

        class _MockOCR(BaseOCRAdapter):
            def __init__(self) -> None:
                self._name = "mock"

            @property
            def name(self) -> str:
                return self._name

            def execute(self, inputs, params, context):
                out = (
                    Path(context.workspace_uri)
                    / f"{context.document_id}.txt"
                )
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("ok", encoding="utf-8")
                return {
                    ArtifactType.RAW_TEXT: Artifact(
                        id=f"{context.document_id}:{self._name}:raw_text",
                        document_id=context.document_id,
                        type=ArtifactType.RAW_TEXT,
                        produced_by_step="ocr",
                        uri=str(out),
                    ),
                }

        img = tmp_path / "doc0.png"
        img.write_bytes(b"x")
        corpus = Corpus(
            name="trou9",
            documents=[Document(
                image_path=img, ground_truth="ok", doc_id="doc0",
            )],
        )

        ws = tmp_path / "ws"
        ws.mkdir()
        args = prepare_preset_args(
            corpus, [_MockOCR()],
            workspace_dir=ws / "gt",
            output_dir=ws / "run",
            output_json=str(tmp_path / "results.json"),
        )

        # Appel direct sans corpus_legacy → doit lever une ValueError
        # avec un message explicite mentionnant ``corpus_legacy``.
        # Avant le fix Trou 9, le message était "Aucun document
        # valide trouvé" — cryptique et pointait à tort vers un
        # corpus mal formé.  Maintenant le wrap du try/except dans
        # ``_persist_legacy_benchmark_json`` enrichit le message pour
        # indiquer le contrat cassé (kwarg ``corpus_legacy``
        # manquant en mode preset).
        with pytest.raises(ValueError) as exc_info:
            RunOrchestrator(ws / "run").execute_preset(
                spec=args.spec,
                corpus_spec=args.corpus_spec,
                extracted_dir=args.extracted_dir,
                pipeline_specs=args.pipeline_specs,
                adapter_resolver=args.adapter_resolver,
                adapter_kwargs=args.adapter_kwargs,
                # corpus_legacy=corpus,  ← INTENTIONNELLEMENT OMIS
            )
        msg = str(exc_info.value)
        assert "corpus_legacy" in msg, (
            f"Message d'erreur ne mentionne pas 'corpus_legacy' "
            f"(régression du fix Trou 9) : {msg!r}"
        )

    def test_corpus_legacy_provided_no_error(self, tmp_path: Path) -> None:
        """Sanity : avec ``corpus_legacy`` fourni, pas d'erreur (path
        nominal du fix 7f313a9)."""
        from picarones.adapters.ocr.base import BaseOCRAdapter
        from picarones.app.services import (
            RunOrchestrator, prepare_preset_args,
        )
        from picarones.domain.artifacts import Artifact, ArtifactType
        from picarones.evaluation.corpus import Corpus, Document

        class _MockOCR(BaseOCRAdapter):
            def __init__(self) -> None:
                self._name = "mock"

            @property
            def name(self) -> str:
                return self._name

            def execute(self, inputs, params, context):
                out = (
                    Path(context.workspace_uri)
                    / f"{context.document_id}.txt"
                )
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("ok", encoding="utf-8")
                return {
                    ArtifactType.RAW_TEXT: Artifact(
                        id=f"{context.document_id}:{self._name}:raw_text",
                        document_id=context.document_id,
                        type=ArtifactType.RAW_TEXT,
                        produced_by_step="ocr",
                        uri=str(out),
                    ),
                }

        img = tmp_path / "doc0.png"
        img.write_bytes(b"x")
        corpus = Corpus(
            name="trou9_ok",
            documents=[Document(
                image_path=img, ground_truth="ok", doc_id="doc0",
            )],
        )

        ws = tmp_path / "ws"
        ws.mkdir()
        out_json = tmp_path / "results.json"
        args = prepare_preset_args(
            corpus, [_MockOCR()],
            workspace_dir=ws / "gt",
            output_dir=ws / "run",
            output_json=str(out_json),
        )

        orch = RunOrchestrator(ws / "run").execute_preset(
            spec=args.spec,
            corpus_spec=args.corpus_spec,
            extracted_dir=args.extracted_dir,
            pipeline_specs=args.pipeline_specs,
            adapter_resolver=args.adapter_resolver,
            adapter_kwargs=args.adapter_kwargs,
            corpus_legacy=corpus,
        )
        assert orch.run_result.n_documents == 1
        assert out_json.exists()
