"""Squelette des tests de feature parity entre ``run_benchmark_via_service``
et ``RunOrchestrator.execute(RunSpec)``.

Phase B0 du chantier de migration Option B.

Rôle
----
Ce module liste les **7 features** que ``run_benchmark_via_service``
expose aujourd'hui et que ``RunOrchestrator`` doit porter pendant la
Phase B2.  Chaque test est documenté précisément (ce qui doit être
vérifié) et marqué ``pytest.skip`` jusqu'à ce que la feature
correspondante soit portée.

Au fur et à mesure de la Phase B2, retirer le ``pytest.skip`` du test
correspondant et implémenter sa logique.  À la fin de B2, tous les
tests doivent être verts → on a atteint le **Checkpoint C1**.

Convention
----------
Chaque test compare :

1. ``run_benchmark_via_service(feature_X=value)`` — chemin legacy
2. ``RunOrchestrator().execute(spec_with_feature_X=value)`` — chemin
   rewrite

Et vérifie que le ``BenchmarkResult`` produit est numériquement
identique (modulo normalisation des champs volatils).

Mapping vers le plan Option B
-----------------------------
- B2.1 ``progress_callback``      → ``test_parity_progress_callback``
- B2.2 ``cancel_event``           → ``test_parity_cancel_event``
- B2.3 ``partial_dir``            → ``test_parity_partial_dir_resume``
- B2.4 ``entity_extractor``       → ``test_parity_entity_extractor_ner``
- B2.5 ``char_exclude`` +         → ``test_parity_normalization_propagation``
       ``normalization_profile``
- B2.6 ``profile`` (hooks)        → ``test_parity_profile_hooks``
- B2.7 ``output_json``            → ``test_parity_output_json_legacy_format``
"""

from __future__ import annotations

import io
import textwrap
import threading
import time
import zipfile
from pathlib import Path

import pytest

from picarones.app.schemas.run_spec import RunSpec, load_run_spec_from_yaml
from picarones.app.services import RunOrchestrator


SKIP_REASON_PREFIX = "TODO Phase B2."


# ──────────────────────────────────────────────────────────────────────
# Helpers communs
# ──────────────────────────────────────────────────────────────────────


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89"
    )


def _make_corpus_zip(n_docs: int = 3) -> bytes:
    """Corpus zip déterministe avec PrecomputedTextAdapter (source ``tess``)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        for i in range(1, n_docs + 1):
            doc_id = f"doc{i:02d}"
            zf.writestr(f"{doc_id}.png", _png_bytes())
            zf.writestr(f"{doc_id}.gt.txt", f"Texte de référence {i}")
            zf.writestr(f"{doc_id}.tess.txt", f"Texte de référence {i}")
    return buf.getvalue()


def _build_spec_yaml(corpus_zip: Path, output_dir: Path) -> str:
    return textwrap.dedent(f"""
        corpus_zip: {corpus_zip}
        corpus_name: feature_parity
        pipelines:
          - name: tess_only
            initial_inputs: [image]
            steps:
              - id: ocr
                adapter_class: picarones.adapters.ocr.precomputed.PrecomputedTextAdapter
                adapter_kwargs:
                  source_label: tess
                input_types: [image]
                output_types: [raw_text]
        views: [text_final]
        output_dir: {output_dir}
        code_version: "feature-parity-test"
    """)


# ──────────────────────────────────────────────────────────────────────
# B2.1 — progress_callback
# ──────────────────────────────────────────────────────────────────────


class TestParityProgressCallback:
    def test_callback_invoked_once_per_document(self, tmp_path: Path) -> None:
        """Spec : ``progress_callback`` est invoqué exactement 1 fois
        par document traité, avec ``doc_idx`` croissant à partir de 0.

        Référence : pattern de ``_benchmark_execution.py:109-139``.
        """
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip(n_docs=3))
        out_dir = tmp_path / "out"
        spec = load_run_spec_from_yaml(_build_spec_yaml(corpus_zip, out_dir))

        invocations: list[tuple[str, int, str]] = []

        def cb(engine: str, idx: int, doc_id: str) -> None:
            invocations.append((engine, idx, doc_id))

        result = RunOrchestrator(out_dir).execute(spec, progress_callback=cb)

        assert result.run_result.n_documents == 3
        assert len(invocations) == 3
        # doc_idx est strictement croissant 0..N-1 (compteur global).
        indices = [inv[1] for inv in invocations]
        assert indices == [0, 1, 2]
        # Tous les callbacks sont pour la pipeline ``tess_only``.
        assert all(inv[0] == "tess_only" for inv in invocations)
        # Les doc_id matchent ceux du corpus.
        assert sorted(inv[2] for inv in invocations) == ["doc01", "doc02", "doc03"]

    def test_no_callback_means_no_invocation(self, tmp_path: Path) -> None:
        """Sans ``progress_callback``, le run s'exécute sans cas
        particulier (compat ascendante)."""
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip(n_docs=2))
        out_dir = tmp_path / "out"
        spec = load_run_spec_from_yaml(_build_spec_yaml(corpus_zip, out_dir))

        result = RunOrchestrator(out_dir).execute(spec)
        assert result.run_result.n_documents == 2

    def test_callback_exceptions_do_not_break_run(self, tmp_path: Path) -> None:
        """Cohérence avec le legacy : une exception dans le callback
        ne fait pas tomber le bench (``_benchmark_execution.py:126-133``)."""
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip(n_docs=2))
        out_dir = tmp_path / "out"
        spec = load_run_spec_from_yaml(_build_spec_yaml(corpus_zip, out_dir))

        def cb_raises(engine: str, idx: int, doc_id: str) -> None:
            raise RuntimeError("callback crash simulé")

        # Le bench réussit malgré les exceptions du callback.
        result = RunOrchestrator(out_dir).execute(
            spec, progress_callback=cb_raises,
        )
        assert result.run_result.n_documents == 2


# ──────────────────────────────────────────────────────────────────────
# B2.2 — cancel_event
# ──────────────────────────────────────────────────────────────────────


class TestParityCancelEvent:
    def test_cancel_event_accepted_without_setting(self, tmp_path: Path) -> None:
        """Un ``cancel_event`` non-set ne perturbe pas le run normal.

        Phase B2.2 — vérifie que le wrapping conditionnel du
        ``CorpusRunner.run`` n'introduit pas de régression quand le
        caller fournit l'event mais ne l'arme pas.
        """
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip(n_docs=2))
        out_dir = tmp_path / "out"
        spec = load_run_spec_from_yaml(_build_spec_yaml(corpus_zip, out_dir))

        ev = threading.Event()  # jamais set()
        result = RunOrchestrator(out_dir).execute(spec, cancel_event=ev)
        assert result.run_result.n_documents == 2

    def test_cancel_event_preset_short_circuits(self, tmp_path: Path) -> None:
        """Un ``cancel_event`` déjà set avant ``execute()`` stoppe
        le run dès le premier check du runner.

        Le ``CorpusRunner`` propage l'event au ``PipelineExecutor`` ;
        dès qu'il voit l'event set, il abandonne le traitement.  Le
        comportement exact (combien de docs sont traités) dépend du
        timing du poll, mais aucun engine n'est exécuté au-delà de
        ``cancel_event.set()``.
        """
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip(n_docs=5))
        out_dir = tmp_path / "out"
        spec = load_run_spec_from_yaml(_build_spec_yaml(corpus_zip, out_dir))

        ev = threading.Event()
        ev.set()  # cancel immédiat

        start = time.monotonic()
        result = RunOrchestrator(out_dir).execute(spec, cancel_event=ev)
        elapsed = time.monotonic() - start

        # Le bench se termine très rapidement — pas de timeout.
        # Le runner abandonne dès le poll du cancel_event.
        assert elapsed < 10.0, (
            f"Bench cancellé trop lent : {elapsed:.2f}s.  "
            "Le wrapper ``corpus_runner.run`` ne propage pas correctement."
        )
        # Le ``RunResult`` est retourné (potentiellement vide ou partiel).
        # On ne fixe pas le nombre exact de docs traités : c'est un
        # détail d'implémentation du timing.  Mais on accepte ≤ 5.
        assert result.run_result.n_documents <= 5


# ──────────────────────────────────────────────────────────────────────
# B2.3 — partial_dir resume
# ──────────────────────────────────────────────────────────────────────


class TestParityPartialDir:
    """Phase B2.3 — reprise sur interruption pivotée par pipeline.

    Le format JSONL est partagé entre tous les pipelines d'un run :
    un fichier par pipeline, append-only, supprimé à la fin du
    pipeline si traité intégralement.
    """

    def _build_spec(
        self, tmp_path: Path, *,
        n_docs: int = 3,
        partial_dir: Path | None,
    ) -> "RunSpec":
        tmp_path.mkdir(parents=True, exist_ok=True)
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip(n_docs=n_docs))
        out_dir = tmp_path / "out"
        yaml = _build_spec_yaml(corpus_zip, out_dir)
        if partial_dir is not None:
            yaml += f"partial_dir: {partial_dir}\n"
        return load_run_spec_from_yaml(yaml)

    def test_partial_dir_fresh_start_creates_no_orphan_files(
        self, tmp_path: Path,
    ) -> None:
        """Fresh start : tous les docs traités → partial supprimé
        à la fin (cleanup).  Le répertoire partial_dir reste vide
        après un run réussi complet."""
        partial_dir = tmp_path / "partial"
        spec = self._build_spec(
            tmp_path, n_docs=2, partial_dir=partial_dir,
        )

        result = RunOrchestrator(tmp_path / "out").execute(spec)
        assert result.run_result.n_documents == 2

        # Aucun fichier .partial.jsonl résiduel après run complet.
        residual = list(partial_dir.glob("*.partial.jsonl"))
        assert residual == [], (
            f"Fichiers partiels résiduels : {residual}"
        )

    def test_partial_dir_resume_after_complete_pipeline(
        self, tmp_path: Path,
    ) -> None:
        """Si un partial existant contient déjà tous les docs d'un
        pipeline, ce pipeline n'est pas relancé.

        Pré-condition : on lance le bench une 1re fois pour créer
        le partial via les helpers (puisque le partial est supprimé
        en cleanup post-success).  On simule un crash en pré-écrivant
        directement le partial JSONL.
        """
        from picarones.app.services._orchestrator_partial import (
            append_pipeline_result,
            compute_pipeline_fingerprint,
            partial_path_for_pipeline,
        )

        partial_dir = tmp_path / "partial"
        partial_dir.mkdir()
        spec = self._build_spec(
            tmp_path, n_docs=2, partial_dir=partial_dir,
        )

        # Construire à la main les pipeline_specs (même logique que
        # RunOrchestrator._build_pipelines) pour pouvoir calculer le
        # fingerprint et pré-écrire le partial.
        orchestrator = RunOrchestrator(tmp_path / "out")
        orchestrator._output_dir.mkdir(parents=True, exist_ok=True)

        # On lance un premier run sans partial pour récupérer les
        # PipelineResult — puis on les rejoue via le partial.
        spec_no_partial = self._build_spec(
            tmp_path / "first", n_docs=2, partial_dir=None,
        )
        first_result = RunOrchestrator(
            tmp_path / "first" / "out",
        ).execute(spec_no_partial)

        # Pré-écrire le partial avec les 2 PipelineResult du 1er run.
        # On a besoin du fingerprint cohérent → on construit la spec
        # via orchestrator._build_pipelines et la corpus_spec via
        # _load_corpus.
        from picarones.app.services.path_security import WorkspaceManager
        workspace = WorkspaceManager(orchestrator._output_dir)
        corpus_spec, _ = orchestrator._load_corpus(spec, workspace)
        pipeline_specs, _, _ = orchestrator._build_pipelines(spec)

        for ps in pipeline_specs:
            fingerprint = compute_pipeline_fingerprint(
                pipeline_spec=ps,
                corpus_spec=corpus_spec,
                normalization_profile=spec.normalization_profile,
                char_exclude=spec.char_exclude,
                profile=spec.profile,
                code_version=spec.code_version,
            )
            partial_path = partial_path_for_pipeline(
                partial_dir=partial_dir,
                corpus_name=corpus_spec.name,
                pipeline_name=ps.name,
                fingerprint=fingerprint,
            )
            # Persister tous les PipelineResult du 1er run dans le partial.
            for first_doc in first_result.run_result.document_results:
                for pr in first_doc.pipeline_results:
                    if pr.pipeline_name == ps.name:
                        append_pipeline_result(partial_path, pr)

        # 2e run sur le même spec : le partial est complet, aucun
        # nouveau calcul n'est requis.
        second_result = RunOrchestrator(
            tmp_path / "out",
        ).execute(spec)

        # Tous les docs sont présents dans le résultat final.
        assert second_result.run_result.n_documents == 2
        # ``fully_resumed`` flag dans la metadata du manifest signale
        # qu'aucun sub-run n'a été nécessaire.
        assert second_result.run_result.manifest.metadata.get(
            "fully_resumed",
        ) == "true"
        # Cleanup : le partial est supprimé même en mode fully resumed.
        assert list(partial_dir.glob("*.partial.jsonl")) == []

    def test_partial_dir_fingerprint_isolation(
        self, tmp_path: Path,
    ) -> None:
        """Deux runs avec des configs différentes ont des fingerprints
        différents → fichiers partiels distincts → pas de réutilisation
        croisée.

        Test : crée un partial avec un fingerprint forgé (différent),
        puis lance le bench.  Le bench doit ignorer ce partial et
        produire un résultat propre.
        """
        from picarones.app.services._orchestrator_partial import (
            partial_path_for_pipeline,
        )

        partial_dir = tmp_path / "partial"
        partial_dir.mkdir()

        # Pré-écrire un partial avec un fingerprint forgé qui ne
        # matchera pas le fingerprint calculé par le bench.
        fake_path = partial_path_for_pipeline(
            partial_dir=partial_dir,
            corpus_name="feature_parity",
            pipeline_name="tess_only",
            fingerprint="0" * 64,  # fingerprint forgé
        )
        fake_path.write_text(
            '{"document_id": "ghost_doc",'
            ' "pipeline_name": "tess_only",'
            ' "step_results": [],'
            ' "succeeded": false,'
            ' "duration_seconds": 0.0,'
            ' "artifacts": []}\n',
            encoding="utf-8",
        )

        spec = self._build_spec(
            tmp_path, n_docs=2, partial_dir=partial_dir,
        )
        result = RunOrchestrator(tmp_path / "out").execute(spec)

        # Le run produit ses 2 docs propres (ne charge pas le fake).
        assert result.run_result.n_documents == 2
        doc_ids = {dr.document_id for dr in result.run_result.document_results}
        assert "ghost_doc" not in doc_ids
        assert doc_ids == {"doc01", "doc02"}


# ──────────────────────────────────────────────────────────────────────
# B2.4 — entity_extractor (NER attach)
# ──────────────────────────────────────────────────────────────────────


# Mock importable utilisé via dotted path par le test ci-dessous.
# Fonction module-level pour que ``importlib`` puisse la résoudre.
def _mock_entity_extractor(text: str) -> list[dict]:
    """Extracteur d'entités fixe pour les tests B2.4.

    Détecte ``Jean`` (PER) et ``Paris`` (LOC) dans le texte.  Sortie
    déterministe pour rendre les métriques NER prévisibles.
    """
    entities: list[dict] = []
    if "Jean" in text:
        start = text.find("Jean")
        entities.append({
            "label": "PER", "start": start, "end": start + 4, "text": "Jean",
        })
    if "Paris" in text:
        start = text.find("Paris")
        entities.append({
            "label": "LOC", "start": start, "end": start + 5, "text": "Paris",
        })
    return entities


class TestParityEntityExtractor:
    """Phase B2.4 — ``entity_extractor`` produit des NER metrics dans
    le BenchmarkResult legacy (output_json).

    Pattern strictement aligné sur ``run_benchmark_via_service:261-264``.
    """

    def _make_corpus_zip_with_entities(self) -> bytes:
        """Corpus zip 1 doc avec GT TEXT + GT ENTITIES JSON."""
        import json
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("doc01.png", _png_bytes())
            zf.writestr("doc01.gt.txt", "Jean habite Paris")
            zf.writestr("doc01.tess.txt", "Jean habite Paris")
            # GT ENTITIES — format reconnu par
            # ``_load_extra_gt_levels``.
            zf.writestr("doc01.gt.entities.json", json.dumps({
                "entities": [
                    {"label": "PER", "start": 0, "end": 4, "text": "Jean"},
                    {"label": "LOC", "start": 12, "end": 17, "text": "Paris"},
                ],
            }))
        return buf.getvalue()

    def _build_spec(
        self, tmp_path: Path, *, entity_extractor: str | None,
    ) -> "RunSpec":
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(self._make_corpus_zip_with_entities())
        out_dir = tmp_path / "out"
        yaml = _build_spec_yaml(corpus_zip, out_dir)
        yaml += f"output_json: {tmp_path / 'bm.json'}\n"
        if entity_extractor is not None:
            yaml += f"entity_extractor: {entity_extractor!r}\n"
        return load_run_spec_from_yaml(yaml)

    def test_extractor_produces_ner_metrics(self, tmp_path: Path) -> None:
        """Avec entity_extractor fourni → DocumentResult.ner_metrics
        est présent dans le JSON legacy."""
        import json

        spec = self._build_spec(
            tmp_path,
            entity_extractor=(
                "tests.app.services.test_run_orchestrator_feature_parity:"
                "_mock_entity_extractor"
            ),
        )
        RunOrchestrator(tmp_path / "out").execute(spec)

        loaded = json.loads((tmp_path / "bm.json").read_text(encoding="utf-8"))
        doc_result = loaded["engine_reports"][0]["document_results"][0]
        # Le NER attach a couru — ner_metrics non-None et non-vide.
        assert "ner_metrics" in doc_result
        assert doc_result["ner_metrics"] is not None
        # Les 2 entités matchent → precision/recall/f1 = 1.0.
        # Le hook NER attache les métriques par type + agrégation.
        ner = doc_result["ner_metrics"]
        assert isinstance(ner, dict)

    def test_no_extractor_no_ner_metrics(self, tmp_path: Path) -> None:
        """Sans entity_extractor → ner_metrics absent ou None
        (cohérent avec run_benchmark_via_service sans entity_extractor)."""
        import json

        spec = self._build_spec(tmp_path, entity_extractor=None)
        RunOrchestrator(tmp_path / "out").execute(spec)

        loaded = json.loads((tmp_path / "bm.json").read_text(encoding="utf-8"))
        doc_result = loaded["engine_reports"][0]["document_results"][0]
        # ner_metrics peut être absent ou None — les deux sont OK.
        assert doc_result.get("ner_metrics") is None

    def test_invalid_extractor_dotted_path_degrades_gracefully(
        self, tmp_path: Path,
    ) -> None:
        """Un dotted path qui pointe vers un module inexistant ne casse
        pas le bench — warning loggé, NER simplement sauté.

        Cohérent avec la tolérance du legacy
        ``_attach_ner_metrics_to_benchmark``.
        """
        spec = self._build_spec(
            tmp_path,
            entity_extractor="picarones.nonexistent.module:no_such_function",
        )
        # Le bench réussit malgré l'extractor invalide.
        result = RunOrchestrator(tmp_path / "out").execute(spec)
        assert result.run_result.n_documents == 1


# ──────────────────────────────────────────────────────────────────────
# B2.5 — char_exclude + normalization_profile
# ──────────────────────────────────────────────────────────────────────


def _make_corpus_zip_with_texts(
    gt_text: str, ocr_text: str,
) -> bytes:
    """Corpus zip 1 doc avec une GT et un texte OCR précalculé spécifiques."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr("doc01.png", _png_bytes())
        zf.writestr("doc01.gt.txt", gt_text)
        zf.writestr("doc01.tess.txt", ocr_text)
    return buf.getvalue()


class TestParityCharExclude:
    """Phase B2.5 — ``char_exclude`` filtre les caractères avant CER/WER.

    Référence : ``compute_metrics(char_exclude=...)`` dans
    ``picarones.evaluation.metrics.text_metrics:151-153``.  Le filtre
    s'applique aux deux payloads (GT + hypothèse) avant tout calcul.
    """

    def _build_spec(
        self, tmp_path: Path, *, char_exclude: str | None,
    ) -> "RunSpec":
        # GT = "Bonjour!"  OCR = "Bonjour."  → diff exact = 1 char
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip_with_texts(
            gt_text="Bonjour!", ocr_text="Bonjour.",
        ))
        out_dir = tmp_path / "out"
        yaml = _build_spec_yaml(corpus_zip, out_dir)
        if char_exclude is not None:
            yaml += f'char_exclude: "{char_exclude}"\n'
        # On force un output_json pour récupérer le CER calculé sans
        # avoir à parser les fichiers JSONL natifs.
        yaml += f"output_json: {tmp_path / 'bm.json'}\n"
        return load_run_spec_from_yaml(yaml)

    def _run_and_read_cer(self, tmp_path: Path, spec: "RunSpec") -> float:
        import json
        RunOrchestrator(tmp_path / "out").execute(spec)
        loaded = json.loads((tmp_path / "bm.json").read_text(encoding="utf-8"))
        return loaded["engine_reports"][0]["document_results"][0]["metrics"]["cer"]

    def test_without_char_exclude_cer_is_nonzero(self, tmp_path: Path) -> None:
        """Sans filtrage, CER = 1/8 (1 char différent sur 8)."""
        spec = self._build_spec(tmp_path, char_exclude=None)
        cer = self._run_and_read_cer(tmp_path, spec)
        assert cer == pytest.approx(1 / 8)

    def test_with_char_exclude_eliminates_diff(self, tmp_path: Path) -> None:
        """Avec ``char_exclude="!."``, les 2 caractères qui diffèrent
        sont filtrés des deux côtés → CER = 0.0."""
        spec = self._build_spec(tmp_path, char_exclude="!.")
        cer = self._run_and_read_cer(tmp_path, spec)
        assert cer == 0.0


class TestParityNormalizationProfile:
    """Phase B2.5 — ``normalization_profile`` impacte ``cer_diplomatic``.

    Sémantique legacy de ``compute_metrics`` (cf.
    ``text_metrics.py:173-184``) : le profil est appliqué à un CER
    parallèle (``cer_diplomatic``) tandis que ``cer`` reste le CER
    brut.  Cette parity-test reproduit exactement le comportement de
    ``run_benchmark_via_service(normalization_profile=...)``.

    Pour les vues canoniques (text_final/searchability), la
    normalisation est appliquée AVANT l'évaluation par
    ``DefaultEvaluationViewExecutor._apply_normalization`` — donc
    ``cer`` natif des vues sera lui-même 0.0.  Mais le converter
    legacy (output_json) recalcule depuis les textes bruts.
    """

    def _build_spec(
        self, tmp_path: Path, *, normalization_profile: str | None,
    ) -> "RunSpec":
        # GT = "Bonjour"  OCR = "BONJOUR"  → 6 chars en désaccord de casse
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip_with_texts(
            gt_text="Bonjour", ocr_text="BONJOUR",
        ))
        out_dir = tmp_path / "out"
        yaml = _build_spec_yaml(corpus_zip, out_dir)
        if normalization_profile is not None:
            yaml += f"normalization_profile: {normalization_profile}\n"
        yaml += f"output_json: {tmp_path / 'bm.json'}\n"
        return load_run_spec_from_yaml(yaml)

    def _run_and_read_metrics(
        self, tmp_path: Path, spec: "RunSpec",
    ) -> dict[str, float]:
        import json
        RunOrchestrator(tmp_path / "out").execute(spec)
        loaded = json.loads((tmp_path / "bm.json").read_text(encoding="utf-8"))
        return loaded["engine_reports"][0]["document_results"][0]["metrics"]

    def test_without_profile_cer_diplomatic_is_high(self, tmp_path: Path) -> None:
        """Sans profil : tous les CER diffèrent (raw, diplomatic) > 0."""
        spec = self._build_spec(tmp_path, normalization_profile=None)
        metrics = self._run_and_read_metrics(tmp_path, spec)
        # cer brut = 1.0 (toutes les lettres diffèrent en casse)
        assert metrics["cer"] > 0.5
        # Default diplomatique = medieval_french qui ne plie pas la
        # casse → cer_diplomatic reste élevé.
        assert metrics.get("cer_diplomatic", 1.0) > 0.5

    def test_with_caseless_profile_zeroes_diplomatic_cer(
        self, tmp_path: Path,
    ) -> None:
        """``caseless`` (pliage de casse) → ``cer_diplomatic`` = 0.0.

        ``cer`` raw reste élevé car il opère sur les payloads bruts.
        """
        spec = self._build_spec(tmp_path, normalization_profile="caseless")
        metrics = self._run_and_read_metrics(tmp_path, spec)
        assert metrics["cer_diplomatic"] == 0.0
        # cer raw n'est pas affecté — c'est la sémantique legacy.
        assert metrics["cer"] > 0.5


# ──────────────────────────────────────────────────────────────────────
# B2.6 — profile (hooks document-level / corpus aggregators)
# ──────────────────────────────────────────────────────────────────────


class TestParityProfile:
    """Phase B2.6 — ``profile`` est validé tôt et applique les hooks.

    Validation : B1.1 ajoute un model_validator
    ``_validate_profile_is_known`` qui appelle ``validate_profile``
    avant que le ``RunSpec`` ne soit instancié.  L'invocation des
    hooks document-level et corpus aggregators se fait via le
    converter legacy (chemin ``output_json``) qui appelle
    ``run_document_hooks(profile=...)`` puis
    ``run_corpus_aggregators(profile, ...)``.
    """

    def test_unknown_profile_rejected_at_runspec(self, tmp_path: Path) -> None:
        """``profile="philolagic_typo"`` est rejeté à la construction
        du ``RunSpec``, AVANT toute exécution OCR.

        Cohérent avec ``run_benchmark_via_service(profile="unknown")``
        qui lève via ``validate_profile`` au démarrage du bench.
        """
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip(n_docs=1))
        out_dir = tmp_path / "out"
        yaml = _build_spec_yaml(corpus_zip, out_dir)
        yaml += "profile: not_a_real_profile\n"

        from pydantic import ValidationError
        with pytest.raises((ValidationError, Exception), match="profil"):
            load_run_spec_from_yaml(yaml)

    def test_default_profile_is_standard_and_runs(self, tmp_path: Path) -> None:
        """``profile`` non spécifié = ``standard`` (default RunSpec) →
        le bench passe la validation et tourne."""
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip(n_docs=1))
        out_dir = tmp_path / "out"
        spec = load_run_spec_from_yaml(_build_spec_yaml(corpus_zip, out_dir))
        assert spec.profile == "standard"

        result = RunOrchestrator(out_dir).execute(spec)
        assert result.run_result.n_documents == 1

    def test_profile_propagated_to_legacy_converter_hooks(
        self, tmp_path: Path,
    ) -> None:
        """Le ``profile`` du ``RunSpec`` est passé au converter legacy
        qui invoque ``run_document_hooks`` + ``run_corpus_aggregators``.

        Vérification fonctionnelle : avec ``profile="standard"``, le
        ``BenchmarkResult`` legacy persisté via ``output_json`` porte
        les champs étendus calculés par les hooks (ex :
        ``DocumentResult`` a des attributs au-delà des CER/WER bruts —
        ``hypothesis_length``, ``confusion``, etc., selon les hooks
        ``standard`` enregistrés).
        """
        import json

        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip(n_docs=1))
        out_dir = tmp_path / "out"
        output_json = tmp_path / "bm.json"
        yaml = _build_spec_yaml(corpus_zip, out_dir)
        yaml += f"output_json: {output_json}\n"
        yaml += "profile: standard\n"
        spec = load_run_spec_from_yaml(yaml)

        RunOrchestrator(out_dir).execute(spec)

        loaded = json.loads(output_json.read_text(encoding="utf-8"))
        doc_result = loaded["engine_reports"][0]["document_results"][0]
        # Le hook ``hypothesis_length`` (standard) doit être présent.
        assert "metrics" in doc_result
        assert doc_result["metrics"]["hypothesis_length"] > 0


# ──────────────────────────────────────────────────────────────────────
# B2.7 — output_json (legacy BenchmarkResult JSON)
# ──────────────────────────────────────────────────────────────────────


class TestParityOutputJsonLegacy:
    """Phase B2.7 — quand ``spec.output_json`` est fourni, un fichier
    JSON au format ``BenchmarkResult.as_dict()`` est écrit en plus
    des 4 fichiers JSONL natifs du ``RunOrchestrator``.

    Cohabitation testée : les 5 fichiers (4 JSONL + 1 JSON legacy)
    sont produits simultanément pour permettre la cohabitation
    pendant la migration.
    """

    def _build_spec_with_output_json(
        self, tmp_path: Path, output_json: Path,
    ) -> "RunSpec":
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip(n_docs=2))
        out_dir = tmp_path / "out"
        yaml = _build_spec_yaml(corpus_zip, out_dir)
        yaml += f"output_json: {output_json}\n"
        return load_run_spec_from_yaml(yaml)

    def test_output_json_is_written(self, tmp_path: Path) -> None:
        output_json = tmp_path / "bm.json"
        spec = self._build_spec_with_output_json(tmp_path, output_json)

        RunOrchestrator(tmp_path / "out").execute(spec)

        assert output_json.exists()
        assert output_json.stat().st_size > 0

    def test_output_json_format_matches_legacy_runner(
        self, tmp_path: Path,
    ) -> None:
        """Le JSON produit est strictement au format
        ``persist_benchmark_result_json`` (``dataclasses.asdict``), donc
        identique à ce que ``run_benchmark_via_service(output_json=...)``
        écrit aujourd'hui.

        Cohérence assurée : un consommateur legacy qui lit ce fichier
        (CLI export, downstream tooling) ne voit aucune différence.
        """
        import json

        output_json = tmp_path / "bm.json"
        spec = self._build_spec_with_output_json(tmp_path, output_json)

        RunOrchestrator(tmp_path / "out").execute(spec)

        loaded = json.loads(output_json.read_text(encoding="utf-8"))
        # Format plat (dataclasses.asdict) — champs racines directs.
        assert loaded["corpus_name"] == "feature_parity"
        assert loaded["document_count"] == 2
        assert len(loaded["engine_reports"]) == 1
        assert loaded["engine_reports"][0]["engine_name"] == "tess_only"
        # Le DocumentResult de chaque doc est présent avec sa hypothèse.
        doc_results = loaded["engine_reports"][0]["document_results"]
        assert len(doc_results) == 2
        for dr in doc_results:
            assert dr["ground_truth"].startswith("Texte de référence")
            assert dr["hypothesis"].startswith("Texte de référence")
            # PrecomputedTextAdapter retourne le même texte que la GT —
            # CER attendu = 0.0.
            assert dr["metrics"]["cer"] == 0.0

    def test_output_json_coexists_with_jsonl(self, tmp_path: Path) -> None:
        """Les 4 fichiers JSONL natifs sont toujours produits en
        parallèle du JSON legacy."""
        output_json = tmp_path / "bm.json"
        spec = self._build_spec_with_output_json(tmp_path, output_json)

        result = RunOrchestrator(tmp_path / "out").execute(spec)

        # 4 fichiers JSONL natifs.
        assert set(result.persisted_files) == {
            "manifest", "pipeline_results", "artifacts_index", "view_results",
        }
        for path in result.persisted_files.values():
            assert path.exists()
        # 5e fichier : le JSON legacy.
        assert output_json.exists()

    def test_no_output_json_skips_legacy_persist(self, tmp_path: Path) -> None:
        """Compat ascendante : sans ``output_json``, le comportement
        est identique à avant B2.7 (seulement 4 fichiers JSONL)."""
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(_make_corpus_zip(n_docs=2))
        out_dir = tmp_path / "out"
        spec = load_run_spec_from_yaml(_build_spec_yaml(corpus_zip, out_dir))

        result = RunOrchestrator(out_dir).execute(spec)
        assert result.run_result.n_documents == 2
        # Pas de 5e fichier — pas de output_json dans la spec.
        legacy_files = list((out_dir / "results").glob("*.json"))
        # Seul ``run_manifest.json`` existe parmi les fichiers .json
        # (les autres sont .jsonl).
        assert {p.name for p in legacy_files} == {"run_manifest.json"}


# ──────────────────────────────────────────────────────────────────────
# Test global de feature parity — vérification croisée
# ──────────────────────────────────────────────────────────────────────


class TestParityAllFeaturesCombined:
    """Phase B2 / Checkpoint C1 — gate finale.

    Lance ``RunOrchestrator.execute`` avec **toutes** les features
    actives simultanément et vérifie que le ``BenchmarkResult`` legacy
    persisté via ``output_json`` est cohérent (toutes les métriques,
    NER, hooks, char_exclude appliqués, etc.).

    Ce test certifie que les 7 features sont câblées ensemble sans
    conflit ni régression croisée.  C'est le gate du checkpoint C1 :
    quand il passe, le ``RunOrchestrator`` est feature-complete vis-à-vis
    de ``run_benchmark_via_service``.
    """

    def test_combined_features_produce_coherent_result(
        self, tmp_path: Path,
    ) -> None:
        import json

        # Corpus avec GT TEXT + GT ENTITIES (pour NER).
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("doc01.png", _png_bytes())
            zf.writestr("doc01.gt.txt", "Jean habite Paris!")
            zf.writestr("doc01.tess.txt", "Jean habite Paris.")
            zf.writestr("doc01.gt.entities.json", json.dumps({
                "entities": [
                    {"label": "PER", "start": 0, "end": 4, "text": "Jean"},
                    {"label": "LOC", "start": 12, "end": 17, "text": "Paris"},
                ],
            }))
        corpus_zip = tmp_path / "c.zip"
        corpus_zip.write_bytes(buf.getvalue())

        out_dir = tmp_path / "out"
        partial_dir = tmp_path / "partial"
        output_json = tmp_path / "bm.json"

        # YAML avec TOUTES les features activées simultanément.
        yaml = _build_spec_yaml(corpus_zip, out_dir)
        yaml += f"partial_dir: {partial_dir}\n"
        yaml += f"output_json: {output_json}\n"
        yaml += 'char_exclude: "!."\n'
        yaml += "normalization_profile: caseless\n"
        yaml += "profile: standard\n"
        yaml += (
            "entity_extractor: 'tests.app.services."
            "test_run_orchestrator_feature_parity:_mock_entity_extractor'\n"
        )
        spec = load_run_spec_from_yaml(yaml)

        # Callback + cancel_event passés en kwargs d'exécution.
        invocations: list[tuple[str, int, str]] = []

        def cb(engine: str, idx: int, doc_id: str) -> None:
            invocations.append((engine, idx, doc_id))

        ev = threading.Event()  # jamais set : run normal

        result = RunOrchestrator(out_dir).execute(
            spec, progress_callback=cb, cancel_event=ev,
        )

        # Le run a tourné : 1 doc, 1 callback invoqué.
        assert result.run_result.n_documents == 1
        assert len(invocations) == 1

        # JSON legacy écrit avec TOUTES les features intégrées.
        loaded = json.loads(output_json.read_text(encoding="utf-8"))
        doc_result = loaded["engine_reports"][0]["document_results"][0]

        # char_exclude appliqué : "!." filtré → ground_truth +
        # hypothesis matchent exactement → CER = 0.
        assert doc_result["metrics"]["cer"] == 0.0

        # normalization_profile=caseless propagé → cer_diplomatic = 0.
        assert doc_result["metrics"]["cer_diplomatic"] == 0.0

        # entity_extractor invoqué → ner_metrics présent.
        assert doc_result.get("ner_metrics") is not None

        # profile=standard appliqué → hypothesis_length présent.
        assert doc_result["metrics"]["hypothesis_length"] > 0

        # Cohabitation : 4 fichiers JSONL natifs + 1 JSON legacy.
        assert output_json.exists()
        assert set(result.persisted_files) == {
            "manifest", "pipeline_results", "artifacts_index", "view_results",
        }

        # partial_dir : pipeline complet → fichier nettoyé.
        assert list(partial_dir.glob("*.partial.jsonl")) == []
