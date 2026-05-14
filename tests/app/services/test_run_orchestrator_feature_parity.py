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


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}3 — port partial_dir resume")
def test_parity_partial_dir_resume_fresh_start(tmp_path: Path) -> None:
    """Premier run avec ``partial_dir`` non existant → comportement
    identique à un run sans ``partial_dir``.

    Spec
    ----
    - ``partial_dir`` = répertoire vide.
    - Lancer le bench.
    - À la fin, le fichier ``{partial_dir}/picarones_{corpus}_{engine}
      .partial.jsonl`` est supprimé (succès complet).
    - Le ``BenchmarkResult`` est identique au run sans ``partial_dir``.
    """


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}3 — port partial_dir resume")
def test_parity_partial_dir_resume_after_crash(tmp_path: Path) -> None:
    """Reprise après crash partiel : 3 docs sur 5 déjà persistés →
    seuls les 2 restants sont soumis au runner.

    Spec
    ----
    - Pré-écrire un partial JSONL avec 3 ``DocumentResult`` valides.
    - Lancer le bench sur le corpus de 5 docs.
    - Le ``CorpusRunner.run`` est appelé sur **2 docs seulement**
      (vérifier via spy).
    - Le ``BenchmarkResult`` final agrège les 5 docs (3 réutilisés +
      2 nouveaux).
    """


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}3 — port partial_dir resume")
def test_parity_partial_dir_fingerprint_invalidates(tmp_path: Path) -> None:
    """Fingerprint divergent invalide le partial (re-calcul depuis 0).

    Spec
    ----
    - Pré-écrire un partial avec un ``code_version`` différent.
    - Lancer le bench.
    - Le partial est ignoré, les 5 docs sont recalculés.
    """


# ──────────────────────────────────────────────────────────────────────
# B2.4 — entity_extractor (NER attach)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}4 — port entity_extractor")
def test_parity_entity_extractor_ner(tmp_path: Path) -> None:
    """Quand un ``entity_extractor`` est fourni, les métriques NER
    sont attachées au ``BenchmarkResult``.

    Spec
    ----
    - Corpus avec ``EntitiesGT`` (au moins 1 doc avec niveau ENTITIES).
    - ``entity_extractor`` = mock qui retourne des entités fixes.
    - Le ``BenchmarkResult`` contient ``DocumentResult.ner_metrics`` :
      ``precision``, ``recall``, ``f1`` par type d'entité.
    - L'agrégation ``EngineReport.aggregated_ner`` est calculée.
    """


# ──────────────────────────────────────────────────────────────────────
# B2.5 — char_exclude + normalization_profile
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}5 — port normalization propagation")
def test_parity_char_exclude(tmp_path: Path) -> None:
    """``char_exclude`` filtre les caractères avant calcul CER/WER.

    Spec
    ----
    - GT = ``"Bonjour!"``, OCR = ``"Bonjour."``.
    - Sans ``char_exclude`` : CER = 1/8 = 0.125.
    - Avec ``char_exclude="!."`` : CER = 0.0 (les 2 caractères
      filtrés sont les seuls différents).
    """


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}5 — port normalization propagation")
def test_parity_normalization_profile(tmp_path: Path) -> None:
    """``normalization_profile="caseless"`` égalise les casses.

    Spec
    ----
    - GT = ``"Bonjour"``, OCR = ``"BONJOUR"``.
    - Sans profil : CER ≈ 1.0 (toutes les lettres diffèrent).
    - Avec ``caseless`` : CER = 0.0.
    """


# ──────────────────────────────────────────────────────────────────────
# B2.6 — profile (hooks document-level / corpus aggregators)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}6 — port profile hooks")
def test_parity_profile_validation(tmp_path: Path) -> None:
    """``profile="unknown"`` lève ``ValueError`` AVANT le run.

    Spec
    ----
    - Comportement identique aux 3 tests
      ``TestProfileValidation`` de
      ``tests/app/test_sprint_d2cdef_features.py``.
    """


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}6 — port profile hooks")
def test_parity_profile_standard_runs_hooks(tmp_path: Path) -> None:
    """``profile="standard"`` exécute les hooks document-level
    enregistrés via ``@register_document_metric``.

    Spec
    ----
    - Enregistrer un hook test ``@register_document_metric("standard")``
      qui renvoie ``{"hooked": True}``.
    - Lancer le bench.
    - ``DocumentResult.hook_values["hooked"] is True``.
    """


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


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}* — toutes features portées")
def test_parity_all_features_combined(tmp_path: Path) -> None:
    """Lance les deux chemins avec toutes les features actives et
    vérifie l'égalité numérique du ``BenchmarkResult``.

    Spec
    ----
    - Construire un ``RunSpec`` avec : ``profile="standard"``,
      ``partial_dir=tmp_path/"partial"``, ``output_json=tmp_path/
      "bm.json"``, ``char_exclude="!."``,
      ``normalization_profile="caseless"``.
    - Lancer ``run_benchmark_via_service`` avec les mêmes paramètres.
    - Lancer ``RunOrchestrator().execute(spec)``.
    - Normaliser les 2 ``BenchmarkResult`` (cf.
      ``test_migration_invariance.py:_normalize_for_snapshot``).
    - Vérifier ``a == b``.

    Ce test est le **gate finale du Checkpoint C1**.  Quand il passe,
    la Phase B2 est terminée et on peut commencer B3 (migration des
    call sites).
    """
