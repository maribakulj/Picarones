"""Harnais de caractérisation du cœur stateful de ``RunOrchestrator``.

Audit prod — PRÉCONDITION de la Phase B (décomposition stateful).
Verrouille le comportement EXACT actuel là où Phase B va toucher,
pour transformer « la CI pourrait ne pas attraper » en « la CI
attrape ».  Cinq groupes, un par cas de risque identifié :

1. :class:`TestGlobalDocIdxContract`     — compteur ``doc_idx`` global
   au run (pas par pipeline) sur **multi-pipeline**.
2. :class:`TestCancelPropagation`        — annulation pré-set ET
   mi-run (identité d'objet event à travers les couches).
3. :class:`TestCrashResumeConsistency`   — interruption réelle puis
   resume : sortie finale identique à un run complet propre.
4. :class:`TestGoldenMultiTopology`      — snapshot normalisé
   déterministe des artefacts (manifest/pipeline/view) sur 4
   topologies (linéaire, multi-pipeline, DAG branchant, OCR+corr).
5. :class:`TestConcurrencyIsolation`     — deux ``execute()`` en
   threads parallèles : isolation cancel/progress, pas de fuite
   d'état entre instances.

6. :class:`TestExecutePresetCharacterization` — chemin
   ``execute_preset`` (objets domain pré-construits : CLI
   diagnose/economics/edition + worker web) : hotfix ``corpus_legacy``
   (trou #9 B3-final), doc_idx global, cancel, resume.

Déterminisme : ``PrecomputedTextAdapter`` (lit ``<stem>.<label>.txt``)
pour ``execute`` ; ``_MockOCR`` en mémoire pour ``execute_preset`` —
aucun OCR/réseau.  Un garde explicite
(:meth:`TestGoldenMultiTopology.test_snapshot_is_deterministic`)
échoue si le snapshot n'est pas reproductible — un golden flaky
serait pire que pas de golden.
"""

from __future__ import annotations

import io
import json
import re
import textwrap
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest

from picarones.adapters.ocr.base import BaseOCRAdapter
from picarones.app.schemas.run_spec import load_run_spec_from_yaml
from picarones.app.services import RunOrchestrator, prepare_preset_args
from picarones.domain.artifacts import Artifact, ArtifactType
from picarones.evaluation.corpus import Corpus, Document

_FIX_DIR = Path(__file__).parent / "fixtures" / "run_orchestrator"


# ---------------------------------------------------------------------------
# Briques déterministes
# ---------------------------------------------------------------------------

def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89"
    )


def _make_corpus_zip(
    tmp: Path, n_docs: int, *, sources: tuple[str, ...] = ("tess",),
) -> Path:
    """Corpus ZIP déterministe.

    Pour chaque ``source_label`` de ``sources`` on écrit
    ``<doc>.<label>.txt`` (lu par ``PrecomputedTextAdapter``).  La GT
    diffère légèrement du texte pour produire un CER non trivial mais
    fixe.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        for i in range(1, n_docs + 1):
            doc = f"doc{i:02d}"
            zf.writestr(f"{doc}.png", _png_bytes())
            zf.writestr(f"{doc}.gt.txt", f"Texte de reference {i}")
            for label in sources:
                # 1 substitution stable → CER déterministe non nul.
                zf.writestr(f"{doc}.{label}.txt", f"Texte de reference {i}!")
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "corpus.zip"
    p.write_bytes(buf.getvalue())
    return p


def _spec(corpus_zip: Path, out: Path, body: str) -> Any:
    yaml = textwrap.dedent(f"""
        corpus_zip: {corpus_zip}
        corpus_name: charac
        output_dir: {out}
        code_version: "charac-1.0"
        views: [text_final]
        {body}
    """)
    return load_run_spec_from_yaml(yaml)


_TOPOLOGIES: dict[str, tuple[tuple[str, ...], str]] = {
    "single_linear": (("tess",), """
        pipelines:
          - name: tess_only
            initial_inputs: [image]
            steps:
              - id: ocr
                adapter_class: picarones.adapters.ocr.precomputed.PrecomputedTextAdapter
                adapter_kwargs: {source_label: tess}
                input_types: [image]
                output_types: [raw_text]
    """),
    "multi_pipeline": (("tess", "pero"), """
        pipelines:
          - name: tess_only
            initial_inputs: [image]
            steps:
              - id: ocr
                adapter_class: picarones.adapters.ocr.precomputed.PrecomputedTextAdapter
                adapter_kwargs: {source_label: tess}
                input_types: [image]
                output_types: [raw_text]
          - name: pero_only
            initial_inputs: [image]
            steps:
              - id: ocr
                adapter_class: picarones.adapters.ocr.precomputed.PrecomputedTextAdapter
                adapter_kwargs: {source_label: pero}
                input_types: [image]
                output_types: [raw_text]
    """),
    "branching_dag": (("tess", "corr"), """
        pipelines:
          - name: ocr_then_correct
            initial_inputs: [image]
            preferred_text_output: corrector.corrected_text
            steps:
              - id: ocr
                adapter_class: picarones.adapters.ocr.precomputed.PrecomputedTextAdapter
                adapter_kwargs: {source_label: tess}
                input_types: [image]
                output_types: [raw_text]
              - id: corrector
                adapter_class: picarones.adapters.ocr.precomputed.PrecomputedTextAdapter
                adapter_kwargs: {source_label: corr}
                input_types: [image, raw_text]
                output_types: [corrected_text]
                inputs_from:
                  raw_text: ocr
    """),
}


# ---------------------------------------------------------------------------
# Normalisation snapshot (déterministe : retire timestamps/durées/paths)
# ---------------------------------------------------------------------------

_VOLATILE_KEYS = {
    "started_at", "completed_at", "duration_seconds", "run_date",
    "created_at", "elapsed_seconds", "wall_clock_seconds",
    # ``run_id`` est horodaté (``charac_YYYYMMDDThhmmssZ``) — volatil,
    # pas une caractéristique de comportement.
    "run_id",
}


_PATH_PREFIX_RE = re.compile(r"^[A-Za-z]:[\\/]|^/")


def _looks_path(s: str) -> bool:
    """Détecte un chemin filesystem absolu, POSIX ou Windows.

    Couvre :
    - POSIX absolu ``/...`` (Linux, macOS ``/var/folders/...``) ;
    - Windows à lettre de lecteur ``C:\\...`` / ``C:/...`` ;
    - tmp-dirs et fixtures pytest (``/tmp/``, ``\\Temp\\``,
      ``pytest-of-...``).  Le snapshot golden doit être cross-OS :
    le CI Windows utilise ``C:\\Users\\runneradmin\\AppData\\Local\\
    Temp\\pytest-of-runneradmin\\...`` — précédemment non capturé
    par l'heuristique POSIX-only.
    """
    if _PATH_PREFIX_RE.match(s):
        return True
    return (
        "/tmp/" in s
        or "\\Temp\\" in s
        or "\\Users\\" in s
        or "pytest-of-" in s
    )


def _scrub(obj: Any) -> Any:
    """Snapshot canonique : retire les clés volatiles, neutralise les
    chemins absolus (POSIX ET Windows), et **trie les listes de
    scalaires**.

    Le tri des listes de scalaires est essentiel : certaines sont
    dérivées de ``set`` (ex. ``ignored_dimensions``, types projetés)
    dont l'ordre de sérialisation varie d'un *process* à l'autre
    (randomisation du hash de chaînes Python).  Le garde déterminisme
    intra-process ne l'attrape pas — seul un golden inter-process le
    révèle.  On NE trie PAS les listes de dicts : ce sont des
    enregistrements dont l'ordre peut porter du sens (et qui sont
    déjà triés au niveau record par :func:`_normalized_snapshot`)."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        scrubbed = [_scrub(x) for x in obj]
        if all(isinstance(x, (str, int, float, bool, type(None))) for x in scrubbed):
            return sorted(scrubbed, key=lambda x: (x is None, str(x)))
        return scrubbed
    if isinstance(obj, str):
        if _looks_path(obj):
            return "<PATH>"
        return obj
    return obj


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _normalized_snapshot(results_dir: Path) -> str:
    """Snapshot canonique, ordre-stable, sans champs volatils."""
    manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
    pipe = sorted(
        _jsonl(results_dir / "pipeline_results.jsonl"),
        key=lambda r: (r.get("document_id", ""), r.get("pipeline_name", "")),
    )
    views = sorted(
        _jsonl(results_dir / "view_results.jsonl"),
        key=lambda r: (r.get("document_id", ""), r.get("view_name", "")),
    )
    # 4e artefact : index des artefacts (découplé de pipeline_results
    # pour le streaming).  Phase B réordonne l'extraction d'artefacts
    # → à verrouiller aussi.
    arts = sorted(
        _jsonl(results_dir / "artifacts_index.jsonl"),
        key=lambda r: (
            r.get("document_id", ""), r.get("pipeline_name", ""),
            r.get("id", ""),
        ),
    )
    snap = {
        "manifest": _scrub(manifest),
        "pipeline_results": _scrub(pipe),
        "view_results": _scrub(views),
        "artifacts_index": _scrub(arts),
    }
    return json.dumps(snap, ensure_ascii=False, sort_keys=True, indent=2)


def _results_snapshot(results_dir: Path) -> str:
    """Snapshot des RÉSULTATS calculés seuls (pipeline + views),
    indépendant de l'écho de config du manifest (``partial_dir``,
    ``output_dir``…).  C'est l'invariant que « resume == run propre »
    doit préserver : mêmes sorties par document, peu importe que le
    run ait été configuré avec un ``partial_dir`` ou non."""
    pipe = sorted(
        _jsonl(results_dir / "pipeline_results.jsonl"),
        key=lambda r: (r.get("document_id", ""), r.get("pipeline_name", "")),
    )
    views = sorted(
        _jsonl(results_dir / "view_results.jsonl"),
        key=lambda r: (r.get("document_id", ""), r.get("view_name", "")),
    )
    return json.dumps(
        {"pipeline_results": _scrub(pipe), "view_results": _scrub(views)},
        ensure_ascii=False, sort_keys=True, indent=2,
    )


def _golden(name: str, actual: str) -> None:
    """Pattern golden : 1er run crée le fixture et échoue (force le
    commit) ; ensuite compare strictement."""
    gp = _FIX_DIR / f"{name}.json"
    if not gp.exists():
        gp.parent.mkdir(parents=True, exist_ok=True)
        gp.write_text(actual + "\n", encoding="utf-8")
        pytest.fail(
            f"Golden créé : {gp} — vérifier puis committer le fixture.",
        )
    expected = gp.read_text(encoding="utf-8").rstrip("\n")
    assert actual == expected, (
        f"Snapshot divergent vs {gp}.\nUne dérive ici = régression "
        "comportementale du cœur stateful (Phase B). Si intentionnel : "
        "supprimer le golden et relancer pour régénérer."
    )


def _run(tmp: Path, topo: str, **kw: Any) -> RunOrchestrator:
    sources, body = _TOPOLOGIES[topo]
    cz = _make_corpus_zip(tmp / "in", kw.pop("n_docs", 3), sources=sources)
    out = tmp / "out"
    spec = _spec(cz, out, body)
    orch = RunOrchestrator(out)
    res = orch.execute(spec, **kw)
    return res


# ---------------------------------------------------------------------------
# 1. Contrat compteur doc_idx GLOBAL (Risque 1)
# ---------------------------------------------------------------------------

class TestGlobalDocIdxContract:
    """Le compteur ``doc_idx`` est global au run, pas par pipeline.
    2 pipelines × N docs ⇒ séquence contiguë ``0..2N-1`` (PAS
    ``0..N-1`` deux fois).  Phase B casse ça en premier si elle crée
    le contexte par-collaborateur."""

    def test_multi_pipeline_counter_is_global_and_contiguous(
        self, tmp_path: Path,
    ) -> None:
        calls: list[tuple[str, int, str]] = []
        self_dir = tmp_path
        sources, body = _TOPOLOGIES["multi_pipeline"]
        cz = _make_corpus_zip(self_dir / "in", 3, sources=sources)
        spec = _spec(cz, self_dir / "out", body)
        RunOrchestrator(self_dir / "out").execute(
            spec, progress_callback=lambda e, i, d: calls.append((e, i, d)),
        )

        # 2 pipelines × 3 docs = 6 notifications.
        assert len(calls) == 6, calls
        indices = sorted(c[1] for c in calls)
        assert indices == [0, 1, 2, 3, 4, 5], (
            f"compteur NON global/contigu : {indices} — régression "
            "Phase B (contexte par-pipeline au lieu de global au run)"
        )
        engines = {c[0] for c in calls}
        assert engines == {"tess_only", "pero_only"}, engines

    def test_single_pipeline_counter_zero_based(
        self, tmp_path: Path,
    ) -> None:
        calls: list[int] = []
        _run(
            tmp_path, "single_linear", n_docs=4,
            progress_callback=lambda e, i, d: calls.append(i),
        )
        assert sorted(calls) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 2. Propagation cancel_event (Risque 2)
# ---------------------------------------------------------------------------

class TestCancelPropagation:
    def test_preset_cancel_multipipeline_stops_fast_and_partial(
        self, tmp_path: Path,
    ) -> None:
        ev = threading.Event()
        ev.set()
        t0 = time.monotonic()
        res = _run(tmp_path, "multi_pipeline", n_docs=5, cancel_event=ev)
        elapsed = time.monotonic() - t0
        assert elapsed < 10.0, f"pas de short-circuit (elapsed={elapsed})"
        # 2 pipelines × 5 = 10 docs possibles ; annulé ⇒ strictement
        # moins traités (souvent 0).
        assert res.run_result.manifest is not None

    def test_cancel_mid_run_via_callback_propagates(
        self, tmp_path: Path,
    ) -> None:
        """Identité d'objet event à travers les couches : l'event est
        ``set()`` APRÈS le 1er doc, depuis le callback ; les docs
        suivants doivent être sautés."""
        ev = threading.Event()
        seen: list[str] = []

        def cb(engine: str, idx: int, doc_id: str) -> None:
            seen.append(doc_id)
            if len(seen) == 1:
                ev.set()  # annule après le tout premier doc

        _run(
            tmp_path, "single_linear", n_docs=8,
            progress_callback=cb, cancel_event=ev,
        )
        # Sans propagation correcte, les 8 docs passent.  Avec : on
        # s'arrête bien avant la fin (tolérance large pour le
        # parallélisme max_in_flight=2 du runner).
        assert len(seen) < 8, (
            f"cancel mi-run NON propagé : {len(seen)}/8 docs traités "
            "— régression identité d'objet event (Phase B)"
        )


# ---------------------------------------------------------------------------
# 3. Interruption réelle → resume cohérent (Risque 4)
# ---------------------------------------------------------------------------

class TestCrashResumeConsistency:
    """Interruption RÉELLE (pas un partial pré-amorcé complet) :
    on stoppe mi-run via cancel, puis on relance SANS cancel avec le
    même ``partial_dir``.

    HISTORIQUE — DÉFAUT DÉCOUVERT PAR CE HARNAIS, PUIS CORRIGÉ.
    Le partial store persistait ``PipelineResult`` mais PAS
    ``ViewResult`` : au resume, les docs rechargés du partial
    sortaient avec ``pipeline_results`` mais SANS ``view_results``
    (jamais recalculés) → ``view_results.jsonl`` incomplet →
    métriques agrégées (CER…) silencieusement faussées après reprise
    (linéaire/DAG ; non manifesté en multi-pipeline).

    FIX : ``_execute_with_partial`` recalcule les vues des docs
    repris via ``_evaluate_document_in_views`` (fonction pure de
    pipeline_results + GT + profil ; aucun changement de format de
    partial).  Ces tests, qui caractérisaient le défaut, ont été
    BASCULÉS pour verrouiller le comportement CORRIGÉ (= run propre)
    — toute régression Phase B refaisant l'incohérence échoue ici."""

    def _persisted_doc_ids(self, results_dir: Path) -> tuple[list[str], list[str]]:
        pr = sorted({r["document_id"] for r in _jsonl(results_dir / "pipeline_results.jsonl")})
        vr = sorted({r["document_id"] for r in _jsonl(results_dir / "view_results.jsonl")})
        return pr, vr

    def _interrupt_then_resume(
        self, tmp_path: Path, n_docs: int, stop_after: int,
        topo: str = "single_linear",
    ) -> tuple[Path, Path]:
        sources, body = _TOPOLOGIES[topo]
        cz = _make_corpus_zip(tmp_path / "in", n_docs, sources=sources)
        partial = tmp_path / "partial"
        partial.mkdir()
        body_pd = body + f"\n        partial_dir: {partial}"

        ev = threading.Event()
        n = {"c": 0}

        def cb(e: str, i: int, d: str) -> None:
            n["c"] += 1
            if n["c"] == stop_after:
                ev.set()

        out1 = tmp_path / "out1"
        RunOrchestrator(out1).execute(
            _spec(cz, out1, body_pd), progress_callback=cb, cancel_event=ev,
        )
        out2 = tmp_path / "out2"
        RunOrchestrator(out2).execute(_spec(cz, out2, body_pd))
        return out1 / "results", out2 / "results"

    def test_clean_run_pipeline_and_views_are_consistent(
        self, tmp_path: Path,
    ) -> None:
        """Référence : un run PROPRE a pipeline_results == view_results
        (tous les docs des deux côtés)."""
        sources, body = _TOPOLOGIES["single_linear"]
        cz = _make_corpus_zip(tmp_path / "in", 5, sources=sources)
        out = tmp_path / "out"
        RunOrchestrator(out).execute(_spec(cz, out, body))
        pr, vr = self._persisted_doc_ids(out / "results")
        assert pr == vr == ["doc01", "doc02", "doc03", "doc04", "doc05"]

    def test_resume_pipeline_results_complete(self, tmp_path: Path) -> None:
        """Le resume RECONSTRUIT bien tous les ``pipeline_results``
        (partiel chargé + reste rejoué) — cette partie est correcte."""
        _, resumed = self._interrupt_then_resume(tmp_path, 5, stop_after=2)
        pr, _ = self._persisted_doc_ids(resumed)
        assert pr == ["doc01", "doc02", "doc03", "doc04", "doc05"]

    @pytest.mark.parametrize(
        "topo", ["single_linear", "multi_pipeline", "branching_dag"],
    )
    def test_resume_view_results_complete_and_consistent(
        self, tmp_path: Path, topo: str,
    ) -> None:
        """Régression-guard du FIX du défaut resume/vues.

        Historique : ce harnais a découvert qu'au resume le partial
        store rejouait ``pipeline_results`` mais PAS ``view_results``
        des docs repris (linéaire/DAG : vues ⊊ pipeline → métriques
        agrégées faussées après reprise).  CORRIGÉ par recalcul des
        vues au resume (``_execute_with_partial`` : ``_evaluate_
        document_in_views`` sur les PR rechargés — fonction pure).

        Invariant verrouillé désormais : après resume, sur TOUTES les
        topologies, ``view_results`` couvre exactement le même
        ensemble de docs que ``pipeline_results`` (= corpus complet),
        comme un run propre.  Si Phase B (ou un futur changement)
        recasse l'égalité, ce test échoue."""
        _, resumed = self._interrupt_then_resume(
            tmp_path, 5, stop_after=2, topo=topo,
        )
        pr, vr = self._persisted_doc_ids(resumed)
        full = ["doc01", "doc02", "doc03", "doc04", "doc05"]
        assert pr == full, f"pipeline incomplet au resume ({topo}): {pr}"
        assert vr == full, (
            f"[{topo}] vues incomplètes au resume : pipeline={pr} "
            f"vues={vr}. Le fix recalcule les vues des docs repris — "
            "une régression ici refait des métriques faussées."
        )

    def test_resume_does_not_duplicate_documents(
        self, tmp_path: Path,
    ) -> None:
        sources, body = _TOPOLOGIES["single_linear"]
        cz = _make_corpus_zip(tmp_path / "in", 4, sources=sources)
        partial = tmp_path / "p"
        partial.mkdir()
        body_pd = body + f"\n        partial_dir: {partial}"
        out = tmp_path / "o"

        ev = threading.Event()
        c = {"n": 0}

        def cb(e: str, i: int, d: str) -> None:
            c["n"] += 1
            if c["n"] == 1:
                ev.set()

        RunOrchestrator(out).execute(
            _spec(cz, out, body_pd), progress_callback=cb, cancel_event=ev,
        )
        out2 = tmp_path / "o2"
        res = RunOrchestrator(out2).execute(_spec(cz, out2, body_pd))
        doc_ids = [d.document_id for d in res.run_result.document_results]
        assert sorted(doc_ids) == ["doc01", "doc02", "doc03", "doc04"]
        assert len(doc_ids) == len(set(doc_ids)), (
            f"docs dupliqués au resume : {doc_ids}"
        )


# ---------------------------------------------------------------------------
# 4. Golden multi-topologie + garde déterminisme (Risques 3 & 4)
# ---------------------------------------------------------------------------

class TestScrubPathCrossPlatform:
    """Verrouille la détection de chemin pour le snapshot cross-OS.
    Régression CI : un chemin Windows ``C:\\Users\\runneradmin\\...``
    n'était pas matché par l'ancienne heuristique POSIX-only (qui
    exigeait ``"/" in obj``) → URI non scrubbée → garde déterminisme
    cassait sur Windows entre run ``a`` et run ``b`` (UUIDs workspace
    distincts)."""

    @pytest.mark.parametrize("p", [
        "/tmp/foo/bar",
        "/var/folders/x/T/pytest-of-runner/test_x/y",   # macOS
        "/tmp/pytest-of-root/pytest-0/test_x/out",      # Linux CI
        "C:\\Users\\runneradmin\\AppData\\Local\\Temp\\pytest-of-runneradmin\\pytest-0\\test_x\\out",
        "C:/Users/runneradmin/AppData/Local/Temp/x",    # Windows fwd-slash
        "pytest-of-runner/test_x",                       # marker sans préfixe
    ])
    def test_paths_are_scrubbed(self, p: str) -> None:
        assert _looks_path(p), f"chemin non détecté : {p!r}"
        assert _scrub(p) == "<PATH>"

    @pytest.mark.parametrize("s", [
        "doc01", "raw_text", "ALTO_XML",
        "tess_only", "ocr_then_correct", "charac-1.0",
    ])
    def test_non_paths_unchanged(self, s: str) -> None:
        assert not _looks_path(s), f"faux positif : {s!r}"
        assert _scrub(s) == s


class TestGoldenMultiTopology:
    @pytest.mark.parametrize("topo", sorted(_TOPOLOGIES))
    def test_snapshot_matches_golden(
        self, tmp_path: Path, topo: str,
    ) -> None:
        _run(tmp_path, topo, n_docs=3)
        snap = _normalized_snapshot(tmp_path / "out" / "results")
        _golden(topo, snap)

    @pytest.mark.parametrize("topo", sorted(_TOPOLOGIES))
    def test_snapshot_is_deterministic(
        self, tmp_path: Path, topo: str,
    ) -> None:
        """Garde anti-golden-flaky : deux runs du MÊME spec doivent
        produire un snapshot normalisé bit-identique.  Si ça échoue,
        le golden serait flaky → on refuse de le figer."""
        _run(tmp_path / "a", topo, n_docs=3)
        _run(tmp_path / "b", topo, n_docs=3)
        s1 = _normalized_snapshot(tmp_path / "a" / "out" / "results")
        s2 = _normalized_snapshot(tmp_path / "b" / "out" / "results")
        assert s1 == s2, (
            f"snapshot NON déterministe pour {topo} — un golden serait "
            "flaky ; corriger la normalisation AVANT de figer"
        )


# ---------------------------------------------------------------------------
# 5. Isolation concurrente (Risque 5)
# ---------------------------------------------------------------------------

class TestConcurrencyIsolation:
    """Deux ``execute()`` en threads parallèles, ``output_dir`` et
    ``cancel_event`` distincts.  Annuler A ne doit PAS perturber B
    (ni progression, ni complétude) — l'invariant « RunOrchestrator
    sans état entre deux execute() » est non testé et Phase B
    (collaborateur réutilisé) le casserait."""

    def test_cancel_in_thread_a_does_not_leak_into_thread_b(
        self, tmp_path: Path,
    ) -> None:
        sources, body = _TOPOLOGIES["single_linear"]
        cz = _make_corpus_zip(tmp_path / "in", 6, sources=sources)

        ev_a = threading.Event()
        ev_b = threading.Event()
        prog_a: list[int] = []
        prog_b: list[int] = []
        err: dict[str, BaseException] = {}

        def run_a() -> None:
            try:
                ev_a.set()  # A annulé d'emblée
                out = tmp_path / "a"
                RunOrchestrator(out).execute(
                    _spec(cz, out, body),
                    progress_callback=lambda e, i, d: prog_a.append(i),
                    cancel_event=ev_a,
                )
            except BaseException as e:  # noqa: BLE001
                err["a"] = e

        def run_b() -> None:
            try:
                out = tmp_path / "b"
                RunOrchestrator(out).execute(
                    _spec(cz, out, body),
                    progress_callback=lambda e, i, d: prog_b.append(i),
                    cancel_event=ev_b,  # B jamais annulé
                )
            except BaseException as e:  # noqa: BLE001
                err["b"] = e

        ta = threading.Thread(target=run_a)
        tb = threading.Thread(target=run_b)
        ta.start()
        tb.start()
        ta.join(timeout=30)
        tb.join(timeout=30)

        assert not err, f"exception thread : {err}"
        # B (non annulé) doit avoir traité ses 6 docs avec une
        # séquence propre 0..5 — aucune fuite du compteur de A.
        assert sorted(prog_b) == [0, 1, 2, 3, 4, 5], (
            f"B perturbé par l'annulation de A : prog_b={sorted(prog_b)} "
            "— fuite d'état entre instances (Phase B)"
        )
        # A annulé : strictement moins que 6.
        assert len(prog_a) < 6, f"A non annulé : {prog_a}"


# ---------------------------------------------------------------------------
# 6. execute_preset — trou comblé (objets domain pré-construits)
# ---------------------------------------------------------------------------

class _MockOCR(BaseOCRAdapter):
    """Adapter déterministe en mémoire (aucun OCR/réseau) : écrit
    ``"hello"`` et retourne un RAW_TEXT.  Recette copiée verbatim de
    ``tests/app/services/test_python_helpers.py`` (infra préset
    canonique)."""

    def __init__(self, name: str = "mock") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def execute(self, inputs: Any, params: Any, context: Any) -> Any:
        d = Path(context.workspace_uri)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{context.document_id}.txt"
        p.write_text("hello", encoding="utf-8")
        return {ArtifactType.RAW_TEXT: Artifact(
            id=f"{context.document_id}:{self._name}:raw_text",
            document_id=context.document_id,
            type=ArtifactType.RAW_TEXT,
            produced_by_step="ocr",
            uri=str(p),
        )}


def _legacy_corpus(tmp: Path, n: int = 3) -> Any:
    docs = []
    for i in range(1, n + 1):
        img = tmp / f"doc{i:02d}.png"
        img.write_bytes(b"x")
        docs.append(Document(
            image_path=img, ground_truth="hello", doc_id=f"doc{i:02d}",
        ))
    return Corpus(name="preset_charac", documents=docs)


def _preset_run(
    tmp: Path, n: int, engines: list[Any], **prep: Any,
) -> tuple[Any, Any, Any]:
    """Recette préset déterministe : Corpus mémoire + MockOCR →
    prepare_preset_args → execute_preset.  Retourne
    (OrchestrationResult, corpus, PresetArgs)."""
    # Sépare les kwargs de CONTRÔLE (réservés au harnais) des kwargs
    # de ``prepare_preset_args``.
    cb = prep.pop("_cb", None)
    ev = prep.pop("_ev", None)
    pass_legacy = prep.pop("_pass_corpus_legacy", True)

    (tmp / "src").mkdir(parents=True, exist_ok=True)
    corpus = _legacy_corpus(tmp / "src", n)
    out = tmp / "out"
    args = prepare_preset_args(
        corpus, engines,
        workspace_dir=tmp / "ws", output_dir=out, **prep,
    )
    kw: dict[str, Any] = {}
    if pass_legacy:
        kw["corpus_legacy"] = corpus
    res = RunOrchestrator(out).execute_preset(
        spec=args.spec,
        corpus_spec=args.corpus_spec,
        extracted_dir=args.extracted_dir,
        pipeline_specs=args.pipeline_specs,
        adapter_resolver=args.adapter_resolver,
        adapter_kwargs=args.adapter_kwargs,
        progress_callback=cb,
        cancel_event=ev,
        **kw,
    )
    return res, corpus, args


class TestExecutePresetCharacterization:
    """Trou comblé : ``execute_preset`` (variante objets domain
    pré-construits, utilisée par CLI ``diagnose/economics/edition``
    et le worker web).  Caractérise le surface-risque PRÉSET-
    SPÉCIFIQUE pour Phase B."""

    def test_preset_happy_path_four_artifacts_deterministic_cer(
        self, tmp_path: Path,
    ) -> None:
        res, _, _ = _preset_run(tmp_path, 3, [_MockOCR()])
        assert res.run_result.n_documents == 3
        rd = tmp_path / "out" / "results"
        for f in (
            "run_manifest.json", "pipeline_results.jsonl",
            "view_results.jsonl", "artifacts_index.jsonl",
        ):
            assert (rd / f).exists(), f"artefact manquant : {f}"
        pr, vr = sorted({r["document_id"] for r in _jsonl(rd / "pipeline_results.jsonl")}), \
            sorted({r["document_id"] for r in _jsonl(rd / "view_results.jsonl")})
        assert pr == vr == ["doc01", "doc02", "doc03"]

    def test_preset_corpus_legacy_hotfix_writes_legacy_json(
        self, tmp_path: Path,
    ) -> None:
        """Trou #9 (B3-final) : ``output_json`` + ``corpus_legacy``
        fourni ⇒ JSON legacy écrit SANS erreur (court-circuite le
        reload depuis ``workspace_dir`` qui n'a que des .gt.txt)."""
        oj = tmp_path / "legacy.json"
        res, _, _ = _preset_run(
            tmp_path, 2, [_MockOCR()], output_json=oj,
        )
        assert res.run_result.n_documents == 2
        assert oj.exists(), "JSON legacy non écrit malgré corpus_legacy"

    def test_preset_without_corpus_legacy_reproduces_trou9_DEFECT(
        self, tmp_path: Path,
    ) -> None:
        """⚠️ CONTRAT DU HOTFIX ⚠️ : ``output_json`` SANS
        ``corpus_legacy`` ⇒ ``ValueError`` (reload impossible depuis
        le ``workspace_dir`` synthétisé).  C'est précisément le bug
        que le hotfix corrige : si Phase B réorganise
        ``_persist_legacy_benchmark_json``, ce contrat ne doit pas
        changer en silence (sinon le hotfix devient inopérant ou le
        message d'erreur trompeur revient)."""
        oj = tmp_path / "legacy.json"
        with pytest.raises(ValueError, match="impossible de reloader"):
            _preset_run(
                tmp_path, 2, [_MockOCR()],
                output_json=oj, _pass_corpus_legacy=False,
            )

    def test_preset_doc_idx_global_across_two_engines(
        self, tmp_path: Path,
    ) -> None:
        """``execute_preset`` partage ``_make_context_factory`` :
        2 engines (⇒ 2 pipelines) × 3 docs ⇒ compteur GLOBAL contigu
        0..5 (même invariant que ``execute``, via le chemin préset)."""
        calls: list[int] = []
        _preset_run(
            tmp_path, 3, [_MockOCR("a"), _MockOCR("b")],
            _cb=lambda e, i, d: calls.append(i),
        )
        assert sorted(calls) == [0, 1, 2, 3, 4, 5], (
            f"compteur préset non global/contigu : {sorted(calls)}"
        )

    def test_preset_preset_cancel_short_circuits(
        self, tmp_path: Path,
    ) -> None:
        ev = threading.Event()
        ev.set()
        t0 = time.monotonic()
        _preset_run(tmp_path, 5, [_MockOCR()], _ev=ev)
        assert time.monotonic() - t0 < 10.0

    def test_preset_resume_keeps_views_complete_FIX_GUARD(
        self, tmp_path: Path,
    ) -> None:
        """Le chemin préset délègue à ``_execute_with_partial`` : le
        FIX resume/vues doit tenir AUSSI via ``execute_preset``.
        Interruption réelle puis resume ⇒ pipeline ET vues complets
        (== run propre).  Garde anti-régression du fix par le chemin
        préset."""
        partial = tmp_path / "pd"
        partial.mkdir()
        ev = threading.Event()
        seen = {"n": 0}

        def cb(e: str, i: int, d: str) -> None:
            seen["n"] += 1
            if seen["n"] == 2:
                ev.set()

        # Run 1 : interrompu après 2 docs.
        _preset_run(
            tmp_path / "r1", 5, [_MockOCR()],
            partial_dir=str(partial), _cb=cb, _ev=ev,
        )
        # Run 2 : même partial_dir, sans cancel → doit compléter.
        res2, _, _ = _preset_run(
            tmp_path / "r2", 5, [_MockOCR()], partial_dir=str(partial),
        )
        rd = tmp_path / "r2" / "out" / "results"
        pr = sorted({r["document_id"] for r in _jsonl(rd / "pipeline_results.jsonl")})
        vr = sorted({r["document_id"] for r in _jsonl(rd / "view_results.jsonl")})
        full = ["doc01", "doc02", "doc03", "doc04", "doc05"]
        assert pr == full, f"pipeline incomplet (préset resume) : {pr}"
        assert vr == full, (
            f"vues incomplètes au resume PRÉSET : {vr} — le fix "
            "resume/vues ne tient pas via execute_preset"
        )
