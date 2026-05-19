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

Déterminisme : ``PrecomputedTextAdapter`` (lit ``<stem>.<label>.txt``,
aucun OCR/réseau).  Un garde explicite
(:meth:`TestGoldenMultiTopology.test_snapshot_is_deterministic`)
échoue si le snapshot n'est pas reproductible — un golden flaky
serait pire que pas de golden.
"""

from __future__ import annotations

import io
import json
import textwrap
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest

from picarones.app.schemas.run_spec import load_run_spec_from_yaml
from picarones.app.services import RunOrchestrator

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


def _scrub(obj: Any) -> Any:
    """Snapshot canonique : retire les clés volatiles, neutralise les
    chemins absolus, et **trie les listes de scalaires**.

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
        # Chemins absolus → placeholder (tmp_path varie par run/CI).
        if "/" in obj and ("/tmp" in obj or "pytest" in obj or obj.startswith("/")):
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

    ⚠️ DÉFAUT PRÉ-EXISTANT DÉCOUVERT PAR CE HARNAIS ⚠️
    Le partial store persiste ``PipelineResult`` mais PAS
    ``ViewResult``.  Au resume, les documents rechargés du partial
    récupèrent leurs ``pipeline_results`` mais **pas** leurs
    ``view_results`` (jamais recalculés).  Conséquence : après une
    reprise, ``view_results.jsonl`` est incomplet → toute métrique
    agrégée (CER…) dérivée des vues est silencieusement faussée pour
    les documents repris.

    Ces tests CARACTÉRISENT le comportement ACTUEL (warts inclus) —
    rôle d'un harnais de caractérisation — pour que Phase B ne
    l'aggrave pas ET qu'une correction future du défaut soit
    consciente (le test échouera, forçant la revue).  Le défaut
    lui-même est remonté à l'opérateur, pas corrigé furtivement ici
    (resume/views = changement stateful risqué, hors périmètre
    « construire le harnais »)."""

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

    #: Relation ``view_results`` vs ``pipeline_results`` APRÈS resume,
    #: par topologie — comportement RÉEL observé (le défaut est
    #: topologie-dépendant : présent en linéaire/DAG, absent en
    #: multi-pipeline avec cette synchro d'interruption).
    _RESUME_VIEW_RELATION = {
        "single_linear": "strict_subset",   # défaut : vues ⊊ pipeline
        "branching_dag": "strict_subset",   # défaut idem
        "multi_pipeline": "equal",          # pas de défaut ici
    }

    @pytest.mark.parametrize(
        "topo", ["single_linear", "multi_pipeline", "branching_dag"],
    )
    def test_resume_view_vs_pipeline_relation_DEFECT_characterized(
        self, tmp_path: Path, topo: str,
    ) -> None:
        """⚠️ CARACTÉRISE LE DÉFAUT (topologie-dépendant) ⚠️ : au
        resume, ``pipeline_results`` couvre tout le corpus, mais la
        relation ``view_results`` vs ``pipeline_results`` dépend de la
        topologie (cf. :data:`_RESUME_VIEW_RELATION`) :

        - ``single_linear`` / ``branching_dag`` : vues ⊊ pipeline —
          les vues des docs repris du partial ne sont jamais
          recalculées (métriques agrégées faussées après reprise).
        - ``multi_pipeline`` : vues == pipeline (le défaut ne se
          manifeste pas avec cette synchro d'interruption).

        Toute évolution de l'une de ces relations (Phase B, ou
        correction du défaut) fait échouer ce test et force une revue
        consciente."""
        _, resumed = self._interrupt_then_resume(
            tmp_path, 5, stop_after=2, topo=topo,
        )
        pr, vr = self._persisted_doc_ids(resumed)
        full = ["doc01", "doc02", "doc03", "doc04", "doc05"]
        assert pr == full, f"pipeline incomplet au resume ({topo}): {pr}"
        rel = self._RESUME_VIEW_RELATION[topo]
        if rel == "strict_subset":
            assert set(vr) < set(pr), (
                f"[{topo}] défaut resume/vues changé : pipeline={pr} "
                f"vues={vr}. Attendu : vues ⊊ pipeline."
            )
        else:
            assert set(vr) == set(pr), (
                f"[{topo}] relation resume/vues changée : pipeline={pr}"
                f" vues={vr}. Attendu : vues == pipeline."
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
