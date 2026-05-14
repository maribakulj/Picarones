"""Phase B1 — extension de ``RunSpec`` avec 7 nouveaux champs.

Tests de la surface API ajoutée pour porter les paramètres legacy de
``run_benchmark_via_service`` dans la spec déclarative pendant le
chantier de migration Option B (cf. ``docs/migration/``).

À ce stade, les champs sont validés mais **pas consommés** par
``RunOrchestrator`` — c'est l'objet des Phases B2.1 à B2.7.  Les tests
ici vérifient donc uniquement :

1. La validation pydantic (types, regex, plage, défaut).
2. L'acceptation des kwargs d'exécution ``progress_callback`` et
   ``cancel_event`` sur :meth:`RunOrchestrator.execute`.
3. Que les feature parity tests (``test_run_orchestrator_feature_parity``)
   peuvent **construire** un RunSpec avec n'importe quel paramètre —
   c'est l'API stable sur laquelle B2 va s'appuyer.
"""

from __future__ import annotations

import io
import textwrap
import threading
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from picarones.app.schemas.run_spec import RunSpec, load_run_spec_from_yaml
from picarones.app.services import RunOrchestrator


# ──────────────────────────────────────────────────────────────────────
# Fixture YAML minimal — réutilisée dans tous les tests
# ──────────────────────────────────────────────────────────────────────


def _minimal_yaml(
    *,
    output_dir: Path,
    corpus_dir: Path = Path("/tmp/picarones-stub-corpus"),
    extra: str = "",
) -> str:
    """YAML minimal valide pour instancier un ``RunSpec``.

    ``corpus_dir`` n'a pas besoin d'exister à ce stade — la validation
    Pydantic vérifie la structure, pas le filesystem.
    """
    return textwrap.dedent(f"""
        corpus_dir: {corpus_dir}
        corpus_name: b1_test
        pipelines:
          - name: only_one
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
    """) + extra


# ──────────────────────────────────────────────────────────────────────
# B1.1 — défauts des 7 nouveaux champs
# ──────────────────────────────────────────────────────────────────────


class TestDefaults:
    def test_all_seven_fields_have_canonical_defaults(self, tmp_path: Path) -> None:
        spec = load_run_spec_from_yaml(_minimal_yaml(output_dir=tmp_path / "out"))

        assert spec.char_exclude is None
        assert spec.normalization_profile is None
        assert spec.partial_dir is None
        assert spec.entity_extractor is None
        assert spec.profile == "standard"
        assert spec.output_json is None
        assert spec.timeout_seconds_per_doc == 60.0

    def test_defaults_match_prepare_preset_args_defaults(
        self, tmp_path: Path,
    ) -> None:
        """Les valeurs par défaut de ``RunSpec`` matchent celles de
        ``prepare_preset_args`` pour cohérence avec l'API publique
        Python (callers qui instancient des adapters).

        Phase B3-final (mai 2026) — ce test remplace l'ancien
        ``test_defaults_match_run_benchmark_via_service_defaults``
        qui inspectait la fonction legacy supprimée.
        """
        import inspect

        from picarones.app.services import prepare_preset_args

        sig = inspect.signature(prepare_preset_args)
        defaults = {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not inspect.Parameter.empty
        }
        spec = load_run_spec_from_yaml(_minimal_yaml(output_dir=tmp_path / "out"))

        assert spec.char_exclude == defaults["char_exclude"]
        assert spec.normalization_profile == defaults["normalization_profile"]
        assert spec.partial_dir == defaults["partial_dir"]
        assert spec.profile == defaults["profile"]
        assert spec.timeout_seconds_per_doc == defaults["timeout_seconds_per_doc"]


# ──────────────────────────────────────────────────────────────────────
# B1.1 — char_exclude
# ──────────────────────────────────────────────────────────────────────


class TestCharExclude:
    def test_accepts_punctuation_string(self, tmp_path: Path) -> None:
        spec = load_run_spec_from_yaml(_minimal_yaml(
            output_dir=tmp_path / "out", extra="char_exclude: \"!?.,;:\"\n",
        ))
        assert spec.char_exclude == "!?.,;:"

    def test_accepts_unicode_string(self, tmp_path: Path) -> None:
        spec = load_run_spec_from_yaml(_minimal_yaml(
            output_dir=tmp_path / "out", extra="char_exclude: \"æœÆŒ\"\n",
        ))
        assert spec.char_exclude == "æœÆŒ"

    def test_rejects_too_long(self, tmp_path: Path) -> None:
        long_value = "x" * 513
        with pytest.raises((ValidationError, Exception)):
            load_run_spec_from_yaml(_minimal_yaml(
                output_dir=tmp_path / "out",
                extra=f'char_exclude: "{long_value}"\n',
            ))


# ──────────────────────────────────────────────────────────────────────
# B1.1 — entity_extractor (dotted path validation)
# ──────────────────────────────────────────────────────────────────────


class TestEntityExtractor:
    @pytest.mark.parametrize("valid_path", [
        "picarones.adapters.ner.spacy:SpacyEntityExtractor",
        "picarones.adapters.ner.spacy.SpacyEntityExtractor",
        "pkg.sub:func",
        "pkg.sub.func",
        "a:b",
        "abc.def.ghi.JKL",
        "module_with_underscore.sub_module:ClassName",
    ])
    def test_accepts_valid_dotted_paths(
        self, tmp_path: Path, valid_path: str,
    ) -> None:
        spec = load_run_spec_from_yaml(_minimal_yaml(
            output_dir=tmp_path / "out",
            extra=f"entity_extractor: {valid_path!r}\n",
        ))
        assert spec.entity_extractor == valid_path

    @pytest.mark.parametrize("invalid_path", [
        "no_dots_at_all",      # pas de séparateur
        ".starts.with.dot",    # commence par un point
        "ends.with.dot.",      # termine par un point
        "has spaces",          # espaces interdits
        "has-dash:Class",      # tirets interdits
        "123starts.with.digit",  # commence par un chiffre
        ":just.colon",         # commence par ``:``
        "module:",             # symbole vide après ``:``
    ])
    def test_rejects_invalid_format(
        self, tmp_path: Path, invalid_path: str,
    ) -> None:
        with pytest.raises((ValidationError, Exception)):
            load_run_spec_from_yaml(_minimal_yaml(
                output_dir=tmp_path / "out",
                extra=f"entity_extractor: {invalid_path!r}\n",
            ))

    def test_none_is_accepted(self, tmp_path: Path) -> None:
        spec = load_run_spec_from_yaml(_minimal_yaml(output_dir=tmp_path / "out"))
        assert spec.entity_extractor is None


# ──────────────────────────────────────────────────────────────────────
# B1.1 — profile (validate_profile)
# ──────────────────────────────────────────────────────────────────────


class TestProfile:
    def test_default_is_standard(self, tmp_path: Path) -> None:
        spec = load_run_spec_from_yaml(_minimal_yaml(output_dir=tmp_path / "out"))
        assert spec.profile == "standard"

    @pytest.mark.parametrize("valid_profile", [
        "standard",
        "diagnostics",
        "economics",
        "pipeline",
        "full",
    ])
    def test_accepts_known_profiles(
        self, tmp_path: Path, valid_profile: str,
    ) -> None:
        spec = load_run_spec_from_yaml(_minimal_yaml(
            output_dir=tmp_path / "out",
            extra=f"profile: {valid_profile}\n",
        ))
        assert spec.profile == valid_profile

    def test_rejects_unknown_profile(self, tmp_path: Path) -> None:
        with pytest.raises((ValidationError, Exception), match="profil"):
            load_run_spec_from_yaml(_minimal_yaml(
                output_dir=tmp_path / "out",
                extra="profile: philolagic_typo\n",
            ))

    def test_validates_at_construction_not_at_runtime(self, tmp_path: Path) -> None:
        """Le rejet d'un profil inconnu se fait à la construction du
        ``RunSpec``, pas au moment où ``execute()`` est appelée.

        Sémantique identique à
        ``run_benchmark_via_service(profile="unknown")`` qui lève
        AVANT toute exécution OCR (cf. ``test_sprint_d2cdef_features.py``).
        """
        with pytest.raises((ValidationError, Exception)):
            RunSpec(
                corpus_dir="/tmp/stub",
                pipelines=[],  # invalide mais on teste profile en premier
                views=("text_final",),
                output_dir=str(tmp_path / "out"),
                profile="not_real",
            )


# ──────────────────────────────────────────────────────────────────────
# B1.1 — timeout_seconds_per_doc (gt=0, le=86400)
# ──────────────────────────────────────────────────────────────────────


class TestTimeout:
    def test_default_is_60(self, tmp_path: Path) -> None:
        spec = load_run_spec_from_yaml(_minimal_yaml(output_dir=tmp_path / "out"))
        assert spec.timeout_seconds_per_doc == 60.0

    def test_accepts_custom_value(self, tmp_path: Path) -> None:
        spec = load_run_spec_from_yaml(_minimal_yaml(
            output_dir=tmp_path / "out",
            extra="timeout_seconds_per_doc: 300.5\n",
        ))
        assert spec.timeout_seconds_per_doc == 300.5

    @pytest.mark.parametrize("invalid_value", [0, -1, -100.5])
    def test_rejects_zero_or_negative(
        self, tmp_path: Path, invalid_value: float,
    ) -> None:
        with pytest.raises((ValidationError, Exception)):
            load_run_spec_from_yaml(_minimal_yaml(
                output_dir=tmp_path / "out",
                extra=f"timeout_seconds_per_doc: {invalid_value}\n",
            ))

    def test_rejects_extreme_values(self, tmp_path: Path) -> None:
        """Plafond à 24h pour éviter qu'un YAML mal formé bloque la CI
        pendant des jours."""
        with pytest.raises((ValidationError, Exception)):
            load_run_spec_from_yaml(_minimal_yaml(
                output_dir=tmp_path / "out",
                extra="timeout_seconds_per_doc: 1000000\n",
            ))


# ──────────────────────────────────────────────────────────────────────
# B1.1 — partial_dir et output_json (chemins facultatifs)
# ──────────────────────────────────────────────────────────────────────


class TestOptionalPaths:
    def test_partial_dir_accepts_path_string(self, tmp_path: Path) -> None:
        spec = load_run_spec_from_yaml(_minimal_yaml(
            output_dir=tmp_path / "out",
            extra=f"partial_dir: {tmp_path / 'partial'}\n",
        ))
        assert spec.partial_dir == str(tmp_path / "partial")

    def test_output_json_accepts_path_string(self, tmp_path: Path) -> None:
        spec = load_run_spec_from_yaml(_minimal_yaml(
            output_dir=tmp_path / "out",
            extra=f"output_json: {tmp_path / 'bm.json'}\n",
        ))
        assert spec.output_json == str(tmp_path / "bm.json")


# ──────────────────────────────────────────────────────────────────────
# B1.1 — normalization_profile (string libre, validation runtime en B2.5)
# ──────────────────────────────────────────────────────────────────────


class TestNormalizationProfile:
    def test_accepts_canonical_profile_names(self, tmp_path: Path) -> None:
        for profile in ["caseless", "medieval_french", "sans_apostrophes"]:
            spec = load_run_spec_from_yaml(_minimal_yaml(
                output_dir=tmp_path / "out",
                extra=f"normalization_profile: {profile}\n",
            ))
            assert spec.normalization_profile == profile

    def test_accepts_unknown_profile_at_schema_level(
        self, tmp_path: Path,
    ) -> None:
        """Phase B1 — pas de validation du contenu, c'est B2.5 qui
        branchera le profil au compute_metrics.  Un YAML peut nommer
        un profil custom qui sera résolu au runtime."""
        spec = load_run_spec_from_yaml(_minimal_yaml(
            output_dir=tmp_path / "out",
            extra="normalization_profile: my_custom_profile\n",
        ))
        assert spec.normalization_profile == "my_custom_profile"


# ──────────────────────────────────────────────────────────────────────
# B1.2 — kwargs d'exécution sur RunOrchestrator.execute()
# ──────────────────────────────────────────────────────────────────────


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89"
    )


def _make_corpus_zip() -> bytes:
    """Corpus zip minimal pour ``PrecomputedTextAdapter`` (1 doc)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr("doc01.png", _png_bytes())
        zf.writestr("doc01.gt.txt", "Bonjour")
        zf.writestr("doc01.tess.txt", "Bonjour")
    return buf.getvalue()


def _build_spec(tmp_path: Path) -> RunSpec:
    """Construit un RunSpec valide pointant vers un corpus_zip réel."""
    corpus_zip = tmp_path / "c.zip"
    corpus_zip.write_bytes(_make_corpus_zip())
    out_dir = tmp_path / "out"
    yaml = textwrap.dedent(f"""
        corpus_zip: {corpus_zip}
        corpus_name: b1_exec_test
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
        output_dir: {out_dir}
        code_version: "1.0.0-b1-test"
    """)
    return load_run_spec_from_yaml(yaml)


class TestExecuteKwargs:
    def test_execute_accepts_progress_callback(self, tmp_path: Path) -> None:
        """``progress_callback`` est accepté en kwarg.

        À ce stade (B1.2), le callback n'est pas encore branché au
        runner — Phase B2.1 le fera.  Ce test vérifie juste que la
        signature accepte le kwarg sans lever et que le run réussit.
        """
        spec = _build_spec(tmp_path)
        invocations: list[tuple] = []

        def cb(engine: str, idx: int, doc_id: str) -> None:
            invocations.append((engine, idx, doc_id))

        result = RunOrchestrator(tmp_path / "out").execute(
            spec, progress_callback=cb,
        )

        assert result.run_result.n_documents == 1
        # Phase B1.2 : le callback n'est pas encore invoqué (B2.1).
        # Quand B2.1 sera fait, ce test sera dé-skippé et l'assertion
        # passera à ``len(invocations) == 1``.

    def test_execute_accepts_cancel_event(self, tmp_path: Path) -> None:
        """``cancel_event`` est accepté en kwarg.

        Phase B2.2 le branchera au CorpusRunner.
        """
        spec = _build_spec(tmp_path)
        ev = threading.Event()

        result = RunOrchestrator(tmp_path / "out").execute(
            spec, cancel_event=ev,
        )

        assert result.run_result.n_documents == 1

    def test_execute_without_new_kwargs_still_works(
        self, tmp_path: Path,
    ) -> None:
        """Compat ascendante : un appel sans les nouveaux kwargs
        fonctionne comme avant."""
        spec = _build_spec(tmp_path)
        result = RunOrchestrator(tmp_path / "out").execute(spec)
        assert result.run_result.n_documents == 1

    def test_execute_stores_kwargs_on_instance(self, tmp_path: Path) -> None:
        """Phase B1.2 — les kwargs sont stockés sur l'instance.

        Quand B2.1/B2.2 brancheront le câblage interne, ils liront
        ``self._progress_callback`` et ``self._cancel_event``.
        """
        spec = _build_spec(tmp_path)

        def cb(engine: str, idx: int, doc_id: str) -> None:
            return None

        ev = threading.Event()
        orch = RunOrchestrator(tmp_path / "out")
        orch.execute(spec, progress_callback=cb, cancel_event=ev)

        # Les kwargs sont accessibles après execute().
        assert orch._progress_callback is cb
        assert orch._cancel_event is ev
