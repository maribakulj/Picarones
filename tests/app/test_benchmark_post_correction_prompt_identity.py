"""Régression — benchmark de post-correction « même modèle, prompts
différents ».

Bug observé en prod (interface web, 2026-05-16) :

    Démarrage du benchmark…
    6 documents chargés.
    Concurrent : tesseract_fra
    Concurrent : tesseract:fra → mistral-small-latest
    Concurrent : tesseract:fra → mistral-small-latest
    Concurrent : tesseract:fra → mistral-small-latest
    Erreur : Adapter resolver : nom 'mistral:mistral-small-latest'
    enregistré deux fois avec des configurations différentes
    (MistralAdapter vs MistralAdapter, états distincts).

Cause racine : le **prompt** ne faisait pas partie de l'identité
logique d'un pipeline de post-correction.

- ``_llm_adapter_name`` = ``provider:model`` ignorait la ``config``
  de l'adapter → deux competitors « même modèle,
  ``max_image_dimension`` différent » obtenaient le même ``name``
  resolver → ``PicaronesError`` « états distincts » à tort (le cas
  exact ci-dessus : un prompt ``text_only`` et un prompt
  ``text_and_image`` avec downscale anti-429).
- ``_engine_from_competitor`` dérivait un ``pipeline_name`` par
  défaut ``{ocr} → {model}`` sans discriminant prompt → N variantes
  de prompt → ``EngineReport.engine_name`` identiques (rapport
  illisible) et clé ``view_results`` partagée (clobber doc-par-doc).
- ``_ocr_llm_pipeline_to_spec`` laissait ``make_ocr_llm_pipeline_spec``
  auto-générer ``ocr_llm_{mode}_{ocr}_to_{llm}`` (sans prompt) →
  ``PipelineResult.pipeline_name`` identiques.

Fix : les trois identifiants (nom resolver LLM, nom pipeline,
nom de spec) sont rendus injectifs sur l'identité logique
= (config OCR, modèle LLM, config LLM, mode, prompt).  Le prompt
continue de voyager dans ``llm_params`` du ``PipelineStep`` (il
n'est pas de l'état d'adapter) : une même instance LLM peut servir
plusieurs prompts, seule une config d'adapter distincte donne un
nom resolver distinct.
"""

from __future__ import annotations

from picarones.app.services import prepare_preset_args
from picarones.app.services._benchmark_adapter_resolver import (
    _llm_adapter_name,
    build_adapter_resolver,
    engine_to_pipeline_spec,
)
from picarones.evaluation.corpus import Corpus
from picarones.interfaces.web.benchmark_utils import _engine_from_competitor
from picarones.interfaces.web.models import PipelineConfig

_BASE = dict(
    engine_name="tesseract",
    ocr_model="fra",
    llm_provider="mistral",
    llm_model="mistral-small-latest",
)


def _comp(prompt_file: str, **over: object) -> PipelineConfig:
    kw: dict = {**_BASE, "pipeline_mode": "text_only", "prompt_file": prompt_file}
    kw.update(over)
    return PipelineConfig(**kw)  # type: ignore[arg-type]


def _specs(engines: list) -> list[str]:
    corpus = Corpus(name="t", documents=[])
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        preset = prepare_preset_args(
            corpus, engines,
            workspace_dir=dp / "gt", output_dir=dp / "run",
            views=("text_final",),
        )
    return [s.name for s in preset.pipeline_specs]


class TestExactProdScenario:
    """OCR seul + 3 pipelines même modèle, prompts différents."""

    def test_user_scenario_no_collision_distinct_engines(self) -> None:
        comps = [
            PipelineConfig(
                name="tesseract_fra", engine_name="tesseract",
                ocr_model="fra",
            ),
            _comp("correction_medieval_french.txt"),
            _comp("correction_early_modern_english.txt"),
            _comp("correction_medieval_english.txt"),
        ]
        engines = [_engine_from_competitor(c) for c in comps]

        # Avant le fix : ``build_adapter_resolver`` levait
        # ``PicaronesError`` quand les configs LLM divergeaient ;
        # sinon les noms se confondaient silencieusement.
        names = [e.name for e in engines]
        assert len(set(names)) == len(names), (
            f"engine.name non distincts : {names}"
        )

        # Resolver : aucune exception (les 3 LLM partagent la même
        # config par défaut → adapter dédupliqué proprement).
        resolver = build_adapter_resolver(engines)
        assert resolver is not None

        spec_names = _specs(engines)
        assert len(set(spec_names)) == len(spec_names), (
            f"spec.name non distincts (view_results se clobberent) : "
            f"{spec_names}"
        )


class TestHardErrorCaseFixed:
    """Le cas exact qui levait « états distincts » : même modèle,
    prompts différents, ``max_image_dimension`` différent."""

    def test_distinct_maxdim_does_not_raise(self) -> None:
        engines = [
            _engine_from_competitor(
                _comp("correction_medieval_french.txt", max_image_dimension=0),
            ),
            _engine_from_competitor(
                _comp(
                    "correction_image_medieval_french.txt",
                    pipeline_mode="text_and_image",
                    max_image_dimension=2048,
                ),
            ),
        ]
        # Ne doit PAS lever (avant le fix : PicaronesError).
        resolver = build_adapter_resolver(engines)
        # Les deux LLM ont des configs distinctes → noms distincts.
        n0 = _llm_adapter_name(engines[0].llm_adapter)
        n1 = _llm_adapter_name(engines[1].llm_adapter)
        assert n0 != n1
        assert resolver(n0) is engines[0].llm_adapter
        assert resolver(n1) is engines[1].llm_adapter


class TestSilentDuplicateFixed:
    """Même config LLM (pas de hard error) mais prompts différents :
    les engines ET les specs doivent rester distincts."""

    def test_same_config_different_prompts_distinct(self) -> None:
        engines = [
            _engine_from_competitor(_comp("correction_medieval_french.txt")),
            _engine_from_competitor(_comp("correction_medieval_english.txt")),
        ]
        assert engines[0].name != engines[1].name
        spec_names = _specs(engines)
        assert spec_names[0] != spec_names[1]

    def test_zero_shot_different_prompts_distinct(self) -> None:
        z = dict(
            engine_name="", llm_provider="mistral",
            llm_model="mistral-small-latest", pipeline_mode="zero_shot",
        )
        engines = [
            _engine_from_competitor(
                PipelineConfig(**z, prompt_file="zero_shot_medieval_french.txt"),
            ),
            _engine_from_competitor(
                PipelineConfig(**z, prompt_file="zero_shot_medieval_english.txt"),
            ),
        ]
        assert engines[0].name != engines[1].name
        s = [engine_to_pipeline_spec(e).name for e in engines]
        assert s[0] != s[1]


class TestIdempotentDedupPreserved:
    """Garde-fou S9 : deux competitors STRICTEMENT identiques
    (même config, même prompt) restent dédupliqués au resolver
    sans explosion de noms ni erreur."""

    def test_identical_competitors_resolver_idempotent(self) -> None:
        engines = [
            _engine_from_competitor(_comp("correction_medieval_french.txt")),
            _engine_from_competitor(_comp("correction_medieval_french.txt")),
        ]
        # Même identité logique → même name resolver LLM →
        # déduplication idempotente (pas de PicaronesError).
        assert _llm_adapter_name(engines[0].llm_adapter) == _llm_adapter_name(
            engines[1].llm_adapter,
        )
        resolver = build_adapter_resolver(engines)
        assert resolver is not None


class TestLLMAdapterNameContract:
    """``_llm_adapter_name`` : injectif sur la config, rétro-compatible
    quand la config est triviale/défaut."""

    def test_default_config_keeps_legacy_name(self) -> None:
        from picarones.adapters.llm.mistral_adapter import MistralAdapter

        # Config vide ET config {max_image_dimension: 0} (défaut web)
        # → name historique ``provider:model`` (fingerprint stable).
        a = MistralAdapter(model="mistral-small-latest")
        b = MistralAdapter(
            model="mistral-small-latest",
            config={"max_image_dimension": 0},
        )
        assert _llm_adapter_name(a) == "mistral:mistral-small-latest"
        assert _llm_adapter_name(b) == "mistral:mistral-small-latest"

    def test_significant_config_yields_distinct_name(self) -> None:
        from picarones.adapters.llm.mistral_adapter import MistralAdapter

        base = MistralAdapter(model="mistral-small-latest")
        downscaled = MistralAdapter(
            model="mistral-small-latest",
            config={"max_image_dimension": 2048},
        )
        hot = MistralAdapter(
            model="mistral-small-latest",
            config={"temperature": 0.7},
        )
        names = {
            _llm_adapter_name(base),
            _llm_adapter_name(downscaled),
            _llm_adapter_name(hot),
        }
        assert len(names) == 3, f"noms non distincts : {names}"
        # Déterminisme : même config → même name.
        again = MistralAdapter(
            model="mistral-small-latest",
            config={"max_image_dimension": 2048},
        )
        assert _llm_adapter_name(again) == _llm_adapter_name(downscaled)
