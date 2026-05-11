"""Sprint S9 — régression pour le bug du resolver d'adapter qui
plantait quand un même OCR apparaît à la fois en standalone et
encapsulé dans un pipeline OCR+LLM.

Bug observé en prod (interface web, 2026-05-10) :

    Démarrage du benchmark…
    6 documents chargés.
    Concurrent : tesseract
    Concurrent : tesseract:fra → mistral-small-latest
    Erreur : Adapter resolver : nom 'tesseract' enregistré deux fois
    avec des instances différentes — collision impossible à résoudre.

Cause : ``_engine_from_competitor`` crée une instance ``TesseractAdapter``
fraîche pour chaque ``CompetitorConfig``.  Quand deux concurrents
partagent le même moteur OCR (l'un seul, l'autre dans un pipeline),
``build_adapter_resolver`` voyait deux instances Python distinctes
sous le même ``name="tesseract"`` et levait ``PicaronesError`` à tort
— les deux instances étant fonctionnellement équivalentes (même
config Tesseract, sans état applicatif).

Fix : le resolver accepte désormais la 2e registration si l'état
public (``__dict__``) est identique.  Configuration vraiment
différente → toujours rejet (le contrat de disambiguation est
préservé).
"""

from __future__ import annotations

import pytest

from picarones.app.services.benchmark_runner import build_adapter_resolver
from picarones.domain.errors import PicaronesError


# ──────────────────────────────────────────────────────────────────────
# Cas qui reproduit le bug en prod
# ──────────────────────────────────────────────────────────────────────


class TestStandaloneAndPipelineWithSameOCR:
    """Le scénario exact qui plantait en prod : Tesseract seul +
    pipeline Tesseract+LLM avec la même config OCR."""

    def test_two_competitors_same_tesseract_config_accepted(self) -> None:
        from picarones.adapters.ocr.tesseract import TesseractAdapter
        from picarones.adapters.llm.mistral_adapter import MistralAdapter
        from picarones.pipeline.llm_pipeline_config import OCRLLMPipelineConfig

        # Competitor 1 : Tesseract seul.
        tesseract_standalone = TesseractAdapter(lang="fra", psm=6)

        # Competitor 2 : pipeline Tesseract → Mistral.  Le Tesseract
        # interne est une AUTRE instance Python mais avec la même config.
        tesseract_in_pipeline = TesseractAdapter(lang="fra", psm=6)
        assert tesseract_standalone is not tesseract_in_pipeline, (
            "test pré-condition : deux instances distinctes"
        )
        pipeline = OCRLLMPipelineConfig(
            ocr_adapter=tesseract_in_pipeline,
            llm_adapter=MistralAdapter(model="mistral-small-latest"),
            mode="text_only",
            prompt_template="correction_medieval_french.txt",
            pipeline_name="tesseract:fra → mistral-small-latest",
        )

        # Le resolver doit accepter cette config — avant le fix
        # S9, il levait ``PicaronesError``.
        resolver = build_adapter_resolver(
            [tesseract_standalone, pipeline],
        )
        # Résolution : le nom ``tesseract`` mappe vers UNE instance
        # (la 1re, par convention idempotente).
        resolved = resolver("tesseract")
        assert resolved is tesseract_standalone


class TestDifferentConfigsStillCollide:
    """Garde-fou : si deux engines partagent le même ``name`` mais
    une config différente, le resolver doit toujours rejeter — sinon
    on cacherait silencieusement un vrai bug utilisateur."""

    def test_different_lang_same_name_raises(self) -> None:
        from picarones.adapters.ocr.tesseract import TesseractAdapter

        # ``name="tesseract"`` par défaut, mais ``lang`` différent →
        # vraies configs distinctes → collision réelle.
        adapter_fra = TesseractAdapter(lang="fra", psm=6)
        adapter_eng = TesseractAdapter(lang="eng", psm=6)

        with pytest.raises(PicaronesError, match="enregistré deux fois|configurations différentes"):
            build_adapter_resolver([adapter_fra, adapter_eng])

    def test_different_psm_same_name_raises(self) -> None:
        from picarones.adapters.ocr.tesseract import TesseractAdapter

        adapter_psm6 = TesseractAdapter(lang="fra", psm=6)
        adapter_psm3 = TesseractAdapter(lang="fra", psm=3)

        with pytest.raises(PicaronesError, match="enregistré deux fois|configurations différentes"):
            build_adapter_resolver([adapter_psm6, adapter_psm3])

    def test_different_types_same_name_raises(self) -> None:
        """Deux types différents avec le même ``name`` (improbable
        en pratique mais théoriquement possible si un caller
        configure manuellement) → toujours rejeté."""
        from picarones.adapters.ocr.tesseract import TesseractAdapter
        from picarones.adapters.ocr.precomputed import (
            PrecomputedTextAdapter,
        )

        # PrecomputedTextAdapter dérive son name de ``source_label`` ;
        # on construit un Tesseract dont le name match exactement
        # celui généré par PrecomputedTextAdapter.
        precomputed = PrecomputedTextAdapter(source_label="bnf")
        # ``precomputed.name`` est "precomputed:bnf" — utilisons un
        # name compatible avec le validateur Tesseract (alphanum + _-).
        tesseract = TesseractAdapter(name=precomputed.name.replace(":", "_"))
        # Renommer mentalement pour forcer la collision : on force
        # le même name côté Tesseract et PrecomputedTextAdapter
        # via attribut interne (path direct au _dict pour test).
        tesseract.__dict__["_name"] = precomputed.name

        assert tesseract.name == precomputed.name
        with pytest.raises(PicaronesError, match="enregistré deux fois|configurations différentes"):
            build_adapter_resolver([tesseract, precomputed])


# ──────────────────────────────────────────────────────────────────────
# Idempotence : même instance enregistrée 2 fois
# ──────────────────────────────────────────────────────────────────────


class TestIdempotentRegistration:
    def test_same_instance_twice_is_idempotent(self) -> None:
        """Si une instance est enregistrée deux fois (par ex. via
        deux pipelines qui partagent la même réf d'adapter OCR),
        c'est trivialement OK."""
        from picarones.adapters.ocr.tesseract import TesseractAdapter

        adapter = TesseractAdapter(lang="fra")
        resolver = build_adapter_resolver([adapter, adapter])
        assert resolver("tesseract") is adapter
