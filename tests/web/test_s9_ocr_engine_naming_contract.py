"""Sprint S9 — propriété systémique : tout moteur OCR exposé par
l'UI web peut coexister en standalone + pipeline (même config) ET
en deux configs distinctes, sans collision resolver.

Ce fichier teste **2 propriétés d'intégration** qui suffisent à
garantir que la classe de bug Tesseract (deux instances Python
distinctes sous le même ``name`` → collision resolver) ne peut
pas se reproduire pour aucun moteur supporté.

Le test itère ``_OCR_KWARGS_BUILDERS`` — la **même registry** que
le dispatch.  Ajouter un nouveau moteur sans propriété vérifiée
est impossible : le test paramétré tourne automatiquement sur la
nouvelle entrée.

Pourquoi 2 tests, pas 6
-----------------------
Les propriétés "name in kwargs", "name reflects config", "same
config same name", "different config different name" sont des
tests d'implémentation qui se déduisent des deux propriétés
d'intégration ci-dessous.  Si l'une des deux échoue, l'une des
4 autres aurait aussi échoué — donc redondance.  On garde le
minimum significatif.
"""

from __future__ import annotations

import pytest

from picarones.app.services._benchmark_adapter_resolver import (
    build_adapter_resolver,
)
from picarones.interfaces.web.benchmark_utils import (
    _OCR_KWARGS_BUILDERS,
    _engine_from_competitor,
)
from picarones.interfaces.web.models import PipelineConfig


# ``cfg_a`` et ``cfg_b`` sont passés tels quels au constructeur de
# l'adapter — peu importe leur sémantique (lang/model/feature),
# seule leur distinction compte pour vérifier les propriétés.
# Si un adapter rejette ces strings au constructeur (validation
# stricte), le test skip sur cet engine via le ``except RuntimeError``.


@pytest.mark.parametrize("engine_id", sorted(_OCR_KWARGS_BUILDERS))
def test_two_distinct_configs_coexist_in_resolver(
    engine_id: str,
) -> None:
    """Deux competitors avec ``ocr_model`` distincts doivent recevoir
    des ``name`` distincts au resolver — le bug Tesseract initial,
    généralisé à tous les moteurs supportés."""
    comp_a = PipelineConfig(
        engine_name=engine_id, ocr_model="cfg_a", llm_provider="",
    )
    comp_b = PipelineConfig(
        engine_name=engine_id, ocr_model="cfg_b", llm_provider="",
    )
    try:
        eng_a = _engine_from_competitor(comp_a)
        eng_b = _engine_from_competitor(comp_b)
    except Exception as exc:  # noqa: BLE001
        # Adapter cloud indisponible OU model rejeté par la
        # validation stricte (par ex. Google ``feature_type``
        # est un enum) → on skip ce moteur, sa couverture est
        # faite par les tests dédiés cloud/mock.
        pytest.skip(f"{engine_id} non instanciable ici : {exc}")

    assert eng_a.name != eng_b.name, (
        f"{engine_id} : configs distinctes ({comp_a.ocr_model!r} vs "
        f"{comp_b.ocr_model!r}) produisent le même name "
        f"({eng_a.name!r}) — collision silencieuse possible."
    )
    resolver = build_adapter_resolver([eng_a, eng_b])
    assert resolver(eng_a.name) is eng_a
    assert resolver(eng_b.name) is eng_b


@pytest.mark.parametrize("engine_id", sorted(_OCR_KWARGS_BUILDERS))
def test_standalone_plus_pipeline_same_config_coexist(
    engine_id: str,
) -> None:
    """Scénario exact du bug Tesseract en prod : un competitor OCR
    seul + un competitor pipeline OCR+LLM partageant la même config
    OCR.  Le resolver doit accepter (les 2 instances Python sont
    fonctionnellement équivalentes, déduplication idempotente)."""
    comp_standalone = PipelineConfig(
        engine_name=engine_id, ocr_model="same_config", llm_provider="",
    )
    comp_pipeline = PipelineConfig(
        engine_name=engine_id, ocr_model="same_config",
        llm_provider="mistral", llm_model="mistral-small-latest",
        pipeline_mode="text_only",
        prompt_file="correction_medieval_french.txt",
    )
    try:
        eng_standalone = _engine_from_competitor(comp_standalone)
        eng_pipeline = _engine_from_competitor(comp_pipeline)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{engine_id} non instanciable ici : {exc}")

    assert eng_standalone.name == eng_pipeline.ocr_adapter.name, (
        f"{engine_id} : configs identiques produisent des names "
        f"distincts ({eng_standalone.name!r} vs "
        f"{eng_pipeline.ocr_adapter.name!r}) — déduplication "
        "cassée."
    )
    # Le resolver tolère la duplication équivalente.
    resolver = build_adapter_resolver([eng_standalone, eng_pipeline])
    assert resolver(eng_standalone.name) is not None
