"""Sprint S9 — propriété systémique : tout moteur OCR exposé par
l'UI web doit avoir un ``name`` qui reflète sa config.

Ce fichier teste **architecturalement** que la classe de bug
Tesseract (deux competitors avec la même config créaient deux
instances Python distinctes sous le même ``name`` → resolver
collision) ne peut pas se reproduire avec un autre moteur OCR.

Trois propriétés invariantes vérifiées sur **chaque** entrée
de ``_OCR_KWARGS_BUILDERS`` (pas seulement Tesseract) :

1. **Le ``name`` est toujours injecté dans les kwargs** (jamais
   l'oubli silencieux qui causait le bug Tesseract).
2. **Deux configs identiques → noms égaux** (déduplication propre
   au resolver via l'équivalence d'état ``__dict__``).
3. **Deux configs distinctes → noms distincts** (pas de collision
   silencieuse côté resolver, registration correcte de deux engines
   logiquement distincts).

Le test itère la **même registry** qu'utilise le dispatch — donc
ajouter un nouveau moteur OCR sans le passer par ces propriétés
est impossible : le test échoue immédiatement sur la nouvelle
entrée non couverte.
"""

from __future__ import annotations

import pytest

from picarones.app.services.benchmark_runner import build_adapter_resolver
from picarones.interfaces.web.benchmark_utils import (
    _OCR_KWARGS_BUILDERS,
    _build_ocr_kwargs,
    _engine_from_competitor,
)
from picarones.interfaces.web.models import CompetitorConfig


# Chaque engine est testé avec deux ``ocr_model`` distinctes.  Pour
# Tesseract on prend deux langues, pour les autres deux models /
# feature_types légitimes.  Si un futur engine n'a pas d'``ocr_model``
# significatif, mettre 2 strings arbitraires non vides.
_ENGINE_SAMPLE_MODELS: dict[str, tuple[str, str]] = {
    "tesseract": ("fra", "eng"),
    "mistral_ocr": ("mistral-ocr-latest", "mistral-ocr-v2"),
    "google_vision": ("DOCUMENT_TEXT_DETECTION", "TEXT_DETECTION"),
    "azure_doc_intel": ("prebuilt-read", "prebuilt-layout"),
}


def test_sample_models_covers_every_supported_engine() -> None:
    """Garde-fou de cohérence : toute entrée de la registry de
    dispatch doit avoir des sample models déclarés ici, sinon le
    test paramétré qui suit n'exercerait pas la nouvelle entrée.

    Si ce test échoue après l'ajout d'un nouveau moteur, ajouter
    deux ``ocr_model`` exemples dans ``_ENGINE_SAMPLE_MODELS``
    ci-dessus — c'est la check-list pour ne pas oublier."""
    missing = set(_OCR_KWARGS_BUILDERS) - set(_ENGINE_SAMPLE_MODELS)
    assert not missing, (
        f"Engines présents dans la registry mais sans sample models "
        f"dans le test : {missing!r}.  Ajouter une entrée dans "
        f"``_ENGINE_SAMPLE_MODELS`` pour couvrir les propriétés "
        f"d'unicité des names."
    )


# ──────────────────────────────────────────────────────────────────────
# Propriété 1 : name toujours présent dans les kwargs
# ──────────────────────────────────────────────────────────────────────


class TestNameAlwaysInjected:
    @pytest.mark.parametrize("engine_id", sorted(_OCR_KWARGS_BUILDERS))
    def test_name_in_kwargs_for_every_engine(self, engine_id: str) -> None:
        """Le ``name`` est systématiquement injecté par
        ``_build_ocr_kwargs``, peu importe le moteur.  C'est la
        garantie centrale anti-régression Tesseract."""
        sample_model = _ENGINE_SAMPLE_MODELS[engine_id][0]
        kwargs = _build_ocr_kwargs(engine_id, sample_model)
        assert "name" in kwargs, (
            f"L'engine {engine_id!r} a perdu le kwarg ``name`` — "
            "le bug de collision resolver peut revenir pour ce moteur."
        )
        assert kwargs["name"], (
            f"L'engine {engine_id!r} a ``name`` vide — "
            "incompatible avec ``_validate_name`` des adapters OCR."
        )

    @pytest.mark.parametrize("engine_id", sorted(_OCR_KWARGS_BUILDERS))
    def test_name_includes_config_for_every_engine(
        self, engine_id: str,
    ) -> None:
        """Le ``name`` dérivé inclut la config (model/lang/feature)
        — sinon deux configs distinctes pourraient avoir le même
        name.  Pour ``engine_id="tesseract"`` + ``ocr_model="fra"``,
        on attend que ``"fra"`` apparaisse (sanitizé) dans le name."""
        sample_model = _ENGINE_SAMPLE_MODELS[engine_id][0]
        kwargs = _build_ocr_kwargs(engine_id, sample_model)
        # Sanitization : ``mistral-ocr-latest`` → ``mistral-ocr-latest``
        # (intact car alphanum+``-``).  Le suffixe doit apparaître
        # tel quel dans le name (modulo la sanitization).
        from picarones.interfaces.web.benchmark_utils import (
            _sanitize_name_suffix,
        )

        expected_suffix = _sanitize_name_suffix(sample_model)
        assert expected_suffix in kwargs["name"], (
            f"Engine {engine_id!r} model {sample_model!r} → "
            f"name {kwargs['name']!r} ne contient pas le suffix "
            f"{expected_suffix!r}.  Deux configs distinctes auraient "
            "potentiellement le même name."
        )


# ──────────────────────────────────────────────────────────────────────
# Propriété 2 : configs identiques → noms égaux (déduplication)
# ──────────────────────────────────────────────────────────────────────


class TestSameConfigSameName:
    @pytest.mark.parametrize("engine_id", sorted(_OCR_KWARGS_BUILDERS))
    def test_identical_configs_produce_identical_names(
        self, engine_id: str,
    ) -> None:
        """Deux competitors avec la même config doivent recevoir
        le même ``name`` au resolver → déduplication automatique.
        C'est le cas Tesseract+Pipeline du bug initial."""
        sample_model = _ENGINE_SAMPLE_MODELS[engine_id][0]
        k1 = _build_ocr_kwargs(engine_id, sample_model)
        k2 = _build_ocr_kwargs(engine_id, sample_model)
        assert k1["name"] == k2["name"]


# ──────────────────────────────────────────────────────────────────────
# Propriété 3 : configs distinctes → noms distincts (anti-collision)
# ──────────────────────────────────────────────────────────────────────


class TestDifferentConfigsDifferentNames:
    @pytest.mark.parametrize("engine_id", sorted(_OCR_KWARGS_BUILDERS))
    def test_different_models_produce_different_names(
        self, engine_id: str,
    ) -> None:
        """Deux competitors avec ``ocr_model`` distincts doivent
        recevoir des ``name`` distincts au resolver — sinon
        collision silencieuse, l'un des deux sera ignoré."""
        model_a, model_b = _ENGINE_SAMPLE_MODELS[engine_id]
        ka = _build_ocr_kwargs(engine_id, model_a)
        kb = _build_ocr_kwargs(engine_id, model_b)
        assert ka["name"] != kb["name"], (
            f"Engine {engine_id!r} produit le même name "
            f"({ka['name']!r}) pour deux models différents "
            f"({model_a!r} vs {model_b!r}) — collision silencieuse "
            "possible au resolver."
        )


# ──────────────────────────────────────────────────────────────────────
# Propriété d'intégration : resolver accepte les configs distinctes
# ──────────────────────────────────────────────────────────────────────


class TestResolverAcceptsAllConfigs:
    """Vérifie de bout-en-bout que pour chaque moteur, deux
    competitors avec des configs distinctes peuvent coexister
    dans un même benchmark sans déclencher la collision resolver.
    Test d'intégration qui aurait pris le bug Tesseract initialement
    — et qui, paramétré sur la registry, le prendra pour tout
    nouveau moteur ajouté."""

    @pytest.mark.parametrize("engine_id", sorted(_OCR_KWARGS_BUILDERS))
    def test_two_distinct_configs_coexist(self, engine_id: str) -> None:
        model_a, model_b = _ENGINE_SAMPLE_MODELS[engine_id]
        comp_a = CompetitorConfig(
            ocr_engine=engine_id, ocr_model=model_a, llm_provider="",
        )
        comp_b = CompetitorConfig(
            ocr_engine=engine_id, ocr_model=model_b, llm_provider="",
        )
        try:
            eng_a = _engine_from_competitor(comp_a)
            eng_b = _engine_from_competitor(comp_b)
        except RuntimeError as exc:
            # Adapter cloud sans SDK → on skip ce moteur pour ce test
            # (la couverture est faite ailleurs avec mock).
            pytest.skip(f"{engine_id} indisponible : {exc}")

        # Le resolver doit accepter les deux sans collision.
        resolver = build_adapter_resolver([eng_a, eng_b])
        # Et les noms doivent être distincts (sinon l'un serait
        # silencieusement déduplique).
        assert eng_a.name != eng_b.name
        assert resolver(eng_a.name) is eng_a
        assert resolver(eng_b.name) is eng_b

    @pytest.mark.parametrize("engine_id", sorted(_OCR_KWARGS_BUILDERS))
    def test_standalone_plus_pipeline_same_config_dedupe(
        self, engine_id: str,
    ) -> None:
        """Le scénario exact du bug Tesseract en prod, généralisé à
        tous les moteurs : OCR seul + pipeline OCR+LLM partageant
        le même OCR → resolver déduplique."""
        sample_model = _ENGINE_SAMPLE_MODELS[engine_id][0]
        comp_standalone = CompetitorConfig(
            ocr_engine=engine_id, ocr_model=sample_model, llm_provider="",
        )
        comp_pipeline = CompetitorConfig(
            ocr_engine=engine_id, ocr_model=sample_model,
            llm_provider="mistral", llm_model="mistral-small-latest",
            pipeline_mode="text_only",
            prompt_file="correction_medieval_french.txt",
        )
        try:
            eng_standalone = _engine_from_competitor(comp_standalone)
            eng_pipeline = _engine_from_competitor(comp_pipeline)
        except RuntimeError as exc:
            pytest.skip(f"{engine_id} indisponible : {exc}")

        # Pas d'exception → le resolver tolère la duplication.
        resolver = build_adapter_resolver(
            [eng_standalone, eng_pipeline],
        )
        # Le standalone et l'OCR interne du pipeline ont le même
        # name (configs identiques).
        assert eng_standalone.name == eng_pipeline.ocr_adapter.name
        # Le resolver retourne une instance pour ce name (peu
        # importe laquelle, équivalentes).
        assert resolver(eng_standalone.name) is not None
