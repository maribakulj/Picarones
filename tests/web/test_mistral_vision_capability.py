"""Régression : classification vision/text-only des modèles Mistral.

Bug rapporté (mai 2026) : ``mistral-small-latest`` n'apparaissait pas
dans la liste des modèles multimodaux côté UI alors qu'il est
multimodal depuis Mistral Small 3.1 (mars 2025).  Cause : il était
listé en dur dans ``engine_utils.MISTRAL_TEXT_ONLY`` (données
obsolètes du temps où Small v2/2409 était text-only).

Ces tests verrouillent la matrice de capacités correcte :

- ``mistral-small-latest`` / ``2503`` / ``2506`` → multimodal
- ``mistral-small-2402`` / ``2409`` / ``2501`` → text-only (versions
  antérieures à 3.1)
- ``ministral-*`` → text-only (modèles edge sans vision, confirmé
  par ``data/pricing.yaml``)
- ``pixtral-*`` → multimodal (inchangé)
- cohérence avec le runtime ``MistralAdapter._TEXT_ONLY_MODELS``
"""

from __future__ import annotations

import pytest


class TestInferMistralCapabilities:
    @pytest.mark.parametrize("model_id", [
        "mistral-small-latest",
        "mistral-small-2503",
        "mistral-small-2506",
        "MISTRAL-SMALL-LATEST",  # casse insensible
    ])
    def test_small_3_1_plus_is_vision(self, model_id: str) -> None:
        from picarones.interfaces.web.engine_utils import (
            infer_mistral_capabilities,
        )
        caps = infer_mistral_capabilities(model_id)
        assert "vision" in caps, (
            f"{model_id} doit être multimodal (Mistral Small 3.1+) — "
            f"reçu {caps}"
        )
        assert "text" in caps

    @pytest.mark.parametrize("model_id", [
        "mistral-small-2402",
        "mistral-small-2409",
        "mistral-small-2501",
    ])
    def test_small_pre_3_1_is_text_only(self, model_id: str) -> None:
        from picarones.interfaces.web.engine_utils import (
            infer_mistral_capabilities,
        )
        caps = infer_mistral_capabilities(model_id)
        assert caps == ["text"], (
            f"{model_id} (antérieur à Small 3.1) doit être text-only — "
            f"reçu {caps}"
        )

    @pytest.mark.parametrize("model_id", [
        "ministral-3b-latest",
        "ministral-8b-latest",
    ])
    def test_ministral_is_text_only(self, model_id: str) -> None:
        """Ministral = modèles edge SANS vision (cf. pricing.yaml
        'Text-only, ne supporte pas le mode multimodal')."""
        from picarones.interfaces.web.engine_utils import (
            infer_mistral_capabilities,
        )
        assert infer_mistral_capabilities(model_id) == ["text"]

    @pytest.mark.parametrize("model_id", [
        "pixtral-12b-2409",
        "pixtral-large-latest",
    ])
    def test_pixtral_is_vision(self, model_id: str) -> None:
        from picarones.interfaces.web.engine_utils import (
            infer_mistral_capabilities,
        )
        assert infer_mistral_capabilities(model_id) == ["text", "vision"]

    @pytest.mark.parametrize("model_id", [
        "mistral-large-latest",
        "mistral-medium-latest",
    ])
    def test_large_medium_are_vision(self, model_id: str) -> None:
        from picarones.interfaces.web.engine_utils import (
            infer_mistral_capabilities,
        )
        assert "vision" in infer_mistral_capabilities(model_id)


class TestRuntimeAdapterConsistency:
    """La classification UI doit être cohérente avec le runtime
    ``MistralAdapter._TEXT_ONLY_MODELS`` : un modèle annoncé
    multimodal côté UI doit effectivement recevoir l'image au
    runtime (sinon l'utilisateur croit que l'image est prise en
    compte alors qu'elle est silencieusement droppée)."""

    def test_small_latest_not_in_runtime_text_only(self) -> None:
        from picarones.adapters.llm.mistral_adapter import (
            _TEXT_ONLY_MODELS,
        )
        # mistral-small-latest multimodal → l'image DOIT être envoyée.
        assert "mistral-small-latest" not in _TEXT_ONLY_MODELS

    def test_old_small_versions_in_runtime_text_only(self) -> None:
        """Les vieilles versions text-only doivent dégrader
        gracieusement (pas d'erreur API si image fournie)."""
        from picarones.adapters.llm.mistral_adapter import (
            _TEXT_ONLY_MODELS,
        )
        for old in (
            "mistral-small-2402", "mistral-small-2409",
            "mistral-small-2501",
        ):
            assert old in _TEXT_ONLY_MODELS, (
                f"{old} (text-only) doit être dans le runtime "
                "_TEXT_ONLY_MODELS pour éviter une erreur API "
                "si une image est fournie en mode text_and_image"
            )

    def test_ui_vision_models_match_runtime(self) -> None:
        """Tout modèle annoncé 'vision' côté UI ne doit PAS être
        dans le runtime text-only (sinon image droppée en silence)."""
        from picarones.adapters.llm.mistral_adapter import (
            _TEXT_ONLY_MODELS,
        )
        from picarones.interfaces.web.engine_utils import (
            MISTRAL_SMALL_VISION,
            infer_mistral_capabilities,
        )
        for model_id in MISTRAL_SMALL_VISION:
            caps = infer_mistral_capabilities(model_id)
            assert "vision" in caps
            assert model_id not in _TEXT_ONLY_MODELS, (
                f"{model_id} annoncé vision côté UI mais présent "
                "dans le runtime _TEXT_ONLY_MODELS → l'image serait "
                "silencieusement droppée"
            )
