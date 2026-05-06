"""Garde-fou de stabilité d'API : les symboles dépréciés au S57
restent accessibles avec ``DeprecationWarning`` jusqu'à la 2.0.

Pour une release institutionnelle, supprimer un symbole exporté du
package public exige une deprecation period publique — un caller
externe (espace HuggingFace tiers, script BnF, notebook de chercheur)
doit pouvoir mettre à jour son code AVANT la cassure dure.

Trois alias couverts :

1. ``picarones.pipeline.spec`` (module entier).
2. ``BaseLLMAdapter.DEFAULT_CORRECTION_PROMPT`` (singulier).
3. ``BaseVLMAdapter.DEFAULT_TRANSCRIPTION_PROMPT`` (singulier).
"""

from __future__ import annotations

import importlib
import sys
import warnings


def test_pipeline_spec_module_emits_deprecation_warning() -> None:
    """``from picarones.pipeline.spec import …`` fonctionne avec un
    ``DeprecationWarning`` qui pointe vers le chemin canonique.
    """
    sys.modules.pop("picarones.pipeline.spec", None)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        importlib.import_module("picarones.pipeline.spec")
    deprecations = [
        w for w in captured if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecations, "DeprecationWarning attendu sur l'import legacy."
    assert "picarones.domain" in str(deprecations[0].message), (
        "Le message du warning doit pointer vers la cible canonique."
    )


def test_pipeline_spec_module_still_resolves_classes() -> None:
    """L'alias résout vers les MÊMES objets que ``picarones.domain``."""
    sys.modules.pop("picarones.pipeline.spec", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from picarones.pipeline.spec import (
            INITIAL_STEP_ID as LegacyInit,
        )
        from picarones.pipeline.spec import (
            PipelineSpec as LegacySpec,
        )
        from picarones.pipeline.spec import (
            PipelineStep as LegacyStep,
        )
    from picarones.domain.pipeline_spec import (
        INITIAL_STEP_ID,
        PipelineSpec,
        PipelineStep,
    )
    assert LegacySpec is PipelineSpec
    assert LegacyStep is PipelineStep
    assert LegacyInit == INITIAL_STEP_ID


def test_default_correction_prompt_singular_emits_warning() -> None:
    """``BaseLLMAdapter.DEFAULT_CORRECTION_PROMPT`` (singulier) reste
    lisible mais émet ``DeprecationWarning``.
    """
    from picarones.adapters.llm.base import BaseLLMAdapter

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        value = BaseLLMAdapter.DEFAULT_CORRECTION_PROMPT
    deprecations = [
        w for w in captured if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecations
    assert "DEFAULT_CORRECTION_PROMPTS" in str(deprecations[0].message)
    # La valeur retournée est cohérente : prompt FR.
    assert "Corrige" in value


def test_default_transcription_prompt_singular_emits_warning() -> None:
    """``BaseVLMAdapter.DEFAULT_TRANSCRIPTION_PROMPT`` (singulier)
    reste lisible mais émet ``DeprecationWarning``.
    """
    from picarones.adapters.vlm.base import BaseVLMAdapter

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        value = BaseVLMAdapter.DEFAULT_TRANSCRIPTION_PROMPT
    deprecations = [
        w for w in captured if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecations
    assert "DEFAULT_TRANSCRIPTION_PROMPTS" in str(deprecations[0].message)
    assert "Transcris" in value
