"""Sprint A14-S40 — PipelineSpec migré dans domain/.

Vérifie que :

1. ``picarones.domain.pipeline_spec`` est le module canonique.
2. ``picarones.domain`` re-exporte ``PipelineSpec``, ``PipelineStep``,
   ``INITIAL_STEP_ID``.
3. ``picarones.pipeline.spec`` continue d'exposer les mêmes classes
   (alias de chemin pour la rétrocompat).
4. Les deux chemins d'import retournent **la même classe**
   (``is`` strict, pas seulement ``==``).
"""

from __future__ import annotations


def test_canonical_path_in_domain() -> None:
    """``picarones.domain.pipeline_spec`` expose les classes canoniques."""
    from picarones.domain.pipeline_spec import (
        INITIAL_STEP_ID,
        PipelineSpec,
        PipelineStep,
    )
    assert PipelineSpec is not None
    assert PipelineStep is not None
    assert INITIAL_STEP_ID == "__initial__"


def test_domain_top_level_reexports() -> None:
    """``picarones.domain`` re-exporte au top-level."""
    from picarones.domain import (
        INITIAL_STEP_ID,
        PipelineSpec,
        PipelineStep,
    )
    assert PipelineSpec is not None
    assert PipelineStep is not None
    assert INITIAL_STEP_ID == "__initial__"


def test_legacy_pipeline_path_aliased() -> None:
    """``picarones.pipeline.spec`` reste un alias de chemin.

    Sprint S57 (audit #26) : émet désormais un ``DeprecationWarning``
    à l'import — vérifié dans ``test_legacy_pipeline_path_emits_warning``.
    """
    from picarones.pipeline.spec import (
        INITIAL_STEP_ID,
        PipelineSpec,
        PipelineStep,
    )
    assert PipelineSpec is not None
    assert PipelineStep is not None
    assert INITIAL_STEP_ID == "__initial__"


def test_legacy_pipeline_path_emits_warning() -> None:
    """Sprint S57 (audit #26) : l'import via ``picarones.pipeline.spec``
    émet un ``DeprecationWarning``.
    """
    import importlib
    import sys
    import warnings

    # Force le re-import pour déclencher le warning module-level.
    sys.modules.pop("picarones.pipeline.spec", None)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        importlib.import_module("picarones.pipeline.spec")
    deprecation = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert deprecation, "DeprecationWarning attendu sur l'import legacy."
    assert "picarones.domain" in str(deprecation[0].message)


def test_all_paths_resolve_to_same_classes() -> None:
    """Les imports depuis les 3 emplacements pointent vers le MÊME objet."""
    from picarones.domain import PipelineSpec as DomainSpec
    from picarones.domain import PipelineStep as DomainStep
    from picarones.domain import INITIAL_STEP_ID as DomainInitial
    from picarones.domain.pipeline_spec import PipelineSpec as CanonSpec
    from picarones.domain.pipeline_spec import PipelineStep as CanonStep
    from picarones.pipeline.spec import PipelineSpec as LegacySpec
    from picarones.pipeline.spec import PipelineStep as LegacyStep
    from picarones.pipeline.spec import INITIAL_STEP_ID as LegacyInitial

    # is strict — toutes les classes pointent vers le même objet.
    assert DomainSpec is CanonSpec
    assert DomainSpec is LegacySpec
    assert DomainStep is CanonStep
    assert DomainStep is LegacyStep
    assert DomainInitial == LegacyInitial


def test_pipeline_module_init_reexports_too() -> None:
    """``picarones.pipeline`` continue d'exposer pour rétrocompat."""
    from picarones.pipeline import PipelineSpec, PipelineStep, INITIAL_STEP_ID
    from picarones.domain.pipeline_spec import (
        PipelineSpec as CanonSpec,
        PipelineStep as CanonStep,
    )
    assert PipelineSpec is CanonSpec
    assert PipelineStep is CanonStep
    assert INITIAL_STEP_ID == "__initial__"
