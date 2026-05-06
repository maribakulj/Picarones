"""``PipelineSpec`` vit en cercle 1 (``picarones.domain``).

Vérifie que :

1. ``picarones.domain.pipeline_spec`` est le module canonique.
2. ``picarones.domain`` re-exporte ``PipelineSpec``, ``PipelineStep``,
   ``INITIAL_STEP_ID`` au top-level.
3. ``picarones.pipeline`` re-exporte aussi (raccourci d'API publique).
4. Les chemins d'import retournent **la même classe** (``is`` strict).
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


def test_all_paths_resolve_to_same_classes() -> None:
    """Les imports valides retournent la MÊME classe (``is`` strict)."""
    from picarones.domain import (
        INITIAL_STEP_ID as DomainInitial,
    )
    from picarones.domain import (
        PipelineSpec as DomainSpec,
    )
    from picarones.domain import (
        PipelineStep as DomainStep,
    )
    from picarones.domain.pipeline_spec import (
        INITIAL_STEP_ID as CanonInitial,
    )
    from picarones.domain.pipeline_spec import (
        PipelineSpec as CanonSpec,
    )
    from picarones.domain.pipeline_spec import (
        PipelineStep as CanonStep,
    )
    from picarones.pipeline import (
        INITIAL_STEP_ID as PkgInitial,
    )
    from picarones.pipeline import (
        PipelineSpec as PkgSpec,
    )
    from picarones.pipeline import (
        PipelineStep as PkgStep,
    )

    assert DomainSpec is CanonSpec
    assert DomainSpec is PkgSpec
    assert DomainStep is CanonStep
    assert DomainStep is PkgStep
    assert DomainInitial == CanonInitial == PkgInitial


def test_legacy_spec_module_removed() -> None:
    """``picarones.pipeline.spec`` n'existe plus — chemin canonique
    unique via ``picarones.domain.pipeline_spec``.
    """
    import importlib

    try:
        importlib.import_module("picarones.pipeline.spec")
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(
            "picarones.pipeline.spec ne devrait plus exister — "
            "importer depuis picarones.domain.",
        )
