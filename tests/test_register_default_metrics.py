"""Phase 5.3 audit code-quality — ``picarones.register_default_metrics``
est exposée comme API publique testable, en remplacement du
side-effect import opaque qui était en tête de ``picarones/__init__.py``.

Avant la Phase 5.3, le ``__init__`` contenait :

.. code-block:: python

    import picarones.evaluation.metrics as _trigger_metric_registration

Côté caller, l'enregistrement des ~37 métriques + ~25 agrégateurs
corpus-level était implicite — pas de point d'entrée nommé, pas de
test direct.  Désormais :

- ``register_default_metrics()`` est exposée dans ``picarones.__all__``.
- L'auto-déclenchement en fin de ``__init__`` reste pour la
  rétrocompat (``from picarones import register_metric`` doit
  marcher hors-test sans setup explicite).
- L'opération est idempotente (appel multiple sans effet de bord
  grâce au cache ``sys.modules``).
"""

from __future__ import annotations


def test_register_default_metrics_is_public_api() -> None:
    """La fonction est exportée dans ``picarones.__all__``."""
    import picarones

    assert "register_default_metrics" in picarones.__all__
    assert callable(picarones.register_default_metrics)


def test_register_default_metrics_is_idempotent() -> None:
    """Plusieurs appels successifs n'ont pas d'effet de bord —
    le module ``picarones.evaluation.metrics`` est cache par
    ``sys.modules`` après le premier import.
    """
    import picarones
    from picarones.evaluation.metric_registry import all_metrics

    snapshot_before = len(all_metrics())
    picarones.register_default_metrics()
    picarones.register_default_metrics()
    picarones.register_default_metrics()
    snapshot_after = len(all_metrics())

    assert snapshot_before == snapshot_after, (
        f"register_default_metrics non idempotent : {snapshot_before} → "
        f"{snapshot_after} métriques.  Le cache sys.modules devrait "
        f"empêcher tout doublon."
    )


def test_auto_trigger_loads_evaluation_metrics_submodules() -> None:
    """Le simple ``import picarones`` doit charger les sous-modules
    listés dans ``picarones/evaluation/metrics/__init__.py`` (ce qui
    déclenche leurs ``@register_metric`` éventuels).

    Limitation connue (drift pré-existant identifié par l'audit
    code-quality, hors scope Phase 5.3) : tous les modules
    ``@register_metric`` ne sont pas dans le ``__init__``.
    L'enregistrement complet des ~37 métriques passe par
    ``builtin_hooks`` et ``philological_hooks`` qui sont eux-mêmes
    chargés à la demande par ``BenchmarkService``.  Ce test vérifie
    uniquement l'invariant minimal : les modules **explicitement**
    listés sont bien chargés.
    """
    import sys

    import picarones  # noqa: F401 — déclenche l'auto-import
    # ``picarones.evaluation.metrics`` doit être chargé.
    assert "picarones.evaluation.metrics" in sys.modules, (
        "Le module ``picarones.evaluation.metrics`` n'est pas dans "
        "``sys.modules`` après ``import picarones`` — "
        "``register_default_metrics`` ne déclenche plus l'import."
    )
    # Et au moins un sous-module concret listé dans son __init__.
    assert "picarones.evaluation.metrics.confusion" in sys.modules


def test_register_default_metrics_signature_takes_no_args() -> None:
    """Contrat d'API : appelable sans argument."""
    import inspect

    import picarones

    sig = inspect.signature(picarones.register_default_metrics)
    assert len(sig.parameters) == 0, (
        f"register_default_metrics doit être sans argument, signature : "
        f"{sig}"
    )
    assert sig.return_annotation in (None, type(None), "None"), (
        f"register_default_metrics doit retourner ``None``, "
        f"annotation actuelle : {sig.return_annotation}"
    )
