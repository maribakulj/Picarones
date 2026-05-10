"""Sprint A14-S3 — interdire les imports par effet de bord dans les nouveaux packages.

Anti-pattern à proscrire : ``picarones/__init__.py`` importe
``picarones.measurements`` au top-level **uniquement** pour
déclencher l'enregistrement des métriques décorées par
``@register_metric``.  Conséquence : tout import du package
charge ~50 sous-modules, exige toutes leurs deps optionnelles, et
fait crasher l'installation minimale (cf. l'épisode ``defusedxml``
au S1).

Ce test garantit que les **nouveaux packages** (créés au S3) ne
reproduisent pas ce pattern.  Pour chaque nouvelle couche, on
mesure le set de modules chargés à l'import du sous-package.  Si
ce set contient des modules externes lourds (numpy, scipy,
fastapi, jinja2, jiwer, ...) **alors que le sous-package est
encore vide**, c'est qu'un ``__init__.py`` fait quelque chose de
suspect.

Note : ce test est volontairement permissif tant que les couches
sont vides — il vérifie surtout l'absence d'import par effet de
bord.  Un test plus strict viendra aux Sprints S5-S6 quand les
premiers contrats du domain seront en place.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


#: Couches du rewrite ciblé (cf. ``test_layer_dependencies.py``).
NEW_LAYERS: tuple[str, ...] = (
    "domain",
    "evaluation",
    "pipeline",
    "formats",
    "adapters",
    "app",
    "interfaces",
    "reports",
)


#: Modules dont l'import est trahi par un side-effect "magique".
#: Si l'un de ces modules est chargé alors qu'on importe juste
#: ``picarones.<layer>`` (qui devrait être un namespace quasi-vide
#: au S3), c'est qu'on a un problème.
SUSPECTED_SIDE_EFFECT_LOADS: frozenset[str] = frozenset({
    "numpy",
    "scipy",
    "jinja2",
    "fastapi",
    "starlette",
    "click",
    "uvicorn",
    "jiwer",
    "rapidfuzz",
    "lxml",
    "yaml",
    "PIL",
})


def _import_in_isolation(module_dotted: str) -> set[str]:
    """Importe ``module_dotted`` et retourne le set des modules
    externes (top-level) chargés **propres au sous-package** au
    passage.

    Subtilité : ``import picarones.<layer>`` déclenche d'abord
    ``import picarones`` (le parent), qui aujourd'hui charge
    ``picarones.measurements`` par effet de bord (cf.
    ``BACKLOG_POST_LIVRAISON.md`` §2.4 — sera supprimé au S20).
    Si on ne pré-charge pas ``picarones``, on impute au sous-package
    tout ce que le parent charge — faux positif.

    Stratégie : pré-charger ``picarones`` une fois pour stabiliser
    ``sys.modules``, puis purger uniquement le sous-package cible
    et mesurer le vrai delta.
    """
    # Pré-charger picarones pour stabiliser le baseline.
    importlib.import_module("picarones")

    # Purger uniquement le sous-package cible (et ses descendants).
    # Ne PAS purger picarones lui-même (impact sur d'autres tests).
    to_purge = [
        m for m in list(sys.modules)
        if m == module_dotted or m.startswith(module_dotted + ".")
    ]
    for m in to_purge:
        del sys.modules[m]

    before = set(sys.modules)
    importlib.import_module(module_dotted)
    after = set(sys.modules)

    # Top-level externes seulement (pas picarones.*, pas stdlib).
    stdlib_names = set(getattr(sys, "stdlib_module_names", ()))
    delta_top = {
        m.split(".")[0] for m in (after - before)
        if "." not in m
    }
    delta_top -= {m for m in delta_top if m.startswith("_")}
    delta_top -= stdlib_names
    delta_top -= {"picarones"}
    return delta_top


@pytest.mark.parametrize(
    "layer",
    NEW_LAYERS,
    ids=lambda x: f"layer-{x}",
)
def test_layer_import_is_side_effect_free(layer: str) -> None:
    """L'import du sous-package d'une nouvelle couche ne doit pas
    charger de lib externe lourde tant que la couche est vide.

    Ce test sera ré-évalué à chaque sprint qui ajoute du code dans
    une couche : à ce moment-là, on mettra à jour les attentes par
    couche (cf. ``EXTERNAL_ALLOWED`` dans
    ``test_layer_dependencies.py``).  Pour S3, toutes les couches
    sont vides → toutes leurs dépendances externes attendues sont
    vides aussi.
    """
    layer_dir = REPO_ROOT / "picarones" / layer
    if not layer_dir.exists():
        pytest.skip(f"Couche {layer} pas encore créée — skip.")

    # Compter les .py non-__init__ dans le sous-package (récursif).
    code_files = [
        p for p in layer_dir.rglob("*.py")
        if p.name != "__init__.py" and "__pycache__" not in p.parts
    ]
    if code_files:
        # Si la couche contient déjà du code, le test est moins
        # strict : on vérifie juste que ``__init__.py`` n'importe
        # rien d'extra par effet de bord.  Une vraie vérif viendra
        # avec des règles dédiées par couche aux Sprints S5+.
        pytest.skip(
            f"Couche {layer} contient déjà du code "
            f"({len(code_files)} fichiers) — règle stricte décalée."
        )

    loaded_externals = _import_in_isolation(f"picarones.{layer}")
    suspect = loaded_externals & SUSPECTED_SIDE_EFFECT_LOADS
    assert not suspect, (
        f"\nL'import de ``picarones.{layer}`` charge des modules externes "
        f"par effet de bord alors que la couche est encore vide :\n"
        f"  {sorted(suspect)}\n\n"
        "C'est l'anti-pattern qu'on cherche à éviter — un ``__init__.py`` "
        "qui fait des imports magiques pour 'amorcer' un registre.\n"
        "Solution : construire le registre explicitement dans un service "
        "(cf. ``picarones/app/services/registry_service.py`` au Sprint S20)."
    )


def test_no_dynamic_registry_trigger_in_new_layers() -> None:
    """Méta-test : aucun ``__init__.py`` du nouveau code ne contient
    le pattern ``import picarones.X as _trigger_...`` qu'on essaie
    de bannir."""
    bad_patterns = (
        "_trigger_metric",
        "_trigger_registration",
        "as _bootstrap",
    )
    offenders: list[str] = []
    for layer in NEW_LAYERS:
        layer_dir = REPO_ROOT / "picarones" / layer
        if not layer_dir.exists():
            continue
        for init_path in layer_dir.rglob("__init__.py"):
            text = init_path.read_text(encoding="utf-8")
            for pattern in bad_patterns:
                if pattern in text:
                    offenders.append(
                        f"{init_path.relative_to(REPO_ROOT)} contient "
                        f"le pattern interdit '{pattern}'"
                    )
    assert not offenders, (
        "\nPattern d'import par effet de bord détecté dans un nouveau "
        "``__init__.py`` :\n"
        + "\n".join(f"  - {o}" for o in offenders)
        + "\n\nLes registres se construisent explicitement dans un "
        "service (cf. ``picarones/evaluation/registry/__init__.py``)."
    )
