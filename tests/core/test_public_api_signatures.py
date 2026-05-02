"""Garde-fou contractuel sur les signatures de l'API publique de ``picarones``.

Sprint A1 (item m-9 de l'audit institutional-readiness-2026-05).

Le module ``tests/core/test_public_api.py`` vérifie déjà *quels* symboles
sont exportés. Ce module-ci verrouille en plus les **valeurs par défaut**
des paramètres des fonctions publiques. Sans ce verrou, un PR peut
silencieusement changer un défaut documenté (ex : ``corpus_lang="fr"``
qui devient ``corpus_lang="en"``) et casser la rétrocompatibilité de
tous les consommateurs externes — y compris des notebooks de chercheurs
pinés sur une version mineure.

Convention : pour ajouter un nouveau paramètre par défaut, mettre à jour
ce fichier ET la documentation publique (CHANGELOG + ``docs/api-stable.md``).
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

import picarones


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signature_defaults(callable_obj: Any) -> dict[str, Any]:
    """Retourne ``{nom_param: default_value}`` pour les paramètres avec défaut.

    Les paramètres sans défaut (positionnels obligatoires) sont omis.
    """
    sig = inspect.signature(callable_obj)
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }


# ---------------------------------------------------------------------------
# load_corpus_from_directory
# ---------------------------------------------------------------------------


def test_load_corpus_from_directory_defaults() -> None:
    """``load_corpus_from_directory`` est l'entrée canonique pour charger un
    corpus depuis un dossier. Ses défauts sont contractuels."""
    defaults = _signature_defaults(picarones.load_corpus_from_directory)

    # Ces clés DOIVENT exister. Si l'une est supprimée, c'est un breaking
    # change qui mérite un tag majeur.
    assert "name" in defaults, (
        "load_corpus_from_directory(name=…) doit avoir un défaut "
        "(actuellement on accepte None pour déduire du nom de dossier)."
    )

    # Le défaut historique de ``name`` est ``None`` (déduction depuis le
    # nom du dossier). Tout changement vers une chaîne fixe casserait les
    # appelants qui s'appuient sur cette déduction.
    assert defaults["name"] is None


# ---------------------------------------------------------------------------
# Symboles publics : pas d'arguments positionnels uniquement non-typés
# ---------------------------------------------------------------------------


def _is_public_callable(name: str) -> bool:
    """Filtre les symboles publics de ``picarones`` qui sont appelables."""
    if name.startswith("_"):
        return False
    obj = getattr(picarones, name, None)
    return callable(obj) and not isinstance(obj, type(picarones))


@pytest.mark.parametrize("symbol", [s for s in picarones.__all__ if _is_public_callable(s)])
def test_public_callable_has_typed_signature(symbol: str) -> None:
    """Toute fonction publique doit avoir des annotations de type.

    Ce garde-fou prépare le passage en strict mypy (Sprint A1, M-4).
    Les classes (Corpus, Document, etc.) sont exclues — leur ``__init__``
    est testé séparément si nécessaire, mais beaucoup sont des dataclasses
    déjà annotées par construction.
    """
    obj = getattr(picarones, symbol)
    if isinstance(obj, type):
        # Les classes sont validées via mypy strict sur core/, pas ici.
        return
    sig = inspect.signature(obj)
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        assert param.annotation is not inspect.Parameter.empty, (
            f"Paramètre `{param_name}` de `picarones.{symbol}` non annoté. "
            f"L'API publique exige un typage explicite (Sprint A1)."
        )


# ---------------------------------------------------------------------------
# compute_at_junction (registre typé Sprint 34)
# ---------------------------------------------------------------------------


def test_compute_at_junction_defaults() -> None:
    """``compute_at_junction`` est l'API consommée par les pipelines composées
    (Sprint 63+). Ses défauts contractuels :
    - ``metric_name`` n'a PAS de défaut (on doit toujours préciser la métrique).
    """
    defaults = _signature_defaults(picarones.compute_at_junction)
    assert "metric_name" not in defaults, (
        "compute_at_junction doit exiger metric_name explicite. "
        "Un défaut introduirait de l'ambiguïté sur la métrique calculée."
    )


# ---------------------------------------------------------------------------
# select_metrics (registre typé Sprint 34)
# ---------------------------------------------------------------------------


def test_select_metrics_signature() -> None:
    """``select_metrics(input_type, output_type)`` est purement positionnel
    sur ses deux types — pas de défauts implicites."""
    defaults = _signature_defaults(picarones.select_metrics)
    assert "input_type" not in defaults
    assert "output_type" not in defaults


# ---------------------------------------------------------------------------
# Méta-test : tout symbole de __all__ existe vraiment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", picarones.__all__)
def test_all_symbols_resolve(symbol: str) -> None:
    """Chaque entrée de ``__all__`` doit pouvoir être résolue."""
    assert hasattr(picarones, symbol), (
        f"`picarones.{symbol}` est dans __all__ mais n'est pas exporté."
    )
