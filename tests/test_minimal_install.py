"""Sprint A14-S2 — A.I.0 P0 : ``import picarones`` doit marcher avec
seulement les dépendances obligatoires.

Avant ce sprint, l'import du package au top-level chaînait des
``import`` par effet de bord (cf. ``picarones/__init__.py:91`` :
``import picarones.measurements as _trigger_metric_registration``)
qui exigeaient au moment du chargement initial des modules
théoriquement optionnels.  Conséquence : un ``pip install picarones``
sur un environnement où, par exemple, ``defusedxml`` n'était pas
résolu (Python 3.13 alpha, mirrors PyPI partiels, etc.) faisait
crasher tout import du package — y compris ``from picarones import
Document`` qui n'a logiquement pas besoin d'XML.

Ce module vérifie deux invariants critiques :

1. **Import OK avec seulement les deps obligatoires** —
   l'API publique du Cercle 1 doit s'importer sans nécessiter
   ``[web]``, ``[ner]``, ``[stats]``, ``[pero]``, ``[hf]``, ``[llm]``,
   ``[ocr-cloud]``, ``[kraken]``.

2. **Les deps obligatoires sont effectivement déclarées** dans
   ``pyproject.toml`` (cohérence entre le code et la spec
   d'installation).

Note d'environnement : ce test ne crée pas un venv vierge en
sous-processus (trop coûteux pour la CI à chaque commit).  Il
vérifie ce qu'on peut vérifier dans le venv courant — la vraie
validation "venv neuf" est faite par la matrice CI (cf.
``.github/workflows/ci.yml``).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────
# 1. Smoke test de l'API publique
# ──────────────────────────────────────────────────────────────────────


PUBLIC_API_NAMES = (
    "Corpus",
    "Document",
    "GTLevel",
    "TextGT",
    "AltoGT",
    "PageGT",
    "EntitiesGT",
    "ReadingOrderGT",
    "load_corpus_from_directory",
    "ArtifactType",
    "BaseModule",
    "BenchmarkResult",
    "DocumentResult",
    "EngineReport",
    "MetricsResult",
    "aggregate_metrics",
    "DetectorRegistry",
    "Fact",
    "FactImportance",
    "FactType",
    "PipelineResult",
    "PipelineRunner",
    "PipelineSpec",
    "PipelineStep",
    "StepResult",
    "MetricSpec",
    "compute_at_junction",
    "register_metric",
    "select_metrics",
)


def test_import_picarones_exposes_public_api() -> None:
    """Tous les noms documentés dans le ``__all__`` du package
    racine doivent être effectivement importables."""
    import picarones

    for name in PUBLIC_API_NAMES:
        assert hasattr(picarones, name), (
            f"``picarones.{name}`` annoncé dans ``__all__`` mais absent "
            "du namespace au moment de l'import."
        )


def test_picarones_all_matches_imports() -> None:
    """``__all__`` ne doit pas mentir."""
    import picarones

    declared = set(picarones.__all__)
    expected = set(PUBLIC_API_NAMES) | {"__version__", "__author__"}
    missing = expected - declared
    assert not missing, (
        f"``__all__`` n'expose pas tous les noms attendus : {missing}"
    )


def test_version_is_set() -> None:
    """``picarones.__version__`` doit être une string non vide."""
    import picarones

    assert isinstance(picarones.__version__, str)
    assert picarones.__version__.strip() != ""


# ──────────────────────────────────────────────────────────────────────
# 2. Cohérence entre les imports top-level et pyproject.toml
# ──────────────────────────────────────────────────────────────────────


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_pyproject_dependencies() -> list[str]:
    """Liste des noms de package des deps obligatoires.

    Volontairement permissif : on garde uniquement le nom (avant
    ``>=``, ``==``, ``[``, etc.) puisque c'est ce qui permet
    ``importlib.util.find_spec``.  Les noms PyPI utilisent ``-``
    mais les modules importés utilisent ``_`` (et ce n'est pas
    toujours symétrique : ``Pillow`` → ``PIL``, ``pyyaml`` →
    ``yaml``).  On gère explicitement le mapping ci-dessous.
    """
    pyproject = _project_root() / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    # Parser TOML léger : on cible juste le bloc ``dependencies = [...]``
    # de [project].  Pour rester sans dépendance externe, on parse à la
    # main une fois la section trouvée.
    in_deps = False
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("dependencies"):
            in_deps = True
            continue
        if in_deps:
            if stripped.startswith("]"):
                break
            if stripped.startswith("#") or not stripped:
                continue
            # ``    "click>=8.1.0",``  →  ``click``
            raw = stripped.strip(",").strip().strip('"').strip("'")
            # Coupe à la première occurrence d'un opérateur de version
            # ou d'un crochet d'extra.
            for sep in (">=", "==", "<=", ">", "<", "~=", "[", ";"):
                idx = raw.find(sep)
                if idx >= 0:
                    raw = raw[:idx]
                    break
            raw = raw.strip()
            if raw:
                out.append(raw)
    return out


# Mapping nom PyPI → nom du module Python à importer.
# Source : https://packaging.python.org/en/latest/discussions/...
# Ne lister que les paires asymétriques.
_NAME_OVERRIDES: dict[str, str] = {
    "Pillow": "PIL",
    "pyyaml": "yaml",
    "PyYAML": "yaml",
    "python-multipart": "multipart",
    "pyaml": "yaml",
}


def _import_name(pypi_name: str) -> str:
    return _NAME_OVERRIDES.get(pypi_name, pypi_name.replace("-", "_"))


def test_required_deps_are_importable() -> None:
    """Toutes les deps déclarées dans ``[project.dependencies]`` doivent
    être effectivement installables/importables.  Garde-fou contre une
    typo ou un nom de package PyPI mal copié."""
    declared = _read_pyproject_dependencies()
    assert declared, (
        "Aucune dépendance obligatoire trouvée dans pyproject.toml — "
        "le parser maison s'est cassé sur le format actuel."
    )
    missing: list[tuple[str, str]] = []
    for pypi in declared:
        mod = _import_name(pypi)
        if importlib.util.find_spec(mod) is None:
            missing.append((pypi, mod))
    assert not missing, (
        "Deps obligatoires déclarées mais introuvables dans le venv "
        "courant.  En CI institutionnelle, c'est un échec dur — un "
        "``pip install picarones`` produit un package qui crashera à "
        f"l'import sur ces noms : {missing}.  Vérifier le mapping "
        "PyPI → module dans ``_NAME_OVERRIDES``."
    )


def test_top_level_externals_are_declared() -> None:
    """Tout package externe chargé par ``import picarones`` doit être
    listé dans ``[project.dependencies]``.

    Garde-fou contre le scénario opposé : on ajoute un ``import foo``
    quelque part dans ``picarones/__init__.py`` (ou dans un module
    chargé par effet de bord depuis ``__init__.py``) sans déclarer
    ``foo`` dans ``pyproject.toml``.  Sur un install propre, le
    package crash.
    """
    # Capture des modules chargés avant et après ``import picarones``.
    before = set(sys.modules)
    importlib.import_module("picarones")
    after = set(sys.modules)

    # On ne garde que les top-level (pas de ``foo.bar``) qui ne sont
    # pas des modules picarones et qui ne sont pas stdlib.
    stdlib_names = set(getattr(sys, "stdlib_module_names", ()))
    candidates = {
        m.split(".")[0] for m in (after - before)
        if "." not in m
    }
    candidates -= {m for m in candidates if m.startswith("_")}
    candidates -= stdlib_names
    candidates -= {"picarones"}
    # Modules implicitement amenés par d'autres déjà déclarés (ex :
    # rapidfuzz vient avec jiwer ; pydantic_core vient avec pydantic ;
    # cython_runtime vient avec rapidfuzz ; pyexpat est en stdlib mais
    # pas toujours dans stdlib_module_names selon la version).
    transitive_allowed = {
        "rapidfuzz",
        "cython_runtime",
        "pyexpat",
        "annotated_types",
        "pydantic",
        "pydantic_core",
        "typing_extensions",
        "typing_inspection",
        "annotated_doc",
        "tomli",  # TOML stdlib uniquement à partir de 3.11 (tomllib)
        "tomllib",
    }
    candidates -= transitive_allowed

    declared = {_import_name(d) for d in _read_pyproject_dependencies()}

    undeclared = candidates - declared
    assert not undeclared, (
        f"Modules externes chargés à ``import picarones`` mais non "
        f"déclarés dans ``[project.dependencies]`` : {sorted(undeclared)}.\n"
        "Soit ajouter ces deps à pyproject.toml, soit déplacer leur "
        "import en lazy load (à l'intérieur d'une fonction qui n'est "
        "pas appelée au top-level)."
    )


# ──────────────────────────────────────────────────────────────────────
# 3. Garde-fou : pas de crash silencieux sur deps optionnelles absentes
# ──────────────────────────────────────────────────────────────────────


def test_optional_deps_not_required_at_top_level() -> None:
    """Les modules dépendant de deps optionnelles doivent s'importer
    en mode dégradé silencieux quand ces deps manquent.

    Exemple : ``picarones.engines.tesseract`` ne doit pas crasher
    l'import si ``pytesseract`` n'est pas installé — il doit échouer
    plus tard, au moment du ``run()``.  Idem pour Pero, Mistral OCR,
    Google Vision, Azure DI.

    On vérifie ici que les modules existent et s'importent même
    quand on n'a pas les engines installés.
    """
    # Liste des modules engines qu'on doit pouvoir au moins charger
    # (pas exécuter) sans planter.
    optional_engine_modules = (
        "picarones.engines.tesseract",
        "picarones.engines.pero_ocr",
        "picarones.engines.mistral_ocr",
        "picarones.engines.google_vision",
        "picarones.engines.azure_doc_intel",
    )
    failed: list[tuple[str, str]] = []
    for mod_name in optional_engine_modules:
        try:
            importlib.import_module(mod_name)
        except ImportError as exc:
            failed.append((mod_name, str(exc)))
    assert not failed, (
        "Modules engines qui plantent à l'import simple — ils doivent "
        "tomber en mode dégradé (warning + fallback) plutôt que de "
        "lever ImportError au top-level.  C'est ce qui permet à un "
        f"installeur minimal d'utiliser le CLI : {failed}"
    )
