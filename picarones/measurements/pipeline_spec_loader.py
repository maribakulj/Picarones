"""Loader YAML pour spécifier des pipelines composées (Sprint 70).

Sprint 70 — Étape 4 / axe B du plan d'évolution 2026 : permet de
décrire une ``PipelineSpec`` (Sprint 63) ou une comparaison de N
pipelines (Sprint 65) dans un fichier **YAML déclaratif**, sans
écrire de code Python.

Philosophie inchangée
---------------------
Picarones reste un **banc d'essai**, pas un atelier de production.
Le YAML ne crée pas de modules — il **référence** des classes
``BaseModule`` que l'utilisateur a installées dans son environnement
(via ``pip install`` ou en plaçant le module dans le ``PYTHONPATH``).

Format YAML — pipeline simple
-----------------------------

.. code-block:: yaml

    name: ocr_then_correct
    steps:
      - name: ocr
        module: my_package.my_ocr.MyOCR
        args:
          tesseract_path: /usr/bin/tesseract
      - name: correct
        module: my_package.correctors.LLMCorrector
        args:
          model: gpt-4
        inputs_from:
          text: ocr

- ``name``         : nom de la pipeline (chaîne)
- ``steps``        : liste d'étapes
- ``steps[*].name`` : nom de l'étape (utilisé dans le rapport)
- ``steps[*].module`` : **chemin Python pointé** vers la classe
                        ``BaseModule`` à instancier
- ``steps[*].args``  : kwargs du constructeur (optionnel)
- ``steps[*].inputs_from`` : map ``{type: source_step}`` pour le
                             DAG branchant Sprint 66 (optionnel)

Format YAML — comparaison de N pipelines
-----------------------------------------

.. code-block:: yaml

    name: comparaison
    pipelines:
      - name: baseline
        steps: [...]
      - name: with_correcteur_a
        steps: [...]

Limites documentées
-------------------
- Les valeurs ``args`` doivent être sérialisables en YAML (str,
  int, float, bool, list, dict).  Pas de support pour des objets
  Python complexes en argument.
- L'import dynamique repose sur ``importlib.import_module`` ;
  la classe doit être accessible depuis l'environnement Python.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from picarones.core.modules import ArtifactType, BaseModule
from picarones.core.pipeline import PipelineSpec, PipelineStep

logger = logging.getLogger(__name__)


class PipelineSpecLoadError(ValueError):
    """Erreur levée lors du chargement d'une spec YAML invalide."""


def _resolve_class(dotted_path: str) -> type:
    """Importe et retourne la classe désignée par ``dotted_path``.

    Format attendu : ``"package.module.ClassName"``.
    """
    if not isinstance(dotted_path, str) or "." not in dotted_path:
        raise PipelineSpecLoadError(
            f"chemin Python invalide : {dotted_path!r} "
            f"(attendu : 'package.module.ClassName')"
        )
    module_path, _sep, class_name = dotted_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise PipelineSpecLoadError(
            f"module {module_path!r} introuvable : {exc}"
        ) from exc
    if not hasattr(module, class_name):
        raise PipelineSpecLoadError(
            f"classe {class_name!r} introuvable dans {module_path!r}"
        )
    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        raise PipelineSpecLoadError(
            f"{dotted_path!r} n'est pas une classe (type : {type(cls).__name__})"
        )
    return cls


def _instantiate_module(dotted_path: str, args: dict[str, Any]) -> BaseModule:
    """Instancie un ``BaseModule`` depuis son dotted path + kwargs."""
    cls = _resolve_class(dotted_path)
    if not issubclass(cls, BaseModule):
        raise PipelineSpecLoadError(
            f"{dotted_path!r} n'est pas une sous-classe de BaseModule"
        )
    try:
        instance = cls(**args)
    except TypeError as exc:
        raise PipelineSpecLoadError(
            f"impossible d'instancier {dotted_path!r} avec args={args!r} : {exc}"
        ) from exc
    return instance


def _parse_inputs_from(
    raw: Any, step_name: str,
) -> dict[ArtifactType, str]:
    """Parse le champ ``inputs_from`` d'un step YAML."""
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise PipelineSpecLoadError(
            f"étape {step_name!r} : ``inputs_from`` doit être un dict, "
            f"pas {type(raw).__name__}"
        )
    out: dict[ArtifactType, str] = {}
    for key, value in raw.items():
        try:
            at = ArtifactType(key)
        except ValueError as exc:
            raise PipelineSpecLoadError(
                f"étape {step_name!r} : type d'artefact inconnu "
                f"dans inputs_from : {key!r}"
            ) from exc
        if not isinstance(value, str) or not value:
            raise PipelineSpecLoadError(
                f"étape {step_name!r} : inputs_from[{key!r}] doit "
                f"être un nom d'étape (str non vide)"
            )
        out[at] = value
    return out


def _build_step(raw: dict, index: int) -> PipelineStep:
    if not isinstance(raw, dict):
        raise PipelineSpecLoadError(
            f"étape {index} : entrée doit être un dict YAML, "
            f"pas {type(raw).__name__}"
        )
    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise PipelineSpecLoadError(
            f"étape {index} : champ ``name`` requis (str)"
        )
    module_path = raw.get("module")
    if not module_path or not isinstance(module_path, str):
        raise PipelineSpecLoadError(
            f"étape {name!r} : champ ``module`` requis (dotted path Python)"
        )
    args = raw.get("args") or {}
    if not isinstance(args, dict):
        raise PipelineSpecLoadError(
            f"étape {name!r} : ``args`` doit être un dict, "
            f"pas {type(args).__name__}"
        )
    instance = _instantiate_module(module_path, args)
    inputs_from = _parse_inputs_from(raw.get("inputs_from"), name)
    return PipelineStep(
        name=name, module=instance, inputs_from=inputs_from,
    )


def load_pipeline_spec_from_dict(data: dict) -> PipelineSpec:
    """Construit une ``PipelineSpec`` depuis un dict (déjà parsé YAML).

    Utile pour les tests qui veulent sauter l'étape de parsing.
    """
    if not isinstance(data, dict):
        raise PipelineSpecLoadError(
            f"document YAML doit être un mapping, pas {type(data).__name__}"
        )
    name = data.get("name")
    if not name or not isinstance(name, str):
        raise PipelineSpecLoadError(
            "champ ``name`` requis au niveau racine"
        )
    raw_steps = data.get("steps")
    if not raw_steps or not isinstance(raw_steps, list):
        raise PipelineSpecLoadError(
            "champ ``steps`` requis (liste non vide)"
        )
    steps = [_build_step(s, i) for i, s in enumerate(raw_steps)]
    return PipelineSpec(name=name, steps=steps)


def load_pipeline_spec_from_yaml(path: Path | str) -> PipelineSpec:
    """Charge un fichier YAML et construit la ``PipelineSpec``.

    Lève ``PipelineSpecLoadError`` si le fichier n'est pas trouvé,
    si le YAML est invalide, ou si la spec ne respecte pas le
    format attendu.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise PipelineSpecLoadError(
            "PyYAML requis pour charger une spec YAML "
            "(pip install pyyaml)"
        ) from exc
    p = Path(path)
    if not p.exists():
        raise PipelineSpecLoadError(f"fichier introuvable : {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PipelineSpecLoadError(f"YAML invalide : {exc}") from exc
    return load_pipeline_spec_from_dict(data)


def load_comparison_specs_from_dict(data: dict) -> list[PipelineSpec]:
    """Construit une liste de ``PipelineSpec`` depuis un dict
    contenant ``pipelines`` (comparaison Sprint 65)."""
    if not isinstance(data, dict):
        raise PipelineSpecLoadError(
            f"document YAML doit être un mapping, pas {type(data).__name__}"
        )
    raw_pipelines = data.get("pipelines")
    if not raw_pipelines or not isinstance(raw_pipelines, list):
        raise PipelineSpecLoadError(
            "champ ``pipelines`` requis (liste non vide)"
        )
    return [load_pipeline_spec_from_dict(p) for p in raw_pipelines]


def load_comparison_specs_from_yaml(
    path: Path | str,
) -> tuple[list[PipelineSpec], dict]:
    """Charge un fichier YAML décrivant une comparaison.

    Retourne un tuple ``(specs, extras)`` où ``extras`` est le
    dict YAML brut (utile pour récupérer ``baseline``,
    ``rankings``, etc. au niveau du document).
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise PipelineSpecLoadError(
            "PyYAML requis pour charger une spec YAML"
        ) from exc
    p = Path(path)
    if not p.exists():
        raise PipelineSpecLoadError(f"fichier introuvable : {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PipelineSpecLoadError(f"YAML invalide : {exc}") from exc
    return load_comparison_specs_from_dict(data), data


__all__ = [
    "PipelineSpecLoadError",
    "load_pipeline_spec_from_dict",
    "load_pipeline_spec_from_yaml",
    "load_comparison_specs_from_dict",
    "load_comparison_specs_from_yaml",
]
