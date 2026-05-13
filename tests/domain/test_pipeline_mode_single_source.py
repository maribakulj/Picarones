"""Phase 7.1 audit code-quality (2026-05) — ``PipelineMode`` doit
avoir **une seule** définition canonique : :data:`picarones.domain.pipeline_spec.PipelineMode`.

Avant la Phase 7.1 :

- ``picarones.pipeline.llm_pipeline_config.OCRLLMMode`` :
  ``Literal["text_only", "text_and_image", "zero_shot"]``
- ``picarones.pipeline.llm_pipeline_builder.OCRLLMPipelineMode`` :
  idem (redéfini)
- ``picarones.interfaces.web.models.PipelineMode`` : idem (redéfini)

Trois définitions identiques mais indépendantes — risque concret de
divergence si l'une ajoute un mode (par exemple ``"hybrid"``) sans
mettre à jour les deux autres.

Refactor : ``domain/pipeline_spec.py`` (couche 1) accueille la
définition canonique ; les trois alias historiques (OCRLLMMode,
OCRLLMPipelineMode, PipelineMode) deviennent des re-exports pour
préserver les imports existants.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import get_args

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "picarones"


def test_pipeline_mode_canonical_definition_exists() -> None:
    """La définition canonique vit dans ``domain/pipeline_spec.py``."""
    from picarones.domain.pipeline_spec import PipelineMode

    args = set(get_args(PipelineMode))
    assert args == {"text_only", "text_and_image", "zero_shot"}, (
        f"PipelineMode canonique doit valoir exactement les 3 littéraux "
        f"canoniques.  Reçu : {args}"
    )


def test_pipeline_mode_aliases_are_identical() -> None:
    """Chaque alias historique pointe vers la même valeur Literal."""
    from picarones.domain.pipeline_spec import PipelineMode as canonical
    from picarones.interfaces.web.models import PipelineMode as web_alias
    from picarones.pipeline.llm_pipeline_builder import OCRLLMPipelineMode as builder_alias
    from picarones.pipeline.llm_pipeline_config import OCRLLMMode as config_alias

    canonical_args = set(get_args(canonical))
    for alias_name, alias in [
        ("interfaces.web.models.PipelineMode", web_alias),
        ("pipeline.llm_pipeline_builder.OCRLLMPipelineMode", builder_alias),
        ("pipeline.llm_pipeline_config.OCRLLMMode", config_alias),
    ]:
        alias_args = set(get_args(alias))
        assert alias_args == canonical_args, (
            f"Alias {alias_name} diverge du canonique : "
            f"{alias_args} != {canonical_args}.  Vérifier que l'alias "
            f"importe bien ``PipelineMode`` depuis ``domain/pipeline_spec``."
        )


def test_no_independent_literal_definition_in_codebase() -> None:
    """Aucun fichier du code source (hors ``domain/pipeline_spec``)
    ne doit redéfinir ``Literal["text_only", "text_and_image", "zero_shot"]``.

    Scan AST.  Refuse toute occurrence d'une annotation de type
    ``Literal[...]`` qui contient ces 3 chaînes — sauf dans le module
    canonique.
    """
    canonical_strings = frozenset({"text_only", "text_and_image", "zero_shot"})
    canonical_file = SOURCE / "domain" / "pipeline_spec.py"

    offenders: list[tuple[str, int]] = []
    for path in SOURCE.rglob("*.py"):
        if path == canonical_file or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # Cherche ``X = Literal["a", "b", ...]`` ou
            # ``foo: Literal["a", "b", ...]`` partout.
            if not isinstance(node, ast.Subscript):
                continue
            val = node.value
            if not (
                (isinstance(val, ast.Name) and val.id == "Literal")
                or (isinstance(val, ast.Attribute) and val.attr == "Literal")
            ):
                continue
            # Extraire les chaînes des arguments.
            slice_node = node.slice
            if isinstance(slice_node, ast.Tuple):
                elts = slice_node.elts
            else:
                elts = [slice_node]
            strings = {
                e.value for e in elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
            if canonical_strings.issubset(strings) and len(strings) == 3:
                rel = path.relative_to(REPO_ROOT).as_posix()
                offenders.append((rel, node.lineno))

    if offenders:
        sample = "\n".join(f"  {p}:{ln}" for p, ln in offenders)
        raise AssertionError(
            "Définition indépendante de ``PipelineMode`` "
            "(``Literal[\"text_only\", \"text_and_image\", \"zero_shot\"]``) "
            "détectée hors du module canonique :\n"
            + sample
            + "\n\nLa Phase 7.1 audit code-quality impose une source de vérité "
            "unique dans ``picarones/domain/pipeline_spec.py``.  Remplacer "
            "la définition locale par ``from picarones.domain.pipeline_spec "
            "import PipelineMode``."
        )


def test_pipeline_mode_in_domain_all() -> None:
    """``PipelineMode`` est exporté par ``picarones.domain.pipeline_spec``."""
    from picarones.domain import pipeline_spec

    assert "PipelineMode" in pipeline_spec.__all__
