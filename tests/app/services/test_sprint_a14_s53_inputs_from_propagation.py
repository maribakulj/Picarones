"""Sprint A14-S53 — propagation inputs_from (fix audit #20).

Avant S53, le YAML loader S39 validait ``StepSpec.inputs_from`` mais
``RunOrchestrator._build_pipelines`` construisait le ``domain.PipelineStep``
sans propager le champ — la validation passait, l'exécution ne profitait
PAS du DAG branchant.  Faux positif de couverture (testé via round-trip
YAML mais pas bout-en-bout).
"""

from __future__ import annotations


from picarones.app.schemas.run_spec import (
    PipelineSpecYaml,
    RunSpec,
    StepSpec,
)
from picarones.app.services import RunOrchestrator
from picarones.domain.artifacts import ArtifactType


def test_orchestrator_propagates_inputs_from_to_pipeline_step(
    tmp_path,
) -> None:
    """Construit un RunSpec avec inputs_from, appelle la méthode
    interne _build_pipelines, vérifie que le PipelineStep produit
    porte bien le inputs_from."""
    spec = RunSpec(
        corpus_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        pipelines=(
            PipelineSpecYaml(
                name="dag",
                initial_inputs=(ArtifactType.IMAGE,),
                steps=(
                    StepSpec(
                        id="ocr_a",
                        adapter_class="my_pkg.A",
                        input_types=(ArtifactType.IMAGE,),
                        output_types=(ArtifactType.RAW_TEXT,),
                    ),
                    StepSpec(
                        id="corrector",
                        adapter_class="my_pkg.B",
                        input_types=(ArtifactType.RAW_TEXT,),
                        output_types=(ArtifactType.CORRECTED_TEXT,),
                        inputs_from={ArtifactType.RAW_TEXT: "ocr_a"},
                    ),
                ),
            ),
        ),
        views=("text_final",),
    )

    orch = RunOrchestrator(output_dir=tmp_path / "out")
    # ``_build_pipelines`` essaie de résoudre adapter_class via
    # importlib — comme my_pkg.A et my_pkg.B n'existent pas, on
    # patch la résolution pour ne tester QUE la propagation
    # inputs_from.
    from unittest.mock import MagicMock, patch
    with patch(
        "picarones.app.services.run_orchestrator.resolve_adapter_class",
        return_value=MagicMock,
    ):
        pipeline_specs, _resolver, _kwargs = orch._build_pipelines(spec)

    assert len(pipeline_specs) == 1
    ps = pipeline_specs[0]
    # Le step "corrector" doit porter inputs_from.
    corrector_step = next(s for s in ps.steps if s.id == "corrector")
    assert ArtifactType.RAW_TEXT in corrector_step.inputs_from
    assert corrector_step.inputs_from[ArtifactType.RAW_TEXT] == "ocr_a"


def test_step_without_inputs_from_yields_empty_dict(tmp_path) -> None:
    spec = RunSpec(
        corpus_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        pipelines=(
            PipelineSpecYaml(
                name="simple",
                initial_inputs=(ArtifactType.IMAGE,),
                steps=(
                    StepSpec(
                        id="ocr",
                        adapter_class="my_pkg.A",
                        input_types=(ArtifactType.IMAGE,),
                        output_types=(ArtifactType.RAW_TEXT,),
                    ),
                ),
            ),
        ),
        views=("text_final",),
    )
    orch = RunOrchestrator(output_dir=tmp_path / "out")
    from unittest.mock import MagicMock, patch
    with patch(
        "picarones.app.services.run_orchestrator.resolve_adapter_class",
        return_value=MagicMock,
    ):
        pipeline_specs, _, _ = orch._build_pipelines(spec)
    assert pipeline_specs[0].steps[0].inputs_from == {}
