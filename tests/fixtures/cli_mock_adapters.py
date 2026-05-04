"""Mock adapters utilisés par les tests CLI S24.

Ces classes implémentent l'interface ``StepExecutor`` minimale
attendue par ``PipelineExecutor`` (S7) et ``BenchmarkService`` (S17).
Importables via dotted path :

::

    tests.fixtures.cli_mock_adapters.MockTextOCR

— exactement le format ``adapter_class`` du ``RunSpec`` (S24).
"""

from __future__ import annotations

from pathlib import Path

from picarones.domain.artifacts import Artifact, ArtifactType


class MockTextOCR:
    """OCR mock : copie le texte GT dans un fichier temp et produit un
    Artifact RAW_TEXT pointant dessus.

    Construit son output en lisant le ``image_uri`` du document, qu'on
    suppose pointer vers une image dont le stem permet de retrouver la
    GT (``foo.png`` → ``foo.gt.txt`` dans le même dossier).  C'est une
    convention du fixture de test, pas du domain.
    """

    name = "mock_text_ocr"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def __init__(self, copy_gt: bool = True) -> None:
        # ``copy_gt=True`` : copie la GT dans la sortie (CER nul).
        # ``copy_gt=False`` : produit du texte vide (cas dégénéré).
        self.copy_gt = copy_gt

    def execute(self, inputs, params, context):
        image_artifact = inputs[ArtifactType.IMAGE]
        image_path = Path(image_artifact.uri)
        # Convention test : la GT vit à <stem>.gt.txt dans le même
        # répertoire que l'image.
        # On retire l'extension image (.png/.jpg/.tif…) pour trouver
        # le stem.
        stem = image_path.stem  # "foo" pour "foo.png"
        gt_path = image_path.parent / f"{stem}.gt.txt"

        out_dir = image_path.parent / "_mock_ocr_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{context.document_id}_text.txt"
        if self.copy_gt and gt_path.exists():
            out_path.write_text(
                gt_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        else:
            out_path.write_text("", encoding="utf-8")

        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:mock_text_ocr:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
                uri=str(out_path),
            ),
        }


class MockBrokenOCR:
    """OCR mock qui lève systématiquement.

    Permet de tester la propagation d'erreurs dans le runner sans
    dépendance externe.
    """

    name = "mock_broken_ocr"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "io"

    def execute(self, inputs, params, context):
        raise RuntimeError("MockBrokenOCR : échec simulé.")


__all__ = ["MockBrokenOCR", "MockTextOCR"]
