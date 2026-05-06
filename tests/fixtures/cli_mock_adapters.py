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


class MockAltoOCR:
    """OCR structuré mock : produit ALTO_XML déterministe sur disque.

    Lit la GT texte (``<stem>.gt.txt`` à côté de l'image) et écrit un
    ALTO contenant exactement ce texte (1 page / 1 bloc / 1 ligne).
    Sert à tester la projection ALTO→texte bout-en-bout dans le CLI
    après le fix du protocole Projector au S25.
    """

    name = "mock_alto_ocr"
    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.ALTO_XML})
    execution_mode = "io"

    def execute(self, inputs, params, context):
        from picarones.formats.alto.types import (
            AltoBBox, AltoDocument, AltoLine, AltoPage, AltoString,
            AltoTextBlock,
        )
        from picarones.formats.alto.writer import write_alto

        image_artifact = inputs[ArtifactType.IMAGE]
        image_path = Path(image_artifact.uri)
        gt_path = image_path.parent / f"{image_path.stem}.gt.txt"
        text = (
            gt_path.read_text(encoding="utf-8") if gt_path.exists()
            else "fallback"
        )

        alto_doc = AltoDocument(pages=(AltoPage(blocks=(AltoTextBlock(lines=(AltoLine(strings=tuple(
            AltoString(content=w, bbox=AltoBBox(hpos=0, vpos=0, width=10, height=10))
            for w in text.split()
        )),),),),),),)

        out_dir = image_path.parent / "_mock_alto_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{context.document_id}.alto.xml"
        out_path.write_bytes(write_alto(alto_doc))

        return {
            ArtifactType.ALTO_XML: Artifact(
                id=f"{context.document_id}:mock_alto_ocr:alto",
                document_id=context.document_id,
                type=ArtifactType.ALTO_XML,
                produced_by_step="ocr",
                uri=str(out_path),
            ),
        }


__all__ = ["MockAltoOCR", "MockBrokenOCR", "MockTextOCR"]
