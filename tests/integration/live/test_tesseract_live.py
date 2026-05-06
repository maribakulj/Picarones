"""Test live TesseractAdapter (skip si binaire absent)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("pytesseract")
pytest.importorskip("PIL")
if shutil.which("tesseract") is None:
    pytest.skip(
        "binaire tesseract absent du PATH — skip live test",
        allow_module_level=True,
    )

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from picarones.adapters.ocr import TesseractAdapter  # noqa: E402
from picarones.domain.artifacts import Artifact, ArtifactType  # noqa: E402
from picarones.pipeline.types import RunContext  # noqa: E402


@pytest.mark.live
def test_tesseract_reads_synthetic_text(tmp_path: Path) -> None:
    """Génère une image avec du texte clair et vérifie que
    Tesseract le retrouve."""
    # Image 400x100 avec "HELLO" en gros (police par défaut).
    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=48)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 20), "HELLO", fill="black", font=font)
    img_path = tmp_path / "synthetic.png"
    img.save(img_path)

    adapter = TesseractAdapter(lang="eng", expose_confidences=False)
    ctx = RunContext(
        document_id="d1", code_version="1.0", pipeline_name="live",
    )
    result = adapter.execute(
        inputs={
            ArtifactType.IMAGE: Artifact(
                id="d1:img", document_id="d1",
                type=ArtifactType.IMAGE, uri=str(img_path),
            ),
        },
        params={},
        context=ctx,
    )
    out_path = Path(result[ArtifactType.RAW_TEXT].uri)
    text = out_path.read_text(encoding="utf-8")
    # Tesseract a au moins capté un caractère raisonnable —
    # on n'assertera pas l'exactitude (police par défaut peut
    # produire des résultats variables) mais on veut du non-vide.
    assert len(text) > 0, "Tesseract a retourné un texte vide"
