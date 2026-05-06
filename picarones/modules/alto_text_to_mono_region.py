"""Reconstructeur ALTO de référence — texte plat → ALTO XML mono-région.

Chantier 1 du plan d'évolution post-Sprint 97.

Pourquoi ce module
------------------
Tout l'échafaudage de l'axe B (Sprints 32-34, 53-54, 63-68, 94-97) suppose
qu'un utilisateur peut brancher un ``BaseModule(input=TEXT, output=ALTO)``
dans une pipeline.  Aucun module réel de ce type n'existait jusqu'à ce
chantier — tous les tests utilisaient des ``MockModule``.

Ce reconstructeur est volontairement **primitif** :

- Une seule ``TextBlock`` couvre toute l'image source.
- Une ``TextLine`` est émise par ligne du texte d'entrée (split sur ``\\n``).
- Une ``String`` est émise par mot (split whitespace).
- Les ``HPOS``/``VPOS``/``WIDTH``/``HEIGHT`` sont distribués uniformément
  sur les dimensions de l'image source (lecture via Pillow si disponible,
  sinon valeurs par défaut documentées).

Cette baseline n'a pas vocation à être un bon reconstructeur — elle a
vocation à être un **point de comparaison stable**.  Un VLM produisant un
ALTO doit faire mieux qu'elle ; c'est mesurable via Layout F1
(:mod:`picarones.measurements.layout`) et via les métriques
``alto_text_cer``/``alto_text_wer`` (:mod:`picarones.measurements.alto_metrics`).

Conformité ALTO 4.2
-------------------
Le XML produit valide contre le schéma ALTO 4.2 (LOC) sur ses éléments
obligatoires.  Le namespace ``http://www.loc.gov/standards/alto/ns-v4#``
est déclaré explicitement.  La sortie est déterministe : deux appels
avec les mêmes entrées produisent le même XML octet par octet.

Exemple
-------
>>> from picarones.domain.artifacts import ArtifactType
>>> from picarones.modules import TextToAltoMonoRegion
>>> module = TextToAltoMonoRegion()
>>> outputs = module.process({
...     ArtifactType.IMAGE: "/path/to/page.png",
...     ArtifactType.TEXT: "Hello world\\nSecond line",
... })
>>> alto_xml = outputs[ArtifactType.ALTO]
>>> "<TextLine" in alto_xml and "Hello" in alto_xml
True
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
from xml.sax.saxutils import escape as _xml_escape

from picarones.domain.artifacts import ArtifactType
from picarones.domain.module_protocol import BaseModule, ExecutionMode

logger = logging.getLogger(__name__)


# Valeurs de repli quand l'image est introuvable ou Pillow indisponible.
# Documentées dans le docstring de classe pour traçabilité.
_DEFAULT_PAGE_WIDTH = 2000
_DEFAULT_PAGE_HEIGHT = 3000


def _read_image_size(image_path: str | Path) -> tuple[int, int]:
    """Retourne ``(width, height)`` en pixels, ou les valeurs par défaut.

    Pillow est une dépendance dure de Picarones (utilisée par les engines
    OCR) ; on la suppose disponible.  Si la lecture échoue (fichier
    manquant, format inconnu), on dégrade en valeurs par défaut + warning
    plutôt que de faire échouer le module — le rapport ALTO peut être
    inspecté visuellement même avec des bbox approximatives.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning(
            "[alto_text_to_mono_region] Pillow indisponible — "
            "dimensions ALTO par défaut %dx%d",
            _DEFAULT_PAGE_WIDTH, _DEFAULT_PAGE_HEIGHT,
        )
        return _DEFAULT_PAGE_WIDTH, _DEFAULT_PAGE_HEIGHT
    try:
        with Image.open(image_path) as img:
            return int(img.width), int(img.height)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[alto_text_to_mono_region] lecture %s impossible (%s) — "
            "dimensions par défaut %dx%d",
            image_path, exc, _DEFAULT_PAGE_WIDTH, _DEFAULT_PAGE_HEIGHT,
        )
        return _DEFAULT_PAGE_WIDTH, _DEFAULT_PAGE_HEIGHT


def _build_alto_xml(
    text: str,
    width: int,
    height: int,
    *,
    image_filename: str = "",
    measurement_unit: str = "pixel",
    processing_software: str = "picarones.modules.TextToAltoMonoRegion",
) -> str:
    """Construit un ALTO XML 4.2 mono-région à partir d'un texte plat.

    Cette fonction est volontairement **pure** (pas d'I/O, pas de side
    effect) : la complexité d'I/O (lecture image, écriture fichier) est
    laissée à l'appelant.  Cela rend la fonction trivialement testable.

    Distribution spatiale (mono-bloc)
    ---------------------------------
    - Le ``PrintSpace`` couvre l'image entière : (0, 0, width, height).
    - Le ``TextBlock`` aussi.
    - Les ``TextLine`` sont distribuées verticalement à pas constant :
      hauteur de ligne = ``height / max(1, n_lines)``.
    - Les ``String`` d'une ligne sont distribuées horizontalement par
      part proportionnelle à la longueur du mot (en caractères).

    Garde-fous
    ----------
    - ``text`` peut contenir tout caractère Unicode ; il est échappé pour
      le XML (``<``, ``>``, ``&``, ``"``).
    - Les lignes vides sont préservées comme ``TextLine`` sans ``String``
      (un blanc reste un blanc).
    - ``width`` et ``height`` doivent être > 0 ; sinon on remplace par
      les valeurs par défaut + warning (ne lève pas).
    """
    if width <= 0 or height <= 0:
        logger.warning(
            "[alto_text_to_mono_region] width/height invalides "
            "(%s, %s) — repli sur valeurs par défaut",
            width, height,
        )
        width, height = _DEFAULT_PAGE_WIDTH, _DEFAULT_PAGE_HEIGHT

    # Découpage du texte en lignes (préserve les lignes vides) puis en
    # mots ; on n'utilise que la séparation whitespace, pas un parser
    # linguistique — la baseline assume un texte déjà tokenisé.
    lines = text.split("\n") if text else []
    n_lines = max(1, len(lines))
    line_h = max(1, height // n_lines)

    parts: list[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        '<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4# '
        'http://www.loc.gov/standards/alto/v4/alto-4-2.xsd">'
    )
    parts.append("<Description>")
    parts.append(f"<MeasurementUnit>{_xml_escape(measurement_unit)}</MeasurementUnit>")
    parts.append("<sourceImageInformation>")
    parts.append(f"<fileName>{_xml_escape(image_filename)}</fileName>")
    parts.append("</sourceImageInformation>")
    parts.append("<OCRProcessing ID=\"OCR_1\">")
    parts.append("<ocrProcessingStep>")
    parts.append("<processingSoftware>")
    parts.append(
        f"<softwareName>{_xml_escape(processing_software)}</softwareName>"
    )
    parts.append("</processingSoftware>")
    parts.append("</ocrProcessingStep>")
    parts.append("</OCRProcessing>")
    parts.append("</Description>")
    parts.append("<Layout>")
    parts.append(
        f'<Page ID="page_1" PHYSICAL_IMG_NR="1" WIDTH="{width}" HEIGHT="{height}">'
    )
    parts.append(
        f'<PrintSpace ID="ps_1" HPOS="0" VPOS="0" WIDTH="{width}" HEIGHT="{height}">'
    )
    parts.append(
        f'<TextBlock ID="tb_1" HPOS="0" VPOS="0" WIDTH="{width}" HEIGHT="{height}">'
    )
    for li, line in enumerate(lines or [""]):
        line_y = li * line_h
        parts.append(
            f'<TextLine ID="tl_{li + 1}" HPOS="0" VPOS="{line_y}" '
            f'WIDTH="{width}" HEIGHT="{line_h}">'
        )
        words = line.split()
        if words:
            # Largeur proportionnelle au nombre de caractères par mot,
            # avec un minimum d'un pixel pour éviter les bbox dégénérées.
            total_chars = sum(len(w) for w in words) or 1
            cursor = 0
            for wi, word in enumerate(words):
                w_width = max(1, (len(word) * width) // total_chars)
                # Le dernier mot occupe le reste de la ligne pour
                # garantir somme(widths) = width (pas de drift visuel).
                if wi == len(words) - 1:
                    w_width = max(1, width - cursor)
                parts.append(
                    f'<String ID="tl_{li + 1}_s_{wi + 1}" '
                    f'HPOS="{cursor}" VPOS="{line_y}" '
                    f'WIDTH="{w_width}" HEIGHT="{line_h}" '
                    f'CONTENT="{_xml_escape(word, {chr(34): "&quot;"})}"/>'
                )
                if wi < len(words) - 1:
                    sp_width = max(1, width // (total_chars + 1))
                    parts.append(
                        f'<SP ID="tl_{li + 1}_sp_{wi + 1}" '
                        f'HPOS="{cursor + w_width}" VPOS="{line_y}" '
                        f'WIDTH="{sp_width}" HEIGHT="{line_h}"/>'
                    )
                    cursor = cursor + w_width + sp_width
                else:
                    cursor = cursor + w_width
        parts.append("</TextLine>")
    parts.append("</TextBlock>")
    parts.append("</PrintSpace>")
    parts.append("</Page>")
    parts.append("</Layout>")
    parts.append("</alto>")
    return "\n".join(parts)


class TextToAltoMonoRegion(BaseModule):
    """Reconstructeur ALTO de référence — TEXT (+ IMAGE) → ALTO mono-région.

    Module **baseline** : produit un ALTO XML 4.2 contenant une seule
    ``TextBlock`` qui couvre l'image source, avec une ``TextLine`` par
    ligne du texte d'entrée et une ``String`` par mot.

    Cette implémentation n'effectue **aucune segmentation visuelle** —
    elle distribue spatialement le texte selon sa structure linéaire
    (lignes par hauteur uniforme, mots par largeur proportionnelle au
    nombre de caractères).  C'est délibéré : un reconstructeur réel,
    pour battre cette baseline en Layout F1, doit apporter une vraie
    intelligence de segmentation.

    Conformité ``BaseModule``
    -------------------------
    - ``input_types = (ArtifactType.IMAGE, ArtifactType.TEXT)``
    - ``output_types = (ArtifactType.ALTO,)``
    - ``execution_mode = "cpu"`` (aucune I/O réseau, calcul local)

    Configuration
    -------------
    Dictionnaire optionnel passé au constructeur :

    - ``name`` (str) : nom affiché du module ; défaut
      ``"alto_text_to_mono_region"``.
    - ``measurement_unit`` (str) : unité ALTO ; défaut ``"pixel"``.
    - ``default_width`` / ``default_height`` (int) : dimensions ALTO
      utilisées quand l'image source est introuvable ; défauts 2000 et
      3000.
    """

    input_types = (ArtifactType.IMAGE, ArtifactType.TEXT)
    output_types = (ArtifactType.ALTO,)
    execution_mode: ExecutionMode = "cpu"

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config: dict = dict(config or {})

    @property
    def name(self) -> str:
        return self.config.get("name", "alto_text_to_mono_region")

    def metadata(self) -> dict:
        return {
            "module_kind": "alto_reconstructor",
            "variant": "mono_region_baseline",
            "deterministic": True,
            "schema": "ALTO 4.2",
        }

    def process(self, inputs: dict[ArtifactType, Any]) -> dict[ArtifactType, Any]:
        self.validate_inputs(inputs)
        image_payload = inputs[ArtifactType.IMAGE]
        text_payload = inputs[ArtifactType.TEXT]

        # ``image_payload`` peut être un chemin (str/Path) — convention
        # historique des engines — ou directement une paire de
        # dimensions ``(width, height)`` pour les usages headless.
        if isinstance(image_payload, tuple) and len(image_payload) == 2:
            width, height = int(image_payload[0]), int(image_payload[1])
            image_filename = ""
        else:
            image_path = Path(image_payload) if image_payload is not None else None
            if image_path is not None and image_path.exists():
                width, height = _read_image_size(image_path)
                image_filename = image_path.name
            else:
                width = int(self.config.get("default_width", _DEFAULT_PAGE_WIDTH))
                height = int(self.config.get("default_height", _DEFAULT_PAGE_HEIGHT))
                image_filename = image_path.name if image_path is not None else ""

        # Le texte peut être passé tel quel (str) ou enveloppé dans un
        # ``TextGT`` selon le contexte d'appel.  On accepte les deux
        # pour rendre l'intégration souple côté pipeline_runner.
        if hasattr(text_payload, "text"):
            text = str(text_payload.text)
        else:
            text = str(text_payload) if text_payload is not None else ""

        alto_xml = _build_alto_xml(
            text=text,
            width=width,
            height=height,
            image_filename=image_filename,
            measurement_unit=self.config.get("measurement_unit", "pixel"),
        )
        return {ArtifactType.ALTO: alto_xml}


__all__ = ["TextToAltoMonoRegion", "_build_alto_xml"]
