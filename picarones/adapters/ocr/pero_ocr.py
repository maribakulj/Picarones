"""``PeroOCRAdapter`` natif — Sprint A14-S31.
implémente directement le contrat du nouveau monde, sans héritage du
parité sera atteinte sur tous les adapters.

Cas d'usage BnF
---------------
Pero OCR (Brno) est un moteur HTR open-source spécialisé pour les
documents historiques manuscrits.  Il produit une sortie structurée
PAGE XML — l'adapter natif extrait le texte plat dans l'ordre de
lecture naturel.  Adapter CPU-bound (PyTorch sur CPU + traitement
d'image) → ``execution_mode="cpu"`` pour ProcessPool.

Configuration
-------------
Constructeur :

- ``name`` (défaut ``"pero_ocr"``) : identifiant de l'instance.
- ``config_path`` : chemin obligatoire vers un fichier ``.ini`` de
  configuration Pero OCR (modèles, paramètres).  Sans ça, Pero OCR
  ne peut pas être instancié.

Comportement
------------
1. Vérifie la présence d'un ``Artifact`` ``IMAGE`` avec URI valide.
2. Lazy-import de ``pero_ocr`` + ``PIL`` + ``numpy`` — message
   explicite si absent.
3. Lazy-init du ``PageParser`` (une seule fois par instance).
4. Charge l'image en numpy array RGB, instancie un ``PageLayout``,
   appelle ``parser.process_page(image, page_layout)``.
5. Extrait le texte plat (``\n`` entre lignes, dans l'ordre des
   regions × lines).
6. Écrit le texte dans ``<stem>.<name>.txt`` à côté de l'image.
7. Retourne un ``Artifact`` ``RAW_TEXT``.

Anti-sur-ingénierie
-------------------
- Pas de support GPU explicite (Pero OCR le gère via la config).
- Pas de retry, pas d'extraction de confidences (à ajouter quand un caller en aura besoin).
- ``_parser`` lazy-init — si l'instance est sérialisée pour
  ProcessPool, le parser est re-instancié dans le worker (cohérent
  avec Pero OCR qui charge ses modèles à l'instanciation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from picarones.adapters.ocr.base import BaseOCRAdapter, OCRAdapterError
from picarones.adapters.output_paths import resolve_output_path
from picarones.domain.artifacts import Artifact, ArtifactType


class PeroOCRAdapter(BaseOCRAdapter):
    """Adapter Pero OCR natif au nouveau contrat (S26).

    Parameters
    ----------
    name:
        Identifiant lisible.  Défaut ``"pero_ocr"``.  Alphanum + ``_-``.
    config_path:
        Chemin vers le fichier ``.ini`` de configuration Pero OCR.
        Obligatoire — sans configuration, Pero OCR ne peut pas être
        instancié.

    Raises
    ------
    OCRAdapterError
        Si ``name`` ou ``config_path`` sont invalides au constructeur.
    """

    input_types = frozenset({ArtifactType.IMAGE})
    output_types = frozenset({ArtifactType.RAW_TEXT})
    execution_mode = "cpu"

    def __init__(
        self,
        *,
        config_path: str | Path,
        name: str = "pero_ocr",
    ) -> None:
        if not name or not name.strip():
            raise OCRAdapterError(
                "PeroOCRAdapter : name vide non autorisé.",
            )
        if not all(c.isalnum() or c in "_-" for c in name):
            raise OCRAdapterError(
                f"PeroOCRAdapter : name invalide {name!r} — "
                "alphanumérique + _ - uniquement.",
            )
        if not config_path:
            raise OCRAdapterError(
                "PeroOCRAdapter : config_path est requis (chemin .ini).",
            )
        self._name = name
        self._config_path = Path(config_path)
        # Le parser est instancié paresseusement au premier execute()
        # pour que la sérialisation ProcessPool fonctionne (un parser
        # contenant des modèles PyTorch n'est pas sérialisable).
        self._parser: Any = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def config_path(self) -> Path:
        return self._config_path

    def _get_parser(self) -> Any:
        """Instancie le PageParser au premier appel (lazy)."""
        if self._parser is not None:
            return self._parser

        try:
            from pero_ocr.document_ocr.page_parser import PageParser
        except ImportError as exc:
            raise OCRAdapterError(
                f"{self.name} : pero-ocr non installé. "
                "Installer avec : pip install pero-ocr",
            ) from exc

        if not self._config_path.exists():
            raise OCRAdapterError(
                f"{self.name} : config_path introuvable "
                f"{self._config_path!r}.",
            )

        import configparser
        parser_config = configparser.ConfigParser()
        parser_config.read(self._config_path)
        try:
            self._parser = PageParser(parser_config)
        except Exception as exc:
            raise OCRAdapterError(
                f"{self.name} : initialisation PageParser échouée "
                f"({type(exc).__name__}: {exc}).",
            ) from exc
        return self._parser

    def execute(
        self,
        inputs: dict[ArtifactType, Artifact],
        params: dict[str, Any],
        context: Any,
    ) -> dict[ArtifactType, Artifact]:
        """Exécute Pero OCR sur l'image fournie.

        Raises
        ------
        OCRAdapterError
            Si l'input est invalide, l'image introuvable, les
            dépendances manquantes, ou Pero OCR lève en interne.
        """
        if ArtifactType.IMAGE not in inputs:
            raise OCRAdapterError(
                f"{self.name} : input IMAGE manquant.",
            )
        image_artifact = inputs[ArtifactType.IMAGE]
        if image_artifact.uri is None:
            raise OCRAdapterError(
                f"{self.name} : artefact image "
                f"{image_artifact.id!r} sans URI.",
            )
        image_path = Path(image_artifact.uri)
        if not image_path.exists():
            raise OCRAdapterError(
                f"{self.name} : image introuvable {image_path!r}.",
            )

        try:
            import numpy as np
            from PIL import Image
            from pero_ocr.document_ocr.layout import PageLayout
        except ImportError as exc:
            raise OCRAdapterError(
                f"{self.name} : pero-ocr/numpy/Pillow non installés. "
                "Installer avec : pip install pero-ocr pillow numpy",
            ) from exc

        parser = self._get_parser()

        try:
            with Image.open(image_path) as pil_image:
                image_array = np.array(pil_image.convert("RGB"))
            page_layout = PageLayout(
                id=image_path.stem,
                page_size=(image_array.shape[0], image_array.shape[1]),
            )
            parser.process_page(image_array, page_layout)
        except Exception as exc:
            raise OCRAdapterError(
                f"{self.name} : Pero OCR a levé sur "
                f"{image_path!r} : {type(exc).__name__}: {exc}",
            ) from exc

        # Extraction du texte plat dans l'ordre regions × lines.
        lines: list[str] = []
        for region in page_layout.regions:
            for line in region.lines:
                if line.transcription:
                    lines.append(line.transcription.strip())
        text = "\n".join(lines)

        text_path = resolve_output_path(
            input_path=image_path,
            adapter_name=self.name,
            suffix="txt",
            context=context,
        )
        text_path.write_text(text, encoding="utf-8")

        return {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:{self.name}:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
                uri=str(text_path),
            ),
        }


__all__ = ["PeroOCRAdapter"]
