"""Interface module générique (Sprint 33 — Phase 0.2 du plan d'évolution).

Pourquoi ce module
------------------
Aujourd'hui ``BaseOCREngine`` (`picarones/engines/base.py`) est typé
implicitement « image → texte » par sa signature.  Cette assomption
empêche d'évaluer dans le même runner :

- un mappeur VLM → ALTO (image → texte + ALTO),
- un rewriter ALTO post-correction (ALTO → ALTO),
- un module NER (texte → entités),
- un reconstructeur de structure (image + texte → ALTO).

``BaseModule`` est l'interface générique dont ``BaseOCREngine`` devient
un cas particulier.  Un module déclare explicitement les types
d'artefacts qu'il **consomme** (``input_types``) et qu'il **produit**
(``output_types``).  Le runner peut alors composer plusieurs modules en
une pipeline (cf. axe B du plan d'évolution).

Rétrocompatibilité
------------------
Aucun adaptateur OCR existant n'est touché par ce sprint.  La méthode
``BaseModule.process`` est implémentée par défaut sur ``BaseOCREngine``
de manière à wrapper l'ancien ``_run_ocr`` — toutes les sous-classes
historiques (Tesseract, Pero OCR, Mistral OCR, Google Vision,
Azure Document Intelligence) continuent à fonctionner sans modification.

Convention sur ``ArtifactType``
-------------------------------
Les valeurs string de ``ArtifactType`` sont volontairement les mêmes que
celles de ``GTLevel`` (Sprint 32) sauf pour ``IMAGE`` qui n'a pas de
correspondance GT.  La conversion entre les deux se fait trivialement
via ``.value`` :

>>> from picarones.core.corpus import GTLevel
>>> from picarones.core.modules import ArtifactType
>>> ArtifactType(GTLevel.TEXT.value) is ArtifactType.TEXT
True
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal


class ArtifactType(str, Enum):
    """Type d'artefact transitant entre modules d'une pipeline.

    Inclut le ``IMAGE`` (entrée typique d'un OCR) et tous les niveaux
    de ``GTLevel`` (Sprint 32) qui peuvent être produits ou consommés
    par un module.
    """

    IMAGE = "image"
    TEXT = "text"
    ALTO = "alto"
    PAGE = "page"
    ENTITIES = "entities"
    READING_ORDER = "reading_order"


ExecutionMode = Literal["io", "cpu"]


class BaseModule(ABC):
    """Interface générique pour tout module exécutable par le runner.

    Un module est une fonction typée d'artefacts vers artefacts.  Il
    déclare ce qu'il consomme et ce qu'il produit, et expose une méthode
    ``process`` qui prend un dictionnaire d'entrées et retourne un
    dictionnaire de sorties.

    Attributs de classe (à surcharger en sous-classe)
    -------------------------------------------------
    input_types : tuple[ArtifactType, ...]
        Types d'artefacts consommés par ``process``.  L'ordre n'a pas de
        signification ; le runner passe un dictionnaire.
    output_types : tuple[ArtifactType, ...]
        Types d'artefacts produits par ``process``.  Tous les types
        listés doivent être présents dans le dict retourné par
        ``process`` (le runner valide).
    execution_mode : ``"io"`` ou ``"cpu"``
        Indique au runner quel exécuteur utiliser : ``ThreadPoolExecutor``
        pour les modules I/O-bound (API, réseau), ``ProcessPoolExecutor``
        pour les CPU-bound (Tesseract, Pero).

    Exemple minimal
    ---------------
    >>> class UpperCaseModule(BaseModule):
    ...     input_types = (ArtifactType.TEXT,)
    ...     output_types = (ArtifactType.TEXT,)
    ...     execution_mode = "cpu"
    ...
    ...     @property
    ...     def name(self) -> str:
    ...         return "uppercase"
    ...
    ...     def process(self, inputs):
    ...         return {ArtifactType.TEXT: inputs[ArtifactType.TEXT].upper()}
    >>> m = UpperCaseModule()
    >>> m.process({ArtifactType.TEXT: "hello"})
    {<ArtifactType.TEXT: 'text'>: 'HELLO'}
    """

    input_types: tuple[ArtifactType, ...] = ()
    output_types: tuple[ArtifactType, ...] = ()
    execution_mode: ExecutionMode = "io"

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifiant unique et stable du module."""

    @abstractmethod
    def process(self, inputs: dict[ArtifactType, Any]) -> dict[ArtifactType, Any]:
        """Exécute le module sur les artefacts d'entrée.

        Parameters
        ----------
        inputs:
            Dictionnaire ``{ArtifactType: payload}``.  Tous les types
            déclarés dans ``input_types`` doivent être présents
            (``validate_inputs`` peut être utilisé pour valider).

        Returns
        -------
        dict[ArtifactType, Any]
            Dictionnaire des sorties produites.  Tous les types déclarés
            dans ``output_types`` doivent être présents.
        """

    def metadata(self) -> dict:
        """Métadonnées libres exposées par le module.

        Sous-classes peuvent surcharger pour exposer la version, la
        license, la citation académique, etc.  Le runner inclut ce dict
        dans le résultat afin que le rapport puisse l'afficher.
        """
        return {}

    # ──────────────────────────────────────────────────────────────────
    # Helpers de validation utilisés par le runner et les tests
    # ──────────────────────────────────────────────────────────────────

    def validate_inputs(self, inputs: dict[ArtifactType, Any]) -> None:
        """Lève ``ValueError`` si un type d'entrée déclaré est manquant."""
        missing = [t for t in self.input_types if t not in inputs]
        if missing:
            raise ValueError(
                f"Module {self.name!r} : entrées manquantes "
                f"{[t.value for t in missing]} (attendues : "
                f"{[t.value for t in self.input_types]})"
            )

    def validate_outputs(self, outputs: dict[ArtifactType, Any]) -> None:
        """Lève ``ValueError`` si un type de sortie déclaré est manquant."""
        missing = [t for t in self.output_types if t not in outputs]
        if missing:
            raise ValueError(
                f"Module {self.name!r} : sorties manquantes "
                f"{[t.value for t in missing]} (déclarées : "
                f"{[t.value for t in self.output_types]})"
            )

    def __repr__(self) -> str:
        ins = ",".join(t.value for t in self.input_types) or "·"
        outs = ",".join(t.value for t in self.output_types) or "·"
        return f"{self.__class__.__name__}(name={self.name!r}, {ins}→{outs})"


__all__ = ["ArtifactType", "BaseModule", "ExecutionMode"]
