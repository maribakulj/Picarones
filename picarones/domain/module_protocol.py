"""``BaseModule`` — interface générique d'un module exécutable.

Un module est une **fonction typée** d'artefacts vers artefacts.  Il
déclare ce qu'il consomme (``input_types``) et ce qu'il produit
(``output_types``), et expose une méthode ``process`` qui prend un
dictionnaire d'entrées et retourne un dictionnaire de sorties.

Usage minimal ::

    class UpperCaseModule(BaseModule):
        input_types = (ArtifactType.RAW_TEXT,)
        output_types = (ArtifactType.RAW_TEXT,)
        execution_mode = "cpu"

        @property
        def name(self) -> str:
            return "uppercase"

        def process(self, inputs):
            txt = inputs[ArtifactType.RAW_TEXT]
            return {ArtifactType.RAW_TEXT: txt.upper()}

Ce module canonique (Phase 4-bis du retrait du legacy) est le
remplacement de ``picarones.core.modules.BaseModule``.  Le shim
legacy ``core/modules.py`` le ré-exporte pour la rétrocompat des
~25 callers (engines, measurements, modules officiels, cli, web,
report) qui le consomment.

Le rewrite a aussi des protocols spécialisés
(``BaseOCRAdapter``, ``BaseLLMAdapter``, ``BaseVLMAdapter`` dans
``picarones.adapters``) qui sont des cas particuliers de
``BaseModule`` typés pour leur domaine.  ``BaseModule`` reste le
contrat **générique** pour les modules contribués par des tiers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from picarones.domain.artifacts import ArtifactType

ExecutionMode = Literal["io", "cpu"]


class BaseModule(ABC):
    """Interface générique pour tout module exécutable par le runner.

    Un module est une fonction typée d'artefacts vers artefacts.  Il
    déclare ce qu'il consomme et ce qu'il produit, et expose une
    méthode ``process`` qui prend un dictionnaire d'entrées et
    retourne un dictionnaire de sorties.

    Attributs de classe (à surcharger en sous-classe)
    -------------------------------------------------
    input_types : tuple[ArtifactType, ...]
        Types d'artefacts consommés par ``process``.  L'ordre n'a
        pas de signification ; le runner passe un dictionnaire.
    output_types : tuple[ArtifactType, ...]
        Types d'artefacts produits par ``process``.  Tous les types
        listés doivent être présents dans le dict retourné par
        ``process`` (le runner valide).
    execution_mode : ``"io"`` ou ``"cpu"``
        Indique au runner quel exécuteur utiliser :
        ``ThreadPoolExecutor`` pour les modules I/O-bound (API,
        réseau), ``ProcessPoolExecutor`` pour les CPU-bound
        (Tesseract, Pero).
    """

    input_types: tuple[ArtifactType, ...] = ()
    output_types: tuple[ArtifactType, ...] = ()
    execution_mode: ExecutionMode = "io"

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifiant unique et stable du module."""

    @abstractmethod
    def process(
        self,
        inputs: dict[ArtifactType, Any],
    ) -> dict[ArtifactType, Any]:
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
            Dictionnaire des sorties produites.  Tous les types
            déclarés dans ``output_types`` doivent être présents.
        """

    def metadata(self) -> dict:
        """Métadonnées libres exposées par le module.

        Sous-classes peuvent surcharger pour exposer la version, la
        license, la citation académique, etc.  Le runner inclut ce
        dict dans le résultat afin que le rapport puisse l'afficher.
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
                f"{[t.value for t in self.input_types]})",
            )

    def validate_outputs(self, outputs: dict[ArtifactType, Any]) -> None:
        """Lève ``ValueError`` si un type de sortie déclaré est manquant."""
        missing = [t for t in self.output_types if t not in outputs]
        if missing:
            raise ValueError(
                f"Module {self.name!r} : sorties manquantes "
                f"{[t.value for t in missing]} (déclarées : "
                f"{[t.value for t in self.output_types]})",
            )

    def __repr__(self) -> str:
        ins = ",".join(t.value for t in self.input_types) or "·"
        outs = ",".join(t.value for t in self.output_types) or "·"
        return f"{self.__class__.__name__}(name={self.name!r}, {ins}→{outs})"


__all__ = ["ArtifactType", "BaseModule", "ExecutionMode"]
