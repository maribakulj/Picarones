"""Interface abstraite commune à tous les adaptateurs moteurs OCR (legacy).

Phase 7.A — module relocalisé depuis ``picarones.engines.base``
vers ``picarones.adapters.legacy_engines.base``.  Le chemin legacy
reste disponible via un shim avec ``DeprecationWarning`` ;
suppression prévue en 2.0.

Cohabite avec ``picarones.adapters.ocr.base.BaseOCRAdapter``
(canonique, ``StepExecutor`` Protocol).  Convergence documentée
dans ``docs/migration/pipeline-convergence-plan.md``
(sub-phases 7.A-7.D, stratégie 4.B).

Refactor du chantier 1 (post-Sprint 97)
---------------------------------------
Les Sprints 47-51 ont fait surcharger ``run()`` par chacun des cinq
adaptateurs OCR pour exposer ``token_confidences`` ; cinq fois la même
structure (chronométrage + extraction native + parsing). Ce module
factorise ce pattern :

- ``_run_with_native(image_path) -> (text, native_response)`` : hook
  par lequel passe désormais ``run()``. Implémentation par défaut qui
  délègue à ``_run_ocr`` (rétrocompat avec les engines historiques et
  avec les engines de test qui n'implémentent que ``_run_ocr``).
- ``_extract_raw_confidences(native) -> list[dict] | None`` : hook
  optionnel à surcharger pour exposer les confidences. Défaut : ``None``.
- ``_normalize_token_confidences(raw)`` : helper commun (filtrage
  tokens vides / négatifs, détection automatique d'échelle 0-100 → 0-1).

Conséquence : la classe se charge seule du chronométrage, de la
gestion d'erreurs et du wrapping en ``EngineResult``. Aucun adaptateur
OCR n'a plus à surcharger ``run()``.

Compat ``BaseModule`` (Sprint 33)
---------------------------------
``process()`` continue de propager le texte sous
``{ArtifactType.TEXT: ...}``. Les ``token_confidences`` ne sont pas
des artefacts — elles vivent dans ``EngineResult`` et restent
accessibles via la propriété ``last_run_result`` après l'exécution.
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from picarones.domain.artifacts import ArtifactType
from picarones.domain.module_protocol import BaseModule

logger = logging.getLogger(__name__)


@dataclass
class EngineResult:
    """Résultat brut produit par un moteur OCR sur une image."""

    engine_name: str
    image_path: str
    text: str
    duration_seconds: float
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    # Sprint 42 — confidences au niveau token (optionnel).
    # Format attendu : liste de dicts ``{"token": str, "confidence": float}``
    # avec ``confidence`` ∈ [0, 1] (ou ∈ [0, 100], normalisé par le runner).
    # ``None`` si le moteur ne fournit pas ce signal — comportement par
    # défaut pour tous les adapters historiques.  Quand renseigné,
    # le runner alimente ``DocumentResult.calibration_metrics``.
    token_confidences: Optional[list[dict[str, Any]]] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def image_sha256(self) -> str:
        return hashlib.sha256(Path(self.image_path).read_bytes()).hexdigest()


class BaseOCREngine(BaseModule):
    """Classe de base dont héritent tous les adaptateurs OCR.

    Sprint 33 — Phase 0.2 : ``BaseOCREngine`` hérite de ``BaseModule`` afin
    que les moteurs OCR existants soient automatiquement utilisables comme
    nœuds d'une pipeline composée (axe B du plan d'évolution).

    Chantier 1 (post-Sprint 97) — factorisation du run() unifié
    ------------------------------------------------------------
    Les sous-classes implémentent **un** des deux contrats suivants :

    1. **Engine sans confidences** : surchargent uniquement ``_run_ocr``
       qui retourne le texte. ``run()`` retourne un ``EngineResult``
       avec ``token_confidences=None``.

    2. **Engine avec confidences natives** : surchargent
       ``_run_with_native`` (un seul appel API qui retourne texte +
       payload natif) et ``_extract_raw_confidences`` (parsing du
       payload natif vers le format runner). ``run()`` les invoque
       et propage les ``token_confidences`` dans le ``EngineResult``.

    Aucune sous-classe n'a plus besoin de surcharger ``run()``.

    Attribut de classe
    ------------------
    execution_mode : ``"io"`` (défaut) ou ``"cpu"``
        Indique au runner quel type d'exécuteur utiliser :
        - ``"io"``  → ``ThreadPoolExecutor``  (moteurs API / réseau)
        - ``"cpu"`` → ``ProcessPoolExecutor`` (moteurs CPU-intensifs : Tesseract, Pero, Kraken)
    """

    # Déclaration BaseModule — un OCR consomme une image et produit du texte.
    input_types = (ArtifactType.IMAGE,)
    output_types = (ArtifactType.TEXT,)
    execution_mode: str = "io"
    """``"io"`` pour ThreadPoolExecutor (défaut), ``"cpu"`` pour ProcessPoolExecutor."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config: dict = config or {}
        # Cache du dernier ``EngineResult`` produit par ``run()`` —
        # exposé via la propriété ``last_run_result`` pour permettre
        # à un orchestrateur (par exemple le pipeline_runner) de
        # consulter les ``token_confidences`` après ``process()``.
        self._last_run_result: Optional[EngineResult] = None

    # ``name`` reste abstrait via héritage de BaseModule (cf.
    # picarones.core.modules) — les sous-classes le surchargent en
    # ``@property`` comme dans BaseModule.

    @abstractmethod
    def version(self) -> str:
        """Retourne la version du moteur (ex : '5.3.0')."""

    @abstractmethod
    def _run_ocr(self, image_path: Path) -> str:
        """Exécute l'OCR et retourne le texte brut extrait.

        Contrat **historique** conservé par rétrocompat. Les
        adaptateurs qui veulent exposer leurs confidences natives
        surchargent en plus ``_run_with_native`` et
        ``_extract_raw_confidences`` (cf. docstring de classe).
        """

    # ──────────────────────────────────────────────────────────────────
    # Hooks pour confidences natives (Chantier 1)
    # ──────────────────────────────────────────────────────────────────

    def _run_with_native(self, image_path: Path) -> tuple[str, Any]:
        """Exécute l'OCR et retourne ``(text, native_response)``.

        Implémentation par défaut : délègue à ``_run_ocr`` et retourne
        ``(text, None)`` — comportement adapté aux engines qui
        n'exposent pas de confidences (ex. tests, moteurs basiques).

        Les adaptateurs avec confidences natives surchargent cette
        méthode pour effectuer un seul appel API qui produit à la
        fois le texte et la structure (dict JSON, page layout, etc.)
        à partir de laquelle ``_extract_raw_confidences`` extraira
        les paires (token, confidence).
        """
        return self._run_ocr(image_path), None

    def _extract_raw_confidences(
        self, native: Any,
    ) -> Optional[list[dict[str, Any]]]:
        """Parse ``native`` et retourne les paires ``(token, conf)``.

        Format attendu : liste de dicts ``{"token": str, "confidence":
        float}`` avec ``confidence`` ∈ [0, 1] **ou** ∈ [0, 100].
        ``_normalize_token_confidences`` détecte l'échelle et normalise.

        Retourne ``None`` quand ``native`` est ``None`` ou que la
        structure ne contient aucune confidence exploitable.

        Implémentation par défaut : ``None`` (pas de confidences).
        """
        return None

    @staticmethod
    def _normalize_token_confidences(
        raw: Optional[list[dict[str, Any]]],
    ) -> Optional[list[dict[str, Any]]]:
        """Filtre les confidences brutes (échelle native conservée).

        - Tokens vides ou ``None`` → écartés.
        - Confidences négatives (Tesseract met -1 pour les non-mots) → écartées.
        - Confidences non convertibles en float → écartées.

        L'échelle native des moteurs ([0, 100] pour Tesseract,
        [0, 1] pour les autres) est conservée. La normalisation finale
        au moment du calcul de calibration est faite dans
        :func:`picarones.measurements.builtin_hooks.calibration_from_engine_result`.

        Retourne ``None`` si aucune entrée n'est exploitable.
        """
        if not raw:
            return None
        cleaned: list[dict[str, Any]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            tok = entry.get("token")
            if not isinstance(tok, str):
                continue
            tok = tok.strip()
            if not tok:
                continue
            conf = entry.get("confidence")
            if conf is None:
                continue
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                continue
            if conf_val < 0:
                continue
            cleaned.append({"token": tok, "confidence": conf_val})
        return cleaned or None

    # ──────────────────────────────────────────────────────────────────
    # Implémentation BaseModule (Sprint 33)
    # ──────────────────────────────────────────────────────────────────

    def process(self, inputs: dict[ArtifactType, Any]) -> dict[ArtifactType, Any]:
        """Exécute le moteur OCR comme un module générique.

        Wrapper rétrocompatible : extrait le chemin image de ``inputs``,
        appelle ``run()``, et retourne la sortie sous forme de dictionnaire
        ``{ArtifactType.TEXT: text}``.  Les erreurs sont conservées dans
        le résultat (cf. ``EngineResult.error``) plutôt que de lever.
        Les ``token_confidences`` restent accessibles via
        ``self.last_run_result.token_confidences`` après l'appel.
        """
        self.validate_inputs(inputs)
        result = self.run(inputs[ArtifactType.IMAGE])
        return {ArtifactType.TEXT: result.text}

    def metadata(self) -> dict:
        """Expose la version du moteur dans les métadonnées du module."""
        return {"engine_version": self._safe_version()}

    @property
    def last_run_result(self) -> Optional[EngineResult]:
        """Dernier ``EngineResult`` produit par ``run()`` (ou ``None``).

        Utile pour récupérer ``token_confidences`` après un appel à
        ``process()`` (qui ne les expose pas dans le bag d'artefacts du
        pipeline_runner — les confidences ne sont pas un type
        d'artefact mais une métadonnée du calcul).
        """
        return self._last_run_result

    # ──────────────────────────────────────────────────────────────────
    # Point d'entrée unifié : run()
    # ──────────────────────────────────────────────────────────────────

    def run(self, image_path: str | Path) -> EngineResult:
        """Exécute l'OCR et retourne un ``EngineResult``.

        Pipeline interne :

        1. ``_run_with_native(image_path)`` → ``(text, native)``
           (par défaut : appelle ``_run_ocr`` et retourne ``(text, None)``).
        2. ``_extract_raw_confidences(native)`` → liste brute ou ``None``
           (par défaut : ``None``).
        3. ``_normalize_token_confidences(raw)`` → format runner Sprint 42
           ou ``None``.

        Toute exception levée par l'étape 1 est capturée et placée dans
        ``EngineResult.error`` ; le texte est alors ``""`` et les
        confidences ``None``. Les exceptions des étapes 2-3 sont
        capturées séparément en warning : on retourne le texte avec
        ``token_confidences=None`` plutôt que de faire échouer toute
        la mesure pour un défaut de calibration.
        """
        image_path = Path(image_path)
        start = time.perf_counter()
        text = ""
        error: Optional[str] = None
        token_confidences: Optional[list[dict[str, Any]]] = None
        try:
            text, native = self._run_with_native(image_path)
        except Exception as exc:  # noqa: BLE001
            text = ""
            error = str(exc)
            native = None
        if error is None:
            try:
                raw = self._extract_raw_confidences(native)
                token_confidences = self._normalize_token_confidences(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] extraction/normalisation des token_confidences "
                    "dégradée : %s",
                    self.name, exc,
                )
                token_confidences = None
        duration = time.perf_counter() - start
        result = EngineResult(
            engine_name=self.name,
            image_path=str(image_path),
            text=text,
            duration_seconds=round(duration, 4),
            error=error,
            metadata={"engine_version": self._safe_version()},
            token_confidences=token_confidences,
        )
        self._last_run_result = result
        return result

    def _safe_version(self) -> str:
        # Sprint 30 — log la stacktrace en DEBUG pour aider au diagnostic
        # quand un moteur retourne ``"unknown"`` (utilisateur qui se
        # demande pourquoi). Ne pollue pas l'output normal (INFO+).
        try:
            return self.version()
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).debug(
                "[%s._safe_version] retourne 'unknown' suite à %s: %s",
                self.__class__.__name__, type(exc).__name__, exc,
                exc_info=True,
            )
            return "unknown"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
