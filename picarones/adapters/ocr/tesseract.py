"""``TesseractAdapter`` — adapter natif pour Tesseract 5.

Implémente le contrat ``BaseOCRAdapter`` (couche 5) :
``execute(inputs, params, context) → dict[ArtifactType, Artifact]``.

Cas d'usage BnF
---------------
Tesseract 5 reste l'OCR open-source de référence pour les corpus
imprimés et certains manuscrits réguliers.  L'adapter est CPU-bound
(Tesseract appelle une lib C en sous-process) — déclaré
``execution_mode="cpu"`` pour que le runner utilise un
``ProcessPoolExecutor``.

Configuration
-------------
Constructeur :

- ``name`` (défaut ``"tesseract"``) : identifiant de l'instance.
  Sert de suffixe au fichier de sortie ``<stem>.<name>.txt`` —
  permet de coexister avec plusieurs configurations Tesseract dans
  un même benchmark.
- ``lang`` (défaut ``"fra"``) : code langue Tesseract (``"fra"``,
  ``"lat"``, ``"eng"``, ``"fra+lat"``).
- ``psm`` (défaut ``6``) : Page Segmentation Mode (0-13).
- ``oem`` (défaut ``3``) : OCR Engine Mode.
- ``tesseract_cmd`` (défaut ``None``) : chemin vers l'exécutable
  ``tesseract`` si non standard.

Comportement
------------
1. Vérifie qu'un ``Artifact`` ``IMAGE`` est présent dans ``inputs``
   et qu'il porte une ``uri`` filesystem.
2. Lazy-import de ``pytesseract`` et ``PIL`` — si absent, lève
   ``OCRAdapterError`` avec message explicite.
3. Applique ``tesseract_cmd`` s'il est fourni.
4. Appelle ``pytesseract.image_to_string`` avec ``lang`` et
   ``--oem N --psm M``.
5. Écrit le texte dans ``<stem>.<name>.txt`` à côté de l'image
   (cohérent avec le pattern ``PrecomputedTextAdapter`` — un caller
   peut relire la sortie via cet adapter pour la comparer dans un
   second run).
6. Retourne un ``Artifact`` ``RAW_TEXT`` pointant vers le fichier
   produit.

Anti-sur-ingénierie
-------------------
- Pas de retry — Tesseract échoue rarement sur une image valide,
  et un appelant peut wrapper si besoin.
- Pas d'extraction de confidences pour l'instant : à ajouter
  quand un caller en aura besoin (un ``ConfidenceArtifact`` typé
  reste à définir).
- Pas de validation de l'encodage de l'image — Tesseract gère.
- Pas de support batch — un appel par image (le runner gère le
  parallélisme inter-documents).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from picarones.adapters.ocr.base import BaseOCRAdapter, OCRAdapterError
from picarones.adapters.output_paths import resolve_output_path
from picarones.domain.artifacts import Artifact, ArtifactType

#: Codes langue Tesseract acceptés : ISO 639-3 (3 lettres ASCII)
#: éventuellement combinés par ``+`` (ex. ``"fra+eng"``).  Le ``lang``
#: étant in fine passé à la ligne de commande Tesseract via
#: pytesseract, on refuse tout caractère qui pourrait être interprété
#: comme un flag ou un séparateur (espaces, ``--``, ``/``, etc.).
#: Phase 1.2 de l'audit code-quality (2026-05).
_TESSERACT_LANG_RE = re.compile(r"^[a-zA-Z]{3,}(?:\+[a-zA-Z]{3,})*$")


class TesseractAdapter(BaseOCRAdapter):
    """Adapter Tesseract 5 natif au nouveau contrat (S26).

    Parameters
    ----------
    name:
        Identifiant lisible de l'instance.  Défaut ``"tesseract"``.
        Doit être alphanumérique + ``_-`` (composant de nom de fichier).
    lang:
        Code langue Tesseract (``"fra"``, ``"lat"``, ``"eng"``, ...).
        Défaut ``"fra"``.
    psm:
        Page Segmentation Mode entre 0 et 13.  Défaut 6
        (single uniform block of text).
    oem:
        OCR Engine Mode (0-3).  Défaut 3 (LSTM, le plus précis).
    tesseract_cmd:
        Chemin custom vers l'exécutable ``tesseract``.  Défaut
        ``None`` (laisse pytesseract trouver l'installation système).
    timeout_seconds:
        Délai max (s) du sous-processus ``tesseract`` par image.
        Défaut ``120.0``.  **Garde-fou critique** : un appel
        ``pytesseract`` sans timeout bloque indéfiniment le thread
        worker si le binaire ``tesseract`` se fige sur une page
        (le ``CorpusRunner`` ne peut pas interrompre un sous-processus
        bloquant — cf. ``pipeline/runner.py``), ce qui gèle tout le
        run sans log ni erreur.  Avec un timeout, pytesseract tue le
        sous-processus et lève : le doc échoue proprement et le run
        continue.  ``0`` désactive explicitement (déconseillé).

    Raises
    ------
    OCRAdapterError
        Si le ``name`` ou les valeurs de ``psm`` / ``oem`` sont
        invalides.
    """

    input_types = frozenset({ArtifactType.IMAGE})
    #: Set **maximal** de types que l'adapter *peut* produire (capacité
    #: de la classe).  Seul ``RAW_TEXT`` est *garanti* : ``CONFIDENCES``
    #: et ``ALTO_XML`` sont best-effort/opt-in — cf.
    #: :pyattr:`effective_output_types` qui pilote la
    #: ``PipelineStep.output_types`` du benchmark mono-step.
    #:
    #: Phase B5 (mai 2026) — ``ALTO_XML`` ajouté au set maximal pour
    #: permettre la production d'un ALTO natif via
    #: ``pytesseract.image_to_alto_xml``.  Activé via le flag
    #: ``expose_alto`` (off par défaut, compat ascendante).
    output_types = frozenset({
        ArtifactType.RAW_TEXT,
        ArtifactType.CONFIDENCES,
        ArtifactType.ALTO_XML,
    })
    execution_mode = "cpu"

    @property
    def effective_output_types(self) -> frozenset[ArtifactType]:
        """Types *garantis* — pilote la ``PipelineStep.output_types``
        du benchmark mono-step (``_canonical_adapter_to_spec``).

        Seul ``RAW_TEXT`` est garanti.  Justification du périmètre :

        - ``CONFIDENCES`` est un sidecar **best-effort** :
          ``_extract_and_persist_confidences`` log + retourne ``None``
          si ``image_to_data`` lève/timeoute, sans invalider l'OCR.
          Surtout, **aucun consommateur côté pipeline** ne lit cet
          artefact (le hook calibration lit
          ``StepResult.token_confidences``, canal distinct que
          Tesseract n'alimente pas).  Le déclarer en sortie requise
          ferait échouer tout le step (``missing_output:
          ['confidences']``) au moindre timeout ``image_to_data`` —
          l'``engine_error`` qui en résulte fait sauter les hooks
          ``requires_success`` (confusion, char_scores, taxonomy,
          structure…) → bug « Analyse des caractères vide » alors que
          ``RAW_TEXT`` est parfaitement valide.
        - ``ALTO_XML`` n'est ajouté que si ``expose_alto`` est
          explicitement activé : c'est un opt-in assumé par
          l'appelant (consommé par une ``AltoView``), donc le
          déclarer comme requis est cohérent avec son intention.

        Conséquence : avec la config par défaut
        (``expose_alto=False``), le step OCR-seul exige uniquement
        ``RAW_TEXT`` — un échec d'extraction sidecar ne masque plus
        l'analyse caractères.
        """
        types = {ArtifactType.RAW_TEXT}
        if self._expose_alto:
            types.add(ArtifactType.ALTO_XML)
        return frozenset(types)

    def __init__(
        self,
        *,
        name: str = "tesseract",
        lang: str = "fra",
        psm: int = 6,
        oem: int = 3,
        tesseract_cmd: str | None = None,
        expose_confidences: bool = True,
        expose_alto: bool = False,
        timeout_seconds: float = 120.0,
    ) -> None:
        if not name or not name.strip():
            raise OCRAdapterError(
                "TesseractAdapter : name vide non autorisé.",
            )
        if not all(c.isalnum() or c in "_-" for c in name):
            raise OCRAdapterError(
                f"TesseractAdapter : name invalide {name!r} — "
                "alphanumérique + _ - uniquement.",
            )
        # Anti-injection ligne de commande Tesseract — refuse les
        # espaces, ``--user-words``, ``/``, etc.  ``lang`` est in fine
        # concaténé à ``tesseract -l <lang>``.
        if not _TESSERACT_LANG_RE.fullmatch(lang):
            raise OCRAdapterError(
                f"TesseractAdapter : lang invalide {lang!r} — "
                "format attendu : code ISO 639-3 (3+ lettres ASCII), "
                "optionnellement combiné via ``+`` (ex. ``fra+eng``).",
            )
        if not 0 <= psm <= 13:
            raise OCRAdapterError(
                f"TesseractAdapter : psm doit être ∈ [0, 13], reçu {psm}.",
            )
        if not 0 <= oem <= 3:
            raise OCRAdapterError(
                f"TesseractAdapter : oem doit être ∈ [0, 3], reçu {oem}.",
            )
        if timeout_seconds < 0:
            raise OCRAdapterError(
                "TesseractAdapter : timeout_seconds doit être >= 0 "
                f"(0 = désactivé), reçu {timeout_seconds}.",
            )
        self._name = name
        self._lang = lang
        self._psm = psm
        self._oem = oem
        self._tesseract_cmd = tesseract_cmd
        self._expose_confidences = expose_confidences
        self._expose_alto = expose_alto
        self._timeout = float(timeout_seconds)

    @property
    def name(self) -> str:
        return self._name

    @property
    def expose_confidences(self) -> bool:
        return self._expose_confidences

    @property
    def expose_alto(self) -> bool:
        return self._expose_alto

    @property
    def lang(self) -> str:
        return self._lang

    @property
    def psm(self) -> int:
        return self._psm

    @property
    def oem(self) -> int:
        return self._oem

    def execute(
        self,
        inputs: dict[ArtifactType, Artifact],
        params: dict[str, Any],
        context: Any,
    ) -> dict[ArtifactType, Artifact]:
        """Exécute Tesseract sur l'image fournie.

        Raises
        ------
        OCRAdapterError
            - input ``IMAGE`` absent ;
            - artefact image sans URI ;
            - fichier image introuvable ;
            - ``pytesseract`` ou ``PIL`` non installé ;
            - erreur Tesseract (lib system manquante, etc.).
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

        # Lazy-import de pytesseract + PIL — si absents, message
        # explicite plutôt qu'``ImportError`` au top-level.
        try:
            import pytesseract  # type: ignore[import-untyped]
            from PIL import Image
        except ImportError as exc:
            raise OCRAdapterError(
                f"{self.name} : pytesseract/Pillow non installés. "
                "Installer avec : pip install pytesseract pillow",
            ) from exc

        # Application du tesseract_cmd custom si fourni.
        if self._tesseract_cmd is not None:
            pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd

        # OCR.
        custom_config = f"--oem {self._oem} --psm {self._psm}"
        try:
            with Image.open(image_path) as image:
                text = pytesseract.image_to_string(
                    image,
                    lang=self._lang,
                    config=custom_config,
                    timeout=self._timeout,
                )
        except Exception as exc:
            raise OCRAdapterError(
                f"{self.name} : Tesseract a levé sur "
                f"{image_path!r} : {type(exc).__name__}: {exc}",
            ) from exc

        text = text.strip()

        # Le helper résout vers le workspace si fourni (sandbox par
        # doc), sinon écrit à côté de l'image — cohérent avec le
        # pattern ``PrecomputedTextAdapter`` qui peut relire la sortie.
        text_path = resolve_output_path(
            input_path=image_path,
            adapter_name=self.name,
            suffix="txt",
            context=context,
        )
        text_path.write_text(text, encoding="utf-8")

        outputs: dict = {
            ArtifactType.RAW_TEXT: Artifact(
                id=f"{context.document_id}:{self.name}:raw_text",
                document_id=context.document_id,
                type=ArtifactType.RAW_TEXT,
                produced_by_step="ocr",
                uri=str(text_path),
            ),
        }

        # Extraction des confidences via image_to_data (best-effort).
        # Si l'extraction échoue, on log et on saute — l'OCR reste
        # valide, seule la calibration est indisponible pour ce doc.
        if self._expose_confidences:
            confidences_artifact = self._extract_and_persist_confidences(
                image_path=image_path,
                text_path=text_path,
                pytesseract_module=pytesseract,
                pil_image_class=Image,
                custom_config=custom_config,
                document_id=context.document_id,
            )
            if confidences_artifact is not None:
                outputs[ArtifactType.CONFIDENCES] = confidences_artifact

        # Phase B5 — production ALTO XML natif (best-effort).
        # Tesseract sait nativement produire un ALTO 4 via
        # ``pytesseract.image_to_alto_xml``.  Désactivé par défaut
        # (compat ascendante : les pipelines existants ne s'attendent
        # pas à un artefact ALTO_XML).  Activer via le constructeur :
        #
        #     TesseractAdapter(expose_alto=True)
        #
        # ou en YAML (PipelineSpec) :
        #
        #     adapter_kwargs: {expose_alto: true}
        #     output_types: [raw_text, alto_xml]
        if self._expose_alto:
            alto_artifact = self._extract_and_persist_alto(
                image_path=image_path,
                text_path=text_path,
                pytesseract_module=pytesseract,
                pil_image_class=Image,
                custom_config=custom_config,
                document_id=context.document_id,
            )
            if alto_artifact is not None:
                outputs[ArtifactType.ALTO_XML] = alto_artifact

        return outputs

    def _extract_and_persist_confidences(
        self,
        *,
        image_path: Path,
        text_path: Path,
        pytesseract_module,
        pil_image_class,
        custom_config: str,
        document_id: str,
    ) -> Artifact | None:
        """Appelle ``image_to_data`` puis écrit le sidecar JSON.

        Retourne l'``Artifact CONFIDENCES`` ou ``None`` si l'extraction
        a échoué (warning loggé, OCR reste valide).
        """
        import logging
        logger = logging.getLogger(__name__)

        from picarones.adapters.ocr.confidences import (
            filter_valid_tokens,
            write_confidences_sidecar,
        )

        try:
            with pil_image_class.open(image_path) as image:
                data = pytesseract_module.image_to_data(
                    image,
                    lang=self._lang,
                    config=custom_config,
                    output_type=pytesseract_module.Output.DICT,
                    timeout=self._timeout,
                )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "[%s] image_to_data indisponible (%s) — calibration "
                "sautée pour ce document.", self._name, exc,
            )
            return None

        # Format Tesseract : dict {"text": [...], "conf": [...]}.
        texts = data.get("text") or []
        confs = data.get("conf") or []
        raw = [
            {"text": t, "confidence": c}
            for t, c in zip(texts, confs)
        ]
        tokens = filter_valid_tokens(raw)
        return write_confidences_sidecar(
            text_path=text_path,
            adapter_name=self._name,
            tokens=tokens,
            document_id=document_id,
            extractor="tesseract",
        )

    def _extract_and_persist_alto(
        self,
        *,
        image_path: Path,
        text_path: Path,
        pytesseract_module,
        pil_image_class,
        custom_config: str,
        document_id: str,
    ) -> Artifact | None:
        """Phase B5 — appelle ``image_to_alto_xml`` puis écrit
        ``<stem>.<name>.alto.xml`` à côté du fichier texte.

        Retourne l'``Artifact ALTO_XML`` ou ``None`` si l'extraction
        échoue ou si la sortie n'est pas un ALTO valide (warning
        loggé, OCR reste valide via ``RAW_TEXT``).

        Validation
        ----------
        L'ALTO produit est passé par ``safe_parse_xml`` (résistance
        XXE/billion-laughs) puis par ``parse_alto`` (vérifie qu'on a
        bien au moins une page + un bloc de texte).  Si la
        validation échoue, on log et on retourne ``None`` plutôt
        que de produire un artefact corrompu en aval.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Sortie attendue : str (ALTO XML 4).
        try:
            with pil_image_class.open(image_path) as image:
                alto_xml = pytesseract_module.image_to_alto_xml(
                    image,
                    lang=self._lang,
                    config=custom_config,
                    timeout=self._timeout,
                )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "[%s] image_to_alto_xml indisponible (%s) — ALTO "
                "sauté pour ce document.", self._name, exc,
            )
            return None

        # ``image_to_alto_xml`` retourne ``bytes`` selon la version de
        # pytesseract.  On normalise vers une string UTF-8.
        if isinstance(alto_xml, bytes):
            try:
                alto_xml = alto_xml.decode("utf-8")
            except UnicodeDecodeError as exc:
                logger.warning(
                    "[%s] ALTO Tesseract non-UTF-8 (%s) — ALTO sauté.",
                    self._name, exc,
                )
                return None

        if not alto_xml or not alto_xml.strip():
            logger.warning(
                "[%s] ALTO Tesseract vide — ALTO sauté.", self._name,
            )
            return None

        # Validation structurelle minimale (résistance XXE +
        # confirmation que c'est bien un ALTO parsable).
        # ``safe_parse_xml`` est volontairement tolérante : elle
        # retourne ``None`` au lieu de lever sur les XML invalides.
        # Pour rejeter proprement un ALTO mal formé, on traite ``None``
        # comme un échec de validation.
        try:
            from picarones.formats._xml_utils import safe_parse_xml
            parsed = safe_parse_xml(alto_xml.encode("utf-8"))
        except Exception as exc:  # noqa: BLE001 — XML mal formé
            logger.warning(
                "[%s] ALTO Tesseract mal formé (%s) — ALTO sauté.",
                self._name, exc,
            )
            return None
        if parsed is None:
            logger.warning(
                "[%s] ALTO Tesseract non-parsable (safe_parse_xml a "
                "retourné None) — ALTO sauté.", self._name,
            )
            return None

        # Persistance à côté du fichier texte (cohérent avec
        # ``write_confidences_sidecar`` : ``<stem>.<name>.alto.xml``).
        alto_path = text_path.with_suffix(".alto.xml")
        try:
            alto_path.write_text(alto_xml, encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "[%s] ALTO non persisté (%s) — ALTO sauté.",
                self._name, exc,
            )
            return None

        return Artifact(
            id=f"{document_id}:{self._name}:alto_xml",
            document_id=document_id,
            type=ArtifactType.ALTO_XML,
            produced_by_step="ocr",
            uri=str(alto_path),
        )


__all__ = ["TesseractAdapter"]
