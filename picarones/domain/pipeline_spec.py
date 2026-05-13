"""``PipelineStep`` et ``PipelineSpec`` — Sprints A14-S6 / S40.

Description **purement déclarative** d'un DAG de transformation
documentaire.  Sérialisable en YAML, versionnable en git, valide
sans avoir besoin d'instancier les modules concrets.

Sprint S40 — migration depuis ``picarones.pipeline.spec``
---------------------------------------------------------
Le module canonique est désormais en cercle 1 (``picarones/domain/``)
— c'est un type pur qui n'a aucune dépendance d'exécution
(``picarones/pipeline/`` qui contient le runtime n'est en fait pas
nécessaire pour décrire la spec).  ``picarones.pipeline.spec`` reste
exposé en re-export pour ne pas casser les callers existants — ce
n'est pas un shim au sens architectural (adaptation d'une API
incompatible) mais un alias de chemin.

Différence avec ``picarones.evaluation.pipeline`` (Sprint 63)
-------------------------------------------------------------
``PipelineStep`` legacy (relocalisé en ``picarones.evaluation.pipeline``)
porte un champ ``module: BaseModule``
— une **instance** d'objet exécutable.  Conséquence : la spec
n'était pas sérialisable en YAML, et un test qui voulait juste
valider la cohérence des types devait instancier des stubs.

Ici, ``PipelineStep`` ne porte qu'un ``adapter_name: str``.  Le
mapping ``nom → instance`` est maintenu par un service applicatif
(``picarones.app.services.adapter_registry`` au S19) et résolu au
moment de l'exécution, pas de la spec.

Bénéfices :

- Le YAML d'une pipeline composée est versionnable en git
  indépendamment de l'environnement Python (BnF peut commit
  ``ocr_llm_alto_remap.yaml`` sans imposer aux contributeurs
  d'avoir tous les SDK installés).
- ``validate_spec`` peut s'exécuter sans instancier aucun module
  → tests rapides et déterministes.
- Le rapport de reproductibilité peut citer le YAML exact, le
  commit du code et la version des adapters utilisés —
  séparation propre de la déclaration et de l'implémentation.

Anti-sur-ingénierie
-------------------
- Pas de typage des ``params`` par adapter ici (chaque adapter
  validera ses propres params au moment de l'exécution).
- Pas de versioning de spec — un nouveau champ se traduit par un
  rebump pydantic.  Si on veut migrer entre versions de schéma,
  on l'ajoutera quand le besoin sera concret.
- Pas d'``outputs_preferred`` (mapping logique "preferred_text =
  step3.RAW_TEXT").  Reporté quand un caller en aura concrètement
  besoin.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from picarones.domain.artifacts import ArtifactType


#: Identifiant d'étape — alphanum + ``_-``.  Doit être un nom court
#: lisible par un humain dans les logs et le rapport.
_STEP_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

#: Modes canoniques d'un pipeline OCR+LLM.  Source de vérité unique
#: depuis la Phase 7.1 audit code-quality (2026-05) — auparavant
#: dupliqué trois fois (``pipeline/llm_pipeline_config.py:25``,
#: ``pipeline/llm_pipeline_builder.py:61``,
#: ``interfaces/web/models.py:71``), avec un risque concret de
#: divergence si l'un des trois ajoutait un nouveau mode.
#:
#: Sémantique :
#:
#: - ``text_only`` — l'OCR amont produit un texte brut, le LLM le
#:   corrige sans voir l'image (post-correction texte pur).
#: - ``text_and_image`` — l'OCR amont produit un texte ; le VLM le
#:   corrige en s'appuyant sur l'image (post-correction multimodale).
#: - ``zero_shot`` — pas d'OCR amont ; un VLM transcrit l'image
#:   directement.
PipelineMode = Literal["text_only", "text_and_image", "zero_shot"]

#: Sentinel pour ``inputs_from`` qui désigne les artefacts initiaux
#: fournis au runner (typiquement ``IMAGE``).
INITIAL_STEP_ID = "__initial__"


class PipelineStep(BaseModel):
    """Une étape déclarative dans un DAG de pipeline.

    Attributs
    ---------
    id:
        Identifiant unique de l'étape dans la pipeline (alphanum +
        ``_-``).  Sert dans les logs, le rapport, et comme cible
        des références ``inputs_from`` des étapes en aval.
    kind:
        Catégorie informationnelle de l'étape (``"ocr"``,
        ``"post_correction"``, ``"alto_remapping"``,
        ``"alto_reconstruction"``, etc.).  Pas de validation
        d'enum — c'est un label libre que les services et le
        rapport peuvent grouper.  Par convention, en
        ``snake_case``.
    adapter_name:
        Nom de l'adapter dans le registre runtime (résolu par
        ``app/services`` au S19).  Convention :
        ``"<provider>:<engine_or_model>"`` (ex : ``"tesseract"``,
        ``"openai:gpt-4o"``, ``"mistral:large"``,
        ``"<vendor>:<custom_module>"``).
    params:
        Paramètres passés à l'adapter au moment de l'exécution.
        Format libre (chaque adapter valide les siens) — typage
        scalaire pour rester sérialisable en YAML.
    input_types:
        Types d'artefacts consommés par l'étape.  Validés par
        ``validate_spec`` contre les outputs des étapes antérieures.
    output_types:
        Types d'artefacts produits.  Validés au runtime par
        l'executor (qui vérifie que tous les types déclarés sont
        bien dans le dict retourné par l'adapter).
    inputs_from:
        DAG branchant (héritage du Sprint 66).  Pour chaque type
        d'entrée, désigne explicitement l'étape source.  La chaîne
        spéciale ``"__initial__"`` désigne les entrées initiales
        du runner.  Si le dict est vide, l'executor prend la
        version la plus récente de chaque type dans le bag.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    kind: str = Field(min_length=1, max_length=64)
    adapter_name: str = Field(min_length=1, max_length=256)
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)
    input_types: tuple[ArtifactType, ...] = Field(default_factory=tuple)
    output_types: tuple[ArtifactType, ...] = Field(default_factory=tuple)
    inputs_from: dict[ArtifactType, str] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _validate_step_id(cls, v: str) -> str:
        if not _STEP_ID_RE.match(v):
            from picarones.domain.errors import PicaronesError
            raise PicaronesError(
                f"step id invalide : {v!r}.  "
                f"Doit matcher {_STEP_ID_RE.pattern!r} (alphanum + _-)."
            )
        if v == INITIAL_STEP_ID:
            from picarones.domain.errors import PicaronesError
            raise PicaronesError(
                f"step id réservé : {INITIAL_STEP_ID!r} désigne "
                "les entrées initiales du runner."
            )
        return v


class PipelineSpec(BaseModel):
    """DAG déclaratif d'une pipeline composée.

    Sérialisable en YAML via ``model_dump()`` + ``yaml.safe_dump``,
    chargeable via ``model_validate(yaml.safe_load(text))``.  Le
    round-trip est testé.

    Attributs
    ---------
    name:
        Nom court de la pipeline (utilisé dans les logs, le cache,
        le rapport).  Convention ``snake_case``.
    description:
        Phrase courte d'introduction affichée dans le rapport.
    initial_inputs:
        Types d'artefacts qui doivent être fournis par le caller
        au moment de l'exécution.  Convention : ``(IMAGE,)`` pour
        une pipeline OCR classique, ``(IMAGE, RAW_TEXT)`` pour
        une post-correction qui part d'un OCR pré-calculé.
    steps:
        Étapes du DAG, ordonnées par dépendance topologique
        d'exécution.  Si une étape ``s2`` dépend de ``s1``, alors
        ``s1`` apparaît avant ``s2``.  ``validate_spec`` détecte
        les violations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    description: str = ""
    initial_inputs: tuple[ArtifactType, ...] = Field(default_factory=tuple)
    steps: tuple[PipelineStep, ...] = Field(default_factory=tuple)

    def step_by_id(self, step_id: str) -> PipelineStep | None:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None


__all__ = ["PipelineMode", "PipelineSpec", "PipelineStep", "INITIAL_STEP_ID"]
