"""Politique de modules contribués — Sprint 97 (B.6).

Sprint 97 — B.6 du plan d'évolution 2026.

Pourquoi ce module
------------------
Avant d'ouvrir Picarones aux contributions externes (axe B —
modules tiers que l'utilisateur amène), il faut un cadre de
qualité explicite : *« un module qui ne passe pas l'audit
n'est pas exécutable. »*

Ce module fournit l'**enveloppe d'audit** :

- ``ModuleManifest`` — métadonnées obligatoires (auteur,
  licence, version, citation, contrat d'entrée/sortie typé).
- ``validate_manifest(manifest)`` — vérifie que tous les champs
  obligatoires sont présents et bien formés.
- ``audit_module(module_class_or_instance, manifest)`` —
  vérifie en plus que la classe respecte le contrat ``BaseModule``
  et que ``input_types``/``output_types`` correspondent au
  manifeste.
- ``AuditResult`` — verdict structuré ``passed/failed`` + liste
  des checks détaillés.

Stratégie d'ouverture
---------------------
Phase fermée actuelle : modules officiels uniquement,
contributions via PR sur le repo principal.  Phase ouverte
future : une fois 5–6 modules officiels stables, ouverture via
``entry_points`` sur PyPI (``picarones-module-X``).  Ce module
prépare la phase ouverte sans la déclencher : tout module
externe devra fournir un ``ModuleManifest`` valide pour être
exécuté.

Pas de SPDX validator
---------------------
On vérifie la présence et la non-vacuité des champs licence ;
on ne valide pas la conformité SPDX du nom (``MIT`` vs
``mit-license`` vs ``MIT License``).  Le chercheur reste
responsable du choix de licence ; l'outil documente, il ne
juge pas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Champs obligatoires d'un ManifestModule (texte non-vide).
_REQUIRED_TEXT_FIELDS = (
    "name", "version", "author", "license",
    "description",
)


@dataclass
class ModuleManifest:
    """Métadonnées d'un module contribué.

    Attributes
    ----------
    name:
        Identifiant unique du module (ex. ``"my-llm-correcteur"``).
    version:
        Version sémantique (ex. ``"1.2.0"``).
    author:
        Auteur ou institution responsable.
    license:
        Identifiant de licence (SPDX recommandé, non validé).
    description:
        Description courte (≤ 1 phrase).
    input_types:
        Liste des types d'entrée (chaînes).  Doit correspondre
        à ``module.input_types`` (Sprint 33).
    output_types:
        Liste des types de sortie.  Doit correspondre à
        ``module.output_types``.
    citation:
        Citation académique (BibTeX, DOI, ou texte libre).
        Optionnel.
    homepage:
        URL du dépôt ou de la page projet. Optionnel.
    picarones_min_version:
        Version minimale de Picarones requise. Optionnel.
    extra:
        Métadonnées libres (clé → valeur).
    """

    name: str
    version: str
    author: str
    license: str
    description: str
    input_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)
    citation: Optional[str] = None
    homepage: Optional[str] = None
    picarones_min_version: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "license": self.license,
            "description": self.description,
            "input_types": list(self.input_types),
            "output_types": list(self.output_types),
            "citation": self.citation,
            "homepage": self.homepage,
            "picarones_min_version": self.picarones_min_version,
            "extra": dict(self.extra),
        }


@dataclass
class AuditCheck:
    """Un check individuel de l'audit."""

    name: str
    passed: bool
    detail: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class AuditResult:
    """Résultat global d'un audit de module."""

    module_name: str
    passed: bool
    checks: list[AuditCheck] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def as_dict(self) -> dict:
        return {
            "module_name": self.module_name,
            "passed": self.passed,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "checks": [c.as_dict() for c in self.checks],
        }


def validate_manifest(manifest: ModuleManifest) -> list[AuditCheck]:
    """Vérifie qu'un manifest est complet et bien formé.

    Returns
    -------
    list[AuditCheck]
        Un check par champ obligatoire + un check pour
        ``input_types``/``output_types`` non vides.
    """
    checks: list[AuditCheck] = []
    for field_name in _REQUIRED_TEXT_FIELDS:
        value = getattr(manifest, field_name, None)
        ok = isinstance(value, str) and bool(value.strip())
        checks.append(AuditCheck(
            name=f"manifest.{field_name}",
            passed=ok,
            detail=None if ok else f"champ '{field_name}' vide ou absent",
        ))
    # input_types / output_types : au moins une entrée chacun
    in_ok = (
        isinstance(manifest.input_types, list)
        and len(manifest.input_types) > 0
        and all(
            isinstance(t, str) and t for t in manifest.input_types
        )
    )
    checks.append(AuditCheck(
        name="manifest.input_types",
        passed=in_ok,
        detail=None if in_ok else "input_types vide ou non-string",
    ))
    out_ok = (
        isinstance(manifest.output_types, list)
        and len(manifest.output_types) > 0
        and all(
            isinstance(t, str) and t for t in manifest.output_types
        )
    )
    checks.append(AuditCheck(
        name="manifest.output_types",
        passed=out_ok,
        detail=None if out_ok else "output_types vide ou non-string",
    ))
    return checks


def _is_base_module(cls: Any) -> bool:
    """Best-effort : vérifie que cls hérite de BaseModule.

    On ne **pas** importer ``BaseModule`` au top-level pour
    éviter les cycles : on inspecte la chaîne de classes par
    leur nom.
    """
    try:
        for base in cls.__mro__:
            if base.__name__ == "BaseModule":
                return True
    except AttributeError:
        return False
    return False


def audit_module(
    module_class_or_instance: Any,
    manifest: ModuleManifest,
) -> AuditResult:
    """Audite un module contribué : interface + manifest.

    Parameters
    ----------
    module_class_or_instance:
        Soit la classe ``BaseModule`` (Sprint 33), soit une
        instance.
    manifest:
        ``ModuleManifest`` correspondant au module.

    Returns
    -------
    AuditResult
        ``passed=True`` ssi tous les checks passent.
    """
    checks = validate_manifest(manifest)

    # Check : héritage de BaseModule
    cls = (
        type(module_class_or_instance)
        if not isinstance(module_class_or_instance, type)
        else module_class_or_instance
    )
    inherits_base = _is_base_module(cls)
    checks.append(AuditCheck(
        name="module.inherits_base_module",
        passed=inherits_base,
        detail=(
            None if inherits_base
            else "la classe n'hérite pas de picarones.domain.module_protocol.BaseModule"
        ),
    ))

    # Check : input_types / output_types correspondent
    declared_in: list[str] = []
    declared_out: list[str] = []
    try:
        instance = (
            module_class_or_instance
            if not isinstance(module_class_or_instance, type)
            else None
        )
        attr_in = getattr(cls, "input_types", None)
        attr_out = getattr(cls, "output_types", None)
        if instance is not None:
            attr_in = getattr(instance, "input_types", attr_in)
            attr_out = getattr(instance, "output_types", attr_out)
        if attr_in is not None:
            declared_in = [
                getattr(t, "value", str(t)) for t in attr_in
            ]
        if attr_out is not None:
            declared_out = [
                getattr(t, "value", str(t)) for t in attr_out
            ]
    except Exception as exc:  # noqa: BLE001
        # Best-effort : si l'introspection échoue (module manifest
        # mal formé), on retombe sur les listes vides ``declared_in``
        # / ``declared_out`` initialisées plus haut.  Audit Sprint S7
        # plutôt que ``pass`` silencieux.
        logger.debug(
            "[module_policy] introspection input/output_types échouée : %s",
            exc,
        )
    # Comparaison case-insensitive : on accepte "TEXT" ou "text"
    # côté manifest, le contrat sémantique est le même.
    #
    # Phase 4-bis : on normalise aussi les aliases legacy
    # (``"text"`` ↔ ``"raw_text"``, etc.) pour qu'un module qui
    # déclare ``ArtifactType.TEXT`` (valeur canonique
    # ``"raw_text"``) soit accepté contre un manifest qui
    # déclare le nom legacy ``"text"`` ou inversement.
    from picarones.domain.artifacts import LEGACY_VALUE_ALIASES
    _LEGACY_TO_CANONICAL = {v: k for k, v in LEGACY_VALUE_ALIASES.items()}

    def _canonicalize(t: str) -> str:
        low = t.lower()
        return _LEGACY_TO_CANONICAL.get(low, low)

    declared_in_lower = sorted(_canonicalize(t) for t in declared_in)
    declared_out_lower = sorted(_canonicalize(t) for t in declared_out)
    manifest_in_lower = sorted(_canonicalize(t) for t in manifest.input_types)
    manifest_out_lower = sorted(_canonicalize(t) for t in manifest.output_types)
    in_match = declared_in_lower == manifest_in_lower
    checks.append(AuditCheck(
        name="module.input_types_match_manifest",
        passed=in_match,
        detail=(
            None if in_match
            else f"déclaré {declared_in} vs manifest {manifest.input_types}"
        ),
    ))
    out_match = declared_out_lower == manifest_out_lower
    checks.append(AuditCheck(
        name="module.output_types_match_manifest",
        passed=out_match,
        detail=(
            None if out_match
            else f"déclaré {declared_out} vs manifest {manifest.output_types}"
        ),
    ))

    # Check : process callable
    has_process = callable(getattr(cls, "process", None))
    checks.append(AuditCheck(
        name="module.has_process",
        passed=has_process,
        detail=None if has_process else "méthode process() absente",
    ))

    passed = all(c.passed for c in checks)
    return AuditResult(
        module_name=manifest.name,
        passed=passed,
        checks=checks,
    )


__all__ = [
    "ModuleManifest",
    "AuditCheck",
    "AuditResult",
    "validate_manifest",
    "audit_module",
]
