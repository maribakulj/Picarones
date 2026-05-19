"""Sérialisation YAML des ``PipelineSpec``

Helpers de chargement / écriture YAML.  Volontairement minces —
``pydantic.model_dump()`` produit déjà un dict imbriqué
sérialisable, et ``yaml.safe_dump`` / ``yaml.safe_load`` sont
suffisants pour le contrat round-trip.

Pourquoi un module dédié plutôt qu'une méthode de classe ?
----------------------------------------------------------
Le ``domain/`` ne doit pas dépendre de PyYAML — c'est une lib
externe que la couche layer permet seulement à ``formats/``,
``app/`` et adjacents.  ``pipeline/`` peut importer pyyaml
(autorisé par les règles du S3), donc le helper vit ici.

API :

    >>> from picarones.pipeline import dump_spec_to_yaml, load_spec_from_yaml
    >>> text = dump_spec_to_yaml(spec)
    >>> spec2 = load_spec_from_yaml(text)
    >>> spec == spec2
    True
"""

from __future__ import annotations

import yaml

from picarones.domain.pipeline_spec import PipelineSpec


def dump_spec_to_yaml(spec: PipelineSpec) -> str:
    """Sérialise une ``PipelineSpec`` en YAML déterministe.

    Le YAML produit est compatible avec ``load_spec_from_yaml``
    et conserve l'ordre des champs et des étapes.
    """
    payload = spec.model_dump(mode="json")
    return yaml.safe_dump(
        payload,
        sort_keys=False,        # conserve l'ordre des champs
        allow_unicode=True,     # préserve accents et caractères spéciaux
        default_flow_style=False,  # style "block" lisible
    )


def load_spec_from_yaml(text: str) -> PipelineSpec:
    """Parse une chaîne YAML et retourne une ``PipelineSpec`` validée.

    Lève ``pydantic.ValidationError`` si le YAML ne respecte pas
    le schéma, ou ``yaml.YAMLError`` si le YAML est mal formé.
    """
    payload = yaml.safe_load(text)
    if payload is None:
        from picarones.domain.errors import PicaronesError
        raise PicaronesError("YAML vide — pas de PipelineSpec à charger")
    return PipelineSpec.model_validate(payload)


__all__ = ["dump_spec_to_yaml", "load_spec_from_yaml"]
