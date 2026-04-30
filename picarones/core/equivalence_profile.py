"""Équivalences diplomatiques granulaires — Sprint 78 (A.I.5).

Sprint 78 — A.I.5 du plan d'évolution 2026.

Pourquoi ce module
------------------
Aujourd'hui les profils de ``picarones/core/normalization.py``
(``medieval_french``, ``early_modern_french``, etc.) appliquent un
**bloc entier** de transformations.  Mais un éditeur peut vouloir
nuancer : *« je tolère ``ſ → s`` mais pas ``u → v`` »* — par
exemple parce qu'il édite un imprimé du XVIᵉ où u/v sont
distinctes mais où le s long doit être normalisé.

Ce module **éclate** chaque profil en règles d'équivalence
**nommées et indépendantes** que l'utilisateur peut activer ou
désactiver une par une.  La couche de calcul retourne le CER
recalculé avec un sous-ensemble personnalisé.

Format
------
Chaque règle a :

- ``name`` : identifiant stable utilisé dans les URLs et l'UX
  (ex. ``"longs_s"``, ``"u_eq_v"``)
- ``source`` : caractère ou séquence à remplacer
- ``target`` : caractère ou séquence cible
- ``description`` : phrase courte FR destinée à l'utilisateur
- ``profile_tag`` : nom du profil dont elle est issue (utile pour
  grouper dans l'UX)

Stratégie de découpage
----------------------
Couche de calcul d'abord (pattern Sprint 71/75/76).  L'UX panneau
avancé (cases à cocher + recalcul JS client + URL state) suivra
dans un sprint dédié — la couche calcul livrée ici est une
fondation suffisante pour qu'un développeur frontend câble la vue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

from picarones.core.normalization import (
    DIPLOMATIC_EN_EARLY_MODERN,
    DIPLOMATIC_FR_EARLY_MODERN,
    DIPLOMATIC_LATIN_MEDIEVAL,
    DIPLOMATIC_MINIMAL,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EquivalenceRule:
    """Une équivalence diplomatique nommée et indépendante."""
    name: str
    source: str
    target: str
    description: str
    profile_tag: str


# Catalogue : on dérive des profils existants en attribuant un nom
# stable à chaque transformation.  Les doublons (ex. ``ſ → s``
# présent dans plusieurs profils) sont fusionnés sous un nom unique
# (le premier rencontré).
def _build_catalog() -> dict[str, EquivalenceRule]:
    catalog: dict[str, EquivalenceRule] = {}

    # Noms canoniques pour les transformations courantes
    canonical_names: dict[tuple[str, str], tuple[str, str]] = {
        ("ſ", "s"):  ("longs_s", "s long ſ → s"),
        ("u", "v"):  ("u_eq_v", "u/v interchangeables (vpon → upon)"),
        ("i", "j"):  ("i_eq_j", "i/j interchangeables (ioy → joy)"),
        ("y", "i"):  ("y_eq_i", "y → i (Latin médiéval)"),
        ("vv", "w"): ("vv_eq_w", "vv → w (anglais moderne)"),
        ("æ", "ae"): ("ae_ligature", "æ → ae"),
        ("œ", "oe"): ("oe_ligature", "œ → oe"),
        ("þ", "th"): ("thorn_th", "þ (thorn) → th"),
        ("ð", "th"): ("eth_th", "ð (eth) → th"),
        ("ȝ", "y"):  ("yogh_y", "ȝ (yogh) → y"),
        ("&", "et"): ("ampersand_et", "& → et (esperluette)"),
        ("ỹ", "yn"): ("y_tilde_yn", "ỹ → yn"),
        ("ꝑ", "per"): ("p_per", "ꝑ → per (abréviation Capelli)"),
        ("ꝓ", "pro"): ("p_pro", "ꝓ → pro (abréviation Capelli)"),
        ("ꝗ", "que"): ("q_que", "ꝗ → que (q barré)"),
    }

    sources = [
        ("medieval_french", DIPLOMATIC_LATIN_MEDIEVAL),
        ("early_modern_french", DIPLOMATIC_FR_EARLY_MODERN),
        ("early_modern_english", DIPLOMATIC_EN_EARLY_MODERN),
        ("minimal", DIPLOMATIC_MINIMAL),
    ]

    for profile_tag, profile_dict in sources:
        for source, target in profile_dict.items():
            key = (source, target)
            if key in canonical_names:
                name, desc = canonical_names[key]
            else:
                # Fallback : générer un nom à partir des codepoints
                name = f"{source}_to_{target}".replace(" ", "_")
                desc = f"{source} → {target}"
            if name in catalog:
                # On garde le profile_tag du premier rencontré, mais
                # on note que la règle est partagée.
                continue
            catalog[name] = EquivalenceRule(
                name=name,
                source=source,
                target=target,
                description=desc,
                profile_tag=profile_tag,
            )
    return catalog


BUILTIN_EQUIVALENCES: dict[str, EquivalenceRule] = _build_catalog()


def list_equivalences_by_profile(
    profile_name: Optional[str] = None,
) -> list[EquivalenceRule]:
    """Liste les règles d'équivalence disponibles.

    Si ``profile_name`` est fourni, ne retourne que les règles dont
    ``profile_tag == profile_name`` (ou les règles dérivées de
    plusieurs profils dont au moins un est ``profile_name``).
    """
    if profile_name is None:
        return list(BUILTIN_EQUIVALENCES.values())
    return [
        rule for rule in BUILTIN_EQUIVALENCES.values()
        if rule.profile_tag == profile_name
    ]


def apply_selected_equivalences(
    text: Optional[str],
    selected_names: Iterable[str],
) -> str:
    """Applique uniquement les règles dont le nom est dans
    ``selected_names``.

    L'ordre d'application est l'ordre du catalogue interne — les
    transformations sont appliquées séquentiellement sur le texte.
    Les règles inconnues sont silencieusement ignorées (avec
    warning).
    """
    if not text:
        return text or ""
    selected_set = set(selected_names)
    if not selected_set:
        return text
    out = text
    for name, rule in BUILTIN_EQUIVALENCES.items():
        if name not in selected_set:
            continue
        out = out.replace(rule.source, rule.target)
    # Détection des règles inconnues (pour logger explicite)
    unknown = selected_set - set(BUILTIN_EQUIVALENCES.keys())
    if unknown:
        logger.warning(
            "[equivalence_profile] règles inconnues ignorées : %s",
            sorted(unknown),
        )
    return out


def compute_cer_with_equivalences(
    reference: Optional[str],
    hypothesis: Optional[str],
    selected_names: Iterable[str],
) -> float:
    """Calcule le CER après application des équivalences sélectionnées
    sur les **deux** côtés (GT et hypothèse).

    Utilise ``picarones.core.metrics.compute_metrics`` et extrait
    le champ ``cer`` du résultat.
    """
    from picarones.core.metrics import compute_metrics

    selected_list = list(selected_names)
    ref = apply_selected_equivalences(reference or "", selected_list)
    hyp = apply_selected_equivalences(hypothesis or "", selected_list)
    result = compute_metrics(ref, hyp)
    return result.cer


__all__ = [
    "EquivalenceRule",
    "BUILTIN_EQUIVALENCES",
    "list_equivalences_by_profile",
    "apply_selected_equivalences",
    "compute_cer_with_equivalences",
]
