"""Tests de cohérence du ``SPECS.md`` avec le code réel.

Sprint A2 (préparation des gates pour B-12 — refonte SPECS en A14).

Le SPECS actuel (mars 2025 + addendum sprints 16-30) est désynchronisé
d'~75 sprints. La refonte intégrale est planifiée en Sprint A14. En
attendant, on pose ici un **garde-fou minimal** qui évite que de
nouvelles divergences ne s'ajoutent silencieusement :

1. Le document doit exister et déclarer une version + date.
2. Toute promesse explicitement *abandonnée* depuis SPECS v1 doit être
   marquée par une balise ``Reporté`` ou ``Abandonné`` (pour qu'un
   primo-lecteur ne s'attende pas à trouver la fonctionnalité).

Le test est délibérément permissif : il ne vérifie pas le contenu
fonctionnel section par section (c'est le rôle de A14). Il garantit
seulement qu'on ne reculera pas davantage.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SPECS_PATH = REPO_ROOT / "SPECS.md"


def _read_specs() -> str:
    if not SPECS_PATH.exists():
        pytest.skip("SPECS.md absent")
    return SPECS_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Existence et meta
# ---------------------------------------------------------------------------


def test_specs_exists() -> None:
    """Pré-requis : SPECS.md doit exister."""
    assert SPECS_PATH.exists(), (
        "SPECS.md absent à la racine. Si retiré volontairement, "
        "supprimer aussi ce test."
    )


def test_specs_declares_version_and_date() -> None:
    """SPECS doit déclarer son numéro de version et sa date (ligne ``Version`` ou ``Date``)."""
    text = _read_specs()
    has_version = bool(re.search(r"\bVersion\s*\d", text, re.IGNORECASE))
    has_date = bool(
        re.search(r"\b(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}", text, re.IGNORECASE)
        or re.search(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}", text, re.IGNORECASE)
    )
    assert has_version and has_date, (
        "SPECS.md doit contenir une ligne 'Version X.Y' et un mois en lettres "
        "(ex: 'Mars 2025'). Manque : "
        f"version={has_version}, date={has_date}."
    )


# ---------------------------------------------------------------------------
# Promesses abandonnées doivent être marquées
# ---------------------------------------------------------------------------

#: Items que SPECS v1 promettait mais qui ne sont pas implémentés
#: au 2 mai 2026. Pour chacun, le SPECS doit soit (a) ne plus en parler,
#: soit (b) l'accompagner d'une balise « Reporté », « Abandonné », « Annulé »
#: ou « Non implémenté » dans un rayon de 200 caractères.
ABANDONED_FEATURES_TO_FLAG = {
    "AWS Textract": ["AWS Textract"],
    "Calamari": ["Calamari"],
    "OCRopus": ["OCRopus"],
    "Recommandation automatique": [
        "Recommandation automatique",
        "recommandation automatique : quel concurrent",
    ],
    "Export PDF": ["Export PDF", "PDF synthétique"],
    "k-means clustering": ["k-means", "Clustering automatique des patterns"],
    "Annotations inline": ["Annotations inline"],
    "Badge SVG qualité": ["Badge de qualité générable", "SVG quality badge"],
}

#: Mots-clés qui marquent un statut « non livré ».
DEPRECATION_MARKERS = (
    "reporté",
    "reportée",
    "abandonné",
    "abandonnée",
    "annulé",
    "annulée",
    "non implémenté",
    "non implémentée",
    "non livré",
    "non livrée",
    "deferred",
    "abandoned",
    "cancelled",
    "not implemented",
)


#: Balises HTML qui ouvrent et ferment un bloc de promesses
#: explicitement déclarées comme abandonnées. Les features listées
#: entre ces deux balises sont acceptées par le test, où qu'elles
#: apparaissent dans le document.
ABANDONED_BLOCK_START = "<!-- specs-check: known-abandoned-start -->"
ABANDONED_BLOCK_END = "<!-- specs-check: known-abandoned-end -->"


def _extract_abandoned_block(text: str) -> str:
    """Retourne le contenu du bloc d'abandon déclaré, ou chaîne vide."""
    start = text.find(ABANDONED_BLOCK_START)
    end = text.find(ABANDONED_BLOCK_END)
    if start == -1 or end == -1 or end < start:
        return ""
    return text[start + len(ABANDONED_BLOCK_START) : end].lower()


def _has_deprecation_nearby(text: str, idx: int, window: int = 200) -> bool:
    """Vrai si l'un des marqueurs est présent dans une fenêtre autour de ``idx``."""
    start = max(0, idx - window)
    end = min(len(text), idx + window)
    snippet = text[start:end].lower()
    return any(marker in snippet for marker in DEPRECATION_MARKERS)


def _is_globally_abandoned(text: str, pattern: str) -> bool:
    """Vrai si ``pattern`` est listé dans le bloc d'abandon global ET
    accompagné d'un marqueur de deprecation dans le bloc."""
    block = _extract_abandoned_block(text)
    if not block:
        return False
    if pattern.lower() not in block:
        return False
    # Le bloc lui-même doit contenir au moins un marqueur de deprecation
    return any(marker in block for marker in DEPRECATION_MARKERS)


@pytest.mark.parametrize("feature_name,patterns", list(ABANDONED_FEATURES_TO_FLAG.items()))
def test_abandoned_feature_marked_or_absent(feature_name: str, patterns: list[str]) -> None:
    """Pour chaque promesse abandonnée, l'une des trois conditions
    suivantes doit être satisfaite :

    1. La feature n'apparaît plus dans SPECS ;
    2. Chaque mention est accompagnée d'un marqueur de deprecation
       dans une fenêtre de 200 chars ;
    3. La feature est listée dans un bloc global
       ``<!-- specs-check: known-abandoned-start -->`` … ``-end -->``
       qui contient lui-même un marqueur de deprecation.

    La condition 3 permet de centraliser la documentation des
    abandons dans un encart unique sans devoir paraphraser à chaque
    occurrence."""
    text = _read_specs()
    text_lower = text.lower()

    # Condition 3 : bloc global
    for pattern in patterns:
        if _is_globally_abandoned(text, pattern):
            return

    unmarked: list[tuple[str, int]] = []
    for pattern in patterns:
        for m in re.finditer(re.escape(pattern.lower()), text_lower):
            if not _has_deprecation_nearby(text, m.start()):
                unmarked.append((pattern, m.start()))

    assert not unmarked, (
        f"Mention(s) de '{feature_name}' dans SPECS.md sans balise de "
        f"deprecation à proximité (mots tolérés : {DEPRECATION_MARKERS}). "
        f"Positions : {unmarked}. "
        f"Soit retirer la mention, soit ajouter une note explicite "
        f"« — reporté / abandonné / non livré », soit lister la feature "
        f"dans le bloc <!-- specs-check: known-abandoned-start --> en tête de SPECS.md."
    )
