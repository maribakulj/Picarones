"""Profils de normalisation unicode pour le calcul du CER diplomatique.

La normalisation diplomatique permet de calculer un CER tenant compte des
équivalences graphiques propres aux documents historiques : ſ=s, u=v, i=j, etc.

En appliquant la même table aux deux textes (GT et OCR), on mesure les erreurs
"substantielles" (transcription erronée) en ignorant les variations graphiques
codifiées connues.

Trois niveaux de normalisation sont disponibles :

1. NFC       : normalisation Unicode canonique (décomposition+recomposition)
2. caseless  : NFC + pliage de casse (casefold)
3. diplomatic: NFC + table de correspondances historiques configurables

Les profils préconfigurés couvrent les cas d'usage patrimoniaux courants.
Ils sont également chargeables depuis un fichier YAML.

Exemple YAML
------------
name: medieval_custom
caseless: false
diplomatic:
  ſ: s
  u: v
  i: j
  y: i
  æ: ae
  œ: oe
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Tables de correspondances diplomatiques préconfigurées
# ---------------------------------------------------------------------------

#: Français médiéval (XIIe–XVe siècle)
DIPLOMATIC_FR_MEDIEVAL: dict[str, str] = {
    "ſ": "s",    # s long → s
    "u": "v",    # u/v interchangeables en position initiale
    "i": "j",    # i/j interchangeables
    "y": "i",    # y vocalique → i
    "æ": "ae",   # ligature æ
    "œ": "oe",   # ligature œ
    "ꝑ": "per",  # abréviation per/par
    "ꝓ": "pro",  # abréviation pro
    "\u0026": "et",  # & → et
}

#: Français moderne / imprimés anciens (XVIe–XVIIIe siècle)
DIPLOMATIC_FR_EARLY_MODERN: dict[str, str] = {
    "ſ": "s",    # s long
    "æ": "ae",
    "œ": "oe",
    "\u0026": "et",
    "ỹ": "yn",   # y tilde
}

#: Latin médiéval
DIPLOMATIC_LATIN_MEDIEVAL: dict[str, str] = {
    "ſ": "s",
    "u": "v",
    "i": "j",
    "y": "i",
    "æ": "ae",
    "œ": "oe",
    "ꝑ": "per",
    "ꝓ": "pro",
    "ꝗ": "que",   # q barré → que
    "\u0026": "et",
}

#: Profil minimal — uniquement NFC + s long
DIPLOMATIC_MINIMAL: dict[str, str] = {
    "ſ": "s",
}


# ---------------------------------------------------------------------------
# Profil de normalisation
# ---------------------------------------------------------------------------

@dataclass
class NormalizationProfile:
    """Décrit une stratégie de normalisation pour le calcul du CER diplomatique.

    Parameters
    ----------
    name:
        Identifiant lisible du profil (ex : ``"medieval_french"``).
    nfc:
        Applique la normalisation Unicode NFC (recommandé, activé par défaut).
    caseless:
        Pliage de casse (casefold) après NFC.
    diplomatic_table:
        Table de correspondances graphiques historiques appliquée caractère
        par caractère sur les deux textes avant calcul du CER.
    description:
        Description courte du profil (affichée dans le rapport HTML).
    """

    name: str
    nfc: bool = True
    caseless: bool = False
    diplomatic_table: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def normalize(self, text: str) -> str:
        """Applique le profil de normalisation à un texte."""
        if self.nfc:
            text = unicodedata.normalize("NFC", text)
        if self.caseless:
            text = text.casefold()
        if self.diplomatic_table:
            text = _apply_diplomatic_table(text, self.diplomatic_table)
        return text

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "nfc": self.nfc,
            "caseless": self.caseless,
            "diplomatic_table": self.diplomatic_table,
            "description": self.description,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NormalizationProfile":
        """Charge un profil depuis un fichier YAML.

        Le fichier YAML doit contenir les clés ``name``, optionnellement
        ``caseless``, ``description`` et ``diplomatic`` (dict str→str).

        Example
        -------
        .. code-block:: yaml

            name: medieval_custom
            caseless: false
            description: Français médiéval personnalisé
            diplomatic:
              ſ: s
              u: v
        """
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "Le package 'pyyaml' est requis pour charger les profils YAML. "
                "Installez-le avec : pip install pyyaml"
            ) from exc

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data.get("name", Path(path).stem),
            nfc=bool(data.get("nfc", True)),
            caseless=bool(data.get("caseless", False)),
            diplomatic_table=data.get("diplomatic", {}),
            description=data.get("description", ""),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "NormalizationProfile":
        """Charge un profil depuis un dictionnaire (ex : section YAML inline)."""
        return cls(
            name=data.get("name", "custom"),
            nfc=bool(data.get("nfc", True)),
            caseless=bool(data.get("caseless", False)),
            diplomatic_table=data.get("diplomatic", {}),
            description=data.get("description", ""),
        )


# ---------------------------------------------------------------------------
# Profils préconfigurés
# ---------------------------------------------------------------------------

def get_builtin_profile(name: str) -> NormalizationProfile:
    """Retourne un profil préconfigurée par son identifiant.

    Identifiants disponibles
    ------------------------
    - ``"medieval_french"``    : français médiéval XIIe–XVe (ſ=s, u=v, i=j, æ=ae, œ=oe…)
    - ``"early_modern_french"`` : imprimés anciens XVIe–XVIIIe (ſ=s, œ=oe, æ=ae…)
    - ``"medieval_latin"``     : latin médiéval (ſ=s, u=v, i=j, ꝑ=per, ꝓ=pro…)
    - ``"minimal"``            : uniquement NFC + s long
    - ``"nfc"``                : NFC seul (sans table diplomatique)
    - ``"caseless"``           : NFC + pliage de casse

    Raises
    ------
    KeyError
        Si le nom n'est pas reconnu.
    """
    profiles = {
        "medieval_french": NormalizationProfile(
            name="medieval_french",
            nfc=True,
            caseless=False,
            diplomatic_table=DIPLOMATIC_FR_MEDIEVAL,
            description="Français médiéval (XIIe–XVe) : ſ=s, u=v, i=j, æ=ae, œ=oe",
        ),
        "early_modern_french": NormalizationProfile(
            name="early_modern_french",
            nfc=True,
            caseless=False,
            diplomatic_table=DIPLOMATIC_FR_EARLY_MODERN,
            description="Imprimés anciens (XVIe–XVIIIe) : ſ=s, æ=ae, œ=oe",
        ),
        "medieval_latin": NormalizationProfile(
            name="medieval_latin",
            nfc=True,
            caseless=False,
            diplomatic_table=DIPLOMATIC_LATIN_MEDIEVAL,
            description="Latin médiéval : ſ=s, u=v, i=j, ꝑ=per, ꝓ=pro",
        ),
        "minimal": NormalizationProfile(
            name="minimal",
            nfc=True,
            caseless=False,
            diplomatic_table=DIPLOMATIC_MINIMAL,
            description="Minimal : NFC + s long seulement",
        ),
        "nfc": NormalizationProfile(
            name="nfc",
            nfc=True,
            caseless=False,
            diplomatic_table={},
            description="Normalisation NFC uniquement",
        ),
        "caseless": NormalizationProfile(
            name="caseless",
            nfc=True,
            caseless=True,
            diplomatic_table={},
            description="NFC + insensible à la casse",
        ),
    }
    if name not in profiles:
        raise KeyError(
            f"Profil de normalisation inconnu : '{name}'. "
            f"Disponibles : {', '.join(profiles)}"
        )
    return profiles[name]


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def _apply_diplomatic_table(text: str, table: dict[str, str]) -> str:
    """Applique une table de correspondances diplomatiques caractère par caractère.

    Les clés multi-caractères (ex : ``"ae"`` → ``"æ"``) sont gérées en priorité
    sur les correspondances simples.
    """
    if not table:
        return text

    # Séparer les clés simples (1 char) des clés multi-chars pour traitement ordonné
    multi_keys = sorted(
        (k for k in table if len(k) > 1), key=len, reverse=True
    )
    simple_table = {k: v for k, v in table.items() if len(k) == 1}

    result = text
    # Remplacements multi-chars en premier (évite les conflits)
    for key in multi_keys:
        result = result.replace(key, table[key])

    # Remplacements char par char
    if simple_table:
        result = "".join(simple_table.get(c, c) for c in result)

    return result


# Profil par défaut utilisé pour le CER diplomatique intégré
DEFAULT_DIPLOMATIC_PROFILE: NormalizationProfile = get_builtin_profile("medieval_french")
