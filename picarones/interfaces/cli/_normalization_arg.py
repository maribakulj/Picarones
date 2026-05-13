"""Phase 3.3 audit code-quality — résolution de l'argument
``--normalization-profile`` côté CLI.

Accepte deux formes :

1. **Identifiant builtin** (``nfc``, ``caseless``, ``medieval_french``,
   ``early_modern_english``...) → résolu via
   :data:`picarones.evaluation.metrics.normalization.NORMALIZATION_PROFILES`.

2. **Chemin vers un fichier YAML** (extension ``.yaml`` ou ``.yml``)
   → chargé via :meth:`NormalizationProfile.from_yaml`.

L'objectif est de permettre aux chercheurs en philologie de
**versionner leur profil custom dans git** plutôt que de le re-saisir
dans l'UI à chaque session.

Le format YAML attendu :

.. code-block:: yaml

    name: medieval_custom
    description: Français médiéval personnalisé pour le corpus BnF
    caseless: false
    nfc: true
    exclude_chars: ".,;:!?"
    diplomatic:
      ſ: s
      u: v
      v: u
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from picarones.formats.text.normalization import NormalizationProfile


def resolve_normalization_profile(
    value: Optional[str],
) -> Optional["NormalizationProfile"]:
    """Résout l'argument utilisateur en :class:`NormalizationProfile`.

    Parameters
    ----------
    value:
        ``None`` (pas d'option passée), un identifiant builtin, ou
        un chemin vers un fichier ``.yaml`` / ``.yml``.

    Returns
    -------
    NormalizationProfile or None
        ``None`` si ``value`` vaut ``None`` ou chaîne vide.

    Raises
    ------
    FileNotFoundError
        Si ``value`` ressemble à un chemin YAML mais que le fichier
        n'existe pas.
    ValueError
        Si ``value`` n'est ni un identifiant connu ni un chemin
        YAML valide, ou si le YAML est mal formé.
    """
    if not value:
        return None

    # Cas 1 : chemin de fichier YAML (extension explicite).
    lowered = value.lower()
    if lowered.endswith((".yaml", ".yml")):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(
                f"Profil YAML introuvable : {value!r}.  Vérifier le "
                f"chemin (relatif au cwd) ou utiliser un identifiant "
                f"builtin (cf. ``picarones info --normalization-profiles``)."
            )
        from picarones.formats.text.normalization import NormalizationProfile
        return NormalizationProfile.from_yaml(path)

    # Cas 2 : identifiant builtin.
    from picarones.evaluation.metrics.normalization import NORMALIZATION_PROFILES
    if value in NORMALIZATION_PROFILES:
        return NORMALIZATION_PROFILES[value]

    # Ni l'un ni l'autre — message d'aide explicite.
    known = ", ".join(sorted(NORMALIZATION_PROFILES.keys()))
    raise ValueError(
        f"Profil de normalisation inconnu : {value!r}.\n"
        f"Soit utiliser un identifiant builtin parmi : {known}\n"
        f"Soit fournir un chemin vers un fichier .yaml ou .yml."
    )


__all__ = ["resolve_normalization_profile"]
