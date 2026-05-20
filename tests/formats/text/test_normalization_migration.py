"""Sprint A14-S9 — migration de ``normalization`` vers ``formats/text/``.

Vérifie que :

1. Le nouveau module ``picarones.formats.text.normalization`` expose
   les 11 profils canoniques.
2. L'ancien re-export ``picarones.measurements.normalization`` continue
   à fonctionner sans erreur (compat ascendante stricte).
3. Les symboles privés utilisés downstream (``_parse_exclude_chars``,
   ``_apply_diplomatic_table``) sont ré-exposés via le re-export.
4. Les deux chemins d'import retournent **le même objet** (pas une
   copie) — preuve que c'est un vrai re-export, pas une duplication.
"""

from __future__ import annotations


def test_new_path_exposes_all_eleven_profiles() -> None:
    from picarones.formats.text.normalization import NORMALIZATION_PROFILES
    expected = {
        "nfc", "caseless", "minimal",
        "medieval_french", "early_modern_french",
        "medieval_latin", "early_modern_english", "medieval_english",
        "secretary_hand", "sans_ponctuation", "sans_apostrophes",
    }
    assert set(NORMALIZATION_PROFILES.keys()) == expected


def test_old_reexport_works() -> None:
    """Compat ascendante : ~50 consommateurs importent depuis l'ancien
    chemin."""
    from picarones.evaluation.metrics.normalization import (
        DEFAULT_DIPLOMATIC_PROFILE,
        NORMALIZATION_PROFILES,
        NormalizationProfile,
        get_builtin_profile,
    )
    assert NormalizationProfile is not None
    assert "medieval_french" in NORMALIZATION_PROFILES
    assert get_builtin_profile("nfc") is not None
    assert DEFAULT_DIPLOMATIC_PROFILE.name == "medieval_french"


def test_private_symbols_reexported() -> None:
    """Les symboles préfixés ``_`` utilisés en aval doivent rester
    importables depuis l'ancien chemin."""
    from picarones.evaluation.metrics.normalization import (
        _apply_diplomatic_table,
        _parse_exclude_chars,
    )
    assert callable(_parse_exclude_chars)
    assert callable(_apply_diplomatic_table)


def test_old_and_new_paths_share_same_objects() -> None:
    """Preuve que c'est un vrai re-export, pas une duplication."""
    from picarones.formats.text.normalization import (
        NORMALIZATION_PROFILES as new_profiles,
        NormalizationProfile as NewProfile,
        get_builtin_profile as new_get,
    )
    from picarones.evaluation.metrics.normalization import (
        NORMALIZATION_PROFILES as old_profiles,
        NormalizationProfile as OldProfile,
        get_builtin_profile as old_get,
    )
    assert new_profiles is old_profiles  # même dict
    assert NewProfile is OldProfile      # même classe
    assert new_get is old_get            # même fonction


def test_apply_profile_works_via_new_path() -> None:
    """Test fonctionnel : un profil chargé depuis le nouveau chemin
    applique bien la normalisation."""
    from picarones.formats.text.normalization import get_builtin_profile
    profile = get_builtin_profile("medieval_french")
    # ſ → s, u → v dans le profil médiéval français.
    normalized = profile.normalize("aſpre")
    assert "ſ" not in normalized
    assert "s" in normalized
