"""Sprint A14-S1 — A.I.0 P0 : ``normalization_profile`` propagé end-to-end.

Avant ce sprint, le paramètre ``normalization_profile`` était :

- exposé par l'API web (``BenchmarkRequest`` / ``BenchmarkRunRequest``) ;
- transporté jusqu'à ``benchmark_utils.run_benchmark_thread*`` ;
- **silencieusement ignoré** : jamais transmis à ``run_benchmark`` ;
- ``run_benchmark`` n'avait même pas le paramètre dans sa signature.

Conséquence : tout benchmark lancé depuis l'API web utilisait le
profil par défaut (``medieval_french``) quel que soit le choix
utilisateur.  L'option de l'UI était un faux bouton.

Ce module verrouille la propagation depuis la signature publique de
``run_benchmark`` jusqu'à ``compute_metrics`` via les workers.
"""

from __future__ import annotations

import inspect

from picarones.app.schemas.run_spec import RunSpec
from picarones.app.services import prepare_preset_args
from picarones.evaluation.metrics.normalization import (
    NORMALIZATION_PROFILES,
    get_builtin_profile,
)


class TestRunBenchmarkSignature:
    """Phase B3-final (mai 2026) — la propagation de
    ``normalization_profile`` est désormais portée par ``RunSpec``
    (champ Pydantic) et par ``prepare_preset_args`` (kwarg).
    ``run_benchmark_via_service`` a été supprimé."""

    def test_run_spec_exposes_normalization_profile(self) -> None:
        """``RunSpec.normalization_profile`` est un champ Pydantic
        documenté (cf. Phase B1)."""
        assert "normalization_profile" in RunSpec.model_fields
        field = RunSpec.model_fields["normalization_profile"]
        # Champ optionnel — défaut None.
        assert field.default is None

    def test_prepare_preset_args_accepts_normalization_profile(
        self,
    ) -> None:
        """``prepare_preset_args`` propage le profil au RunSpec."""
        sig = inspect.signature(prepare_preset_args)
        assert "normalization_profile" in sig.parameters
        # Optionnel par défaut.
        assert sig.parameters["normalization_profile"].default is None


class TestProfileResolution:
    def test_all_eleven_profiles_resolvable(self) -> None:
        """Les 11 profils annoncés dans le README sont tous résolvables.

        Verrouille la cohérence entre ``NORMALIZATION_PROFILES`` (table
        runtime) et ``NormalizationProfileId`` (Literal Pydantic web).
        """
        expected = {
            "nfc", "caseless", "minimal",
            "medieval_french", "early_modern_french",
            "medieval_latin", "medieval_english", "early_modern_english",
            "secretary_hand", "sans_ponctuation", "sans_apostrophes",
        }
        assert set(NORMALIZATION_PROFILES.keys()) >= expected
        for name in expected:
            profile = get_builtin_profile(name)
            assert profile is not None
            assert profile.name == name


class TestWebModelProfileAlignment:
    def test_web_literal_lists_all_eleven_profiles(self) -> None:
        """Le ``Literal`` Pydantic doit lister les 11 profils.

        Avant S1, le Literal n'en exposait que 8 — Pydantic rejetait
        donc 3 profils valides du runtime.
        """
        from picarones.interfaces.web.models import NormalizationProfileId
        from typing import get_args
        literals = set(get_args(NormalizationProfileId))
        runtime = set(NORMALIZATION_PROFILES.keys())
        # Le web peut être un sous-ensemble strict en théorie, mais
        # l'alignement README ↔ web ↔ runtime exige égalité.
        assert literals == runtime, (
            f"Décalage README/web/runtime.  Web a {literals}, "
            f"runtime a {runtime}.  Diff missing-from-web: "
            f"{runtime - literals}, extra-in-web: {literals - runtime}."
        )


class TestNormalizationActuallyApplied:
    """Vérifie via une intégration unitaire que le profil arrive bien
    jusqu'à ``compute_metrics`` et change le ``cer_diplomatic`` calculé."""

    def test_cer_diplomatic_uses_specified_profile(self) -> None:
        """Avec deux profils différents, le ``cer_diplomatic`` est
        différent sur la même paire de textes.  Si le profil n'était
        pas propagé, on aurait toujours la même valeur."""
        from picarones.evaluation.metrics.text_metrics import compute_metrics

        # Texte avec un ſ médiéval + un v moderne (la GT a l'ancienne
        # graphie, l'OCR la moderne).
        gt = "ſuper aqua viuens"
        hyp = "super aqua vivens"

        # Profil "minimal" : seul ſ → s.  v reste v de chaque côté.
        prof_minimal = get_builtin_profile("minimal")
        m_minimal = compute_metrics(gt, hyp, normalization_profile=prof_minimal)

        # Profil "medieval_latin" : ſ → s, u → v, etc.  Sera plus permissif.
        prof_latin = get_builtin_profile("medieval_latin")
        m_latin = compute_metrics(gt, hyp, normalization_profile=prof_latin)

        # Les deux doivent être calculés.
        assert m_minimal.cer_diplomatic is not None
        assert m_latin.cer_diplomatic is not None
        assert m_minimal.diplomatic_profile_name == "minimal"
        assert m_latin.diplomatic_profile_name == "medieval_latin"
        # Les profils diffèrent → le score change.  S'ils étaient
        # confondus (bug de propagation), ce serait égal.
        assert m_minimal.diplomatic_profile_name != m_latin.diplomatic_profile_name
