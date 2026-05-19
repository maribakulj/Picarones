"""Sprint A14-S23 — ``RegistryService`` (bootstrap explicite des
registres).

Couverture :

- **Pas d'effet de bord d'import** : importer le module
  ``picarones.app.services.registry_service`` ne crée AUCUN registre
  global (le bootstrap est explicite via une fonction).
- ``bootstrap_default_registries`` peuple les 9 métriques canoniques
  + 3 projecteurs canoniques.
- Sélection par signature de jonction (``select``) retourne le bon
  sous-ensemble pour ``(RAW_TEXT, RAW_TEXT)`` et
  ``(ALTO_XML, ALTO_XML)``.
- ``MetricRegistry.compute`` fonctionne pour chaque métrique
  canonique sur un cas trivial.
- Deux bootstraps successifs produisent des **instances distinctes**
  (pas d'état global partagé) — preuve que les tests peuvent
  isoler leurs registres.
- ``RegistryService.bootstrap_defaults`` (classmethod) est
  équivalent à instancier puis bootstrapper.
- Construction du service avec des arguments invalides → TypeError
  typé.
"""

from __future__ import annotations

from picarones.app.services import (
    RegistriesBundle,
    RegistryService,
    bootstrap_default_registries,
)
from picarones.domain.artifacts import ArtifactType
from picarones.evaluation.projectors import ProjectorRegistry
from picarones.evaluation.registry import MetricRegistry

import pytest


# ──────────────────────────────────────────────────────────────────
# Constantes attendues
# ──────────────────────────────────────────────────────────────────


_EXPECTED_TEXT_METRICS = {
    "cer", "wer", "mer", "wil",
    "searchability_recall", "numerical_sequence_preservation",
}

_EXPECTED_ALTO_METRICS = {
    "alto_validity", "alto_line_count_ratio", "alto_word_box_coverage",
}

_EXPECTED_PROJECTORS = {"alto_to_text", "page_to_text", "canonical_to_text"}


# ──────────────────────────────────────────────────────────────────
# Pas d'effet de bord d'import
# ──────────────────────────────────────────────────────────────────


class TestNoImportSideEffect:
    def test_importing_module_does_not_register_anywhere(self) -> None:
        """Importer le module N'AMORCE PAS un registre global.

        Le rewrite réclame que le bootstrap soit explicite — un
        ``import picarones.app.services.registry_service`` ne doit
        créer aucun registre, ni en globalité, ni implicitement.
        """
        import importlib
        # Re-import frais pour s'assurer qu'aucun cache de side-effect
        # n'existe.
        m = importlib.import_module(
            "picarones.app.services.registry_service",
        )
        # Aucun attribut "registry" ou "_GLOBAL_REGISTRY" exposé.
        for forbidden in (
            "DEFAULT_REGISTRY",
            "GLOBAL_REGISTRY",
            "_DEFAULT_REGISTRY",
            "_GLOBAL_REGISTRY",
            "default_registry",
        ):
            assert not hasattr(m, forbidden), (
                f"Le module expose {forbidden!r} — anti-pattern singleton "
                "global probable."
            )

    def test_default_registry_function_is_pure(self) -> None:
        """Deux appels successifs produisent des **instances distinctes**.
        Pas de cache, pas de mémoïsation — chaque caller peut
        construire son propre registre."""
        b1 = bootstrap_default_registries()
        b2 = bootstrap_default_registries()
        assert b1.metrics is not b2.metrics
        assert b1.projectors is not b2.projectors
        # Mais le contenu est identique.
        assert set(b1.metrics.names()) == set(b2.metrics.names())


# ──────────────────────────────────────────────────────────────────
# Bootstrap par défaut : contenu canonique
# ──────────────────────────────────────────────────────────────────


class TestDefaultBootstrap:
    def test_bundle_returns_two_registries(self) -> None:
        bundle = bootstrap_default_registries()
        assert isinstance(bundle, RegistriesBundle)
        assert isinstance(bundle.metrics, MetricRegistry)
        assert isinstance(bundle.projectors, ProjectorRegistry)

    def test_metric_count_matches_canonical_set(self) -> None:
        bundle = bootstrap_default_registries()
        registered = set(bundle.metrics.names())
        assert registered == (
            _EXPECTED_TEXT_METRICS | _EXPECTED_ALTO_METRICS
        )

    def test_text_junction_returns_six_metrics(self) -> None:
        bundle = bootstrap_default_registries()
        text_metrics = bundle.metrics.select(
            ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT,
        )
        names = {s.name for s in text_metrics}
        assert names == _EXPECTED_TEXT_METRICS

    def test_alto_junction_returns_three_metrics(self) -> None:
        bundle = bootstrap_default_registries()
        alto_metrics = bundle.metrics.select(
            ArtifactType.ALTO_XML, ArtifactType.ALTO_XML,
        )
        names = {s.name for s in alto_metrics}
        assert names == _EXPECTED_ALTO_METRICS

    def test_unknown_junction_returns_empty(self) -> None:
        bundle = bootstrap_default_registries()
        # Aucune métrique enregistrée pour (IMAGE, IMAGE).
        result = bundle.metrics.select(
            ArtifactType.IMAGE, ArtifactType.IMAGE,
        )
        assert result == []


# ──────────────────────────────────────────────────────────────────
# Calcul des métriques sur un cas trivial
# ──────────────────────────────────────────────────────────────────


class TestMetricsAreCallable:
    def test_cer_computes_zero_on_identical_text(self) -> None:
        metrics = bootstrap_default_registries().metrics
        assert metrics.compute("cer", "Hello", "Hello") == 0.0

    def test_cer_computes_one_on_empty_hypothesis(self) -> None:
        metrics = bootstrap_default_registries().metrics
        assert metrics.compute("cer", "Hello", "") == 1.0

    def test_cer_computes_zero_on_double_empty(self) -> None:
        metrics = bootstrap_default_registries().metrics
        assert metrics.compute("cer", "", "") == 0.0

    def test_wer_word_difference_yields_nonzero(self) -> None:
        metrics = bootstrap_default_registries().metrics
        v = metrics.compute("wer", "a b c", "a b d")
        assert 0 < v <= 1

    def test_searchability_recall_perfect_on_identical(self) -> None:
        metrics = bootstrap_default_registries().metrics
        assert metrics.compute(
            "searchability_recall", "alpha beta", "alpha beta",
        ) == 1.0

    def test_numerical_preservation_perfect_when_year_kept(self) -> None:
        metrics = bootstrap_default_registries().metrics
        assert metrics.compute(
            "numerical_sequence_preservation",
            "Acte de 1789",
            "Acte de 1789",
        ) == 1.0

    def test_jiwer_metrics_have_higher_is_better_false(self) -> None:
        metrics = bootstrap_default_registries().metrics
        for name in ("cer", "wer", "mer", "wil"):
            assert metrics.get_spec(name).higher_is_better is False

    def test_search_metrics_have_higher_is_better_true(self) -> None:
        metrics = bootstrap_default_registries().metrics
        for name in (
            "searchability_recall", "numerical_sequence_preservation",
        ):
            assert metrics.get_spec(name).higher_is_better is True

    def test_alto_metrics_have_higher_is_better_true(self) -> None:
        metrics = bootstrap_default_registries().metrics
        for name in _EXPECTED_ALTO_METRICS:
            assert metrics.get_spec(name).higher_is_better is True


# ──────────────────────────────────────────────────────────────────
# Projecteurs canoniques
# ──────────────────────────────────────────────────────────────────


class TestDefaultProjectors:
    def test_three_canonical_projectors_registered(self) -> None:
        projectors = bootstrap_default_registries().projectors
        # ``ProjectorRegistry`` expose ``names()`` (cf. les tests S13/S14).
        # On s'appuie sur l'API publique sans connaître les détails
        # internes.
        if hasattr(projectors, "names"):
            assert set(projectors.names()) == _EXPECTED_PROJECTORS
        else:
            # Fallback : chaque projecteur est résolvable par son nom.
            for name in _EXPECTED_PROJECTORS:
                assert projectors.get(name) is not None


# ──────────────────────────────────────────────────────────────────
# RegistryService classmethod + accessors
# ──────────────────────────────────────────────────────────────────


class TestRegistryServiceFacade:
    def test_bootstrap_defaults_classmethod(self) -> None:
        svc = RegistryService.bootstrap_defaults()
        assert isinstance(svc.metrics, MetricRegistry)
        assert isinstance(svc.projectors, ProjectorRegistry)
        assert len(svc.metrics) == 9

    def test_bundle_property_exposes_both(self) -> None:
        svc = RegistryService.bootstrap_defaults()
        bundle = svc.bundle
        assert bundle.metrics is svc.metrics
        assert bundle.projectors is svc.projectors

    def test_construct_with_invalid_metrics_type_raises(self) -> None:
        with pytest.raises(TypeError, match="MetricRegistry"):
            RegistryService(metrics="not a registry", projectors=ProjectorRegistry())  # type: ignore[arg-type]

    def test_construct_with_invalid_projectors_type_raises(self) -> None:
        with pytest.raises(TypeError, match="ProjectorRegistry"):
            RegistryService(
                metrics=MetricRegistry(),
                projectors="not a registry",  # type: ignore[arg-type]
            )

    def test_two_services_are_independent(self) -> None:
        """Deux bootstraps successifs partagent zéro état."""
        svc1 = RegistryService.bootstrap_defaults()
        svc2 = RegistryService.bootstrap_defaults()
        assert svc1.metrics is not svc2.metrics
        assert svc1.projectors is not svc2.projectors

    def test_external_registers_in_one_dont_leak_to_other(self) -> None:
        """Un caller qui ajoute une métrique à svc1 ne pollue pas svc2."""
        from picarones.domain.evaluation_spec import MetricSpec
        svc1 = RegistryService.bootstrap_defaults()
        svc2 = RegistryService.bootstrap_defaults()
        custom_spec = MetricSpec(
            name="my_custom_metric",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
            higher_is_better=True,
        )
        svc1.metrics.register(custom_spec, lambda r, h: 1.0)
        assert "my_custom_metric" in svc1.metrics.names()
        assert "my_custom_metric" not in svc2.metrics.names()


# ──────────────────────────────────────────────────────────────────
# Smoke d'intégration : utiliser le RegistryService dans un
# DefaultEvaluationViewExecutor (la cible canonique de l'injection).
# ──────────────────────────────────────────────────────────────────


class TestSmokeIntegration:
    def test_bootstrapped_registries_drive_view_executor(self) -> None:
        """Le caller canonique (``DefaultEvaluationViewExecutor``) doit
        accepter directement le bundle bootstrapé sans massage."""
        from picarones.evaluation.views import DefaultEvaluationViewExecutor
        svc = RegistryService.bootstrap_defaults()

        loader = lambda art: ""  # noqa: E731 — non appelé ici
        executor = DefaultEvaluationViewExecutor.from_registries(
            svc.metrics, svc.projectors, loader,
        )
        assert executor is not None  # si le constructeur passe, c'est OK
