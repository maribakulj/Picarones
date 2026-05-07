"""Tests Sprint 34 — registre typé de métriques (Phase 0.3).

Vérifie :

1. ``register_metric`` accepte les métriques typées et les expose via
   ``all_metrics`` / ``get_metric`` / ``select_metrics``.
2. La sélection par signature de types est exacte (pas de coercion).
3. ``compute_at_junction`` calcule toutes les métriques applicables et
   tolère les erreurs d'une métrique sans casser les autres.
4. Les métriques natives (``builtin_metrics``) produisent les mêmes
   valeurs que ``jiwer`` directement (parité numérique avec
   ``compute_metrics`` legacy).
5. Le double enregistrement avec le même nom est interdit.
6. Une signature à 1 ou 3 éléments est rejetée.
7. Le stub typé hétérogène ``(TEXT, ALTO)`` se calcule sans erreur.
"""

from __future__ import annotations

import pytest

from picarones.core.metric_registry import (
    MetricSpec,
    all_metrics,
    compute_at_junction,
    get_metric,
    register_metric,
    select_metrics,
)
from picarones.domain.artifacts import ArtifactType


# Force l'import du module qui enregistre les métriques natives. Les
# tests s'exécutent avec ce registre peuplé ; on n'utilise pas
# ``_reset_registry_for_tests`` parce qu'on veut justement tester l'état
# par défaut visible par le runner en production.
import picarones.measurements.builtin_metrics  # noqa: F401


# ──────────────────────────────────────────────────────────────────────────
# 1 & 2. Enregistrement et sélection par signature
# ──────────────────────────────────────────────────────────────────────────


class TestRegistryBasics:
    def test_builtin_metrics_loaded(self) -> None:
        names = {spec.name for spec in all_metrics()}
        assert {"cer", "wer", "mer", "wil"} <= names

    def test_get_metric_returns_spec(self) -> None:
        spec = get_metric("cer")
        assert isinstance(spec, MetricSpec)
        assert spec.input_types == (ArtifactType.TEXT, ArtifactType.TEXT)
        assert spec.higher_is_better is False

    def test_get_metric_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_metric("definitely_not_registered_42")

    def test_select_text_text_includes_cer_wer(self) -> None:
        selected = select_metrics((ArtifactType.TEXT, ArtifactType.TEXT))
        names = {spec.name for spec in selected}
        assert "cer" in names
        assert "wer" in names

    def test_select_alto_alto_excludes_text_metrics(self) -> None:
        selected = select_metrics((ArtifactType.ALTO, ArtifactType.ALTO))
        names = {spec.name for spec in selected}
        assert "cer" not in names
        assert "wer" not in names

    def test_select_text_alto_returns_heterogeneous_metric(self) -> None:
        selected = select_metrics((ArtifactType.TEXT, ArtifactType.ALTO))
        names = {spec.name for spec in selected}
        assert "text_preservation_after_reconstruction" in names

    def test_select_returns_empty_when_no_match(self) -> None:
        # ENTITIES → READING_ORDER : aucune métrique enregistrée à ce jour
        assert select_metrics((ArtifactType.ENTITIES, ArtifactType.READING_ORDER)) == []


# ──────────────────────────────────────────────────────────────────────────
# 3. compute_at_junction — calcul orchestré et résilience
# ──────────────────────────────────────────────────────────────────────────


class TestComputeAtJunction:
    def test_returns_all_applicable_metrics(self) -> None:
        out = compute_at_junction(
            "hello world",
            "hello wrld",
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        # Au moins les 4 métriques natives doivent être présentes
        for name in ("cer", "wer", "mer", "wil"):
            assert name in out
            assert isinstance(out[name], float)
            assert 0.0 <= out[name] <= 1.0

    def test_empty_dict_when_no_metric_applies(self) -> None:
        # Un type d'artefact sans métrique enregistrée
        out = compute_at_junction(
            [], [],
            (ArtifactType.ENTITIES, ArtifactType.READING_ORDER),
        )
        assert out == {}

    def test_skip_on_error_default_true(self) -> None:
        """Une métrique qui lève est ignorée, les autres tournent."""

        @register_metric(
            name="_test_always_raises",
            input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
            description="Test only",
        )
        def _broken(ref: str, hyp: str) -> float:
            raise RuntimeError("intentional failure")

        try:
            out = compute_at_junction(
                "abc", "abd",
                (ArtifactType.TEXT, ArtifactType.TEXT),
            )
            assert "_test_always_raises" not in out
            # Les natives sont toujours là
            assert "cer" in out
        finally:
            # Nettoyage manuel — pas d'API publique, on écrit dans le dict.
            from picarones.core.metric_registry import _METRIC_REGISTRY

            _METRIC_REGISTRY.pop("_test_always_raises", None)

    def test_skip_on_error_false_propagates(self) -> None:
        @register_metric(
            name="_test_propagates",
            input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
        )
        def _broken(ref: str, hyp: str) -> float:
            raise RuntimeError("propagate me")

        try:
            with pytest.raises(RuntimeError, match="propagate me"):
                compute_at_junction(
                    "x", "y",
                    (ArtifactType.TEXT, ArtifactType.TEXT),
                    skip_on_error=False,
                )
        finally:
            from picarones.core.metric_registry import _METRIC_REGISTRY

            _METRIC_REGISTRY.pop("_test_propagates", None)


# ──────────────────────────────────────────────────────────────────────────
# 4. Parité numérique avec compute_metrics legacy
# ──────────────────────────────────────────────────────────────────────────


class TestParityWithLegacy:
    """Le critère « rapport identique octet par octet » du Sprint 34
    se traduit en : les métriques enregistrées produisent les mêmes
    chiffres que ``compute_metrics`` historique sur les mêmes paires."""

    @pytest.mark.parametrize(
        "ref,hyp",
        [
            ("hello world", "hello wrld"),
            ("Le manuscrit médiéval", "Le manuscript medieval"),
            ("abcdef", "abcdef"),  # cas parfait
            ("a", "b"),
        ],
    )
    def test_cer_matches_compute_metrics(self, ref: str, hyp: str) -> None:
        from picarones.measurements.metrics import compute_metrics

        legacy = compute_metrics(ref, hyp)
        registered = compute_at_junction(
            ref, hyp,
            (ArtifactType.TEXT, ArtifactType.TEXT),
        )
        # On compare au CER brut, pas aux variantes (NFC, caseless,
        # diplomatic) qui sont des métriques distinctes non encore
        # enregistrées.
        assert registered["cer"] == pytest.approx(legacy.cer, abs=1e-9)
        assert registered["wer"] == pytest.approx(legacy.wer, abs=1e-9)
        assert registered["mer"] == pytest.approx(legacy.mer, abs=1e-9)
        assert registered["wil"] == pytest.approx(legacy.wil, abs=1e-9)


# ──────────────────────────────────────────────────────────────────────────
# 5 & 6. Garde-fous d'enregistrement
# ──────────────────────────────────────────────────────────────────────────


class TestRegistrationGuards:
    def test_double_register_same_name_raises(self) -> None:
        @register_metric(
            name="_test_duplicate",
            input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
        )
        def _first(ref: str, hyp: str) -> float:
            return 0.0

        try:
            with pytest.raises(ValueError, match="déjà enregistrée"):

                @register_metric(
                    name="_test_duplicate",
                    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
                )
                def _second(ref: str, hyp: str) -> float:
                    return 1.0
        finally:
            from picarones.core.metric_registry import _METRIC_REGISTRY

            _METRIC_REGISTRY.pop("_test_duplicate", None)

    def test_re_register_same_function_tolerated(self) -> None:
        """Ré-importer le module ne doit pas lever (cas réel : pytest
        recharge un module entre fichiers de tests)."""

        def _func(ref: str, hyp: str) -> float:
            return 0.0

        register_metric(
            name="_test_idempotent",
            input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
        )(_func)
        # Second appel avec la même fonction → tolérance
        register_metric(
            name="_test_idempotent",
            input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
        )(_func)

        from picarones.core.metric_registry import _METRIC_REGISTRY

        _METRIC_REGISTRY.pop("_test_idempotent", None)

    def test_input_types_must_be_pair(self) -> None:
        with pytest.raises(ValueError, match="couple"):

            @register_metric(
                name="_bad_arity_3",
                input_types=(  # type: ignore[arg-type]
                    ArtifactType.TEXT,
                    ArtifactType.TEXT,
                    ArtifactType.TEXT,
                ),
            )
            def _f(a, b, c):
                return 0.0


# ──────────────────────────────────────────────────────────────────────────
# 7. Stub TEXT → ALTO opérationnel
# ──────────────────────────────────────────────────────────────────────────


class TestHeterogeneousJunction:
    def test_text_preservation_runs(self) -> None:
        ref = "le manuscrit médiéval"
        alto = (
            '<?xml version="1.0"?><alto>'
            '<String CONTENT="le"/><String CONTENT="manuscrit"/>'
            '<String CONTENT="médiéval"/></alto>'
        )

        out = compute_at_junction(
            ref, alto,
            (ArtifactType.TEXT, ArtifactType.ALTO),
        )
        assert "text_preservation_after_reconstruction" in out
        assert out["text_preservation_after_reconstruction"] == pytest.approx(1.0)

    def test_text_preservation_partial(self) -> None:
        ref = "alpha beta gamma"
        alto = '<?xml version="1.0"?><alto><String CONTENT="alpha"/></alto>'

        score = compute_at_junction(
            ref, alto,
            (ArtifactType.TEXT, ArtifactType.ALTO),
        )["text_preservation_after_reconstruction"]
        # 1 token sur 3 préservé
        assert score == pytest.approx(1 / 3, abs=1e-9)

    def test_text_preservation_metric_marked_higher_is_better(self) -> None:
        spec = get_metric("text_preservation_after_reconstruction")
        assert spec.higher_is_better is True
