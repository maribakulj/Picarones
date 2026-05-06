"""Sprint A14-S5 — ``MetricRegistry`` instancié explicitement.

Vérifie le contrat critique du S5 : pas de singleton global, pas
de side-effect d'import, association explicite ``MetricSpec ↔
Callable``, sélection par signature de types.

Anti-pattern testé négativement : ``import picarones.evaluation``
ne doit PAS auto-enregistrer de métrique.
"""

from __future__ import annotations

import pytest

from picarones.domain import ArtifactType, MetricSpec
from picarones.evaluation.registry import (
    MetricNotFoundError,
    MetricRegistrationError,
    MetricRegistry,
)


def _cer(reference: str, hypothesis: str) -> float:
    """Stub CER pour les tests."""
    return 0.0 if reference == hypothesis else 1.0


def _wer(reference: str, hypothesis: str) -> float:
    return 0.0 if reference == hypothesis else 1.0


def _ner_f1(ref_entities: list[dict], hyp_entities: list[dict]) -> float:
    return 1.0


# ──────────────────────────────────────────────────────────────────────
# Instanciation et état initial
# ──────────────────────────────────────────────────────────────────────


class TestEmptyRegistry:
    def test_starts_empty(self) -> None:
        reg = MetricRegistry()
        assert len(reg) == 0
        assert reg.names() == []

    def test_unknown_metric_raises(self) -> None:
        reg = MetricRegistry()
        with pytest.raises(MetricNotFoundError):
            reg.get_spec("cer")
        with pytest.raises(MetricNotFoundError):
            reg.get_callable("cer")


# ──────────────────────────────────────────────────────────────────────
# Enregistrement
# ──────────────────────────────────────────────────────────────────────


class TestRegistration:
    def test_register_one_metric(self) -> None:
        reg = MetricRegistry()
        spec = MetricSpec(
            name="cer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
        )
        reg.register(spec, _cer)
        assert "cer" in reg
        assert len(reg) == 1
        assert reg.get_spec("cer") is spec
        assert reg.get_callable("cer") is _cer

    def test_register_non_callable_raises(self) -> None:
        reg = MetricRegistry()
        spec = MetricSpec(
            name="cer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
        )
        with pytest.raises(MetricRegistrationError, match="callable"):
            reg.register(spec, "not_a_function")  # type: ignore[arg-type]

    def test_duplicate_name_with_different_func_raises(self) -> None:
        reg = MetricRegistry()
        spec = MetricSpec(
            name="cer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
        )
        reg.register(spec, _cer)
        with pytest.raises(MetricRegistrationError, match="déjà enregistrée"):
            reg.register(spec, _wer)  # même spec, autre callable

    def test_idempotent_re_registration(self) -> None:
        """Re-enregistrer la même spec + même callable est silencieux
        (utile pour les tests qui re-instancient le service)."""
        reg = MetricRegistry()
        spec = MetricSpec(
            name="cer",
            input_types=(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT),
        )
        reg.register(spec, _cer)
        reg.register(spec, _cer)  # ne lève pas
        assert len(reg) == 1


# ──────────────────────────────────────────────────────────────────────
# Sélection par signature de types
# ──────────────────────────────────────────────────────────────────────


class TestSelectByTypes:
    def _filled_registry(self) -> MetricRegistry:
        reg = MetricRegistry()
        reg.register(
            MetricSpec(name="cer", input_types=(
                ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT,
            )),
            _cer,
        )
        reg.register(
            MetricSpec(name="wer", input_types=(
                ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT,
            )),
            _wer,
        )
        reg.register(
            MetricSpec(name="ner_f1", input_types=(
                ArtifactType.ENTITIES, ArtifactType.ENTITIES,
            ), higher_is_better=True),
            _ner_f1,
        )
        return reg

    def test_select_text_text(self) -> None:
        reg = self._filled_registry()
        selected = reg.select(ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT)
        names = sorted(s.name for s in selected)
        assert names == ["cer", "wer"]

    def test_select_entities(self) -> None:
        reg = self._filled_registry()
        selected = reg.select(ArtifactType.ENTITIES, ArtifactType.ENTITIES)
        assert [s.name for s in selected] == ["ner_f1"]

    def test_select_no_match(self) -> None:
        reg = self._filled_registry()
        selected = reg.select(ArtifactType.IMAGE, ArtifactType.IMAGE)
        assert selected == []

    def test_select_distinguishes_text_subtypes(self) -> None:
        """Important : RAW_TEXT et CORRECTED_TEXT sont des types distincts.
        Une métrique enregistrée pour (RAW_TEXT, RAW_TEXT) ne s'applique
        pas automatiquement à (CORRECTED_TEXT, RAW_TEXT)."""
        reg = self._filled_registry()
        selected = reg.select(ArtifactType.CORRECTED_TEXT, ArtifactType.RAW_TEXT)
        assert selected == []


# ──────────────────────────────────────────────────────────────────────
# Calcul
# ──────────────────────────────────────────────────────────────────────


class TestCompute:
    def test_compute_named(self) -> None:
        reg = MetricRegistry()
        reg.register(
            MetricSpec(name="cer", input_types=(
                ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT,
            )),
            _cer,
        )
        assert reg.compute("cer", "hello", "hello") == 0.0
        assert reg.compute("cer", "hello", "world") == 1.0

    def test_compute_unknown_raises(self) -> None:
        reg = MetricRegistry()
        with pytest.raises(MetricNotFoundError):
            reg.compute("missing", "x", "y")

    def test_compute_at_junction_runs_all_applicable(self) -> None:
        reg = MetricRegistry()
        reg.register(
            MetricSpec(name="cer", input_types=(
                ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT,
            )),
            _cer,
        )
        reg.register(
            MetricSpec(name="wer", input_types=(
                ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT,
            )),
            _wer,
        )
        reg.register(
            MetricSpec(name="ner_f1", input_types=(
                ArtifactType.ENTITIES, ArtifactType.ENTITIES,
            )),
            _ner_f1,
        )
        out = reg.compute_at_junction(
            "hello", "hello",
            ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT,
        )
        assert set(out.keys()) == {"cer", "wer"}
        assert out["cer"] == 0.0
        assert "ner_f1" not in out  # mauvaise signature

    def test_compute_at_junction_propagates_exceptions(self) -> None:
        """Le S5 ne capture pas les exceptions des métriques.
        C'est l'EvaluationViewExecutor (S13) qui décidera quoi en
        faire dans son ProjectionReport."""
        def _broken(r: str, h: str) -> float:
            raise RuntimeError("boom")
        reg = MetricRegistry()
        reg.register(
            MetricSpec(name="broken", input_types=(
                ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT,
            )),
            _broken,
        )
        with pytest.raises(RuntimeError, match="boom"):
            reg.compute_at_junction(
                "x", "y",
                ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT,
            )


# ──────────────────────────────────────────────────────────────────────
# Anti-pattern : pas de singleton global
# ──────────────────────────────────────────────────────────────────────


class TestNoGlobalSingleton:
    def test_two_registries_are_independent(self) -> None:
        """Différence cruciale avec l'ancien
        ``picarones.core.metric_registry`` qui a un dict global :
        deux ``MetricRegistry()`` ne se partagent rien."""
        reg_a = MetricRegistry()
        reg_b = MetricRegistry()
        spec = MetricSpec(name="cer", input_types=(
            ArtifactType.RAW_TEXT, ArtifactType.RAW_TEXT,
        ))
        reg_a.register(spec, _cer)
        assert "cer" in reg_a
        assert "cer" not in reg_b
        assert len(reg_a) == 1
        assert len(reg_b) == 0
