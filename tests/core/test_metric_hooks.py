"""Tests du système de profils + registre de hooks (chantier 2 post-Sprint 97).

Couvre :

- :mod:`picarones.evaluation.metric_hooks` : profils, registre, décorateurs,
  sélection par profil, exécution avec gestion d'erreurs.
- :mod:`picarones.measurements.builtin_hooks` : enregistre les 12+12 hooks
  historiques sur le profil ``standard``.
- Rétrocompat : les fonctions privées ``_aggregate_*`` et
  ``_calibration_from_engine_result`` restent accessibles depuis
  ``picarones.measurements.runner`` (tests Sprint 13/42).
- Le profil ``standard`` (défaut) couvre **exactement** les 12 hooks
  documentaires et 12 agrégateurs historiques.
- Le profil ``minimal`` n'active aucun hook (bench rapide).
- Un profil inconnu lève ``ValueError``.
"""

from __future__ import annotations


import pytest


# ──────────────────────────────────────────────────────────────────────────
# 1. Profils : constantes + validation
# ──────────────────────────────────────────────────────────────────────────


class TestProfiles:
    def test_known_profiles_complete(self):
        from picarones.evaluation.metric_hooks import KNOWN_PROFILES

        assert KNOWN_PROFILES == frozenset({
            "minimal", "standard", "philological", "diagnostics",
            "economics", "pipeline", "full",
        })

    def test_validate_profile_accepts_known(self):
        from picarones.evaluation.metric_hooks import validate_profile

        for p in ["minimal", "standard", "philological", "diagnostics",
                  "economics", "pipeline", "full"]:
            validate_profile(p)  # ne lève pas

    def test_validate_profile_rejects_unknown(self):
        from picarones.evaluation.metric_hooks import validate_profile

        with pytest.raises(ValueError, match="profil inconnu"):
            validate_profile("philolagic")

    def test_validate_profile_rejects_empty(self):
        from picarones.evaluation.metric_hooks import validate_profile

        with pytest.raises(ValueError):
            validate_profile("")


# ──────────────────────────────────────────────────────────────────────────
# 2. Registre des hooks builtin
# ──────────────────────────────────────────────────────────────────────────


class TestBuiltinHooksRegistration:
    def test_twelve_document_hooks_registered(self):
        # Import déclenche l'enregistrement via décorateurs.
        import picarones.measurements.builtin_hooks  # noqa: F401
        from picarones.evaluation.metric_hooks import _all_document_hook_names

        names = set(_all_document_hook_names())
        expected = {
            "confusion", "char_scores", "taxonomy", "structure",
            "image_quality", "line_metrics", "hallucination",
            "calibration", "philological", "searchability",
            "numerical_sequences", "readability",
        }
        assert expected.issubset(names), f"manquants : {expected - names}"

    def test_twelve_corpus_aggregators_registered(self):
        import picarones.measurements.builtin_hooks  # noqa: F401
        from picarones.evaluation.metric_hooks import _all_corpus_aggregator_names

        names = set(_all_corpus_aggregator_names())
        expected = {
            "confusion", "char_scores", "taxonomy", "structure",
            "image_quality", "line_metrics", "hallucination",
            "calibration", "philological", "searchability",
            "numerical_sequences", "readability",
        }
        assert expected.issubset(names), f"manquants : {expected - names}"

    def test_standard_profile_activates_all_hooks(self):
        import picarones.measurements.builtin_hooks  # noqa: F401
        from picarones.evaluation.metric_hooks import (
            select_corpus_aggregators, select_document_hooks,
        )

        doc_hooks = select_document_hooks("standard")
        agg_hooks = select_corpus_aggregators("standard")
        assert len(doc_hooks) == 12, [h.name for h in doc_hooks]
        assert len(agg_hooks) == 12, [a.name for a in agg_hooks]

    def test_minimal_profile_activates_zero_hooks(self):
        import picarones.measurements.builtin_hooks  # noqa: F401
        from picarones.evaluation.metric_hooks import (
            select_corpus_aggregators, select_document_hooks,
        )

        assert select_document_hooks("minimal") == []
        assert select_corpus_aggregators("minimal") == []

    def test_standard_attribute_names_match_documentresult(self):
        """Les attributs déclarés par les hooks doivent correspondre aux
        champs réels du DocumentResult — sinon le runner planterait à
        l'instanciation du dataclass."""
        import picarones.measurements.builtin_hooks  # noqa: F401
        from dataclasses import fields

        from picarones.evaluation.metric_hooks import select_document_hooks
        from picarones.evaluation.benchmark_result import DocumentResult

        doc_fields = {f.name for f in fields(DocumentResult)}
        for hook in select_document_hooks("standard"):
            assert hook.attribute in doc_fields, (
                f"hook '{hook.name}' a attribute='{hook.attribute}' "
                f"qui n'est pas un champ du DocumentResult"
            )

    def test_aggregator_attribute_names_match_enginereport(self):
        import picarones.measurements.builtin_hooks  # noqa: F401
        from dataclasses import fields

        from picarones.evaluation.metric_hooks import select_corpus_aggregators
        from picarones.evaluation.benchmark_result import EngineReport

        report_fields = {f.name for f in fields(EngineReport)}
        for agg in select_corpus_aggregators("standard"):
            assert agg.attribute in report_fields, (
                f"agrégateur '{agg.name}' a attribute='{agg.attribute}' "
                f"qui n'est pas un champ du EngineReport"
            )


# ──────────────────────────────────────────────────────────────────────────
# 3. run_document_hooks : exécution avec gestion d'erreurs
# ──────────────────────────────────────────────────────────────────────────


class _MockEngineResult:
    """Mock d'EngineResult pour tester sans dépendance OCR."""

    def __init__(self, *, success=True, text="hello world", token_confidences=None):
        self.success = success
        self.text = text if success else ""
        self.error = None if success else "boom"
        self.token_confidences = token_confidences


class TestRunDocumentHooks:
    def test_minimal_profile_returns_empty_dict(self):
        from picarones.evaluation.metric_hooks import run_document_hooks

        result = run_document_hooks(
            "minimal",
            ground_truth="hello world",
            hypothesis="hello world",
            image_path="/tmp/x.png",
            corpus_lang="fr",
            ocr_result=_MockEngineResult(),
        )
        assert result == {}

    def test_hook_exception_does_not_propagate(self, caplog):
        """Un hook qui lève doit être loggé en warning, pas faire
        échouer le calcul des autres hooks."""
        import picarones.evaluation.metric_hooks as mh

        # Crée un profil de test isolé via un hook qui lève
        custom_profile_name = "standard"

        @mh.register_document_metric(
            name="failing_test_hook_chantier2",
            attribute="image_path",  # peu importe — on vérifie qu'il rate
            profiles=(custom_profile_name,),
        )
        def _fail(**_):
            raise RuntimeError("intentional failure")

        with caplog.at_level("WARNING"):
            result = mh.run_document_hooks(
                custom_profile_name,
                ground_truth="x",
                hypothesis="x",
                image_path="/tmp/x.png",
                corpus_lang="fr",
                ocr_result=_MockEngineResult(),
            )
        # Le hook a échoué donc son attribut n'est pas dans le résultat
        assert "image_path" not in result or result.get("image_path") != "RAISED"
        # Vérification : le warning explicite est bien apparu
        assert any(
            "failing_test_hook_chantier2" in r.message and "fonctionnalité dégradée" in r.message
            for r in caplog.records
        )

    def test_requires_success_skips_failed_ocr(self):
        """Un hook ``requires_success=True`` ne doit pas être appelé si
        ``ocr_result.success`` est False."""
        import picarones.evaluation.metric_hooks as mh

        called = []

        @mh.register_document_metric(
            name="needs_success_chantier2",
            attribute="image_path",
            profiles=("standard",),
            requires_success=True,
        )
        def _hook(**kwargs):
            called.append(True)
            return "called"

        # Avec OCR échoué, le hook ne doit pas être appelé
        mh.run_document_hooks(
            "standard",
            ground_truth="x",
            hypothesis="",
            image_path="/tmp/x.png",
            corpus_lang="fr",
            ocr_result=_MockEngineResult(success=False),
        )
        assert called == []  # hook sauté

    def test_requires_token_confidences_skips_when_absent(self):
        """Un hook ``requires_token_confidences=True`` doit être sauté
        quand ``ocr_result.token_confidences`` est None."""
        import picarones.evaluation.metric_hooks as mh

        called = []

        @mh.register_document_metric(
            name="needs_tokens_chantier2",
            attribute="image_path",
            profiles=("standard",),
            requires_token_confidences=True,
        )
        def _hook(**_):
            called.append(True)

        mh.run_document_hooks(
            "standard",
            ground_truth="x",
            hypothesis="x",
            image_path="/tmp/x.png",
            corpus_lang="fr",
            ocr_result=_MockEngineResult(token_confidences=None),
        )
        assert called == []
class TestDecoratorIdempotence:
    def test_register_same_func_twice_is_silent(self):
        """Ré-import d'un module en test ne doit pas lever sur le
        décorateur déjà appliqué."""
        from picarones.evaluation.metric_hooks import register_document_metric

        @register_document_metric(
            name="reimport_test_chantier2",
            attribute="image_path",
            profiles=("standard",),
        )
        def _hook(**_):
            return None

        # Re-application (simule ré-import) → pas d'erreur
        decorator = register_document_metric(
            name="reimport_test_chantier2",
            attribute="image_path",
            profiles=("standard",),
        )
        result = decorator(_hook)
        assert result is _hook

    def test_register_different_func_same_name_raises(self):
        from picarones.evaluation.metric_hooks import register_document_metric

        @register_document_metric(
            name="conflict_test_chantier2",
            attribute="image_path",
            profiles=("standard",),
        )
        def _hook_a(**_):
            return None

        with pytest.raises(ValueError, match="déjà enregistré"):
            @register_document_metric(
                name="conflict_test_chantier2",
                attribute="image_path",
                profiles=("standard",),
            )
            def _hook_b(**_):
                return None

    def test_register_unknown_profile_raises(self):
        from picarones.evaluation.metric_hooks import register_document_metric

        with pytest.raises(ValueError, match="profils inconnus"):
            @register_document_metric(
                name="bad_profile_chantier2",
                attribute="image_path",
                profiles=("philolagic",),
            )
            def _hook(**_):
                return None
