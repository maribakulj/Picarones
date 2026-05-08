"""Test de stabilité de l'API publique de Picarones (Cercle 1).

Phase D du chantier de refonte en 3 cercles. Ce test est le **filet de
sécurité contractuel** documenté dans :doc:`docs/api-stable.md` : il
échoue dès qu'un nom listé dans le contrat de stabilité du Cercle 1
disparaît, change de type (class ↔ function), ou perd un argument
attendu.

Discipline
----------
Toute modification d'un test ici doit être accompagnée d'une mise à
jour de ``docs/api-stable.md`` et **justifiée par une RFC** si elle
casse la rétrocompat. Ce test est la traduction technique d'un
engagement public.

Si une PR doit ajouter un nom à l'API publique, suivre dans l'ordre :

1. Documenter le nom dans ``docs/api-stable.md``.
2. Ajouter le test correspondant ici.
3. Implémenter / exposer le nom.

Si une PR doit casser un nom de l'API publique :

1. RFC + bump majeur (``2.0.0``).
2. Mise à jour de ``docs/api-stable.md`` (suppression).
3. Mise à jour des tests ici.

Les noms historiques rétrocompat (Cercle 2 / Cercle 3 via shims) ne
sont **pas** couverts par ce test — ils ont leurs propres tests dans
``tests/test_phaseA_migration.py``, ``test_phaseB_migration.py``, etc.
"""

from __future__ import annotations

import importlib
import inspect

import pytest


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _get_attr(module_path: str, name: str):
    mod = importlib.import_module(module_path)
    assert hasattr(mod, name), (
        f"API publique cassée : {module_path}.{name} a disparu"
    )
    return getattr(mod, name)


def _assert_class(module_path: str, name: str, *, abstract: bool = False):
    obj = _get_attr(module_path, name)
    assert inspect.isclass(obj), (
        f"{module_path}.{name} : attendu class, obtenu {type(obj).__name__}"
    )
    if abstract:
        assert inspect.isabstract(obj) or hasattr(obj, "__abstractmethods__"), (
            f"{module_path}.{name} : attendu classe abstraite"
        )
    return obj


def _assert_function(module_path: str, name: str):
    obj = _get_attr(module_path, name)
    assert callable(obj), (
        f"{module_path}.{name} : attendu callable, obtenu {type(obj).__name__}"
    )
    return obj


# ──────────────────────────────────────────────────────────────────────────
# 1. picarones.evaluation.corpus — modèle Document/Corpus + GT multi-niveaux (canonique)
# ──────────────────────────────────────────────────────────────────────────


class TestCorpusApi:
    @pytest.mark.parametrize("name", [
        "Document", "Corpus",
        "TextGT", "AltoGT", "PageGT", "EntitiesGT", "ReadingOrderGT",
    ])
    def test_class_exists(self, name):
        _assert_class("picarones.evaluation.corpus", name)

    def test_load_corpus_from_directory_exists(self):
        _assert_function("picarones.evaluation.corpus", "load_corpus_from_directory")

    def test_gt_suffixes_constant(self):
        from picarones.domain.artifacts import ArtifactType
        from picarones.evaluation.corpus import GT_SUFFIXES

        assert isinstance(GT_SUFFIXES, dict)
        # Chacun des 5 niveaux GT (ArtifactType) doit avoir un suffixe
        for level in (
            ArtifactType.RAW_TEXT,
            ArtifactType.ALTO_XML,
            ArtifactType.PAGE_XML,
            ArtifactType.ENTITIES,
            ArtifactType.READING_ORDER,
        ):
            assert level in GT_SUFFIXES, (
                f"GT_SUFFIXES manque le niveau {level}"
            )


# ──────────────────────────────────────────────────────────────────────────
# 2. picarones.domain — BaseModule + ArtifactType (canoniques)
# ──────────────────────────────────────────────────────────────────────────


class TestModulesApi:
    def test_artifact_type_values(self):
        from picarones.domain.artifacts import ArtifactType

        names = {member.value for member in ArtifactType}
        # Phase 4-bis : ``ArtifactType`` canonique (``domain.artifacts``)
        # — 10 valeurs.  L'ancien set legacy (``image, text, alto, page,
        # entities, reading_order``) reste accessible via les aliases
        # ``TEXT``/``ALTO``/``PAGE`` qui pointent vers les valeurs
        # canoniques ``raw_text``/``alto_xml``/``page_xml``.  Les
        # aliases n'apparaissent pas dans cette itération (Python
        # masque les membres aliasés dans ``__members__`` itérable).
        assert names == {
            "image",
            "raw_text",
            "corrected_text",
            "alto_xml",
            "page_xml",
            "canonical_document",
            "entities",
            "reading_order",
            "alignment",
            "confidences",
        }

    def test_basemodule_is_abstract(self):
        cls = _assert_class("picarones.domain.module_protocol", "BaseModule")
        # Doit avoir `process` abstrait
        assert "process" in cls.__abstractmethods__ or hasattr(cls, "process")

    def test_basemodule_class_attributes(self):
        from picarones.domain.module_protocol import BaseModule

        # Contrat : ces attributs de classe sont lisibles depuis la base
        assert hasattr(BaseModule, "input_types")
        assert hasattr(BaseModule, "output_types")
        assert hasattr(BaseModule, "execution_mode")
        assert hasattr(BaseModule, "validate_inputs")
        assert hasattr(BaseModule, "validate_outputs")
        assert hasattr(BaseModule, "metadata")


# ──────────────────────────────────────────────────────────────────────────
# 3. picarones.evaluation.benchmark_result — modèles de résultats (canonique)
# ──────────────────────────────────────────────────────────────────────────


class TestResultsApi:
    @pytest.mark.parametrize("name", [
        "DocumentResult", "EngineReport", "BenchmarkResult",
    ])
    def test_class_exists(self, name):
        _assert_class("picarones.evaluation.benchmark_result", name)


# ──────────────────────────────────────────────────────────────────────────
# 4. picarones.measurements.metrics — métriques de base
# ──────────────────────────────────────────────────────────────────────────


class TestMetricsApi:
    def test_metrics_result_class(self):
        _assert_class("picarones.measurements.metrics", "MetricsResult")

    @pytest.mark.parametrize("name", [
        "compute_metrics", "aggregate_metrics",
    ])
    def test_function_exists(self, name):
        _assert_function("picarones.measurements.metrics", name)

    def test_compute_metrics_signature(self):
        """``compute_metrics(reference, hypothesis, char_exclude=None)`` est
        contractuel — les 2 premiers args sont positionnels, le 3ᵉ keyword."""
        from picarones.measurements.metrics import compute_metrics
        sig = inspect.signature(compute_metrics)
        params = list(sig.parameters.values())
        # Au moins 2 paramètres positionnels (reference, hypothesis)
        positional = [p for p in params
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                      and p.default is p.empty]
        assert len(positional) >= 2, (
            f"compute_metrics doit accepter >= 2 args positionnels — "
            f"signature actuelle : {sig}"
        )


# ──────────────────────────────────────────────────────────────────────────
# 5. picarones.measurements.runner — run_benchmark
# ──────────────────────────────────────────────────────────────────────────


class TestRunnerApi:
    def test_run_benchmark_exists(self):
        try:
            _assert_function("picarones.measurements.runner", "run_benchmark")
        except ImportError as exc:
            if "tqdm" in str(exc):
                pytest.skip("tqdm non installé en sandbox")
            raise

    def test_run_benchmark_keyword_args(self):
        """Les paramètres clés (corpus, engines, profile…) doivent rester
        accessibles. Ajout d'un argument requis = breaking change."""
        try:
            from picarones.measurements.runner import run_benchmark
        except ImportError as exc:
            if "tqdm" in str(exc):
                pytest.skip("tqdm non installé")
            raise
        sig = inspect.signature(run_benchmark)
        params = sig.parameters
        # Arguments contractuels — leur présence est garantie
        for name in [
            "corpus", "engines", "output_json", "show_progress",
            "char_exclude", "max_workers", "timeout_seconds",
            "profile",
        ]:
            assert name in params, (
                f"run_benchmark : argument '{name}' a disparu (signature : {sig})"
            )


# ──────────────────────────────────────────────────────────────────────────
# 6. (anciennement) ``picarones.pipeline.legacy_*`` — supprimé en Phase 7.D
# ──────────────────────────────────────────────────────────────────────────
# Les modules ``pipeline.legacy_runner``, ``legacy_pipeline_benchmark``,
# ``legacy_pipeline_comparison`` et ``measurements.pipeline_spec_loader``
# ont été supprimés en Phase 7.D (mai 2026). L'API canonique vit dans
# ``picarones.pipeline.executor`` (``PipelineExecutor``) et
# ``picarones.domain.pipeline_spec`` (``PipelineSpec``, ``PipelineStep``).


# ──────────────────────────────────────────────────────────────────────────
# 7. picarones.evaluation.metric_registry — registre typé (canonique)
# ──────────────────────────────────────────────────────────────────────────


class TestMetricRegistryApi:
    def test_metric_spec_class(self):
        _assert_class("picarones.evaluation.metric_registry", "MetricSpec")

    @pytest.mark.parametrize("name", [
        "register_metric", "get_metric", "all_metrics",
        "select_metrics", "compute_at_junction",
    ])
    def test_function_exists(self, name):
        _assert_function("picarones.evaluation.metric_registry", name)

    def test_register_metric_keyword_only(self):
        """``register_metric`` est exclusivement keyword-only sur ``name``,
        ``input_types`` etc. — décorateur factory."""
        from picarones.evaluation.metric_registry import register_metric
        sig = inspect.signature(register_metric)
        for name in ["name", "input_types", "description"]:
            assert name in sig.parameters, (
                f"register_metric : keyword '{name}' manquant"
            )


# ──────────────────────────────────────────────────────────────────────────
# 8. picarones.evaluation.metric_hooks — profils + registre de hooks (canonique)
# ──────────────────────────────────────────────────────────────────────────


class TestMetricHooksApi:
    @pytest.mark.parametrize("profile_name", [
        "PROFILE_MINIMAL", "PROFILE_STANDARD", "PROFILE_PHILOLOGICAL",
        "PROFILE_DIAGNOSTICS", "PROFILE_ECONOMICS", "PROFILE_PIPELINE",
        "PROFILE_FULL",
    ])
    def test_profile_constant_exists(self, profile_name):
        from picarones.evaluation import metric_hooks
        assert hasattr(metric_hooks, profile_name), (
            f"Profil {profile_name} disparu"
        )
        assert isinstance(getattr(metric_hooks, profile_name), str)

    def test_known_profiles_set(self):
        from picarones.evaluation.metric_hooks import KNOWN_PROFILES

        assert isinstance(KNOWN_PROFILES, frozenset)
        # Les 7 profils contractuels
        assert len(KNOWN_PROFILES) == 7

    @pytest.mark.parametrize("name", [
        "DocumentMetricHook", "CorpusMetricAggregator",
    ])
    def test_class_exists(self, name):
        _assert_class("picarones.evaluation.metric_hooks", name)

    @pytest.mark.parametrize("name", [
        "validate_profile",
        "register_document_metric", "register_corpus_aggregator",
        "select_document_hooks", "select_corpus_aggregators",
        "run_document_hooks", "run_corpus_aggregators",
    ])
    def test_function_exists(self, name):
        _assert_function("picarones.evaluation.metric_hooks", name)


# ──────────────────────────────────────────────────────────────────────────
# 9. picarones.measurements.builtin_metrics — CER/WER/MER/WIL natifs
# ──────────────────────────────────────────────────────────────────────────


class TestBuiltinMetricsApi:
    @pytest.mark.parametrize("name", [
        "cer", "wer", "mer", "wil",
        "text_preservation_after_reconstruction",
    ])
    def test_function_exists(self, name):
        _assert_function("picarones.measurements.builtin_metrics", name)


# ──────────────────────────────────────────────────────────────────────────
# 10. picarones.measurements.alto_metrics — métriques (ALTO, ALTO)
# ──────────────────────────────────────────────────────────────────────────


class TestAltoMetricsApi:
    def test_extract_text_from_alto(self):
        _assert_function("picarones.measurements.alto_metrics", "extract_text_from_alto")

    @pytest.mark.parametrize("name", [
        "alto_text_cer", "alto_text_wer",
        "alto_text_mer", "alto_text_wil",
    ])
    def test_alto_metric_function(self, name):
        _assert_function("picarones.measurements.alto_metrics", name)


# ──────────────────────────────────────────────────────────────────────────
# 11. picarones.web.jobs — JobStore (utilisé par web/)
# ──────────────────────────────────────────────────────────────────────────


class TestJobsApi:
    def test_job_store(self):
        _assert_class("picarones.web.jobs", "JobStore")

    @pytest.mark.parametrize("name", [
        "get_default_store", "reset_default_store",
    ])
    def test_function_exists(self, name):
        _assert_function("picarones.web.jobs", name)


# ──────────────────────────────────────────────────────────────────────────
# 12. Anti-régression : aucune fuite de Cercle 2/3 dans le Cercle 1
# ──────────────────────────────────────────────────────────────────────────


class TestCercle1IsLean:
    """``picarones/core/`` ne doit contenir que les modules Cercle 1 réels
    (les autres sont des shims). Ce test garde-fou empêche un module
    métrique d'être réintroduit dans le cœur sans RFC."""

    # Modules Cercle 1 — abstractions pures (corpus, contrats, registres).
    # Tout module avec de la logique métier (calcul, orchestration)
    # appartient au Cercle 2 (``measurements/``) ou au Cercle 3
    # (``extras/``, ``report/``).
    EXPECTED_CERCLE1: set[str] = set()
    # Phase 1 du retrait du legacy a déplacé `facts.py`,
    # `diff_utils.py` et `xml_utils.py` vers leurs canoniques
    # (`domain/facts.py`, `evaluation/_diff_utils.py`,
    # `formats/_xml_utils.py`).  Les fichiers `core/X.py`
    # restent comme shims re-export avec DeprecationWarning
    # (< 30 lignes), donc ne comptent plus comme "real_modules"
    # au sens de ce test.
    # Phase 4-bis a fait pareil pour `modules.py` (canonique :
    # `domain/module_protocol.py` + `domain/artifacts.py`).
    # Phase 4-ter a fait pareil pour `metric_registry.py`,
    # `metric_hooks.py` (canonique : `evaluation/metric_*.py`),
    # `metrics.py` (canonique : `evaluation/metric_result.py`)
    # et `results.py` (canonique :
    # `evaluation/benchmark_result.py`).
    # Phase 4-quater a fait pareil pour `corpus.py`
    # (canonique : `evaluation/corpus.py`).
    # Phase 5.C.batch7 a fait pareil pour `pipeline.py`
    # (canonique : `evaluation/pipeline.py`).  Désormais
    # ``core/`` ne contient plus que des shims < 30 lignes.

    def test_cercle1_files_lean(self):
        from pathlib import Path

        repo = Path(__file__).parent.parent.parent
        core_dir = repo / "picarones" / "core"

        real_modules = set()
        for path in core_dir.glob("*.py"):
            content = path.read_text(encoding="utf-8")
            n_lines = len(
                [line for line in content.splitlines() if line.strip()],
            )
            # Un shim a < 30 lignes ; un module Cercle 1 a > 30 lignes
            if n_lines > 30:
                real_modules.add(path.name)

        unexpected = real_modules - self.EXPECTED_CERCLE1
        assert not unexpected, (
            f"Modules non-Cercle 1 réintroduits dans core/ : {unexpected}. "
            "Soit les déplacer dans measurements/ (Cercle 2) ou extras/ "
            "(Cercle 3), soit ajouter à EXPECTED_CERCLE1 + api-stable.md "
            "via RFC."
        )

        missing = self.EXPECTED_CERCLE1 - real_modules
        assert not missing, (
            f"Modules Cercle 1 manquants : {missing}. Restaurer ou retirer "
            "de EXPECTED_CERCLE1."
        )


# ──────────────────────────────────────────────────────────────────────────
# 13. Doc api-stable.md présente et complète
# ──────────────────────────────────────────────────────────────────────────


class TestApiStableDoc:
    def test_doc_exists(self):
        from pathlib import Path

        # S60 — la doc a migré sous ``docs/reference/`` (Diataxis).
        path = (
            Path(__file__).parent.parent.parent
            / "docs"
            / "reference"
            / "api-stable.md"
        )
        assert path.exists(), "docs/reference/api-stable.md manquant"
        content = path.read_text(encoding="utf-8")
        # Présence des sections (1 par module canonique)
        for module in [
            "picarones.evaluation.corpus",
            "picarones.domain.artifacts",
            "picarones.domain.module_protocol",
            "picarones.evaluation.benchmark_result",
            "picarones.measurements.metrics",
            "picarones.measurements.runner",
            "picarones.evaluation.metric_registry",
            "picarones.evaluation.metric_hooks",
            "picarones.measurements.builtin_metrics",
            "picarones.measurements.alto_metrics",
            "picarones.web.jobs",
        ]:
            assert module in content, (
                f"docs/api-stable.md ne mentionne pas {module}"
            )

    def test_doc_mentions_stability_policy(self):
        from pathlib import Path

        path = (
            Path(__file__).parent.parent.parent
            / "docs"
            / "reference"
            / "api-stable.md"
        )
        content = path.read_text(encoding="utf-8")
        # Les sections clés du contrat
        assert "Politique de stabilité" in content
        assert "Ce que nous garantissons" in content
        assert "Bump majeur" in content
