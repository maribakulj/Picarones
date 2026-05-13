"""Garde-fou contre la croissance silencieuse des fichiers.

Chaque fichier listé dans :data:`FILE_BUDGETS` a un budget en lignes.
Si un fichier dépasse son budget, le test échoue et la PR est forcée
à choisir entre :

1. **Refactor** pour rentrer dans le budget (extraire un sous-module,
   factoriser, supprimer du code mort).
2. **Relever le budget délibérément** : modifier la valeur dans ce
   fichier en l'expliquant dans le message de commit. La hausse devient
   un acte conscient, plus une dérive silencieuse.

Calibration : snapshot v1.0.0 (2026-05-02), ``current + ~15 %`` de marge
pour l'évolution naturelle. Les god-modules historiques (statistics,
generator, runner) gardent un budget proche de leur taille actuelle ; le
choix de les dégonfler est une décision dédiée à un sprint de refactor,
pas un sous-produit de l'invariant.

Re-calibrer à chaque release tag.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# Format : chemin relatif → max_lines.
# Seuls les fichiers ≥ 400 lignes sont surveillés (les petits fichiers
# n'ont pas besoin de budget — leur croissance est gérée par les tests
# de couverture, pas par un seuil dur).
FILE_BUDGETS: dict[str, int] = {
    # Sprint D.1 (plan v2.0) — adapter de compat run_benchmark legacy
    # → BenchmarkService rewrite.  Module qui présente l'API
    # historique mono-call ``run_benchmark(corpus, engines, ...)``
    # consommée par les interfaces CLI/web.
    # Sprint D.2.b a ajouté ~260 LOC pour la branche resumable.
    # Sprint D.2.c-f a ajouté ~190 LOC : NER attach + over_normalization
    # + validate_profile.
    # Sprint H.2.c a retiré ``_ocr_only_to_spec`` legacy + simplifié
    # ``build_adapter_resolver`` (canonique uniquement).
    # Sprint H.4 — module renommé ``_legacy_runner_adapter`` →
    # ``benchmark_runner`` (drop le préfixe legacy : c'est l'entry
    # point canonique des interfaces vers ``BenchmarkService``).
    # Phase 2.3 du chantier post-rewrite — ajout de
    # ``_engine_config_for_fingerprint`` (~50 LOC) pour distinguer les
    # runs avec configs différentes (psm/lang/model/prompt) au niveau
    # du fichier partial.
    "picarones/app/services/benchmark_runner.py": 1750,  # actuel ~1700
    # --- God-modules : budget actuel + 15 % de marge.
    # Le rétrécissement sera l'objet d'un sprint de refactor dédié.
    # statistics.py (1128 lignes) a été éclaté en sous-package
    # ``picarones/measurements/statistics/`` lors du sprint
    # « découpage de statistics.py » (2026-05-02). Plus aucun fichier
    # de la famille ne dépasse 350 lignes, donc aucune entrée requise.
    # runner.py (1019 lignes) a été éclaté en sous-package
    # ``picarones/measurements/runner/`` lors du sprint
    # « découpage de runner.py » (2026-05-03).  Le sous-package a été
    # supprimé en Sprint D.6.b du plan v2.0 — son entrée dans
    # ``FILE_BUDGETS`` a été retirée.
    # --- Refactor (sprint « découpage de generator.py ») : passé de
    # 1063 à 431 lignes via extraction vers picarones/report/assets.py
    # et le sous-package picarones/report/report_data/. Budget serré
    # à 500 pour verrouiller le gain ; toute croissance > 500 sera
    # un signal pour redécouper.
    # Phase 5.E : ``report/generator.py`` est désormais un shim ;
    # canonique dans ``reports/html/generator.py``.
    "picarones/reports/html/generator.py": 550,        # actuel 471
    # --- Fichiers métier larges.
    # (Phase 7.D — ``reports/html/renderers/pipeline.py`` supprimé.)
    # Phase 4-ter : ``core/results.py`` est désormais un shim
    # (≤ 25 l).  Le contenu canonique vit dans ``evaluation/`` ;
    # même budget pour la même raison historique (modèles
    # BenchmarkResult/EngineReport/DocumentResult).
    # Phase 2.2 du chantier post-rewrite — ajout de
    # ``DocumentResult.from_dict``, ``EngineReport.from_dict``,
    # ``BenchmarkResult.from_dict`` et ``BenchmarkResult.from_json_object``
    # pour restaurer la fidélité du round-trip JSON (taxonomy,
    # hallucination, philological, etc.).
    "picarones/evaluation/benchmark_result.py": 880,      # actuel ~826
    # Phase 5.C : ``report/philological_render.py`` est désormais
    # un shim (≤ 25 l).  Le contenu canonique vit dans
    # ``reports/html/renderers/philological.py``.
    "picarones/reports/html/renderers/philological.py": 700,  # actuel 601
    # Sprint E.1 du plan v2.0 — module migré vers ``evaluation/metrics/``.
    "picarones/evaluation/metrics/modern_archives.py": 700,  # actuel 599
    # Sprint E.4 du plan v2.0 — migré vers ``evaluation/metrics/``.
    "picarones/evaluation/metrics/builtin_hooks.py": 700,  # actuel 590
    # Sprint E.5 du plan v2.0 — modules ``history`` et ``robustness``
    # migrés depuis ``measurements/`` vers la couche canonique.
    "picarones/evaluation/metrics/history.py": 720,        # actuel 615
    "picarones/evaluation/metrics/robustness.py": 850,     # actuel 742
    # Sprint S9 — helpers ``_sanitize_name_suffix`` / ``_ocr_adapter_name``
    # + registry ``_OCR_KWARGS_BUILDERS`` qui rendent impossible
    # l'oubli du ``name=`` côté factory web (fix collision resolver).
    "picarones/interfaces/web/benchmark_utils.py": 600,    # actuel 520
    # (Phase 7.D — ``pipeline/legacy_runner.py`` et
    # ``pipeline/legacy_pipeline_benchmark.py`` supprimés.)
    # Phase 8 — importers IIIF/Gallica déplacés vers ``adapters/corpus/``.
    "picarones/adapters/corpus/iiif.py": 675,             # actuel 567
    "picarones/adapters/corpus/gallica.py": 675,          # actuel 563
    # Sprint A14-S10 + Lot D — déplacés depuis measurements/.
    # L'ancien emplacement (shim) a été supprimé au Lot D ; seul le
    # canonique reste dans evaluation/metrics/.
    "picarones/evaluation/metrics/levers.py": 675,        # actuel 561
    "picarones/evaluation/metrics/inter_engine.py": 575,  # actuel 484
    "picarones/adapters/corpus/escriptorium.py": 650,     # actuel 553 (Phase 8)
    # Sprint A14-S1 — A.I.0 P0 : ajout de validated_path,
    # validated_prompt_filename, safe_report_name et compute_workspace_roots.
    # Ces helpers seront extraits dans ``picarones/web/path_security.py``
    # lors du Sprint S20 du rewrite ciblé (création couche app/services/).
    # Sprint F du plan v2.0 — déplacé vers ``interfaces/web/``.
    "picarones/interfaces/web/security.py": 850,  # actuel 751
    # Sprint A14-S8 — CorpusRunner introduit pour orchestrer les
    # pipelines composées sur un corpus avec backpressure / timeout
    # réel / annulation propre.  Budget stable, l'extension
    # ProcessPoolExecutor (S11) restera dans cette enveloppe.
    "picarones/pipeline/runner.py": 550,                  # actuel 462
    # Sprint A14-S28 — PipelineExecutor refondu pour consommer un
    # ExecutionPlan (run_plan) tout en gardant run(spec) comme sucre.
    # PipelinePlanner introduit pour transformer une PipelineSpec en
    # plan immuable (validation + bindings + jonctions de métriques).
    # Sprint A14-S47 — branchement ArtifactStore : +60 lignes (lookup
    # cache avant exec, persistance après succès, helpers privés).
    "picarones/pipeline/executor.py": 600,                # actuel 541
    "picarones/pipeline/planner.py": 465,                 # actuel 403
    # Sprint A14-S29 — ArtifactStore (ABC + 2 implémentations) avec
    # hash multi-paramètres pour adresser la critique d'audit n° 14
    # « hash multi-paramètres + reprise par hash ».
    "picarones/adapters/storage/artifact_store.py": 580,  # actuel 504
    # Sprint A14-S37 + S52 + S56 — JobStore SQLite : POST/GET/DELETE,
    # JobStoreError, schema_version table (S56) + busy_timeout 30s +
    # WAL mode pour les jobs concurrents.
    "picarones/adapters/storage/job_store.py": 500,       # actuel 421
    # Sprint A14-S41 — artifacts_index.jsonl séparé.
    "picarones/app/services/benchmark_service.py": 470,   # actuel 400
    # ``BaseLLMAdapter`` implémente le contrat ``StepExecutor``
    # (input_types, output_types, execute) en plus de complete().
    # Sprint S7 — descripteur ``_DeprecatedAttribute`` + alias
    # ``DEFAULT_CORRECTION_PROMPT`` (singulier) supprimés (period
    # de deprecation expirée à v2.0).
    "picarones/adapters/llm/base.py": 520,                # actuel ~440
    # Phase 4-quater : ``core/corpus.py`` est désormais un shim
    # (≤ 30 l).  Le contenu canonique vit dans ``evaluation/`` ;
    # même budget pour la même raison historique
    # (Document/Corpus/GTLevel + 5 payloads + load_corpus_from_directory).
    "picarones/evaluation/corpus.py": 600,                # actuel 533
    # Sprint H.1 du plan v2.0 — ``fixtures.py`` migré vers
    # ``evaluation/synthetic.py``.
    "picarones/evaluation/synthetic.py": 600,             # actuel 510
    # Phase 5.C.batch7 + Lot D : le shim
    # ``measurements/roman_numerals.py`` a été supprimé.  Seul le
    # canonique ``evaluation/metrics/roman_numerals.py`` reste.
    "picarones/evaluation/metrics/roman_numerals.py": 575,  # actuel 484
    # Sprint A14-S11 + Lot I — déplacés depuis extras/importers/.
    # Les shims ``extras/importers/{htr_united, huggingface,
    # _fallback_log}`` ont été supprimés au Lot I (mai 2026).
    "picarones/adapters/corpus/htr_united.py": 575,       # actuel 473
    "picarones/adapters/corpus/huggingface.py": 550,      # actuel 464
    # Sprint G du plan v2.0 — déplacé vers ``interfaces/cli/``.
    "picarones/interfaces/cli/_workflows.py": 550,  # actuel 469
    # ``__init__.py`` du legacy CLI — plus gros que les autres car il
    # contient les commandes ``info``, ``engines``, ``metrics``,
    # ``report``, ``demo``.
    "picarones/interfaces/cli/__init__.py": 500,    # actuel 396
    # Phase 4-ter : ``core/metric_hooks.py`` est désormais un shim
    # (≤ 80 l).  Le contenu canonique vit dans ``evaluation/`` ;
    # même budget pour la même raison historique (centralise les
    # hooks document/corpus, croissance maîtrisée).
    "picarones/evaluation/metric_hooks.py": 500,          # actuel 427
    # Phase 5.C.batch7 : ``measurements/numerical_sequences.py`` est
    # désormais un shim ; canonique dans
    # ``evaluation/metrics/numerical_sequences.py``.
    "picarones/evaluation/metrics/numerical_sequences.py": 500,  # actuel 428
    # Sprint A14-S9 + Lot D — déplacé depuis measurements/normalization.py.
    # Le shim a été supprimé au Lot D ; seul le canonique reste.
    "picarones/formats/text/normalization.py": 500,       # actuel 420
    # Phase 5.E : ``report/comparison.py`` est désormais un shim ;
    # canonique dans ``reports/html/comparison.py``.
    "picarones/reports/html/comparison.py": 500,       # actuel 414
    # --- Module mutualisé créé par le sprint des render helpers
    # (Sprint « consolidation des renderers » 2026-05-02). Budget
    # calibré sur la taille post-documentation des conventions.
    # Phase 5 : ``report/render_helpers.py`` est désormais un shim
    # (≤ 25 l).  Le contenu canonique vit dans
    # ``reports/_helpers/`` ; même budget pour la même raison
    # historique (consolidation des 25 helpers de couleur).
    "picarones/reports/_helpers/render_helpers.py": 480,  # actuel 428
    # --- Services applicatifs et orchestration du rewrite ciblé.
    # Budgets calibrés à current + 15 % de marge.  La CLI elle-même
    # reste mince (~110 lignes) — toute logique métier vit dans
    # ``app/services/``.
    "picarones/app/services/corpus_service.py": 625,      # actuel 541
    "picarones/app/services/path_security.py": 470,       # actuel 410
    "picarones/app/services/run_orchestrator.py": 500,    # actuel 432
    # Le rendu HTML vit en couche ``reports/`` (cible documentée
    # du rewrite — un rapport est un format de sortie, pas un
    # service métier).
    "picarones/reports/html/render.py": 700,           # actuel 615
}


def _line_count(path: Path) -> int:
    """Compte les lignes physiques (y compris vides)."""
    return len(path.read_text(encoding="utf-8").splitlines())


@pytest.mark.parametrize(
    ("rel_path", "budget"),
    sorted(FILE_BUDGETS.items()),
)
def test_file_size_within_budget(rel_path: str, budget: int) -> None:
    """Chaque fichier surveillé doit rester ≤ budget."""
    path = REPO_ROOT / rel_path
    assert path.exists(), (
        f"Fichier disparu : {rel_path}. "
        "Retire l'entrée de FILE_BUDGETS dans "
        "tests/architecture/test_file_budgets.py."
    )
    actual = _line_count(path)
    assert actual <= budget, (
        f"\n{rel_path} a {actual} lignes (budget {budget}).\n\n"
        "Soit refactor pour rentrer dans le budget, soit relève le budget "
        "consciemment dans tests/architecture/test_file_budgets.py "
        "avec une justification dans le message de commit."
    )


def test_no_orphaned_budget_entries() -> None:
    """Toute entrée de FILE_BUDGETS doit pointer vers un fichier existant."""
    missing = [p for p in FILE_BUDGETS if not (REPO_ROOT / p).exists()]
    assert not missing, (
        f"Entrées orphelines dans FILE_BUDGETS : {missing}. "
        "Le fichier a été déplacé/supprimé — retire l'entrée."
    )


def test_budget_table_covers_all_large_files() -> None:
    """Tout fichier ≥ 400 lignes doit avoir une entrée dans FILE_BUDGETS.

    Empêche un fichier nouveau ou subitement gros d'échapper à la
    surveillance. Si un fichier dépasse 400 lignes, ajoute-le à
    FILE_BUDGETS avec son budget (current + 15 %).
    """
    threshold = 400
    untracked: list[tuple[str, int]] = []
    for path in (REPO_ROOT / "picarones").rglob("*.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in FILE_BUDGETS:
            continue
        count = _line_count(path)
        if count >= threshold:
            untracked.append((rel, count))
    assert not untracked, (
        f"\nFichiers ≥ {threshold} lignes non surveillés :\n"
        + "\n".join(f"  {p} ({n} lignes)" for p, n in sorted(untracked))
        + "\n\nAjoute-les à FILE_BUDGETS dans "
        "tests/architecture/test_file_budgets.py avec budget = current + ~15 %."
    )
