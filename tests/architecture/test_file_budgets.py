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
    # ``benchmark_runner`` : façade publique de ``BenchmarkService``,
    # entry point pour CLI/web.  God-module historique en cours de
    # split (Phase 6 audit code-quality, 1700 → 1329 LOC).
    # Extractions effectuées (rounds 1-6) :
    # - ``_benchmark_ner`` (NER aggregation, ~100 LOC)
    # - ``_benchmark_persistence`` (JSON dump, ~15 LOC)
    # - ``_benchmark_adapter_resolver`` (engine→spec + resolver, ~250 LOC)
    # - ``_benchmark_conversions`` (document/corpus + helpers GT, ~230 LOC)
    # - ``_benchmark_execution`` (BenchmarkService orchestration, ~140 LOC)
    # - ``_benchmark_helpers`` (extract_*, build_*, fingerprint, ~260 LOC)
    # - ``_benchmark_converter`` (RunResult→BenchmarkResult, ~200 LOC)
    # - ``_benchmark_orchestration`` (unified + with_partial, ~250 LOC)
    # Bilan cumulé : **1700 → 299 LOC (-82 %)** — façade pure
    # ``run_benchmark_via_service`` + bloc de re-exports.
    "picarones/app/services/benchmark_runner.py": 320,  # actuel 299
    # --- God-modules : budget actuel + 15 % de marge.
    # Le rétrécissement sera l'objet d'un sprint de refactor dédié.
    # Phase 4.6 audit code-quality (2026-05) — commentaires retirés :
    # ils décrivaient des modules supprimés en v2.0 (``measurements/``,
    # ``core/``, ``report/``, ``pipelines/legacy_*``) qui ne sont plus
    # référencés ailleurs.  L'historique reste accessible via git log
    # + CHANGELOG.
    "picarones/reports/html/generator.py": 550,        # actuel 471
    "picarones/evaluation/benchmark_result.py": 880,      # actuel ~826
    "picarones/reports/html/renderers/philological.py": 700,  # actuel 601
    "picarones/evaluation/metrics/modern_archives.py": 700,  # actuel 599
    "picarones/evaluation/metrics/builtin_hooks.py": 700,  # actuel 590
    "picarones/evaluation/metrics/history.py": 720,        # actuel 615
    # Phase 3.1 audit code-quality (2026-05) : retrait des 5 helpers
    # pure-Python ``_apply_*`` + stub ``_degrade_pure_python`` (~300 LOC
    # mortes) → budget dégonflé de 850 à 650.
    "picarones/evaluation/metrics/robustness.py": 650,     # actuel 578
    "picarones/interfaces/web/benchmark_utils.py": 600,    # actuel 520
    # Importers IIIF / Gallica.
    "picarones/adapters/corpus/iiif.py": 675,             # actuel 567
    "picarones/adapters/corpus/gallica.py": 675,          # actuel 563
    "picarones/evaluation/metrics/levers.py": 675,        # actuel 561
    "picarones/evaluation/metrics/inter_engine.py": 575,  # actuel 484
    "picarones/adapters/corpus/escriptorium.py": 650,     # actuel 553
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
    "picarones/adapters/llm/base.py": 520,                # actuel ~440
    "picarones/evaluation/corpus.py": 600,                # actuel 533
    "picarones/evaluation/synthetic.py": 600,             # actuel 510
    "picarones/evaluation/metrics/roman_numerals.py": 575,  # actuel 484
    "picarones/adapters/corpus/htr_united.py": 575,       # actuel 473
    "picarones/adapters/corpus/huggingface.py": 550,      # actuel 464
    # Phase 3.3 audit code-quality (2026-05) — option
    # ``--normalization-profile`` + résolution builtin/YAML (~30 LOC).
    "picarones/interfaces/cli/_workflows.py": 660,  # actuel ~621
    # ``__init__.py`` du CLI : commandes ``info``, ``engines``,
    # ``metrics``, ``report``, ``demo`` regroupées.
    "picarones/interfaces/cli/__init__.py": 500,    # actuel 396
    "picarones/evaluation/metric_hooks.py": 500,          # actuel 427
    "picarones/evaluation/metrics/numerical_sequences.py": 500,  # actuel 428
    "picarones/formats/text/normalization.py": 500,       # actuel 420
    "picarones/reports/html/comparison.py": 500,       # actuel 414
    # Renderers HTML — helpers couleur + format mutualisés.
    "picarones/reports/_helpers/render_helpers.py": 480,  # actuel 428
    # --- Services applicatifs (couche 6).  Budgets ``current + 15 %``.
    "picarones/app/services/corpus_service.py": 625,      # actuel 541
    "picarones/app/services/path_security.py": 470,       # actuel 410
    "picarones/app/services/run_orchestrator.py": 1500,   # actuel 1267 — Phase B4 migration Option B (+ execute_preset + workspace_uri propagation)
    "picarones/app/schemas/run_spec.py": 620,             # actuel 530 — Phase B1 migration Option B (+90 LOC : 7 nouveaux champs + 2 validators)
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
