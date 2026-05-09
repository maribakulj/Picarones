"""Garde-fou contre la dérive doc-vs-code.

Scanne ``CLAUDE.md``, ``README.md``, ``docs/**/*.md`` à la recherche de
chemins de la forme ``picarones/.../X.py`` et vérifie qu'ils existent
dans le repo.

Snapshot v1.0.0 (2026-05-02) : **119 chemins cassés**, presque tous
dans ``CLAUDE.md`` et ``CHANGELOG.md`` qui décrivent systématiquement
des modules sous ``picarones/core/...`` alors qu'ils vivent dans
``picarones/measurements/...``. C'est une dette documentaire connue
qu'il faut résorber par paliers.

Test ratchet : le nombre de chemins cassés ne peut que diminuer. Pour
le faire baisser :

1. Soit corriger le chemin dans la doc.
2. Soit déplacer le module au chemin documenté (rare — la doc se
   trompe presque toujours).
3. Soit retirer la référence devenue obsolète.

Puis abaisser :data:`BROKEN_PATHS_BASELINE` du même montant.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Snapshot. Doit baisser, jamais monter.
#:
#: Historique :
#: - 119 (initial v1.0.0, dette pré-existante CLAUDE.md/CHANGELOG.md
#:   qui décrivent des modules sous ``picarones/core/...`` alors qu'ils
#:   vivent dans ``picarones/measurements/...``).
#: - 122 (sprint « découpage de statistics.py », 2026-05-02) : 3 audits
#:   historiques référencent ``picarones/measurements/statistics.py``
#:   qui est maintenant un sous-package. Baseline relevée.
#: - 72 (sprint « zéro dette actionnable », 2026-05-02) : 50 chemins
#:   massivement corrigés — 44 dans CLAUDE.md + 6 dans docs vivants.
#: - 73 (sprint « découpage de runner.py », 2026-05-03) :
#:   ``picarones/measurements/runner.py`` est désormais un sous-package
#:   ``runner/``. ``docs/user/writing-a-pipeline-module.md`` a été
#:   corrigé en place ; un audit historique
#:   (``docs/audits/institutional-readiness-2026-05.md``) référence
#:   l'ancien chemin et reste intouché par convention.
#: - 77 (sprint « Lot A — core.{modules,facts} → domain », 2026-05-07) :
#:   suppression des shims ``picarones/core/modules.py`` et
#:   ``picarones/core/facts.py``.  Deux références demeurent dans
#:   ``CHANGELOG.md`` (journal versionné) et
#:   ``docs/roadmap/evolution-2026.md`` (plan stratégique historique
#:   décrivant la création initiale du module).
#: - 80 (sprint « Lot B — core.metric_* → evaluation », 2026-05-07) :
#:   suppression des shims ``picarones/core/metric_registry.py``,
#:   ``picarones/core/metric_hooks.py`` et
#:   ``picarones/core/metrics.py``.  Trois nouvelles références
#:   héritées : deux dans ``CHANGELOG.md`` (intouchable) + une
#:   dans ``docs/migration/executor-equivalence.md`` (audit
#:   historique de la migration legacy → executor).  Le doc actif
#:   ``docs/reference/normalization-profiles.md`` a été corrigé
#:   en place vers ``picarones/evaluation/metric_hooks.py``.
#: - 83 (sprint « Lot C — core.{results,corpus,pipeline} → evaluation »,
#:   2026-05-07) : suppression des shims ``picarones/core/results.py``,
#:   ``picarones/core/corpus.py`` et ``picarones/core/pipeline.py``.
#:   Trois nouvelles références héritées : deux dans ``CHANGELOG.md``
#:   (intouchable) + une dans ``docs/roadmap/evolution-2026.md``
#:   (plan stratégique historique).  Le doc actif
#:   ``docs/reference/api-stable.md`` a été migré vers les chemins
#:   canoniques ``picarones.evaluation.{benchmark_result, corpus,
#:   pipeline}``.
#: - 88 (sprint « Lot D — measurements/X (34 shims) → evaluation/metrics »,
#:   2026-05-07) : suppression des 34 shims plats de ``measurements/``.
#:   Cinq nouveaux chemins cassés héritage : 4 dans ``docs/audits/*.md``
#:   (intouchable) + 1 dans ``docs/roadmap/evolution-2026.md``
#:   (plan stratégique historique).  Les docs actifs ``CLAUDE.md``,
#:   ``README.md`` et ``SPECS.md`` ont été corrigés en place vers
#:   ``picarones/formats/text/normalization.py``.
#: - 94 (sprint « Lot E — engines/ + modules/ → adapters/legacy_* »,
#:   2026-05-07) : suppression des 8 shims ``picarones/engines/`` et
#:   ``picarones/modules/``.  Six nouveaux chemins cassés héritage :
#:   5 dans ``CHANGELOG.md`` (intouchable) + 1 dans
#:   ``docs/audits/remediation-plan-2026-05.md`` (intouchable) — les
#:   audits citant ``aws_textract`` / ``kraken`` étaient déjà cassés
#:   avant la migration (ces moteurs n'ont jamais été implémentés).
#:   ``SPECS.md`` a été corrigé en place vers
#:   ``picarones/adapters/legacy_engines/base.py``.
#: - 132 (sprint « Lot F — report/ → reports/ », 2026-05-07) :
#:   suppression des 37 shims ``picarones/report/`` (29 *_render.py,
#:   2 helpers, 6 modules + glossary).  38 nouveaux chemins cassés
#:   héritage : 29 dans ``CHANGELOG.md`` + 8 dans ``docs/audits/*.md``
#:   et ``docs/migration/legacy-retirement-plan.md`` — tous
#:   intouchables.  Le doc actif ``docs/reference/views.md`` a été
#:   corrigé en place vers les chemins ``picarones/reports/html/{views,
#:   generator, renderers, templates}``.
#: - 134 (sprint « Lot G — core/{diff_utils, xml_utils} », 2026-05-07) :
#:   suppression des 2 derniers shims de ``picarones/core/``.  Le
#:   sous-paquet ``core/`` n'existe plus du tout.  Deux nouveaux
#:   chemins cassés héritage dans ``CHANGELOG.md`` (intouchable).
#: - 138 (sprints « Lots H + I », 2026-05-07) : suppression du
#:   sous-paquet ``measurements/statistics/`` (Lot H, 9 shims) et
#:   des 3 shims ``extras/importers/{htr_united, huggingface,
#:   _fallback_log}`` (Lot I).  Quatre nouveaux chemins cassés
#:   héritage répartis dans ``docs/audits/*.md`` (intouchables).
#:
#: Les chemins cassés restants sont **TOUS** dans :
#: - ``CHANGELOG.md`` : journal historique versionné, intouchable.
#: - ``docs/audits/*.md`` : audits historiques, intouchables.
#: - ``docs/roadmap/evolution-2026.md`` : plan stratégique historique.
#: - ``docs/migration/{executor-equivalence, legacy-retirement-plan}.md`` :
#:   audits/plans historiques (citent des chemins legacy à des fins
#:   de comparaison).
# Phase 7.B.2 : +3 broken paths — la doc référence
# ``picarones.evaluation.pipeline_benchmark`` /
# ``pipeline_comparison`` / ``pipeline`` qui ont migré vers
# ``picarones.pipeline.legacy_*``.  Les docs concernées
# (CHANGELOG.md, audits, sub-plans) gardent volontairement les
# anciens chemins pour la traçabilité historique.
# Sprint H.5 : -11 broken paths — fix des refs actives dans
# docs/how-to/cli-workflows.md, narrative-engine, normalization-profiles,
# doc-consistency, SESSION_HANDOVER.
# Sprint H.2.d : +1 — la suppression de ``adapters/legacy_engines/``
# et ``adapters/legacy_pipelines/`` casse 1 ref active de plus dans
# les docs migration restantes (la majorité des refs cassées
# pointaient déjà vers ces paquets dans CHANGELOG/audits historiques,
# d'où l'impact limité).
BROKEN_PATHS_BASELINE = 162

#: Patrons de fichiers de documentation à scanner.
DOC_GLOBS: tuple[str, ...] = (
    "CLAUDE.md",
    "README.md",
    "CHANGELOG.md",
    "SPECS.md",
    "docs/**/*.md",
)

#: Pattern minimal d'un chemin Python dans le repo.
PATH_PATTERN: re.Pattern[str] = re.compile(
    r"picarones/[a-z_][a-z_0-9]*(?:/[a-z_][a-z_0-9]*)*\.py"
)


def _doc_files() -> list[Path]:
    files: list[Path] = []
    for glob in DOC_GLOBS:
        files.extend(REPO_ROOT.glob(glob))
    return sorted({f for f in files if f.is_file()})


def _broken_paths() -> list[tuple[str, str]]:
    """Liste des (doc_relatif, chemin_cassé), dédoublonnée et triée."""
    broken: set[tuple[str, str]] = set()
    for doc in _doc_files():
        try:
            text = doc.read_text(encoding="utf-8")
        except OSError:
            continue
        rel_doc = doc.relative_to(REPO_ROOT).as_posix()
        for match in PATH_PATTERN.findall(text):
            if not (REPO_ROOT / match).exists():
                broken.add((rel_doc, match))
    return sorted(broken)


def test_broken_doc_paths_below_baseline() -> None:
    """Le nombre de chemins cassés ne peut que diminuer."""
    broken = _broken_paths()
    if len(broken) > BROKEN_PATHS_BASELINE:
        sample = "\n".join(f"  {doc} → {path}" for doc, path in broken[:30])
        more = f"\n  ... ({len(broken) - 30} de plus)" if len(broken) > 30 else ""
        raise AssertionError(
            f"\n{len(broken)} chemins de doc cassés (baseline "
            f"{BROKEN_PATHS_BASELINE}).\n"
            f"Régression : la doc référence un fichier qui n'existe pas.\n\n"
            f"Échantillon :\n{sample}{more}\n\n"
            "Soit corrige le chemin, soit le code, soit retire la référence."
        )


def test_baseline_must_be_tightened_when_progress_made() -> None:
    """Si on est sous le baseline, mettre à jour :data:`BROKEN_PATHS_BASELINE`.

    Verrouille chaque correction de doc pour empêcher une régression
    future de glisser sous le seuil obsolète.
    """
    broken = _broken_paths()
    assert len(broken) >= BROKEN_PATHS_BASELINE, (
        f"\nExcellent : {len(broken)} chemins cassés vs baseline "
        f"{BROKEN_PATHS_BASELINE}.\n\n"
        f"Mets à jour BROKEN_PATHS_BASELINE = {len(broken)} dans "
        "tests/architecture/test_doc_paths.py pour verrouiller le gain."
    )
