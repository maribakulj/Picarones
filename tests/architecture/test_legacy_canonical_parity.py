"""Test architectural — parité legacy ↔ canonique.

Pourquoi ce test
----------------
Le retrait du legacy se fait par phases (cf. :doc:`docs/migration/legacy-retirement-plan.md`).
À chaque phase, des symboles publics legacy migrent vers leur
équivalent canonique.  Sans garde-fou, deux risques :

1. **Suppression silencieuse d'une feature** : un symbole legacy
   disparaît sans équivalent canonique → la feature est perdue.
2. **Drift de l'API** : un nouveau symbole legacy est ajouté
   sans équivalent canonique → la dette de migration grossit.

Ce test maintient un **journal de bord vivant** :
:data:`LEGACY_PARITY` est une table 3-états qui pour chaque
symbole legacy connu déclare :

- ``canonical: <module.symbol>`` — symbole équivalent dans
  l'arbre canonique.
- ``dropped: <raison>`` — feature volontairement abandonnée.
- ``unmigrated: <cible prévue>`` — migration prévue, à venir.

Limites du test
---------------
**Ce test ne vérifie que la présence de symbole, pas le
comportement.**  Deux symboles peuvent porter le même nom et
avoir des sémantiques différentes (cf. ``ArtifactType`` 6 vs 10
valeurs).  Les différences comportementales sont signalées par
le champ optionnel ``behavior_diff`` qui sert de mémoire à
l'équipe.

Pour la vérification comportementale réelle :

- Les tests unitaires métier couvrent les usages individuels.
- Le test d'intégration ``tests/integration/test_sprint_a14_s12_executor_equivalence.py``
  compare des comportements bout-en-bout.
- La régression bit-for-bit sur les rapports HTML (cible Phase
  11 du retrait du legacy) couvrira la sortie utilisateur finale.

Maintenance
-----------
- À chaque migration d'un symbole, ajouter ou mettre à jour son
  entrée dans :data:`LEGACY_PARITY`.
- Si un nouveau symbole legacy est introduit (rare mais possible
  pour patcher un bug bloquant), l'inscrire avec
  ``"unmigrated": "<cible>"``.
- Si un symbole est supprimé du legacy, retirer son entrée.
- ``BOOTSTRAP_BASELINE`` autorise temporairement N symboles non
  trackés ; à diminuer à chaque session de migration.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import warnings
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Paquets legacy (cf. ``test_no_legacy_imports_in_rewrite``).
LEGACY_PACKAGES: tuple[str, ...] = (
    "measurements",
    "llm",
    "pipelines",
)

#: Combien de symboles legacy peuvent être absents de
#: :data:`LEGACY_PARITY` sans faire échouer le test.  À diminuer
#: à chaque session de migration : on cible 0 quand le retrait
#: est complet.
BOOTSTRAP_BASELINE = 78


# ──────────────────────────────────────────────────────────────────
# Table de parité legacy ↔ canonique
# ──────────────────────────────────────────────────────────────────

#: Entrée de :data:`LEGACY_PARITY`.  Exactement une des trois
#: clés (``canonical``, ``dropped``, ``unmigrated``) est
#: renseignée par entrée.
ParityEntry = dict[str, Any]


#: Mapping ``"<legacy_module>.<symbol>" → entry``.  Les entrées
#: sont ajoutées au fil des phases de migration ; la table est le
#: journal de bord vivant du retrait du legacy.
#:
#: État initial : seed des migrations déjà effectuées (Phases
#: 1, 4-bis, 4-ter, 4-quater, 7.A).  À étendre à chaque sprint.
LEGACY_PARITY: dict[str, ParityEntry] = {
    # ──────────────────────────────────────────────────────────
    # Phase 1 — diff_utils, xml_utils, facts
    # ──────────────────────────────────────────────────────────
    # Lot G (mai 2026) : ``picarones.core`` entièrement supprimé.
    # Les helpers ``compute_word_diff``, ``compute_char_diff``,
    # ``diff_stats`` vivent désormais uniquement dans
    # ``picarones.evaluation._diff_utils`` ; ``safe_parse_xml``
    # uniquement dans ``picarones.formats._xml_utils``.
    # ``core.facts`` et ``core.modules`` avaient déjà été
    # supprimés en Lot A.  Les entrées correspondantes ont été
    # retirées de cette table pour garder l'alignement avec
    # l'arbre legacy réellement présent sur disque.
    # ──────────────────────────────────────────────────────────
    # Phase 4-ter — metric_registry, metric_hooks, metrics
    # ──────────────────────────────────────────────────────────
    # ``core.metric_registry``, ``core.metric_hooks`` et
    # ``core.metrics`` ont été supprimés (Lot B de la migration
    # core → evaluation).  Les symboles publics
    # (MetricSpec, register_metric, compute_at_junction, …,
    # PROFILE_*, KNOWN_PROFILES, MetricsResult, aggregate_metrics)
    # sont exposés depuis
    # ``picarones.evaluation.{metric_registry, metric_hooks,
    # metric_result}``.  Comme pour le Lot A, les entrées sont
    # retirées en même temps que les shims pour garder la table
    # alignée avec l'arbre legacy réellement présent sur disque.
    # ──────────────────────────────────────────────────────────
    # Phase 4-ter résiduel + 4-quater + 5.C.batch7 — results,
    # corpus, pipeline (Lot C)
    # ──────────────────────────────────────────────────────────
    # ``core.results``, ``core.corpus`` et ``core.pipeline`` ont
    # été supprimés (Lot C de la migration core → evaluation).
    # Les symboles publics (BenchmarkResult, EngineReport,
    # DocumentResult, Document, Corpus, GTLevel, TextGT, AltoGT,
    # PageGT, EntitiesGT, ReadingOrderGT, load_corpus_from_directory,
    # PipelineRunner, PipelineSpec, PipelineStep, PipelineResult,
    # StepResult) sont exposés depuis
    # ``picarones.evaluation.{benchmark_result, corpus, pipeline}``.
    # Comme pour les Lots A et B, les entrées sont retirées en
    # même temps que les shims pour garder la table alignée avec
    # l'arbre legacy réellement présent sur disque.
    # Note 7.B-7.D : le ``PipelineRunner`` canonique reste
    # transitoire et délègue progressivement à
    # ``PipelineExecutor`` (cf. pipeline-convergence-plan.md).
    # ──────────────────────────────────────────────────────────
    # Phase 7.A + Lot E — engines, modules
    # ──────────────────────────────────────────────────────────
    # ``picarones/engines/`` et ``picarones/modules/`` ont été
    # supprimés (Lot E de la migration legacy → adapters/legacy_*).
    # Les classes (BaseOCREngine, EngineResult, TesseractEngine,
    # PeroOCREngine, MistralOCREngine, GoogleVisionEngine,
    # AzureDocIntelEngine, engine_from_name, TextToAltoMonoRegion)
    # sont exposées depuis
    # ``picarones.adapters.legacy_engines.{...}`` et
    # ``picarones.adapters.legacy_modules.alto_text_to_mono_region``.
    # Comme pour les Lots précédents, les entrées ont été retirées
    # pour garder la table alignée avec l'arbre legacy réellement
    # présent sur disque.
    # Note 7.D : ces adapters legacy seront eux-mêmes refondus en
    # ``BaseOCRAdapter (StepExecutor)`` lors de la convergence
    # pipeline (cf. pipeline-convergence-plan.md).
}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _resolve(dotted: str) -> Any | None:
    """Résout ``"package.module.symbol"`` en l'objet effectif.

    Retourne ``None`` si la résolution échoue (module introuvable
    ou symbole absent).  Émet un warning silencieux : on
    ignore les ``DeprecationWarning`` des shims.
    """
    parts = dotted.rsplit(".", 1)
    if len(parts) != 2:
        return None
    module_path, symbol = parts
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            module = importlib.import_module(module_path)
    except ImportError:
        return None
    return getattr(module, symbol, None)


def _signatures_compatible(legacy_obj: Any, canonical_obj: Any) -> bool:
    """Vérifie que deux callables ont des signatures compatibles.

    Compatible = même nombre de paramètres positionnels.  Pour
    les classes, on inspecte ``__init__``.  Si l'un des deux
    n'est pas inspectable (e.g. C-builtin), on retourne ``True``
    (on ne peut pas comparer).
    """
    try:
        sig_legacy = inspect.signature(legacy_obj)
        sig_canonical = inspect.signature(canonical_obj)
    except (TypeError, ValueError):
        # Pas un callable inspectable : on tolère
        return True
    pos_legacy = [
        p for p in sig_legacy.parameters.values()
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    pos_canonical = [
        p for p in sig_canonical.parameters.values()
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    return len(pos_legacy) == len(pos_canonical)


def _scan_legacy_public_symbols() -> set[str]:
    """Liste tous les symboles publics top-level dans les
    paquets legacy.

    Scanne via AST (statique, plus stable que ``import + dir``).
    Retourne les ``"<full_module_path>.<symbol_name>"``.
    """
    out: set[str] = set()
    for pkg in LEGACY_PACKAGES:
        root = REPO_ROOT / "picarones" / pkg
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if path.name == "__init__.py":
                continue  # __init__ : on ne tracke que les modules nommés
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            module_name = ".".join(
                ["picarones", *path.relative_to(REPO_ROOT / "picarones").with_suffix("").parts]
            )
            for node in tree.body:
                names: list[str] = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            names.append(target.id)
                for sym in names:
                    if sym.startswith("_"):
                        continue
                    out.add(f"{module_name}.{sym}")
    return out


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "legacy_path,entry",
    sorted(LEGACY_PARITY.items()),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_each_entry_is_resolvable(legacy_path: str, entry: ParityEntry) -> None:
    """Chaque entrée doit avoir exactement un état renseigné.

    Validation par entrée :

    - ``canonical`` : les deux symboles existent + signatures
      compatibles (warning si non).
    - ``dropped``   : justification non vide.
    - ``unmigrated``: cible non vide (le canonique peut ne pas
      encore exister, c'est tout l'intérêt de cet état).
    """
    states = {"canonical", "dropped", "unmigrated"}
    present = states & entry.keys()
    assert len(present) == 1, (
        f"{legacy_path} doit avoir exactement un de {states}, "
        f"trouvé : {sorted(present)}"
    )
    state = next(iter(present))
    if state == "canonical":
        canonical_path = entry["canonical"]
        legacy_obj = _resolve(legacy_path)
        canonical_obj = _resolve(canonical_path)
        assert legacy_obj is not None, (
            f"Symbole legacy ``{legacy_path}`` introuvable.  "
            "Si le symbole a été supprimé, retire son entrée de "
            "LEGACY_PARITY."
        )
        assert canonical_obj is not None, (
            f"Symbole canonique ``{canonical_path}`` introuvable.  "
            "Vérifie le chemin canonique ou que le module existe."
        )
        assert _signatures_compatible(legacy_obj, canonical_obj), (
            f"Signatures incompatibles entre ``{legacy_path}`` et "
            f"``{canonical_path}``.  Mets à jour ``behavior_diff`` "
            "pour documenter le changement, ou aligne les signatures."
        )
    elif state == "dropped":
        reason = entry["dropped"]
        assert isinstance(reason, str) and reason.strip(), (
            f"{legacy_path} marqué dropped sans justification.  "
            "Ajoute la raison du drop pour traçabilité."
        )
    elif state == "unmigrated":
        target = entry["unmigrated"]
        assert isinstance(target, str) and target.strip(), (
            f"{legacy_path} marqué unmigrated sans cible.  "
            "Ajoute le chemin canonique prévu."
        )


def test_no_untracked_legacy_symbol_above_baseline() -> None:
    """Tout symbole public legacy doit être tracé dans :data:`LEGACY_PARITY`.

    Mode bootstrap : :data:`BOOTSTRAP_BASELINE` autorise N
    symboles non trackés.  À diminuer à chaque sprint de
    migration ; cible 0 quand la migration est complète.
    """
    public_symbols = _scan_legacy_public_symbols()
    untracked = public_symbols - LEGACY_PARITY.keys()
    if len(untracked) > BOOTSTRAP_BASELINE:
        sample = "\n".join(f"  {s}" for s in sorted(untracked)[:30])
        more = (
            f"\n  ... ({len(untracked) - 30} de plus)"
            if len(untracked) > 30
            else ""
        )
        raise AssertionError(
            f"\n{len(untracked)} symbole(s) legacy non tracé(s) "
            f"dans LEGACY_PARITY (baseline {BOOTSTRAP_BASELINE}).\n\n"
            f"{sample}{more}\n\n"
            "Ajoute chaque symbole à LEGACY_PARITY avec son état :\n"
            "  - ``canonical: <module.symbol>`` si déjà migré\n"
            "  - ``dropped: <raison>`` si abandonné\n"
            "  - ``unmigrated: <cible prévue>`` si encore à venir\n\n"
            "Ou abaisse BOOTSTRAP_BASELINE si on est sous le seuil "
            "(faute de quoi le test ne progresse plus)."
        )


def test_baseline_should_tighten_when_progress() -> None:
    """Si on est sous le baseline, abaisser BOOTSTRAP_BASELINE.

    Ce test est l'inverse du précédent : il rappelle que le
    baseline doit suivre la progression.  Pareil pattern que
    ``test_doc_paths::test_baseline_must_be_tightened_when_progress_made``.
    """
    public_symbols = _scan_legacy_public_symbols()
    untracked = public_symbols - LEGACY_PARITY.keys()
    assert len(untracked) >= BOOTSTRAP_BASELINE, (
        f"\nExcellent : {len(untracked)} symboles non tracés vs "
        f"baseline {BOOTSTRAP_BASELINE}.\n"
        "Mets à jour BOOTSTRAP_BASELINE dans "
        "tests/architecture/test_legacy_canonical_parity.py."
    )


def test_canonical_paths_dont_themselves_use_legacy() -> None:
    """Les cibles canoniques ne doivent pas pointer vers du legacy.

    Cas pathologique : on déclare ``canonical:
    picarones.evaluation.X`` mais ``picarones.evaluation.X``
    importe en interne depuis ``picarones.core.Y``.  Ce serait un
    bug d'aiguillage.

    Ce test ne couvre pas le cas (couverture par
    ``test_no_legacy_imports_in_rewrite``) ; ici on se contente
    de vérifier que les cibles canoniques sont dans des paquets
    rewrite reconnus.
    """
    REWRITE_PREFIXES = (
        "picarones.domain.",
        "picarones.formats.",
        "picarones.evaluation.",
        "picarones.pipeline.",
        "picarones.adapters.",
        "picarones.app.",
        "picarones.reports_v2.",
        "picarones.interfaces.",
    )
    misrouted: list[tuple[str, str]] = []
    for legacy, entry in LEGACY_PARITY.items():
        canonical = entry.get("canonical")
        if not canonical:
            continue
        if not canonical.startswith(REWRITE_PREFIXES):
            misrouted.append((legacy, canonical))
    assert not misrouted, (
        "Cibles canoniques en dehors des paquets rewrite :\n"
        + "\n".join(f"  {legacy} → {canonical}" for legacy, canonical in misrouted)
    )
