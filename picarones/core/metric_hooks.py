"""Registre typé des hooks de métriques document-level et corpus-level.

Chantier 2 du plan d'évolution post-Sprint 97.

Pourquoi ce module
------------------
Avant ce chantier, ``picarones.core.runner._compute_document_result``
contenait **11 imports tardifs codés en dur** vers
``picarones.core.confusion``, ``char_scores``, ``taxonomy``, ``structure``,
``image_quality``, ``line_metrics``, ``hallucination``,
``philological_runner``, ``searchability_runner``,
``numerical_sequences_runner``, ``readability_runner`` — chacun enrobé
dans un ``try/except Exception`` qui logue un warning. Symétriquement,
la phase d'agrégation contenait 11 fonctions ``_aggregate_*`` ou
``aggregate_*``. Ajouter une nouvelle métrique exigeait de patcher
``runner.py`` à deux endroits, ce qui rendait le fichier monolithique
(1322 lignes) et fragile.

Ce module centralise le mécanisme :

- **Profils** (``minimal`` / ``standard`` / ``philological`` /
  ``diagnostics`` / ``pipeline`` / ``full``) — l'utilisateur choisit
  quel sous-ensemble de métriques calculer selon son use case.
- **Hooks document-level** (:class:`DocumentMetricHook`) enregistrés via
  :func:`register_document_metric` — fonctions appelées pour chaque
  document, leur retour remplit un attribut nommé du ``DocumentResult``.
- **Agrégateurs corpus-level** (:class:`CorpusMetricAggregator`)
  enregistrés via :func:`register_corpus_aggregator` — fonctions
  appelées une fois par moteur pour synthétiser les
  ``DocumentResult`` en attributs ``aggregated_*`` du ``EngineReport``.

Rétrocompat stricte
-------------------
Le profil ``standard`` (défaut) active exactement les 11 hooks et 11
agrégateurs historiques. Comportement, ordre d'exécution, gestion
d'erreurs et octets de sortie : strictement identiques à avant le
chantier 2. La preuve est dans ``tests/test_metric_hooks.py``
(cas-tests qui comparent profil ``standard`` vs comportement legacy
sur fixtures).

Comment ajouter un hook
-----------------------
Pour ajouter une métrique document-level :

>>> from picarones.core.metric_hooks import (
...     register_document_metric, PROFILE_STANDARD, PROFILE_FULL,
... )
>>>
>>> @register_document_metric(
...     name="my_metric",
...     attribute="my_metric",  # nom du champ dans DocumentResult
...     profiles=(PROFILE_STANDARD, PROFILE_FULL),
...     requires_success=True,
... )
... def my_hook(*, ground_truth, hypothesis, image_path, corpus_lang,
...             ocr_result):
...     # Imports tardifs OK ici — le coût n'est payé que si le hook
...     # est dans le profil actif.
...     from my_pkg import compute_my_metric
...     return compute_my_metric(ground_truth, hypothesis)

Pour un nouveau profil, l'ajouter à :data:`KNOWN_PROFILES` (et
référencer dans la doc utilisateur ``docs/profiles/``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Profils
# ──────────────────────────────────────────────────────────────────────────

PROFILE_MINIMAL = "minimal"
"""Profil le plus léger : juste CER/WER de ``compute_metrics``. Aucun
hook document-level ni agrégateur ne s'exécute. Cible : tests rapides
ou bench massif où seul le CER global compte."""

PROFILE_STANDARD = "standard"
"""Profil par défaut. Active tous les hooks et agrégateurs historiques
de Picarones (Sprints 5 + 10 + 39+42 + 55-60 + 84-87). Comportement
strictement identique au runner pré-chantier-2."""

PROFILE_PHILOLOGICAL = "philological"
"""Profil orienté édition critique : standard + emphase philologique.
Aujourd'hui équivalent à standard ; réservé pour des hooks futurs
spécifiques aux corpus médiévaux et imprimés anciens."""

PROFILE_DIAGNOSTICS = "diagnostics"
"""Profil orienté diagnostic : standard + leviers d'amélioration,
prédiction de complexité, baseline historique. Réservé pour des
hooks futurs (chantiers 3-4)."""

PROFILE_ECONOMICS = "economics"
"""Profil orienté décision budget : minimal + métriques économiques
(throughput effectif, coût marginal). Réservé pour des hooks futurs."""

PROFILE_PIPELINE = "pipeline"
"""Profil pour les benchmarks de pipelines composées (axe B). Active
les hooks pertinents aux jonctions du DAG. Réservé pour des hooks
futurs spécifiques aux pipelines."""

PROFILE_FULL = "full"
"""Profil exhaustif : tous les hooks de tous les profils. Coût
maximal mais reproductibilité scientifique maximale."""

KNOWN_PROFILES: frozenset[str] = frozenset({
    PROFILE_MINIMAL,
    PROFILE_STANDARD,
    PROFILE_PHILOLOGICAL,
    PROFILE_DIAGNOSTICS,
    PROFILE_ECONOMICS,
    PROFILE_PIPELINE,
    PROFILE_FULL,
})


def validate_profile(profile: str) -> None:
    """Lève ``ValueError`` si ``profile`` n'est pas connu.

    Le runner appelle cette fonction au démarrage pour rejeter
    rapidement une faute de frappe utilisateur (``--profile philolagic``).
    """
    if profile not in KNOWN_PROFILES:
        raise ValueError(
            f"profil inconnu : {profile!r}. "
            f"Profils valides : {sorted(KNOWN_PROFILES)}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Modèles de hook
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DocumentMetricHook:
    """Hook calculé pour chaque document.

    Attributs
    ---------
    name:
        Identifiant lisible utilisé dans les logs et les warnings.
    attribute:
        Nom du champ du :class:`DocumentResult` à remplir (par
        exemple ``"confusion_matrix"`` ou ``"taxonomy"``). Doit
        correspondre exactement à un attribut existant — le runner
        passe le résultat via ``setattr``.
    profiles:
        Ensemble des profils dans lesquels ce hook s'active.
    func:
        Fonction calculant la métrique. Signature attendue :
        ``func(*, ground_truth, hypothesis, image_path, corpus_lang,
        ocr_result) -> Any``. Tous les arguments sont passés en
        keyword pour que les hooks puissent ignorer ceux qu'ils
        n'utilisent pas avec ``**_``.
    requires_success:
        Si ``True``, le hook n'est appelé que quand
        ``ocr_result.success`` (texte hyp non-vide). Évite de gaspiller
        du temps sur des documents en erreur OCR.
    requires_token_confidences:
        Si ``True``, le hook n'est appelé que quand
        ``ocr_result.token_confidences`` est non-vide. Réservé à la
        calibration (Sprint 42).
    """

    name: str
    attribute: str
    profiles: frozenset[str]
    func: Callable[..., Any]
    requires_success: bool = False
    requires_token_confidences: bool = False


@dataclass(frozen=True)
class CorpusMetricAggregator:
    """Agrégateur calculé une fois par moteur sur tous les documents.

    Attributs
    ---------
    name:
        Identifiant lisible.
    attribute:
        Nom du champ du :class:`EngineReport` à remplir (par
        exemple ``"aggregated_confusion"``).
    profiles:
        Profils dans lesquels l'agrégateur s'active.
    func:
        ``func(document_results: list[DocumentResult]) -> Any``.
    """

    name: str
    attribute: str
    profiles: frozenset[str]
    func: Callable[..., Any]


# ──────────────────────────────────────────────────────────────────────────
# Registres globaux
# ──────────────────────────────────────────────────────────────────────────


_DOCUMENT_HOOKS: list[DocumentMetricHook] = []
_CORPUS_AGGREGATORS: list[CorpusMetricAggregator] = []


def _check_profiles(profiles: Iterable[str]) -> frozenset[str]:
    frozen = frozenset(profiles)
    unknown = frozen - KNOWN_PROFILES
    if unknown:
        raise ValueError(
            f"profils inconnus : {sorted(unknown)}. "
            f"Profils valides : {sorted(KNOWN_PROFILES)}"
        )
    return frozen


def register_document_metric(
    *,
    name: str,
    attribute: str,
    profiles: Iterable[str],
    requires_success: bool = False,
    requires_token_confidences: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Décorateur d'enregistrement d'un hook document-level."""
    profiles_set = _check_profiles(profiles)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Idempotence : si le même func est ré-enregistré (ré-import
        # du module en test), on ignore silencieusement.  Si un autre
        # func tente le même ``name``, on lève.
        for existing in _DOCUMENT_HOOKS:
            if existing.name == name:
                if existing.func is func:
                    return func
                raise ValueError(
                    f"hook document '{name}' déjà enregistré par "
                    f"{existing.func.__module__}.{existing.func.__qualname__}"
                )
        _DOCUMENT_HOOKS.append(DocumentMetricHook(
            name=name,
            attribute=attribute,
            profiles=profiles_set,
            func=func,
            requires_success=requires_success,
            requires_token_confidences=requires_token_confidences,
        ))
        return func

    return decorator


def register_corpus_aggregator(
    *,
    name: str,
    attribute: str,
    profiles: Iterable[str],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Décorateur d'enregistrement d'un agrégateur corpus-level."""
    profiles_set = _check_profiles(profiles)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        for existing in _CORPUS_AGGREGATORS:
            if existing.name == name:
                if existing.func is func:
                    return func
                raise ValueError(
                    f"agrégateur corpus '{name}' déjà enregistré par "
                    f"{existing.func.__module__}.{existing.func.__qualname__}"
                )
        _CORPUS_AGGREGATORS.append(CorpusMetricAggregator(
            name=name,
            attribute=attribute,
            profiles=profiles_set,
            func=func,
        ))
        return func

    return decorator


# ──────────────────────────────────────────────────────────────────────────
# Sélection + exécution selon profil
# ──────────────────────────────────────────────────────────────────────────


def select_document_hooks(profile: str) -> list[DocumentMetricHook]:
    """Retourne les hooks document-level actifs pour ``profile``.

    L'ordre d'enregistrement est préservé pour garantir que les
    warnings et logs apparaissent dans le même ordre qu'avant le
    chantier 2 (cf. test de rétrocompat).
    """
    validate_profile(profile)
    return [h for h in _DOCUMENT_HOOKS if profile in h.profiles]


def select_corpus_aggregators(profile: str) -> list[CorpusMetricAggregator]:
    """Retourne les agrégateurs corpus-level actifs pour ``profile``."""
    validate_profile(profile)
    return [a for a in _CORPUS_AGGREGATORS if profile in a.profiles]


def run_document_hooks(
    profile: str,
    *,
    ground_truth: str,
    hypothesis: str,
    image_path: str,
    corpus_lang: str,
    ocr_result: Any,
) -> dict[str, Any]:
    """Exécute tous les hooks document-level actifs pour ``profile``.

    Retourne un dict ``{attribute_name: value}`` que le runner peut
    appliquer au ``DocumentResult`` via ``setattr`` ou ``**kwargs``.

    Pré-conditions :
    - les hooks à ``requires_success=True`` ne tournent que si
      ``ocr_result.success`` ;
    - les hooks à ``requires_token_confidences=True`` ne tournent
      que si ``ocr_result.token_confidences`` est non vide.

    Toute exception levée par un hook est loggée en warning et
    le hook est sauté (``attribute`` absent du dict retourné). Aucun
    hook ne fait jamais échouer le calcul des autres — discipline
    historique préservée.
    """
    out: dict[str, Any] = {}
    for hook in select_document_hooks(profile):
        if hook.requires_success and not getattr(ocr_result, "success", False):
            continue
        if hook.requires_token_confidences and not getattr(
            ocr_result, "token_confidences", None,
        ):
            continue
        try:
            value = hook.func(
                ground_truth=ground_truth,
                hypothesis=hypothesis,
                image_path=image_path,
                corpus_lang=corpus_lang,
                ocr_result=ocr_result,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] fonctionnalité dégradée : %s", hook.name, exc,
            )
            continue
        if value is not None:
            out[hook.attribute] = value
    return out


def run_corpus_aggregators(
    profile: str,
    document_results: list,
) -> dict[str, Any]:
    """Exécute tous les agrégateurs corpus-level pour ``profile``.

    Retourne un dict ``{attribute_name: value}`` à appliquer au
    ``EngineReport``. Comme pour les hooks doc-level, une exception
    dans un agrégateur est loggée et l'agrégateur sauté.
    """
    out: dict[str, Any] = {}
    for agg in select_corpus_aggregators(profile):
        try:
            value = agg.func(document_results)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[aggregate_%s] fonctionnalité dégradée : %s", agg.name, exc,
            )
            continue
        if value is not None:
            out[agg.attribute] = value
    return out


# ──────────────────────────────────────────────────────────────────────────
# Helpers test-only
# ──────────────────────────────────────────────────────────────────────────


def _reset_for_tests() -> None:
    """Vide les registres globaux. **Réservé aux tests** — désactive
    toutes les métriques en production."""
    _DOCUMENT_HOOKS.clear()
    _CORPUS_AGGREGATORS.clear()


def _all_document_hook_names() -> list[str]:
    return [h.name for h in _DOCUMENT_HOOKS]


def _all_corpus_aggregator_names() -> list[str]:
    return [a.name for a in _CORPUS_AGGREGATORS]


__all__ = [
    "PROFILE_MINIMAL",
    "PROFILE_STANDARD",
    "PROFILE_PHILOLOGICAL",
    "PROFILE_DIAGNOSTICS",
    "PROFILE_ECONOMICS",
    "PROFILE_PIPELINE",
    "PROFILE_FULL",
    "KNOWN_PROFILES",
    "validate_profile",
    "DocumentMetricHook",
    "CorpusMetricAggregator",
    "register_document_metric",
    "register_corpus_aggregator",
    "select_document_hooks",
    "select_corpus_aggregators",
    "run_document_hooks",
    "run_corpus_aggregators",
]
