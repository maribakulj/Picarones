"""Squelette des tests de feature parity entre ``run_benchmark_via_service``
et ``RunOrchestrator.execute(RunSpec)``.

Phase B0 du chantier de migration Option B.

Rôle
----
Ce module liste les **7 features** que ``run_benchmark_via_service``
expose aujourd'hui et que ``RunOrchestrator`` doit porter pendant la
Phase B2.  Chaque test est documenté précisément (ce qui doit être
vérifié) et marqué ``pytest.skip`` jusqu'à ce que la feature
correspondante soit portée.

Au fur et à mesure de la Phase B2, retirer le ``pytest.skip`` du test
correspondant et implémenter sa logique.  À la fin de B2, tous les
tests doivent être verts → on a atteint le **Checkpoint C1**.

Convention
----------
Chaque test compare :

1. ``run_benchmark_via_service(feature_X=value)`` — chemin legacy
2. ``RunOrchestrator().execute(spec_with_feature_X=value)`` — chemin
   rewrite

Et vérifie que le ``BenchmarkResult`` produit est numériquement
identique (modulo normalisation des champs volatils).

Mapping vers le plan Option B
-----------------------------
- B2.1 ``progress_callback``      → ``test_parity_progress_callback``
- B2.2 ``cancel_event``           → ``test_parity_cancel_event``
- B2.3 ``partial_dir``            → ``test_parity_partial_dir_resume``
- B2.4 ``entity_extractor``       → ``test_parity_entity_extractor_ner``
- B2.5 ``char_exclude`` +         → ``test_parity_normalization_propagation``
       ``normalization_profile``
- B2.6 ``profile`` (hooks)        → ``test_parity_profile_hooks``
- B2.7 ``output_json``            → ``test_parity_output_json_legacy_format``
"""

from __future__ import annotations

from pathlib import Path

import pytest


SKIP_REASON_PREFIX = "TODO Phase B2."


# ──────────────────────────────────────────────────────────────────────
# B2.1 — progress_callback
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}1 — port progress_callback")
def test_parity_progress_callback(tmp_path: Path) -> None:
    """``progress_callback`` est appelé avec ``(engine_name, doc_idx,
    doc_id)`` dans les deux chemins.

    Spec
    ----
    - Lancer un benchmark à 1 engine × 3 docs.
    - Le callback est invoqué exactement 3 fois (1 par doc).
    - Les arguments matchent : ``engine_name`` = nom de l'adapter,
      ``doc_idx`` = compteur global croissant (0, 1, 2), ``doc_id``
      = ID du document.
    - Le compteur est partagé entre threads via verrou
      (cf. ``_benchmark_execution.py:109-139``).

    Cible de port
    -------------
    Étendre ``RunOrchestrator._make_context_factory`` pour qu'il
    accepte un ``progress_callback`` et reproduise le pattern
    (verrou + compteur ``doc_idx``).
    """


# ──────────────────────────────────────────────────────────────────────
# B2.2 — cancel_event
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}2 — port cancel_event")
def test_parity_cancel_event(tmp_path: Path) -> None:
    """Un ``threading.Event.set()`` arrête le run en cours.

    Spec
    ----
    - Lancer un benchmark à 1 engine × 10 docs.
    - Après ~2 docs traités, appeler ``cancel_event.set()``.
    - Le run doit s'arrêter rapidement (< 1 s de marge).
    - Le ``BenchmarkResult`` retourné contient les 2 premiers docs
      (ou plus, selon timing) mais pas les 10.

    Cible de port
    -------------
    Wrapper ``CorpusRunner.run`` dans ``RunOrchestrator`` pour qu'il
    injecte le ``cancel_event`` dans ses kwargs (cf.
    ``_benchmark_execution.py:142-149``).
    """


# ──────────────────────────────────────────────────────────────────────
# B2.3 — partial_dir resume
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}3 — port partial_dir resume")
def test_parity_partial_dir_resume_fresh_start(tmp_path: Path) -> None:
    """Premier run avec ``partial_dir`` non existant → comportement
    identique à un run sans ``partial_dir``.

    Spec
    ----
    - ``partial_dir`` = répertoire vide.
    - Lancer le bench.
    - À la fin, le fichier ``{partial_dir}/picarones_{corpus}_{engine}
      .partial.jsonl`` est supprimé (succès complet).
    - Le ``BenchmarkResult`` est identique au run sans ``partial_dir``.
    """


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}3 — port partial_dir resume")
def test_parity_partial_dir_resume_after_crash(tmp_path: Path) -> None:
    """Reprise après crash partiel : 3 docs sur 5 déjà persistés →
    seuls les 2 restants sont soumis au runner.

    Spec
    ----
    - Pré-écrire un partial JSONL avec 3 ``DocumentResult`` valides.
    - Lancer le bench sur le corpus de 5 docs.
    - Le ``CorpusRunner.run`` est appelé sur **2 docs seulement**
      (vérifier via spy).
    - Le ``BenchmarkResult`` final agrège les 5 docs (3 réutilisés +
      2 nouveaux).
    """


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}3 — port partial_dir resume")
def test_parity_partial_dir_fingerprint_invalidates(tmp_path: Path) -> None:
    """Fingerprint divergent invalide le partial (re-calcul depuis 0).

    Spec
    ----
    - Pré-écrire un partial avec un ``code_version`` différent.
    - Lancer le bench.
    - Le partial est ignoré, les 5 docs sont recalculés.
    """


# ──────────────────────────────────────────────────────────────────────
# B2.4 — entity_extractor (NER attach)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}4 — port entity_extractor")
def test_parity_entity_extractor_ner(tmp_path: Path) -> None:
    """Quand un ``entity_extractor`` est fourni, les métriques NER
    sont attachées au ``BenchmarkResult``.

    Spec
    ----
    - Corpus avec ``EntitiesGT`` (au moins 1 doc avec niveau ENTITIES).
    - ``entity_extractor`` = mock qui retourne des entités fixes.
    - Le ``BenchmarkResult`` contient ``DocumentResult.ner_metrics`` :
      ``precision``, ``recall``, ``f1`` par type d'entité.
    - L'agrégation ``EngineReport.aggregated_ner`` est calculée.
    """


# ──────────────────────────────────────────────────────────────────────
# B2.5 — char_exclude + normalization_profile
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}5 — port normalization propagation")
def test_parity_char_exclude(tmp_path: Path) -> None:
    """``char_exclude`` filtre les caractères avant calcul CER/WER.

    Spec
    ----
    - GT = ``"Bonjour!"``, OCR = ``"Bonjour."``.
    - Sans ``char_exclude`` : CER = 1/8 = 0.125.
    - Avec ``char_exclude="!."`` : CER = 0.0 (les 2 caractères
      filtrés sont les seuls différents).
    """


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}5 — port normalization propagation")
def test_parity_normalization_profile(tmp_path: Path) -> None:
    """``normalization_profile="caseless"`` égalise les casses.

    Spec
    ----
    - GT = ``"Bonjour"``, OCR = ``"BONJOUR"``.
    - Sans profil : CER ≈ 1.0 (toutes les lettres diffèrent).
    - Avec ``caseless`` : CER = 0.0.
    """


# ──────────────────────────────────────────────────────────────────────
# B2.6 — profile (hooks document-level / corpus aggregators)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}6 — port profile hooks")
def test_parity_profile_validation(tmp_path: Path) -> None:
    """``profile="unknown"`` lève ``ValueError`` AVANT le run.

    Spec
    ----
    - Comportement identique aux 3 tests
      ``TestProfileValidation`` de
      ``tests/app/test_sprint_d2cdef_features.py``.
    """


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}6 — port profile hooks")
def test_parity_profile_standard_runs_hooks(tmp_path: Path) -> None:
    """``profile="standard"`` exécute les hooks document-level
    enregistrés via ``@register_document_metric``.

    Spec
    ----
    - Enregistrer un hook test ``@register_document_metric("standard")``
      qui renvoie ``{"hooked": True}``.
    - Lancer le bench.
    - ``DocumentResult.hook_values["hooked"] is True``.
    """


# ──────────────────────────────────────────────────────────────────────
# B2.7 — output_json (legacy BenchmarkResult JSON)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}7 — port output_json legacy")
def test_parity_output_json_legacy_format(tmp_path: Path) -> None:
    """Quand ``output_json`` est fourni, un fichier JSON au format
    ``BenchmarkResult.as_dict()`` est écrit en plus des 4 fichiers
    JSONL natifs du ``RunOrchestrator``.

    Spec
    ----
    - Lancer ``RunOrchestrator().execute(spec_with_output_json)``.
    - Vérifier que ``output_json`` existe et contient un JSON
      désérialisable via ``BenchmarkResult.from_json_object``.
    - Vérifier que les 4 fichiers JSONL natifs sont aussi écrits
      (cohabitation).
    """


# ──────────────────────────────────────────────────────────────────────
# Test global de feature parity — vérification croisée
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason=f"{SKIP_REASON_PREFIX}* — toutes features portées")
def test_parity_all_features_combined(tmp_path: Path) -> None:
    """Lance les deux chemins avec toutes les features actives et
    vérifie l'égalité numérique du ``BenchmarkResult``.

    Spec
    ----
    - Construire un ``RunSpec`` avec : ``profile="standard"``,
      ``partial_dir=tmp_path/"partial"``, ``output_json=tmp_path/
      "bm.json"``, ``char_exclude="!."``,
      ``normalization_profile="caseless"``.
    - Lancer ``run_benchmark_via_service`` avec les mêmes paramètres.
    - Lancer ``RunOrchestrator().execute(spec)``.
    - Normaliser les 2 ``BenchmarkResult`` (cf.
      ``test_migration_invariance.py:_normalize_for_snapshot``).
    - Vérifier ``a == b``.

    Ce test est le **gate finale du Checkpoint C1**.  Quand il passe,
    la Phase B2 est terminée et on peut commencer B3 (migration des
    call sites).
    """
