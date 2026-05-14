"""Test d'invariance run-to-run pour la migration Option B.

Phase B0 du chantier de migration ``run_benchmark_via_service`` →
``RunOrchestrator.execute(RunSpec)``.

Rôle
----
Ce test exécute un benchmark **déterministe** (corpus mini de 2 docs +
``PrecomputedTextAdapter``) via la façade actuelle
``run_benchmark_via_service`` et compare son ``BenchmarkResult``
normalisé à un snapshot JSON enregistré dans
``tests/integration/snapshots/migration_invariance.json``.

Pourquoi
--------
Pendant la migration vers ``RunOrchestrator``, on porte 7 features
(``progress_callback``, ``cancel_event``, ``partial_dir``,
``entity_extractor``, ``char_exclude``, ``normalization_profile``,
``profile``, ``output_json``).  Chaque port doit préserver
**exactement** le comportement numérique du chemin existant.  Ce test
sert de filet de sécurité : si une refactorisation interne modifie le
résultat (CER, agrégation, ordre des engines, structure du JSON), le
snapshot diverge et la CI échoue.

Le test n'utilise **aucune** dépendance externe (pas de Tesseract, pas
de réseau).  Le ``PrecomputedTextAdapter`` lit un fichier texte écrit
sur disque — sortie 100% déterministe.

Mise à jour du snapshot
-----------------------
Si une modification **volontaire** change le résultat (ex. nouveau
champ dans ``BenchmarkResult``), régénérer le snapshot :

    PICARONES_UPDATE_SNAPSHOT=1 python -m pytest \
        tests/integration/test_migration_invariance.py

Et inspecter le diff git du snapshot avant commit.

Normalisation
-------------
Les champs volatils sont neutralisés avant comparaison :

- ``picarones_version`` → ``"PINNED"``
- ``run_date`` → ``"PINNED"``
- ``corpus.source`` → ``"FIXTURES/corpus"``
- ``image_path`` → ``"FIXTURES/docN.png"``
- ``duration_seconds`` → ``0.0``
- Tout autre champ contenant le ``tmp_path`` → remplacé par
  ``"FIXTURES/..."``

Cela garantit que le snapshot reste stable cross-OS et cross-run.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from picarones.adapters.ocr.precomputed import PrecomputedTextAdapter
from picarones.app.services.benchmark_runner import run_benchmark_via_service
from picarones.evaluation.corpus import Corpus, Document


SNAPSHOT_PATH = (
    Path(__file__).parent / "snapshots" / "migration_invariance.json"
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures déterministes
# ──────────────────────────────────────────────────────────────────────


def _make_invariance_corpus(tmp_path: Path) -> Corpus:
    """Corpus mini de 2 documents avec GT + texte précalculé.

    Le texte précalculé est légèrement différent de la GT pour produire
    des métriques CER/WER non triviales (et donc plus discriminantes
    dans le snapshot).
    """
    documents: list[Document] = []

    # Doc 1 : GT = "Bonjour le monde", OCR = "Bonjour le monde" → CER 0.0
    doc1_img = tmp_path / "doc1.png"
    doc1_img.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG header minimal
    doc1_ocr = tmp_path / "doc1.invariance.txt"
    doc1_ocr.write_text("Bonjour le monde", encoding="utf-8")
    documents.append(Document(
        image_path=doc1_img,
        ground_truth="Bonjour le monde",
        doc_id="doc1",
    ))

    # Doc 2 : GT = "Hello world", OCR = "Helio world" → CER non nul
    doc2_img = tmp_path / "doc2.png"
    doc2_img.write_bytes(b"\x89PNG\r\n\x1a\n")
    doc2_ocr = tmp_path / "doc2.invariance.txt"
    doc2_ocr.write_text("Helio world", encoding="utf-8")
    documents.append(Document(
        image_path=doc2_img,
        ground_truth="Hello world",
        doc_id="doc2",
    ))

    return Corpus(name="invariance_corpus", documents=documents)


def _make_invariance_engine() -> PrecomputedTextAdapter:
    """``PrecomputedTextAdapter`` qui lit ``<stem>.invariance.txt``."""
    return PrecomputedTextAdapter(source_label="invariance")


# ──────────────────────────────────────────────────────────────────────
# Normalisation du snapshot
# ──────────────────────────────────────────────────────────────────────


def _normalize_for_snapshot(data: Any, tmp_path: Path) -> Any:
    """Normalise récursivement les champs volatils du ``BenchmarkResult``.

    Remplace ``tmp_path`` par ``"FIXTURES"`` dans toutes les valeurs
    string.  Neutralise les champs explicitement volatils
    (``duration_seconds``, ``run_date``, ``picarones_version``,
    ``engine_version``, ``code_version``).
    """
    tmp_str = str(tmp_path)
    # Pattern pour matcher tmp_path/quelque-chose (pour les chemins
    # absolus qui n'apparaissent pas en clé mais en valeur string).
    tmp_re = re.compile(re.escape(tmp_str))

    def _normalize(value: Any, *, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {k: _normalize(v, key=k) for k, v in value.items()}
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        if isinstance(value, str):
            return tmp_re.sub("FIXTURES", value)
        if isinstance(value, float):
            # Neutralise les durées (volatiles d'un run à l'autre).
            if key == "duration_seconds":
                return 0.0
            # Garde les autres floats avec une précision raisonnable
            # pour absorber le bruit de calcul minimum.
            return round(value, 6)
        return value

    normalized = _normalize(data)

    # Champs volatils au niveau racine — neutralisés en post-traitement
    # parce que leur valeur ne contient pas ``tmp_path``.
    if isinstance(normalized, dict):
        for volatile_key in ("picarones_version", "run_date"):
            if volatile_key in normalized:
                normalized[volatile_key] = "PINNED"

        # engine_version peut apparaître dans chaque engine_report.
        for report in normalized.get("engine_reports", []):
            if "engine_version" in report:
                report["engine_version"] = "PINNED"
            # Les pipeline_info portent parfois des chemins ou metadata.
            pipeline_info = report.get("pipeline_info")
            if isinstance(pipeline_info, dict):
                if "code_version" in pipeline_info:
                    pipeline_info["code_version"] = "PINNED"

    return normalized


# ──────────────────────────────────────────────────────────────────────
# Comparaison snapshot
# ──────────────────────────────────────────────────────────────────────


def _load_snapshot() -> dict | None:
    if not SNAPSHOT_PATH.exists():
        return None
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _write_snapshot(data: dict) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _should_update_snapshot() -> bool:
    return os.environ.get("PICARONES_UPDATE_SNAPSHOT") == "1"


# ──────────────────────────────────────────────────────────────────────
# Test principal
# ──────────────────────────────────────────────────────────────────────


def test_run_benchmark_via_service_invariance(tmp_path: Path) -> None:
    """Snapshot d'invariance du comportement actuel.

    Ce test est le filet de sécurité de la migration Option B.  Il doit
    rester vert à chaque étape du chantier (B1, B2, B3, B4, ...) tant
    que ``run_benchmark_via_service`` est la façade publique.

    Quand la migration sera terminée et ``run_benchmark_via_service``
    supprimée (Phase B8), ce test sera retiré ou migré vers
    ``RunOrchestrator.execute()``.
    """
    corpus = _make_invariance_corpus(tmp_path)
    engine = _make_invariance_engine()

    benchmark_result = run_benchmark_via_service(
        corpus=corpus,
        engines=[engine],
        code_version="invariance-test-1.0.0",
    )

    actual_normalized = _normalize_for_snapshot(
        benchmark_result.as_dict(), tmp_path,
    )

    snapshot = _load_snapshot()
    if snapshot is None or _should_update_snapshot():
        _write_snapshot(actual_normalized)
        if snapshot is None:
            pytest.skip(
                f"Snapshot créé pour la première fois à "
                f"{SNAPSHOT_PATH.relative_to(Path.cwd())}. "
                f"Vérifier son contenu puis ré-exécuter le test."
            )
        else:
            # Mode update explicite : on a écrit, le test passe sans
            # vérification additionnelle.  L'opérateur est responsable
            # d'inspecter le diff git.
            return

    assert actual_normalized == snapshot, (
        "BenchmarkResult diverge du snapshot d'invariance.\n"
        f"Snapshot : {SNAPSHOT_PATH}\n"
        "Si la divergence est intentionnelle, régénérer avec :\n"
        "    PICARONES_UPDATE_SNAPSHOT=1 python -m pytest "
        f"{Path(__file__).relative_to(Path.cwd())}\n"
        "et inspecter le diff git du snapshot avant commit."
    )


# ──────────────────────────────────────────────────────────────────────
# Test annexe — vérifie que la normalisation elle-même est stable
# ──────────────────────────────────────────────────────────────────────


def test_normalization_is_idempotent(tmp_path: Path) -> None:
    """La normalisation d'un dict déjà normalisé ne le change pas.

    Garantit qu'on peut ré-appliquer la normalisation sans dériver.
    Test pédagogique de la mécanique du snapshot.
    """
    sample = {
        "picarones_version": "2.0.0",
        "run_date": "2026-05-14T12:00:00Z",
        "corpus": {"source": str(tmp_path / "corpus.zip")},
        "engine_reports": [
            {
                "engine_version": "1.2.3",
                "document_results": [
                    {
                        "image_path": str(tmp_path / "doc1.png"),
                        "duration_seconds": 0.123456,
                        "metrics": {"cer": 0.05},
                    },
                ],
            },
        ],
    }

    once = _normalize_for_snapshot(sample, tmp_path)
    twice = _normalize_for_snapshot(once, tmp_path)

    assert once == twice
    assert once["picarones_version"] == "PINNED"
    assert once["run_date"] == "PINNED"
    assert once["engine_reports"][0]["engine_version"] == "PINNED"
    assert once["engine_reports"][0]["document_results"][0]["duration_seconds"] == 0.0
    assert "FIXTURES" in once["corpus"]["source"]
    assert "FIXTURES" in once["engine_reports"][0]["document_results"][0]["image_path"]
