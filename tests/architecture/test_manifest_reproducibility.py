"""Garde-fou de reproductibilité du ``RunManifest``.

L'audit S58 a relevé que ``RunManifest.dependencies_lock`` n'était
jamais peuplé et que ``pipeline_specs`` ne contenait que les noms,
rompant la promesse documentée *« à code_version + corpus + specs +
dependencies_lock identiques, ré-exécuter doit donner les mêmes
résultats »*.

Ces tests verrouillent le contrat :

1. ``capture_dependencies_lock()`` retourne un dict non vide trié.
2. ``RunManifest`` accepte des ``pipeline_specs`` complètes (steps,
   adapter_name, params, inputs_from), pas seulement des noms.
3. ``adapter_kwargs`` permet de reconstituer les constructeurs
   d'adapters (model, temperature, etc.).
4. La sérialisation est déterministe : deux manifests à entrée
   identique produisent les mêmes octets JSON.
"""

from __future__ import annotations

from datetime import datetime, timezone

from picarones.app.services.dependencies import capture_dependencies_lock
from picarones.domain.artifacts import ArtifactType
from picarones.domain.pipeline_spec import PipelineSpec, PipelineStep
from picarones.domain.run_manifest import RunManifest


def test_capture_dependencies_lock_non_empty_and_sorted() -> None:
    """``capture_dependencies_lock()`` retourne ≥ 1 paquet (pydantic
    au minimum) et trié alphabétiquement (case-insensitive).
    """
    lock = capture_dependencies_lock()
    assert len(lock) > 0, "lock vide — picarones lui-même doit être listé."
    keys = list(lock.keys())
    assert keys == sorted(keys, key=str.lower), (
        "lock non trié — le manifest ne sera pas bit-for-bit "
        "reproductible cross-environnement."
    )
    # pydantic est une dépendance ferme du projet — sa présence prouve
    # que la capture marche sur l'env réel.
    assert any(k.lower() == "pydantic" for k in lock)


def test_run_manifest_carries_full_pipeline_specs() -> None:
    """Le manifest doit porter les ``PipelineSpec`` complètes, pas
    seulement les noms.  Sans ça, un relecteur 5 ans plus tard ne peut
    pas reconstituer le DAG sans accès au YAML d'origine.
    """
    step = PipelineStep(
        id="ocr",
        kind="ocr",
        adapter_name="tesseract",
        input_types=(ArtifactType.IMAGE,),
        output_types=(ArtifactType.RAW_TEXT,),
        params={"lang": "fra"},
    )
    spec = PipelineSpec(name="tess_only", steps=(step,))

    manifest = RunManifest(
        run_id="r1",
        corpus_name="c1",
        n_documents=1,
        pipeline_specs=(spec,),
        adapter_kwargs={"tesseract": {"lang": "fra", "psm": 6}},
        view_specs=(),
        code_version="1.0.0-test",
        started_at=datetime.now(tz=timezone.utc),
        completed_at=datetime.now(tz=timezone.utc),
        dependencies_lock={"pydantic": "2.5.0"},
    )

    assert manifest.pipeline_specs == (spec,)
    # Vue rétrocompat dérivée des specs.
    assert manifest.pipeline_names == ("tess_only",)
    # Les kwargs d'instanciation sont tracés.
    assert manifest.adapter_kwargs["tesseract"]["psm"] == 6
    # Le step complet est reconstituable.
    assert manifest.pipeline_specs[0].steps[0].params == {"lang": "fra"}


def test_run_manifest_serialization_is_deterministic() -> None:
    """Deux manifests à entrée identique produisent les mêmes
    octets JSON — pré-requis pour le hash d'intégrité que la BnF
    peut citer dans une publication.
    """
    common = dict(
        run_id="r1",
        corpus_name="c1",
        n_documents=42,
        pipeline_specs=(),
        adapter_kwargs={"a": {"k": 1}, "b": {"k": 2}},
        view_specs=(),
        code_version="1.0.0",
        started_at=datetime(2026, 5, 6, tzinfo=timezone.utc),
        completed_at=datetime(2026, 5, 6, tzinfo=timezone.utc),
        dependencies_lock={"pkg-a": "1.0", "pkg-b": "2.0"},
        metadata={"note": "test"},
    )
    m1 = RunManifest(**common)
    m2 = RunManifest(**common)
    assert m1.model_dump_json() == m2.model_dump_json()


def test_run_manifest_rejects_extra_fields() -> None:
    """``extra="forbid"`` — le contrat du manifest n'évolue pas
    silencieusement.  Tout nouveau champ exige un ajout explicite
    au modèle (et donc une revue).
    """
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RunManifest(
            run_id="r1",
            corpus_name="c1",
            n_documents=1,
            code_version="1.0",
            started_at=datetime.now(tz=timezone.utc),
            completed_at=datetime.now(tz=timezone.utc),
            unknown_field="nope",  # type: ignore[call-arg]
        )
