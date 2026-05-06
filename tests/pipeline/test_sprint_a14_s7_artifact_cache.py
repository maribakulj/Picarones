"""Sprint A14-S7 — ``ArtifactCache`` minimal.

Vérifie compute_key déterministe, get/put basique, et garde-fou
"un seul input sans content_hash → pas de clé".
"""

from __future__ import annotations

from picarones.domain import Artifact, ArtifactType
from picarones.pipeline import ArtifactCache, PipelineStep


def _hashed_artifact(
    suffix: str, type_: ArtifactType, content_hash: str | None = None,
) -> Artifact:
    return Artifact(
        id=f"d1:{suffix}",
        document_id="d1",
        type=type_,
        content_hash=content_hash,
    )


def _ocr_step() -> PipelineStep:
    return PipelineStep(
        id="ocr", kind="ocr", adapter_name="tesseract",
        params={"lang": "fra"},
        input_types=(ArtifactType.IMAGE,),
        output_types=(ArtifactType.RAW_TEXT,),
    )


class TestComputeKey:
    def test_returns_string_when_all_inputs_have_hash(self) -> None:
        cache = ArtifactCache()
        img = _hashed_artifact("img", ArtifactType.IMAGE, "a" * 64)
        key = cache.compute_key(_ocr_step(), {ArtifactType.IMAGE: img}, "1.0.0")
        assert key is not None
        assert len(key) == 64  # SHA-256 hex

    def test_deterministic(self) -> None:
        cache = ArtifactCache()
        img = _hashed_artifact("img", ArtifactType.IMAGE, "a" * 64)
        k1 = cache.compute_key(_ocr_step(), {ArtifactType.IMAGE: img}, "1.0.0")
        k2 = cache.compute_key(_ocr_step(), {ArtifactType.IMAGE: img}, "1.0.0")
        assert k1 == k2

    def test_different_content_hash_different_key(self) -> None:
        cache = ArtifactCache()
        img_a = _hashed_artifact("a", ArtifactType.IMAGE, "a" * 64)
        img_b = _hashed_artifact("b", ArtifactType.IMAGE, "b" * 64)
        k_a = cache.compute_key(_ocr_step(), {ArtifactType.IMAGE: img_a}, "1.0.0")
        k_b = cache.compute_key(_ocr_step(), {ArtifactType.IMAGE: img_b}, "1.0.0")
        assert k_a != k_b

    def test_different_code_version_different_key(self) -> None:
        cache = ArtifactCache()
        img = _hashed_artifact("img", ArtifactType.IMAGE, "a" * 64)
        k1 = cache.compute_key(_ocr_step(), {ArtifactType.IMAGE: img}, "1.0.0")
        k2 = cache.compute_key(_ocr_step(), {ArtifactType.IMAGE: img}, "2.0.0")
        assert k1 != k2

    def test_different_step_params_different_key(self) -> None:
        cache = ArtifactCache()
        img = _hashed_artifact("img", ArtifactType.IMAGE, "a" * 64)
        step_fra = PipelineStep(
            id="ocr", kind="ocr", adapter_name="tesseract",
            params={"lang": "fra"},
            input_types=(ArtifactType.IMAGE,),
            output_types=(ArtifactType.RAW_TEXT,),
        )
        step_eng = PipelineStep(
            id="ocr", kind="ocr", adapter_name="tesseract",
            params={"lang": "eng"},
            input_types=(ArtifactType.IMAGE,),
            output_types=(ArtifactType.RAW_TEXT,),
        )
        k_fra = cache.compute_key(step_fra, {ArtifactType.IMAGE: img}, "1.0.0")
        k_eng = cache.compute_key(step_eng, {ArtifactType.IMAGE: img}, "1.0.0")
        assert k_fra != k_eng

    def test_returns_none_when_input_has_no_hash(self) -> None:
        cache = ArtifactCache()
        img = _hashed_artifact("img", ArtifactType.IMAGE, content_hash=None)
        key = cache.compute_key(_ocr_step(), {ArtifactType.IMAGE: img}, "1.0.0")
        assert key is None


class TestGetPutClear:
    def test_get_miss_returns_none(self) -> None:
        cache = ArtifactCache()
        assert cache.get("non_existent") is None

    def test_put_then_get_returns_outputs(self) -> None:
        cache = ArtifactCache()
        artifacts = {
            ArtifactType.RAW_TEXT: _hashed_artifact(
                "raw", ArtifactType.RAW_TEXT, "f" * 64,
            ),
        }
        cache.put("k1", artifacts)
        cached = cache.get("k1")
        assert cached is not None
        assert ArtifactType.RAW_TEXT in cached

    def test_put_with_none_key_is_noop(self) -> None:
        cache = ArtifactCache()
        cache.put(None, {ArtifactType.RAW_TEXT: _hashed_artifact(
            "raw", ArtifactType.RAW_TEXT, "f" * 64,
        )})
        assert len(cache) == 0

    def test_get_with_none_key_returns_none(self) -> None:
        cache = ArtifactCache()
        assert cache.get(None) is None

    def test_clear(self) -> None:
        cache = ArtifactCache()
        cache.put("k", {ArtifactType.RAW_TEXT: _hashed_artifact(
            "raw", ArtifactType.RAW_TEXT, "f" * 64,
        )})
        assert len(cache) == 1
        cache.clear()
        assert len(cache) == 0

    def test_contains(self) -> None:
        cache = ArtifactCache()
        cache.put("foo", {})
        assert "foo" in cache
        assert "bar" not in cache

    def test_keys(self) -> None:
        cache = ArtifactCache()
        cache.put("a", {})
        cache.put("b", {})
        assert sorted(cache.keys()) == ["a", "b"]

    def test_put_makes_defensive_copy(self) -> None:
        """Modifier le dict d'origine après put() ne doit pas
        affecter le contenu du cache."""
        cache = ArtifactCache()
        artifacts = {
            ArtifactType.RAW_TEXT: _hashed_artifact(
                "raw", ArtifactType.RAW_TEXT, "f" * 64,
            ),
        }
        cache.put("k", artifacts)
        artifacts.clear()
        cached = cache.get("k")
        assert cached is not None
        assert ArtifactType.RAW_TEXT in cached
