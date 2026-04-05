from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mata.annotate.ai_assist import AIAssist
from mata.annotate.server import AnnotateServer


class _FakeDetectAdapter:
    def __init__(self, instances: list[SimpleNamespace]) -> None:
        self.instances = instances
        self.calls: list[tuple[str, float]] = []

    def predict(self, image_path: str, threshold: float = 0.3):
        self.calls.append((image_path, threshold))
        return SimpleNamespace(instances=self.instances)


def test_detect_assist_converts_xyxy_to_xywh() -> None:
    assist = AIAssist()
    assist._detect_adapter = _FakeDetectAdapter(
        [
            SimpleNamespace(
                bbox=(10, 20, 60, 90),
                label=1,
                label_name="person",
                score=0.95,
            )
        ]
    )

    candidates = assist.detect_assist("image.jpg", threshold=0.3)

    assert candidates == [
        {
            "bbox_xywh": [10.0, 20.0, 50.0, 70.0],
            "bbox_xyxy": [10.0, 20.0, 60.0, 90.0],
            "label": "person",
            "label_id": 1,
            "score": 0.95,
            "source": "detect",
        }
    ]


def test_detect_assist_filters_low_scores_after_prediction() -> None:
    adapter = _FakeDetectAdapter(
        [
            SimpleNamespace(bbox=(0, 0, 10, 10), label=0, label_name="keep", score=0.8),
            SimpleNamespace(bbox=(10, 10, 20, 20), label=1, label_name="drop", score=0.2),
        ]
    )
    assist = AIAssist()
    assist._detect_adapter = adapter

    candidates = assist.detect_assist("image.jpg", threshold=0.5)

    assert adapter.calls == [("image.jpg", 0.5)]
    assert len(candidates) == 1
    assert candidates[0]["label"] == "keep"


def test_detect_assist_uses_class_map_override() -> None:
    assist = AIAssist()
    assist._detect_adapter = _FakeDetectAdapter(
        [SimpleNamespace(bbox=(1, 2, 6, 8), label=3, label_name="old-name", score=0.91)]
    )

    candidates = assist.detect_assist("image.jpg", class_map={"3": "car"})

    assert candidates[0]["label"] == "car"


def test_detect_assist_lazy_loads_and_caches_model(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeDetectAdapter(
        [SimpleNamespace(bbox=(1, 2, 6, 8), label=0, label_name="cat", score=0.9)]
    )
    load_calls: list[tuple[str, str]] = []

    def fake_load(task: str, model: str, **kwargs):
        load_calls.append((task, model))
        return adapter

    import mata

    monkeypatch.setattr(mata, "load", fake_load)

    assist = AIAssist(detect_model="fake-detector")

    first = assist.detect_assist("image.jpg")
    second = assist.detect_assist("image-2.jpg")

    assert len(first) == 1
    assert len(second) == 1
    assert load_calls == [("detect", "fake-detector")]
    assert adapter.calls == [("image.jpg", 0.3), ("image-2.jpg", 0.3)]


def test_detect_assist_requires_configured_model_when_adapter_missing() -> None:
    assist = AIAssist()

    with pytest.raises(ValueError, match="No detection model configured"):
        assist.detect_assist("image.jpg")


def test_server_preloads_detect_model_when_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = _FakeDetectAdapter([])
    load_calls: list[tuple[str, str]] = []

    def fake_load(task: str, model: str, **kwargs):
        load_calls.append((task, model))
        return adapter

    import mata

    monkeypatch.setattr(mata, "load", fake_load)

    server = AnnotateServer(data_root=str(tmp_path), port=0, detect_model="fake-detector")
    try:
        assert isinstance(server.ai_assist, AIAssist)
        assert load_calls == [("detect", "fake-detector")]
    finally:
        server._httpd.server_close()


# ---------------------------------------------------------------------------
# Task E3: CLIP classification suggestions
# ---------------------------------------------------------------------------


class _FakeClassifyAdapter:
    """Minimal stand-in for a CLIP/classify adapter."""

    def __init__(self, predictions: list[SimpleNamespace]) -> None:
        self.predictions = predictions
        self.calls: list[tuple[str, list[str]]] = []

    def predict(self, image_path: str, text_prompts: list[str] | None = None):
        self.calls.append((image_path, list(text_prompts or [])))
        return SimpleNamespace(predictions=self.predictions)


def _make_classify_adapter(labels_scores: list[tuple[str, float]]) -> _FakeClassifyAdapter:
    preds = [SimpleNamespace(label=lbl, score=score) for lbl, score in labels_scores]
    return _FakeClassifyAdapter(preds)


def test_classify_assist_returns_sorted_suggestions() -> None:
    adapter = _make_classify_adapter([("dog", 0.10), ("bird", 0.05), ("cat", 0.85)])
    assist = AIAssist()
    assist._embed_adapter = adapter

    suggestions = assist.classify_assist("cat.jpg", ["cat", "dog", "bird"])

    assert suggestions[0]["label"] == "cat"
    assert suggestions[0]["score"] == pytest.approx(0.85)
    assert suggestions[1]["label"] == "dog"
    assert suggestions[2]["label"] == "bird"


def test_classify_assist_all_have_clip_source() -> None:
    adapter = _make_classify_adapter([("a", 0.6), ("b", 0.4)])
    assist = AIAssist()
    assist._embed_adapter = adapter

    suggestions = assist.classify_assist("img.jpg", ["a", "b"])

    assert all(s["source"] == "clip" for s in suggestions)


def test_classify_assist_passes_class_names_as_text_prompts() -> None:
    adapter = _make_classify_adapter([("cat", 0.9), ("dog", 0.1)])
    assist = AIAssist()
    assist._embed_adapter = adapter

    assist.classify_assist("img.jpg", ["cat", "dog"])

    assert adapter.calls == [("img.jpg", ["cat", "dog"])]


def test_classify_assist_scores_are_floats() -> None:
    import numpy as np

    adapter = _make_classify_adapter([])
    # Simulate numpy scalars that need to be cast
    adapter.predictions = [
        SimpleNamespace(label="cat", score=np.float32(0.7)),
        SimpleNamespace(label="dog", score=np.float32(0.3)),
    ]
    assist = AIAssist()
    assist._embed_adapter = adapter

    suggestions = assist.classify_assist("img.jpg", ["cat", "dog"])

    for s in suggestions:
        assert isinstance(s["score"], float)


def test_classify_assist_lazy_loads_model(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_classify_adapter([("cat", 0.9)])
    load_calls: list[tuple[str, str]] = []

    def fake_load(task: str, model: str, **kwargs):
        load_calls.append((task, model))
        return adapter

    import mata

    monkeypatch.setattr(mata, "load", fake_load)

    assist = AIAssist(embed_model="openai/clip-vit-base-patch32")

    first = assist.classify_assist("img1.jpg", ["cat"])
    second = assist.classify_assist("img2.jpg", ["cat"])

    assert len(first) == 1
    assert len(second) == 1
    # load called exactly once; CLIP loaded as "classify" task
    assert load_calls == [("classify", "openai/clip-vit-base-patch32")]
    assert adapter.calls == [("img1.jpg", ["cat"]), ("img2.jpg", ["cat"])]


def test_classify_assist_requires_configured_model_when_adapter_missing() -> None:
    assist = AIAssist()

    with pytest.raises(ValueError, match="No embed model configured"):
        assist.classify_assist("img.jpg", ["cat"])


def test_load_embed_caches_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_classify_adapter([])
    load_calls: list = []

    def fake_load(task: str, model: str, **kwargs):
        load_calls.append((task, model))
        return adapter

    import mata

    monkeypatch.setattr(mata, "load", fake_load)

    assist = AIAssist()
    assist.load_embed("openai/clip-vit-base-patch32")
    assist.load_embed("openai/clip-vit-base-patch32")  # second call — should not reload

    assert load_calls == [("classify", "openai/clip-vit-base-patch32")]


def test_load_embed_raises_without_model() -> None:
    assist = AIAssist()

    with pytest.raises(ValueError, match="No embed model configured"):
        assist.load_embed()


def test_classify_assist_empty_predictions_returns_empty_list() -> None:
    adapter = _FakeClassifyAdapter([])
    assist = AIAssist()
    assist._embed_adapter = adapter

    suggestions = assist.classify_assist("img.jpg", ["cat", "dog"])

    assert suggestions == []


# ---------------------------------------------------------------------------
# Task E2: VLM auto-annotation
# ---------------------------------------------------------------------------


class _FakeVLMAdapter:
    """Minimal stand-in for a HuggingFace VLM adapter."""

    def __init__(self, instances=None, entities=None, text: str = "") -> None:
        self._instances = instances or []
        self._entities = entities or []
        self._text = text
        self.calls: list[dict] = []

    def predict(self, image=None, prompt=None, output_mode=None,
                max_new_tokens=None, auto_promote=False, **kwargs):
        self.calls.append({
            "image": image,
            "prompt": prompt,
            "output_mode": output_mode,
            "max_new_tokens": max_new_tokens,
            "auto_promote": auto_promote,
        })
        return SimpleNamespace(
            instances=self._instances,
            entities=self._entities,
            text=self._text,
            meta={"text": self._text},
        )


def _make_vlm_instance(bbox, label_id=1, label_name="cat", score=0.8):
    return SimpleNamespace(
        bbox=bbox,
        label=label_id,
        label_name=label_name,
        score=score,
    )


def _make_vlm_entity(label: str, score: float = 0.7):
    return SimpleNamespace(label=label, score=score)


def test_vlm_assist_returns_instances_as_candidates() -> None:
    inst = _make_vlm_instance((10.0, 20.0, 60.0, 90.0), label_id=1, label_name="cat", score=0.9)
    adapter = _FakeVLMAdapter(instances=[inst])
    assist = AIAssist()
    assist._vlm_adapter = adapter

    candidates = assist.vlm_assist("image.jpg", class_names=["cat"])

    assert len(candidates) == 1
    c = candidates[0]
    assert c["bbox_xywh"] == [10.0, 20.0, 50.0, 70.0]
    assert c["bbox_xyxy"] == [10.0, 20.0, 60.0, 90.0]
    assert c["label"] == "cat"
    assert c["label_id"] == 1
    assert c["score"] == pytest.approx(0.9)
    assert c["source"] == "vlm"


def test_vlm_assist_entities_have_vlm_entity_source() -> None:
    ent = _make_vlm_entity("dog", score=0.6)
    adapter = _FakeVLMAdapter(entities=[ent])
    assist = AIAssist()
    assist._vlm_adapter = adapter

    candidates = assist.vlm_assist("image.jpg")

    assert len(candidates) == 1
    assert candidates[0]["label"] == "dog"
    assert candidates[0]["score"] == pytest.approx(0.6)
    assert candidates[0]["source"] == "vlm_entity"
    assert "bbox_xywh" not in candidates[0]


def test_vlm_assist_builds_prompt_from_class_names() -> None:
    adapter = _FakeVLMAdapter()
    assist = AIAssist()
    assist._vlm_adapter = adapter

    assist.vlm_assist("image.jpg", class_names=["cat", "dog"])

    prompt = adapter.calls[0]["prompt"]
    assert "cat" in prompt
    assert "dog" in prompt
    assert "x1, y1, x2, y2" in prompt


def test_vlm_assist_custom_prompt_overrides_class_names() -> None:
    adapter = _FakeVLMAdapter()
    assist = AIAssist()
    assist._vlm_adapter = adapter

    assist.vlm_assist("image.jpg", class_names=["cat"], prompt="My custom prompt")

    assert adapter.calls[0]["prompt"] == "My custom prompt"


def test_vlm_assist_uses_output_mode_detect_and_auto_promote() -> None:
    adapter = _FakeVLMAdapter()
    assist = AIAssist()
    assist._vlm_adapter = adapter

    assist.vlm_assist("image.jpg")

    assert adapter.calls[0]["output_mode"] == "detect"
    assert adapter.calls[0]["auto_promote"] is True


def test_vlm_assist_max_new_tokens_forwarded() -> None:
    adapter = _FakeVLMAdapter()
    assist = AIAssist()
    assist._vlm_adapter = adapter

    assist.vlm_assist("image.jpg", max_new_tokens=512)

    assert adapter.calls[0]["max_new_tokens"] == 512


def test_vlm_assist_returns_empty_list_on_failure() -> None:
    class _BrokenAdapter:
        def predict(self, **kwargs):
            raise RuntimeError("VLM exploded")

    assist = AIAssist()
    assist._vlm_adapter = _BrokenAdapter()

    candidates = assist.vlm_assist("image.jpg")

    assert candidates == []


def test_vlm_assist_returns_empty_list_on_no_results() -> None:
    adapter = _FakeVLMAdapter()
    assist = AIAssist()
    assist._vlm_adapter = adapter

    candidates = assist.vlm_assist("image.jpg")

    assert candidates == []


def test_vlm_assist_handles_none_score_on_instance() -> None:
    inst = _make_vlm_instance((0, 0, 50, 50), score=None)
    adapter = _FakeVLMAdapter(instances=[inst])
    assist = AIAssist()
    assist._vlm_adapter = adapter

    candidates = assist.vlm_assist("image.jpg")

    assert candidates[0]["score"] == pytest.approx(0.5)


def test_vlm_assist_handles_none_label_name_on_instance() -> None:
    inst = _make_vlm_instance((0, 0, 50, 50), label_name=None, score=0.9)
    adapter = _FakeVLMAdapter(instances=[inst])
    assist = AIAssist()
    assist._vlm_adapter = adapter

    candidates = assist.vlm_assist("image.jpg")

    assert candidates[0]["label"] == "object"


def test_vlm_assist_skips_entity_with_empty_label() -> None:
    ent = _make_vlm_entity("", score=0.5)
    adapter = _FakeVLMAdapter(entities=[ent])
    assist = AIAssist()
    assist._vlm_adapter = adapter

    candidates = assist.vlm_assist("image.jpg")

    assert candidates == []


def test_vlm_assist_lazy_loads_and_caches_model(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeVLMAdapter()
    load_calls: list[tuple[str, str]] = []

    def fake_load(task: str, model: str, **kwargs):
        load_calls.append((task, model))
        return adapter

    import mata
    monkeypatch.setattr(mata, "load", fake_load)

    assist = AIAssist(vlm_model="Qwen/Qwen3-VL-2B-Instruct")

    assist.vlm_assist("img1.jpg")
    assist.vlm_assist("img2.jpg")

    assert load_calls == [("vlm", "Qwen/Qwen3-VL-2B-Instruct")]
    assert len(adapter.calls) == 2


def test_vlm_assist_requires_configured_model_when_adapter_missing() -> None:
    assist = AIAssist()

    with pytest.raises(ValueError, match="No VLM model configured"):
        assist.vlm_assist("image.jpg")


def test_vlm_describe_returns_text() -> None:
    adapter = _FakeVLMAdapter(text="A cat sitting on a mat.")
    assist = AIAssist()
    assist._vlm_adapter = adapter

    description = assist.vlm_describe("image.jpg")

    assert description == "A cat sitting on a mat."


def test_vlm_describe_uses_default_prompt() -> None:
    adapter = _FakeVLMAdapter(text="A scene.")
    assist = AIAssist()
    assist._vlm_adapter = adapter

    assist.vlm_describe("image.jpg")

    assert "Describe" in adapter.calls[0]["prompt"]


def test_vlm_describe_uses_custom_prompt() -> None:
    adapter = _FakeVLMAdapter(text="Result")
    assist = AIAssist()
    assist._vlm_adapter = adapter

    assist.vlm_describe("image.jpg", prompt="What is the dominant color?")

    assert adapter.calls[0]["prompt"] == "What is the dominant color?"


def test_vlm_describe_returns_empty_string_on_failure() -> None:
    class _BrokenAdapter:
        def predict(self, **kwargs):
            raise RuntimeError("VLM offline")

    assist = AIAssist()
    assist._vlm_adapter = _BrokenAdapter()

    result = assist.vlm_describe("image.jpg")

    assert result == ""


def test_load_vlm_caches_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeVLMAdapter()
    load_calls: list = []

    def fake_load(task: str, model: str, **kwargs):
        load_calls.append((task, model))
        return adapter

    import mata
    monkeypatch.setattr(mata, "load", fake_load)

    assist = AIAssist()
    assist.load_vlm("Qwen/Qwen3-VL-2B-Instruct")
    assist.load_vlm("Qwen/Qwen3-VL-2B-Instruct")  # second call — should not reload

    assert load_calls == [("vlm", "Qwen/Qwen3-VL-2B-Instruct")]


def test_load_vlm_raises_without_model() -> None:
    assist = AIAssist()

    with pytest.raises(ValueError, match="No VLM model configured"):
        assist.load_vlm()


def test_server_preloads_vlm_model_when_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _FakeVLMAdapter()
    load_calls: list[tuple[str, str]] = []

    def fake_load(task: str, model: str, **kwargs):
        load_calls.append((task, model))
        return adapter

    import mata
    monkeypatch.setattr(mata, "load", fake_load)

    server = AnnotateServer(data_root=str(tmp_path), port=0, vlm_model="Qwen/Qwen3-VL-2B-Instruct")
    try:
        assert isinstance(server.ai_assist, AIAssist)
        assert ("vlm", "Qwen/Qwen3-VL-2B-Instruct") in load_calls
    finally:
        server._httpd.server_close()


# ---------------------------------------------------------------------------
# Task G3: Comprehensive cross-adapter source field & score normalization tests
# ---------------------------------------------------------------------------


def test_all_candidates_have_source_field() -> None:
    """All three adapter types must set a non-empty 'source' field."""
    # --- detect ---
    detect_adapter = _FakeDetectAdapter(
        [SimpleNamespace(bbox=(0, 0, 10, 10), label=1, label_name="car", score=0.9)]
    )
    detect_assist = AIAssist()
    detect_assist._detect_adapter = detect_adapter
    detect_candidates = detect_assist.detect_assist("img.jpg")
    assert all("source" in c and c["source"] == "detect" for c in detect_candidates)

    # --- vlm instance ---
    vlm_inst = _make_vlm_instance((5, 5, 30, 30), label_id=2, label_name="person", score=0.75)
    vlm_ent = _make_vlm_entity("scene", score=0.5)
    vlm_adapter = _FakeVLMAdapter(instances=[vlm_inst], entities=[vlm_ent])
    vlm_assist = AIAssist()
    vlm_assist._vlm_adapter = vlm_adapter
    vlm_candidates = vlm_assist.vlm_assist("img.jpg")
    sources = {c["source"] for c in vlm_candidates}
    assert sources == {"vlm", "vlm_entity"}

    # --- clip ---
    clip_adapter = _make_classify_adapter([("cat", 0.6), ("dog", 0.4)])
    clip_assist = AIAssist()
    clip_assist._embed_adapter = clip_adapter
    clip_candidates = clip_assist.classify_assist("img.jpg", ["cat", "dog"])
    assert all("source" in c and c["source"] == "clip" for c in clip_candidates)


def test_classify_assist_score_normalization() -> None:
    """CLIP scores should sum to ~1.0 when the adapter returns softmax-normalized values."""
    # Simulate softmax-normalized scores from a CLIP adapter
    scores = [0.70, 0.20, 0.10]
    labels = ["cat", "dog", "bird"]
    adapter = _make_classify_adapter(list(zip(labels, scores)))
    assist = AIAssist()
    assist._embed_adapter = adapter

    suggestions = assist.classify_assist("cat.jpg", labels)

    total = sum(s["score"] for s in suggestions)
    assert total == pytest.approx(1.0, abs=1e-5)
    # Result should be sorted descending
    assert suggestions[0]["label"] == "cat"
    assert suggestions[0]["score"] == pytest.approx(0.70)