"""Tests for ValkeyStore and ValkeyLoad graph nodes — Task B1, B2."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.classifications import Classifications
from mata.core.artifacts.depth_map import DepthMap
from mata.core.artifacts.detections import Detections
from mata.core.artifacts.masks import Masks
from mata.core.graph.context import ExecutionContext
from mata.core.graph.graph import CompiledGraph, Graph
from mata.core.graph.node import Node
from mata.core.types import Classification, ClassifyResult, DepthResult, Instance, VisionResult
from mata.nodes.valkey_load import ValkeyLoad
from mata.nodes.valkey_store import ValkeyStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_instance(label: str = "cat", score: float = 0.9) -> Instance:
    return Instance(label=0, label_name=label, score=score, bbox=[10, 20, 100, 200])


def _make_vision_result(n: int = 2) -> VisionResult:
    instances = [_make_instance(score=0.9 - i * 0.1) for i in range(n)]
    return VisionResult(instances=instances)


def _make_detections(n: int = 2) -> Detections:
    return Detections.from_vision_result(_make_vision_result(n))


def _make_masks() -> Masks:
    mask_arr = np.ones((64, 64), dtype=bool)
    instances = [
        Instance(label=0, label_name="cat", score=0.9, bbox=[10, 20, 100, 200], mask=mask_arr),
        Instance(label=1, label_name="dog", score=0.8, bbox=[50, 60, 150, 250], mask=mask_arr),
    ]
    return Masks(instances=instances)


def _make_classifications() -> Classifications:
    preds = [
        Classification(label=0, score=0.95, label_name="cat"),
        Classification(label=1, score=0.05, label_name="dog"),
    ]
    return Classifications(predictions=tuple(preds))


def _make_depth_map() -> DepthMap:
    depth_arr = np.ones((64, 64), dtype=np.float32)
    result = DepthResult(depth=depth_arr, normalized=depth_arr)
    return DepthMap.from_depth_result(result)


def _make_ctx() -> ExecutionContext:
    return ExecutionContext(providers={}, device="cpu")


# ---------------------------------------------------------------------------
# TestValkeyStoreNode
# ---------------------------------------------------------------------------


class TestValkeyStoreNode:
    """Tests for ValkeyStore sink node."""

    def test_inherits_from_node(self):
        node = ValkeyStore(src="dets", url="valkey://localhost:6379", key="test:key")
        assert isinstance(node, Node)

    def test_declares_inputs_outputs(self):
        # inputs/outputs are set dynamically in __init__ based on src
        node = ValkeyStore(src="dets", url="valkey://localhost:6379", key="test:key")
        assert "dets" in node.inputs
        assert "dets" in node.outputs
        assert node.inputs["dets"] is Artifact
        assert node.outputs["dets"] is Artifact

    def test_importable_from_mata_nodes(self):
        from mata.nodes import ValkeyStore as VS  # noqa: F401, N817

        assert VS is ValkeyStore

    def test_run_stores_artifact(self):
        artifact = _make_detections()
        ctx = _make_ctx()
        node = ValkeyStore(src="dets", url="valkey://localhost:6379", key="test:dets")

        mock_result = MagicMock()
        mock_result.to_json.return_value = '{"instances": []}'

        with patch(
            "mata.nodes.valkey_store.ValkeyStore._artifact_to_serializable",
            return_value=mock_result,
        ):
            with patch("mata.core.exporters.valkey_exporter.export_valkey") as mock_export:
                node.run(ctx, artifact=artifact)
                assert mock_export.called

    def test_passthrough_artifact_unchanged(self):
        artifact = _make_detections()
        ctx = _make_ctx()
        node = ValkeyStore(src="dets", url="valkey://localhost:6379", key="test:dets", out="dets_out")

        mock_result = MagicMock()
        mock_result.to_json.return_value = "{}"

        with patch(
            "mata.nodes.valkey_store.ValkeyStore._artifact_to_serializable",
            return_value=mock_result,
        ):
            with patch("mata.core.exporters.valkey_exporter.export_valkey"):
                result = node.run(ctx, artifact=artifact)

        # The artifact must be returned unchanged
        assert "dets_out" in result
        assert result["dets_out"] is artifact

    def test_key_template_node_placeholder(self):
        artifact = _make_detections()
        ctx = _make_ctx()
        node = ValkeyStore(src="dets", url="valkey://localhost:6379", key="pipeline:{node}:result")

        captured_keys = []

        mock_result = MagicMock()
        mock_result.to_json.return_value = "{}"

        with patch(
            "mata.nodes.valkey_store.ValkeyStore._artifact_to_serializable",
            return_value=mock_result,
        ):
            with patch("mata.core.exporters.valkey_exporter.export_valkey") as mock_export:
                node.run(ctx, artifact=artifact)
                call_kwargs = mock_export.call_args
                captured_keys.append(call_kwargs[1]["key"] if call_kwargs[1] else call_kwargs[0][2])

        # {node} should be replaced with node name
        resolved_key = captured_keys[0]
        assert "{node}" not in resolved_key
        assert "ValkeyStore" in resolved_key

    def test_key_template_timestamp_placeholder(self):
        artifact = _make_detections()
        ctx = _make_ctx()
        node = ValkeyStore(src="dets", url="valkey://localhost:6379", key="pipeline:frame:{timestamp}")

        mock_result = MagicMock()
        mock_result.to_json.return_value = "{}"

        with patch(
            "mata.nodes.valkey_store.ValkeyStore._artifact_to_serializable",
            return_value=mock_result,
        ):
            with patch("mata.core.exporters.valkey_exporter.export_valkey") as mock_export:
                node.run(ctx, artifact=artifact)

        call_kwargs = mock_export.call_args
        key_arg = call_kwargs[1].get("key") or call_kwargs[0][2]
        assert "{timestamp}" not in key_arg
        # Should be numeric (timestamp)
        ts_part = key_arg.split(":")[-1]
        assert ts_part.isdigit()

    def test_ttl_parameter_forwarded(self):
        artifact = _make_detections()
        ctx = _make_ctx()
        node = ValkeyStore(src="dets", url="valkey://localhost:6379", key="test:key", ttl=300)

        mock_result = MagicMock()
        mock_result.to_json.return_value = "{}"

        with patch(
            "mata.nodes.valkey_store.ValkeyStore._artifact_to_serializable",
            return_value=mock_result,
        ):
            with patch("mata.core.exporters.valkey_exporter.export_valkey") as mock_export:
                node.run(ctx, artifact=artifact)

        call_kwargs = mock_export.call_args
        assert call_kwargs[1].get("ttl") == 300 or (len(call_kwargs[0]) > 3 and call_kwargs[0][3] == 300)

    def test_serializer_parameter_forwarded(self):
        artifact = _make_detections()
        ctx = _make_ctx()
        node = ValkeyStore(src="dets", url="valkey://localhost:6379", key="test:key", serializer="msgpack")

        mock_result = MagicMock()
        mock_result.to_json.return_value = "{}"

        with patch(
            "mata.nodes.valkey_store.ValkeyStore._artifact_to_serializable",
            return_value=mock_result,
        ):
            with patch("mata.core.exporters.valkey_exporter.export_valkey") as mock_export:
                node.run(ctx, artifact=artifact)

        call_kwargs = mock_export.call_args
        assert call_kwargs[1].get("serializer") == "msgpack"

    def test_graph_compilation(self):
        """ValkeyStore compiles successfully in a Graph DAG."""
        from unittest.mock import Mock

        from mata.nodes.detect import Detect

        mock_detector = Mock()
        mock_detector.predict = Mock(return_value=_make_vision_result())

        # ValkeyStore.inputs is keyed by src so auto-wiring picks up "dets"
        graph = (
            Graph()
            .then(Detect(using="detr", out="dets"))
            .then(ValkeyStore(src="dets", url="valkey://localhost:6379", key="test:dets"))
        )

        compiled = graph.compile(providers={"detr": mock_detector})
        assert isinstance(compiled, CompiledGraph)
        assert compiled.validation_result.valid
        assert len(compiled.nodes) == 2

    def test_graph_execution_with_detect(self):
        """ValkeyStore executes cleanly in a full graph pipeline."""
        from unittest.mock import Mock

        from PIL import Image as PILImage

        from mata.core.artifacts.image import Image
        from mata.core.graph.scheduler import SyncScheduler
        from mata.nodes.detect import Detect

        vision_result = _make_vision_result()
        mock_detector = Mock()
        mock_detector.predict = Mock(return_value=vision_result)

        compile_providers = {"detr": mock_detector}
        ctx_providers = {"detect": {"detr": mock_detector}}

        # ValkeyStore.inputs is keyed by src so auto-wiring picks up "dets"
        graph = (
            Graph()
            .then(Detect(using="detr", out="dets"))
            .then(ValkeyStore(src="dets", url="valkey://localhost:6379", key="test:dets"))
        )

        compiled = graph.compile(providers=compile_providers)
        ctx = ExecutionContext(providers=ctx_providers, device="cpu")

        pil_img = PILImage.new("RGB", (64, 64), color=(128, 128, 128))
        image_artifact = Image.from_pil(pil_img)

        with patch("mata.core.exporters.valkey_exporter.export_valkey"):
            scheduler = SyncScheduler()
            result = scheduler.execute(
                compiled,
                ctx,
                initial_artifacts={"input.image": image_artifact},
            )

        # The store node should pass through; result contains all artifacts
        assert result is not None

    def test_artifact_to_serializable_detections(self):
        artifact = _make_detections()
        result = ValkeyStore._artifact_to_serializable(artifact)
        assert isinstance(result, VisionResult)

    def test_artifact_to_serializable_masks(self):
        artifact = _make_masks()
        result = ValkeyStore._artifact_to_serializable(artifact)
        assert isinstance(result, VisionResult)

    def test_artifact_to_serializable_classifications(self):
        artifact = _make_classifications()
        result = ValkeyStore._artifact_to_serializable(artifact)
        assert isinstance(result, ClassifyResult)

    def test_artifact_to_serializable_depth(self):
        artifact = _make_depth_map()
        result = ValkeyStore._artifact_to_serializable(artifact)
        assert isinstance(result, DepthResult)

    def test_artifact_to_serializable_fallback(self):
        """Unknown artifact types are returned as-is (fallback)."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class CustomArtifact(Artifact):
            def to_dict(self) -> dict:
                return {}

            @classmethod
            def from_dict(cls, data: dict) -> CustomArtifact:
                return cls()

        artifact = CustomArtifact()
        result = ValkeyStore._artifact_to_serializable(artifact)
        # Fallback: returns the artifact itself
        assert result is artifact

    def test_default_output_name_same_as_src(self):
        """out defaults to src when not provided."""
        node = ValkeyStore(src="my_dets", url="valkey://localhost:6379", key="k")
        assert node.output_name == "my_dets"

    def test_custom_output_name(self):
        node = ValkeyStore(src="my_dets", url="valkey://localhost:6379", key="k", out="stored")
        assert node.output_name == "stored"


# ---------------------------------------------------------------------------
# TestValkeyLoadNode
# ---------------------------------------------------------------------------


class TestValkeyLoadNode:
    """Tests for ValkeyLoad source node."""

    def test_inherits_from_node(self):
        node = ValkeyLoad(url="valkey://localhost:6379", key="test:key")
        assert isinstance(node, Node)

    def test_no_inputs_declared(self):
        assert ValkeyLoad.inputs == {}

    def test_importable_from_mata_nodes(self):
        from mata.nodes import ValkeyLoad as VL  # noqa: F401, N817

        assert VL is ValkeyLoad

    def test_run_loads_artifact(self):
        vision_result = _make_vision_result()
        ctx = _make_ctx()
        node = ValkeyLoad(url="valkey://localhost:6379", key="test:key", out="loaded_dets")

        with patch("mata.core.exporters.valkey_exporter.load_valkey", return_value=vision_result):
            result = node.run(ctx)

        assert "loaded_dets" in result
        assert isinstance(result["loaded_dets"], Detections)

    def test_result_to_artifact_vision(self):
        vision_result = _make_vision_result()
        artifact = ValkeyLoad._result_to_artifact(vision_result)
        assert isinstance(artifact, Detections)

    def test_result_to_artifact_classify(self):
        classify_result = ClassifyResult(predictions=[Classification(label=0, score=0.9, label_name="cat")])
        artifact = ValkeyLoad._result_to_artifact(classify_result)
        assert isinstance(artifact, Classifications)

    def test_result_to_artifact_depth(self):
        depth_arr = np.ones((32, 32), dtype=np.float32)
        depth_result = DepthResult(depth=depth_arr, normalized=depth_arr)
        artifact = ValkeyLoad._result_to_artifact(depth_result)
        assert isinstance(artifact, DepthMap)

    def test_result_to_artifact_unknown_raises(self):
        with pytest.raises(TypeError, match="Cannot convert"):
            ValkeyLoad._result_to_artifact("not_a_result")

    def test_missing_key_raises(self):
        ctx = _make_ctx()
        node = ValkeyLoad(url="valkey://localhost:6379", key="nonexistent:key")

        with patch(
            "mata.core.exporters.valkey_exporter.load_valkey",
            side_effect=KeyError("nonexistent:key"),
        ):
            with pytest.raises(KeyError):
                node.run(ctx)

    def test_graph_compilation_as_source(self):
        """ValkeyLoad (source node with no inputs) compiles in a Graph DAG."""
        graph = Graph().then(ValkeyLoad(url="valkey://localhost:6379", key="upstream:dets", out="dets"))

        compiled = graph.compile(providers={})
        assert isinstance(compiled, CompiledGraph)
        assert compiled.validation_result.valid
        assert len(compiled.nodes) == 1

    def test_graph_chain_load_then_filter(self):
        """ValkeyLoad → Filter chain compiles and validates correctly."""
        from mata.nodes.filter import Filter

        graph = (
            Graph()
            .then(ValkeyLoad(url="valkey://localhost:6379", key="upstream:dets", out="dets"))
            .then(Filter(src="dets", out="filtered", score_gt=0.5))
        )

        compiled = graph.compile(providers={})
        assert isinstance(compiled, CompiledGraph)
        assert compiled.validation_result.valid
        assert len(compiled.nodes) == 2

    def test_default_output_name(self):
        node = ValkeyLoad(url="valkey://localhost:6379", key="test:key")
        assert node.output_name == "loaded"

    def test_custom_output_name(self):
        node = ValkeyLoad(url="valkey://localhost:6379", key="test:key", out="my_artifact")
        assert node.output_name == "my_artifact"

    def test_result_type_forwarded_to_load_valkey(self):
        """result_type parameter is passed to load_valkey."""
        vision_result = _make_vision_result()
        ctx = _make_ctx()
        node = ValkeyLoad(
            url="valkey://localhost:6379",
            key="test:key",
            result_type="vision",
        )

        with patch("mata.core.exporters.valkey_exporter.load_valkey", return_value=vision_result) as mock_load:
            node.run(ctx)
            mock_load.assert_called_once_with(
                url="valkey://localhost:6379",
                key="test:key",
                result_type="vision",
            )

    def test_load_and_store_round_trip(self):
        """ValkeyLoad and ValkeyStore can be used in the same graph."""
        graph = (
            Graph()
            .then(ValkeyLoad(url="valkey://localhost:6379", key="upstream:dets", out="dets"))
            .then(ValkeyStore(src="dets", url="valkey://localhost:6379", key="downstream:{timestamp}"))
        )

        compiled = graph.compile(providers={})
        assert isinstance(compiled, CompiledGraph)
        assert compiled.validation_result.valid
        assert len(compiled.nodes) == 2
