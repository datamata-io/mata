"""Generate the tiny YOLO-layout ONNX fixture used by tests.

Run once from the repo root, then commit the generated .onnx and .json:

    .venv/bin/python tests/fixtures/make_tiny_yolo_onnx.py

Requires the `onnx` package (dev-only; CI only needs onnxruntime to run the test).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).parent
ANCHORS = 10
NUM_CLASSES = 2
INPUT = 64


def build() -> None:
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, INPUT, INPUT])
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4 + NUM_CLASSES, ANCHORS])

    base = np.zeros((1, 4 + NUM_CLASSES, ANCHORS), dtype=np.float32)
    # one anchor: cx, cy, w, h in input pixels; class 0 score 0.9, class 1 score 0.1
    base[0, 0, 0], base[0, 1, 0], base[0, 2, 0], base[0, 3, 0] = 32.0, 32.0, 16.0, 16.0
    base[0, 4, 0], base[0, 5, 0] = 0.9, 0.1

    nodes = [
        # keep the declared input connected so ONNX checkers accept the graph
        helper.make_node("ReduceMean", ["images"], ["mean"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Constant", [], ["zero"], value=numpy_helper.from_array(np.array([0.0], np.float32))),
        helper.make_node("Mul", ["mean", "zero"], ["delta"]),
        helper.make_node("Constant", [], ["base"], value=numpy_helper.from_array(base)),
        helper.make_node("Add", ["base", "delta"], ["output0"]),
    ]
    graph = helper.make_graph(nodes, "tiny_yolo", [images], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, HERE / "tiny_yolo.onnx")

    expected = {
        "input_size": INPUT,
        "detections": [{"bbox": [24.0, 24.0, 40.0, 40.0], "score": 0.9, "label": 0}],
    }
    (HERE / "tiny_yolo_expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print("wrote", HERE / "tiny_yolo.onnx", (HERE / "tiny_yolo.onnx").stat().st_size, "bytes")


def build_dynamic() -> None:
    """Dynamic-batch variant of the tiny YOLO fixture (for predict_batch tests)."""
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, INPUT, INPUT])
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", 4 + NUM_CLASSES, ANCHORS])

    base = np.zeros((1, 4 + NUM_CLASSES, ANCHORS), dtype=np.float32)
    base[0, 0, 0], base[0, 1, 0], base[0, 2, 0], base[0, 3, 0] = 32.0, 32.0, 16.0, 16.0
    base[0, 4, 0], base[0, 5, 0] = 0.9, 0.1

    nodes = [
        helper.make_node("ReduceMean", ["images"], ["mean"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Constant", [], ["axes"], value=numpy_helper.from_array(np.array([1, 2], np.int64))),
        helper.make_node("Unsqueeze", ["mean", "axes"], ["mean3"]),
        helper.make_node("Constant", [], ["zero"], value=numpy_helper.from_array(np.array([0.0], np.float32))),
        helper.make_node("Mul", ["mean3", "zero"], ["delta"]),
        helper.make_node("Constant", [], ["base"], value=numpy_helper.from_array(base)),
        helper.make_node("Add", ["base", "delta"], ["output0"]),
    ]
    graph = helper.make_graph(nodes, "tiny_yolo_batch", [images], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, HERE / "tiny_yolo_batch.onnx")
    print("wrote", HERE / "tiny_yolo_batch.onnx", (HERE / "tiny_yolo_batch.onnx").stat().st_size, "bytes")


if __name__ == "__main__":
    build()
    build_dynamic()
