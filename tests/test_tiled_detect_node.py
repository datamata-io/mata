"""Tests for the TiledDetect graph node."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from mata.core.artifacts.detections import Detections
from mata.core.artifacts.image import Image
from mata.core.graph import Graph
from mata.core.graph.context import ExecutionContext
from mata.core.types import Instance, VisionResult
from mata.nodes.tiled_detect import TiledDetect


def _make_ctx(provider) -> ExecutionContext:
    return ExecutionContext(providers={"detect": {"fake": provider}}, device="cpu")


def _provider(bbox=(5, 5, 25, 25), score=0.9, label=0):
    provider = MagicMock()
    provider.predict.return_value = VisionResult(
        instances=[Instance(bbox=bbox, score=score, label=label, label_name="obj")]
    )
    return provider


def _image(width=100, height=100) -> Image:
    return Image(data=np.zeros((height, width, 3), dtype=np.uint8), width=width, height=height)


def test_tiles_are_offset_into_source_coordinates():
    provider = _provider()
    node = TiledDetect(using="fake", slice_size=60, overlap=0.0, include_full_image=False)
    result = node.run(_make_ctx(provider), image=_image())
    dets = result["dets"]

    assert provider.predict.call_count == 4  # 2 columns x 2 rows
    assert len(dets.instances) == 4
    assert {tuple(int(v) for v in inst.bbox) for inst in dets.instances} == {
        (5, 5, 25, 25),
        (45, 5, 65, 25),
        (5, 45, 25, 65),
        (45, 45, 65, 65),
    }


def test_full_image_pass_adds_one_call_and_merges_the_duplicate():
    provider = _provider()
    node = TiledDetect(using="fake", slice_size=60, overlap=0.0, include_full_image=True)
    result = node.run(_make_ctx(provider), image=_image())
    dets = result["dets"]

    assert provider.predict.call_count == 5
    assert len(dets.instances) == 4  # full-image box duplicates the top-left tile
    assert dets.meta["tiles"] == 5
    assert dets.meta["raw_detections"] == 5


def test_empty_provider_result_yields_empty_detections():
    provider = MagicMock()
    provider.predict.return_value = VisionResult(instances=[])
    node = TiledDetect(using="fake", slice_size=60, overlap=0.0)
    result = node.run(_make_ctx(provider), image=_image())
    assert result["dets"].instances == []


def test_accepts_detections_artifact_from_provider():
    provider = MagicMock()
    provider.predict.return_value = Detections(instances=[Instance(bbox=(1, 2, 11, 12), score=0.7, label=0)])
    node = TiledDetect(using="fake", slice_size=60, overlap=0.0, include_full_image=False)
    result = node.run(_make_ctx(provider), image=_image())
    assert len(result["dets"].instances) == 4


def test_custom_output_name():
    provider = _provider()
    node = TiledDetect(using="fake", out="tiled", slice_size=60, overlap=0.0)
    assert "tiled" in node.run(_make_ctx(provider), image=_image())


def test_small_image_runs_once():
    provider = _provider()
    node = TiledDetect(using="fake", slice_size=640, overlap=0.2, include_full_image=True)
    result = node.run(_make_ctx(provider), image=_image(50, 40))
    assert provider.predict.call_count == 1
    assert len(result["dets"].instances) == 1


def test_nmm_mode_unions_seam_duplicates():
    provider = MagicMock()
    provider.predict.return_value = VisionResult(instances=[Instance(bbox=(0, 0, 60, 60), score=0.9, label=0)])
    node = TiledDetect(
        using="fake",
        slice_size=(60, 60),
        overlap=0.5,
        include_full_image=False,
        iou_threshold=0.5,
        merge="nmm",
    )
    result = node.run(_make_ctx(provider), image=_image(70, 60))
    dets = result["dets"].instances
    assert len(dets) == 1
    assert tuple(dets[0].bbox) == (0, 0, 70, 60)


def test_tile_local_masks_and_keypoints_are_dropped():
    provider = MagicMock()
    provider.predict.return_value = VisionResult(
        instances=[
            Instance(
                bbox=(1, 2, 11, 12),
                mask=np.ones((10, 10), dtype=bool),
                keypoints=np.zeros((1, 3), dtype=np.float32),
                score=0.7,
                label=0,
            )
        ]
    )
    node = TiledDetect(using="fake", slice_size=640, overlap=0.0, include_full_image=False)
    dets = node.run(_make_ctx(provider), image=_image())["dets"].instances
    assert len(dets) == 1
    assert dets[0].mask is None
    assert dets[0].keypoints is None


def test_tiled_detect_is_exported_and_runs_in_a_graph():
    from mata.nodes import TiledDetect as ExportedTiledDetect

    assert ExportedTiledDetect is TiledDetect

    provider = _provider()
    graph = Graph("tiled").then(
        TiledDetect(using="detector", out="dets", slice_size=60, overlap=0.0, include_full_image=False)
    )
    result = graph.run(image=np.zeros((100, 100, 3), dtype=np.uint8), providers={"detector": provider})
    dets = result.channels["dets"]
    assert len(dets.instances) == 4
