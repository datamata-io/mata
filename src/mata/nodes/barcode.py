"""Barcode node — barcode/QR detection and decoding task node.

Runs a barcode provider on either a whole image or a set of ROI crops
and returns a :class:`~mata.core.artifacts.barcode_data.BarcodeData` artifact.

When the input is a :class:`~mata.core.artifacts.rois.ROIs` artifact (e.g.
from ``ExtractROIs``), each crop is processed individually and the resulting
:class:`~mata.core.artifacts.barcode_data.BarcodeData` carries ``instance_ids``
aligned to the source ROI identifiers so a downstream ``Fuse`` node can
cross-reference detections with their decoded barcodes.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from mata.core.artifacts.barcode_data import BarcodeData, BarcodeEntry
from mata.core.artifacts.base import Artifact
from mata.core.artifacts.image import Image
from mata.core.artifacts.rois import ROIs
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class Barcode(Node):
    """Barcode/QR code detection and decoding node.

    Accepts either a whole :class:`~mata.core.artifacts.image.Image` or a
    :class:`~mata.core.artifacts.rois.ROIs` artifact produced by
    ``ExtractROIs``. When ``ROIs`` are provided, each crop is processed
    individually and the ``instance_ids`` of the output artifact align with
    those of the source ROIs so that ``Fuse`` can correlate results.

    Args:
        using: Name of the barcode provider registered in the execution context
            (e.g. ``"pyzbar"``).
        out: Key under which the output artifact is stored
            (default ``"barcodes"``).
        src: Optional input artifact name override (useful when the incoming
            artifact is keyed under a custom name, e.g. ``"rois"``).
        name: Optional human-readable node name.
        **kwargs: Extra keyword arguments forwarded to the provider's
            ``predict()`` call.

    Inputs:
        image (Image): Input image artifact  *or*
        rois (ROIs): ROI crops from ``ExtractROIs``.

    Outputs:
        barcodes (BarcodeData): Decoded barcode artifact (key is ``out``).

    Example (standalone)::

        from mata.nodes import Barcode

        node = Barcode(using="pyzbar", out="codes")
        result = node.run(ctx, image=img)
        for entry in result["codes"].entries:
            print(entry.data, entry.type)

    Example (ROI pipeline)::

        graph = (
            Detect(using="detector", out="dets")
            >> Filter(src="dets", label_in=["barcode", "qr_code"], out="bc_dets")
            >> ExtractROIs(src_dets="bc_dets", out="rois")
            >> Barcode(using="pyzbar", src="rois", out="codes")
        )
    """

    inputs: dict[str, type[Artifact]] = {"image": Image}
    outputs: dict[str, type[Artifact]] = {"barcodes": BarcodeData}

    def __init__(
        self,
        using: str,
        out: str = "barcodes",
        src: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name)
        self.provider_name = using
        self.output_name = out
        self.src = src
        self.kwargs = kwargs
        # Teach the scheduler which artifact to wire based on src.
        if src:
            self.inputs = {src: Artifact}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, ctx: ExecutionContext, **inputs: Artifact) -> dict[str, Artifact]:
        """Execute barcode reading on the input image or ROI crops.

        Args:
            ctx: Execution context with providers and metrics.
            **inputs: Keyword-mapped input artifacts. The node resolves the
                relevant artifact in priority order:

                1. The key named by ``self.src`` (if set).
                2. An artifact keyed ``"rois"`` (ROIs input).
                3. An artifact keyed ``"image"`` (Image input).
                4. The first available artifact (fallback).

        Returns:
            Dict with a single key (``self.output_name``) mapping to a
            :class:`~mata.core.artifacts.barcode_data.BarcodeData` artifact.

        Raises:
            ValueError: If the input artifact is neither an ``Image`` nor
                ``ROIs`` instance, or if no inputs are provided.
            KeyError: If the barcode provider is not found in the context.
        """
        provider = ctx.get_provider("barcode", self.provider_name)

        # Resolve input artifact -------------------------------------------
        artifact: Artifact | None = None
        if self.src and self.src in inputs:
            artifact = inputs[self.src]
        elif "rois" in inputs:
            artifact = inputs["rois"]
        elif "image" in inputs:
            artifact = inputs["image"]
        elif inputs:
            artifact = next(iter(inputs.values()))

        if artifact is None:
            raise ValueError(f"Barcode node '{self.name}' received no inputs. " "Provide an Image or ROIs artifact.")

        # Dispatch ----------------------------------------------------------
        start = time.time()
        if isinstance(artifact, ROIs):
            result = self._run_on_rois(provider, artifact)
        elif isinstance(artifact, Image):
            result = self._run_on_image(provider, artifact)
        else:
            raise ValueError(
                f"Barcode node '{self.name}' expected an Image or ROIs input, " f"got {type(artifact).__name__}."
            )
        latency_ms = (time.time() - start) * 1000

        # Metrics -----------------------------------------------------------
        ctx.record_metric(self.name, "latency_ms", latency_ms)
        ctx.record_metric(self.name, "num_barcodes", len(result.entries))

        return {self.output_name: result}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_on_image(self, provider: Any, image: Image) -> BarcodeData:
        """Run barcode decoding on a single whole image."""
        raw = provider.predict(image.to_pil(), **self.kwargs)
        if isinstance(raw, BarcodeData):
            return raw
        return BarcodeData.from_barcode_result(raw, instance_ids=())

    def _run_on_rois(self, provider: Any, rois: ROIs) -> BarcodeData:
        """Run barcode decoding on each ROI crop and aggregate results.

        Each decoded barcode entry is tagged with the ``instance_id`` of its
        source ROI crop so that downstream ``Fuse`` nodes can correlate barcode
        output with the original detections.
        """
        all_entries: list[BarcodeEntry] = []
        all_instance_ids: list[str] = []

        for crop, inst_id in zip(rois.roi_images, rois.instance_ids):
            raw = provider.predict(crop, **self.kwargs)
            if isinstance(raw, BarcodeData):
                entries = raw.entries
            else:
                entries = BarcodeData.from_barcode_result(raw, instance_ids=()).entries

            for entry in entries:
                all_entries.append(entry)
                all_instance_ids.append(inst_id)

        return BarcodeData(
            entries=tuple(all_entries),
            instance_ids=tuple(all_instance_ids),
        )
