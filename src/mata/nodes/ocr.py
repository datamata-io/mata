"""OCR node — optical character recognition task node.

Runs an OCR provider on either a whole image or a set of ROI crops and
returns an :class:`~mata.core.artifacts.ocr_text.OCRText` artifact.

When the input is an :class:`~mata.core.artifacts.rois.ROIs` artifact (e.g.
from ``ExtractROIs``), each crop is processed individually and the resulting
:class:`~mata.core.artifacts.ocr_text.OCRText` carries ``instance_ids``
aligned to the source ROI identifiers so a downstream ``Fuse`` node can
cross-reference detections with their extracted text.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from mata.core.artifacts.base import Artifact
from mata.core.artifacts.image import Image
from mata.core.artifacts.ocr_text import OCRText, TextBlock
from mata.core.artifacts.rois import ROIs
from mata.core.graph.node import Node

if TYPE_CHECKING:
    from mata.core.graph.context import ExecutionContext


class OCR(Node):
    """Optical character recognition node.

    Accepts either a whole :class:`~mata.core.artifacts.image.Image` or a
    :class:`~mata.core.artifacts.rois.ROIs` artifact produced by
    ``ExtractROIs``.  When ``ROIs`` are provided the OCR provider is called
    once per crop and the ``instance_ids`` of the output artifact align with
    those of the source ROIs so that ``Fuse`` can correlate results.

    Args:
        using: Name of the OCR provider registered in the execution context
            (e.g. ``"my_ocr"``, ``"tesseract"``).
        out: Key under which the output artifact is stored
            (default ``"ocr"``).
        src: Optional input artifact name override (useful when the incoming
            artifact is keyed under a custom name, e.g. ``"rois"``).
        name: Optional human-readable node name.
        **kwargs: Extra keyword arguments forwarded to the provider's
            ``recognize()`` call.

    Inputs:
        image (Image): Input image artifact  *or*
        rois (ROIs): ROI crops from ``ExtractROIs``.

    Outputs:
        ocr (OCRText): Recognized text artifact (key is ``out``).

    Example (standalone)::

        from mata.nodes import OCR

        node = OCR(using="easyocr_provider", out="text")
        result = node.run(ctx, image=img)
        print(result["text"].full_text)

    Example (ROI pipeline)::

        graph = (
            Detect(using="detector", out="dets")
            >> Filter(src="dets", label_in=["sign"], out="sign_dets")
            >> ExtractROIs(src_dets="sign_dets", out="rois")
            >> OCR(using="easyocr_provider", src="rois", out="ocr_result")
            >> Fuse(out="final", dets="sign_dets", ocr="ocr_result")
        )
    """

    inputs: dict[str, type[Artifact]] = {"image": Image}
    outputs: dict[str, type[Artifact]] = {"ocr": OCRText}

    def __init__(
        self,
        using: str,
        out: str = "ocr",
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
        # Without this, the scheduler always injects "image" (raw input) and
        # the ROIs crop arg is never provided, so OCR runs on the full image.
        if src:
            self.inputs = {src: Artifact}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, ctx: ExecutionContext, **inputs: Artifact) -> dict[str, Artifact]:
        """Execute OCR on the input image or ROI crops.

        Args:
            ctx: Execution context with providers and metrics.
            **inputs: Keyword-mapped input artifacts.  The node resolves the
                relevant artifact in priority order:

                1. The key named by ``self.src`` (if set).
                2. An artifact keyed ``"rois"`` (ROIs input).
                3. An artifact keyed ``"image"`` (Image input).
                4. The first available artifact (fallback).

        Returns:
            Dict with a single key (``self.output_name``) mapping to an
            :class:`~mata.core.artifacts.ocr_text.OCRText` artifact.

        Raises:
            ValueError: If the input artifact is neither an ``Image`` nor
                ``ROIs`` instance.
            KeyError: If the OCR provider is not found in the context.
        """
        recognizer = ctx.get_provider("ocr", self.provider_name)

        # Auto-wrap raw adapters that expose predict() but not recognize().
        # This lets users pass mata.load("ocr", …) adapters directly as
        # providers without having to call wrap_ocr() manually.
        if not hasattr(recognizer, "recognize"):
            from mata.adapters.wrappers.ocr_wrapper import OCRWrapper

            recognizer = OCRWrapper(recognizer)

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
            raise ValueError(f"OCR node '{self.name}' received no inputs. " "Provide an Image or ROIs artifact.")

        # Dispatch ----------------------------------------------------------
        start = time.time()
        if isinstance(artifact, ROIs):
            ocr_result = self._run_on_rois(recognizer, artifact)
        elif isinstance(artifact, Image):
            ocr_result = self._run_on_image(recognizer, artifact)
        else:
            raise ValueError(
                f"OCR node '{self.name}' expected an Image or ROIs input, " f"got {type(artifact).__name__}."
            )
        latency_ms = (time.time() - start) * 1000

        # Metrics -----------------------------------------------------------
        ctx.record_metric(self.name, "latency_ms", latency_ms)
        ctx.record_metric(self.name, "num_text_blocks", len(ocr_result.text_blocks))

        return {self.output_name: ocr_result}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_on_image(self, recognizer: Any, image: Image) -> OCRText:
        """Run OCR on a single whole image."""
        raw = recognizer.recognize(image, **self.kwargs)
        if isinstance(raw, OCRText):
            return raw
        # raw is assumed to be an OCRResult from the adapter
        return OCRText.from_ocr_result(raw, instance_ids=())

    def _run_on_rois(self, recognizer: Any, rois: ROIs) -> OCRText:
        """Run OCR on each ROI crop and aggregate results.

        Each recognized text block is tagged with the ``instance_id`` of its
        source ROI crop so that downstream ``Fuse`` nodes can correlate OCR
        output with the original detections.
        """
        all_blocks: list[TextBlock] = []
        all_instance_ids: list[str] = []

        for crop, inst_id in zip(rois.roi_images, rois.instance_ids):
            # Wrap crop as an Image artifact ---------------------------------
            roi_image = self._wrap_crop(crop)

            raw = recognizer.recognize(roi_image, **self.kwargs)
            if isinstance(raw, OCRText):
                crop_blocks = raw.text_blocks
            else:
                crop_ocr_text = OCRText.from_ocr_result(raw, instance_ids=())
                crop_blocks = crop_ocr_text.text_blocks

            for block in crop_blocks:
                all_blocks.append(block)
                all_instance_ids.append(inst_id)

        full_text = "\n".join(b.text for b in all_blocks)
        return OCRText(
            text_blocks=tuple(all_blocks),
            full_text=full_text,
            instance_ids=tuple(all_instance_ids),
            meta={
                "mode": "rois",
                "num_rois": len(rois.roi_images),
                **self.kwargs,
            },
        )

    @staticmethod
    def _wrap_crop(crop: Any) -> Image:
        """Wrap a PIL Image or numpy array crop as an Image artifact."""
        try:
            from PIL import Image as PILImage

            if isinstance(crop, PILImage.Image):
                return Image.from_pil(crop)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(crop, np.ndarray):
                return Image.from_numpy(crop)
        except ImportError:
            pass

        # Last resort: assume it has a PIL-compatible interface
        return Image.from_pil(crop)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        extra = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        parts = f"OCR(using='{self.provider_name}', out='{self.output_name}'"
        if self.src:
            parts += f", src='{self.src}'"
        if extra:
            parts += f", {extra}"
        return parts + ")"
