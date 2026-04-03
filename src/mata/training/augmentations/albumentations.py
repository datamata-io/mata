"""Optional albumentations integration for MATA augmentation pipelines.

Wraps a user-provided ``albumentations.Compose`` pipeline and handles the
format conversions required to interoperate with MATA's target dict convention
(xyxy absolute pixel coordinates for boxes, ``[N, H, W]`` binary tensors for
masks).

Usage::

    import albumentations as A
    from mata.training.augmentations.albumentations import AlbumentationsWrapper

    transform = A.Compose(
        [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )
    aug = AlbumentationsWrapper(transform)
    image_t, target_t = aug(pil_image, target)
"""

from __future__ import annotations

from typing import Any


class AlbumentationsWrapper:
    """Wraps a user-provided ``albumentations.Compose`` pipeline.

    Handles the following format conversions automatically:

    * **Image**: PIL Image → numpy ``uint8 (H, W, 3)`` → back to
      ``torch.Tensor (3, H, W)`` float32 in ``[0, 1]``.
    * **Bounding boxes**: MATA xyxy absolute pixels (same as albumentations
      ``pascal_voc`` format) — passed as a list of ``[x_min, y_min, x_max, y_max]``
      and reconstructed back to a ``torch.Tensor [N, 4]``.
    * **Labels**: aligned with boxes via albumentations ``label_fields``
      mechanism; reconstructed as ``torch.Tensor [N]`` long.
    * **Masks**: ``torch.Tensor [N, H, W]`` → list of ``(H, W)`` numpy arrays
      → back to ``torch.Tensor [N, H, W]``.

    .. note::
        For detection / segmentation tasks the user-provided ``A.Compose``
        **must** include ``bbox_params=A.BboxParams(format="pascal_voc",
        label_fields=["class_labels"])`` otherwise bounding boxes will not be
        transformed.

    Args:
        transform: A pre-built ``albumentations.Compose`` pipeline.

    Raises:
        ImportError: If ``albumentations`` is not installed.
    """

    def __init__(self, transform: Any) -> None:
        try:
            import albumentations  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "albumentations is required for AlbumentationsWrapper. " "Install it with: pip install albumentations"
            ) from exc

        self._transform = transform

    def __call__(self, image: Any, target: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Apply the albumentations pipeline to image and target.

        Args:
            image: PIL Image or ``torch.Tensor`` of shape ``(C, H, W)``.
            target: Dict with optional ``"boxes"`` (Tensor ``[N, 4]`` xyxy),
                ``"labels"`` (Tensor ``[N]``), and ``"masks"`` (Tensor
                ``[N, H, W]`` binary).

        Returns:
            ``(image_tensor, target_dict)`` where the image is a
            ``torch.Tensor (3, H, W)`` float32 in ``[0, 1]``, and boxes /
            labels / masks in the target dict are updated accordingly.
        """
        import numpy as np
        import torch

        # ── Convert image to numpy uint8 (H, W, 3) ──────────────────────────
        if isinstance(image, torch.Tensor):
            # Tensor (C, H, W) in [0, 1] or [0, 255] → numpy uint8
            img_np = image.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        else:
            # Assume PIL Image
            img_np = np.array(image, dtype=np.uint8)

        # ── Build albumentations call kwargs ─────────────────────────────────
        call_kwargs: dict[str, Any] = {"image": img_np}

        raw_boxes = target.get("boxes")
        raw_labels = target.get("labels")
        raw_masks = target.get("masks")

        has_boxes = raw_boxes is not None and len(raw_boxes) > 0
        has_masks = raw_masks is not None and len(raw_masks) > 0

        if has_boxes:
            boxes_list = raw_boxes.tolist() if isinstance(raw_boxes, torch.Tensor) else list(raw_boxes)
            call_kwargs["bboxes"] = boxes_list

            if raw_labels is not None:
                labels_list = raw_labels.tolist() if isinstance(raw_labels, torch.Tensor) else list(raw_labels)
            else:
                labels_list = [0] * len(boxes_list)
            call_kwargs["class_labels"] = labels_list
        else:
            call_kwargs["bboxes"] = []
            call_kwargs["class_labels"] = []

        if has_masks:
            # albumentations expects a list of (H, W) uint8 arrays
            masks_np = raw_masks.cpu().numpy() if isinstance(raw_masks, torch.Tensor) else np.array(raw_masks)
            call_kwargs["masks"] = [masks_np[i] for i in range(masks_np.shape[0])]

        # ── Apply the albumentations pipeline ────────────────────────────────
        result = self._transform(**call_kwargs)

        # ── Convert outputs back to torch ────────────────────────────────────
        out_img = torch.from_numpy(result["image"]).permute(2, 0, 1).float() / 255.0

        new_target = dict(target)

        if has_boxes or "bboxes" in result:
            out_bboxes = result.get("bboxes", [])
            out_labels = result.get("class_labels", [])
            new_target["boxes"] = (
                torch.tensor(out_bboxes, dtype=torch.float32)
                if out_bboxes
                else torch.zeros((0, 4), dtype=torch.float32)
            )
            new_target["labels"] = (
                torch.tensor(out_labels, dtype=torch.long) if out_labels else torch.zeros(0, dtype=torch.long)
            )

        if has_masks and "masks" in result:
            out_masks = result["masks"]
            if out_masks:
                new_target["masks"] = torch.from_numpy(np.stack(out_masks, axis=0))
            else:
                h, w = img_np.shape[:2]
                new_target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

        return out_img, new_target
