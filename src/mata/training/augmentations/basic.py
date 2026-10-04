"""Built-in augmentation pipelines using torchvision.transforms.v2.

Detection and segmentation augmentations are coordinate-aware: bounding boxes
and masks are transformed alongside the image using tv_tensors wrappers,
ensuring spatial consistency across all transforms.
"""

from __future__ import annotations

from typing import Any

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BasicDetectionAugmentation:
    """Resize, horizontal flip, color jitter, and normalize for detection.

    Detection-aware: bounding boxes are transformed alongside the image using
    ``torchvision.transforms.v2`` and ``tv_tensors.BoundingBoxes``.

    Args:
        size: Target size. In train mode the shorter edge is resized to this
            value (aspect ratio preserved); in val mode the image is resized to
            ``(size, size)``.
        flip_prob: Probability of random horizontal flip (train only).
        jitter_brightness: ColorJitter brightness factor (train only).
        jitter_contrast: ColorJitter contrast factor (train only).
        jitter_saturation: ColorJitter saturation factor (train only).
        jitter_hue: ColorJitter hue factor (train only).
        mean: Normalization mean. Defaults to ImageNet mean.
        std: Normalization std. Defaults to ImageNet std.
        train: If ``True`` use full training augmentations; if ``False`` use
            val-mode transforms (resize + normalize only).
    """

    def __init__(
        self,
        size: int = 800,
        flip_prob: float = 0.5,
        jitter_brightness: float = 0.2,
        jitter_contrast: float = 0.2,
        jitter_saturation: float = 0.2,
        jitter_hue: float = 0.1,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        train: bool = True,
    ) -> None:
        import torch
        from torchvision.transforms import v2

        self.size = size
        self.train = train
        self._mean = mean if mean is not None else IMAGENET_MEAN
        self._std = std if std is not None else IMAGENET_STD

        if train:
            self._transform = v2.Compose(
                [
                    v2.Resize(size),
                    v2.RandomHorizontalFlip(p=flip_prob),
                    v2.ColorJitter(
                        brightness=jitter_brightness,
                        contrast=jitter_contrast,
                        saturation=jitter_saturation,
                        hue=jitter_hue,
                    ),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=self._mean, std=self._std),
                ]
            )
        else:
            self._transform = v2.Compose(
                [
                    v2.Resize((size, size)),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=self._mean, std=self._std),
                ]
            )

    def __call__(self, image: Any, target: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Apply augmentations to image and target dict.

        Args:
            image: PIL Image or ``torch.Tensor`` of shape ``(C, H, W)``.
            target: Dict with optional ``"boxes"`` key (Tensor ``[N, 4]`` xyxy)
                and ``"labels"`` key (Tensor ``[N]``).

        Returns:
            ``(image_tensor, target_dict)`` with transformed image and boxes.
        """
        import torch
        from torchvision import tv_tensors

        # Resolve image spatial dimensions
        if hasattr(image, "size") and not isinstance(image, torch.Tensor):
            # PIL Image: .size returns (width, height)
            w, h = image.size
        else:
            h, w = image.shape[-2], image.shape[-1]

        # Wrap boxes as BoundingBoxes tv_tensor (empty if absent)
        raw_boxes = target.get("boxes")
        if raw_boxes is not None and len(raw_boxes) > 0:
            tv_boxes = tv_tensors.BoundingBoxes(
                raw_boxes.to(dtype=torch.float32),
                format="XYXY",
                canvas_size=(h, w),
            )
        else:
            tv_boxes = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format="XYXY",
                canvas_size=(h, w),
            )

        image_t, tv_boxes_t = self._transform(image, tv_boxes)

        new_target = dict(target)
        new_target["boxes"] = tv_boxes_t.as_subclass(torch.Tensor)
        return image_t, new_target


class BasicClassificationAugmentation:
    """Resize, crop, horizontal flip, color jitter, and normalize for classification.

    Training mode uses ``RandomResizedCrop`` + ``RandomHorizontalFlip`` +
    ``ColorJitter``.  Validation mode uses ``Resize`` + ``CenterCrop``.

    Args:
        size: Output crop size (default: 224).
        flip_prob: Probability of random horizontal flip (train only).
        jitter_brightness: ColorJitter brightness factor (train only).
        jitter_contrast: ColorJitter contrast factor (train only).
        jitter_saturation: ColorJitter saturation factor (train only).
        jitter_hue: ColorJitter hue factor (train only).
        mean: Normalization mean. Defaults to ImageNet mean.
        std: Normalization std. Defaults to ImageNet std.
        train: If ``True`` use full training augmentations; if ``False`` use
            val-mode transforms.
    """

    def __init__(
        self,
        size: int = 224,
        flip_prob: float = 0.5,
        jitter_brightness: float = 0.2,
        jitter_contrast: float = 0.2,
        jitter_saturation: float = 0.2,
        jitter_hue: float = 0.1,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        train: bool = True,
    ) -> None:
        import torch
        from torchvision.transforms import v2

        self.size = size
        self.train = train
        self._mean = mean if mean is not None else IMAGENET_MEAN
        self._std = std if std is not None else IMAGENET_STD

        # Standard ImageNet val resize: larger edge → 256/224 * size
        val_resize = int(size * 256 / 224)

        if train:
            self._transform = v2.Compose(
                [
                    v2.RandomResizedCrop(size),
                    v2.RandomHorizontalFlip(p=flip_prob),
                    v2.ColorJitter(
                        brightness=jitter_brightness,
                        contrast=jitter_contrast,
                        saturation=jitter_saturation,
                        hue=jitter_hue,
                    ),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=self._mean, std=self._std),
                ]
            )
        else:
            self._transform = v2.Compose(
                [
                    v2.Resize(val_resize),
                    v2.CenterCrop(size),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=self._mean, std=self._std),
                ]
            )

    def __call__(self, image: Any, target: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Apply augmentations to image and classification target.

        Args:
            image: PIL Image or ``torch.Tensor`` of shape ``(C, H, W)``.
            target: Dict with ``"label"`` key (int).

        Returns:
            ``(image_tensor[3, size, size], target_dict)`` unchanged target.
        """
        image_t = self._transform(image)
        return image_t, target


class BasicSegmentationAugmentation:
    """Resize, horizontal flip, color jitter, and normalize for segmentation.

    Segmentation-aware: both bounding boxes and instance masks are transformed
    alongside the image using ``tv_tensors.BoundingBoxes`` and ``tv_tensors.Mask``.

    Args:
        size: Target size. In train mode the shorter edge is resized to this
            value (aspect ratio preserved); in val mode the image is resized to
            ``(size, size)``.
        flip_prob: Probability of random horizontal flip (train only).
        jitter_brightness: ColorJitter brightness factor (train only).
        jitter_contrast: ColorJitter contrast factor (train only).
        jitter_saturation: ColorJitter saturation factor (train only).
        jitter_hue: ColorJitter hue factor (train only).
        mean: Normalization mean. Defaults to ImageNet mean.
        std: Normalization std. Defaults to ImageNet std.
        train: If ``True`` use full training augmentations; if ``False`` use
            val-mode transforms (resize + normalize only).
    """

    def __init__(
        self,
        size: int = 800,
        flip_prob: float = 0.5,
        jitter_brightness: float = 0.2,
        jitter_contrast: float = 0.2,
        jitter_saturation: float = 0.2,
        jitter_hue: float = 0.1,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        train: bool = True,
    ) -> None:
        import torch
        from torchvision.transforms import v2

        self.size = size
        self.train = train
        self._mean = mean if mean is not None else IMAGENET_MEAN
        self._std = std if std is not None else IMAGENET_STD

        if train:
            self._transform = v2.Compose(
                [
                    v2.Resize(size),
                    v2.RandomHorizontalFlip(p=flip_prob),
                    v2.ColorJitter(
                        brightness=jitter_brightness,
                        contrast=jitter_contrast,
                        saturation=jitter_saturation,
                        hue=jitter_hue,
                    ),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=self._mean, std=self._std),
                ]
            )
        else:
            self._transform = v2.Compose(
                [
                    v2.Resize((size, size)),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=self._mean, std=self._std),
                ]
            )

    def __call__(self, image: Any, target: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Apply augmentations to image, boxes, and masks.

        Args:
            image: PIL Image or ``torch.Tensor`` of shape ``(C, H, W)``.
            target: Dict with optional ``"boxes"`` (Tensor ``[N, 4]`` xyxy),
                ``"labels"`` (Tensor ``[N]``), and ``"masks"`` (Tensor
                ``[N, H, W]`` binary).

        Returns:
            ``(image_tensor, target_dict)`` with consistently transformed boxes
            and masks.
        """
        import torch
        from torchvision import tv_tensors

        # Resolve image spatial dimensions
        if hasattr(image, "size") and not isinstance(image, torch.Tensor):
            w, h = image.size  # PIL: (width, height)
        else:
            h, w = image.shape[-2], image.shape[-1]

        # Wrap boxes
        raw_boxes = target.get("boxes")
        if raw_boxes is not None and len(raw_boxes) > 0:
            tv_boxes = tv_tensors.BoundingBoxes(
                raw_boxes.to(dtype=torch.float32),
                format="XYXY",
                canvas_size=(h, w),
            )
        else:
            tv_boxes = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format="XYXY",
                canvas_size=(h, w),
            )

        # Wrap masks — shape [N, H, W]
        raw_masks = target.get("masks")
        has_masks = raw_masks is not None and len(raw_masks) > 0
        if has_masks:
            tv_masks = tv_tensors.Mask(raw_masks)
        else:
            tv_masks = tv_tensors.Mask(torch.zeros((0, h, w), dtype=torch.uint8))

        image_t, tv_boxes_t, tv_masks_t = self._transform(image, tv_boxes, tv_masks)

        new_target = dict(target)
        new_target["boxes"] = tv_boxes_t.as_subclass(torch.Tensor)
        new_target["masks"] = tv_masks_t.as_subclass(torch.Tensor)
        return image_t, new_target
