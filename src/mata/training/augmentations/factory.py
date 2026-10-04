"""Factory for building augmentation pipelines from task name and config dicts.

Usage::

    from mata.training.augmentations.factory import AugmentationFactory

    # Default training augmentations
    aug = AugmentationFactory.create("detect")

    # Default val augmentations
    aug = AugmentationFactory.create("classify", train=False)

    # Custom config (size override)
    aug = AugmentationFactory.create("segment", config={"size": 640})

    # Use albumentations pipeline
    import albumentations as A
    aug = AugmentationFactory.create(
        "detect",
        config={
            "type": "albumentations",
            "transform": A.Compose(
                [A.HorizontalFlip(p=0.5)],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
            ),
        },
    )
"""

from __future__ import annotations

from typing import Any


class AugmentationFactory:
    """Build augmentation pipelines from task name and optional config dicts.

    All factory methods are static — no instantiation required.
    """

    @staticmethod
    def create(
        task: str,
        config: dict[str, Any] | None = None,
        train: bool = True,
    ) -> Any:
        """Build an augmentation pipeline.

        Args:
            task: Vision task — ``"detect"``, ``"classify"``, or ``"segment"``.
            config: Optional configuration dict.  When ``config["type"]`` is
                ``"albumentations"`` the factory returns an
                :class:`~mata.training.augmentations.albumentations.AlbumentationsWrapper`
                around ``config["transform"]`` (a pre-built
                ``albumentations.Compose`` instance).  Otherwise
                ``"size"`` and per-augmentation keyword arguments are forwarded
                to the corresponding ``BasicXxxAugmentation`` class.
            train: If ``True`` (default) return training augmentations; if
                ``False`` return val-mode transforms (resize + normalize only).

        Returns:
            A callable ``(image, target) → (image_tensor, target_dict)``.

        Raises:
            ValueError: If ``task`` is not one of the supported values.
            ImportError: If ``config["type"] == "albumentations"`` but
                ``albumentations`` is not installed.
        """
        _VALID_TASKS = {"detect", "classify", "segment"}
        if task not in _VALID_TASKS:
            raise ValueError(f"Unknown task {task!r}. Expected one of: " + ", ".join(sorted(_VALID_TASKS)))

        # ── albumentations route ─────────────────────────────────────────────
        if config is not None and config.get("type") == "albumentations":
            from mata.training.augmentations.albumentations import AlbumentationsWrapper

            transform = config.get("transform")
            if transform is None:
                raise ValueError(
                    "config['transform'] must be a pre-built albumentations.Compose "
                    "instance when config['type'] == 'albumentations'."
                )
            return AlbumentationsWrapper(transform)

        # ── basic augmentation route ─────────────────────────────────────────
        # Extract common kwargs (size + per-channel jitter params)
        kwargs: dict[str, Any] = {}
        if config:
            for key in (
                "size",
                "flip_prob",
                "jitter_brightness",
                "jitter_contrast",
                "jitter_saturation",
                "jitter_hue",
                "mean",
                "std",
            ):
                if key in config:
                    kwargs[key] = config[key]

        kwargs["train"] = train

        if task == "detect":
            from mata.training.augmentations.basic import BasicDetectionAugmentation

            return BasicDetectionAugmentation(**kwargs)

        if task == "classify":
            from mata.training.augmentations.basic import BasicClassificationAugmentation

            return BasicClassificationAugmentation(**kwargs)

        # task == "segment"
        from mata.training.augmentations.basic import BasicSegmentationAugmentation

        return BasicSegmentationAugmentation(**kwargs)
