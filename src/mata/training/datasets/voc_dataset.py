"""Pascal VOC format dataset for detection training."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from mata.core.exceptions import TrainingError
from mata.training.datasets.base import TrainingDataset, _ensure_torch


class VOCDetectionDataset(TrainingDataset):
    """Pascal VOC format dataset for object detection training.

    Expected directory structure::

        root/
        ├── JPEGImages/        # source images (*.jpg / *.jpeg / *.png)
        ├── Annotations/       # per-image XML annotation files
        └── ImageSets/
            └── Main/
                ├── train.txt
                ├── val.txt
                ├── trainval.txt
                └── test.txt

    Args:
        root: Path to the VOC dataset root directory.
        image_set: Dataset split — ``"train"``, ``"val"``, ``"trainval"``, or
            ``"test"``.  Image IDs are read from
            ``ImageSets/Main/{image_set}.txt``.  If that file does not exist,
            all ``.xml`` files in ``Annotations/`` are used instead.
        transforms: Optional callable ``(image, target) -> (image, target)``
            applied after loading each sample.
        skip_difficult: When ``True`` (the default), objects marked with
            ``<difficult>1</difficult>`` in the XML are excluded from the
            returned targets.

    Example::

        dataset = VOCDetectionDataset("/data/VOC2012")
        image, target = dataset[0]
        # target = {"boxes": Tensor[N,4], "labels": Tensor[N], "image_id": 0}
        print(dataset.class_names)  # {0: "aeroplane", 1: "bicycle", ...}
    """

    def __init__(
        self,
        root: str | Path,
        image_set: str = "trainval",
        transforms: Any | None = None,
        skip_difficult: bool = True,
    ) -> None:
        super().__init__(task="detect", transforms=transforms)
        self.root = Path(root)
        self.image_set = image_set
        self.skip_difficult = skip_difficult
        self._annotations_dir = self.root / "Annotations"
        self._images_dir = self.root / "JPEGImages"

        if not self._annotations_dir.is_dir():
            raise TrainingError(
                f"VOC Annotations directory not found: {self._annotations_dir}. "
                "Expected structure: root/Annotations/*.xml"
            )
        if not self._images_dir.is_dir():
            raise TrainingError(
                f"VOC JPEGImages directory not found: {self._images_dir}. " "Expected structure: root/JPEGImages/"
            )

        self._image_ids: list[str] = self._load_image_ids(image_set)
        self._class_names: dict[int, str] = self._discover_class_names()
        self._class_to_idx: dict[str, int] = {v: k for k, v in self._class_names.items()}

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        torch = _ensure_torch()
        image_id = self._image_ids[index]

        img_path = self._find_image_path(image_id)
        image = self._load_image(img_path)

        ann_path = self._annotations_dir / f"{image_id}.xml"
        boxes, labels = self._parse_annotation(ann_path)

        target: dict[str, Any] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": index,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_image_ids(self, image_set: str) -> list[str]:
        """Return image IDs for the requested split.

        Reads ``ImageSets/Main/{image_set}.txt`` when present; each line may be
        ``"id"`` or ``"id confidence"`` (the per-class list variant). Falls back
        to scanning all ``*.xml`` files in ``Annotations/`` if the file is absent.
        """
        image_sets_file = self.root / "ImageSets" / "Main" / f"{image_set}.txt"
        if image_sets_file.is_file():
            ids: list[str] = []
            for line in image_sets_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if parts:
                    ids.append(parts[0])
            if not ids:
                raise TrainingError(f"ImageSets file is empty: {image_sets_file}")
            return ids

        # Fallback: derive IDs from annotation filenames
        xml_files = sorted(self._annotations_dir.glob("*.xml"))
        if not xml_files:
            raise TrainingError(f"No XML annotation files found in {self._annotations_dir}")
        return [f.stem for f in xml_files]

    def _discover_class_names(self) -> dict[int, str]:
        """Scan annotation files to build a deterministic (sorted) class mapping."""
        names: set[str] = set()
        for image_id in self._image_ids:
            ann_path = self._annotations_dir / f"{image_id}.xml"
            if not ann_path.is_file():
                continue
            try:
                tree = ET.parse(ann_path)
            except ET.ParseError:
                continue
            for obj in tree.getroot().findall("object"):
                name_el = obj.find("name")
                if name_el is not None and name_el.text:
                    names.add(name_el.text.strip())
        return {idx: name for idx, name in enumerate(sorted(names))}

    def _find_image_path(self, image_id: str) -> Path:
        """Locate an image file, trying the most common extensions in order."""
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = self._images_dir / f"{image_id}{ext}"
            if candidate.is_file():
                return candidate
        raise TrainingError(
            f"Image not found for ID '{image_id}' in {self._images_dir}. " "Tried .jpg, .jpeg, .png extensions."
        )

    def _parse_annotation(self, ann_path: Path) -> tuple[list[list[float]], list[int]]:
        """Parse a VOC XML file and return boxes + labels.

        Returns:
            boxes:  List of ``[xmin, ymin, xmax, ymax]`` bounding boxes.
            labels: Corresponding 0-indexed class labels.

        Missing or malformed annotation files yield empty lists (no error).
        """
        boxes: list[list[float]] = []
        labels: list[int] = []

        if not ann_path.is_file():
            return boxes, labels

        try:
            tree = ET.parse(ann_path)
        except ET.ParseError:
            return boxes, labels

        for obj in tree.getroot().findall("object"):
            # Honour the difficult flag
            if self.skip_difficult:
                difficult_el = obj.find("difficult")
                if difficult_el is not None and difficult_el.text is not None:
                    try:
                        if int(difficult_el.text.strip()) == 1:
                            continue
                    except ValueError:
                        pass

            name_el = obj.find("name")
            if name_el is None or not name_el.text:
                continue
            class_name = name_el.text.strip()
            if class_name not in self._class_to_idx:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            try:
                xmin = float(bndbox.findtext("xmin", default="0"))
                ymin = float(bndbox.findtext("ymin", default="0"))
                xmax = float(bndbox.findtext("xmax", default="0"))
                ymax = float(bndbox.findtext("ymax", default="0"))
            except (ValueError, TypeError):
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self._class_to_idx[class_name])

        return boxes, labels
