"""
Convert data/imagenet/imagenet_val_labels.json → data/imagenet/imagenet_val_coco.json

The MATA validator (DatasetLoader) expects a COCO-format annotations file.
This script wraps the flat {filename: class_index} mapping produced by
generate_imagenet_val_labels.py into the required COCO structure:

    {
      "images":      [{id, file_name, width, height}, ...],
      "annotations": [{id, image_id, category_id, bbox, area, iscrowd}, ...],
      "categories":  [{id, name}, ...]
    }

For classification evaluation:
  - Each image gets exactly one annotation.
  - category_id = class_index + 1  (COCO ids are 1-based).
  - bbox / area are set to zero (unused by the classify validator).

Usage:
    python scripts/convert_imagenet_to_coco.py
"""
import json
from pathlib import Path

SRC     = Path("data/imagenet/imagenet_val_labels.json")
OUT     = Path("data/imagenet/imagenet_val_coco.json")
VAL_DIR = Path("data/imagenet/val")

# Optional: path to the ImageNet synset list to attach human-readable names.
# If omitted, categories are named after the synset ID (e.g. "n01440764").
SYNSET_FILE: Path | None = None  # e.g. Path("data/imagenet/synsets.txt")


def load_synset_names(synset_file: Path) -> dict[str, str]:
    """Return {synset: human_name} from a text file (one 'nXXXXXXXX name' per line)."""
    mapping: dict[str, str] = {}
    with synset_file.open() as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def main() -> None:
    # ── Load source files ─────────────────────────────────────────────────────
    print(f"Loading {SRC} …")
    with SRC.open() as f:
        labels: dict[str, int] = json.load(f)

    print(f"  {len(labels):,} image entries, "
          f"{len(set(labels.values()))} unique classes")

    # Optional synset names
    synset_names: dict[str, str] = {}
    if SYNSET_FILE and SYNSET_FILE.exists():
        synset_names = load_synset_names(SYNSET_FILE)
        print(f"  Loaded {len(synset_names)} synset names from {SYNSET_FILE}")

    # ── Build reverse lookup: class_index → synset ────────────────────────────
    # Re-derive from the XM labels so names match exactly how indices were assigned.
    xml_dir = Path("data/imagenet/bbox_val")
    idx_to_synset: dict[int, str] = {}

    if xml_dir.exists():
        import xml.etree.ElementTree as ET
        synset_per_image: dict[str, str] = {}
        for xml_path in sorted(xml_dir.glob("*.xml")):
            root = ET.parse(xml_path).getroot()
            fn_el = root.find("filename")
            name_el = root.find(".//object/name")
            if fn_el is not None and name_el is not None:
                fn = fn_el.text.strip()
                if not fn.endswith(".JPEG"):
                    fn += ".JPEG"
                synset_per_image[fn] = name_el.text.strip()
        unique_synsets = sorted(set(synset_per_image.values()))
        idx_to_synset = {i: s for i, s in enumerate(unique_synsets)}
        print(f"  Rebuilt {len(idx_to_synset)} synset → index mappings from XML files")

    # ── Assemble COCO structures ──────────────────────────────────────────────
    categories: list[dict] = []
    seen_indices = sorted(set(labels.values()))
    for class_idx in seen_indices:
        synset = idx_to_synset.get(class_idx, f"class_{class_idx:04d}")
        name   = synset_names.get(synset, synset)
        categories.append({
            "id":   class_idx + 1,   # COCO category IDs are 1-indexed
            "name": name,
            "supercategory": "object",
        })

    images_list: list[dict] = []
    annotations_list: list[dict] = []

    for img_id, (filename, class_idx) in enumerate(sorted(labels.items()), start=1):
        images_list.append({
            "id":        img_id,
            "file_name": filename,
            "width":     0,    # unused by the classify validator
            "height":    0,
        })
        annotations_list.append({
            "id":          img_id,
            "image_id":    img_id,
            "category_id": class_idx + 1,   # 1-indexed
            "bbox":        [0, 0, 0, 0],
            "area":        0,
            "iscrowd":     0,
        })

    coco = {
        "images":      images_list,
        "annotations": annotations_list,
        "categories":  categories,
    }

    # ── Write ─────────────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(coco, f)

    print(f"  Wrote {OUT}")
    print(f"  {len(images_list):,} images, "
          f"{len(annotations_list):,} annotations, "
          f"{len(categories)} categories")
    print("Done.")


if __name__ == "__main__":
    main()
