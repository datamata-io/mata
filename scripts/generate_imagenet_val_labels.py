"""
Generate data/imagenet/imagenet_val_labels.json from bbox_val/ XML annotations.

The XML files ship with the official ILSVRC2012 validation devkit and contain
the ground-truth synset (e.g. n01751748) for every validation image.
Class indices are assigned by sorting synsets alphabetically — the standard
ILSVRC2012 ordering used by PyTorch / torchvision.

Usage:
    python scripts/generate_imagenet_val_labels.py
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path


BBOX_DIR = Path("data/imagenet/bbox_val")
OUTPUT   = Path("data/imagenet/imagenet_val_labels.json")


def main():
    # ── Step 1: parse every XML ───────────────────────────────────────────────
    print(f"Scanning {BBOX_DIR} …")
    synset_per_image: dict[str, str] = {}

    xml_files = sorted(BBOX_DIR.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {BBOX_DIR}")

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename_el = root.find("filename")
        if filename_el is None:
            print(f"  WARNING: no <filename> in {xml_path.name}, skipping")
            continue
        # Normalise to the JPEG name used in val/
        filename = filename_el.text.strip()
        if not filename.endswith(".JPEG"):
            filename += ".JPEG"

        # Use the first annotated object's synset as the image label
        name_el = root.find(".//object/name")
        if name_el is None:
            print(f"  WARNING: no <object/name> in {xml_path.name}, skipping")
            continue

        synset_per_image[filename] = name_el.text.strip()

    print(f"  Parsed {len(synset_per_image):,} images")

    # ── Step 2: build synset → index (alphabetical = ILSVRC2012 standard) ────
    unique_synsets = sorted(set(synset_per_image.values()))
    if len(unique_synsets) != 1000:
        print(f"  WARNING: found {len(unique_synsets)} unique synsets (expected 1000)")
    synset_to_idx = {s: i for i, s in enumerate(unique_synsets)}

    # ── Step 3: write JSON ────────────────────────────────────────────────────
    labels = {fn: synset_to_idx[syn] for fn, syn in synset_per_image.items()}

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(labels, f)

    print(f"  Wrote {OUTPUT}  ({len(labels):,} entries, {len(unique_synsets)} classes)")
    print("Done.")


if __name__ == "__main__":
    main()
