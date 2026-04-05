"""Annotation Workflow Quickstart — MATA Framework

Demonstrates the full mata.annotate() workflow:
  1. Start the annotation server (non-blocking, no browser)
  2. Create a dataset and add sample images
  3. Save COCO-format annotations via the REST API
  4. Export the dataset as a training-ready YAML config
  5. Verify the exported COCO JSON is valid
  6. (Optional) Trigger mata.train() with the exported dataset

Prerequisites:
  - MATA installed: pip install mata
  - Sample images in data/ or the script will use coco_mini if available

Run:
    python examples/annotate/quickstart.py
"""

from __future__ import annotations

import http.client
import json
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these to match your environment
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data")            # Where datasets are stored
DATASET_NAME = "annotate_demo"      # Name for the demo dataset
PORT = 8710                         # Server port (set 0 for random OS-assigned)
HOST = "127.0.0.1"


# ---------------------------------------------------------------------------
# HTTP helpers (no external dependencies — pure stdlib)
# ---------------------------------------------------------------------------

def _get(port: int, path: str) -> tuple[int, object]:
    conn = http.client.HTTPConnection(HOST, port, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    try:
        return resp.status, json.loads(raw)
    except json.JSONDecodeError:
        return resp.status, raw


def _post(port: int, path: str, body: dict | None = None) -> tuple[int, object]:
    payload = json.dumps(body or {}).encode()
    conn = http.client.HTTPConnection(HOST, port, timeout=10)
    conn.request(
        "POST",
        path,
        body=payload,
        headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
    )
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    try:
        return resp.status, json.loads(raw)
    except json.JSONDecodeError:
        return resp.status, raw


# ---------------------------------------------------------------------------
# Step 1: Start the annotation server
# ---------------------------------------------------------------------------

def start_server() -> object:
    """Start the annotation server in the background (non-blocking)."""
    import mata  # noqa: PLC0415

    print(f"[1] Starting annotation server on {HOST}:{PORT} ...")
    server = mata.annotate(
        str(DATA_ROOT),
        port=PORT,
        block=False,
        open_browser=False,
    )
    # Wait briefly to ensure the server thread is ready
    time.sleep(0.3)
    print(f"    Server URL: {server.url}")

    # Confirm health
    status, body = _get(server.port, "/api/health")
    assert status == 200 and body.get("status") == "ok", f"Health check failed: {body}"
    print("    Health: OK")
    return server


# ---------------------------------------------------------------------------
# Step 2: Prepare the dataset directory and sample images
# ---------------------------------------------------------------------------

def prepare_dataset(server_port: int) -> Path:
    """Create a demo dataset and populate it with sample images."""
    dataset_dir = DATA_ROOT / DATASET_NAME
    images_dir = dataset_dir / "images"

    # Remove existing demo dataset so this script is idempotent
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    # Register dataset via API — this creates images/ and annotations/ directories
    status, body = _post(server_port, f"/api/datasets/{DATASET_NAME}")
    assert status in (200, 201), f"Dataset creation failed: {body}"
    print(f"    Dataset registered: {DATASET_NAME}")

    # Prefer real images from coco_mini if available, otherwise synthesise stubs
    source_images = list((DATA_ROOT / "coco_mini" / "images").glob("*.jpg"))[:3]
    if not source_images:
        # Fall back: create 2 minimal JPEG stubs (SOI marker only — enough for
        # the server's image-list endpoint; real inference would need valid JPEG)
        for i in range(1, 3):
            (images_dir / f"sample_{i:03d}.jpg").write_bytes(
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
                b"\xff\xd9"
            )
        print(f"[2] Created {DATASET_NAME} with 2 synthetic stub images")
    else:
        for src in source_images:
            shutil.copy(src, images_dir / src.name)
        print(f"[2] Created {DATASET_NAME} with {len(source_images)} images from coco_mini")

    return dataset_dir


# ---------------------------------------------------------------------------
# Step 3: List images and save COCO annotations
# ---------------------------------------------------------------------------

def annotate_dataset(server_port: int, dataset_dir: Path) -> None:
    """Push a sample COCO annotation payload to the server."""
    # List images known to the server
    status, body = _get(server_port, f"/api/datasets/{DATASET_NAME}/images")
    assert status == 200, f"Image listing failed: {body}"
    # API returns list of dicts: [{"filename": "...", "size_bytes": ...}, ...]
    raw_items: list = body if isinstance(body, list) else body.get("images", [])
    image_files: list[str] = [
        item["filename"] if isinstance(item, dict) else item
        for item in raw_items
    ]
    print(f"[3] Images in dataset: {image_files}")

    if not image_files:
        print("    No images found — skipping annotation step")
        return

    # Build a minimal COCO payload with one bounding-box annotation per image
    images = []
    annotations = []
    categories = [{"id": 1, "name": "object", "supercategory": "object"}]

    for idx, fname in enumerate(image_files, start=1):
        images.append({"id": idx, "file_name": fname, "width": 640, "height": 480})
        annotations.append({
            "id": idx,
            "image_id": idx,
            "category_id": 1,
            "bbox": [10.0, 10.0, 100.0, 80.0],   # x, y, w, h  (COCO xywh)
            "area": 8000.0,
            "iscrowd": 0,
            "segmentation": [],
        })

    coco_payload = {
        "info": {"description": "annotate_demo", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    status, body = _post(
        server_port,
        f"/api/datasets/{DATASET_NAME}/annotations",
        coco_payload,
    )
    assert status in (200, 201), f"Annotation save failed: {body}"
    print(f"    Saved {len(annotations)} annotation(s) for {len(images)} image(s)")


# ---------------------------------------------------------------------------
# Step 4 & 5: Export dataset and verify COCO output
# ---------------------------------------------------------------------------

def export_and_verify(server_port: int, dataset_dir: Path) -> Path:
    """Export the dataset to YAML + COCO JSON, then validate the output."""
    from mata.annotate.coco_io import load_annotations, validate_coco  # noqa: PLC0415

    status, body = _post(server_port, f"/api/datasets/{DATASET_NAME}/export")
    assert status in (200, 201), f"Export failed: {body}"
    print(f"[4] Export response: {body}")

    # Locate produced files
    yaml_path = dataset_dir / "dataset.yaml"
    ann_dir = dataset_dir / "annotations"

    if yaml_path.exists():
        print(f"[5] dataset.yaml  -> {yaml_path}")
    else:
        print("[5] NOTE: dataset.yaml not yet written (export created COCO JSON only)")

    # Find any COCO JSON files and validate them
    coco_files = list(ann_dir.glob("*.json")) if ann_dir.exists() else []
    if not coco_files:
        coco_files = list(dataset_dir.rglob("*.json"))

    validated = 0
    for coco_file in coco_files:
        coco = load_annotations(coco_file)
        warnings = validate_coco(coco)
        if warnings:
            print(f"    COCO warnings in {coco_file.name}: {warnings}")
        else:
            print(f"    {coco_file.name}: VALID (no warnings)")
        validated += 1

    if validated == 0:
        print("    No COCO JSON files found to validate (annotations may be empty)")

    return yaml_path


# ---------------------------------------------------------------------------
# (Optional) Step 6: Trigger training
# ---------------------------------------------------------------------------

def trigger_training(server_port: int, yaml_path: Path) -> None:
    """Demonstrate the training trigger endpoint (mocked — no GPU required)."""
    if not yaml_path.exists():
        print("[6] Skipping training trigger — dataset.yaml not present")
        return

    # mata.train() would normally be called here.  We just show the API shape.
    print("[6] Training trigger example (not executing real training):")
    print(f"    mata.train('detect', model='facebook/detr-resnet-50',")
    print(f"               data='{yaml_path}', epochs=10)")
    print("    Use `mata train detect --model ... --data ... --epochs 10` from CLI")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    server = start_server()
    try:
        dataset_dir = prepare_dataset(server.port)
        annotate_dataset(server.port, dataset_dir)
        yaml_path = export_and_verify(server.port, dataset_dir)
        trigger_training(server.port, yaml_path)
        print("\nQuickstart complete — annotation workflow demonstrated successfully.")
    finally:
        server.shutdown()
        print("Server stopped.")


if __name__ == "__main__":
    main()
