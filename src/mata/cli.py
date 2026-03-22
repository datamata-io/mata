"""MATA command-line interface.

Provides console-level UX parity with Ultralytics CLI.

Usage examples:
    mata run detect image.jpg --model facebook/detr-resnet-50 --save
    mata run classify image.jpg --model openai/clip-vit-base-patch32 --text "cat,dog"
    mata run embed image.jpg --model openai/clip-vit-base-patch32
    mata track video.mp4 --model facebook/detr-resnet-50 --tracker botsort --save
    mata val detect --model facebook/detr-resnet-50 --data coco.yaml
    mata export detect ./model.pt --format onnx          # stub: coming in v2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mata",
        description="MATA — Model-Agnostic Task Architecture for Computer Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mata run detect image.jpg --model facebook/detr-resnet-50 --save
  mata run classify image.jpg --model openai/clip-vit-base-patch32 --text "cat,dog"
  mata recognize image.jpg --gallery gallery.npz --model openai/clip-vit-base-patch32
  mata track video.mp4 --model facebook/detr-resnet-50 --tracker botsort --save
  mata val detect --model facebook/detr-resnet-50 --data coco.yaml
  mata export detect ./model.pt --format onnx   # coming in v2.0
""",
    )
    parser.add_argument(
        "--version", action="version", version=_get_version()
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v = quiet, -vv = verbose)",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    _add_run_parser(subparsers)
    _add_recognize_parser(subparsers)
    _add_track_parser(subparsers)
    _add_val_parser(subparsers)
    _add_export_parser(subparsers)

    return parser


def _get_version() -> str:
    try:
        import mata
        return f"mata {mata.__version__}"
    except Exception:
        return "mata (unknown version)"


def _add_recognize_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "recognize",
        help="Identify an image against a gallery of known embeddings",
        description="Gallery-based recognition. Embeds the input image and runs cosine "
                    "similarity search against a pre-built .npz gallery file.",
    )
    p.add_argument("input", help="Input image path")
    p.add_argument("--gallery", "-g", required=True,
                   help="Path to .npz gallery file (created with Gallery.save())")
    p.add_argument("--model", "-m", default=None,
                   help="Embed model ID, path, or alias (default: registry default)")
    p.add_argument("--top-k", type=int, default=1,
                   help="Number of top matches to return (default: 1)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Minimum cosine similarity threshold (default: gallery default)")
    p.add_argument("--device", default=None, help="Device: cpu, cuda, cuda:0, mps")
    p.add_argument("--json", dest="output_json", action="store_true",
                   help="Output raw JSON to stdout")


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Run inference on an image",
        description="Run one-shot inference. Wraps mata.run().",
    )
    p.add_argument("task", help='Task type: detect, segment, classify, depth, embed, ocr, vlm, barcode')
    p.add_argument("input", help="Input image path (file or URL)")
    p.add_argument("--model", "-m", default=None, help="Model ID, path, or config alias")
    p.add_argument("--conf", type=float, default=None, help="Confidence threshold (detect/segment)")
    p.add_argument("--device", default=None, help="Device: cpu, cuda, cuda:0, mps")
    p.add_argument("--text", default=None, help="Comma-separated text prompts for zero-shot tasks")
    p.add_argument("--prompt", default=None, help="Text prompt for VLM tasks")
    p.add_argument("--save", action="store_true", help="Save annotated result to disk")
    p.add_argument("--save-dir", default="runs/", help="Directory to save results (default: runs/)")
    p.add_argument("--json", dest="output_json", action="store_true", help="Output raw JSON to stdout")


def _add_track_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "track",
        help="Track objects in a video or stream",
        description="Run multi-object tracking. Wraps mata.track().",
    )
    p.add_argument("source", help="Video file, RTSP stream, or webcam index")
    p.add_argument("--model", "-m", default=None, help="Detection model ID, path, or alias")
    p.add_argument("--tracker", default="botsort", help="Tracker: botsort (default) or bytetrack")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    p.add_argument("--iou", type=float, default=0.7, help="IoU threshold (default: 0.7)")
    p.add_argument("--device", default=None, help="Device: cpu, cuda, cuda:0, mps")
    p.add_argument("--save", action="store_true", help="Save annotated output video")
    p.add_argument("--show", action="store_true", help="Display tracking results in a window")
    p.add_argument("--save-dir", default="runs/", help="Directory to save results (default: runs/)")
    p.add_argument("--reid-model", default=None, help="ReID model for appearance-based tracking")
    p.add_argument("--json", dest="output_json", action="store_true", help="Print per-frame JSON to stdout")


def _add_val_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "val",
        help="Evaluate a model on a dataset",
        description="Run model evaluation/validation. Wraps mata.val().",
    )
    p.add_argument("task", help="Task type: detect, segment, classify, depth, ocr")
    p.add_argument("--model", "-m", default=None, help="Model ID, path, or config alias")
    p.add_argument("--data", required=True, help="Dataset YAML config path")
    p.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (default: 0.001)")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for mAP (default: 0.5)")
    p.add_argument("--device", default=None, help="Device: cpu, cuda, cuda:0, mps")
    p.add_argument("--split", default="val", help="Dataset split to evaluate (default: val)")
    p.add_argument("--save-dir", default="runs/val/", help="Directory for plots/CSV (default: runs/val/)")
    p.add_argument("--plots", action="store_true", help="Save PR/F1/confusion plots")
    p.add_argument("--json", dest="output_json", action="store_true", help="Output metrics as JSON")


def _add_export_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "export",
        help="Export a model to a portable format [coming in v2.0]",
        description="Export a model to ONNX, TorchScript, etc. Stub — full support in v2.0.",
    )
    p.add_argument("task", help="Task type: detect, segment, classify")
    p.add_argument("model", help="Model ID or local path")
    p.add_argument("--format", default="onnx", choices=["onnx", "torchscript"], help="Export format (default: onnx)")
    p.add_argument("--quantize", default=None, choices=["int8", "fp16"], help="Quantization precision")
    p.add_argument("--output", "-o", default=None, help="Output file path")


# ------------------------------------------------------------------
# Command handlers
# ------------------------------------------------------------------


def _cmd_recognize(args: argparse.Namespace) -> int:
    import mata

    try:
        gallery = mata.Gallery.load(args.gallery)
    except Exception as exc:
        print(f"ERROR: could not load gallery '{args.gallery}': {exc}", file=sys.stderr)
        return 1

    kwargs: dict = {}
    if args.device is not None:
        kwargs["device"] = args.device

    try:
        result = mata.run(
            "recognize",
            args.input,
            model=args.model,
            gallery=gallery,
            top_k=args.top_k,
            threshold=args.threshold,
            **kwargs,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.output_json:
        print(result.to_dict().__class__.__name__)
        print(json.dumps(result.to_dict(), indent=2))
    else:
        entries = getattr(result, "entries", [])
        if not entries:
            print("No matches found.")
        else:
            entry = entries[0]
            print(f"Best match:  {entry.label}  (similarity={entry.similarity:.4f})")
            all_m = entry.all_matches
            if len(all_m) > 1:
                print(f"Top-{len(all_m)} matches:")
                for m in all_m:
                    print(f"  {m['label']:<30} {m['similarity']:.4f}")

    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    import mata

    kwargs: dict = {}
    if args.conf is not None:
        kwargs["threshold"] = args.conf
    if args.device is not None:
        kwargs["device"] = args.device
    if args.text is not None:
        kwargs["text_prompts"] = [t.strip() for t in args.text.split(",")]
    if args.prompt is not None:
        kwargs["prompt"] = args.prompt

    try:
        result = mata.run(args.task, args.input, model=args.model, **kwargs)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Output
    if args.output_json:
        if hasattr(result, "to_json"):
            print(result.to_json(indent=2))
        else:
            # Raw numpy from embed task
            import numpy as np
            if isinstance(result, np.ndarray):
                print(json.dumps({"embeddings": result.tolist()}))
            else:
                print(json.dumps(str(result)))
    else:
        _print_result(args.task, result)

    # Save
    if args.save:
        _save_result(args.task, result, args.input, args.save_dir)

    return 0


def _cmd_track(args: argparse.Namespace) -> int:
    import mata

    kwargs: dict = {}
    if args.device is not None:
        kwargs["device"] = args.device
    if args.reid_model is not None:
        kwargs["reid_model"] = args.reid_model

    try:
        results = mata.track(
            args.source,
            model=args.model,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            show=args.show,
            **kwargs,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    result_list = list(results)
    if args.output_json:
        print(json.dumps([r.to_dict() for r in result_list if hasattr(r, "to_dict")], indent=2))
    else:
        n_frames = len(result_list)
        total_tracks = sum(len(r.instances) for r in result_list if hasattr(r, "instances"))
        print(f"Tracked {n_frames} frames | total detections: {total_tracks}")
        if args.save:
            print(f"Saved output to: {args.save_dir}")

    return 0


def _cmd_val(args: argparse.Namespace) -> int:
    import mata

    try:
        metrics = mata.val(
            args.task,
            model=args.model,
            data=args.data,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            split=args.split,
            save_dir=args.save_dir,
            plots=args.plots,
            verbose=not args.output_json,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.output_json:
        if hasattr(metrics, "to_dict"):
            print(json.dumps(metrics.to_dict(), indent=2))
        else:
            print(json.dumps(str(metrics)))

    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    print(
        f"mata export: Coming in v2.0.\n"
        f"  Task:   {args.task}\n"
        f"  Model:  {args.model}\n"
        f"  Format: {args.format}\n"
        f"  Quantize: {args.quantize or 'none'}\n"
        f"\nSee CHANGELOG.md for the v2.0 roadmap."
    )
    return 0


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------


def _print_result(task: str, result: object) -> None:
    """Print a human-readable summary of the result."""
    import numpy as np

    if task in ("detect", "segment"):
        instances = getattr(result, "instances", [])
        print(f"Detected {len(instances)} instance(s):")
        for inst in instances[:20]:
            name = getattr(inst, "label_name", None) or f"class_{getattr(inst, 'label', '?')}"
            score = getattr(inst, "score", 0.0)
            bbox = getattr(inst, "bbox", None)
            bbox_str = f"  bbox={[round(v) for v in bbox]}" if bbox else ""
            print(f"  {name:<20} score={score:.3f}{bbox_str}")
        if len(instances) > 20:
            print(f"  ... and {len(instances) - 20} more.")

    elif task == "classify":
        classifications = getattr(result, "classifications", [])
        print(f"Top-{min(5, len(classifications))} classifications:")
        for clf in classifications[:5]:
            name = getattr(clf, "label_name", None) or f"class_{getattr(clf, 'label', '?')}"
            score = getattr(clf, "score", 0.0)
            print(f"  {name:<30} {score:.4f}")

    elif task == "depth":
        depth_map = getattr(result, "depth_map", getattr(result, "depth", None))
        if depth_map is not None and isinstance(depth_map, np.ndarray):
            print(f"Depth map: shape={depth_map.shape}, min={depth_map.min():.3f}, max={depth_map.max():.3f}")

    elif task == "embed":
        if isinstance(result, np.ndarray):
            print(f"Embeddings: shape={result.shape}, dtype={result.dtype}")
        elif hasattr(result, "embeddings"):
            emb = result.embeddings
            print(f"Embeddings: shape={emb.shape}, dtype={emb.dtype}")

    elif task == "ocr":
        regions = getattr(result, "regions", [])
        print(f"OCR: {len(regions)} text region(s)")
        for r in regions[:10]:
            print(f"  {r.text!r:<40} score={r.score:.3f}")

    elif task == "barcode":
        barcodes = getattr(result, "barcodes", [])
        print(f"Barcodes: {len(barcodes)} found")
        for b in barcodes:
            print(f"  [{b.type}] {b.data}")

    elif task == "vlm":
        text = getattr(result, "text", None)
        if text:
            print(f"VLM response:\n{text}")

    else:
        # Generic fallback
        if hasattr(result, "to_dict"):
            print(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            print(str(result))


def _save_result(task: str, result: object, input_path: str, save_dir: str) -> None:
    """Save result to the save directory."""
    import numpy as np

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    try:
        if task in ("detect", "segment", "classify", "ocr", "barcode"):
            out_file = out_dir / f"{stem}_result.json"
            if hasattr(result, "save"):
                result.save(str(out_file))
            print(f"Saved: {out_file}")

        elif task == "embed":
            out_file = out_dir / f"{stem}_embeddings.npz"
            if isinstance(result, np.ndarray):
                np.savez(str(out_file), embeddings=result)
            elif hasattr(result, "save"):
                result.save(str(out_file))
            print(f"Saved: {out_file}")

        elif task == "depth":
            out_file = out_dir / f"{stem}_depth.png"
            if hasattr(result, "save"):
                result.save(str(out_file))
            print(f"Saved: {out_file}")

        else:
            out_file = out_dir / f"{stem}_result.json"
            if hasattr(result, "save"):
                result.save(str(out_file))
            elif hasattr(result, "to_json"):
                out_file.write_text(result.to_json(indent=2), encoding="utf-8")
            print(f"Saved: {out_file}")

    except Exception as exc:
        print(f"Warning: could not save result — {exc}", file=sys.stderr)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Set verbosity
    if args.verbose == 1:
        import mata
        mata.verbose(1)
    elif args.verbose >= 2:
        import mata
        mata.verbose(2)

    dispatch = {
        "run": _cmd_run,
        "recognize": _cmd_recognize,
        "track": _cmd_track,
        "val": _cmd_val,
        "export": _cmd_export,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
