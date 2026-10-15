"""Command-line entry point. Run python main.py --help."""
import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description="Geospatial super-resolution for continuous optical GeoTIFFs")
    commands = p.add_subparsers(dest="command", required=True)
    info = commands.add_parser("inspect", help="Inspect pixel size, CRS and bands")
    info.add_argument("input")
    prep = commands.add_parser("prepare", help="Create HR patches from geographically separate train/validation scenes")
    prep.add_argument("--train-dir", required=True)
    prep.add_argument("--val-dir", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--scale", type=int, choices=[2, 4, 8], default=4)
    prep.add_argument("--patch-size", type=int, default=128)
    prep.add_argument("--stride", type=int, default=128)
    prep.add_argument("--value-scale", type=float, default=10000.0, help="Divide stored values by this after subtracting offset; choose for your product")
    prep.add_argument("--value-offset", type=float, default=0.0)
    prep.add_argument("--max-patches", type=int, default=10000, help="Maximum per split")
    prep.add_argument("--lr-train-dir", help="Optional real LR scenes, filenames matched to HR training scenes")
    prep.add_argument("--lr-val-dir", help="Optional real LR scenes, filenames matched to HR validation scenes")
    tr = commands.add_parser("train", help="Train and save the best validation checkpoint")
    tr.add_argument("--data", required=True)
    tr.add_argument("--output", required=True)
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--batch-size", type=int, default=8)
    tr.add_argument("--features", type=int, default=64)
    tr.add_argument("--blocks", type=int, default=8)
    tr.add_argument("--learning-rate", type=float, default=1e-4)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--patience", type=int, default=10, help="Early stopping patience; 0 disables")
    tr.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ev = commands.add_parser("evaluate", help="Compare a checkpoint against bicubic on prepared validation pairs")
    ev.add_argument("--data", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ev.add_argument("--data-range", type=float, default=1.0)
    ev.add_argument("--report", help="Optional JSON report path")
    comp = commands.add_parser("compare", help="Evaluate an output against an aligned independent reference GeoTIFF")
    comp.add_argument("prediction")
    comp.add_argument("reference")
    comp.add_argument("--data-range", type=float, required=True, help="Physical/stored signal range used for PSNR, e.g. 1 for reflectance")
    comp.add_argument("--tile-size", type=int, default=512)
    comp.add_argument("--apply-scale-offset", action="store_true")
    comp.add_argument("--report")
    pred = commands.add_parser("predict", help="Write a georeferenced output raster")
    pred.add_argument("input_path")
    pred.add_argument("output_path")
    pred.add_argument("--scale", type=int, choices=[2, 4, 8], default=4)
    pred.add_argument("--method", choices=["model", "bicubic", "bilinear", "nearest"], default="model")
    pred.add_argument("--checkpoint")
    pred.add_argument("--tile-size", type=int, default=256)
    pred.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    pred.add_argument("--ensemble", action="store_true", help="Average eight rotated/flipped predictions (slower)")
    pred.add_argument("--build-overviews", action="store_true", help="Add average overviews for faster GIS viewing")
    return p


def main():
    p = parser()
    args = vars(p.parse_args())
    command = args.pop("command")
    try:
        if command == "inspect":
            from src.geospatial_utils import inspect_raster
            result = inspect_raster(args["input"])
        elif command == "prepare":
            from src.preprocessing import prepare
            result = prepare(**args)
        elif command == "train":
            from src.train import train
            result = train(**args)
        elif command in ("evaluate", "compare"):
            if command == "evaluate":
                from src.train import evaluate as operation
            else:
                from src.validation import compare_rasters as operation
            from src.reporting import save_json
            report = args.pop("report")
            if report and Path(report).exists():
                raise ValueError("Report already exists. Choose a new report path.")
            result = operation(**args)
            if report:
                save_json(report, result)
        else:
            from src.predict import predict
            result = {"output": predict(**args)}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
        p.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
