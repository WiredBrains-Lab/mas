#!/usr/bin/env python3
"""Run fixed-width concept-decoding windows locally or as a Slurm array.

Arguments after ``--`` are passed unchanged to run_concept_decoding_cli.py.
This wrapper owns --window-start, --window-end, and --output-dir so every task
has a deterministic directory and can be collected after an HPC array finishes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DECODING_CLI = ROOT / "scripts" / "run_concept_decoding_cli.py"
MANAGED_FLAGS = {"--window-start", "--window-end", "--output-dir"}
SUMMARY_FIELDS = (
    "accuracy",
    "balanced_accuracy",
    "mean_percentile_rank",
    "top1_accuracy",
    "top1_balanced_accuracy",
    "top3_accuracy",
    "top5_accuracy",
    "chance_accuracy",
    "top1_chance_accuracy",
    "permutation_p_value_accuracy",
    "permutation_p_value_balanced_accuracy",
    "n_trials",
    "n_train",
    "n_validation",
    "n_test",
    "n_classes",
    "n_runs",
    "n_features",
)


def temporal_windows(start_s: float, stop_s: float, width_ms: float, step_ms: float) -> list[tuple[float, float]]:
    """Return fixed-width half-open windows whose end does not exceed stop_s."""
    values = (start_s, stop_s, width_ms, step_ms)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Sweep bounds, width, and step must be finite.")
    width_s = float(width_ms) / 1000.0
    step_s = float(step_ms) / 1000.0
    if width_s <= 0 or step_s <= 0:
        raise ValueError("Window width and step must be positive.")
    if float(stop_s) <= float(start_s) or width_s > float(stop_s) - float(start_s) + 1e-12:
        raise ValueError("Sweep stop must allow at least one complete window after sweep start.")
    count = int(math.floor(((float(stop_s) - float(start_s)) - width_s + 1e-12) / step_s)) + 1
    return [
        (round(float(start_s) + index * step_s, 9), round(float(start_s) + index * step_s + width_s, 9))
        for index in range(count)
    ]


def _time_token(value_s: float) -> str:
    value_ms = int(round(float(value_s) * 1000.0))
    prefix = "m" if value_ms < 0 else "p"
    return f"{prefix}{abs(value_ms):04d}ms"


def window_directory(root: Path, index: int, start_s: float, end_s: float) -> Path:
    return root / f"window_{index:03d}_{_time_token(start_s)}_{_time_token(end_s)}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_manifest(root: Path, windows: list[tuple[float, float]], args: argparse.Namespace, decoder_args: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "temporal_windows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("task_index", "window_start_s", "window_end_s", "output_dir"))
        writer.writeheader()
        for index, (start_s, end_s) in enumerate(windows):
            writer.writerow({
                "task_index": index,
                "window_start_s": start_s,
                "window_end_s": end_s,
                "output_dir": str(window_directory(root, index, start_s, end_s)),
            })
    _write_json(root / "sweep_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_start_s": args.sweep_start,
        "sweep_stop_s": args.sweep_stop,
        "window_width_ms": args.window_width_ms,
        "step_ms": args.step_ms,
        "n_windows": len(windows),
        "array_range": f"0-{len(windows) - 1}",
        "decoder_cli": str(DECODING_CLI),
        "decoder_args": decoder_args,
    })


def read_windows_manifest(path: Path) -> list[tuple[float, float]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        windows = [(float(row["window_start_s"]), float(row["window_end_s"])) for row in rows]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot read temporal window manifest {path}: {exc}") from exc
    if not windows:
        raise ValueError(f"Temporal window manifest is empty: {path}")
    return windows


def collect_results(root: Path, windows: list[tuple[float, float]]) -> Path:
    rows: list[dict[str, Any]] = []
    for index, (start_s, end_s) in enumerate(windows):
        output_dir = window_directory(root, index, start_s, end_s)
        status_path = output_dir / "temporal_task_status.json"
        summary_path = output_dir / "summary.json"
        status: dict[str, Any] = {}
        summary: dict[str, Any] = {}
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                status = {"status": "invalid_status"}
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                summary = {}
        row = {
            "task_index": index,
            "window_start_s": start_s,
            "window_end_s": end_s,
            "window_center_s": round((start_s + end_s) / 2.0, 9),
            "status": status.get("status", "complete" if summary else "pending"),
            "return_code": status.get("return_code", 0 if summary else ""),
            "output_dir": str(output_dir),
        }
        for field in SUMMARY_FIELDS:
            row[field] = summary.get(field, "")
        rows.append(row)
    output = root / "temporal_window_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def run_window(root: Path, index: int, window: tuple[float, float], decoder_args: list[str]) -> int:
    start_s, end_s = window
    output_dir = window_directory(root, index, start_s, end_s)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(DECODING_CLI),
        *decoder_args,
        "--window-start",
        str(start_s),
        "--window-end",
        str(end_s),
        "--output-dir",
        str(output_dir),
    ]
    (output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    _write_json(output_dir / "temporal_task_status.json", {
        "status": "running",
        "task_index": index,
        "window_start_s": start_s,
        "window_end_s": end_s,
        "command": command,
    })
    completed = subprocess.run(command, cwd=ROOT)
    status = "complete" if completed.returncode == 0 else "failed"
    _write_json(output_dir / "temporal_task_status.json", {
        "status": status,
        "return_code": completed.returncode,
        "task_index": index,
        "window_start_s": start_s,
        "window_end_s": end_s,
        "command": command,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    return int(completed.returncode)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    argv = list(sys.argv[1:] if argv is None else argv)
    decoder_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        decoder_args = argv[split + 1:]
        argv = argv[:split]
    parser = argparse.ArgumentParser(
        description="Run a deterministic temporal-window sweep through the existing concept-decoding CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep-output", required=True, help="Root directory for the manifest and per-window outputs.")
    parser.add_argument("--sweep-start", type=float, default=-0.2, help="First possible window start, seconds from stimulus onset.")
    parser.add_argument("--sweep-stop", type=float, default=0.6, help="Latest allowed window end, seconds from stimulus onset.")
    parser.add_argument("--window-width-ms", type=float, default=50.0, help="Width of every decoding window.")
    parser.add_argument("--step-ms", type=float, default=20.0, help="Distance between adjacent window starts.")
    parser.add_argument("--task-index", type=int, help="Run only this zero-based window; defaults to SLURM_ARRAY_TASK_ID when set.")
    parser.add_argument("--manifest-only", action="store_true", help="Write the window manifest without running decoding.")
    parser.add_argument("--collect-only", action="store_true", help="Collect existing task outputs without running decoding.")
    parser.add_argument("--continue-on-error", action="store_true", help="When running all windows locally, continue after a failed window.")
    args = parser.parse_args(argv)
    args.sweep_output = str(Path(args.sweep_output).expanduser().resolve())
    if args.task_index is None and os.environ.get("SLURM_ARRAY_TASK_ID", "").strip():
        try:
            args.task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        except ValueError:
            parser.error("SLURM_ARRAY_TASK_ID must be an integer")
    if args.manifest_only and args.collect_only:
        parser.error("--manifest-only and --collect-only are mutually exclusive")
    used_managed = [token.split("=", 1)[0] for token in decoder_args if token.split("=", 1)[0] in MANAGED_FLAGS]
    if used_managed:
        parser.error(f"Pass {', '.join(sorted(set(used_managed)))} to the sweep wrapper, not after --")
    if not args.manifest_only and not args.collect_only and not decoder_args:
        parser.error("Decoder arguments are required after -- (for example -- --config configs/decoding.hpc.toml)")
    return args, decoder_args


def main(argv: list[str] | None = None) -> int:
    args, decoder_args = parse_args(argv)
    root = Path(args.sweep_output)
    manifest_path = root / "temporal_windows.csv"
    if (args.task_index is not None or args.collect_only) and not manifest_path.is_file():
        print(
            f"Missing {manifest_path}. Run this command once with --manifest-only before submitting the array.",
            file=sys.stderr,
        )
        return 2
    try:
        if args.task_index is not None or args.collect_only:
            windows = read_windows_manifest(manifest_path)
        else:
            windows = temporal_windows(args.sweep_start, args.sweep_stop, args.window_width_ms, args.step_ms)
    except ValueError as exc:
        print(f"Temporal sweep configuration error: {exc}", file=sys.stderr)
        return 2
    if not args.collect_only and args.task_index is None:
        write_manifest(root, windows, args, decoder_args)
    if args.manifest_only:
        print(f"Wrote {len(windows)} windows to {root / 'temporal_windows.csv'}; Slurm array range: 0-{len(windows) - 1}")
        return 0
    if args.collect_only:
        print(f"Collected results in {collect_results(root, windows)}")
        return 0
    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(windows):
            print(f"Task index {args.task_index} is outside 0-{len(windows) - 1}.", file=sys.stderr)
            return 2
        return run_window(root, args.task_index, windows[args.task_index], decoder_args)
    return_code = 0
    for index, window in enumerate(windows):
        code = run_window(root, index, window, decoder_args)
        collect_results(root, windows)
        if code:
            return_code = code
            if not args.continue_on_error:
                break
    print(f"Collected results in {collect_results(root, windows)}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
