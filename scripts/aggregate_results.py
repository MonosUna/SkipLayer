#!/usr/bin/env python3
"""Aggregate `metrics_summary.json` produced by
[`JsonlFileLogger`](src/loggers/jsonl_logger.py:1) across all runs in
`outputs/` and print one comparison table.

Usage:
    python scripts/aggregate_results.py [outputs_dir]
        [--csv path/to/results.csv]
        [--keys key1,key2,...]

If ``--keys`` is omitted, prints a curated set of high-signal metrics:
generate split's match-vs-full-model accuracy at various skip counts,
candidate-share aggregates and skip-percent stats.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

DEFAULT_KEYS = [
    "generate/skip_metric/skip_match_accuracy",
    "generate/skip_metric/full_log_prob/mean",
    "generate/skip_metric/skip_log_prob/mean",
    "generate/count/samples",
    "generate/candidate_count/eligible_total",
    "generate/candidate_count/protected_total",
]


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "metrics_summary.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def collect_runs(outputs_dir: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    if not outputs_dir.exists():
        print(f"No outputs dir found at {outputs_dir}", file=sys.stderr)
        return runs
    for child in sorted(outputs_dir.iterdir()):
        if not child.is_dir():
            continue
        summary = load_summary(child)
        if summary:
            runs[child.name] = summary
    return runs


def select_keys(runs: dict[str, dict], explicit_keys: list[str] | None) -> list[str]:
    if explicit_keys:
        return explicit_keys
    # Auto-pick: any DEFAULT_KEYS that appear in at least one run.
    keys = [k for k in DEFAULT_KEYS if any(k in s for s in runs.values())]
    if keys:
        return keys
    # Fallback: union of all numeric keys across runs.
    seen = set()
    for s in runs.values():
        for k, v in s.items():
            if isinstance(v, (int, float)) and not k.startswith("_"):
                seen.add(k)
    return sorted(seen)


def fmt_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_table(runs: dict[str, dict], keys: list[str]) -> None:
    if not runs:
        print("No runs with metrics_summary.json found.")
        return
    run_names = list(runs.keys())
    name_w = max(len("run"), *(len(n) for n in run_names))
    col_w = [max(len(k), 8) for k in keys]
    header = "run".ljust(name_w) + " | " + " | ".join(
        k.ljust(w) for k, w in zip(keys, col_w, strict=False)
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for name in run_names:
        row = [name.ljust(name_w)]
        for k, w in zip(keys, col_w, strict=False):
            v = runs[name].get(k, "—")
            row.append(fmt_value(v).ljust(w))
        print(" | ".join(row))


def write_csv(runs: dict[str, dict], keys: list[str], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", *keys])
        for name, summary in runs.items():
            writer.writerow([name, *(summary.get(k, "") for k in keys)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "outputs_dir",
        nargs="?",
        default="outputs",
        help="Directory with per-run subdirectories (default: ./outputs).",
    )
    parser.add_argument(
        "--csv", default=None, help="Optional path to dump full table as CSV."
    )
    parser.add_argument(
        "--keys",
        default=None,
        help="Comma-separated metric keys to display (default: curated set).",
    )
    args = parser.parse_args()

    runs = collect_runs(Path(args.outputs_dir))
    explicit = args.keys.split(",") if args.keys else None
    keys = select_keys(runs, explicit)
    print_table(runs, keys)
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(runs, keys, out)
        print(f"\nSaved CSV: {out.resolve()}")


if __name__ == "__main__":
    main()
