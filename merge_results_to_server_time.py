#!/usr/bin/env python3
"""Merge local and server results using server-normalized runtimes.

Server rows keep their measured runtime. Local rows are converted to predicted
server runtime with problem-level ratios estimated from the overlapping
medium-main run.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


BASE_FIELDS = ["status", "strategy_id", "policy_id", "problem", "instance", "final_cost", "gap", "time"]
EXTRA_FIELDS = [
    "original_time",
    "runtime_scale",
    "runtime_scale_basis",
    "source_tree",
    "source_run",
    "source_csv",
]
MERGED_FIELDS = BASE_FIELDS + EXTRA_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge results/ and results_server/results with local runtimes converted to server-predicted time."
    )
    parser.add_argument("--local-root", type=Path, default=Path("results"))
    parser.add_argument("--server-root", type=Path, default=Path("results_server/results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_merged"))
    parser.add_argument("--basis-run", default="medium-main", help="run name used to estimate local/server runtime ratios")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def result_csvs(root: Path) -> list[Path]:
    return sorted(root.glob("**/results.csv"))


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("strategy_id", ""),
        row.get("policy_id", ""),
        row.get("problem", ""),
        row.get("instance", ""),
        row.get("status", ""),
    )


def runtime_scales(local_root: Path, server_root: Path, basis_run: str) -> dict[str, float]:
    local_rows = {
        row_key(row): row
        for row in read_csv(local_root / basis_run / "results.csv")
        if row.get("status") == "ok"
    }
    server_rows = {
        row_key(row): row
        for row in read_csv(server_root / basis_run / "results.csv")
        if row.get("status") == "ok"
    }
    local_times: dict[str, list[float]] = defaultdict(list)
    server_times: dict[str, list[float]] = defaultdict(list)
    for key in sorted(set(local_rows) & set(server_rows)):
        problem = key[2]
        local_time = numeric(local_rows[key].get("time"))
        server_time = numeric(server_rows[key].get("time"))
        if local_time is None or server_time is None or server_time <= 0.0:
            continue
        local_times[problem].append(local_time)
        server_times[problem].append(server_time)
    scales: dict[str, float] = {}
    for problem, values in local_times.items():
        server_values = server_times[problem]
        if values and server_values:
            scales[problem] = (sum(values) / len(values)) / (sum(server_values) / len(server_values))
    return scales


def normalize_row(
    row: dict[str, str],
    *,
    source_tree: str,
    source_run: str,
    source_csv: Path,
    scales: dict[str, float],
    basis_run: str,
) -> dict[str, str]:
    merged = {field: row.get(field, "") for field in BASE_FIELDS}
    original_time = row.get("time", "")
    scale = 1.0
    basis = "server-measured"
    if source_tree == "local":
        scale = scales.get(row.get("problem", ""), 1.0)
        basis = f"local-divided-by-{basis_run}-{row.get('problem', '')}-ratio"
        time_value = numeric(original_time)
        if time_value is not None and scale > 0.0:
            merged["time"] = f"{time_value / scale:.6f}"
    merged.update(
        {
            "original_time": original_time,
            "runtime_scale": f"{scale:.12g}",
            "runtime_scale_basis": basis,
            "source_tree": source_tree,
            "source_run": source_run,
            "source_csv": str(source_csv),
        }
    )
    return merged


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MERGED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} already exists; move or remove it before regenerating")
    scales = runtime_scales(args.local_root, args.server_root, args.basis_run)
    if not scales:
        raise SystemExit("Could not estimate local/server runtime scales")

    all_rows: list[dict[str, str]] = []
    per_run_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seen_server_rows: set[tuple[str, tuple[str, str, str, str, str]]] = set()

    for csv_path in result_csvs(args.server_root):
        source_run = str(csv_path.parent.relative_to(args.server_root))
        for row in read_csv(csv_path):
            merged = normalize_row(
                row,
                source_tree="server",
                source_run=source_run,
                source_csv=csv_path,
                scales=scales,
                basis_run=args.basis_run,
            )
            all_rows.append(merged)
            per_run_rows[("server", source_run)].append(merged)
            seen_server_rows.add((source_run, row_key(row)))

    skipped_local_duplicates = 0
    for csv_path in result_csvs(args.local_root):
        source_run = str(csv_path.parent.relative_to(args.local_root))
        for row in read_csv(csv_path):
            if (source_run, row_key(row)) in seen_server_rows:
                skipped_local_duplicates += 1
                continue
            merged = normalize_row(
                row,
                source_tree="local",
                source_run=source_run,
                source_csv=csv_path,
                scales=scales,
                basis_run=args.basis_run,
            )
            all_rows.append(merged)
            per_run_rows[("local", source_run)].append(merged)

    write_csv(args.out_dir / "results.csv", all_rows)
    for (source_tree, source_run), rows in sorted(per_run_rows.items()):
        write_csv(args.out_dir / source_tree / source_run / "results.csv", rows)

    summary = {
        "local_root": str(args.local_root),
        "server_root": str(args.server_root),
        "basis_run": args.basis_run,
        "runtime_scales_local_over_server": scales,
        "merged_rows": len(all_rows),
        "skipped_local_duplicates": skipped_local_duplicates,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
