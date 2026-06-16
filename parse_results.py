#!/usr/bin/env python3
"""Summarize STAR results.csv files by problem.

By default this scans every results.csv below results/. You can also pass one
or more result directories or CSV files explicitly.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_RESULTS_DIR = Path("results")
BENCH_ROOT = Path("survey/0_data_survey")
GROUP_FIELDS = ("problem", "strategy_id", "policy_id")
SUMMARY_FIELDS = (
    "source",
    "problem",
    "size_group",
    "strategy_id",
    "policy_id",
    "rows",
    "ok",
    "failed",
    "avg_gap",
    "avg_time",
    "total_time",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse STAR results.csv files and report average gap/time by problem."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "results.csv files or directories containing results.csv; "
            "defaults to every results.csv under results/"
        ),
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="combine all input CSVs before grouping instead of keeping each result directory separate",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="include failed/unsupported rows in gap/time averages when numeric values are present",
    )
    parser.add_argument(
        "--by-size-group",
        action="store_true",
        help="also group rows by instance size: <1K, [1K,10K), >=10K",
    )
    return parser.parse_args()


def result_csvs(paths: Iterable[Path]) -> list[Path]:
    candidates = list(paths)
    if not candidates:
        return sorted(DEFAULT_RESULTS_DIR.glob("**/results.csv"))

    csvs: list[Path] = []
    for path in candidates:
        if path.is_dir():
            csv_path = path / "results.csv"
            if csv_path.exists():
                csvs.append(csv_path)
            else:
                csvs.extend(sorted(path.glob("**/results.csv")))
        elif path.name == "results.csv":
            csvs.append(path)
        else:
            raise SystemExit(f"Not a results.csv file or result directory: {path}")
    return sorted(dict.fromkeys(csvs))


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_dimension_and_name(path: Path) -> tuple[int | None, str]:
    dimension: int | None = None
    name = path.stem
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("NAME"):
            name = line.split(":", 1)[-1].strip() if ":" in line else line.split()[-1].strip()
        elif line.startswith("DIMENSION"):
            value = line.split(":", 1)[-1] if ":" in line else line.split()[-1]
            try:
                dimension = int(value.strip())
            except ValueError:
                dimension = None
        elif line in {"NODE_COORD_SECTION", "DEMAND_SECTION"} and dimension is not None:
            break
    return dimension, name


def instance_dimensions() -> dict[tuple[str, str], int]:
    dimensions: dict[tuple[str, str], int] = {}
    for problem, directory, suffix in (
        ("tsp", BENCH_ROOT / "survey_bench_tsp", ".tsp"),
        ("cvrp", BENCH_ROOT / "survey_bench_cvrp", ".vrp"),
    ):
        for path in directory.glob(f"*{suffix}"):
            dimension, name = read_dimension_and_name(path)
            if dimension is None:
                continue
            dimensions[(problem, path.stem)] = dimension
            dimensions[(problem, name)] = dimension
    return dimensions


def size_group(dimension: int | None) -> str:
    if dimension is None:
        return "unknown"
    if dimension < 1000:
        return "<1K"
    if dimension < 10000:
        return "[1K,10K)"
    return ">=10K"


def summarize(
    rows: Iterable[dict[str, str]],
    *,
    include_all_rows: bool,
    by_size_group: bool,
    dimensions: dict[tuple[str, str], int],
) -> list[dict[str, str]]:
    group_fields = ("problem", "size_group", "strategy_id", "policy_id") if by_size_group else (
        "problem",
        "strategy_id",
        "policy_id",
    )
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        row = dict(row)
        if by_size_group:
            problem = row.get("problem", "")
            instance = row.get("instance", "")
            row["size_group"] = size_group(dimensions.get((problem, instance)))
        else:
            row["size_group"] = ""
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)

    summaries: list[dict[str, str]] = []
    for key, group_rows in sorted(groups.items()):
        averaged_rows = group_rows if include_all_rows else [
            row for row in group_rows if row.get("status") == "ok"
        ]
        gaps = [value for row in averaged_rows if (value := numeric(row.get("gap"))) is not None]
        times = [value for row in averaged_rows if (value := numeric(row.get("time"))) is not None]
        ok_count = sum(1 for row in group_rows if row.get("status") == "ok")
        failed_count = len(group_rows) - ok_count
        summaries.append(
            {
                "problem": key[0],
                "size_group": key[1] if by_size_group else "",
                "strategy_id": key[2] if by_size_group else key[1],
                "policy_id": key[3] if by_size_group else key[2],
                "rows": str(len(group_rows)),
                "ok": str(ok_count),
                "failed": str(failed_count),
                "avg_gap": f"{sum(gaps) / len(gaps):.6f}" if gaps else "n/a",
                "avg_time": f"{sum(times) / len(times):.6f}" if times else "n/a",
                "total_time": f"{sum(times):.6f}" if times else "n/a",
            }
        )
    return summaries


def print_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No rows to summarize.")
        return
    widths = {
        field: max(len(field), *(len(row.get(field, "")) for row in rows))
        for field in SUMMARY_FIELDS
    }
    print("  ".join(field.ljust(widths[field]) for field in SUMMARY_FIELDS))
    print("  ".join("-" * widths[field] for field in SUMMARY_FIELDS))
    for row in rows:
        print("  ".join(row.get(field, "").ljust(widths[field]) for field in SUMMARY_FIELDS))


def main() -> int:
    args = parse_args()
    csv_paths = result_csvs(args.paths)
    if not csv_paths:
        raise SystemExit("No results.csv files found.")
    dimensions = instance_dimensions() if args.by_size_group else {}

    output_rows: list[dict[str, str]] = []
    if args.combined:
        all_rows: list[dict[str, str]] = []
        for csv_path in csv_paths:
            all_rows.extend(read_rows(csv_path))
        for row in summarize(
            all_rows,
            include_all_rows=args.all_rows,
            by_size_group=args.by_size_group,
            dimensions=dimensions,
        ):
            output_rows.append({"source": "combined", **row})
    else:
        for csv_path in csv_paths:
            result_dir = str(csv_path.parent)
            rows = read_rows(csv_path)
            for row in summarize(
                rows,
                include_all_rows=args.all_rows,
                by_size_group=args.by_size_group,
                dimensions=dimensions,
            ):
                output_rows.append({"source": result_dir, **row})

    print_table(output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
