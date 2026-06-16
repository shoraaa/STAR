#!/usr/bin/env python3
"""Prepare STAR benchmark instances in the text formats expected by TTPL.

The TTPL test scripts do not read TSPLIB/VRPLIB files directly.  Their
``--test_in_tsplib`` and ``--test_in_vrplib`` modes read one-line summary files
under ``lehd/*/data/test``.  This script converts the STAR survey benchmark
files into those TTPL summary files without modifying the raw benchmark files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from STAR.core import CVRP_BENCH_DIR, TSP_BENCH_DIR, instance_paths, load_cvrp, load_tsp


TTPL_ROOT = ROOT / "TTPL" / "TTPL"
TTPL_TSP_TEST_DIR = TTPL_ROOT / "lehd" / "TSP" / "data" / "test"
TTPL_CVRP_TEST_DIR = TTPL_ROOT / "lehd" / "CVRP" / "data" / "test"


def _select_paths(problem: str, size: str, limit: int | None) -> list[Path]:
    paths = instance_paths(problem, size)
    if limit is not None:
        paths = paths[:limit]
    return paths


def _write_tsp(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for path in paths:
            instance = load_tsp(path)
            if instance.bks_cost is None:
                raise ValueError(f"missing BKS cost for {path}")
            coords = []
            for node in sorted(instance.coords):
                x, y = instance.coords[node]
                coords.extend([str(float(x)), str(float(y))])
            row = [instance.name, str(float(instance.bks_cost)), *coords]
            handle.write(",".join(row) + "\n")


def _write_cvrp(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for path in paths:
            instance = load_cvrp(path)
            if instance.bks_cost is None:
                raise ValueError(f"missing BKS cost for {path}")
            if instance.capacity is None:
                raise ValueError(f"missing capacity for {path}")

            depot = instance.depot
            customers = [node for node in sorted(instance.coords) if node != depot]
            depot_x, depot_y = instance.coords[depot]
            customer_coords = []
            for node in customers:
                x, y = instance.coords[node]
                customer_coords.extend([str(float(x)), str(float(y))])
            demands = [str(int(instance.demands.get(depot, 0)))]
            demands.extend(str(int(instance.demands[node])) for node in customers)

            row = [
                "['name'",
                f"'{instance.name}'",
                "'depot'",
                str(float(depot_x)),
                str(float(depot_y)),
                "'customer'",
                *customer_coords,
                "'demand'",
                *demands,
                "'capacity'",
                str(int(instance.capacity)),
                "'cost'",
                str(float(instance.bks_cost)),
            ]
            handle.write(", ".join(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=["tsp", "cvrp"])
    parser.add_argument(
        "--size",
        choices=["dev", "dev-medium3", "dev-medium", "small", "medium", "large"],
        default="medium",
        help="STAR size bucket to export.",
    )
    parser.add_argument("--limit", type=int, default=None, help="optional prefix limit")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output file; defaults to TTPL's lehd/<problem>/data/test directory",
    )
    args = parser.parse_args()

    paths = _select_paths(args.problem, args.size, args.limit)
    if not paths:
        source_dir = TSP_BENCH_DIR if args.problem == "tsp" else CVRP_BENCH_DIR
        raise SystemExit(f"no {args.problem} instances found for size={args.size} in {source_dir}")

    if args.output is None:
        default_dir = TTPL_TSP_TEST_DIR if args.problem == "tsp" else TTPL_CVRP_TEST_DIR
        suffix = "tsplib" if args.problem == "tsp" else "vrplib"
        output = default_dir / f"STAR_{suffix}_{args.size}_n{len(paths)}.txt"
    else:
        output = args.output

    if args.problem == "tsp":
        _write_tsp(paths, output)
    else:
        _write_cvrp(paths, output)

    print(f"wrote {len(paths)} {args.problem.upper()} instances to {output}")


if __name__ == "__main__":
    main()
