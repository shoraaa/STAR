#!/usr/bin/env python3
"""Evaluate LKH-3 through the PyLKH wrapper on the TSPLIB survey set."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytz


HERE = Path(__file__).resolve()
SURVEY_ROOT = HERE.parents[3]
REPO_ROOT = SURVEY_ROOT.parent
NN_TSP_DIR = SURVEY_ROOT / "heuristics" / "nearest neighbor" / "TSP"
sys.path.insert(0, str(NN_TSP_DIR))

try:
    import lkh
except ImportError as exc:
    raise SystemExit("lkh is required for this baseline. Install it in the repo environment.") from exc

from LIBUtils import TSPLIBReader, tsplib_cost  # noqa: E402
from nearest_neighbor_survey import read_tsplib_dimension, tsplib_total_distance  # noqa: E402


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def selected_tsp_files(lib_path: str) -> list[str]:
    low = int(os.environ.get("NRS_SURVEY_SIZE_LOW", "0"))
    high = int(os.environ.get("NRS_SURVEY_SIZE_HIGH", "100001"))
    candidates: list[tuple[int, str, str]] = []
    for root, _dirs, files in os.walk(lib_path):
        for fname in files:
            if not fname.endswith(".tsp"):
                continue
            tsp_path = os.path.join(root, fname)
            dimension = read_tsplib_dimension(tsp_path)
            if dimension is None or not (low <= dimension < high):
                continue
            candidates.append((dimension, fname, tsp_path))
    candidates.sort()
    if env_flag("NRS_EVAL_DEBUG_SMALLEST") and candidates:
        return [candidates[0][2]]
    return [path for _dimension, _fname, path in candidates]


def lkh_binary() -> str:
    configured = os.environ.get("NRS_LKH_EXECUTABLE")
    if configured:
        return configured
    return str(REPO_ROOT / "survey" / "heuristics" / "LKH-3.0.14" / "LKH")


def solve_one_tsplib_instance(
    tsp_path: str,
    solver: str,
    max_trials: int,
    runs: int,
    time_limit: int,
) -> dict[str, object] | None:
    name, dimension, locs, edge_weight_type = TSPLIBReader(tsp_path)
    if name is None:
        return None
    optimal = tsplib_cost.get(name)
    if optimal is None:
        raise ValueError(f"optimal value (BKS) of instance {name} not found in tsplib_cost")

    start_time = time.time()
    routes = lkh.solve(
        solver=solver,
        problem_file=tsp_path,
        max_trials=max_trials,
        runs=runs,
        time_limit=time_limit,
    )
    elapsed = time.time() - start_time
    if not routes:
        raise ValueError("LKH returned no tour")

    # PyLKH exposes TSPLIB node IDs as 1-based integers.
    tour = [int(node) - 1 for node in routes[0]]
    if tour[0] != tour[-1]:
        tour.append(tour[0])

    unique = np.unique(np.array(tour[:-1], dtype=int))
    if len(unique) != dimension:
        raise ValueError(f"tour invalid: visited {len(unique)} unique nodes, expected {dimension}")

    cities = np.array(locs, dtype=np.float64)
    total_distance = tsplib_total_distance(cities, tour, edge_weight_type)
    bks = float(optimal)
    gap = (total_distance - bks) * 100.0 / bks
    return {
        "name": name,
        "dimension": int(dimension),
        "bks": bks,
        "cost": float(total_distance),
        "gap": float(gap),
        "time": float(elapsed),
    }


def bucket_label(lo: int, hi: int) -> str:
    if (lo, hi) == (0, 1000):
        return "[0, 1000)"
    if (lo, hi) == (1000, 10000):
        return "[1000, 10000)"
    return "[10000, 100000]"


def main() -> int:
    lib_path = os.environ.get("NRS_SURVEY_TSP_DIR", str(SURVEY_ROOT / "0_data_survey" / "survey_bench_tsp"))
    solver = lkh_binary()
    max_trials = int(os.environ.get("NRS_LKH_MAX_TRIALS", "1000000"))
    runs = int(os.environ.get("NRS_LKH_RUNS", "1"))
    time_limit = int(os.environ.get("NRS_LKH_TIME_LIMIT", "1000"))
    scale_ranges = [(0, 1000), (1000, 10000), (10000, 100001)]
    bucket_stats = {rng: {"gaps": [], "times": []} for rng in scale_ranges}
    all_gaps: list[float] = []
    all_times: list[float] = []

    tz = pytz.timezone("Asia/Shanghai")
    process_start_time = datetime.now(tz)
    method_log_root = os.environ.get("NRS_METHOD_LOG_ROOT")
    if method_log_root:
        log_dir = Path(method_log_root) / "lkh_tsp"
    else:
        log_dir = HERE.parent / "result_survey_tsp_lkh" / f"{process_start_time:%Y%m%d_%H%M%S}_LKH_TSPLIB"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run_log.txt"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    tsp_files = selected_tsp_files(lib_path)
    log(f"Total TSPLIB instances detected: {len(tsp_files)}")
    log("#################  LKH Test on TSPLIB_Survey  #################")
    log(f"TSPLIB folder: {lib_path}")
    log(f"LKH executable: {solver}")
    log(f"LKH max_trials: {max_trials}, runs: {runs}, time_limit: {time_limit}s")
    log("-----------------------------------------------------------------")

    total_start = time.time()
    solved = 0
    emitted_second = 0
    gap_sum = 0.0
    for idx, tsp_path in enumerate(tsp_files, start=1):
        try:
            result = solve_one_tsplib_instance(
                tsp_path,
                solver=solver,
                max_trials=max_trials,
                runs=runs,
                time_limit=time_limit,
            )
            if result is None:
                log(f"[SKIP {idx}/{len(tsp_files)}] {tsp_path}, unsupported format")
                continue
        except Exception as exc:
            log(f"[SKIP {idx}/{len(tsp_files)}] {tsp_path}, error: {exc}")
            continue

        solved += 1
        gap = float(result["gap"])
        elapsed = float(result["time"])
        dim = int(result["dimension"])
        gap_sum += gap
        all_gaps.append(gap)
        all_times.append(elapsed)
        for rng in scale_ranges:
            lo, hi = rng
            if lo <= dim < hi:
                bucket_stats[rng]["gaps"].append(gap)
                bucket_stats[rng]["times"].append(elapsed)
                break
        log(
            f"[{idx}/{len(tsp_files)}] Instance: {result['name']}, dim: {dim}, "
            f"BKS: {result['bks']:.0f}, LKH cost: {result['cost']:.0f}, "
            f"GAP: {gap:.3f}%, time: {elapsed:.3f}s"
        )
        current_second = max(1, int(time.time() - total_start))
        while emitted_second < current_second:
            emitted_second += 1
            log(
                f"[GAP_OVER_TIME] elapsed_second={emitted_second} "
                f"solved_count={solved} avg_gap={gap_sum / solved:.6f} "
                f"last_instance={result['name']} last_size={dim}"
            )

    total_time = time.time() - total_start
    log("-----------------------------------------------------------------")
    log(f"All instances found: {len(tsp_files)}, solved: {solved}")
    log(f"Total time: {total_time:.2f}s, avg time per solved instance: {(total_time / solved) if solved else 0:.2f}s")
    log("#################  Bucket Summary  #################")
    for rng in scale_ranges:
        gaps = bucket_stats[rng]["gaps"]
        times = bucket_stats[rng]["times"]
        avg_gap = float(np.mean(gaps)) if gaps else 0.0
        avg_time = float(np.mean(times)) if times else 0.0
        log(f"{bucket_label(*rng)}, number: {len(gaps)}, avg GAP: {avg_gap:.3f}%, avg time: {avg_time:.3f}s")

    avg_all_gap = float(np.mean(all_gaps)) if all_gaps else 0.0
    avg_all_time = float(np.mean(all_times)) if all_times else 0.0
    log("###################################  Overall Summary  ##########################################")
    log(f"All solved instances, number: {len(all_gaps)}, avg GAP: {avg_all_gap:.3f}%, avg time: {avg_all_time:.3f}s")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
