#!/usr/bin/env python3
"""Evaluate PyVRP/HGS on the CVRPLIB survey set."""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from logging import getLogger
from pathlib import Path

import numpy as np


try:
    from pyvrp import read, solve
    from pyvrp.stop import MaxIterations, MaxRuntime
except ImportError as exc:
    raise SystemExit("pyvrp is required for this baseline. Run with the repo environment, e.g. .venv/bin/python.") from exc


HERE = Path(__file__).resolve()
SURVEY_ROOT = HERE.parents[3]


def cvrplib_header(filename: str) -> tuple[str, int, float | None, str | None]:
    name = Path(filename).stem
    dimension = 0
    edge_weight_type = None
    with open(filename, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("NAME"):
                name = line.strip().split()[-1]
            elif line.startswith("DIMENSION"):
                dimension = int(float(line.strip().split()[-1])) - 1
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.strip().split()[-1]

    cost = None
    sol_path = filename.replace(".vrp", ".sol")
    if os.path.exists(sol_path):
        with open(sol_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("Cost"):
                    cost = float(line.split()[-1])
                    break
    return name, dimension, cost, edge_weight_type


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def survey_scale_ranges() -> list[list[int]]:
    if "NRS_SURVEY_SIZE_LOW" in os.environ and "NRS_SURVEY_SIZE_HIGH" in os.environ:
        return [[int(os.environ["NRS_SURVEY_SIZE_LOW"]), int(os.environ["NRS_SURVEY_SIZE_HIGH"])]]
    return [[0, 1000], [1000, 10000], [10000, 100001]]


def round_func_for(edge_weight_type: str | None) -> str:
    if edge_weight_type == "EUC_2D":
        return "round"
    if edge_weight_type == "CEIL_2D":
        # PyVRP does not expose a ceil string in 0.13.x, but it accepts callables.
        return "ceil"
    return "round"


def pyvrp_read(filename: str, edge_weight_type: str | None):
    if edge_weight_type == "CEIL_2D":
        return read(filename, round_func=np.ceil)
    return read(filename, round_func=round_func_for(edge_weight_type))


def stopping_criterion():
    if "NRS_PYVRP_MAX_ITERATIONS" in os.environ:
        return MaxIterations(int(os.environ["NRS_PYVRP_MAX_ITERATIONS"]))
    return MaxRuntime(float(os.environ.get("NRS_PYVRP_MAX_RUNTIME", "1.0")))


class PyVRPCVRPTester:
    def __init__(self, tester_params):
        self.tester_params = tester_params
        self.logger = getLogger("trainer")
        self.gap_set_less_1000: list[float] = []
        self.gap_set_less_10000: list[float] = []
        self.gap_set_less_100000: list[float] = []
        self.time_sum_less_1000 = 0.0
        self.time_sum_less_10000 = 0.0
        self.time_sum_less_100000 = 0.0
        self.all_instance_num = 0
        self.all_solved_instance_num = 0

    def run_lib(self) -> None:
        filename = self.tester_params["filename"]
        start_time_all = time.time()

        for scale_range in survey_scale_ranges():
            self.logger.info("#################  Test scale range: %s  #################", scale_range)
            self._run_one_scale_range_lib(filename, scale_range)

        elapsed = time.time() - start_time_all
        avg_time = elapsed / self.all_solved_instance_num if self.all_solved_instance_num else 0.0
        self.logger.info(
            "All scale ranges done, solved instance number: %s/%s, total time: %.2fs, avg time per instance: %.2fs",
            self.all_solved_instance_num,
            self.all_instance_num,
            elapsed,
            avg_time,
        )
        self._log_bucket("[0, 1000)", self.gap_set_less_1000, self.time_sum_less_1000)
        self._log_bucket("[1000, 10000)", self.gap_set_less_10000, self.time_sum_less_10000)
        self._log_bucket("[10000, 100000]", self.gap_set_less_100000, self.time_sum_less_100000)

        all_gaps = self.gap_set_less_1000 + self.gap_set_less_10000 + self.gap_set_less_100000
        all_times_count = len(all_gaps)
        total_time = self.time_sum_less_1000 + self.time_sum_less_10000 + self.time_sum_less_100000
        self.logger.info("#######################################################")
        self.logger.info(
            "All solved instances, number: %s, avg GAP: %.3f%%, avg time: %.3fs",
            all_times_count,
            float(np.mean(all_gaps)) if all_gaps else 0.0,
            total_time / all_times_count if all_times_count else 0.0,
        )
        self.logger.info("#################  All Done  #################")

    def _log_bucket(self, label: str, gaps: list[float], time_sum: float) -> None:
        avg_gap = float(np.mean(gaps)) if gaps else 0.0
        avg_time = time_sum / len(gaps) if gaps else 0.0
        self.logger.info("%s, number: %s, avg GAP: %.3f%%, avg time: %.3fs", label, len(gaps), avg_gap, avg_time)

    def _run_one_scale_range_lib(self, filename: str, scale_range: list[int]) -> None:
        candidates = []
        for root, _dirs, files in os.walk(filename):
            for file in files:
                if not file.endswith(".vrp"):
                    continue
                vrp_path = os.path.join(root, file)
                name, dimension, optimal, edge_weight_type = cvrplib_header(vrp_path)
                if edge_weight_type not in {"EUC_2D", "CEIL_2D"}:
                    continue
                if not (scale_range[0] <= dimension < scale_range[1]):
                    continue
                if optimal is None:
                    self.logger.info("Instance %s: .sol not found or cost missing, skip.", name)
                    continue
                candidates.append((dimension, name, vrp_path, optimal, edge_weight_type))

        candidates.sort(key=lambda item: (item[0], item[1]))
        if env_flag("NRS_EVAL_DEBUG_SMALLEST") and candidates:
            candidates = candidates[:1]

        start_time_range = time.time()
        for dimension, name, vrp_path, optimal, edge_weight_type in candidates:
            self.all_instance_num += 1
            self.logger.info("===============================================================")
            self.logger.info("Instance name: %s, problem_size: %s, edge_weight: %s", name, dimension, edge_weight_type)

            inst_start = time.time()
            try:
                data = pyvrp_read(vrp_path, edge_weight_type)
                result = solve(
                    data,
                    stopping_criterion(),
                    seed=int(os.environ.get("NRS_PYVRP_SEED", "0")),
                    display=False,
                    collect_stats=False,
                )
                if not result.is_feasible():
                    raise ValueError("PyVRP returned an infeasible solution")
                score = float(result.cost())
            except Exception as exc:
                self.logger.info("Error occurred in instance %s, dimension: %s, skip it!", name, dimension)
                self.logger.info("Error message: %s", exc)
                continue

            inst_time = time.time() - inst_start
            self.all_solved_instance_num += 1
            gap = (score - optimal) * 100.0 / optimal

            if dimension < 1000:
                self.gap_set_less_1000.append(gap)
                self.time_sum_less_1000 += inst_time
            elif 1000 <= dimension < 10000:
                self.gap_set_less_10000.append(gap)
                self.time_sum_less_10000 += inst_time
            elif 10000 <= dimension <= 100000:
                self.gap_set_less_100000.append(gap)
                self.time_sum_less_100000 += inst_time
            else:
                raise ValueError(f"dimension should be less than 100000, but got {dimension}")

            self.logger.info(
                "Instance: %s, Dimension: %s, BKS: %.4f, HGS cost: %.3f, Gap: %.3f%%, Time: %.3fs",
                name,
                dimension,
                optimal,
                score,
                gap,
                inst_time,
            )

        elapsed = time.time() - start_time_range
        self.logger.info(
            "scale_range: %s, instance number: %s, total time: %.2fs",
            scale_range,
            len(candidates),
            elapsed,
        )
        self.logger.info("===============================================================")


def main() -> int:
    time_str = time.strftime("%Y%m%d_%H%M%S")
    rand_str = f"{random.randint(0, 9999):04d}"
    method_log_root = os.environ.get("NRS_METHOD_LOG_ROOT")
    if method_log_root:
        log_dir = Path(method_log_root) / "pyvrp_cvrp"
    else:
        log_dir = HERE.parent / "result_survey_cvrp_pyvrp" / f"{time_str}_{rand_str}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run_log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    logger = getLogger("trainer")
    logger.info("===== PyVRP/HGS CVRP Tester (ICAM-style log) =====")
    logger.info("Log directory: %s", log_dir)
    logger.info(
        "Stopping: iterations=%s, runtime=%ss",
        os.environ.get("NRS_PYVRP_MAX_ITERATIONS", ""),
        os.environ.get("NRS_PYVRP_MAX_RUNTIME", "1.0"),
    )

    tester = PyVRPCVRPTester(
        tester_params={
            "filename": os.environ.get(
                "NRS_SURVEY_CVRP_DIR",
                str(SURVEY_ROOT / "0_data_survey" / "survey_bench_cvrp"),
            )
        }
    )
    tester.run_lib()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
