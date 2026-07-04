#!/usr/bin/env python3
"""Evaluate HGS-CVRP through the PyHygese wrapper on the CVRPLIB survey set."""

from __future__ import annotations

import logging
import math
import os
import random
import re
import time
from logging import getLogger
from pathlib import Path

import numpy as np

try:
    from hygese import AlgorithmParameters, Solver
except ImportError as exc:
    raise SystemExit("hygese is required for this baseline. Install it in the repo environment.") from exc


HERE = Path(__file__).resolve()
SURVEY_ROOT = HERE.parents[3]


def cvrplib_instance(filename: str) -> tuple[str, int, list[list[float]], list[int], float, float | None, str | None]:
    name = Path(filename).stem
    dimension = 0
    capacity = 0.0
    edge_weight_type = None
    coords: list[list[float]] = []
    demands: list[int] = []
    section = ""
    with open(filename, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("NAME"):
                name = line.split()[-1]
            elif line.startswith("DIMENSION"):
                dimension = int(float(line.split()[-1])) - 1
            elif line.startswith("CAPACITY"):
                capacity = float(line.split()[-1])
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split()[-1]
            elif line.startswith("NODE_COORD_SECTION"):
                section = "coords"
            elif line.startswith("DEMAND_SECTION"):
                section = "demands"
            elif line.startswith("DEPOT_SECTION"):
                section = ""
            elif section == "coords":
                parts = line.split()
                if len(parts) >= 3:
                    coords.append([float(parts[1]), float(parts[2])])
            elif section == "demands":
                parts = line.split()
                if len(parts) >= 2:
                    demands.append(int(float(parts[-1])))

    optimal = None
    sol_path = filename.replace(".vrp", ".sol")
    if os.path.exists(sol_path):
        with open(sol_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("Cost"):
                    optimal = float(line.split()[-1])
                    break
    return name, dimension, coords, demands, capacity, optimal, edge_weight_type


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def survey_scale_ranges() -> list[list[int]]:
    if "NRS_SURVEY_SIZE_LOW" in os.environ and "NRS_SURVEY_SIZE_HIGH" in os.environ:
        return [[int(os.environ["NRS_SURVEY_SIZE_LOW"]), int(os.environ["NRS_SURVEY_SIZE_HIGH"])]]
    return [[0, 1000], [1000, 10000], [10000, 100001]]


def tsplib_distance_matrix(coords: np.ndarray, edge_weight_type: str | None) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    raw = np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float64)
    if edge_weight_type == "CEIL_2D":
        return np.ceil(raw)
    if edge_weight_type == "EUC_2D":
        return np.floor(raw + 0.5)
    return raw


def data_for_hygese(
    name: str,
    coords: list[list[float]],
    demands: list[int],
    capacity: float,
    edge_weight_type: str | None,
) -> dict[str, object]:
    arr = np.array(coords, dtype=np.float64)
    data: dict[str, object] = {
        "x_coordinates": arr[:, 0],
        "y_coordinates": arr[:, 1],
        "demands": demands,
        "vehicle_capacity": capacity,
        "depot": 0,
    }
    if edge_weight_type == "CEIL_2D":
        data["distance_matrix"] = tsplib_distance_matrix(arr, edge_weight_type)
    if env_flag("NRS_HYGESE_USE_NAME_FLEET"):
        match = re.search(r"-k(?P<k>\d+)$", name)
        if match:
            data["num_vehicles"] = int(match.group("k"))
    return data


def default_iteration_budget(size: int) -> int:
    return max(1, size // 3)


def solve_one(
    name: str,
    dimension: int,
    coords: list[list[float]],
    demands: list[int],
    capacity: float,
    edge_weight_type: str | None,
    nb_iter_override: int | None,
) -> tuple[float, int, int]:
    nb_iter = nb_iter_override if nb_iter_override is not None else default_iteration_budget(dimension)
    params = AlgorithmParameters(
        seed=int(os.environ.get("NRS_HYGESE_SEED", "0")),
        nbIter=nb_iter,
        timeLimit=float(os.environ.get("NRS_HYGESE_TIME_LIMIT", "3600")),
    )
    data = data_for_hygese(name, coords, demands, capacity, edge_weight_type)
    result = Solver(params, verbose=False).solve_cvrp(data, rounding=edge_weight_type != "CEIL_2D")
    if not math.isfinite(result.cost) or result.cost <= 0:
        raise ValueError(f"Hygese returned no feasible route, cost={result.cost}")
    return float(result.cost), int(result.n_routes), int(nb_iter)


class HygeseCVRPTester:
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
        self.progress_start_time = time.time()
        self.emitted_second = 0
        self.progress_gap_sum = 0.0
        nb_iter_override = os.environ.get("NRS_HYGESE_NB_ITER")
        self.nb_iter_override = int(nb_iter_override) if nb_iter_override is not None else None

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
        total_time = self.time_sum_less_1000 + self.time_sum_less_10000 + self.time_sum_less_100000
        self.logger.info("#######################################################")
        self.logger.info(
            "All solved instances, number: %s, avg GAP: %.3f%%, avg time: %.3fs",
            len(all_gaps),
            float(np.mean(all_gaps)) if all_gaps else 0.0,
            total_time / len(all_gaps) if all_gaps else 0.0,
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
                instance = cvrplib_instance(vrp_path)
                name, dimension, coords, demands, capacity, optimal, edge_weight_type = instance
                if edge_weight_type not in {"EUC_2D", "CEIL_2D"}:
                    continue
                if not (scale_range[0] <= dimension < scale_range[1]):
                    continue
                if optimal is None:
                    self.logger.info("Instance %s: .sol not found or cost missing, skip.", name)
                    continue
                candidates.append((dimension, name, coords, demands, capacity, optimal, edge_weight_type))

        candidates.sort(key=lambda item: (item[0], item[1]))
        if env_flag("NRS_EVAL_DEBUG_SMALLEST") and candidates:
            candidates = candidates[:1]

        start_time_range = time.time()
        for dimension, name, coords, demands, capacity, optimal, edge_weight_type in candidates:
            self.all_instance_num += 1
            self.logger.info("===============================================================")
            self.logger.info("Instance name: %s, problem_size: %s, edge_weight: %s", name, dimension, edge_weight_type)
            inst_start = time.time()
            try:
                score, route_count, nb_iter = solve_one(
                    name,
                    dimension,
                    coords,
                    demands,
                    capacity,
                    edge_weight_type,
                    self.nb_iter_override,
                )
            except Exception as exc:
                self.logger.info("Error occurred in instance %s, dimension: %s, skip it!", name, dimension)
                self.logger.info("Error message: %s", exc)
                continue
            inst_time = time.time() - inst_start
            self.all_solved_instance_num += 1
            gap = (score - optimal) * 100.0 / optimal
            self.progress_gap_sum += gap
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
            self.logger.info("Hygese routes: %s, nbIter: %s", route_count, nb_iter)
            current_second = max(1, int(time.time() - self.progress_start_time))
            while self.emitted_second < current_second:
                self.emitted_second += 1
                self.logger.info(
                    "[GAP_OVER_TIME] elapsed_second=%s solved_count=%s avg_gap=%.6f last_instance=%s last_size=%s",
                    self.emitted_second,
                    self.all_solved_instance_num,
                    self.progress_gap_sum / self.all_solved_instance_num,
                    name,
                    dimension,
                )

        elapsed = time.time() - start_time_range
        self.logger.info("scale_range: %s, instance number: %s, total time: %.2fs", scale_range, len(candidates), elapsed)
        self.logger.info("===============================================================")


def main() -> int:
    time_str = time.strftime("%Y%m%d_%H%M%S")
    rand_str = f"{random.randint(0, 9999):04d}"
    method_log_root = os.environ.get("NRS_METHOD_LOG_ROOT")
    if method_log_root:
        log_dir = Path(method_log_root) / "hygese_cvrp"
    else:
        log_dir = HERE.parent / "result_survey_cvrp_hygese" / f"{time_str}_{rand_str}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = getLogger("trainer")
    logger.info("===== Hygese/HGS CVRP Tester (ICAM-style log) =====")
    logger.info("Log directory: %s", log_dir)
    logger.info(
        "Hygese params: nbIter=%s, timeLimit=%s, seed=%s",
        os.environ.get("NRS_HYGESE_NB_ITER", "n//3 per instance"),
        os.environ.get("NRS_HYGESE_TIME_LIMIT", "3600"),
        os.environ.get("NRS_HYGESE_SEED", "0"),
    )
    tester = HygeseCVRPTester(
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
