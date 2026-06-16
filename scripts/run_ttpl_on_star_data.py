#!/usr/bin/env python3
"""Run the cloned TTPL testers on STAR-exported benchmark lists."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_ttpl_data import (  # noqa: E402
    TTPL_CVRP_TEST_DIR,
    TTPL_TSP_TEST_DIR,
    _select_paths,
    _write_cvrp,
    _write_tsp,
)

TTPL_ROOT = ROOT / "TTPL" / "TTPL"
DEFAULT_RESULTS_DIR = ROOT / "results_ttpl"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _run_slug(args: argparse.Namespace, timestamp: str) -> str:
    mode = "mvdf" if args.mvdf else "nomvdf"
    start = "random" if args.random_insertion else "model"
    return (
        f"{timestamp}-ttpl-{args.problem}-{args.size}-{mode}-{start}-"
        f"rrc{args.rrc_budget}-range{args.rrc_range}"
    )


def _default_out_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RESULTS_DIR / _run_slug(args, timestamp)


@contextlib.contextmanager
def _stream_to_log(log_path: Path | None):
    if log_path is None:
        yield None
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        stdout = Tee(sys.stdout, log_file)
        stderr = Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"streaming log to {log_path}")
            yield log_path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ttpl_relative_data_path(problem: str, data_path: Path) -> str:
    base = TTPL_ROOT / "lehd" / ("TSP" if problem == "tsp" else "CVRP") / "data"
    return str(data_path.resolve().relative_to(base.resolve()))


def _prepare_data_for_size(problem: str, size: str, limit: int | None) -> tuple[Path, int]:
    paths = _select_paths(problem, size, limit)
    if not paths:
        raise SystemExit(f"no {problem} instances found for size={size}")

    if problem == "tsp":
        output = TTPL_TSP_TEST_DIR / f"STAR_tsplib_{size}_n{len(paths)}.txt"
        _write_tsp(paths, output)
    else:
        output = TTPL_CVRP_TEST_DIR / f"STAR_vrplib_{size}_n{len(paths)}.txt"
        _write_cvrp(paths, output)
    print(f"prepared {len(paths)} {problem.upper()} instances at {output}")
    return output, len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=["tsp", "cvrp"])
    parser.add_argument(
        "--size",
        choices=["dev", "dev-medium3", "dev-medium", "small", "medium", "large"],
        default="medium",
        help="STAR size bucket to export and evaluate when --data is not provided.",
    )
    parser.add_argument("--limit", type=int, default=None, help="optional prefix limit for quick smoke runs")
    parser.add_argument("--data", type=Path, default=None, help="existing TTPL-format data file")
    parser.add_argument("--episodes", type=int, default=None, help="number of rows to evaluate")
    parser.add_argument("--cuda-device-num", type=int, default=0)
    parser.add_argument("--projection", default=None, help="projection function name")
    parser.add_argument("--rrc-budget", type=int, default=0)
    parser.add_argument("--rrc-range", type=int, default=1000)
    parser.add_argument("--mvdf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-insertion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--k-nearest-nodes", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="run output directory; defaults to timestamped results_ttpl/<run>/",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="path for streamed stdout/stderr log; defaults to <out-dir>/run.log",
    )
    parser.add_argument("--no-log", action="store_true", help="disable automatic streamed log file")
    args = parser.parse_args()

    args.out_dir = (args.out_dir or _default_out_dir(args)).resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = None if args.no_log else (args.log_path or args.out_dir / "run.log")
    with _stream_to_log(log_path):
        print(f"output directory: {args.out_dir}")
        _run(args)


def _run(args: argparse.Namespace) -> None:
    if args.data is None:
        data_path, episodes = _prepare_data_for_size(args.problem, args.size, args.limit)
    else:
        data_path = args.data
        if args.episodes is None:
            raise SystemExit("--episodes is required when --data is provided")
        episodes = args.episodes
    data_path = data_path.resolve()

    if args.problem == "tsp":
        script = TTPL_ROOT / "lehd" / "TSP" / "test_tsp.py"
        module = _load_module("_ttpl_test_tsp", script)
        data_rel = _ttpl_relative_data_path("tsp", data_path)
        module.test_paras[0] = [data_rel, episodes, 1]
        tt_args = SimpleNamespace(
            cuda_device_num=args.cuda_device_num,
            problem_size=0,
            test_in_tsplib=True,
            RRC_budget=args.rrc_budget,
            RRC_range=args.rrc_range,
            random_insertion=args.random_insertion,
            knearest=True,
            k_nearest_nodes=args.k_nearest_nodes,
            coor_projection=True,
            counter_current=0,
            projection=args.projection or "projection_5k",
            MVDF=args.mvdf,
            model_load_epoch=150,
            model_load_path="result/TSP100_model",
        )
    else:
        script = TTPL_ROOT / "lehd" / "CVRP" / "test_cvrp.py"
        module = _load_module("_ttpl_test_cvrp", script)
        data_rel = _ttpl_relative_data_path("cvrp", data_path)
        module.test_paras[0] = [data_rel, episodes, 1, 0]
        tt_args = SimpleNamespace(
            cuda_device_num=args.cuda_device_num,
            problem_size=0,
            test_in_vrplib=True,
            RRC_budget=args.rrc_budget,
            RRC_range=args.rrc_range,
            random_insertion=args.random_insertion,
            knearest=True,
            k_nearest_nodes=args.k_nearest_nodes,
            coor_projection=True,
            counter_current=0,
            projection=args.projection or "projection_10k",
            MVDF=args.mvdf,
            model_load_epoch=40,
            model_load_path="result/CVRP100_model",
        )

    module.main_test(tt_args)


if __name__ == "__main__":
    main()
