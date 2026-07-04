#!/usr/bin/env python3
"""Run original NRS survey entrypoints and normalize their logs to CSV."""

from __future__ import annotations

import argparse
import csv
import os
import re
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
SURVEY = ROOT / "survey" / "NRS"
SURVEY_DATA = ROOT / "survey" / "0_data_survey"
SURVEY_TSP_DIR = SURVEY_DATA / "survey_bench_tsp"
SURVEY_CVRP_DIR = SURVEY_DATA / "survey_bench_cvrp"


@dataclass(frozen=True)
class OriginalJob:
    method: str
    problem: str
    script: Path
    script_args: tuple[str, ...] = ()
    dev_bounds: tuple[int, int] | None = None

    @property
    def job_id(self) -> str:
        return f"{self.method}_{self.problem}"


JOBS: tuple[OriginalJob, ...] = (
    OriginalJob("nn", "tsp", ROOT / "survey/heuristics/nearest neighbor/TSP/nearest_neighbor_survey.py"),
    OriginalJob("nn", "cvrp", ROOT / "survey/heuristics/nearest neighbor/CVRP/nearest_neighbor_survey_cvrp.py"),
    OriginalJob("elkai", "tsp", ROOT / "survey/heuristics/elkai/TSP/elkai_survey.py"),
    OriginalJob("pyvrp", "cvrp", ROOT / "survey/heuristics/pyvrp/CVRP/pyvrp_survey_cvrp.py"),
    OriginalJob("lkh", "tsp", ROOT / "survey/heuristics/lkh/TSP/lkh_survey.py"),
    OriginalJob("hygese", "cvrp", ROOT / "survey/heuristics/hygese/CVRP/hygese_survey_cvrp.py"),
    OriginalJob("bq", "tsp", SURVEY / "Construction/single-stage/appending/1_BQ/test_tsp_survey.py", ("--seed", "0")),
    OriginalJob("bq", "cvrp", SURVEY / "Construction/single-stage/appending/1_BQ/test_cvrp_survey.py", ("--seed", "0")),
    OriginalJob("lehd", "tsp", SURVEY / "Construction/single-stage/appending/2_LEHD/TSP/test_survey.py"),
    OriginalJob("lehd", "cvrp", SURVEY / "Construction/single-stage/appending/2_LEHD/CVRP/test_cvrp_survey.py"),
    OriginalJob("sil", "tsp", SURVEY / "Construction/single-stage/appending/3_SIL/TSP/Test_All/test_survey.py", dev_bounds=(1000, 10000)),
    OriginalJob("sil", "cvrp", SURVEY / "Construction/single-stage/appending/3_SIL/CVRP/Test_All/test_survey.py", dev_bounds=(1000, 10000)),
    OriginalJob(
        "lehd_rrc",
        "tsp",
        SURVEY / "Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/1_LEHD/TSP/test_survey.py",
    ),
    OriginalJob(
        "lehd_rrc",
        "cvrp",
        SURVEY / "Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/1_LEHD/CVRP/test_cvrp_survey.py",
    ),
    OriginalJob(
        "sil_prc",
        "tsp",
        SURVEY / "Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/2_SIL/TSP/Test_All/test_survey.py",
        dev_bounds=(1000, 10000),
    ),
    OriginalJob(
        "sil_prc",
        "cvrp",
        SURVEY / "Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/2_SIL/CVRP/Test_All/test_survey.py",
        dev_bounds=(1000, 10000),
    ),
    OriginalJob(
        "drhg",
        "tsp",
        SURVEY / "Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/3_DRHG/TSP/test_survey.py",
        dev_bounds=(1000, 10000),
    ),
    OriginalJob(
        "drhg",
        "cvrp",
        SURVEY / "Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/3_DRHG/CVRP/test_cvrp_survey.py",
        dev_bounds=(1000, 10000),
    ),
    OriginalJob(
        "dgl",
        "tsp",
        SURVEY / "Construction/single-stage/appending/8_DGL/TSP/test_survey.py",
    ),
    OriginalJob(
        "l2c_insert",
        "tsp",
        SURVEY / "Construction/single-stage/insertion/L2C_Insert/L2C_Insert/TSP/Test/test_survey.py",
    ),
    OriginalJob(
        "gfacs",
        "tsp",
        SURVEY / "Improvement/population_based/GFACS/tsp_nls/test_survey.py",
    ),
    OriginalJob(
        "gensco",
        "tsp",
        SURVEY / "Improvement/single_solution_based/large neighborhood/direct LNS (non-restricted)/GenSCO/test_survey.py",
    ),
    OriginalJob(
        "fast_t2t",
        "tsp",
        SURVEY / "Improvement/single_solution_based/large neighborhood/indirect LNS/Fast-T2T_new/test_survey.py",
    ),
    OriginalJob(
        "l2c_insert_imp",
        "tsp",
        SURVEY / "Improvement/single_solution_based/large neighborhood/direct LNS (non-restricted)/L2C_Insert/L2C_Insert/TSP/Test/test_survey.py",
    ),
    OriginalJob(
        "icam",
        "tsp",
        SURVEY / "Construction/single-stage/appending/4_ICAM/ICAM_TSP/tsp_test_lib.py",
    ),
    OriginalJob(
        "icam",
        "cvrp",
        SURVEY / "Construction/single-stage/appending/4_ICAM/ICAM_CVRP/cvrp_test_lib.py",
    ),
    OriginalJob(
        "elg",
        "tsp",
        SURVEY / "Construction/single-stage/appending/5_ELG/TSP/test_tsplib_survey.py",
    ),
    OriginalJob(
        "elg",
        "cvrp",
        SURVEY / "Construction/single-stage/appending/5_ELG/CVRP/test_vrplib_survey.py",
    ),
    OriginalJob(
        "invit",
        "tsp",
        SURVEY / "Construction/single-stage/appending/6_INViT/train_survey_3v.py",
        ("--problem", "tsp", "--nb_epochs", "0", "--checkpoint_model", "24-04-11--16-19-47-n100-gpu0"),
    ),
    OriginalJob(
        "invit",
        "cvrp",
        SURVEY / "Construction/single-stage/appending/6_INViT/train_survey_3v.py",
        ("--problem", "cvrp", "--nb_epochs", "0", "--checkpoint_model", "24-04-11--16-55-30-n100-gpu0"),
    ),
    OriginalJob(
        "l2r",
        "tsp",
        SURVEY / "Construction/single-stage/appending/7_L2R/TSP/tsp_test_lib.py",
    ),
    OriginalJob(
        "l2r",
        "cvrp",
        SURVEY / "Construction/single-stage/appending/7_L2R/CVRP/cvrp_test_main_lib_survey.py",
    ),
    OriginalJob(
        "reld",
        "cvrp",
        SURVEY / "Construction/single-stage/appending/9_ReLD/CVRP/test_cvrplib_survey.py",
    ),
    OriginalJob(
        "neuopt",
        "tsp",
        SURVEY / "Improvement/single_solution_based/small neighborhood/sequential/NeuOpt/neuopt_survey_lib.py",
        ("--problem", "tsp"),
    ),
    OriginalJob(
        "neuopt",
        "cvrp",
        SURVEY / "Improvement/single_solution_based/small neighborhood/sequential/NeuOpt/neuopt_survey_lib.py",
        ("--problem", "cvrp"),
    ),
)


INSTANCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\[Timeout\]\s*Instance\s+(?P<name>[^\s]+)\s+\(n=(?P<size>\d+)\)\s*\|\s*"
        r"opt=(?P<opt>[-+0-9.eE]+|OOM|NaN|inf)\s*\|\s*time=(?P<time>[-+0-9.eE]+)s?",
        re.IGNORECASE,
    ),
    re.compile(
        r"Instance:\s*(?P<name>[^,\n]+),\s*Dimension:\s*(?P<size>\d+),\s*"
        r"Cost:\s*(?P<cost>[-+0-9.eE]+|OOM|NaN|inf),\s*Optimal:\s*(?P<opt>[-+0-9.eE]+|OOM|NaN|inf),\s*"
        r"Gap:\s*(?P<gap>[-+0-9.eE]+|OOM|NaN|inf)%?,\s*Time:\s*(?P<time>[-+0-9.eE]+|OOM|NaN|inf)s?"
    ),
    re.compile(
        r"Instance:\s*(?P<name>[^,\n]+),\s*size:\s*(?P<size>\d+),\s*"
        r"opt:\s*(?P<opt>[-+0-9.eE]+|OOM|NaN|inf),\s*student:\s*(?P<cost>[-+0-9.eE]+|OOM|NaN|inf),\s*"
        r"gap:\s*(?P<gap>[-+0-9.eE]+|OOM|NaN|inf)%?,\s*time:\s*(?P<time>[-+0-9.eE]+|OOM|NaN|inf)s?",
        re.IGNORECASE,
    ),
    re.compile(
        r"Dim:\s*(?P<size>\d+),\s*Teacher:\s*(?P<opt>[-+0-9.eE]+|OOM|NaN|inf),\s*"
        r"Student:\s*(?P<cost>[-+0-9.eE]+|OOM|NaN|inf),\s*Gap:\s*(?P<gap>[-+0-9.eE]+|OOM|NaN|inf)%?,\s*"
        r"Time:\s*(?P<time>[-+0-9.eE]+|OOM|NaN|inf)s?"
    ),
    re.compile(
        r"\[Inst\]\s*(?P<name>[^\s]+)\s*\(n=(?P<size>\d+)\)\s*"
        r"opt=(?P<opt>[-+0-9.eE]+|OOM|NaN|inf),\s*stu=(?P<cost>[-+0-9.eE]+|OOM|NaN|inf),\s*"
        r"gap=(?P<gap>[-+0-9.eE]+|OOM|NaN|inf)%?,\s*time=(?P<time>[-+0-9.eE]+|OOM|NaN|inf)s?"
    ),
    re.compile(
        r"Instance:\s*(?P<name>[^,\n]+),\s*Cost:\s*(?P<cost>[-+0-9.eE]+|OOM|NaN|inf),\s*"
        r"Optimal:\s*(?P<opt>[-+0-9.eE]+|OOM|NaN|inf),\s*Gap:\s*(?P<gap>[-+0-9.eE]+|OOM|NaN|inf)%?,\s*"
        r"Time:\s*(?P<time>[-+0-9.eE]+|OOM|NaN|inf)s?"
    ),
    re.compile(
        r"\[OK\]\s*inst:\s*(?P<name>[^,\n]+),\s*n:\s*(?P<size>\d+),\s*"
        r"gap:\s*(?P<gap>[-+0-9.eE]+|OOM|NaN|inf)%?,\s*time:\s*(?P<time>[-+0-9.eE]+|OOM|NaN|inf)s?",
        re.IGNORECASE,
    ),
    re.compile(
        r"Instance\s+(?P<name>[^\s|]+)\s+\(n=(?P<size>\d+)\)\s*\|\s*"
        r"optimal=(?P<opt>[-+0-9.eE]+|OOM|NaN|inf)\s*\|\s*score=(?P<cost>[-+0-9.eE]+|OOM|NaN|inf)\s*\|\s*"
        r"gap=(?P<gap>[-+0-9.eE]+|OOM|NaN|inf)%?\s*\|\s*time=(?P<time>[-+0-9.eE]+|OOM|NaN|inf)s?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\[(?P<name>[^\]]+)\]\s*score=(?P<cost>[-+0-9.eE]+|OOM|NaN|inf),\s*"
        r"opt=(?P<opt>[-+0-9.eE]+|OOM|NaN|inf),\s*gap=(?P<gap>[-+0-9.eE]+|OOM|NaN|inf)%?,\s*"
        r"time=(?P<time>[-+0-9.eE]+|OOM|NaN|inf)s?",
        re.IGNORECASE,
    ),
    re.compile(
        r"Instance:\s*(?P<name>[^:]+):\s*Size:\s*(?P<size>\d+),\s*Score:\s*(?P<cost>[-+0-9.eE]+|OOM|NaN|inf),\s*Gap:\s*(?P<gap>[-+0-9.eE]+|OOM|NaN|inf)%?,\s*Time:\s*(?P<time>[-+0-9.eE]+|OOM|NaN|inf)s?",
        re.IGNORECASE,
    ),
    re.compile(
        r"Instance:\s*(?P<name>[^,]+),\s*dim:\s*(?P<size>\d+),\s*teacher:\s*(?P<opt>[-+0-9.eE]+|OOM|NaN|inf),\s*student:\s*(?P<cost>[-+0-9.eE]+|OOM|NaN|inf),\s*gap:\s*(?P<gap>[-+0-9.eE]+|OOM|NaN|inf)%?,\s*time:\s*(?P<time>[-+0-9.eE]+|OOM|NaN|inf)s?",
        re.IGNORECASE,
    ),
    re.compile(
        r"Instance\s+(?P<name>[^,]+),\s*dim=(?P<size>\d+),\s*length=(?P<cost>[-+0-9.eE]+|OOM|NaN|inf),\s*optimal=(?P<opt>[-+0-9.eE]+|OOM|NaN|inf),\s*gap=(?P<gap>[-+0-9.eE]+|OOM|NaN|inf)%?,\s*time=(?P<time>[-+0-9.eE]+|OOM|NaN|inf)s?",
        re.IGNORECASE,
    ),
    re.compile(
        r"Instance:\s*(?P<name>[^,]+),\s*dim:\s*(?P<size>\d+),\s*BKS:\s*(?P<opt>[-+0-9.eE]+|OOM|NaN|inf),\s*"
        r"(?:NN cost|LKH cost|HGS cost|cost):\s*(?P<cost>[-+0-9.eE]+|OOM|NaN|inf),\s*"
        r"GAP:\s*(?P<gap>[-+0-9.eE]+|OOM|NaN|inf)%?,\s*time:\s*(?P<time>[-+0-9.eE]+|OOM|NaN|inf)s?",
        re.IGNORECASE,
    ),
    re.compile(
        r"Instance:\s*(?P<name>[^,]+),\s*Dimension:\s*(?P<size>\d+),\s*BKS:\s*(?P<opt>[-+0-9.eE]+|OOM|NaN|inf),\s*"
        r"(?:NN cost|LKH cost|HGS cost|cost):\s*(?P<cost>[-+0-9.eE]+|OOM|NaN|inf),\s*"
        r"Gap:\s*(?P<gap>[-+0-9.eE]+|OOM|NaN|inf)%?,\s*Time:\s*(?P<time>[-+0-9.eE]+|OOM|NaN|inf)s?",
        re.IGNORECASE,
    ),
    re.compile(
        r"Instance:\s*(?P<name>[^,]+),\s*dim:\s*(?P<size>\d+),\s*optimal cost:\s*(?P<opt>[-+0-9.eE]+|OOM|NaN|inf),\s*Pred cost:\s*(?P<cost>[-+0-9.eE]+|OOM|NaN|inf),\s*Gap:\s*(?P<gap>[-+0-9.eE]+|OOM|NaN|inf)%?,\s*Time:\s*(?P<time>[-+0-9.eE]+|OOM|NaN|inf)s?",
        re.IGNORECASE,
    ),
)

FAST_T2T_INSTANCE_PATTERN = re.compile(
    r"Instance name:\s*(?P<name>[^,\n]+),\s*problem_size:\s*(?P<size>\d+).*?"
    r"Instance name:\s*(?P=name),\s*optimal score:\s*(?P<opt>[-+0-9.eE]+).*?"
    r"No aug score:\s*(?P<cost>[-+0-9.eE]+),\s*No aug gap:\s*(?P<gap>[-+0-9.eE]+)%.*?"
    r"Instance time \(incl\. dist\+test\):\s*(?P<time>[-+0-9.eE]+)s",
    re.DOTALL,
)

SUMMARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\[(?P<bucket>[^\]]+)\]\s*count:\s*(?P<count>\d+),\s*"
        r"mean time:\s*(?P<time>[-+0-9.eE]+)s,\s*mean gap:\s*(?P<gap>[-+0-9.eE]+)%",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<bucket>0_1000|1000_10000|10000_100000|total)\s*\|\s*(?P<count>\d+)\s*\|\s*"
        r"(?P<time>[-+0-9.eE]+)\s*\|\s*(?P<gap>[-+0-9.eE]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<bucket>\[[^\]]+\))\s*num=(?P<count>\d+),\s*"
        r"avg_gap=(?P<gap>[-+0-9.eE]+)%,\s*avg_time=(?P<time>[-+0-9.eE]+)s",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<bucket>\[[^\]]+\]?)\s*:\s*count=(?P<count>\d+),\s*"
        r"avg_gap=(?P<gap>[-+0-9.eE]+)%,\s*avg_time=(?P<time>[-+0-9.eE]+)s",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<bucket>\[[^\]]+\]?\)?)\s*,\s*number:\s*(?P<count>\d+),\s*"
        r"avg\s*(?:GAP|gap(?:\(no aug\))?):\s*(?P<gap>[-+0-9.eE]+)%,\s*avg time:\s*(?P<time>[-+0-9.eE]+)s",
        re.IGNORECASE,
    ),
)

GAP_OVER_TIME_PATTERN = re.compile(
    r"\[GAP_OVER_TIME\]\s*"
    r"elapsed_second=(?P<elapsed_second>\d+)\s+"
    r"solved_count=(?P<solved_count>\d+)\s+"
    r"avg_gap=(?P<avg_gap>[-+0-9.eE]+|NaN|inf)\s+"
    r"last_instance=(?P<last_instance>\S*)\s+"
    r"last_size=(?P<last_size>\S*)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run original NRS Survey scripts and normalize stdout/log output to CSV."
    )
    parser.add_argument("--methods", nargs="+", default=["all"], help="Methods or job ids, e.g. bq sil_prc lehd_tsp.")
    parser.add_argument("--skip", nargs="+", default=[], help="Methods or job ids to skip, e.g. lehd_rrc lehd_rrc_tsp.")
    parser.add_argument("--problems", nargs="+", choices=["tsp", "cvrp"], default=["tsp", "cvrp"])
    parser.add_argument(
        "--size",
        choices=["dev", "small", "medium", "large", "full"],
        default="dev",
        help="Size range passed to original scripts through NRS_* env vars.",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to results/original-<timestamp>.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run original scripts.")
    parser.add_argument("--timeout", type=float, default=None, help="Optional per-job timeout in seconds.")
    parser.add_argument("--stream", action="store_true", help="Stream each child job's stdout/stderr while still saving raw logs.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running remaining jobs after failure.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed job, including --methods all.")
    parser.add_argument("--list", action="store_true", help="List available original jobs and exit.")
    parser.add_argument("--no-skip-src-copy", action="store_true", help="Allow original scripts to copy source snapshots.")
    parser.add_argument("--smoke-episodes", type=int, default=1, help="Episode cap for --size dev where supported.")
    return parser.parse_args()


def size_bounds(size: str) -> tuple[int, int] | None:
    if size == "full":
        return None
    if size == "dev":
        return (0, 1000)
    if size == "small":
        return (0, 1000)
    if size == "medium":
        return (1000, 10000)
    if size == "large":
        return (10000, 100001)
    raise ValueError(size)


def job_size_bounds(args: argparse.Namespace, job: OriginalJob) -> tuple[int, int] | None:
    if args.size == "dev" and job.dev_bounds is not None:
        return job.dev_bounds
    return size_bounds(args.size)


def select_jobs(methods: list[str], problems: list[str], skip: list[str] | None = None) -> list[OriginalJob]:
    if methods == ["all"]:
        selected = [job for job in JOBS if job.problem in problems]
    else:
        wanted = set(methods)
        selected = [
            job
            for job in JOBS
            if job.problem in problems and (job.method in wanted or job.job_id in wanted)
        ]
    if skip:
        skipped = set(skip)
        selected = [job for job in selected if job.method not in skipped and job.job_id not in skipped]
    if not selected:
        raise SystemExit("no original jobs selected")
    return selected


def missing_jobs(jobs: Iterable[OriginalJob]) -> list[OriginalJob]:
    return [job for job in jobs if not job.script.exists()]


def bucket_for_size(size: int | None) -> str:
    if size is None:
        return ""
    if size < 1000:
        return "<1K"
    if size < 10000:
        return "[1K,10K)"
    return ">=10K"


def infer_size(name: str, problem: str) -> int | None:
    if problem == "cvrp":
        match = re.search(r"-n(?P<n>\d+)-", name)
        if match:
            return max(0, int(match.group("n")) - 1)
    match = re.search(r"(?P<n>\d+)$", name)
    if match:
        return int(match.group("n"))
    return None


def normalize_summary_bucket(bucket: str) -> str:
    cleaned = bucket.strip().lower().replace("_", "-")
    cleaned = cleaned.replace(" ", "")
    if cleaned in {"0-1000", "0,1000", "[0,1000)", "<1k"}:
        return "<1K"
    if cleaned in {"1000-10000", "1000,10000", "[1000,10000)", "1k-10k", "[1k,10k)"}:
        return "[1K,10K)"
    if cleaned in {"10000-100000", "10000,100000", "[10000,100000]", "10k-100k", ">=10k"}:
        return ">=10K"
    if cleaned == "total":
        return "total"
    return bucket.strip()


def discover_logs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {p for p in root.rglob("*") if p.is_file() and p.suffix in {".log", ".txt"}}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_tsplib_instance(path: Path) -> tuple[str, float | None, list[tuple[float, float]]]:
    name = path.stem
    coords: list[tuple[float, float]] = []
    in_coords = False
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("NAME"):
            name = line.split()[-1]
        elif line.startswith("NODE_COORD_SECTION"):
            in_coords = True
        elif line.startswith("EOF"):
            break
        elif in_coords:
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))

    opt = None
    opt_path = SURVEY_DATA / "survey_bench_opt_tsp_same_file_name.py"
    if opt_path.exists():
        text = opt_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(rf"['\"]{re.escape(name)}['\"]\s*:\s*([-+0-9.eE]+)", text)
        if match:
            opt = float(match.group(1))
    return name, opt, coords


def read_cvrplib_instance(path: Path) -> tuple[str, float | None, list[tuple[float, float]], list[int], int]:
    name = path.stem
    coords: list[tuple[float, float]] = []
    demands: list[int] = []
    capacity = 0
    section = ""
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("NAME"):
            name = line.split()[-1]
        elif line.startswith("CAPACITY"):
            capacity = int(float(line.split()[-1]))
        elif line.startswith("NODE_COORD_SECTION"):
            section = "coords"
        elif line.startswith("DEMAND_SECTION"):
            section = "demands"
        elif line.startswith("DEPOT_SECTION"):
            section = ""
        elif section == "coords":
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
        elif section == "demands":
            parts = line.split()
            if len(parts) >= 2:
                demands.append(int(float(parts[-1])))

    opt = None
    sol_path = path.with_suffix(".sol")
    if sol_path.exists():
        for raw_line in sol_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if raw_line.startswith("Cost"):
                opt = float(raw_line.split()[-1])
                break
    return name, opt, coords, demands, capacity


def write_sil_tsp_smoke_fixture(target: Path) -> None:
    if target.exists():
        return
    candidates = sorted(SURVEY_TSP_DIR.glob("*.tsp"))
    source = next((path for path in candidates if "C1k" in path.name), candidates[0] if candidates else None)
    if source is None:
        return
    name, opt, coords = read_tsplib_instance(source)
    if not coords:
        return
    fields = [name, str(opt or 0.0)]
    for x, y in coords:
        fields.extend((f"{x:g}", f"{y:g}"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(",".join(fields) + "\n", encoding="utf-8")


def write_sil_cvrp_smoke_fixture(target: Path) -> None:
    if target.exists():
        return
    candidates = sorted(SURVEY_CVRP_DIR.glob("X-n1001-*.vrp"))
    source = candidates[0] if candidates else next(iter(sorted(SURVEY_CVRP_DIR.glob("*.vrp"))), None)
    if source is None:
        return
    name, opt, coords, demands, capacity = read_cvrplib_instance(source)
    if len(coords) < 2 or len(demands) != len(coords):
        return
    depot = coords[0]
    customers = coords[1:]
    fields = ["['name'", name, "'depot'", f"{depot[0]:g}", f"{depot[1]:g}", "'customer'"]
    for x, y in customers:
        fields.extend((f"{x:g}", f"{y:g}"))
    fields.append("'demand'")
    fields.extend(str(demand) for demand in demands)
    fields.extend(("'capacity'", str(capacity), "'cost'", str(opt or 0.0), "']"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(", ".join(fields) + "\n", encoding="utf-8")


def ensure_sil_smoke_data(args: argparse.Namespace, job: OriginalJob) -> None:
    if args.size != "dev" or job.method not in {"sil", "sil_prc"}:
        return
    if job.problem == "tsp":
        data_path = job.script.parent.parent / "data/test_set/TSPlib_scale_ge_1K_n33_ascending.txt"
        write_sil_tsp_smoke_fixture(data_path)
    elif job.problem == "cvrp":
        data_path = job.script.parent.parent / "data/test_set/CVRPlib_scale_larger_than1000_Li_X_XXL_n14.txt"
        write_sil_cvrp_smoke_fixture(data_path)


def parse_instance_rows(text: str, job: OriginalJob, log_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    if job.job_id == "fast_t2t_tsp":
        for match in FAST_T2T_INSTANCE_PATTERN.finditer(text):
            data = match.groupdict()
            size = int(data["size"])
            status, note = classify_instance_status(data["gap"], data["time"])
            rows.append(
                {
                    "status": status,
                    "method": job.method,
                    "problem": job.problem,
                    "job_id": job.job_id,
                    "instance": data["name"],
                    "size": data["size"],
                    "size_group": bucket_for_size(size),
                    "cost": data["cost"],
                    "bks": data["opt"],
                    "gap_percent": data["gap"],
                    "time_seconds": data["time"],
                    "source_log": str(log_path),
                    "note": note,
                }
            )
        if rows:
            return rows

    for line in text.splitlines():
        for pattern in INSTANCE_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            data = match.groupdict()
            name = data.get("name") or ""
            size = int(data["size"]) if data.get("size") else infer_size(name, job.problem)
            key = (job.job_id, name, str(size or ""), data["time"])
            if key in seen:
                break
            seen.add(key)
            
            gap_str = data.get("gap", "")
            time_str = data.get("time", "")
            status, note = classify_instance_status(gap_str, time_str, line)
                
            rows.append(
                {
                    "status": status,
                    "method": job.method,
                    "problem": job.problem,
                    "job_id": job.job_id,
                    "instance": name,
                    "size": str(size or ""),
                    "size_group": bucket_for_size(size),
                    "cost": data.get("cost", ""),
                    "bks": data.get("opt", ""),
                    "gap_percent": gap_str,
                    "time_seconds": time_str,
                    "source_log": str(log_path),
                    "note": note,
                }
            )
            break
    return rows


def classify_instance_status(gap_str: str, time_str: str, line: str = "") -> tuple[str, str]:
    if "TIMEOUT" in line.upper():
        return "FAILED", "timeout"
    for value in (gap_str, time_str):
        upper = value.upper()
        if "OOM" in upper or "NAN" in upper or "INF" in upper:
            return "FAILED", value.strip()
    if "OOM" in line.upper():
        return "FAILED", "OOM"
    if gap_str:
        try:
            if float(gap_str) > 100.0:
                return "FAILED", "gap > 100%"
        except ValueError:
            pass
    if time_str:
        try:
            if float(time_str) > 1000.0:
                return "FAILED", "time > 1000s"
        except ValueError:
            pass
    return "ok", ""


def parse_summary_rows(text: str, job: OriginalJob, log_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for line in text.splitlines():
        for pattern in SUMMARY_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            data = match.groupdict()
            bucket = normalize_summary_bucket(data["bucket"])
            key = (job.job_id, bucket, data["count"])
            if key in seen:
                break
            seen.add(key)
            rows.append(
                {
                    "status": "summary",
                    "method": job.method,
                    "problem": job.problem,
                    "job_id": job.job_id,
                    "size_group": bucket,
                    "count": data["count"],
                    "avg_gap_percent": data["gap"],
                    "total_time_seconds": "",
                    "avg_time_seconds": data["time"],
                    "source_log": str(log_path),
                }
            )
            break
    return rows


def parse_gap_over_time_rows(text: str, job: OriginalJob, _log_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in text.splitlines():
        match = GAP_OVER_TIME_PATTERN.search(line)
        if not match:
            continue
        data = match.groupdict()
        key = (job.job_id, data["elapsed_second"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "method": job.method,
                "problem": job.problem,
                "job_id": job.job_id,
                "elapsed_second": data["elapsed_second"],
                "solved_count": data["solved_count"],
                "avg_gap_percent": data["avg_gap"],
                "last_instance": data["last_instance"],
                "last_size": data["last_size"],
            }
        )
    return rows


def summarize(rows: Iterable[dict[str, str]], job: OriginalJob) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status") not in {"ok", "FAILED"}:
            continue
        groups.setdefault(row["size_group"], []).append(row)

    summary: list[dict[str, str]] = []
    for bucket in ("<1K", "[1K,10K)", ">=10K"):
        bucket_rows = groups.get(bucket, [])
        if not bucket_rows:
            continue
        solved_rows = [row for row in bucket_rows if row.get("status") == "ok"]
        failed_count = len(bucket_rows) - len(solved_rows)
        gaps = [float(row["gap_percent"]) for row in solved_rows if row.get("gap_percent")]
        times = [float(row["time_seconds"]) for row in solved_rows if row.get("time_seconds")]
        summary.append(
            {
                "status": "computed",
                "method": job.method,
                "problem": job.problem,
                "job_id": job.job_id,
                "size_group": bucket,
                "count": str(len(solved_rows)),
                "solved_count": str(len(solved_rows)),
                "failed_count": str(failed_count),
                "total_count": str(len(bucket_rows)),
                "reliability_percent": f"{100.0 * len(solved_rows) / len(bucket_rows):.6f}",
                "avg_gap_percent": f"{sum(gaps) / len(gaps):.6f}" if gaps else "",
                "total_time_seconds": f"{sum(times):.6f}" if times else "",
                "avg_time_seconds": f"{sum(times) / len(times):.6f}" if times else "",
                "source_log": "parsed instance rows",
            }
        )
    return summary


def dedupe_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for row in rows:
        if row.get("status") == "ok":
            key = (
                row.get("job_id", ""),
                row.get("instance", ""),
                row.get("size", ""),
                row.get("cost", ""),
                row.get("bks", ""),
                row.get("time_seconds", ""),
            )
            if key in seen:
                continue
            seen.add(key)
        deduped.append(row)
    return deduped


def dedupe_gap_time_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.get("job_id", ""), row.get("elapsed_second", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def gap_over_time_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    by_job: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status") != "ok" or not row.get("instance"):
            continue
        gap = parse_float(row.get("gap_percent", ""))
        elapsed = parse_float(row.get("time_seconds", ""))
        if gap is None or elapsed is None:
            continue
        by_job.setdefault(row["job_id"], []).append(row)

    timeline: list[dict[str, str]] = []
    for job_id, job_rows in by_job.items():
        if not job_rows:
            continue

        cumulative_time = 0.0
        completed: list[tuple[float, float, dict[str, str]]] = []
        for row in job_rows:
            elapsed = parse_float(row.get("time_seconds", "")) or 0.0
            gap = parse_float(row.get("gap_percent", ""))
            if gap is None:
                continue
            cumulative_time += max(0.0, elapsed)
            completed.append((cumulative_time, gap, row))

        if not completed:
            continue

        max_second = max(1, int(completed[-1][0] + 0.999999))
        next_idx = 0
        gap_sum = 0.0
        latest_row = completed[0][2]
        for second in range(1, max_second + 1):
            while next_idx < len(completed) and completed[next_idx][0] <= second:
                _done_time, gap, latest_row = completed[next_idx]
                gap_sum += gap
                next_idx += 1
            solved_count = next_idx
            avg_gap = gap_sum / solved_count if solved_count else None
            timeline.append(
                {
                    "method": latest_row.get("method", ""),
                    "problem": latest_row.get("problem", ""),
                    "job_id": job_id,
                    "elapsed_second": str(second),
                    "solved_count": str(solved_count),
                    "avg_gap_percent": f"{avg_gap:.6f}" if avg_gap is not None else "",
                    "last_instance": latest_row.get("instance", "") if solved_count else "",
                    "last_size": latest_row.get("size", "") if solved_count else "",
                }
            )
    return timeline


def plot_gap_over_time(rows: list[dict[str, str]], path: Path) -> bool:
    if not rows:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[original] warning: could not import matplotlib for gap plot: {exc}", file=sys.stderr)
        return False

    by_job: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if not row.get("avg_gap_percent"):
            continue
        by_job.setdefault(row["job_id"], []).append(row)

    if not by_job:
        return False

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for job_id, job_rows in sorted(by_job.items()):
        seconds = [int(row["elapsed_second"]) for row in job_rows]
        gaps = [float(row["avg_gap_percent"]) for row in job_rows]
        ax.plot(seconds, gaps, marker="o", linewidth=1.8, markersize=3, label=job_id)

    ax.set_xlabel("Elapsed seconds within method")
    ax.set_ylabel("Cumulative average gap (%)")
    ax.set_title("Original Baselines: Average Gap Over Time")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def job_env(args: argparse.Namespace, out_dir: Path, job: OriginalJob) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "NRS_EVAL_SIZE_LOW",
        "NRS_EVAL_SIZE_HIGH",
        "NRS_SURVEY_SIZE_LOW",
        "NRS_SURVEY_SIZE_HIGH",
        "NRS_EVAL_SIZE_BUCKET",
        "NRS_EVAL_DEBUG_SMALLEST",
        "NRS_SMOKE",
        "NRS_SMOKE_EPISODES",
    ):
        env.pop(name, None)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(name, "20")
    env["NRS_METHOD_LOG_ROOT"] = str(out_dir / "method_logs" / job.job_id)
    env["NRS_SURVEY_TSP_DIR"] = str(os.path.relpath(SURVEY_TSP_DIR, job.script.parent))
    env["NRS_SURVEY_CVRP_DIR"] = str(os.path.relpath(SURVEY_CVRP_DIR, job.script.parent))
    if not args.no_skip_src_copy:
        env["NRS_SKIP_SRC_COPY"] = "1"
    if args.timeout is not None:
        env.setdefault("NRS_INSTANCE_TIMEOUT", str(args.timeout))

    bounds = job_size_bounds(args, job)
    if args.size == "dev":
        env["NRS_SMOKE"] = "1"
        env["NRS_SMOKE_EPISODES"] = str(args.smoke_episodes)
        if job.method in {"nn", "elkai", "pyvrp", "lkh", "hygese"}:
            env.setdefault("NRS_EVAL_DEBUG_SMALLEST", "1")
        if job.method == "pyvrp":
            env.setdefault("NRS_PYVRP_MAX_ITERATIONS", "1")
        if job.method == "lkh":
            env.setdefault("NRS_LKH_MAX_TRIALS", "1")
            env.setdefault("NRS_LKH_RUNS", "1")
        if job.method == "hygese":
            env.setdefault("NRS_HYGESE_NB_ITER", "1")
            env.setdefault("NRS_HYGESE_TIME_LIMIT", "0")
    if bounds is not None:
        low, high = bounds
        env["NRS_EVAL_SIZE_LOW"] = str(low)
        env["NRS_EVAL_SIZE_HIGH"] = str(high)
        env["NRS_SURVEY_SIZE_LOW"] = str(low)
        env["NRS_SURVEY_SIZE_HIGH"] = str(high)
        env["NRS_EVAL_SIZE_BUCKET"] = args.size
    return env


def ensure_sil_prc_checkpoints(job: OriginalJob) -> None:
    if job.method != "sil_prc":
        return
    if job.problem == "tsp":
        source_dir = SURVEY / "Construction/single-stage/appending/3_SIL/TSP/Test_All/result"
        target_dir = SURVEY / "Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/2_SIL/TSP/Test_All/result"
        prefix = "checkpoint-tsp"
    elif job.problem == "cvrp":
        source_dir = SURVEY / "Construction/single-stage/appending/3_SIL/CVRP/Test_All/result"
        target_dir = SURVEY / "Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/2_SIL/CVRP/Test_All/result"
        prefix = "checkpoint-cvrp"
    else:
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    # We use 1k model for each method as requested
    for suffix in ("1k", "5k", "10k", "50k", "100k"):
        source = source_dir / f"{prefix}1k.pt"
        target = target_dir / f"{prefix}{suffix}.pt"
        if target.exists() and not target.is_symlink():
            target.unlink()
        if target.is_symlink():
            target.unlink()
        if source.exists():
            target.symlink_to(os.path.relpath(source, start=target_dir))


def ensure_fast_t2t_cython(args: argparse.Namespace, job: OriginalJob) -> None:
    if job.job_id != "fast_t2t_tsp":
        return
    ext_dir = job.script.parent / "diffusion/utils/cython_merge"
    if list(ext_dir.glob("cython_merge*.so")):
        return
    subprocess.run(
        [args.python, "setup.py", "build_ext", "--inplace"],
        cwd=str(ext_dir),
        check=True,
    )


def supports_instance_timeout(job: OriginalJob) -> bool:
    return job.job_id in {"lehd_rrc_tsp"}


def run_child_process(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout: float | None,
    stream: bool,
    job_id: str,
) -> subprocess.CompletedProcess[str]:
    if not stream:
        return subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, ("stdout", stdout_parts, sys.stdout))
    selector.register(proc.stderr, selectors.EVENT_READ, ("stderr", stderr_parts, sys.stderr))

    start = time.perf_counter()
    timed_out = False
    while selector.get_map():
        if timeout is not None and time.perf_counter() - start > timeout:
            timed_out = True
            proc.kill()
        for key, _ in selector.select(timeout=0.2):
            stream_name, parts, output = key.data
            line = key.fileobj.readline()
            if line:
                parts.append(line)
                print(f"[{job_id} {stream_name}] {line}", end="", file=output, flush=True)
            else:
                selector.unregister(key.fileobj)

    returncode = proc.wait()
    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    if timed_out:
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def run_job(
    args: argparse.Namespace,
    out_dir: Path,
    job: OriginalJob,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if not job.script.exists():
        failure = {
            "status": "missing_script",
            "method": job.method,
            "problem": job.problem,
            "job_id": job.job_id,
            "instance": "",
            "size": "",
            "size_group": "",
            "cost": "",
            "bks": "",
            "gap_percent": "",
            "time_seconds": "",
            "source_log": str(job.script),
            "note": "script not found",
        }
        return [failure], [], []

    ensure_sil_smoke_data(args, job)
    ensure_sil_prc_checkpoints(job)
    ensure_fast_t2t_cython(args, job)
    raw_dir = out_dir / "raw" / job.job_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = raw_dir / "stdout.txt"
    stderr_path = raw_dir / "stderr.txt"
    command_path = raw_dir / "command.txt"
    command = [args.python, str(job.script), *job.script_args]
    command_path.write_text(" ".join(command) + "\n", encoding="utf-8")

    log_root = out_dir / "method_logs" / job.job_id
    before_logs = discover_logs(log_root)
    start = time.perf_counter()
    process_timeout = None if args.timeout is not None and supports_instance_timeout(job) else args.timeout
    proc = run_child_process(
        command,
        cwd=job.script.parent,
        env=job_env(args, out_dir, job),
        timeout=process_timeout,
        stream=args.stream,
        job_id=job.job_id,
    )
    elapsed = time.perf_counter() - start
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")

    texts: list[tuple[Path, str]] = [(stdout_path, proc.stdout), (stderr_path, proc.stderr)]
    after_logs = discover_logs(log_root)
    for log_path in sorted(after_logs - before_logs):
        texts.append((log_path, read_text(log_path)))

    rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    gap_time_rows: list[dict[str, str]] = []
    for log_path, text in texts:
        rows.extend(parse_instance_rows(text, job, log_path))
        summary_rows.extend(parse_summary_rows(text, job, log_path))
        gap_time_rows.extend(parse_gap_over_time_rows(text, job, log_path))

    rows = dedupe_rows(rows)
    gap_time_rows = dedupe_gap_time_rows(gap_time_rows)
    summary_rows.extend(summarize(rows, job))
    if proc.returncode != 0:
        note = f"exit code {proc.returncode}"
        if "out of memory" in proc.stderr.lower() or "oom" in proc.stderr.lower() or "out of memory" in proc.stdout.lower() or "oom" in proc.stdout.lower():
            note = "OOM"
            
        print(f"[original] ERROR: {job.job_id} failed with exit code {proc.returncode}", file=sys.stderr)
        if proc.stderr.strip():
            print(f"[original] STDERR:\n{proc.stderr.strip()}", file=sys.stderr)
        
        rows.append(
            {
                "status": "FAILED" if note == "OOM" else "failed",
                "method": job.method,
                "problem": job.problem,
                "job_id": job.job_id,
                "instance": "",
                "size": "",
                "size_group": "",
                "cost": "",
                "bks": "",
                "gap_percent": "",
                "time_seconds": f"{elapsed:.6f}",
                "source_log": str(stderr_path),
                "note": note,
            }
        )
    elif not rows:
        rows.append(
            {
                "status": "no_instance_rows",
                "method": job.method,
                "problem": job.problem,
                "job_id": job.job_id,
                "instance": "",
                "size": "",
                "size_group": "",
                "cost": "",
                "bks": "",
                "gap_percent": "",
                "time_seconds": f"{elapsed:.6f}",
                "source_log": str(stdout_path),
                "note": "script finished but parser found no per-instance rows",
            }
        )
    return rows, summary_rows, gap_time_rows


def main() -> int:
    args = parse_args()
    if args.list:
        for job in JOBS:
            print(f"{job.job_id}\t{job.method}\t{job.problem}\t{job.script.relative_to(ROOT)}")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / f"original-{timestamp}"
    out_dir = out_dir.resolve()
    jobs = select_jobs(args.methods, args.problems, args.skip)
    continue_after_error = (args.methods == ["all"] and not args.fail_fast) or args.continue_on_error
    missing = missing_jobs(jobs)
    if missing:
        print(
            "[original] warning: some selected jobs are not present in this checkout. "
            "The pretrained-assets archive restores checkpoints/data files, not method "
            "source trees; commit the now-unignored method source directories for fresh clones.",
            file=sys.stderr,
        )
        for job in missing:
            print(f"[original] missing {job.job_id}: {job.script.relative_to(ROOT)}", file=sys.stderr)
        if not continue_after_error:
            print(
                "[original] stopping at the first missing/failed job by default; "
                "use --continue-on-error to run the available jobs.",
                file=sys.stderr,
            )

    all_rows: list[dict[str, str]] = []
    all_summary_rows: list[dict[str, str]] = []
    all_gap_time_rows: list[dict[str, str]] = []
    for job in jobs:
        print(f"[original] running {job.job_id}: {job.script.relative_to(ROOT)}", flush=True)
        try:
            rows, summary_rows, gap_time_rows = run_job(args, out_dir, job)
        except subprocess.TimeoutExpired as exc:
            rows = [
                {
                    "status": "timeout",
                    "method": job.method,
                    "problem": job.problem,
                    "job_id": job.job_id,
                    "instance": "",
                    "size": "",
                    "size_group": "",
                    "cost": "",
                    "bks": "",
                    "gap_percent": "",
                    "time_seconds": str(args.timeout or ""),
                    "source_log": "",
                    "note": f"timeout after {exc.timeout}s",
                }
            ]
            summary_rows = []
            gap_time_rows = []
        all_rows.extend(rows)
        all_summary_rows.extend(summary_rows)
        all_gap_time_rows.extend(gap_time_rows)
        failed_statuses = {"failed", "FAILED", "timeout", "missing_script"}
        if any(row["status"] in failed_statuses for row in rows) and not continue_after_error:
            break

    result_fields = [
        "status",
        "method",
        "problem",
        "job_id",
        "instance",
        "size",
        "size_group",
        "cost",
        "bks",
        "gap_percent",
        "time_seconds",
        "source_log",
        "note",
    ]
    write_csv(
        out_dir / "original_results.csv",
        all_rows,
        result_fields,
    )
    write_csv(
        out_dir / "original_instances.csv",
        [row for row in all_rows if row.get("status") in {"ok", "FAILED"} and row.get("instance")],
        result_fields,
    )
    write_csv(
        out_dir / "original_summary.csv",
        all_summary_rows,
        [
            "status",
            "method",
            "problem",
            "job_id",
            "size_group",
            "count",
            "solved_count",
            "failed_count",
            "total_count",
            "reliability_percent",
            "avg_gap_percent",
            "total_time_seconds",
            "avg_time_seconds",
            "source_log",
        ],
    )
    fallback_gap_timeline = gap_over_time_rows(all_rows)
    explicit_jobs = {row["job_id"] for row in all_gap_time_rows}
    gap_timeline = all_gap_time_rows + [
        row for row in fallback_gap_timeline if row.get("job_id") not in explicit_jobs
    ]
    write_csv(
        out_dir / "original_gap_over_time.csv",
        gap_timeline,
        [
            "method",
            "problem",
            "job_id",
            "elapsed_second",
            "solved_count",
            "avg_gap_percent",
            "last_instance",
            "last_size",
        ],
    )
    gap_plot_path = out_dir / "original_gap_over_time.png"
    wrote_gap_plot = plot_gap_over_time(gap_timeline, gap_plot_path)
    print(f"[original] wrote {out_dir / 'original_results.csv'}")
    print(f"[original] wrote {out_dir / 'original_instances.csv'}")
    print(f"[original] wrote {out_dir / 'original_summary.csv'}")
    print(f"[original] wrote {out_dir / 'original_gap_over_time.csv'}")
    if wrote_gap_plot:
        print(f"[original] wrote {gap_plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
