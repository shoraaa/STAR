#!/usr/bin/env python3
"""In-process policy/search experiments for neural routing.

This module deliberately does not launch the original test scripts.  Search
strategies and append policies either run in-process, or are reported as
unsupported with the concrete reason.
"""

from __future__ import annotations

import argparse
import importlib.util
import csv
import json
import math
import os
import random
import sys
import time
import weakref
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np
import torch

from survey.NRS import native_forward_swap

try:
    from STAR import _STAR as STAR
except Exception:  # pragma: no cover - optional compiled acceleration
    STAR = None


CUDA_AVAILABLE = torch.cuda.is_available()
DEFAULT_TORCH_DEVICE = torch.device("cuda", 0) if CUDA_AVAILABLE else torch.device("cpu")
ROOT = Path(__file__).resolve().parents[1] / "survey"
DEFAULT_OUT_DIR = ROOT / "results" / "faithful-inprocess"
DEFAULT_TSP = ROOT / "0_data_survey/survey_bench_tsp/national_wi29.tsp"
DEFAULT_CVRP = ROOT / "0_data_survey/survey_bench_cvrp/X-n101-k25.vrp"
TSP_BENCH_DIR = ROOT / "0_data_survey/survey_bench_tsp"
CVRP_BENCH_DIR = ROOT / "0_data_survey/survey_bench_cvrp"
LEHD_TSP_DIR = ROOT / "NRS/Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/1_LEHD/TSP"
SIL_CVRP_DIR = ROOT / "NRS/Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/2_SIL/CVRP/Test_All"
LEHD_CVRP_DIR = ROOT / "NRS/Construction/single-stage/appending/2_LEHD/CVRP"
LEHD_RRC_CVRP_DIR = ROOT / "NRS/Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/1_LEHD/CVRP"
SIL_PRC_TSP_DIR = ROOT / "NRS/Construction/single-stage/appending/3_SIL/TSP/Test_All"
SIL_PRC_CVRP_DIR = ROOT / "NRS/Construction/single-stage/appending/3_SIL/CVRP/Test_All"

FIELDS = [
    "status",
    "strategy_id",
    "policy_id",
    "problem",
    "instance",
    "final_cost",
    "gap",
    "time",
]

STAR_PROGRESS_FIELDS = [
    "strategy_id",
    "policy_id",
    "problem",
    "instance",
    "completed_iterations",
    "iteration",
    "total_iterations",
    "samples",
    "min_new_edges",
    "refine_k",
    "refine",
    "memory",
    "memory_update_mode",
    "advantage_scale",
    "source_cost",
    "best_cost",
    "best_gap",
    "elapsed_seconds",
    "iteration_seconds",
]

ORIGINAL_SMALLEST_GAPS = {
    # Original native-script smallest rows observed with NRS_FORWARD_OVERRIDE.
    # A reimplementation is only treated as passing when it matches these gaps.
    ("lehd_rrc", "lehd", "tsp"): 0.000,
    ("sil_prc", "sil", "tsp"): 0.000,
    ("greedy", "lehd", "tsp"): 0.072456,
    ("greedy", "sil", "tsp"): 0.000,
    ("greedy", "lehd", "cvrp"): 10.952847,
    ("greedy", "sil", "cvrp"): 141.923091,
}


@dataclass(frozen=True)
class Instance:
    name: str
    problem: str
    coords: dict[int, tuple[float, float]]
    demands: dict[int, int]
    capacity: int | None
    bks_cost: float | None
    source_path: Path | None = None
    edge_weight_type: str = "EUC_2D"

    @property
    def depot(self) -> int:
        return 1


class AppendPolicy(Protocol):
    policy_id: str

    def select_next(
        self,
        instance: Instance,
        current: int,
        candidates: Sequence[int],
        rng: random.Random,
        prefix: Sequence[int],
        *,
        repair: bool = False,
    ) -> int:
        ...


def normalize_coords_for_policy(raw: torch.Tensor, policy_id: str, problem: str) -> torch.Tensor:
    if problem == "cvrp" and policy_id == "elg":
        xy_min = raw.amin(dim=1, keepdim=True)
        xy_max = raw.amax(dim=1, keepdim=True)
        return (raw - xy_min) / (xy_max - xy_min).clamp_min(1e-12)
    if policy_id in {"icam", "elg"}:
        xy_min = raw.amin(dim=1, keepdim=True)
        xy_max = raw.amax(dim=1, keepdim=True)
        scale = (xy_max - xy_min).amax(dim=-1, keepdim=True).clamp_min(1e-12)
        return (raw - xy_min) / scale
    scale = (raw.max() - raw.min()).clamp_min(1e-12)
    return (raw - raw.min()) / scale


@dataclass
class NearestPolicy:
    policy_id: str = "nearest"

    def select_next(
        self,
        instance: Instance,
        current: int,
        candidates: Sequence[int],
        rng: random.Random,
        prefix: Sequence[int],
        *,
        repair: bool = False,
    ) -> int:
        del prefix, repair
        del rng
        return min(candidates, key=lambda node: (distance(instance, current, node), node))


@dataclass
class SoftDistPolicy:
    policy_id: str = "softdist"
    temperature: float = 0.0051
    knn: int = 50
    _neighbors: dict[tuple[int, int], list[tuple[int, float]]] = field(default_factory=dict, init=False, repr=False)

    def select_next(
        self,
        instance: Instance,
        current: int,
        candidates: Sequence[int],
        rng: random.Random,
        prefix: Sequence[int],
        *,
        repair: bool = False,
    ) -> int:
        del prefix, repair
        candidate_set = set(candidates)
        pool: list[tuple[int, float]] = []
        for node, dist_value in self._nearest_distances(instance, current):
            if node in candidate_set:
                pool.append((node, dist_value))
                if len(pool) >= self.knn:
                    break
        if not pool:
            raise ValueError("softdist received no legal candidates")
        logits = [-dist_value / max(self.temperature, 1e-12) for _node, dist_value in pool]
        max_logit = max(logits)
        weights = [math.exp(logit - max_logit) for logit in logits]
        total = sum(weights)
        threshold = rng.random() * total
        running = 0.0
        for (node, _dist_value), weight in zip(pool, weights):
            running += weight
            if running >= threshold:
                return node
        return pool[-1][0]

    def _nearest_distances(self, instance: Instance, current: int) -> list[tuple[int, float]]:
        key = (id(instance), current)
        cached = self._neighbors.get(key)
        if cached is not None:
            return cached
        neighbors = [
            (node, normalized_distance(instance, current, node))
            for node in instance.coords
            if node != current
        ]
        neighbors.sort(key=lambda item: (item[1], item[0]))
        self._neighbors[key] = neighbors
        return neighbors


@dataclass
class NativeTSPNeuralPolicy:
    """In-process wrapper around the original checkpoint-backed TSP append model."""

    policy_id: str

    def __post_init__(self) -> None:
        spec = native_forward_swap._neural_policy(self.policy_id, "tsp")
        if spec is None:
            raise ValueError(f"unknown neural TSP append policy: {self.policy_id}")
        self._spec = spec
        self._node_ids: list[int] = []
        self._node_to_index: dict[int, int] = {}
        self._coords_tensor: torch.Tensor | None = None
        self._solution_tensor: torch.Tensor | None = None
        self._state: SimpleNamespace | None = None
        self._encoded_tensor: torch.Tensor | None = None
        self._encoded_device: torch.device | None = None

    def select_next(
        self,
        instance: Instance,
        current: int,
        candidates: Sequence[int],
        rng: random.Random,
        prefix: Sequence[int],
        *,
        repair: bool = False,
    ) -> int:
        del current, rng
        if instance.problem != "tsp":
            raise ValueError(f"{self.policy_id} neural adapter is currently extracted for TSP only")
        self._ensure_instance(instance)
        assert self._coords_tensor is not None
        if not prefix:
            raise ValueError(f"{self.policy_id} neural adapter requires the caller-provided selected prefix")
        selected = torch.tensor(
            [[self._node_to_index[node] for node in prefix]],
            dtype=torch.long,
            device=self._coords_tensor.device,
        )
        assert self._solution_tensor is not None
        assert self._state is not None
        with original_torch_device_context():
            with torch.no_grad():
                picked, _prob, _misc, selected_student = native_forward_swap._call_tsp_neural(
                    self._spec,
                    self._state,
                    selected,
                    self._solution_tensor,
                    len(prefix),
                    decode_method="greedy",
                    repair=repair,
                )
        picked_index = int(selected_student.reshape(-1)[0].item() if hasattr(selected_student, "reshape") else picked.reshape(-1)[0].item())
        if picked_index < 0 or picked_index >= len(self._node_ids):
            raise ValueError(f"{self.policy_id} returned out-of-range TSP node index {picked_index}")
        picked_node = self._node_ids[picked_index]
        return picked_node

    def action_probabilities(
        self,
        instance: Instance,
        prefix: Sequence[int],
        *,
        repair: bool = False,
    ) -> dict[int, float]:
        if instance.problem != "tsp":
            raise ValueError(f"{self.policy_id} neural adapter is currently extracted for TSP only")
        self._ensure_instance(instance)
        assert self._coords_tensor is not None
        assert self._solution_tensor is not None
        assert self._state is not None
        if not prefix:
            raise ValueError(f"{self.policy_id} neural adapter requires the caller-provided selected prefix")

        selected = torch.tensor(
            [[self._node_to_index[node] for node in prefix]],
            dtype=torch.long,
            device=self._coords_tensor.device,
        )
        with original_torch_device_context():
            with torch.no_grad():
                probs = tsp_neural_action_probabilities(
                    self._spec,
                    self._state,
                    selected,
                    self._solution_tensor,
                    len(prefix),
                    repair=repair,
                )
        row = probs.reshape(-1).detach().to(device="cpu", dtype=torch.float64)
        return {
            self._node_ids[index]: float(value)
            for index, value in enumerate(row.tolist())
            if value > 0.0
        }

    def action_probability_vector(
        self,
        instance: Instance,
        prefix: Sequence[int],
        *,
        repair: bool = False,
        allowed_candidates: Sequence[int] | None = None,
    ) -> list[float]:
        if instance.problem != "tsp":
            raise ValueError(f"{self.policy_id} neural adapter is currently extracted for TSP only")
        self._ensure_instance(instance)
        assert self._coords_tensor is not None
        assert self._solution_tensor is not None
        assert self._state is not None
        if not prefix:
            raise ValueError(f"{self.policy_id} neural adapter requires the caller-provided selected prefix")
        selected_prefix = tsp_decoder_selected_prefix(instance, prefix, allowed_candidates)
        selected = torch.tensor(
            [[self._node_to_index[node] for node in selected_prefix]],
            dtype=torch.long,
            device=self._coords_tensor.device,
        )
        with original_torch_device_context():
            with torch.no_grad():
                probs = tsp_neural_action_probabilities(
                    self._spec,
                    self._state,
                    selected,
                    self._solution_tensor,
                    len(prefix),
                    repair=repair,
                )
        return probs.reshape(-1).detach().to(device="cpu", dtype=torch.float64).tolist()

    def action_probability_vectors_batch(
        self,
        instance: Instance,
        prefixes: Sequence[Sequence[int]],
        *,
        repair: bool = False,
        allowed_candidates: Sequence[Sequence[int] | None] | None = None,
    ) -> list[list[float]]:
        if instance.problem != "tsp":
            raise ValueError(f"{self.policy_id} neural adapter is currently extracted for TSP only")
        if self.policy_id not in {"lehd", "sil"}:
            raise ValueError(f"batched STAR TSP neural decode is currently implemented for LEHD/SIL, not {self.policy_id}")
        if not prefixes:
            return []
        self._ensure_instance(instance)
        assert self._coords_tensor is not None
        assert self._solution_tensor is not None
        assert self._state is not None
        if allowed_candidates is None:
            allowed_candidates = [None] * len(prefixes)
        if len(allowed_candidates) != len(prefixes):
            raise ValueError("allowed candidate batch size does not match prefix batch size")

        selected_prefixes: list[list[int]] = []
        for prefix, allowed in zip(prefixes, allowed_candidates):
            if not prefix:
                raise ValueError(f"{self.policy_id} neural adapter requires the caller-provided selected prefix")
            selected_prefixes.append(tsp_decoder_selected_prefix(instance, prefix, allowed))

        results: list[list[float] | None] = [None] * len(prefixes)
        groups: dict[tuple[int, int], list[int]] = {}
        for index, selected_prefix in enumerate(selected_prefixes):
            groups.setdefault((len(selected_prefix), len(prefixes[index])), []).append(index)

        for (_selected_len, current_step), group_indices in groups.items():
            selected = torch.tensor(
                [
                    [self._node_to_index[node] for node in selected_prefixes[index]]
                    for index in group_indices
                ],
                dtype=torch.long,
                device=self._coords_tensor.device,
            )
            with original_torch_device_context():
                with torch.no_grad():
                    probs = tsp_neural_action_probabilities_batch(
                        self._spec,
                        self._state,
                        selected,
                        current_step=current_step,
                        encoded=self._encoded_for_decode(),
                        repair=repair,
                    )
            rows = probs.detach().to(device="cpu", dtype=torch.float64)
            for row_index, original_index in enumerate(group_indices):
                results[original_index] = rows[row_index].reshape(-1).tolist()

        if any(row is None for row in results):
            raise RuntimeError("batched TSP neural probability extraction left an unset result row")
        return [row for row in results if row is not None]

    def action_candidate_probabilities_batch(
        self,
        instance: Instance,
        prefixes: Sequence[Sequence[int]],
        candidates: Sequence[Sequence[int]],
        *,
        repair: bool = False,
        allowed_candidates: Sequence[Sequence[int] | None] | None = None,
    ) -> list[list[float]]:
        if len(prefixes) != len(candidates):
            raise ValueError("candidate batch size does not match prefix batch size")
        if instance.problem != "tsp":
            raise ValueError(f"{self.policy_id} neural adapter is currently extracted for TSP only")
        if self.policy_id not in {"lehd", "sil"}:
            raise ValueError(f"batched STAR TSP neural decode is currently implemented for LEHD/SIL, not {self.policy_id}")
        if not prefixes:
            return []
        self._ensure_instance(instance)
        assert self._coords_tensor is not None
        assert self._state is not None
        if allowed_candidates is None:
            allowed_candidates = [None] * len(prefixes)
        if len(allowed_candidates) != len(prefixes):
            raise ValueError("allowed candidate batch size does not match prefix batch size")

        selected_prefixes: list[list[int]] = []
        for prefix, allowed in zip(prefixes, allowed_candidates):
            if not prefix:
                raise ValueError(f"{self.policy_id} neural adapter requires the caller-provided selected prefix")
            selected_prefixes.append(tsp_decoder_selected_prefix(instance, prefix, allowed))

        results: list[list[float] | None] = [None] * len(prefixes)
        groups: dict[tuple[int, int], list[int]] = {}
        for index, selected_prefix in enumerate(selected_prefixes):
            groups.setdefault((len(selected_prefix), len(prefixes[index])), []).append(index)

        for (_selected_len, current_step), group_indices in groups.items():
            selected = torch.tensor(
                [
                    [self._node_to_index[node] for node in selected_prefixes[index]]
                    for index in group_indices
                ],
                dtype=torch.long,
                device=self._coords_tensor.device,
            )
            with original_torch_device_context():
                with torch.no_grad():
                    max_candidates = max(len(candidates[index]) for index in group_indices)
                    probs = tsp_neural_action_probabilities_batch(
                        self._spec,
                        self._state,
                        selected,
                        current_step=current_step,
                        encoded=self._encoded_for_decode(),
                        candidate_k=max_candidates,
                        repair=repair,
                    )
                    padded_candidate_indices = []
                    for index in group_indices:
                        row = [self._node_to_index[node] for node in candidates[index]]
                        if not row:
                            raise ValueError("candidate probability extraction received an empty candidate row")
                        row = row + [row[-1]] * (max_candidates - len(row))
                        padded_candidate_indices.append(row)
                    candidate_indices = torch.tensor(padded_candidate_indices, dtype=torch.long, device=probs.device)
                    candidate_probs = probs.gather(1, candidate_indices)
            rows = candidate_probs.detach().to(device="cpu", dtype=torch.float64)
            for row_index, original_index in enumerate(group_indices):
                results[original_index] = rows[row_index, : len(candidates[original_index])].reshape(-1).tolist()

        if any(row is None for row in results):
            raise RuntimeError("batched TSP neural candidate probability extraction left an unset result row")
        return [row for row in results if row is not None]

    def _ensure_instance(self, instance: Instance) -> None:
        node_ids = sorted(instance.coords)
        if node_ids == self._node_ids and self._coords_tensor is not None:
            return
        raw = torch.tensor([instance.coords[node] for node in node_ids], dtype=torch.float32).unsqueeze(0)
        raw_dist = None
        if self.policy_id == "bq":
            raw64 = raw.to(dtype=torch.float64)
            raw_dist = torch.linalg.vector_norm(raw64[:, :, None, :] - raw64[:, None, :, :], dim=-1)
            if instance.edge_weight_type == "CEIL_2D":
                raw_dist = torch.ceil(raw_dist)
            elif instance.edge_weight_type == "EUC_2D":
                raw_dist = torch.floor(raw_dist + 0.5)
            raw_dist = raw_dist.to(dtype=torch.float32)
        normalized = normalize_coords_for_policy(raw, self.policy_id, "tsp")
        if CUDA_AVAILABLE:
            raw = raw.to(device=DEFAULT_TORCH_DEVICE)
            if raw_dist is not None:
                raw_dist = raw_dist.to(device=DEFAULT_TORCH_DEVICE)
            normalized = normalized.to(device=DEFAULT_TORCH_DEVICE)
        self._node_ids = node_ids
        self._node_to_index = {node: index for index, node in enumerate(node_ids)}
        self._coords_tensor = normalized
        self._solution_tensor = torch.arange(len(self._node_ids), dtype=torch.long, device=normalized.device).unsqueeze(0)
        self._state = SimpleNamespace(
            data=normalized,
            raw_coords=raw,
            raw_dist_matrix=raw_dist,
            edge_weight_type=instance.edge_weight_type,
            first_node=None,
            current_node=None,
        )
        self._encoded_tensor = None
        self._encoded_device = None

    def _encoded_for_decode(self) -> torch.Tensor | None:
        if self.policy_id not in {"lehd", "sil"}:
            return None
        assert self._state is not None
        coords = native_forward_swap._coords_from_state(self._state)
        target_device = native_forward_swap._forward_device(coords.device)
        if self._encoded_tensor is not None and self._encoded_device == target_device:
            return self._encoded_tensor
        neural_model = native_forward_swap._load_neural_model(self._spec, target_device)
        state_on_device = native_forward_swap._state_to_device(self._state, target_device)
        original_device = coords.device
        with original_torch_device_context():
            with torch.no_grad():
                with native_forward_swap._cpu_default_tensor_type(target_device, original_device):
                    self._encoded_tensor = neural_model.encoder(state_on_device.data)
        self._encoded_device = target_device
        return self._encoded_tensor


@dataclass
class NativeCVRPNeuralPolicy:
    """In-process wrapper around the original checkpoint-backed CVRP append model."""

    policy_id: str

    def __post_init__(self) -> None:
        spec = native_forward_swap._neural_policy(self.policy_id, "cvrp")
        if spec is None:
            raise ValueError(f"unknown neural CVRP append policy: {self.policy_id}")
        self._spec = spec
        self._node_ids: list[int] = []
        self._node_to_index: dict[int, int] = {}
        self._base_problems: torch.Tensor | None = None
        self._solution_tensor: torch.Tensor | None = None
        self._raw_capacity_tensor: torch.Tensor | None = None

    def select_next(
        self,
        instance: Instance,
        current: int,
        candidates: Sequence[int],
        rng: random.Random,
        prefix: Sequence[int],
        *,
        repair: bool = False,
    ) -> int:
        del instance, current, candidates, rng, prefix, repair
        raise ValueError(
            f"{self.policy_id} CVRP neural policy is exact-env only; use the extracted original CVRP route-state loop"
        )

    def select_next_with_flag(
        self,
        instance: Instance,
        candidates: Sequence[int],
        prefix: Sequence[int],
    ) -> tuple[int, int]:
        del instance, candidates, prefix
        raise ValueError(
            f"{self.policy_id} CVRP neural policy is exact-env only; use the extracted original CVRP route-state loop"
        )

    def action_probabilities(
        self,
        instance: Instance,
        prefix: Sequence[int],
        remaining_capacity: int,
    ) -> tuple[dict[int, float], dict[int, float]]:
        return self.action_probabilities_batch(instance, [prefix], [remaining_capacity])[0]

    def action_probabilities_batch(
        self,
        instance: Instance,
        prefixes: Sequence[Sequence[int]],
        remaining_capacities: Sequence[int],
    ) -> list[tuple[dict[int, float], dict[int, float]]]:
        rows = self.action_probability_rows_batch(instance, prefixes, remaining_capacities)
        self._ensure_instance(instance)
        split_line = len(self._node_ids) - 1
        result: list[tuple[dict[int, float], dict[int, float]]] = []
        for row in rows:
            direct: dict[int, float] = {}
            via_depot: dict[int, float] = {}
            for offset in range(split_line):
                node = self._node_ids[offset + 1]
                direct[node] = float(row[offset])
                via_depot[node] = float(row[split_line + offset])
            result.append((direct, via_depot))
        return result

    def action_probability_rows_batch(
        self,
        instance: Instance,
        prefixes: Sequence[Sequence[int]],
        remaining_capacities: Sequence[int],
    ) -> list[list[float]]:
        if self.policy_id != "sil":
            raise ValueError(f"CVRP probability extraction is currently implemented for SIL, not {self.policy_id}")
        if len(prefixes) != len(remaining_capacities):
            raise ValueError("prefix and remaining-capacity batch sizes must match")
        if not prefixes:
            return []
        self._ensure_instance(instance)
        if self._node_ids[0] != instance.depot:
            raise ValueError("SIL CVRP probability-row extraction expects the depot to be the first sorted node")
        assert self._base_problems is not None
        assert self._raw_capacity_tensor is not None
        for prefix in prefixes:
            if not prefix:
                raise ValueError("SIL CVRP probability extraction requires non-empty selected prefixes")
        lengths = {len(prefix) for prefix in prefixes}
        if len(lengths) > 1:
            grouped_results: list[list[float] | None] = [None] * len(prefixes)
            for length in sorted(lengths):
                indices = [index for index, prefix in enumerate(prefixes) if len(prefix) == length]
                rows = self.action_probability_rows_batch(
                    instance,
                    [prefixes[index] for index in indices],
                    [remaining_capacities[index] for index in indices],
                )
                for index, row in zip(indices, rows):
                    grouped_results[index] = row
            if any(row is None for row in grouped_results):
                raise RuntimeError("CVRP SIL probability batch grouping left an unset row")
            return [row for row in grouped_results if row is not None]

        max_prefix_len = max(len(prefix) for prefix in prefixes)
        selected_rows: list[list[int]] = []
        for prefix in prefixes:
            row = [self._node_to_index[node] for node in prefix]
            row = row + [row[-1]] * (max_prefix_len - len(row))
            selected_rows.append(row)
        problems = self._base_problems.expand(len(prefixes), -1, -1).clone()
        problems[:, :, 3] = torch.tensor(remaining_capacities, dtype=problems.dtype, device=problems.device).view(-1, 1)
        selected = torch.tensor(selected_rows, dtype=torch.long, device=problems.device)
        state = SimpleNamespace(problems=problems, first_node=None, current_node=None)
        original_device = problems.device
        target_device = native_forward_swap._forward_device(original_device)
        neural_model = native_forward_swap._load_neural_model(self._spec, target_device)
        state_on_device = native_forward_swap._state_to_device(state, target_device)
        selected_on_device = selected.to(device=target_device, dtype=torch.long)
        raw_capacity = self._raw_capacity_tensor.to(device=target_device)

        with original_torch_device_context():
            with torch.no_grad():
                with native_forward_swap._cpu_default_tensor_type(target_device, original_device):
                    capacity = float(raw_capacity.reshape(-1)[0].item())
                    encoded = neural_model.encoder(state_on_device.problems, capacity)
                    remaining = state_on_device.problems[:, 1, 3]
                    probs = neural_model.decoder(
                        encoded,
                        state_on_device.problems,
                        selected_on_device,
                        max_prefix_len,
                        capacity,
                        remaining,
                    )

        rows = probs.detach().to(device="cpu", dtype=torch.float64)
        split_line = len(self._node_ids) - 1
        if rows.size(1) < 2 * split_line:
            raise ValueError(f"SIL CVRP probability vector has unexpected length {rows.size(1)} for split {split_line}")
        return [row_tensor.reshape(-1).tolist() for row_tensor in rows]

    def _ensure_instance(self, instance: Instance) -> None:
        node_ids = sorted(instance.coords)
        if node_ids == self._node_ids and self._base_problems is not None:
            return
        raw = torch.tensor([instance.coords[node] for node in node_ids], dtype=torch.float32).unsqueeze(0)
        coords = normalize_coords_for_policy(raw, self.policy_id, "cvrp")
        if CUDA_AVAILABLE:
            coords = coords.to(device=DEFAULT_TORCH_DEVICE)
        demands = torch.tensor(
            [[instance.demands.get(node, 0) for node in node_ids]],
            dtype=torch.float32,
            device=coords.device,
        )
        capacities = torch.full(
            (1, len(node_ids)),
            float(instance.capacity or 0),
            dtype=torch.float32,
            device=coords.device,
        )
        self._node_ids = node_ids
        self._node_to_index = {node: index for index, node in enumerate(node_ids)}
        self._base_problems = torch.cat((coords, demands.unsqueeze(-1), capacities.unsqueeze(-1)), dim=2)
        self._solution_tensor = torch.zeros((1, len(node_ids) - 1, 2), dtype=torch.long, device=coords.device)
        self._raw_capacity_tensor = torch.tensor([float(instance.capacity or 0)], device=coords.device)


@dataclass
class CVRPDecision:
    node: int
    route_break: bool


@dataclass
class SparseEdgeMemory:
    """Sparse Smooth-MMAS-style edge memory over only k nearest outgoing edges."""

    instance: Instance
    k: int = 32
    rho: float = 0.9
    tau_min: float = 1.0 / 32.0
    tau_max: float = 1.0
    alpha: float = 1.0
    values: dict[tuple[int, int], float] = field(default_factory=dict)
    neighbor_sets: dict[int, set[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("memory k must be positive")
        if self.rho < 0.0 or self.rho > 1.0:
            raise ValueError("memory rho must be in [0, 1]")
        if self.tau_min <= 0.0 or self.tau_max <= 0.0 or self.tau_min > self.tau_max:
            raise ValueError("memory tau bounds must satisfy 0 < tau_min <= tau_max")
        if self.alpha < 0.0:
            raise ValueError("memory alpha must be non-negative")
        self.values = {}
        self.neighbor_sets = {}
        node_ids = sorted(self.instance.coords)
        neighbor_order = STAR_neighbor_order(self.instance, self.k)
        neighbor_rows = zip(node_ids, neighbor_order)
        for node, neighbors in neighbor_rows:
            self.neighbor_sets[node] = set(neighbors)
            for other in neighbors:
                self.values[(node, other)] = 1.0

    def weight(self, current: int, picked: int) -> float:
        tau = self.values.get((current, picked), 1.0)
        if self.alpha == 1.0:
            return tau
        return tau ** self.alpha

    def update_from_tsp(self, route: Sequence[int]) -> None:
        best_edges: set[tuple[int, int]] = set()
        for index, u in enumerate(route):
            v = route[(index + 1) % len(route)]
            best_edges.add((u, v))
            best_edges.add((v, u))
        self._smooth_update(best_edges)

    def update_from_cvrp(self, routes: Sequence[Sequence[int]]) -> None:
        best_edges: set[tuple[int, int]] = set()
        depot = self.instance.depot
        for route in routes:
            previous = depot
            for node in route:
                best_edges.add((previous, node))
                best_edges.add((node, previous))
                previous = node
            best_edges.add((previous, depot))
            best_edges.add((depot, previous))
        self._smooth_update(best_edges)

    def _smooth_update(self, best_edges: set[tuple[int, int]]) -> None:
        decay = 1.0 - self.rho
        for edge, tau in list(self.values.items()):
            target = self.tau_max if edge in best_edges else self.tau_min
            self.values[edge] = decay * tau + self.rho * target

    def update_from_advantage_edges(self, edges: set[tuple[int, int]], strength: float) -> int:
        strength = max(0.0, min(1.0, float(strength)))
        target_high = self.tau_min + strength * (self.tau_max - self.tau_min)
        decay = 1.0 - self.rho
        reinforced = 0
        for edge, tau in list(self.values.items()):
            if edge in edges:
                self.values[edge] = decay * tau + self.rho * target_high
                reinforced += 1
            else:
                self.values[edge] = decay * tau + self.rho * self.tau_min
        return reinforced

    def boost_advantage_edges(self, edges: set[tuple[int, int]], strength: float) -> int:
        strength = max(0.0, min(1.0, float(strength)))
        reinforced = 0
        for edge in edges:
            if edge not in self.values:
                continue
            current = self.values[edge]
            self.values[edge] = current + strength * (self.tau_max - current)
            reinforced += 1
        return reinforced


@dataclass
class TspStartInfo:
    mode: str
    node: int
    successor: int
    score: float = 0.0
    cost_score: float = 0.0
    policy_score: float = 0.0
    memory_score: float = 0.0
    successor_prob: float | None = None
    best_alt_prob: float | None = None


@dataclass
class TspPerturbResult:
    route: list[int]
    changed: set[int]
    introduced_edges: set[tuple[int, int]] = field(default_factory=set)
    removed_edges: set[tuple[int, int]] = field(default_factory=set)
    start_info: TspStartInfo | None = None

    def __iter__(self):
        yield self.route
        yield self.changed


@dataclass
class TspRefinedCandidate:
    route: list[int]
    cost: float
    perturb: TspPerturbResult


@dataclass
class CvrpPerturbResult:
    routes: list[list[int]]
    changed: set[int]
    introduced_edges: set[tuple[int, int]] = field(default_factory=set)
    removed_edges: set[tuple[int, int]] = field(default_factory=set)

    def __iter__(self):
        yield self.routes
        yield self.changed


@dataclass
class CvrpRefinedCandidate:
    routes: list[list[int]]
    cost: float
    perturb: CvrpPerturbResult


@dataclass
class CvrpPerturbState:
    candidate: list[list[int]]
    source_route_id: dict[int, int]
    source_edges: set[tuple[int, int]]
    source_memory_edges: set[tuple[int, int]]
    current: int
    remaining_capacity: int
    prefix: list[int]
    visited: set[int]
    changed: set[int]
    new_edges_cross: int = 0
    steps: int = 0
    done: bool = False


@dataclass(frozen=True)
class UnsupportedPolicy:
    policy_id: str
    reason: str


def append_policy(policy_id: str, problem: str | None = None) -> AppendPolicy | UnsupportedPolicy:
    if policy_id == "nearest":
        return NearestPolicy()
    if policy_id == "softdist":
        return SoftDistPolicy()
    if policy_id == "random_insertion":
        return UnsupportedPolicy(policy_id, "random insertion is an insertion constructor, not an append-forward policy")
    if problem == "cvrp" and policy_id in {"bq", "lehd", "sil", "icam", "elg"}:
        return NativeCVRPNeuralPolicy(policy_id)
    if problem == "cvrp" and policy_id in {"drhg", "invit", "dgl", "reld"}:
        return UnsupportedPolicy(
            policy_id,
            "exact original CVRP route-state loop has not been extracted for this policy; refusing generic neural forward",
        )
    if problem == "tsp" and policy_id in {"bq", "lehd", "sil", "icam", "elg"}:
        return NativeTSPNeuralPolicy(policy_id)
    if problem == "tsp" and policy_id in {"drhg", "invit", "dgl", "reld"}:
        return UnsupportedPolicy(
            policy_id,
            "exact original TSP route-state loop has not been extracted for this policy; refusing generic neural forward",
        )
    return UnsupportedPolicy(policy_id, f"unknown append policy: {policy_id}")


def append_policy_ids() -> list[str]:
    return ["nearest", "softdist", "random_insertion", "bq", "lehd", "sil", "drhg", "icam", "elg", "invit", "dgl", "reld"]


def tsp_decoder_selected_prefix(
    instance: Instance,
    prefix: Sequence[int],
    allowed_candidates: Sequence[int] | None = None,
) -> list[int]:
    if allowed_candidates is None:
        return list(prefix)
    if not prefix:
        raise ValueError("TSP neural adapter requires a non-empty selected prefix")
    if len(prefix) == 1:
        return list(prefix)

    prefix_set = set(prefix)
    allowed = set(allowed_candidates)
    forbidden = [
        node
        for node in sorted(instance.coords)
        if node not in allowed and node not in prefix_set
    ]
    return [prefix[0]] + forbidden + list(prefix[1:])


def tsp_neural_action_probabilities(
    spec: native_forward_swap.NeuralPolicySpec,
    state: Any,
    selected_node_list: torch.Tensor,
    solution: torch.Tensor,
    current_step: int,
    *,
    repair: bool = False,
) -> torch.Tensor:
    del solution, repair
    coords = native_forward_swap._coords_from_state(state)
    original_device = coords.device
    target_device = native_forward_swap._forward_device(original_device)
    if spec.family == "invit":
        raise ValueError("STAR memory needs next-node probabilities; INViT TSP path currently exposes only a decoded tour")
    neural_model = native_forward_swap._load_neural_model(spec, target_device)
    state_on_device = native_forward_swap._state_to_device(state, target_device)
    selected = selected_node_list.to(device=target_device, dtype=torch.long)

    with native_forward_swap._cpu_default_tensor_type(target_device, original_device):
        if spec.family == "bq":
            coords_on_device = native_forward_swap._coords_from_state(state_on_device)
            batch, nodes, _ = coords_on_device.shape
            selected_bq = selected
            if selected_bq.numel() == 0 or selected_bq.size(1) == 0:
                selected_bq = torch.zeros(batch, 1, dtype=torch.long, device=coords_on_device.device)
            visited = native_forward_swap._visited_mask(nodes, selected_bq, 0)
            local_inputs, local_to_global = native_forward_swap._bq_local_tsp_inputs(
                coords_on_device,
                selected_bq,
                visited,
                getattr(state_on_device, "raw_coords", coords_on_device),
                getattr(state_on_device, "edge_weight_type", "EUC_2D"),
                getattr(state_on_device, "raw_dist_matrix", None),
            )
            scores = neural_model(local_inputs)
            local_probs = torch.softmax(scores, dim=1)
            probs = torch.zeros(batch, nodes, dtype=local_probs.dtype, device=coords_on_device.device)
            probs.scatter_(1, local_to_global.to(device=coords_on_device.device), local_probs)
            return probs.to(device=original_device)

        if spec.family == "lehd":
            encoded = neural_model.encoder(state_on_device.data)
            return neural_model.decoder(encoded, selected).to(device=original_device)

        if spec.family == "sil":
            encoded = neural_model.encoder(state_on_device.data)
            batch = state_on_device.data.size(0)
            problem_size = state_on_device.data.size(1)
            probs = neural_model.decoder(
                encoded,
                state_on_device.data,
                state_on_device.first_node,
                state_on_device.current_node,
                selected,
                current_step,
                batch,
                problem_size,
                mode="test",
            )
            return probs.to(device=original_device)

        if spec.family == "icam":
            coords_on_device = native_forward_swap._coords_from_state(state_on_device)
            selected_2d = native_forward_swap._selected_2d(selected, coords_on_device.device)
            reset_state = SimpleNamespace(
                problems=coords_on_device,
                dist=native_forward_swap._pairwise_distances(coords_on_device),
                log_scale=math.log2(max(2, coords_on_device.size(1))),
            )
            neural_model.pre_forward(reset_state)
            native_forward_swap._set_tsp_decoder_first_query(neural_model, selected_2d)
            step_state = native_forward_swap._tsp_step_state(coords_on_device, selected_2d)
            cur_dist, _cur_theta, _relative_xy = native_forward_swap._tsp_local_features(coords_on_device, selected_2d)
            current_node = step_state.current_node
            encoded_last_node = neural_model.encoded_nodes.gather(
                1,
                current_node[:, :, None].expand(current_node.size(0), current_node.size(1), neural_model.encoded_nodes.size(2)),
            )
            probs = neural_model.decoder(encoded_last_node, cur_dist, neural_model.log_scale, ninf_mask=step_state.ninf_mask)
            return probs.squeeze(1).to(device=original_device)

        if spec.family == "elg":
            coords_on_device = native_forward_swap._coords_from_state(state_on_device)
            selected_2d = native_forward_swap._selected_2d(selected, coords_on_device.device)
            neural_model.pre_forward(SimpleNamespace(problems=coords_on_device))
            native_forward_swap._set_tsp_decoder_first_query(neural_model, selected_2d)
            step_state = native_forward_swap._tsp_step_state(coords_on_device, selected_2d)
            cur_dist, cur_theta, relative_xy = native_forward_swap._tsp_local_features(coords_on_device, selected_2d)
            current_node = step_state.current_node
            encoded_last_node = neural_model.encoded_nodes.gather(
                1,
                current_node[:, :, None].expand(current_node.size(0), current_node.size(1), neural_model.encoded_nodes.size(2)),
            )
            probs = neural_model.decoder(
                encoded_last_node,
                cur_dist=cur_dist,
                cur_theta=cur_theta,
                xy=relative_xy,
                ninf_mask=step_state.ninf_mask,
            )
            return probs.squeeze(1).to(device=original_device)

    raise ValueError(f"STAR memory probability extraction is not implemented for TSP policy family {spec.family}")


def tsp_neural_action_probabilities_batch(
    spec: native_forward_swap.NeuralPolicySpec,
    state: Any,
    selected_node_list: torch.Tensor,
    *,
    current_step: int,
    encoded: torch.Tensor | None = None,
    candidate_k: int | None = None,
    repair: bool = False,
) -> torch.Tensor:
    del repair
    if spec.family not in {"lehd", "sil"}:
        raise ValueError(f"batched TSP neural probability extraction is implemented for LEHD/SIL, not {spec.family}")
    coords = native_forward_swap._coords_from_state(state)
    original_device = coords.device
    target_device = native_forward_swap._forward_device(original_device)
    neural_model = native_forward_swap._load_neural_model(spec, target_device)
    state_on_device = native_forward_swap._state_to_device(state, target_device)
    selected = selected_node_list.to(device=target_device, dtype=torch.long)

    with native_forward_swap._cpu_default_tensor_type(target_device, original_device):
        if encoded is None:
            encoded = neural_model.encoder(state_on_device.data)
        else:
            encoded = encoded.to(device=target_device)
        if encoded.size(0) != selected.size(0):
            encoded = encoded.expand(selected.size(0), -1, -1)
        if spec.family == "lehd":
            return neural_model.decoder(encoded, selected).to(device=original_device)
        data = state_on_device.data
        if data.size(0) != selected.size(0):
            data = data.expand(selected.size(0), -1, -1)
        batch = data.size(0)
        problem_size = data.size(1)
        decoder_params = getattr(neural_model.decoder, "model_params", None)
        old_k_nearest = None
        old_use_k_nearest = None
        if candidate_k is not None and isinstance(decoder_params, dict):
            old_k_nearest = decoder_params.get("k_nearest_num")
            old_use_k_nearest = decoder_params.get("use_k_nearest")
            decoder_params["k_nearest_num"] = max(1, int(candidate_k))
            decoder_params["use_k_nearest"] = True
        try:
            probs = neural_model.decoder(
                encoded,
                data,
                getattr(state_on_device, "first_node", None),
                getattr(state_on_device, "current_node", None),
                selected,
                current_step,
                batch,
                problem_size,
                mode="test",
            )
        finally:
            if candidate_k is not None and isinstance(decoder_params, dict):
                decoder_params["k_nearest_num"] = old_k_nearest
                decoder_params["use_k_nearest"] = old_use_k_nearest
        return probs.to(device=original_device)


class SearchStrategy(Protocol):
    strategy_id: str

    def run(self, instance: Instance, policy: AppendPolicy, rng: random.Random) -> tuple[float, float, bool]:
        ...


@dataclass
class STARStrategy:
    """STAR: Scoped Test-time Adapt and Refine."""

    strategy_id: str = "STAR"
    iterations: int = 100
    min_new_edges: int = 24
    refine_k: int = 64
    refine: bool = True
    neural_knn_k: int = 32
    neural_backup_k: int = 32
    neural_knn_mask: bool = True
    STAR_samples: int = 64
    memory: bool = True
    memory_k: int = 32
    memory_rho: float = 0.9
    memory_tau_min: float | None = None
    memory_tau_max: float = 1.0
    memory_alpha: float = 1.0
    start_mode: str = "random"
    start_probes: int = 32
    start_cost_weight: float = 1.0
    start_policy_weight: float = 1.0
    start_memory_weight: float = 1.0
    memory_update_mode: str = "auto"
    advantage_scale: float = 100.0
    advantage_min: float = 0.0
    STAR_trace: bool = False
    STAR_profile: bool = False
    STAR_trace_rows: list[dict[str, str]] = field(default_factory=list)
    STAR_profile_rows: list[dict[str, str]] = field(default_factory=list)
    STAR_progress_callback: Callable[[dict[str, str]], None] | None = field(default=None, repr=False)

    def run(self, instance: Instance, policy: AppendPolicy, rng: random.Random) -> tuple[float, float, bool]:
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if self.min_new_edges < 0:
            raise ValueError("min-new-edges must be non-negative")
        if self.refine and self.refine_k <= 0:
            raise ValueError("refine-k must be positive")
        if self.neural_knn_k <= 0:
            raise ValueError("neural-knn-k must be positive")
        if self.neural_backup_k < 0:
            raise ValueError("neural-backup-k must be non-negative")
        if self.STAR_samples <= 0:
            raise ValueError("STAR-samples must be positive")
        if self.memory and self.memory_k <= 0:
            raise ValueError("memory-k must be positive")
        if self.start_mode not in {"random", "cost", "policy-disagreement", "hybrid"}:
            raise ValueError(f"unknown STAR-start-mode: {self.start_mode}")
        if self.start_probes <= 0:
            raise ValueError("STAR-start-probes must be positive")
        if self.memory_update_mode not in {"auto", "source", "advantage-introduced", "source-advantage"}:
            raise ValueError(f"unknown STAR-memory-update-mode: {self.memory_update_mode}")
        effective_memory_update_mode = self.effective_memory_update_mode(instance)
        if self.advantage_scale < 0.0:
            raise ValueError("STAR-advantage-scale must be non-negative")
        if self.advantage_min < 0.0:
            raise ValueError("STAR-advantage-min must be non-negative")
        if instance.problem != "tsp" and self.start_mode != "random":
            raise ValueError("STAR start modes other than random are currently implemented for TSP only")
        if instance.problem not in {"tsp", "cvrp"} and effective_memory_update_mode != "source":
            raise ValueError("STAR advantage memory updates are currently implemented for TSP/CVRP only")
        if effective_memory_update_mode != "source" and not self.memory:
            raise ValueError("STAR advantage memory update modes require STAR memory")
        if self.start_mode == "policy-disagreement" and self.start_policy_weight > 0.0 and not isinstance(policy, NativeTSPNeuralPolicy):
            raise ValueError(f"STAR-start-mode {self.start_mode} with policy weight requires a TSP neural policy")
        if self.start_mode == "hybrid" and self.start_memory_weight > 0.0 and not self.memory:
            raise ValueError("STAR-start-mode hybrid with memory weight requires STAR memory")
        run_t0 = time.perf_counter()

        if instance.problem == "tsp":
            route = tsp_initial_multi_start_cpp(instance, starts=8, k=self.neural_knn_k)
            initial = tsp_cost(instance, route)
            if self.use_cpp_nearest_tsp(policy):
                context = STAR_context(instance)
                best = list(
                    context.run_nearest_tsp(
                        route,
                        self.iterations,
                        self.min_new_edges,
                        self.refine_k,
                        self.neural_knn_k,
                        self.neural_backup_k,
                        int(rng.randrange(0, 2**63)),
                        self.refine,
                    )
                )
                return initial, tsp_cost(instance, best), validate_tsp(instance, best)
            source = route
            best = route
            best_cost = initial
            source_cost = initial
            self.record_STAR_progress(
                instance,
                policy,
                completed_iterations=0,
                iteration=-1,
                source_cost=source_cost,
                best_cost=best_cost,
                elapsed_seconds=time.perf_counter() - run_t0,
                iteration_seconds=0.0,
            )
            edge_memory = self.make_edge_memory(instance) if self.memory else None
            neighbor_order = STAR_neighbor_order(
                instance,
                max(
                    self.memory_k if self.memory else 0,
                    self.neural_knn_k + self.neural_backup_k,
                    self.refine_k if self.refine else 0,
                ),
            )
            if edge_memory is not None:
                edge_memory.update_from_tsp(source)
            if (
                self.STAR_samples > 1
                and isinstance(policy, NearestPolicy)
                and edge_memory is None
                and STAR is not None
            ):
                context = STAR_context(instance)
                best = list(
                    context.run_nearest_tsp_samples(
                        best,
                        self.iterations,
                        self.min_new_edges,
                        self.STAR_samples,
                        self.refine_k,
                        self.neural_knn_k,
                        self.neural_backup_k,
                        int(rng.randrange(0, 2**63)),
                        self.refine,
                    )
                )
                return initial, tsp_cost(instance, best), validate_tsp(instance, best)
            for iteration in range(self.iterations):
                iter_t0 = time.perf_counter()
                perturb_profile: dict[str, Any] | None = {
                    "instance": instance.name,
                    "problem": instance.problem,
                    "policy_id": getattr(policy, "policy_id", str(policy)),
                    "iteration": str(iteration),
                    "samples": str(self.STAR_samples),
                    "min_new_edges": str(self.min_new_edges),
                    "refine_k": str(self.refine_k),
                    "neural_knn_k": str(self.neural_knn_k),
                    "neural_backup_k": str(self.neural_backup_k),
                } if self.STAR_profile else None
                if isinstance(policy, NativeTSPNeuralPolicy) and policy.policy_id in {"lehd", "sil"}:
                    candidates = perturb_tsp_batched_neural(
                        instance,
                        source,
                        policy,
                        rng,
                        self.min_new_edges,
                        self.STAR_samples,
                        edge_memory=edge_memory,
                        neural_knn_k=self.neural_knn_k,
                        neural_backup_k=self.neural_backup_k,
                        neural_knn_mask=self.neural_knn_mask,
                        start_mode=self.start_mode,
                        start_probes=self.start_probes,
                        start_cost_weight=self.start_cost_weight,
                        start_policy_weight=self.start_policy_weight,
                        start_memory_weight=self.start_memory_weight,
                        profile=perturb_profile,
                        neighbor_order=neighbor_order,
                    )
                elif self.STAR_samples > 1:
                    if not isinstance(policy, NearestPolicy):
                        raise ValueError(
                            "STAR-samples > 1 is currently implemented only for STAR + nearest/lehd/sil + tsp; "
                            f"got policy {getattr(policy, 'policy_id', policy)}"
                        )
                    candidates = perturb_tsp_batched_nearest(
                        instance,
                        source,
                        policy,
                        rng,
                        self.min_new_edges,
                        self.STAR_samples,
                        edge_memory=edge_memory,
                        neural_knn_k=self.neural_knn_k,
                        neural_backup_k=self.neural_backup_k,
                        start_mode=self.start_mode,
                        start_probes=self.start_probes,
                        start_cost_weight=self.start_cost_weight,
                        start_policy_weight=self.start_policy_weight,
                        start_memory_weight=self.start_memory_weight,
                        profile=perturb_profile,
                        neighbor_order=neighbor_order,
                    )
                else:
                    perturb = perturb_tsp(
                        instance,
                        source,
                        policy,
                        rng,
                        self.min_new_edges,
                        edge_memory=edge_memory,
                        neural_knn_k=self.neural_knn_k,
                        neural_backup_k=self.neural_backup_k,
                        neural_knn_mask=self.neural_knn_mask,
                        start_mode=self.start_mode,
                        start_probes=self.start_probes,
                        start_cost_weight=self.start_cost_weight,
                        start_policy_weight=self.start_policy_weight,
                        start_memory_weight=self.start_memory_weight,
                    )
                    candidates = [perturb]
                source_before_cost = source_cost if "source_cost" in locals() else tsp_cost(instance, source)
                refine_t0 = time.perf_counter()
                refined_candidates = refine_tsp_candidates_local(
                    instance,
                    candidates,
                    self.refine_k,
                    refine=self.refine,
                    neighbor_order=neighbor_order,
                )
                refine_time = time.perf_counter() - refine_t0
                update_time = 0.0
                if refined_candidates:
                    update_t0 = time.perf_counter()
                    self.update_tsp_memory_from_candidates(edge_memory, source_before_cost, refined_candidates)
                    update_time += time.perf_counter() - update_t0
                    if self.STAR_trace:
                        self.record_tsp_STAR_trace(instance, policy, iteration, source_before_cost, refined_candidates)
                    selected = min(refined_candidates, key=lambda item: item.cost)
                    source, source_cost = selected.route, selected.cost
                    if source_cost <= best_cost + 1e-9:
                        best = list(source)
                        best_cost = source_cost
                if edge_memory is not None and effective_memory_update_mode == "source":
                    update_t0 = time.perf_counter()
                    edge_memory.update_from_tsp(source)
                    update_time += time.perf_counter() - update_t0
                if perturb_profile is not None:
                    perturb_profile["refine_seconds"] = f"{refine_time:.9f}"
                    perturb_profile["memory_update_seconds"] = f"{update_time:.9f}"
                    perturb_profile["iteration_seconds"] = f"{time.perf_counter() - iter_t0:.9f}"
                    self.STAR_profile_rows.append({key: str(value) for key, value in perturb_profile.items()})
                iter_seconds = time.perf_counter() - iter_t0
                self.record_STAR_progress(
                    instance,
                    policy,
                    completed_iterations=iteration + 1,
                    iteration=iteration,
                    source_cost=source_cost,
                    best_cost=best_cost,
                    elapsed_seconds=time.perf_counter() - run_t0,
                    iteration_seconds=iter_seconds,
                )
            return initial, best_cost, validate_tsp(instance, best)

        if instance.problem == "cvrp":
            routes = greedy_cvrp_policy_multi_start(instance, policy, rng, starts=8)
            initial = cvrp_cost(instance, routes)
            source_routes = routes
            source_cost = initial
            best_routes = routes
            best_cost = initial
            self.record_STAR_progress(
                instance,
                policy,
                completed_iterations=0,
                iteration=-1,
                source_cost=source_cost,
                best_cost=best_cost,
                elapsed_seconds=time.perf_counter() - run_t0,
                iteration_seconds=0.0,
            )
            edge_memory = self.make_edge_memory(instance) if self.memory else None
            neighbor_order = STAR_neighbor_order(
                instance,
                max(
                    self.memory_k if self.memory else 0,
                    self.neural_knn_k + self.neural_backup_k,
                ),
            )
            for iteration in range(self.iterations):
                iter_t0 = time.perf_counter()
                perturb_profile: dict[str, Any] | None = {
                    "instance": instance.name,
                    "problem": instance.problem,
                    "policy_id": getattr(policy, "policy_id", str(policy)),
                    "iteration": str(iteration),
                    "samples": str(self.STAR_samples),
                    "min_new_edges": str(self.min_new_edges),
                    "refine_k": str(self.refine_k),
                    "neural_knn_k": str(self.neural_knn_k),
                    "neural_backup_k": str(self.neural_backup_k),
                } if self.STAR_profile else None
                if isinstance(policy, NativeCVRPNeuralPolicy) and policy.policy_id == "sil":
                    perturb_candidates = perturb_cvrp_batched_sil(
                        instance,
                        source_routes,
                        policy,
                        rng,
                        self.min_new_edges,
                        self.STAR_samples,
                        edge_memory=edge_memory,
                        neural_knn_k=self.neural_knn_k,
                        neural_backup_k=self.neural_backup_k,
                        profile=perturb_profile,
                        neighbor_order=neighbor_order,
                    )
                else:
                    perturb_candidates = [
                        perturb_cvrp(instance, source_routes, policy, rng, self.min_new_edges, edge_memory=edge_memory)
                        for _sample in range(self.STAR_samples)
                    ]
                refine_t0 = time.perf_counter()
                refined_candidates = refine_cvrp_candidates_local(
                    instance,
                    perturb_candidates,
                    self.refine_k,
                    refine=self.refine,
                )
                refine_time = time.perf_counter() - refine_t0
                update_time = 0.0
                valid_candidates = [candidate for candidate in refined_candidates if validate_cvrp(instance, candidate.routes)]
                if valid_candidates:
                    update_t0 = time.perf_counter()
                    self.update_cvrp_memory_from_candidates(edge_memory, source_cost, valid_candidates)
                    update_time += time.perf_counter() - update_t0
                    selected = min(valid_candidates, key=lambda item: item.cost)
                    source_routes, source_cost = selected.routes, selected.cost
                    if source_cost <= best_cost + 1e-9:
                        best_routes = [list(route) for route in source_routes]
                        best_cost = source_cost
                if edge_memory is not None and effective_memory_update_mode == "source":
                    update_t0 = time.perf_counter()
                    edge_memory.update_from_cvrp(source_routes)
                    update_time += time.perf_counter() - update_t0
                if perturb_profile is not None:
                    perturb_profile["refine_seconds"] = f"{refine_time:.9f}"
                    perturb_profile["memory_update_seconds"] = f"{update_time:.9f}"
                    perturb_profile["iteration_seconds"] = f"{time.perf_counter() - iter_t0:.9f}"
                    self.STAR_profile_rows.append({key: str(value) for key, value in perturb_profile.items()})
                iter_seconds = time.perf_counter() - iter_t0
                self.record_STAR_progress(
                    instance,
                    policy,
                    completed_iterations=iteration + 1,
                    iteration=iteration,
                    source_cost=source_cost,
                    best_cost=best_cost,
                    elapsed_seconds=time.perf_counter() - run_t0,
                    iteration_seconds=iter_seconds,
                )
            return initial, best_cost, validate_cvrp(instance, best_routes)

        raise ValueError(f"unsupported problem for STAR: {instance.problem}")

    def record_STAR_progress(
        self,
        instance: Instance,
        policy: AppendPolicy,
        *,
        completed_iterations: int,
        iteration: int,
        source_cost: float,
        best_cost: float,
        elapsed_seconds: float,
        iteration_seconds: float,
    ) -> None:
        if self.STAR_progress_callback is None:
            return
        best_gap = ((best_cost - instance.bks_cost) / instance.bks_cost * 100) if instance.bks_cost else ""
        self.STAR_progress_callback(
            {
                "strategy_id": self.strategy_id,
                "policy_id": getattr(policy, "policy_id", str(policy)),
                "problem": instance.problem,
                "instance": instance.name,
                "completed_iterations": str(completed_iterations),
                "iteration": str(iteration),
                "total_iterations": str(self.iterations),
                "samples": str(self.STAR_samples),
                "min_new_edges": str(self.min_new_edges),
                "refine_k": str(self.refine_k),
                "refine": str(self.refine),
                "memory": str(self.memory),
                "memory_update_mode": self.effective_memory_update_mode(instance),
                "advantage_scale": f"{self.advantage_scale:.12g}",
                "source_cost": f"{source_cost:.6f}",
                "best_cost": f"{best_cost:.6f}",
                "best_gap": f"{best_gap:.6f}" if isinstance(best_gap, float) else "",
                "elapsed_seconds": f"{elapsed_seconds:.6f}",
                "iteration_seconds": f"{iteration_seconds:.6f}",
            }
        )

    def make_edge_memory(self, instance: Instance) -> SparseEdgeMemory:
        tau_min = self.memory_tau_min if self.memory_tau_min is not None else 1.0 / max(1, self.memory_k)
        return SparseEdgeMemory(
            instance=instance,
            k=self.memory_k,
            rho=self.memory_rho,
            tau_min=tau_min,
            tau_max=self.memory_tau_max,
            alpha=self.memory_alpha,
        )

    def effective_memory_update_mode(self, instance: Instance) -> str:
        if not self.memory:
            return "source"
        if self.memory_update_mode == "auto":
            return "advantage-introduced" if instance.problem in {"tsp", "cvrp"} else "source"
        return self.memory_update_mode

    def update_tsp_memory_from_candidates(
        self,
        edge_memory: SparseEdgeMemory | None,
        source_cost: float,
        refined_candidates: Sequence[TspRefinedCandidate],
    ) -> None:
        mode = self.effective_memory_update_mode(edge_memory.instance) if edge_memory is not None else self.memory_update_mode
        if edge_memory is None or mode == "source":
            return
        if mode == "source-advantage":
            selected = min(refined_candidates, key=lambda item: item.cost)
            edge_memory.update_from_tsp(selected.route)
        positive_edges: set[tuple[int, int]] = set()
        best_strength = 0.0
        for candidate in refined_candidates:
            if source_cost <= 0.0:
                continue
            advantage = (source_cost - candidate.cost) / source_cost
            if advantage <= self.advantage_min:
                continue
            positive_edges.update(candidate.perturb.introduced_edges)
            best_strength = max(best_strength, advantage * self.advantage_scale)
        if positive_edges:
            if mode == "source-advantage":
                edge_memory.boost_advantage_edges(positive_edges, best_strength)
            else:
                edge_memory.update_from_advantage_edges(positive_edges, best_strength)

    def update_cvrp_memory_from_candidates(
        self,
        edge_memory: SparseEdgeMemory | None,
        source_cost: float,
        refined_candidates: Sequence[CvrpRefinedCandidate],
    ) -> None:
        mode = self.effective_memory_update_mode(edge_memory.instance) if edge_memory is not None else self.memory_update_mode
        if edge_memory is None or mode == "source":
            return
        if mode == "source-advantage":
            selected = min(refined_candidates, key=lambda item: item.cost)
            edge_memory.update_from_cvrp(selected.routes)
        positive_edges: set[tuple[int, int]] = set()
        best_strength = 0.0
        for candidate in refined_candidates:
            if source_cost <= 0.0:
                continue
            advantage = (source_cost - candidate.cost) / source_cost
            if advantage <= self.advantage_min:
                continue
            positive_edges.update(candidate.perturb.introduced_edges)
            best_strength = max(best_strength, advantage * self.advantage_scale)
        if positive_edges:
            if mode == "source-advantage":
                edge_memory.boost_advantage_edges(positive_edges, best_strength)
            else:
                edge_memory.update_from_advantage_edges(positive_edges, best_strength)

    def record_tsp_STAR_trace(
        self,
        instance: Instance,
        policy: AppendPolicy,
        iteration: int,
        source_cost: float,
        refined_candidates: Sequence[TspRefinedCandidate],
    ) -> None:
        for sample_index, candidate in enumerate(refined_candidates):
            info = candidate.perturb.start_info
            advantage = (source_cost - candidate.cost) / source_cost if source_cost > 0.0 else 0.0
            self.STAR_trace_rows.append(
                {
                    "instance": instance.name,
                    "iteration": str(iteration),
                    "sample": str(sample_index),
                    "policy_id": getattr(policy, "policy_id", str(policy)),
                    "start_mode": info.mode if info is not None else self.start_mode,
                    "start_node": str(info.node) if info is not None else "",
                    "successor": str(info.successor) if info is not None else "",
                    "start_score": f"{info.score:.12g}" if info is not None else "",
                    "cost_score": f"{info.cost_score:.12g}" if info is not None else "",
                    "policy_score": f"{info.policy_score:.12g}" if info is not None else "",
                    "memory_score": f"{info.memory_score:.12g}" if info is not None else "",
                    "successor_prob": f"{info.successor_prob:.12g}" if info is not None and info.successor_prob is not None else "",
                    "best_alt_prob": f"{info.best_alt_prob:.12g}" if info is not None and info.best_alt_prob is not None else "",
                    "source_cost": f"{source_cost:.6f}",
                    "refined_cost": f"{candidate.cost:.6f}",
                    "advantage": f"{advantage:.12g}",
                    "introduced_edges": str(len(candidate.perturb.introduced_edges)),
                    "removed_edges": str(len(candidate.perturb.removed_edges)),
                    "changed_nodes": str(len(candidate.perturb.changed)),
                    "memory_update_mode": self.effective_memory_update_mode(instance),
                }
            )

    def use_cpp_nearest_tsp(self, policy: AppendPolicy) -> bool:
        backend = os.environ.get("NRS_STAR_NEAREST_TSP_BACKEND", "python").strip().lower()
        if backend in {"python", "py"}:
            return False
        if backend not in {"cpp", "c++"}:
            raise ValueError(f"unknown NRS_STAR_NEAREST_TSP_BACKEND: {backend}")
        return (
            STAR is not None
            and hasattr(STAR, "STAR")
            and isinstance(policy, NearestPolicy)
            and not self.memory
        )


@dataclass
class RRCStrategy:
    """Restricted reconstruction: contiguous subpath destroy and append repair."""

    strategy_id: str = "lehd_rrc"
    iterations: int = 100

    def run(self, instance: Instance, policy: AppendPolicy, rng: random.Random) -> tuple[float, float, bool]:
        del rng
        if instance.problem == "tsp":
            return lehd_rrc_tsp(instance, policy, budget=self.iterations)
        if instance.problem == "cvrp":
            return lehd_rrc_cvrp(instance, policy, budget=self.iterations)
        raise ValueError(f"unsupported problem for lehd_rrc: {instance.problem}")


@dataclass
class PRCStrategy:
    """Parallel reconstruction using the original SIL PRC environment loop."""

    strategy_id: str = "sil_prc"
    iterations: int = 100

    def run(self, instance: Instance, policy: AppendPolicy, rng: random.Random) -> tuple[float, float, bool]:
        del rng
        if instance.problem == "tsp":
            return sil_prc_tsp(instance, policy, budget=self.iterations)
        if instance.problem == "cvrp":
            return sil_prc_cvrp(instance, policy, budget=self.iterations)
        raise ValueError(f"unsupported problem for sil_prc: {instance.problem}")


@dataclass
class GreedyStrategy:
    strategy_id: str = "greedy"

    def run(self, instance: Instance, policy: AppendPolicy, rng: random.Random) -> tuple[float, float, bool]:
        if instance.problem == "tsp":
            route = greedy_tsp(instance, policy, rng)
            cost = tsp_cost(instance, route)
            return cost, cost, validate_tsp(instance, route)
        if isinstance(policy, NativeCVRPNeuralPolicy):
            if policy.policy_id == "sil":
                return exact_cvrp_greedy_with_env(
                    instance,
                    policy,
                    env_path=SIL_CVRP_DIR / "VRPEnv.py",
                    import_roots=(SIL_CVRP_DIR, SIL_CVRP_DIR.parent, SIL_CVRP_DIR.parent.parent),
                    env_kwargs=sil_cvrp_env_kwargs(),
                    distance_kwargs={"test_in_vrplib": True},
                    load_kwargs={"only_test": True},
                )
            if policy.policy_id == "lehd":
                return exact_cvrp_greedy_with_env(
                    instance,
                    policy,
                    env_path=LEHD_CVRP_DIR / "VRPEnv_inCVRPlib.py",
                    import_roots=(LEHD_CVRP_DIR, LEHD_CVRP_DIR.parent, LEHD_CVRP_DIR.parent.parent),
                    env_kwargs=lehd_cvrp_env_kwargs(),
                    distance_kwargs={},
                    load_kwargs={},
                )
            if policy.policy_id in {"bq", "icam", "elg"}:
                routes = greedy_cvrp_policy(instance, policy, rng)
                cost = cvrp_cost(instance, routes)
                return cost, cost, validate_cvrp(instance, routes)
            raise ValueError(f"exact CVRP route state is not extracted for {policy.policy_id}")
        routes = greedy_cvrp(instance, policy, rng)
        cost = cvrp_cost(instance, routes)
        return cost, cost, validate_cvrp(instance, routes)


@dataclass(frozen=True)
class UnsupportedStrategy:
    strategy_id: str
    reason: str


def search_strategy(
    strategy_id: str,
    *,
    iterations: int | None = None,
    min_new_edges: int = 24,
    refine_k: int = 64,
    refine: bool = True,
    neural_knn_k: int = 32,
    neural_backup_k: int = 32,
    neural_knn_mask: bool = True,
    STAR_samples: int = 64,
    memory: bool = True,
    memory_k: int = 32,
    memory_rho: float = 0.9,
    memory_tau_min: float | None = None,
    memory_tau_max: float = 1.0,
    memory_alpha: float = 1.0,
    start_mode: str = "random",
    start_probes: int = 32,
    start_cost_weight: float = 1.0,
    start_policy_weight: float = 1.0,
    start_memory_weight: float = 1.0,
    memory_update_mode: str = "auto",
    advantage_scale: float = 100.0,
    advantage_min: float = 0.0,
    STAR_trace: bool = False,
    STAR_profile: bool = False,
    STAR_progress_callback: Callable[[dict[str, str]], None] | None = None,
) -> SearchStrategy | UnsupportedStrategy:
    if strategy_id.lower() == "star":
        strategy_id = "STAR"
    if strategy_id == "STAR":
        return STARStrategy(
            iterations=100 if iterations is None else iterations,
            min_new_edges=min_new_edges,
            refine_k=refine_k,
            refine=refine,
            neural_knn_k=neural_knn_k,
            neural_backup_k=neural_backup_k,
            neural_knn_mask=neural_knn_mask,
            STAR_samples=STAR_samples,
            memory=memory,
            memory_k=memory_k,
            memory_rho=memory_rho,
            memory_tau_min=memory_tau_min,
            memory_tau_max=memory_tau_max,
            memory_alpha=memory_alpha,
            start_mode=start_mode,
            start_probes=start_probes,
            start_cost_weight=start_cost_weight,
            start_policy_weight=start_policy_weight,
            start_memory_weight=start_memory_weight,
            memory_update_mode=memory_update_mode,
            advantage_scale=advantage_scale,
            advantage_min=advantage_min,
            STAR_trace=STAR_trace,
            STAR_profile=STAR_profile,
            STAR_progress_callback=STAR_progress_callback,
        )
    if strategy_id in {"lehd_rrc", "rrc"}:
        return RRCStrategy(iterations=1000 if iterations is None else iterations)
    if strategy_id in {"sil_prc", "prc"}:
        return PRCStrategy(iterations=1000 if iterations is None else iterations)
    if strategy_id == "greedy":
        return GreedyStrategy()
    if strategy_id == "drhg":
        return UnsupportedStrategy(
            strategy_id,
            "DRHG state coupling/endpoint-mask search logic has not been extracted into an in-process implementation yet",
        )
    return UnsupportedStrategy(strategy_id, f"unknown search strategy: {strategy_id}")


def search_strategy_ids() -> list[str]:
    return ["greedy", "lehd_rrc", "sil_prc", "rrc", "prc", "drhg", "STAR"]


def sil_cvrp_env_kwargs() -> dict[str, Any]:
    return {
        "pomo_size": 1,
        "k_nearest": 1,
        "beam_width": 16,
        "decode_method": "greedy",
        "mode": "test",
        "test_in_vrplib": True,
        "vrplib_path": str(DEFAULT_CVRP.parent),
        "data_path": str(DEFAULT_CVRP.parent),
        "load_way": "allin",
        "sub_path": False,
        "budget": 0,
        "PRC": False,
        "repair_max_sub_length": 1000,
        "random_insertion": False,
    }


def lehd_cvrp_env_kwargs() -> dict[str, Any]:
    return {
        "mode": "test",
        "test_in_vrplib": True,
        "vrplib_path": str(DEFAULT_CVRP.parent),
        "data_path": str(DEFAULT_CVRP.parent),
        "sub_path": False,
        "RRC_budget": 0,
    }


def sil_prc_tsp_env_kwargs(budget: int) -> dict[str, Any]:
    return {
        "mode": "test",
        "decode_method": "greedy",
        "test_in_tsplib": True,
        "tsplib_path": str(DEFAULT_TSP.parent),
        "data_path": str(SIL_PRC_TSP_DIR / "data/re_generate_test_TSP100_0423_n1w.txt"),
        "sub_path": False,
        "budget": budget,
        "PRC": True,
        "repair_max_sub_length": 1000,
    }


def sil_prc_cvrp_env_kwargs(budget: int) -> dict[str, Any]:
    return {
        "pomo_size": 1,
        "k_nearest": 1,
        "beam_width": 16,
        "decode_method": "greedy",
        "mode": "test",
        "test_in_vrplib": True,
        "vrplib_path": str(DEFAULT_CVRP.parent),
        "data_path": str(DEFAULT_CVRP.parent),
        "load_way": "allin",
        "sub_path": False,
        "budget": budget,
        "PRC": True,
        "repair_max_sub_length": 1000,
        "random_insertion": False,
    }


def exact_cvrp_greedy_with_env(
    instance: Instance,
    policy: NativeCVRPNeuralPolicy,
    *,
    env_path: Path,
    import_roots: Sequence[Path],
    env_kwargs: dict[str, Any],
    distance_kwargs: dict[str, Any],
    load_kwargs: dict[str, Any],
) -> tuple[float, float, bool]:
    if instance.problem != "cvrp":
        raise ValueError("exact CVRP greedy requires CVRP instance")

    instance_path = instance.source_path or DEFAULT_CVRP
    env_kwargs = dict(env_kwargs)
    if "vrplib_path" in env_kwargs:
        env_kwargs["vrplib_path"] = str(instance_path.parent)
    if "data_path" in env_kwargs:
        env_kwargs["data_path"] = str(instance_path.parent)

    env_module = import_module_from_path(
        f"_nrs_{policy.policy_id}_cvrp_env",
        env_path,
        import_roots=import_roots,
    )
    env = env_module.VRPEnv(**env_kwargs)

    with original_torch_device_context():
        with torch.no_grad():
            env.load_problems(0, 1, instance_path.stem, **load_kwargs)
            env.reset("test")
            state, _reward, _reward_student, done = env.pre_step()
            origin_problem = env.problems.clone().detach()
            current_step = 0
            while not done:
                _loss, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = native_forward_swap._call_cvrp_neural(
                    policy._spec,
                    state,
                    env.selected_node_list,
                    env.solution,
                    current_step,
                    raw_data_capacity=env.raw_data_capacity,
                    decode_method="greedy",
                )
                if current_step == 0:
                    selected_flag_teacher = torch.ones(1, dtype=torch.int, device=env.problems.device)
                    selected_flag_student = selected_flag_teacher
                current_step += 1
                state, _reward, _reward_student, done = env.step(
                    selected_teacher,
                    selected_student,
                    selected_flag_teacher,
                    selected_flag_student,
                )

            solution = torch.cat(
                (
                    env.selected_student_list.reshape(1, -1, 1),
                    env.selected_student_flag.reshape(1, -1, 1),
                ),
                dim=2,
            )
            cost = float(env._get_travel_distance_2(origin_problem, solution, **distance_kwargs).mean().item())
            routes = cvrp_routes_from_node_flags(instance, solution)
            return cost, cost, validate_cvrp(instance, routes)


def cvrp_routes_from_node_flags(instance: Instance, solution: torch.Tensor) -> list[list[int]]:
    routes: list[list[int]] = []
    current: list[int] = []
    for node_index, flag in solution.reshape(-1, 2).detach().cpu().long().tolist():
        node = int(node_index) + 1
        if int(flag) == 1 and current:
            routes.append(current)
            current = []
        if node != instance.depot:
            current.append(node)
    if current:
        routes.append(current)
    return routes


def lehd_rrc_tsp(instance: Instance, policy: AppendPolicy, budget: int) -> tuple[float, float, bool]:
    if instance.bks_cost is None:
        raise ValueError("LEHD RRC TSPLIB run requires a BKS/optimal value")

    instance_path = instance.source_path or DEFAULT_TSP
    env_module = import_module_from_path(
        "_nrs_lehd_tsp_env",
        LEHD_TSP_DIR / "TSPEnv_inTSPlib.py",
        import_roots=(LEHD_TSP_DIR, LEHD_TSP_DIR.parent, LEHD_TSP_DIR.parent.parent),
    )
    env = env_module.TSPEnv(
        mode="test",
        test_in_tsplib=True,
        tsplib_path=str(instance_path.parent),
        data_path=str(LEHD_TSP_DIR / "data/re_generate_test_TSP100_0423_n1w.txt"),
        sub_path=False,
        RRC_budget=budget,
    )

    with original_torch_device_context():
        with torch.no_grad():
            env.load_problems(0, 1, instance_path.stem, len(instance.coords), float(instance.bks_cost))
            origin_problem = env.problems
            env.reset("test")
            state, _reward, _reward_student, done = env.pre_step()
            current_step = 0
            while not done:
                if current_step == 0:
                    selected_teacher = torch.zeros(1, dtype=torch.int64, device=env.problems.device)
                    selected_student = selected_teacher
                else:
                    selected_teacher, _prob, _misc, selected_student = tsp_forward_from_policy(
                        policy, state, env.selected_node_list, env.solution, current_step
                    )
                current_step += 1
                state, _reward, _reward_student, done = env.step(selected_teacher, selected_student)

            best_select_node_list = env.selected_node_list
            initial = float(env._get_travel_distance_2(origin_problem, best_select_node_list).mean().item())

            for _ in range(budget):
                env.load_problems(0, 1, instance_path.stem, len(instance.coords), float(instance.bks_cost))

                if_inverse = torch.randint(low=0, high=100, size=[1], device=env.problems.device)[0] >= 50
                if bool(if_inverse):
                    best_select_node_list = torch.flip(best_select_node_list, dims=[1])

                partial_solution_length, first_node_index, length_of_subpath, double_solution = env.destroy_solution(
                    env.problems, best_select_node_list
                )
                before_reward = partial_solution_length

                current_step = 0
                env.reset("test")
                state, reward, reward_student, done = env.pre_step()
                while not done:
                    if current_step == 0:
                        selected_teacher = env.solution[:, -1]
                        selected_student = env.solution[:, -1]
                    elif current_step == 1:
                        selected_teacher = env.solution[:, 0]
                        selected_student = env.solution[:, 0]
                    else:
                        selected_teacher, _prob, _misc, selected_student = tsp_forward_from_policy(
                            policy, state, env.selected_node_list, env.solution, current_step, repair=True
                        )
                    current_step += 1
                    state, reward, reward_student, done = env.step(selected_teacher, selected_student)

                after_repair_sub_solution = torch.roll(env.selected_node_list, shifts=-1, dims=1)
                best_select_node_list = lehd_decide_whether_to_repair_solution(
                    after_repair_sub_solution,
                    before_reward,
                    reward_student,
                    first_node_index,
                    length_of_subpath,
                    double_solution,
                )

            final = float(env._get_travel_distance_2(origin_problem, best_select_node_list).mean().item())
            route = [int(node) + 1 for node in best_select_node_list.reshape(-1).detach().cpu().tolist()]
            return initial, final, validate_tsp(instance, route)


def lehd_rrc_cvrp(instance: Instance, policy: AppendPolicy, budget: int) -> tuple[float, float, bool]:
    if instance.bks_cost is None:
        raise ValueError("LEHD RRC CVRPLIB run requires a BKS/optimal value")

    instance_path = instance.source_path or DEFAULT_CVRP
    env_module = import_module_from_path(
        "_nrs_lehd_rrc_cvrp_env",
        LEHD_RRC_CVRP_DIR / "VRPEnv_inCVRPlib.py",
        import_roots=(LEHD_RRC_CVRP_DIR, LEHD_RRC_CVRP_DIR.parent, LEHD_RRC_CVRP_DIR.parent.parent),
    )
    env_kwargs = lehd_cvrp_env_kwargs()
    env_kwargs["vrplib_path"] = str(instance_path.parent)
    env_kwargs["data_path"] = str(instance_path.parent)
    env_kwargs["RRC_budget"] = budget
    env = env_module.VRPEnv(**env_kwargs)

    with original_torch_device_context():
        with torch.no_grad():
            torch.manual_seed(12)
            env.load_problems(0, 1, instance_path.stem)
            env.reset("test")
            state, _reward, _reward_student, done = env.pre_step()
            origin_problem = env.problems.clone().detach()

            current_step = 0
            while not done:
                _loss, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = cvrp_forward_from_policy(
                    policy,
                    state,
                    env.selected_node_list,
                    env.solution,
                    current_step,
                    raw_data_capacity=env.raw_data_capacity,
                )
                if current_step == 0:
                    selected_flag_teacher = torch.ones(1, dtype=torch.int, device=env.problems.device)
                    selected_flag_student = selected_flag_teacher
                current_step += 1
                state, _reward, _reward_student, done = env.step(
                    selected_teacher,
                    selected_student,
                    selected_flag_teacher,
                    selected_flag_student,
                )

            best_select_node_list = torch.cat(
                (
                    env.selected_student_list.reshape(1, -1, 1),
                    env.selected_student_flag.reshape(1, -1, 1),
                ),
                dim=2,
            )
            initial = float(env._get_travel_distance_2(origin_problem, best_select_node_list).mean().item())

            for _ in range(budget):
                if CUDA_AVAILABLE:
                    torch.cuda.empty_cache()

                env.load_problems(0, 1, instance_path.stem)
                best_select_node_list = env.vrp_whole_and_solution_subrandom_inverse(best_select_node_list)
                partial_solution_length, first_node_index, length_of_subpath, double_solution = env.destroy_solution(
                    env.problems,
                    best_select_node_list,
                )
                before_reward = partial_solution_length

                current_step = 0
                env.reset("test")
                state, _reward, reward_student, done = env.pre_step()
                while not done:
                    if current_step == 0:
                        selected_teacher = env.solution[:, 0, 0]
                        selected_flag_teacher = env.solution[:, 0, 1]
                        selected_student = selected_teacher
                        selected_flag_student = selected_flag_teacher
                    else:
                        _loss, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = cvrp_forward_from_policy(
                            policy,
                            state,
                            env.selected_node_list,
                            env.solution,
                            current_step,
                            raw_data_capacity=env.raw_data_capacity,
                        )
                    current_step += 1
                    state, _reward, reward_student, done = env.step(
                        selected_teacher,
                        selected_student,
                        selected_flag_teacher,
                        selected_flag_student,
                    )

                after_repair_sub_solution = torch.cat(
                    (env.selected_student_list.unsqueeze(2), env.selected_student_flag.unsqueeze(2)),
                    dim=2,
                )
                best_select_node_list = lehd_cvrp_decide_whether_to_repair_solution(
                    after_repair_sub_solution,
                    before_reward,
                    -reward_student,
                    first_node_index,
                    length_of_subpath,
                    double_solution,
                )

            final = float(env._get_travel_distance_2(origin_problem, best_select_node_list).mean().item())
            routes = cvrp_routes_from_node_flags(instance, best_select_node_list)
            return initial, final, validate_cvrp(instance, routes)


def sil_prc_tsp(instance: Instance, policy: AppendPolicy, budget: int) -> tuple[float, float, bool]:
    if instance.bks_cost is None:
        raise ValueError("SIL PRC TSPLIB run requires a BKS/optimal value")

    instance_path = instance.source_path or DEFAULT_TSP
    env_module = import_module_from_path(
        "_nrs_sil_prc_tsp_env",
        SIL_PRC_TSP_DIR / "TSPEnv.py",
        import_roots=(SIL_PRC_TSP_DIR, SIL_PRC_TSP_DIR.parent, SIL_PRC_TSP_DIR.parent.parent),
    )
    env_kwargs = sil_prc_tsp_env_kwargs(budget)
    env_kwargs["tsplib_path"] = str(instance_path.parent)
    env = env_module.TSPEnv(**env_kwargs)

    with original_torch_device_context():
        with torch.no_grad():
            torch.manual_seed(12)
            env.load_problems(0, 1, instance_path.stem, len(instance.coords), float(instance.bks_cost))
            origin_problem = env.problems.clone().detach()
            env.reset("test")
            state, _reward, _reward_student, done = env.pre_step()

            if budget > 0:
                with temporary_import_roots((SIL_PRC_TSP_DIR.parent.parent,)):
                    insertion_module = importlib.import_module("utils.insertion")
                dataset = origin_problem.clone().cpu().numpy()
                problem_size = dataset.shape[1]
                orders = [torch.randperm(problem_size)]
                pi_all = [
                    insertion_module.random_insertion(problem_instance, orders[order_id])[0]
                    for order_id in range(len(orders))
                    for problem_instance in dataset
                ]
                best_select_node_list = torch.tensor(np.array(pi_all, dtype=np.int64), device=env.problems.device)
            else:
                current_step = 0
                while not done:
                    if current_step == 0:
                        selected_teacher = torch.zeros(1, dtype=torch.int64, device=env.problems.device)
                        selected_student = selected_teacher
                    else:
                        selected_teacher, _prob, _misc, selected_student = tsp_forward_from_policy(
                            policy, state, env.selected_node_list, env.solution, current_step
                        )
                    current_step += 1
                    state, _reward, _reward_student, done = env.step(selected_teacher, selected_student)

                best_select_node_list = env.selected_node_list
            initial = float(env._get_travel_distance_2(origin_problem, best_select_node_list, test_in_tsplib=True).mean().item())

            origin_problem_size = origin_problem.shape[1]
            repair_max_sub_length = min(origin_problem_size, env_kwargs["repair_max_sub_length"])
            if repair_max_sub_length <= 4:
                length_all = torch.full((budget,), repair_max_sub_length, dtype=torch.long, device=env.problems.device)
            elif origin_problem_size <= 1000:
                length_all = torch.randint(low=4, high=repair_max_sub_length, size=[budget], device=env.problems.device)
            else:
                length_all = torch.randint(low=4, high=repair_max_sub_length + 1, size=[budget], device=env.problems.device)
            first_index_all = torch.randint(low=0, high=origin_problem_size, size=[budget], device=env.problems.device)

            for bbbb in range(budget):
                env.problems = origin_problem.clone().detach()
                best_select_node_list = env.random_inverse_solution(best_select_node_list)
                partial_solution_length, _first_node_index, _length_of_subpath, double_solution, origin_sub_solution, index4, factor = (
                    env.destroy_solution_PRC(env.problems, best_select_node_list, length_all[bbbb], first_index_all[bbbb])
                )
                before_reward = partial_solution_length
                env.batch_size = env.solution.shape[0]

                current_step = 0
                env.reset("test")
                state, _reward, reward_student, done = env.pre_step()
                while not done:
                    if current_step == 0:
                        selected_teacher = env.solution[:, -1]
                        selected_student = env.solution[:, -1]
                    elif current_step == 1:
                        selected_teacher = env.solution[:, 0]
                        selected_student = env.solution[:, 0]
                    else:
                        selected_teacher, _prob, _misc, selected_student = tsp_forward_from_policy(
                            policy, state, env.selected_node_list, env.solution, current_step, repair=True
                        )
                    current_step += 1
                    state, _reward, reward_student, done = env.step(selected_teacher, selected_student)

                after_repair_sub_solution = torch.roll(env.selected_node_list, shifts=-1, dims=1)
                best_select_node_list = env.decide_whether_to_repair_solution_PRC(
                    after_repair_sub_solution,
                    before_reward,
                    reward_student,
                    double_solution,
                    1,
                    origin_sub_solution,
                    index4,
                    factor,
                )

            final = float(env._get_travel_distance_2(origin_problem, best_select_node_list, test_in_tsplib=True).mean().item())
            route = [int(node) + 1 for node in best_select_node_list.reshape(-1).detach().cpu().tolist()]
            return initial, final, validate_tsp(instance, route)


def sil_prc_cvrp(instance: Instance, policy: AppendPolicy, budget: int) -> tuple[float, float, bool]:
    if instance.bks_cost is None:
        raise ValueError("SIL PRC CVRPLIB run requires a BKS/optimal value")

    instance_path = instance.source_path or DEFAULT_CVRP
    env_module = import_module_from_path(
        "_nrs_sil_prc_cvrp_env",
        SIL_PRC_CVRP_DIR / "VRPEnv.py",
        import_roots=(SIL_PRC_CVRP_DIR, SIL_PRC_CVRP_DIR.parent, SIL_PRC_CVRP_DIR.parent.parent),
    )
    env_kwargs = sil_prc_cvrp_env_kwargs(budget)
    env_kwargs["vrplib_path"] = str(instance_path.parent)
    env_kwargs["data_path"] = str(instance_path.parent)
    env = env_module.VRPEnv(**env_kwargs)

    with original_torch_device_context():
        with torch.no_grad():
            torch.manual_seed(12)
            env.load_problems(0, 1, instance_path.stem, only_test=True)
            env.reset("test")
            state, _reward, _reward_student, done = env.pre_step()
            origin_problem = env.problems.clone().detach()

            if budget > 0:
                with temporary_import_roots((SIL_PRC_CVRP_DIR.parent.parent,)):
                    best_select_node_list = env.random_insert(origin_problem)
            else:
                current_step = 0
                while not done:
                    _loss, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = cvrp_forward_from_policy(
                        policy,
                        state,
                        env.selected_node_list,
                        env.solution,
                        current_step,
                        raw_data_capacity=env.raw_data_capacity,
                    )
                    if current_step == 0:
                        selected_flag_teacher = torch.ones(1, dtype=torch.int, device=env.problems.device)
                        selected_flag_student = selected_flag_teacher
                    current_step += 1
                    state, _reward, _reward_student, done = env.step(
                        selected_teacher,
                        selected_student,
                        selected_flag_teacher,
                        selected_flag_student,
                    )

                best_select_node_list = torch.cat(
                    (
                        env.selected_student_list.reshape(1, -1, 1),
                        env.selected_student_flag.reshape(1, -1, 1),
                    ),
                    dim=2,
                )
            initial = float(env._get_travel_distance_2(origin_problem, best_select_node_list, test_in_vrplib=True).mean().item())

            origin_problem_size = origin_problem.shape[1]
            max_length = min(origin_problem_size, env_kwargs["repair_max_sub_length"])
            if max_length <= 4:
                length_all = torch.full((budget,), max_length, dtype=torch.long, device=env.problems.device)
            elif origin_problem_size <= 1000:
                length_all = torch.randint(low=4, high=max_length, size=[budget], device=env.problems.device)
            else:
                length_all = torch.randint(low=4, high=max_length + 1, size=[budget], device=env.problems.device)
            first_index_all = torch.randint(low=0, high=origin_problem_size, size=[budget], device=env.problems.device)

            for bbbb in range(budget):
                if CUDA_AVAILABLE:
                    torch.cuda.empty_cache()
                best_select_node_list = env.Rearrange_solution_caller(origin_problem, best_select_node_list)
                env.load_problems(0, 1, instance_path.stem, only_test=True)
                best_select_node_list = env.vrp_whole_and_solution_subrandom_inverse(best_select_node_list)

                if origin_problem_size <= 1000 and int(origin_problem_size / int(length_all[bbbb].item())) <= 1:
                    partial_solution_length, first_node_index, length_of_subpath, double_solution = env.destroy_solution(
                        env.problems,
                        best_select_node_list,
                        length_all[bbbb],
                        first_index_all[bbbb],
                    )
                    use_prc_accept = False
                    origin_sub_solution = index4 = factor = None
                else:
                    partial_solution_length, _first_node_index, _end_node_index, _length_of_subpath, double_solution, origin_sub_solution, index4, factor = (
                        env.destroy_solution_PRC(env.problems, best_select_node_list, length_all[bbbb], first_index_all[bbbb])
                    )
                    first_node_index = length_of_subpath = None
                    use_prc_accept = True

                before_reward = partial_solution_length
                env.batch_size = env.solution.shape[0]
                current_step = 0
                env.reset("test")
                state, _reward, reward_student, done = env.pre_step()
                while not done:
                    if current_step == 0:
                        selected_teacher = env.solution[:, 0, 0]
                        selected_flag_teacher = env.solution[:, 0, 1]
                        selected_student = selected_teacher
                        selected_flag_student = selected_flag_teacher
                    else:
                        _loss, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = cvrp_forward_from_policy(
                            policy,
                            state,
                            env.selected_node_list,
                            env.solution,
                            current_step,
                            raw_data_capacity=env.raw_data_capacity,
                        )
                    current_step += 1
                    state, _reward, reward_student, done = env.step(
                        selected_teacher,
                        selected_student,
                        selected_flag_teacher,
                        selected_flag_student,
                    )

                after_repair_sub_solution = torch.cat(
                    (env.selected_student_list.unsqueeze(2), env.selected_student_flag.unsqueeze(2)),
                    dim=2,
                )
                if use_prc_accept:
                    assert origin_sub_solution is not None and index4 is not None and factor is not None
                    best_select_node_list = env.decide_whether_to_repair_solution_V2(
                        after_repair_sub_solution,
                        before_reward,
                        reward_student,
                        double_solution,
                        origin_sub_solution,
                        index4,
                        1,
                        factor,
                    )
                else:
                    assert first_node_index is not None and length_of_subpath is not None
                    best_select_node_list = env.decide_whether_to_repair_solution(
                        after_repair_sub_solution,
                        before_reward,
                        reward_student,
                        first_node_index,
                        length_of_subpath,
                        double_solution,
                    )

            final = float(env._get_travel_distance_2(origin_problem, best_select_node_list, test_in_vrplib=True).mean().item())
            routes = cvrp_routes_from_node_flags(instance, best_select_node_list)
            return initial, final, validate_cvrp(instance, routes)


def tsp_forward_from_policy(
    policy: AppendPolicy,
    state: Any,
    selected_node_list: torch.Tensor,
    solution: torch.Tensor | None,
    current_step: int,
    *,
    repair: bool = False,
) -> tuple[torch.Tensor, Any, Any, torch.Tensor]:
    if isinstance(policy, NativeTSPNeuralPolicy):
        if solution is None:
            solution = torch.arange(state.data.size(1), dtype=torch.long, device=state.data.device).unsqueeze(0)
        return native_forward_swap._call_tsp_neural(
            policy._spec,
            state,
            selected_node_list,
            solution,
            current_step,
            decode_method="greedy",
            repair=repair,
        )
    if isinstance(policy, NearestPolicy):
        selected = native_forward_swap._select_next_node(state.data, selected_node_list, "nearest")
        return selected, None, None, selected
    if isinstance(policy, SoftDistPolicy):
        selected = native_forward_swap._select_next_node(state.data, selected_node_list, "softdist")
        return selected, None, None, selected
    raise ValueError(f"policy {getattr(policy, 'policy_id', policy)} cannot run in LEHD RRC TSP")


def cvrp_forward_from_policy(
    policy: AppendPolicy,
    state: Any,
    selected_node_list: torch.Tensor,
    solution: torch.Tensor,
    current_step: int,
    *,
    raw_data_capacity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(policy, NativeCVRPNeuralPolicy):
        return native_forward_swap._call_cvrp_neural(
            policy._spec,
            state,
            selected_node_list,
            solution,
            current_step,
            raw_data_capacity=raw_data_capacity,
            decode_method="greedy",
        )
    if isinstance(policy, NearestPolicy):
        return cvrp_heuristic_forward(state, selected_node_list, "nearest")
    if isinstance(policy, SoftDistPolicy):
        return cvrp_heuristic_forward(state, selected_node_list, "softdist")
    raise ValueError(f"policy {getattr(policy, 'policy_id', policy)} cannot run in CVRP reference env loop")


def cvrp_heuristic_forward(
    state: Any,
    selected_node_list: torch.Tensor,
    policy_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    problems = state.problems
    selected = native_forward_swap._select_next_node(problems[:, :, :2], selected_node_list, policy_name, start_index=1)
    demand = native_forward_swap._gather_feature(problems, selected, feature=2)
    remaining = problems[:, 0, 3].to(device=problems.device)
    selected_flag = (demand > remaining + 1e-9).to(dtype=torch.int, device=problems.device)
    loss_node = torch.zeros(selected.size(0), dtype=problems.dtype, device=problems.device)
    return loss_node, selected, selected, selected_flag, selected_flag.clone()


def lehd_cvrp_decide_whether_to_repair_solution(
    after_repair_sub_solution: torch.Tensor,
    before_reward: torch.Tensor,
    after_reward: torch.Tensor,
    first_node_index: torch.Tensor,
    length_of_subpath: torch.Tensor,
    double_solution: torch.Tensor,
) -> torch.Tensor:
    the_whole_problem_size = int(double_solution.shape[1] / 2)
    batch_size = len(double_solution)
    temp = torch.arange(double_solution.shape[1], device=double_solution.device)
    x3 = temp >= first_node_index[:, None].long()
    x4 = temp < (first_node_index[:, None] + length_of_subpath).long()
    x5 = x3 * x4
    origin_sub_solution = double_solution[x5.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, length_of_subpath, 2)
    sorted_origin, _ = torch.sort(origin_sub_solution[:, :, 0], dim=1, descending=False)
    index = torch.arange(batch_size, device=double_solution.device)[:, None].repeat(1, sorted_origin.shape[1])
    after_repair_sub_solution[:, :, 0] = sorted_origin[index, after_repair_sub_solution[:, :, 0] - 1]
    if_repair = before_reward > after_reward
    repair_double_solution = double_solution[if_repair]
    repair_double_solution[x5[if_repair].unsqueeze(2).repeat(1, 1, 2)] = after_repair_sub_solution[if_repair].ravel()
    double_solution[if_repair] = repair_double_solution
    x6 = temp >= (first_node_index[:, None] + length_of_subpath - the_whole_problem_size).long()
    x7 = temp < (first_node_index[:, None] + length_of_subpath).long()
    x8 = x6 * x7
    return double_solution[x8.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, the_whole_problem_size, -1)


def lehd_decide_whether_to_repair_solution(
    after_repair_sub_solution: torch.Tensor,
    before_reward: torch.Tensor,
    after_reward: torch.Tensor,
    first_node_index: torch.Tensor,
    length_of_subpath: torch.Tensor,
    double_solution: torch.Tensor,
) -> torch.Tensor:
    the_whole_problem_size = int(double_solution.shape[1] / 2)
    other_part_1 = double_solution[:, :first_node_index]
    other_part_2 = double_solution[:, first_node_index + length_of_subpath :]
    origin_sub_solution = double_solution[:, first_node_index : first_node_index + length_of_subpath]
    sorted_origin, _ = torch.sort(origin_sub_solution, dim=1, descending=False)
    index = torch.arange(sorted_origin.shape[0], device=sorted_origin.device)[:, None].repeat(1, sorted_origin.shape[1])
    repaired_global_nodes = sorted_origin[index, after_repair_sub_solution]
    if_repair = before_reward > after_reward
    double_solution[if_repair] = torch.cat(
        (other_part_1[if_repair], repaired_global_nodes[if_repair], other_part_2[if_repair]),
        dim=1,
    )
    return double_solution[:, first_node_index : first_node_index + the_whole_problem_size]


def import_module_from_path(name: str, path: Path, import_roots: Sequence[Path]) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    with temporary_import_roots(import_roots):
        spec.loader.exec_module(module)
    return module


@contextmanager
def temporary_import_roots(roots: Sequence[Path]):
    previous_path = list(sys.path)
    try:
        for root in reversed([str(path) for path in roots if path.exists()]):
            if root not in sys.path:
                sys.path.insert(0, root)
        yield
    finally:
        sys.path[:] = previous_path


@contextmanager
def original_torch_device_context():
    previous_type = torch.tensor([]).type()
    if CUDA_AVAILABLE:
        torch.cuda.set_device(0)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    try:
        yield
    finally:
        torch.set_default_tensor_type(previous_type)


def greedy_tsp(instance: Instance, policy: AppendPolicy, rng: random.Random) -> list[int]:
    route = [instance.depot]
    remaining = set(instance.coords) - {instance.depot}
    while remaining:
        node = select_tsp_decision(instance, policy, route[-1], sorted(remaining), rng, route)
        route.append(node)
        remaining.remove(node)
    return route


def greedy_tsp_from_start(instance: Instance, policy: AppendPolicy, rng: random.Random, start: int) -> list[int]:
    if start not in instance.coords:
        raise ValueError(f"start node {start} not found in TSP instance")
    route = [start]
    remaining = set(instance.coords) - {start}
    while remaining:
        node = select_tsp_decision(instance, policy, route[-1], sorted(remaining), rng, route)
        route.append(node)
        remaining.remove(node)
    return route


def greedy_tsp_multi_start(instance: Instance, policy: AppendPolicy, rng: random.Random, starts: int = 8) -> list[int]:
    node_ids = sorted(instance.coords)
    if not node_ids:
        return []
    best_route: list[int] | None = None
    best_cost = math.inf
    for start in node_ids[: max(1, min(starts, len(node_ids)))]:
        route = greedy_tsp_from_start(instance, policy, rng, start)
        cost = tsp_cost(instance, route)
        if cost < best_cost:
            best_route = route
            best_cost = cost
    if best_route is None:
        raise ValueError("failed to construct multi-start TSP route")
    return best_route


def tsp_initial_multi_start_cpp(instance: Instance, starts: int = 8, k: int = 32) -> list[int]:
    if instance.problem != "tsp":
        raise ValueError("TSP multi-start initializer requested for non-TSP instance")
    return list(STAR_context(instance).greedy_tsp_multi_start(starts, k))


def greedy_cvrp(instance: Instance, policy: AppendPolicy, rng: random.Random) -> list[list[int]]:
    if instance.capacity is None:
        raise ValueError("CVRP capacity missing")
    if isinstance(policy, NativeCVRPNeuralPolicy):
        raise ValueError(
            f"{policy.policy_id} CVRP neural policy is exact-env only; generic CVRP greedy is not a faithful route-state loop"
        )
    remaining = set(instance.coords) - {instance.depot}
    routes: list[list[int]] = []
    selected_history: list[int] = []
    while remaining:
        route: list[int] = []
        load = 0
        current = instance.depot
        while True:
            feasible = sorted(node for node in remaining if load + instance.demands.get(node, 0) <= instance.capacity)
            if not feasible:
                break
            prefix = selected_history + route
            node = policy.select_next(instance, current, feasible, rng, prefix)
            route.append(node)
            remaining.remove(node)
            load += instance.demands.get(node, 0)
            current = node
        if not route:
            raise ValueError("no feasible CVRP customer")
        selected_history.extend(route)
        routes.append(route)
    return routes


def select_tsp_decision(
    instance: Instance,
    policy: AppendPolicy,
    current: int,
    candidates: Sequence[int],
    rng: random.Random,
    prefix: Sequence[int],
    edge_memory: SparseEdgeMemory | None = None,
    neural_knn_k: int = 32,
    neural_backup_k: int = 32,
    neural_knn_mask: bool = True,
) -> int:
    if isinstance(policy, NativeTSPNeuralPolicy):
        picked = select_tsp_neural_knn_action(
            instance,
            policy,
            current,
            candidates,
            rng,
            prefix,
            edge_memory=edge_memory,
            neural_knn_k=neural_knn_k,
            neural_backup_k=neural_backup_k,
            neural_knn_mask=neural_knn_mask,
        )
    elif edge_memory is not None:
        picked = select_tsp_decision_with_memory(instance, policy, current, candidates, rng, prefix, edge_memory)
    else:
        picked = policy.select_next(instance, current, candidates, rng, prefix, repair=True)
    if picked not in candidates:
        raise ValueError(
            f"{getattr(policy, 'policy_id', policy)} picked invalid TSP node {picked}; "
            f"{len(candidates)} legal candidates remain"
        )
    return picked


def select_tsp_neural_knn_action(
    instance: Instance,
    policy: NativeTSPNeuralPolicy,
    current: int,
    candidates: Sequence[int],
    rng: random.Random,
    prefix: Sequence[int],
    *,
    edge_memory: SparseEdgeMemory | None = None,
    neural_knn_k: int = 32,
    neural_backup_k: int = 32,
    neural_knn_mask: bool = True,
) -> int:
    legal = set(candidates)
    neighbor_order = STAR_tsp_neighbor_order(instance, neural_knn_k + neural_backup_k)
    order = STAR_tsp_neighbor_row(instance, neighbor_order, current)
    primary_candidates = [node for node in order[:neural_knn_k] if node in legal]
    if not primary_candidates:
        for node in order[neural_knn_k : neural_knn_k + neural_backup_k]:
            if node in legal:
                return node
        return min(candidates, key=lambda node: (raw_euclidean_distance(instance, current, node), node))

    allowed_candidates = primary_candidates if neural_knn_mask and policy.policy_id == "lehd" else None
    probabilities = policy.action_probability_vector(instance, prefix, repair=True, allowed_candidates=allowed_candidates)
    nodes, coords, _demands = STAR_srr_instance_payload(instance)
    memory_weights: list[float] = []
    if edge_memory is not None:
        memory_weights = [edge_memory.weight(current, node) for node in nodes]
    node_to_index = {node: index for index, node in enumerate(nodes)}
    candidate_weights = tsp_candidate_heuristic_weights(
        instance,
        current,
        primary_candidates,
        [probabilities[node_to_index[node]] for node in primary_candidates],
        edge_memory=edge_memory,
    )
    if STAR is not None and hasattr(STAR, "select_tsp_candidate_weight_actions_batch"):
        picked = STAR.select_tsp_candidate_weight_actions_batch(
            [primary_candidates],
            [candidate_weights],
            [rng.random()],
        )
        return int(picked[0])
    if STAR is not None and hasattr(STAR, "select_tsp_candidate_actions_batch"):
        picked = STAR.select_tsp_candidate_actions_batch(
            nodes,
            [current],
            [primary_candidates],
            [probabilities],
            [memory_weights] if memory_weights else [],
            [rng.random()],
        )
        return int(picked[0])
    if STAR is not None and hasattr(STAR, "select_tsp_knn_action"):
        return int(
            STAR.select_tsp_knn_action(
                nodes,
                coords,
                instance.edge_weight_type,
                current,
                primary_candidates,
                probabilities,
                memory_weights,
                neural_knn_k,
                0,
                rng.random(),
            )
        )
    return select_tsp_knn_action_python(
        instance,
        current,
        primary_candidates,
        probabilities,
        nodes,
        memory_weights,
        neural_knn_k,
        0,
        rng,
    )


def select_tsp_knn_action_python(
    instance: Instance,
    current: int,
    candidates: Sequence[int],
    probabilities: Sequence[float],
    nodes: Sequence[int],
    memory_weights: Sequence[float],
    neural_knn_k: int,
    neural_backup_k: int,
    rng: random.Random,
) -> int:
    node_to_index = {node: index for index, node in enumerate(nodes)}
    legal = set(candidates)
    order = nearest_nodes_excluding(instance, current, current, neural_knn_k + neural_backup_k)
    primary = []
    for node in order[:neural_knn_k]:
        if node not in legal:
            continue
        index = node_to_index[node]
        weight = float(probabilities[index])
        if memory_weights:
            weight *= float(memory_weights[index])
        if weight > 0.0 and math.isfinite(weight):
            primary.append((node, weight))
    if primary:
        return sample_weighted(primary, sum(weight for _node, weight in primary), rng)
    for node in order[neural_knn_k : neural_knn_k + neural_backup_k]:
        if node in legal:
            return node
    return min(candidates, key=lambda node: (raw_euclidean_distance(instance, current, node), node))


def select_tsp_decision_with_memory(
    instance: Instance,
    policy: AppendPolicy,
    current: int,
    candidates: Sequence[int],
    rng: random.Random,
    prefix: Sequence[int],
    edge_memory: SparseEdgeMemory,
) -> int:
    candidate_set = set(candidates)
    if isinstance(policy, NativeTSPNeuralPolicy):
        probs = policy.action_probabilities(instance, prefix, repair=True)
        weighted = [
            (node, probs.get(node, 0.0) * edge_memory.weight(current, node))
            for node in candidates
        ]
        total = sum(weight for _node, weight in weighted)
        if total <= 0.0:
            raise ValueError(f"{policy.policy_id} produced no positive probability for legal TSP candidates")
        return sample_weighted(weighted, total, rng)
    if isinstance(policy, SoftDistPolicy):
        pool: list[tuple[int, float]] = []
        for node, dist_value in policy._nearest_distances(instance, current):
            if node in candidate_set:
                score = math.exp(-dist_value / max(policy.temperature, 1e-12)) * edge_memory.weight(current, node)
                pool.append((node, score))
                if len(pool) >= policy.knn:
                    break
        if not pool:
            raise ValueError("softdist received no legal candidates")
        return sample_weighted(pool, sum(weight for _node, weight in pool), rng)
    if isinstance(policy, NearestPolicy):
        return max(
            candidates,
            key=lambda node: (
                edge_memory.weight(current, node),
                -distance(instance, current, node),
                -node,
            ),
        )
    return policy.select_next(instance, current, candidates, rng, prefix, repair=True)


def sample_weighted(items: Sequence[tuple[int, float]], total: float, rng: random.Random) -> int:
    threshold = rng.random() * total
    running = 0.0
    for node, weight in items:
        running += weight
        if running >= threshold:
            return node
    return items[-1][0]


def greedy_cvrp_policy(instance: Instance, policy: AppendPolicy, rng: random.Random) -> list[list[int]]:
    return greedy_cvrp_policy_from_start(instance, policy, rng, start_node=None)


def greedy_cvrp_policy_multi_start(instance: Instance, policy: AppendPolicy, rng: random.Random, starts: int = 8) -> list[list[int]]:
    if instance.capacity is None:
        raise ValueError("CVRP capacity missing")
    customers = sorted(set(instance.coords) - {instance.depot})
    if not customers:
        return []
    start_nodes: list[int | None] = [None]
    depot_order = sorted(customers, key=lambda node: (distance(instance, instance.depot, node), node))
    start_nodes.extend(depot_order[: max(0, starts - 1)])
    while len(start_nodes) < starts and len(start_nodes) < len(customers) + 1:
        node = rng.choice(customers)
        if node not in start_nodes:
            start_nodes.append(node)

    best_routes: list[list[int]] | None = None
    best_cost = float("inf")
    for node in start_nodes[: max(1, starts)]:
        routes = greedy_cvrp_policy_from_start(instance, policy, rng, start_node=node)
        cost = cvrp_cost(instance, routes)
        if cost < best_cost:
            best_routes = routes
            best_cost = cost
    if best_routes is None:
        raise RuntimeError("CVRP multi-start greedy failed to construct a route")
    return best_routes


def greedy_cvrp_policy_from_start(
    instance: Instance,
    policy: AppendPolicy,
    rng: random.Random,
    start_node: int | None,
) -> list[list[int]]:
    if instance.capacity is None:
        raise ValueError("CVRP capacity missing")
    if isinstance(policy, NativeCVRPNeuralPolicy) and policy.policy_id not in {"bq", "lehd", "sil", "icam", "elg"}:
        raise ValueError(f"STAR CVRP currently supports nearest/softdist/bq/lehd/sil/icam/elg, not {policy.policy_id}")

    remaining = set(instance.coords) - {instance.depot}
    routes: list[list[int]] = []
    selected_history: list[int] = []
    current_route: list[int] = []
    current = instance.depot
    remaining_capacity = instance.capacity
    if start_node is not None:
        if start_node not in remaining:
            raise ValueError(f"invalid CVRP start node {start_node}")
        demand = instance.demands.get(start_node, 0)
        if demand > remaining_capacity:
            raise ValueError(f"customer {start_node} demand exceeds vehicle capacity")
        current_route.append(start_node)
        selected_history.append(start_node)
        remaining.remove(start_node)
        remaining_capacity -= demand
        current = start_node

    while remaining:
        decision = select_cvrp_decision(instance, policy, current, sorted(remaining), rng, selected_history, remaining_capacity)
        demand = instance.demands.get(decision.node, 0)
        if decision.route_break or demand > remaining_capacity:
            if current_route:
                routes.append(current_route)
                current_route = []
            current = instance.depot
            remaining_capacity = instance.capacity
            if demand > remaining_capacity:
                raise ValueError(f"customer {decision.node} demand exceeds vehicle capacity")

        current_route.append(decision.node)
        selected_history.append(decision.node)
        remaining.remove(decision.node)
        remaining_capacity -= demand
        current = decision.node

    if current_route:
        routes.append(current_route)
    return routes


def symmetric_edges(edges: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for u, v in edges:
        result.add((u, v))
        result.add((v, u))
    return result


def tsp_relocation_edge_delta(
    route: Sequence[int],
    positions: dict[int, int],
    current: int,
    picked: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    current_pos = positions[current]
    picked_pos = positions[picked]
    successor = route[(current_pos + 1) % len(route)]
    picked_pred = route[picked_pos - 1]
    picked_succ = route[(picked_pos + 1) % len(route)]
    added = symmetric_edges(
        [
            (current, picked),
            (picked, successor),
            (picked_pred, picked_succ),
        ]
    )
    removed = symmetric_edges(
        [
            (current, successor),
            (picked_pred, picked),
            (picked, picked_succ),
        ]
    )
    return added, removed


def choose_tsp_perturb_start(
    instance: Instance,
    route: Sequence[int],
    positions: dict[int, int],
    policy: AppendPolicy,
    rng: random.Random,
    *,
    edge_memory: SparseEdgeMemory | None,
    neural_knn_k: int,
    neural_knn_mask: bool,
    start_mode: str,
    start_probes: int,
    cost_weight: float,
    policy_weight: float,
    memory_weight: float,
) -> tuple[int, TspStartInfo]:
    if start_mode == "random":
        node = rng.choice(list(route))
        successor = route[(positions[node] + 1) % len(route)]
        return node, TspStartInfo(mode=start_mode, node=node, successor=successor)
    scored = build_tsp_perturb_start_scores(
        instance,
        route,
        positions,
        policy,
        rng,
        edge_memory=edge_memory,
        neural_knn_k=neural_knn_k,
        neural_knn_mask=neural_knn_mask,
        start_mode=start_mode,
        start_probes=start_probes,
        cost_weight=cost_weight,
        policy_weight=policy_weight,
        memory_weight=memory_weight,
    )
    return sample_tsp_perturb_start(scored, rng)


def build_tsp_perturb_start_scores(
    instance: Instance,
    route: Sequence[int],
    positions: dict[int, int],
    policy: AppendPolicy,
    rng: random.Random,
    *,
    edge_memory: SparseEdgeMemory | None,
    neural_knn_k: int,
    neural_knn_mask: bool,
    start_mode: str,
    start_probes: int,
    cost_weight: float,
    policy_weight: float,
    memory_weight: float,
) -> list[tuple[int, float, TspStartInfo]]:
    if start_mode == "random":
        return [
            (
                node,
                0.0,
                TspStartInfo(mode=start_mode, node=node, successor=route[(positions[node] + 1) % len(route)]),
            )
            for node in route
        ]

    if start_mode == "policy-disagreement" and policy_weight > 0.0 and not isinstance(policy, NativeTSPNeuralPolicy):
        raise ValueError(f"STAR-start-mode {start_mode} requires a TSP neural policy for policy-disagreement scoring")
    if start_mode == "hybrid" and memory_weight > 0.0 and edge_memory is None:
        raise ValueError("STAR-start-mode hybrid with memory weight requires STAR memory")

    probe_count = min(len(route), max(1, start_probes))
    probes = rng.sample(list(route), probe_count) if probe_count < len(route) else list(route)
    successors = {node: route[(positions[node] + 1) % len(route)] for node in probes}
    edge_costs = {node: raw_euclidean_distance(instance, node, successors[node]) for node in probes}
    max_edge_cost = max(edge_costs.values()) if edge_costs else 1.0

    policy_scores: dict[int, tuple[float, float | None, float | None]] = {}
    if start_mode in {"policy-disagreement", "hybrid"} and policy_weight > 0.0:
        neighbor_order = STAR_tsp_neighbor_order(instance, neural_knn_k)
        jobs: list[tuple[int, list[int]]] = []
        for node in probes:
            remaining = [candidate for candidate in route if candidate != node]
            primary = tsp_primary_candidates(instance, node, remaining, neural_knn_k, neighbor_order)
            if primary:
                jobs.append((node, primary))
        if jobs and isinstance(policy, NativeTSPNeuralPolicy):
            prefixes_batch = [[node] for node, _primary in jobs]
            candidates_batch = [primary for _node, primary in jobs]
            allowed_batch = candidates_batch if neural_knn_mask and policy.policy_id == "lehd" else None
            probs_batch = policy.action_candidate_probabilities_batch(
                instance,
                prefixes_batch,
                candidates_batch,
                repair=True,
                allowed_candidates=allowed_batch,
            )
            for (node, primary), probs in zip(jobs, probs_batch):
                successor = successors[node]
                successor_prob = 0.0
                best_alt = 0.0
                for candidate, prob in zip(primary, probs):
                    prob_value = float(prob)
                    if candidate == successor:
                        successor_prob = prob_value
                    else:
                        best_alt = max(best_alt, prob_value)
                policy_scores[node] = (max(0.0, best_alt - successor_prob), successor_prob, best_alt)
        elif jobs and isinstance(policy, NearestPolicy):
            for node, primary in jobs:
                successor = successors[node]
                raw_weights = [1.0 / max(raw_euclidean_distance(instance, node, candidate), 1e-12) for candidate in primary]
                total = sum(raw_weights)
                successor_prob = 0.0
                best_alt = 0.0
                for candidate, raw_weight in zip(primary, raw_weights):
                    prob_value = raw_weight / total if total > 0.0 and math.isfinite(total) else 1.0 / len(primary)
                    if candidate == successor:
                        successor_prob = prob_value
                    else:
                        best_alt = max(best_alt, prob_value)
                policy_scores[node] = (max(0.0, best_alt - successor_prob), successor_prob, best_alt)
        elif jobs:
            raise ValueError(f"STAR-start-mode {start_mode} policy scoring supports neural TSP policies and nearest heuristic, not {getattr(policy, 'policy_id', policy)}")

    scored: list[tuple[int, float, TspStartInfo]] = []
    for node in probes:
        successor = successors[node]
        cost_score = edge_costs[node] / max(max_edge_cost, 1e-12)
        policy_score, successor_prob, best_alt_prob = policy_scores.get(node, (0.0, None, None))
        if start_mode == "policy-disagreement":
            score = policy_score
        elif start_mode == "cost":
            score = cost_score
        else:
            memory_score = 0.0
            if edge_memory is not None:
                memory_score = max(0.0, (edge_memory.tau_max - edge_memory.weight(node, successor)) / max(edge_memory.tau_max, 1e-12))
            score = cost_weight * cost_score + policy_weight * policy_score + memory_weight * memory_score
            info = TspStartInfo(
                mode=start_mode,
                node=node,
                successor=successor,
                score=score,
                cost_score=cost_score,
                policy_score=policy_score,
                memory_score=memory_score,
                successor_prob=successor_prob,
                best_alt_prob=best_alt_prob,
            )
            scored.append((node, score, info))
            continue
        info = TspStartInfo(
            mode=start_mode,
            node=node,
            successor=successor,
            score=score,
            cost_score=cost_score,
            policy_score=policy_score,
            successor_prob=successor_prob,
            best_alt_prob=best_alt_prob,
        )
        scored.append((node, score, info))

    return scored


def sample_tsp_perturb_start(scored: Sequence[tuple[int, float, TspStartInfo]], rng: random.Random) -> tuple[int, TspStartInfo]:
    if not scored:
        raise ValueError("cannot sample STAR start from an empty score list")
    max_score = max(score for _node, score, _info in scored)
    weights = [(node, math.exp(score - max_score)) for node, score, _info in scored]
    picked = sample_weighted(weights, sum(weight for _node, weight in weights), rng)
    for node, _score, info in scored:
        if node == picked:
            return node, info
    node = scored[-1][0]
    return node, scored[-1][2]


def perturb_tsp(
    instance: Instance,
    route: Sequence[int],
    policy: AppendPolicy,
    rng: random.Random,
    min_new_edges: int,
    *,
    edge_memory: SparseEdgeMemory | None = None,
    neural_knn_k: int = 32,
    neural_backup_k: int = 32,
    neural_knn_mask: bool = True,
    start_mode: str = "random",
    start_probes: int = 32,
    start_cost_weight: float = 1.0,
    start_policy_weight: float = 1.0,
    start_memory_weight: float = 1.0,
) -> TspPerturbResult:
    if not route:
        return TspPerturbResult([], set())
    candidate = list(route)
    if len(candidate) <= 2 or min_new_edges == 0:
        return TspPerturbResult(candidate, set())

    positions = tsp_positions(candidate)
    node_to_visit_index = {node: index for index, node in enumerate(candidate)}
    visited_flags = [False] * len(candidate)
    current, start_info = choose_tsp_perturb_start(
        instance,
        candidate,
        positions,
        policy,
        rng,
        edge_memory=edge_memory,
        neural_knn_k=neural_knn_k,
        neural_knn_mask=neural_knn_mask,
        start_mode=start_mode,
        start_probes=start_probes,
        cost_weight=start_cost_weight,
        policy_weight=start_policy_weight,
        memory_weight=start_memory_weight,
    )
    prefix = [current]
    visited_flags[node_to_visit_index[current]] = True
    visited_count = 1
    changed: set[int] = {current}
    introduced_edges: set[tuple[int, int]] = set()
    removed_edges: set[tuple[int, int]] = set()
    new_edges = 0

    while new_edges < min_new_edges and visited_count < len(candidate):
        remaining = [node for node in candidate if not visited_flags[node_to_visit_index[node]]]
        current_pos = positions[current]
        successor = candidate[(current_pos + 1) % len(candidate)]
        picked = select_tsp_decision(
            instance,
            policy,
            current,
            remaining,
            rng,
            prefix,
            edge_memory=edge_memory,
            neural_knn_k=neural_knn_k,
            neural_backup_k=neural_backup_k,
            neural_knn_mask=neural_knn_mask,
        )
        picked_visit_index = node_to_visit_index.get(picked)
        if picked_visit_index is None or visited_flags[picked_visit_index]:
            raise ValueError(f"{getattr(policy, 'policy_id', policy)} picked invalid TSP node {picked}")

        is_new_edge = picked != successor
        if is_new_edge:
            added, removed = tsp_relocation_edge_delta(candidate, positions, current, picked)
            introduced_edges.update(added)
            removed_edges.update(removed)
            picked_pos = positions[picked]
            picked_pred = candidate[picked_pos - 1]
            relocate_after_inplace(candidate, positions, current, picked)
            changed.update({current, picked, picked_pred})
            new_edges += 1

        visited_flags[picked_visit_index] = True
        visited_count += 1
        prefix.append(picked)
        current = picked
    return TspPerturbResult(candidate, changed, introduced_edges, removed_edges, start_info)


@dataclass
class BatchedTspPerturbState:
    candidate: list[int]
    positions: dict[int, int]
    node_to_visit_index: dict[int, int]
    visited_flags: list[bool]
    current: int
    prefix: list[int]
    changed: set[int]
    introduced_edges: set[tuple[int, int]] = field(default_factory=set)
    removed_edges: set[tuple[int, int]] = field(default_factory=set)
    start_info: TspStartInfo | None = None
    new_edges: int = 0
    visited_count: int = 1
    active: bool = True


def make_batched_tsp_perturb_state(
    instance: Instance,
    route: Sequence[int],
    policy: AppendPolicy,
    rng: random.Random,
    *,
    edge_memory: SparseEdgeMemory | None,
    neural_knn_k: int,
    neural_knn_mask: bool,
    start_mode: str,
    start_probes: int,
    start_cost_weight: float,
    start_policy_weight: float,
    start_memory_weight: float,
) -> BatchedTspPerturbState:
    candidate = list(route)
    positions = tsp_positions(candidate)
    current, start_info = choose_tsp_perturb_start(
        instance,
        candidate,
        positions,
        policy,
        rng,
        edge_memory=edge_memory,
        neural_knn_k=neural_knn_k,
        neural_knn_mask=neural_knn_mask,
        start_mode=start_mode,
        start_probes=start_probes,
        cost_weight=start_cost_weight,
        policy_weight=start_policy_weight,
        memory_weight=start_memory_weight,
    )
    return make_batched_tsp_perturb_state_from_start(candidate, positions, current, start_info)


def make_batched_tsp_perturb_state_from_start(
    candidate: list[int],
    positions: dict[int, int],
    current: int,
    start_info: TspStartInfo,
) -> BatchedTspPerturbState:
    node_to_visit_index = {node: index for index, node in enumerate(candidate)}
    visited_flags = [False] * len(candidate)
    visited_flags[node_to_visit_index[current]] = True
    return BatchedTspPerturbState(
        candidate=candidate,
        positions=positions,
        node_to_visit_index=node_to_visit_index,
        visited_flags=visited_flags,
        current=current,
        prefix=[current],
        changed={current},
        start_info=start_info,
    )


def batched_tsp_remaining(state: BatchedTspPerturbState) -> list[int]:
    return [
        node
        for node in state.candidate
        if not state.visited_flags[state.node_to_visit_index[node]]
    ]


def tsp_primary_candidates(
    instance: Instance,
    current: int,
    candidates: Sequence[int],
    neural_knn_k: int,
    neighbor_order: Sequence[Sequence[int]] | None = None,
) -> list[int]:
    legal = set(candidates)
    order = (
        STAR_tsp_neighbor_row(instance, neighbor_order, current)
        if neighbor_order is not None
        else nearest_nodes_excluding(instance, current, current, neural_knn_k)
    )
    return [node for node in order[:neural_knn_k] if node in legal]


def tsp_backup_or_global_candidate(
    instance: Instance,
    current: int,
    candidates: Sequence[int],
    neural_knn_k: int,
    neural_backup_k: int,
    neighbor_order: Sequence[Sequence[int]] | None = None,
) -> int:
    legal = set(candidates)
    order = (
        STAR_tsp_neighbor_row(instance, neighbor_order, current)
        if neighbor_order is not None
        else nearest_nodes_excluding(instance, current, current, neural_knn_k + neural_backup_k)
    )
    for node in order[neural_knn_k : neural_knn_k + neural_backup_k]:
        if node in legal:
            return node
    return min(candidates, key=lambda node: (raw_euclidean_distance(instance, current, node), node))


def tsp_candidate_heuristic_weights(
    instance: Instance,
    current: int,
    candidates: Sequence[int],
    base_weights: Sequence[float],
    *,
    edge_memory: SparseEdgeMemory | None = None,
) -> list[float]:
    if len(candidates) != len(base_weights):
        raise ValueError("candidate weights must align with candidates")
    weights: list[float] = []
    for node, base_weight in zip(candidates, base_weights):
        d = raw_euclidean_distance(instance, current, node)
        weight = float(base_weight) / max(d, 1e-12)
        if edge_memory is not None:
            weight *= edge_memory.weight(current, node)
        weights.append(weight if math.isfinite(weight) and weight > 0.0 else 0.0)
    return weights


def apply_tsp_perturb_pick(
    state: BatchedTspPerturbState,
    picked: int,
    min_new_edges: int,
) -> None:
    picked_visit_index = state.node_to_visit_index.get(picked)
    if picked_visit_index is None or state.visited_flags[picked_visit_index]:
        raise ValueError(f"batched STAR picked invalid TSP node {picked}")
    current_pos = state.positions[state.current]
    successor = state.candidate[(current_pos + 1) % len(state.candidate)]
    if picked != successor:
        added, removed = tsp_relocation_edge_delta(state.candidate, state.positions, state.current, picked)
        state.introduced_edges.update(added)
        state.removed_edges.update(removed)
        picked_pos = state.positions[picked]
        picked_pred = state.candidate[picked_pos - 1]
        relocate_after_inplace(state.candidate, state.positions, state.current, picked)
        state.changed.update({state.current, picked, picked_pred})
        state.new_edges += 1
    state.visited_flags[picked_visit_index] = True
    state.visited_count += 1
    state.prefix.append(picked)
    state.current = picked
    state.active = state.new_edges < min_new_edges and state.visited_count < len(state.candidate)


def perturb_tsp_batched_neural(
    instance: Instance,
    route: Sequence[int],
    policy: NativeTSPNeuralPolicy,
    rng: random.Random,
    min_new_edges: int,
    samples: int,
    *,
    edge_memory: SparseEdgeMemory | None = None,
    neural_knn_k: int = 32,
    neural_backup_k: int = 32,
    neural_knn_mask: bool = True,
    start_mode: str = "random",
    start_probes: int = 32,
    start_cost_weight: float = 1.0,
    start_policy_weight: float = 1.0,
    start_memory_weight: float = 1.0,
    profile: dict[str, Any] | None = None,
    neighbor_order: Sequence[Sequence[int]] | None = None,
) -> list[TspPerturbResult]:
    if policy.policy_id not in {"lehd", "sil"}:
        raise ValueError(f"batched STAR TSP neural perturbation is implemented for LEHD/SIL, not {policy.policy_id}")
    if samples <= 0:
        raise ValueError("STAR-samples must be positive")
    if not route:
        return [TspPerturbResult([], set()) for _ in range(samples)]
    if len(route) <= 2 or min_new_edges == 0:
        return [TspPerturbResult(list(route), set()) for _ in range(samples)]

    if start_mode != "random" or STAR is None or not hasattr(STAR, "TspPerturbBatch"):
        return perturb_tsp_batched_neural_python(
            instance,
            route,
            policy,
            rng,
            min_new_edges,
            samples,
            edge_memory=edge_memory,
            neural_knn_k=neural_knn_k,
            neural_backup_k=neural_backup_k,
            neural_knn_mask=neural_knn_mask,
            start_mode=start_mode,
            start_probes=start_probes,
            start_cost_weight=start_cost_weight,
            start_policy_weight=start_policy_weight,
            start_memory_weight=start_memory_weight,
            profile=profile,
            neighbor_order=neighbor_order,
        )

    init_t0 = time.perf_counter()
    context = STAR_context(instance)
    if neighbor_order is not None:
        context.neighbor_order(len(neighbor_order[0]) if neighbor_order else 0)
    engine = context.tsp_perturb_batch(
        list(route),
        int(samples),
        int(min_new_edges),
        int(neural_knn_k),
        int(neural_backup_k),
        int(rng.randrange(0, 2**63 - 1)),
    )
    init_time = time.perf_counter() - init_t0
    memory_values: list[tuple[int, int, float]] = []
    if edge_memory is not None:
        memory_values = [(u, v, edge_memory.weight(u, v)) for (u, v) in edge_memory.values]

    request_time = 0.0
    neural_time = 0.0
    step_time = 0.0
    decode_batches = 0
    decode_rows = 0
    while True:
        request_t0 = time.perf_counter()
        state_indices, prefixes_batch, candidates_batch = engine.requests()
        request_time += time.perf_counter() - request_t0
        if not state_indices:
            break
        allowed_batch = [
            list(candidates) if candidates else None
            for candidates in candidates_batch
        ] if neural_knn_mask and policy.policy_id == "lehd" else None
        neural_t0 = time.perf_counter()
        probabilities_batch = policy.action_candidate_probabilities_batch(
            instance,
            prefixes_batch,
            candidates_batch,
            repair=True,
            allowed_candidates=allowed_batch,
        )
        neural_time += time.perf_counter() - neural_t0
        decode_batches += 1
        decode_rows += len(state_indices)
        step_t0 = time.perf_counter()
        engine.step_candidates(
            state_indices,
            candidates_batch,
            probabilities_batch,
            memory_values,
            [rng.random() for _state_index in state_indices],
        )
        step_time += time.perf_counter() - step_t0

    result_t0 = time.perf_counter()
    routes_batch, changeds_batch, introduced_batch, removed_batch = engine.results()
    result_time = time.perf_counter() - result_t0
    if profile is not None:
        profile.update(
            {
                "perturb_init_seconds": f"{init_time:.9f}",
                "perturb_request_seconds": f"{request_time:.9f}",
                "neural_forward_seconds": f"{neural_time:.9f}",
                "cpp_step_seconds": f"{step_time:.9f}",
                "perturb_result_seconds": f"{result_time:.9f}",
                "decode_batches": str(decode_batches),
                "decode_rows": str(decode_rows),
                "avg_decode_batch": f"{(decode_rows / decode_batches) if decode_batches else 0.0:.6f}",
            }
        )
    return [
        TspPerturbResult(
            list(route_row),
            set(int(node) for node in changed),
            introduced_edges={(int(u), int(v)) for u, v in introduced},
            removed_edges={(int(u), int(v)) for u, v in removed},
        )
        for route_row, changed, introduced, removed in zip(
            routes_batch,
            changeds_batch,
            introduced_batch,
            removed_batch,
        )
    ]


def perturb_tsp_batched_neural_python(
    instance: Instance,
    route: Sequence[int],
    policy: NativeTSPNeuralPolicy,
    rng: random.Random,
    min_new_edges: int,
    samples: int,
    *,
    edge_memory: SparseEdgeMemory | None = None,
    neural_knn_k: int = 32,
    neural_backup_k: int = 32,
    neural_knn_mask: bool = True,
    start_mode: str = "random",
    start_probes: int = 32,
    start_cost_weight: float = 1.0,
    start_policy_weight: float = 1.0,
    start_memory_weight: float = 1.0,
    profile: dict[str, Any] | None = None,
    neighbor_order: Sequence[Sequence[int]] | None = None,
) -> list[TspPerturbResult]:
    init_t0 = time.perf_counter()
    base_route = list(route)
    base_positions = tsp_positions(base_route)
    start_scores = build_tsp_perturb_start_scores(
        instance,
        base_route,
        base_positions,
        policy,
        rng,
        edge_memory=edge_memory,
        neural_knn_k=neural_knn_k,
        neural_knn_mask=neural_knn_mask,
        start_mode=start_mode,
        start_probes=start_probes,
        cost_weight=start_cost_weight,
        policy_weight=start_policy_weight,
        memory_weight=start_memory_weight,
    )
    states = []
    for _sample in range(samples):
        current, start_info = sample_tsp_perturb_start(start_scores, rng)
        candidate = list(route)
        positions = tsp_positions(candidate)
        states.append(make_batched_tsp_perturb_state_from_start(candidate, positions, current, start_info))
    init_time = time.perf_counter() - init_t0

    request_time = 0.0
    neural_time = 0.0
    step_time = 0.0
    decode_batches = 0
    decode_rows = 0
    while any(state.active for state in states):
        request_t0 = time.perf_counter()
        jobs: list[tuple[int, list[int]]] = []
        for state_index, state in enumerate(states):
            if not state.active:
                continue
            remaining = batched_tsp_remaining(state)
            primary = tsp_primary_candidates(instance, state.current, remaining, neural_knn_k, neighbor_order)
            if primary:
                jobs.append((state_index, primary))
                continue
            picked = tsp_backup_or_global_candidate(
                instance,
                state.current,
                remaining,
                neural_knn_k,
                neural_backup_k,
                neighbor_order,
            )
            apply_tsp_perturb_pick(state, picked, min_new_edges)
        request_time += time.perf_counter() - request_t0
        if not jobs:
            continue

        prefixes_batch = [states[state_index].prefix for state_index, _primary in jobs]
        candidates_batch = [primary for _state_index, primary in jobs]
        allowed_batch = [
            list(candidates) if candidates else None
            for candidates in candidates_batch
        ] if neural_knn_mask and policy.policy_id == "lehd" else None
        neural_t0 = time.perf_counter()
        probabilities_batch = policy.action_candidate_probabilities_batch(
            instance,
            prefixes_batch,
            candidates_batch,
            repair=True,
            allowed_candidates=allowed_batch,
        )
        neural_time += time.perf_counter() - neural_t0
        decode_batches += 1
        decode_rows += len(jobs)

        step_t0 = time.perf_counter()
        candidate_weights_batch = [
            tsp_candidate_heuristic_weights(
                instance,
                states[state_index].current,
                primary,
                probabilities,
                edge_memory=edge_memory,
            )
            for (state_index, primary), probabilities in zip(jobs, probabilities_batch)
        ]
        if any(sum(weights) <= 0.0 for weights in candidate_weights_batch):
            raise ValueError("neural TSP STAR produced no positive probability for legal k-NN actions")
        if STAR is not None and hasattr(STAR, "select_tsp_candidate_weight_actions_batch"):
            picked_batch = [
                int(node)
                for node in STAR.select_tsp_candidate_weight_actions_batch(
                    candidates_batch,
                    candidate_weights_batch,
                    [rng.random() for _job in jobs],
                )
            ]
        else:
            picked_batch = [
                sample_weighted(list(zip(primary, weights)), sum(weights), rng)
                for (_state_index, primary), weights in zip(jobs, candidate_weights_batch)
            ]
        for (state_index, _primary), picked in zip(jobs, picked_batch):
            apply_tsp_perturb_pick(states[state_index], picked, min_new_edges)
        step_time += time.perf_counter() - step_t0

    if profile is not None:
        profile.update(
            {
                "perturb_backend": "python",
                "perturb_init_seconds": f"{init_time:.9f}",
                "perturb_request_seconds": f"{request_time:.9f}",
                "neural_forward_seconds": f"{neural_time:.9f}",
                "cpp_step_seconds": f"{step_time:.9f}",
                "perturb_result_seconds": "0.000000000",
                "decode_batches": str(decode_batches),
                "decode_rows": str(decode_rows),
                "avg_decode_batch": f"{(decode_rows / decode_batches) if decode_batches else 0.0:.6f}",
            }
        )
    return [
        TspPerturbResult(
            list(state.candidate),
            set(state.changed),
            introduced_edges=set(state.introduced_edges),
            removed_edges=set(state.removed_edges),
            start_info=state.start_info,
        )
        for state in states
    ]


def perturb_tsp_batched_nearest(
    instance: Instance,
    route: Sequence[int],
    policy: AppendPolicy,
    rng: random.Random,
    min_new_edges: int,
    samples: int,
    *,
    edge_memory: SparseEdgeMemory | None = None,
    neural_knn_k: int = 32,
    neural_backup_k: int = 32,
    start_mode: str = "random",
    start_probes: int = 32,
    start_cost_weight: float = 1.0,
    start_policy_weight: float = 1.0,
    start_memory_weight: float = 1.0,
    profile: dict[str, Any] | None = None,
    neighbor_order: Sequence[Sequence[int]] | None = None,
) -> list[TspPerturbResult]:
    del profile, neighbor_order
    if samples <= 0:
        raise ValueError("STAR-samples must be positive")
    if not route:
        return [TspPerturbResult([], set()) for _ in range(samples)]
    if len(route) <= 2 or min_new_edges == 0:
        return [TspPerturbResult(list(route), set()) for _ in range(samples)]

    base_route = list(route)
    base_positions = tsp_positions(base_route)
    start_scores = build_tsp_perturb_start_scores(
        instance,
        base_route,
        base_positions,
        policy,
        rng,
        edge_memory=edge_memory,
        neural_knn_k=neural_knn_k,
        neural_knn_mask=False,
        start_mode=start_mode,
        start_probes=start_probes,
        cost_weight=start_cost_weight,
        policy_weight=start_policy_weight,
        memory_weight=start_memory_weight,
    )
    states = []
    for _ in range(samples):
        current, start_info = sample_tsp_perturb_start(start_scores, rng)
        candidate = list(route)
        positions = tsp_positions(candidate)
        states.append(make_batched_tsp_perturb_state_from_start(candidate, positions, current, start_info))
    nodes = sorted(instance.coords)
    node_to_index = {node: index for index, node in enumerate(nodes)}

    while any(state.active for state in states):
        jobs: list[tuple[int, list[int]]] = []
        for state_index, state in enumerate(states):
            if not state.active:
                continue
            remaining = batched_tsp_remaining(state)
            primary = tsp_primary_candidates(instance, state.current, remaining, neural_knn_k)
            if primary:
                jobs.append((state_index, primary))
                continue
            picked = tsp_backup_or_global_candidate(instance, state.current, remaining, neural_knn_k, neural_backup_k)
            apply_tsp_perturb_pick(state, picked, min_new_edges)

        if not jobs:
            continue

        probabilities_batch: list[list[float]] = []
        for state_index, primary in jobs:
            state = states[state_index]
            probabilities = [0.0] * len(nodes)
            total_h = 0.0
            for node in primary:
                d = raw_euclidean_distance(instance, state.current, node)
                h = 1.0 / max(d, 1e-12)
                probabilities[node_to_index[node]] = h
                total_h += h
            if total_h <= 0.0 or not math.isfinite(total_h):
                uniform = 1.0 / len(primary)
                for node in primary:
                    probabilities[node_to_index[node]] = uniform
            else:
                for node in primary:
                    node_index = node_to_index[node]
                    probabilities[node_index] /= total_h
            probabilities_batch.append(probabilities)

        if STAR is not None and hasattr(STAR, "select_tsp_candidate_actions_batch"):
            picked_batch = [
                int(node)
                for node in STAR.select_tsp_candidate_actions_batch(
                    nodes,
                    [states[state_index].current for state_index, _primary in jobs],
                    [primary for _state_index, primary in jobs],
                    probabilities_batch,
                    [],
                    [rng.random() for _job in jobs],
                )
            ]
            for (state_index, _primary), picked in zip(jobs, picked_batch):
                apply_tsp_perturb_pick(states[state_index], picked, min_new_edges)
        else:
            for (state_index, primary), probabilities in zip(jobs, probabilities_batch):
                state = states[state_index]
                picked = select_tsp_knn_action_python(
                    instance,
                    state.current,
                    primary,
                    probabilities,
                    nodes,
                    [],
                    neural_knn_k,
                    0,
                    rng,
                )
                apply_tsp_perturb_pick(state, picked, min_new_edges)

    return [
        TspPerturbResult(state.candidate, state.changed, state.introduced_edges, state.removed_edges, state.start_info)
        for state in states
    ]


def perturb_cvrp(
    instance: Instance,
    routes: Sequence[Sequence[int]],
    policy: AppendPolicy,
    rng: random.Random,
    min_new_edges: int,
    *,
    edge_memory: SparseEdgeMemory | None = None,
) -> CvrpPerturbResult:
    candidate = [list(route) for route in routes if route]
    customers = [node for route in candidate for node in route]
    if len(customers) <= 1 or min_new_edges == 0:
        return CvrpPerturbResult(candidate, set())

    source_route_id = source_cvrp_route_ids(candidate)
    source_edges = source_cvrp_edges(candidate)
    source_memory_edges = cvrp_memory_edges(instance, candidate)
    current = rng.choice(customers)
    current_route_index, _current_pos = find_route_pos(candidate, current)
    remaining_capacity = max(0, (instance.capacity or 0) - route_prefix_load(instance, candidate[current_route_index], current))
    prefix = [current]
    visited = {current}
    changed: set[int] = set()
    new_edges_cross = 0
    steps = 0
    max_steps = max(1, len(customers) * 4)

    while new_edges_cross < min_new_edges and len(visited) < len(customers) and steps <= max_steps:
        remaining = sorted(set(customers) - visited)
        if edge_memory is None:
            decision = select_cvrp_decision(instance, policy, current, remaining, rng, prefix, remaining_capacity)
        else:
            decision = select_cvrp_decision(instance, policy, current, remaining, rng, prefix, remaining_capacity, edge_memory=edge_memory)
        picked = decision.node
        if picked not in remaining:
            raise ValueError(f"{getattr(policy, 'policy_id', policy)} picked invalid CVRP node {picked}")

        route_index, pos = find_route_pos(candidate, current)
        successor = candidate[route_index][pos + 1] if pos + 1 < len(candidate[route_index]) else instance.depot
        demand = instance.demands.get(picked, 0)
        force_break = decision.route_break or demand > remaining_capacity
        transition_to = instance.depot if force_break else picked
        is_new = (normalize_cvrp_edge(current, transition_to) not in source_edges)

        if force_break:
            candidate = split_cvrp_after_current(candidate, current)
            changed.add(current)
            if is_new:
                new_edges_cross += 1
            target_route_index, _target_pos = find_route_pos(candidate, current)
            insert_route_index = min(target_route_index + 1, len(candidate) - 1)
            route_successor = candidate[insert_route_index][0] if candidate[insert_route_index] else None
            if picked == route_successor:
                remaining_capacity = max(0, (instance.capacity or 0) - demand)
            else:
                candidate = relocate_cvrp_customer_after_current_route(candidate, current, picked)
                changed.add(picked)
                if route_successor is not None:
                    changed.add(route_successor)
            remaining_capacity = max(0, (instance.capacity or 0) - demand)
        elif picked == successor:
            remaining_capacity -= demand
        else:
            changed.update({current, picked})
            if successor != instance.depot:
                changed.add(successor)
            candidate = relocate_cvrp_customer(instance, candidate, current, picked, False)
            route_index, _pos = find_route_pos(candidate, picked)
            remaining_capacity = max(0, (instance.capacity or 0) - route_prefix_load(instance, candidate[route_index], picked))
            if is_new:
                u_route = source_route_id.get(current, -1)
                v_route = source_route_id.get(picked, -2)
                if current == instance.depot or u_route != v_route:
                    new_edges_cross += 1

        visited.add(picked)
        prefix.append(picked)
        current = picked
        steps += 1

    final_routes = [route for route in candidate if route]
    final_memory_edges = cvrp_memory_edges(instance, final_routes)
    return CvrpPerturbResult(
        final_routes,
        changed,
        introduced_edges=final_memory_edges - source_memory_edges,
        removed_edges=source_memory_edges - final_memory_edges,
    )


def make_cvrp_perturb_state(
    instance: Instance,
    routes: Sequence[Sequence[int]],
    rng: random.Random,
) -> CvrpPerturbState:
    candidate = [list(route) for route in routes if route]
    customers = [node for route in candidate for node in route]
    current = rng.choice(customers)
    current_route_index, _current_pos = find_route_pos(candidate, current)
    return CvrpPerturbState(
        candidate=candidate,
        source_route_id=source_cvrp_route_ids(candidate),
        source_edges=source_cvrp_edges(candidate),
        source_memory_edges=cvrp_memory_edges(instance, candidate),
        current=current,
        remaining_capacity=max(0, (instance.capacity or 0) - route_prefix_load(instance, candidate[current_route_index], current)),
        prefix=[current],
        visited={current},
        changed=set(),
    )


def cvrp_apply_decision(
    instance: Instance,
    state: CvrpPerturbState,
    picked: int,
    route_break: bool,
) -> None:
    route_index, pos = find_route_pos(state.candidate, state.current)
    successor = state.candidate[route_index][pos + 1] if pos + 1 < len(state.candidate[route_index]) else instance.depot
    demand = instance.demands.get(picked, 0)
    force_break = route_break or demand > state.remaining_capacity
    transition_to = instance.depot if force_break else picked
    is_new = normalize_cvrp_edge(state.current, transition_to) not in state.source_edges

    if force_break:
        state.candidate = split_cvrp_after_current(state.candidate, state.current)
        state.changed.add(state.current)
        if is_new:
            state.new_edges_cross += 1
        target_route_index, _target_pos = find_route_pos(state.candidate, state.current)
        insert_route_index = min(target_route_index + 1, len(state.candidate) - 1)
        route_successor = state.candidate[insert_route_index][0] if state.candidate[insert_route_index] else None
        if picked == route_successor:
            state.remaining_capacity = max(0, (instance.capacity or 0) - demand)
        else:
            state.candidate = relocate_cvrp_customer_after_current_route(state.candidate, state.current, picked)
            state.changed.add(picked)
            if route_successor is not None:
                state.changed.add(route_successor)
        state.remaining_capacity = max(0, (instance.capacity or 0) - demand)
    elif picked == successor:
        state.remaining_capacity -= demand
    else:
        state.changed.update({state.current, picked})
        if successor != instance.depot:
            state.changed.add(successor)
        state.candidate = relocate_cvrp_customer(instance, state.candidate, state.current, picked, False)
        route_index, _pos = find_route_pos(state.candidate, picked)
        state.remaining_capacity = max(0, (instance.capacity or 0) - route_prefix_load(instance, state.candidate[route_index], picked))
        if is_new:
            u_route = state.source_route_id.get(state.current, -1)
            v_route = state.source_route_id.get(picked, -2)
            if state.current == instance.depot or u_route != v_route:
                state.new_edges_cross += 1

    state.visited.add(picked)
    state.prefix.append(picked)
    state.current = picked
    state.steps += 1


def finish_cvrp_perturb_state(instance: Instance, state: CvrpPerturbState) -> CvrpPerturbResult:
    final_routes = [route for route in state.candidate if route]
    final_memory_edges = cvrp_memory_edges(instance, final_routes)
    return CvrpPerturbResult(
        final_routes,
        state.changed,
        introduced_edges=final_memory_edges - state.source_memory_edges,
        removed_edges=state.source_memory_edges - final_memory_edges,
    )


def perturb_cvrp_batched_sil(
    instance: Instance,
    routes: Sequence[Sequence[int]],
    policy: NativeCVRPNeuralPolicy,
    rng: random.Random,
    min_new_edges: int,
    samples: int,
    *,
    edge_memory: SparseEdgeMemory | None = None,
    neural_knn_k: int = 32,
    neural_backup_k: int = 32,
    profile: dict[str, Any] | None = None,
    neighbor_order: Sequence[Sequence[int]] | None = None,
) -> list[CvrpPerturbResult]:
    candidate_routes = [list(route) for route in routes if route]
    customers = [node for route in candidate_routes for node in route]
    if len(customers) <= 1 or min_new_edges == 0:
        return [CvrpPerturbResult([list(route) for route in candidate_routes], set()) for _sample in range(samples)]

    init_t0 = time.perf_counter()
    context = STAR_context(instance)
    if neighbor_order is not None:
        context.neighbor_order(len(neighbor_order[0]) if neighbor_order else 0)
    engine = context.cvrp_perturb_batch(
        candidate_routes,
        int(samples),
        int(min_new_edges),
        int(neural_knn_k),
        int(neural_backup_k),
        int(rng.randrange(0, 2**63 - 1)),
    )
    init_time = time.perf_counter() - init_t0
    memory_values: list[tuple[int, int, float]] = []
    if edge_memory is not None:
        memory_values = [(u, v, edge_memory.weight(u, v)) for (u, v) in edge_memory.values]

    request_time = 0.0
    neural_time = 0.0
    step_time = 0.0
    decode_batches = 0
    decode_rows = 0
    while True:
        request_t0 = time.perf_counter()
        state_indices, prefixes, remaining_capacities = engine.requests()
        request_time += time.perf_counter() - request_t0
        if not state_indices:
            break
        neural_t0 = time.perf_counter()
        prob_rows = policy.action_probability_rows_batch(instance, prefixes, remaining_capacities)
        neural_time += time.perf_counter() - neural_t0
        decode_batches += 1
        decode_rows += len(state_indices)
        step_t0 = time.perf_counter()
        engine.step(
            state_indices,
            prob_rows,
            memory_values,
            [rng.random() for _state_index in state_indices],
        )
        step_time += time.perf_counter() - step_t0

    result_t0 = time.perf_counter()
    routes_batch, changeds_batch, introduced_batch, removed_batch = engine.results()
    result_time = time.perf_counter() - result_t0
    if profile is not None:
        profile.update(
            {
                "perturb_init_seconds": f"{init_time:.9f}",
                "perturb_request_seconds": f"{request_time:.9f}",
                "neural_forward_seconds": f"{neural_time:.9f}",
                "cpp_step_seconds": f"{step_time:.9f}",
                "perturb_result_seconds": f"{result_time:.9f}",
                "decode_batches": str(decode_batches),
                "decode_rows": str(decode_rows),
                "avg_decode_batch": f"{(decode_rows / decode_batches) if decode_batches else 0.0:.6f}",
            }
        )
    return [
        CvrpPerturbResult(
            [list(route) for route in routes],
            set(int(node) for node in changed),
            introduced_edges={(int(u), int(v)) for u, v in introduced},
            removed_edges={(int(u), int(v)) for u, v in removed},
        )
        for routes, changed, introduced, removed in zip(routes_batch, changeds_batch, introduced_batch, removed_batch)
    ]


def select_cvrp_decision(
    instance: Instance,
    policy: AppendPolicy,
    current: int,
    candidates: Sequence[int],
    rng: random.Random,
    prefix: Sequence[int],
    remaining_capacity: int,
    edge_memory: SparseEdgeMemory | None = None,
) -> CVRPDecision:
    if not candidates:
        raise ValueError("no CVRP candidates")

    capacity = instance.capacity or 0
    feasible_now = [node for node in candidates if instance.demands.get(node, 0) <= remaining_capacity]
    feasible_full = [node for node in candidates if instance.demands.get(node, 0) <= capacity]
    if not feasible_full:
        raise ValueError("no feasible CVRP customer under vehicle capacity")

    if isinstance(policy, NativeCVRPNeuralPolicy):
        if edge_memory is not None:
            return select_cvrp_neural_decision_with_memory(
                instance,
                policy,
                current,
                feasible_now,
                feasible_full,
                rng,
                prefix,
                remaining_capacity,
                edge_memory,
            )
        if policy.policy_id not in {"bq", "lehd", "sil", "icam", "elg"}:
            raise ValueError(f"STAR CVRP neural adapter is currently extracted for bq/lehd/sil/icam/elg, not {policy.policy_id}")
        picked, route_break = neural_cvrp_select_next(instance, policy, prefix, remaining_capacity)
        if picked not in candidates:
            raise ValueError(
                f"{policy.policy_id} picked invalid CVRP node {picked}; "
                f"{len(candidates)} legal customers remain"
            )
        return CVRPDecision(picked, route_break or picked not in feasible_now)

    if isinstance(policy, SoftDistPolicy):
        if feasible_now:
            if current != instance.depot:
                nearest_feasible = min(feasible_now, key=lambda node: (distance(instance, current, node), node))
                if distance(instance, current, instance.depot) < distance(instance, current, nearest_feasible):
                    picked = select_policy_with_optional_memory(instance, policy, instance.depot, feasible_full, rng, prefix, edge_memory)
                    return CVRPDecision(picked, True)
            picked = select_policy_with_optional_memory(instance, policy, current, feasible_now, rng, prefix, edge_memory)
            return CVRPDecision(picked, False)
        picked = select_policy_with_optional_memory(instance, policy, instance.depot, feasible_full, rng, prefix, edge_memory)
        return CVRPDecision(picked, True)

    if feasible_now:
        if current != instance.depot:
            nearest_feasible = min(feasible_now, key=lambda node: (distance(instance, current, node), node))
            if distance(instance, current, instance.depot) < distance(instance, current, nearest_feasible):
                picked = select_policy_with_optional_memory(instance, policy, instance.depot, feasible_full, rng, prefix, edge_memory)
                return CVRPDecision(picked, True)
        picked = select_policy_with_optional_memory(instance, policy, current, feasible_now, rng, prefix, edge_memory)
        return CVRPDecision(picked, False)
    picked = select_policy_with_optional_memory(instance, policy, instance.depot, feasible_full, rng, prefix, edge_memory)
    return CVRPDecision(picked, True)


def select_cvrp_neural_decision_with_memory(
    instance: Instance,
    policy: NativeCVRPNeuralPolicy,
    current: int,
    feasible_now: Sequence[int],
    feasible_full: Sequence[int],
    rng: random.Random,
    prefix: Sequence[int],
    remaining_capacity: int,
    edge_memory: SparseEdgeMemory,
) -> CVRPDecision:
    if policy.policy_id != "sil":
        raise ValueError(
            f"STAR edge memory for CVRP neural policy {policy.policy_id} needs probability-vector extraction; "
            "currently implemented only for SIL"
        )
    direct_probs, via_depot_probs = policy.action_probabilities(instance, prefix, remaining_capacity)
    pool: list[tuple[CVRPDecision, float]] = []

    for node in feasible_now:
        probability = direct_probs.get(node, 0.0)
        weight = probability * edge_memory.weight(current, node)
        if weight > 0.0 and math.isfinite(weight):
            pool.append((CVRPDecision(node, False), weight))

    depot = instance.depot
    for node in feasible_full:
        probability = via_depot_probs.get(node, 0.0)
        weight = probability * edge_memory.weight(current, depot) * edge_memory.weight(depot, node)
        if weight > 0.0 and math.isfinite(weight):
            pool.append((CVRPDecision(node, True), weight))

    total = sum(weight for _decision, weight in pool)
    if total <= 0.0:
        raise ValueError(f"{policy.policy_id} produced no positive probability for legal CVRP STAR actions")
    return sample_weighted(pool, total, rng)


def select_policy_with_optional_memory(
    instance: Instance,
    policy: AppendPolicy,
    current: int,
    candidates: Sequence[int],
    rng: random.Random,
    prefix: Sequence[int],
    edge_memory: SparseEdgeMemory | None,
) -> int:
    if edge_memory is None:
        return policy.select_next(instance, current, candidates, rng, prefix, repair=True)
    if instance.problem == "tsp":
        return select_tsp_decision_with_memory(instance, policy, current, candidates, rng, prefix, edge_memory)
    if isinstance(policy, NativeCVRPNeuralPolicy):
        raise ValueError("STAR edge memory for CVRP neural policies needs probability-vector extraction; refusing decoded-node bias")
    if isinstance(policy, SoftDistPolicy):
        candidate_set = set(candidates)
        pool: list[tuple[int, float]] = []
        for node, dist_value in policy._nearest_distances(instance, current):
            if node in candidate_set:
                pool.append((node, math.exp(-dist_value / max(policy.temperature, 1e-12)) * edge_memory.weight(current, node)))
                if len(pool) >= policy.knn:
                    break
        if not pool:
            raise ValueError("softdist received no legal candidates")
        return sample_weighted(pool, sum(weight for _node, weight in pool), rng)
    if isinstance(policy, NearestPolicy):
        return max(candidates, key=lambda node: (edge_memory.weight(current, node), -distance(instance, current, node), -node))
    return policy.select_next(instance, current, candidates, rng, prefix, repair=True)


def neural_cvrp_select_next(
    instance: Instance,
    policy: NativeCVRPNeuralPolicy,
    prefix: Sequence[int],
    remaining_capacity: int,
) -> tuple[int, bool]:
    policy._ensure_instance(instance)
    assert policy._base_problems is not None
    assert policy._solution_tensor is not None
    assert policy._raw_capacity_tensor is not None
    problems = policy._base_problems.clone()
    problems[:, :, 3] = float(remaining_capacity)
    selected = torch.tensor([[policy._node_to_index[node] for node in prefix]], dtype=torch.long, device=problems.device)
    state = SimpleNamespace(problems=problems, first_node=None, current_node=None)
    with original_torch_device_context():
        with torch.no_grad():
            _loss, _selected_teacher, selected_student, _flag_teacher, flag_student = native_forward_swap._call_cvrp_neural(
                policy._spec,
                state,
                selected,
                policy._solution_tensor,
                len(prefix),
                raw_data_capacity=policy._raw_capacity_tensor,
                decode_method="greedy",
            )
    picked_index = int(selected_student.reshape(-1)[0].item())
    route_break = bool(int(flag_student.reshape(-1)[0].item()))
    if picked_index <= 0 or picked_index >= len(policy._node_ids):
        raise ValueError(f"{policy.policy_id} returned out-of-range CVRP customer index {picked_index}")
    return policy._node_ids[picked_index], route_break


def refine_tsp_local(
    instance: Instance,
    route: Sequence[int],
    changed: set[int],
    refine_k: int,
    *,
    neighbor_order: Sequence[Sequence[int]] | None = None,
) -> list[int]:
    candidate = list(route)
    if len(candidate) < 4 or not changed:
        return candidate
    return list(STAR_context(instance).refine_tsp(candidate, list(changed), refine_k))


def refine_tsp_candidates_local(
    instance: Instance,
    candidates: Sequence[TspPerturbResult],
    refine_k: int,
    *,
    refine: bool = True,
    neighbor_order: Sequence[Sequence[int]] | None = None,
) -> list[TspRefinedCandidate]:
    if not candidates:
        return []
    if not refine:
        return [
            TspRefinedCandidate(list(perturb.route), tsp_cost(instance, perturb.route), perturb)
            for perturb in candidates
        ]
    if len(candidates) > 1:
        routes = [list(candidate.route) for candidate in candidates]
        changeds = [list(candidate.changed) for candidate in candidates]
        refined_routes, costs = STAR_context(instance).refine_tsp_batch(routes, changeds, refine_k)
        return [
            TspRefinedCandidate(list(route), float(cost), perturb)
            for route, cost, perturb in zip(refined_routes, costs, candidates)
        ]

    refined: list[TspRefinedCandidate] = []
    for perturb in candidates:
        candidate = refine_tsp_local(
            instance,
            perturb.route,
            perturb.changed,
            refine_k,
            neighbor_order=neighbor_order,
        ) if perturb.changed else list(perturb.route)
        refined.append(TspRefinedCandidate(candidate, tsp_cost(instance, candidate), perturb))
    return refined


def tsp_positions(route: Sequence[int]) -> dict[int, int]:
    return {node: index for index, node in enumerate(route)}


def tsp_successor(route: Sequence[int], positions: dict[int, int], node: int) -> int:
    index = positions[node]
    return route[(index + 1) % len(route)]


def tsp_predecessor(route: Sequence[int], positions: dict[int, int], node: int) -> int:
    index = positions[node]
    return route[(index - 1) % len(route)]


def flip_tsp_section(route: list[int], start_node: int, end_node: int, positions: dict[int, int] | None = None) -> None:
    if start_node == end_node:
        return
    if positions is None:
        positions = tsp_positions(route)
    n = len(route)
    first = positions[start_node]
    last = positions[end_node]
    if first > last:
        first, last = last, first

    segment_length = last - first
    remaining_length = n - segment_length
    if segment_length <= remaining_length:
        route[first:last] = reversed(route[first:last])
        update_tsp_positions(route, positions, range(first, last))
        return

    indices = list(range(last, n)) + list(range(0, first))
    values = [route[index] for index in indices]
    for index, value in zip(indices, reversed(values)):
        route[index] = value
        positions[value] = index


def update_tsp_positions(route: Sequence[int], positions: dict[int, int], indices: Iterable[int]) -> None:
    for index in indices:
        positions[route[index]] = index


def refine_cvrp_local(
    instance: Instance,
    routes: Sequence[Sequence[int]],
    changed: set[int],
    refine_k: int,
) -> list[list[int]]:
    candidate = [list(route) for route in routes if route]
    checklist = [node for node in changed if node != instance.depot]
    if checklist:
        require_cpp_STAR_srr("cvrp", "CVRP STAR SRR refinement", "refine_cvrp_srr")
        nodes, coords, demands = STAR_srr_instance_payload(instance)
        return [
            list(route)
            for route in STAR.refine_cvrp_srr(
                nodes,
                coords,
                demands,
                instance.depot,
                instance.capacity or 0,
                instance.edge_weight_type,
                candidate,
                checklist,
                refine_k,
            )
            if route
        ]
    return [route for route in candidate if route]


def refine_cvrp_candidates_local(
    instance: Instance,
    candidates: Sequence[CvrpPerturbResult],
    refine_k: int,
    *,
    refine: bool = True,
) -> list[CvrpRefinedCandidate]:
    if not candidates:
        return []
    if not refine:
        return [
            CvrpRefinedCandidate([list(route) for route in perturb.routes if route], cvrp_cost(instance, perturb.routes), perturb)
            for perturb in candidates
        ]
    if len(candidates) > 1:
        require_cpp_STAR_srr("cvrp", "batched CVRP STAR SRR refinement", "refine_cvrp_srr_batch")
        nodes, coords, demands = STAR_srr_instance_payload(instance)
        routes_batch = [[list(route) for route in perturb.routes] for perturb in candidates]
        changeds = [list(perturb.changed) for perturb in candidates]
        refined_routes_batch, costs = STAR.refine_cvrp_srr_batch(
            nodes,
            coords,
            demands,
            instance.depot,
            instance.capacity or 0,
            instance.edge_weight_type,
            routes_batch,
            changeds,
            refine_k,
        )
        return [
            CvrpRefinedCandidate([list(route) for route in routes if route], float(cost), perturb)
            for routes, cost, perturb in zip(refined_routes_batch, costs, candidates)
        ]

    refined: list[CvrpRefinedCandidate] = []
    for perturb in candidates:
        routes = refine_cvrp_local(instance, perturb.routes, perturb.changed, refine_k) if perturb.changed else [list(route) for route in perturb.routes]
        refined.append(CvrpRefinedCandidate(routes, cvrp_cost(instance, routes), perturb))
    return refined


def require_cpp_STAR_srr(problem: str, feature: str, symbol: str | None = None) -> None:
    if STAR is None:
        raise RuntimeError(f"{feature} requires the STAR C++ extension")
    if symbol is not None and not hasattr(STAR, symbol):
        raise RuntimeError(f"{feature} requires STAR.{symbol}")


def use_cpp_STAR_srr(problem: str) -> bool:
    return STAR is not None


def refresh_instance_cache_ref(instance: Instance) -> bool:
    key = id(instance)
    ref = _INSTANCE_CACHE_REFS.get(key)
    if ref is None or ref() is instance:
        if ref is None:
            _INSTANCE_CACHE_REFS[key] = weakref.ref(instance)
        return False
    _purge_instance_caches(key)
    _INSTANCE_CACHE_REFS[key] = weakref.ref(instance)
    return True


def STAR_srr_instance_payload(instance: Instance) -> tuple[list[int], list[list[float]], list[int]]:
    nodes = sorted(instance.coords)
    coords = [[float(instance.coords[node][0]), float(instance.coords[node][1])] for node in nodes]
    demands = [int(instance.demands.get(node, 0)) for node in nodes]
    return nodes, coords, demands


_STAR_CONTEXT_CACHE: dict[int, Any] = {}
_INSTANCE_ARRAY_CACHE: dict[int, tuple[list[int], dict[int, int], np.ndarray]] = {}
_INSTANCE_CACHE_REFS: dict[int, weakref.ReferenceType[Instance]] = {}

_INSTANCE_CACHE_SWEEP_TICKS = 0
_INSTANCE_CACHE_SWEEP_EVERY = 512


def _purge_instance_caches(instance_key: int) -> None:
    _STAR_CONTEXT_CACHE.pop(instance_key, None)
    _INSTANCE_ARRAY_CACHE.pop(instance_key, None)


def _sweep_dead_instance_caches() -> None:
    dead_keys = [key for key, ref in _INSTANCE_CACHE_REFS.items() if ref() is None]
    for key in dead_keys:
        _purge_instance_caches(key)
        _INSTANCE_CACHE_REFS.pop(key, None)


def instance_cache_key(instance: Instance) -> int:
    global _INSTANCE_CACHE_SWEEP_TICKS
    _INSTANCE_CACHE_SWEEP_TICKS += 1
    if _INSTANCE_CACHE_SWEEP_TICKS >= _INSTANCE_CACHE_SWEEP_EVERY:
        _INSTANCE_CACHE_SWEEP_TICKS = 0
        _sweep_dead_instance_caches()

    key = id(instance)
    ref = _INSTANCE_CACHE_REFS.get(key)
    if ref is None or ref() is not instance:
        _purge_instance_caches(key)
        _INSTANCE_CACHE_REFS[key] = weakref.ref(instance)
    return key


def nearest_nodes_excluding(instance: Instance, node: int, excluded: int, limit: int) -> list[int]:
    if limit <= 0:
        return []
    if node not in instance.coords:
        raise ValueError(f"node {node} not found in instance.coords")
    if excluded not in instance.coords:
        raise ValueError(f"excluded node {excluded} not found in instance.coords")
    total = min(max(0, len(instance.coords) - 1), limit + int(excluded != node))
    result: list[int] = []
    for other in STAR_neighbor_row(instance, STAR_neighbor_order(instance, total), node):
        if other != excluded:
            result.append(other)
            if len(result) >= limit:
                break
    return result


def STAR_neighbor_order(instance: Instance, total: int) -> list[list[int]]:
    total = max(0, min(total, max(0, len(instance.coords) - 1)))
    return [
        [int(node) for node in row]
        for row in STAR_context(instance).neighbor_order(total)
    ]


def STAR_context(instance: Instance) -> Any:
    instance_key = instance_cache_key(instance)
    cached = _STAR_CONTEXT_CACHE.get(instance_key)
    if cached is not None:
        return cached
    if STAR is None or not hasattr(STAR, "STAR"):
        raise ValueError("C++ STAR context extension is required for STAR operations")
    nodes, coords, demands = STAR_srr_instance_payload(instance)
    context = STAR.STAR(
        nodes,
        coords,
        demands,
        int(instance.depot),
        int(instance.capacity or 0),
        instance.edge_weight_type,
    )
    _STAR_CONTEXT_CACHE[instance_key] = context
    return context


def STAR_tsp_neighbor_order(instance: Instance, total: int) -> list[list[int]]:
    if instance.problem != "tsp":
        raise ValueError("STAR TSP neighbor order requested for non-TSP instance")
    return STAR_neighbor_order(instance, total)


def STAR_neighbor_row(instance: Instance, neighbor_order: Sequence[Sequence[int]], node: int) -> Sequence[int]:
    _node_ids, node_to_index, _coords = instance_arrays(instance)
    return neighbor_order[node_to_index[node]]


def STAR_tsp_neighbor_row(instance: Instance, neighbor_order: Sequence[Sequence[int]], node: int) -> Sequence[int]:
    return STAR_neighbor_row(instance, neighbor_order, node)


def instance_arrays(instance: Instance) -> tuple[list[int], dict[int, int], np.ndarray]:
    key = instance_cache_key(instance)
    cached = _INSTANCE_ARRAY_CACHE.get(key)
    if cached is not None:
        return cached
    node_ids = sorted(instance.coords)
    node_to_index = {node: index for index, node in enumerate(node_ids)}
    coords = np.asarray([instance.coords[node] for node in node_ids], dtype=np.float64)
    cached = (node_ids, node_to_index, coords)
    _INSTANCE_ARRAY_CACHE[key] = cached
    return cached


def relocate_after(route: Sequence[int], target: int, node: int) -> list[int]:
    candidate = list(route)
    if target == node:
        return candidate
    candidate.remove(node)
    target_index = candidate.index(target)
    candidate.insert(target_index + 1, node)
    return candidate


def relocate_after_inplace(route: list[int], positions: dict[int, int], target: int, node: int) -> None:
    if target == node:
        return
    node_pos = positions[node]
    target_pos = positions[target]
    if (target_pos + 1) % len(route) == node_pos:
        return
    node_value = route[node_pos]
    if target_pos < node_pos:
        for index in range(node_pos, target_pos + 1, -1):
            route[index] = route[index - 1]
        route[target_pos + 1] = node_value
        update_tsp_positions(route, positions, range(target_pos + 1, node_pos + 1))
    else:
        for index in range(node_pos, target_pos):
            route[index] = route[index + 1]
        route[target_pos] = node_value
        update_tsp_positions(route, positions, range(node_pos, target_pos + 1))


def relocate_cvrp_customer(
    instance: Instance,
    routes: Sequence[Sequence[int]],
    current: int,
    picked: int,
    route_break: bool,
) -> list[list[int]]:
    candidate = [list(route) for route in routes if route]
    current_route, current_pos = find_route_pos(candidate, current)
    if route_break:
        remove_customer(candidate, picked)
        current_route, _current_pos = find_route_pos(candidate, current)
        candidate.insert(current_route + 1, [picked])
        return [route for route in candidate if route]

    remove_customer(candidate, picked)
    current_route, current_pos = find_route_pos(candidate, current)
    if route_load(instance, candidate[current_route]) + instance.demands.get(picked, 0) > (instance.capacity or 0):
        candidate.insert(current_route + 1, [picked])
    else:
        candidate[current_route].insert(current_pos + 1, picked)
    return [route for route in candidate if route]


def source_cvrp_route_ids(routes: Sequence[Sequence[int]]) -> dict[int, int]:
    return {node: route_index for route_index, route in enumerate(routes) for node in route}


def source_cvrp_edges(routes: Sequence[Sequence[int]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for route in routes:
        previous = 1
        for node in route:
            edges.add(normalize_cvrp_edge(previous, node))
            previous = node
        edges.add(normalize_cvrp_edge(previous, 1))
    return edges


def cvrp_memory_edges(instance: Instance, routes: Sequence[Sequence[int]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    depot = instance.depot
    for route in routes:
        previous = depot
        for node in route:
            edges.add((previous, node))
            edges.add((node, previous))
            previous = node
        edges.add((previous, depot))
        edges.add((depot, previous))
    return edges


def normalize_cvrp_edge(a: int, b: int) -> tuple[int, int]:
    if a == 1:
        return (1, b)
    if b == 1:
        return (1, a)
    return (a, b) if a <= b else (b, a)


def split_cvrp_after_current(routes: Sequence[Sequence[int]], current: int) -> list[list[int]]:
    candidate = [list(route) for route in routes if route]
    route_index, pos = find_route_pos(candidate, current)
    route = candidate[route_index]
    suffix = route[pos + 1 :]
    candidate[route_index] = route[: pos + 1]
    if suffix:
        candidate.insert(route_index + 1, suffix)
    else:
        candidate.insert(route_index + 1, [])
    return candidate


def relocate_cvrp_customer_after_current_route(
    routes: Sequence[Sequence[int]],
    current: int,
    picked: int,
) -> list[list[int]]:
    candidate = [list(route) for route in routes]
    remove_customer(candidate, picked)
    current_route_index, _pos = find_route_pos(candidate, current)
    insert_route_index = min(current_route_index + 1, len(candidate) - 1)
    if insert_route_index < 0:
        candidate.append([picked])
    else:
        candidate[insert_route_index].insert(0, picked)
    return [route for route in candidate if route]


def find_route_pos(routes: Sequence[Sequence[int]], node: int) -> tuple[int, int]:
    for route_index, route in enumerate(routes):
        for pos, value in enumerate(route):
            if value == node:
                return route_index, pos
    raise ValueError(f"node {node} not found in routes")


def remove_customer(routes: list[list[int]], node: int) -> None:
    route_index, pos = find_route_pos(routes, node)
    del routes[route_index][pos]
    if not routes[route_index]:
        del routes[route_index]


def route_prefix_load(instance: Instance, route: Sequence[int], through_node: int) -> int:
    load = 0
    for node in route:
        load += instance.demands.get(node, 0)
        if node == through_node:
            break
    return load


def tsp_cost(instance: Instance, route: Sequence[int]) -> float:
    return sum(distance(instance, route[i], route[(i + 1) % len(route)]) for i in range(len(route)))


def cvrp_cost(instance: Instance, routes: Sequence[Sequence[int]]) -> float:
    total = 0.0
    for route in routes:
        current = instance.depot
        for node in route:
            total += distance(instance, current, node)
            current = node
        total += distance(instance, current, instance.depot)
    return total


def validate_tsp(instance: Instance, route: Sequence[int]) -> bool:
    return sorted(route) == sorted(instance.coords)


def validate_cvrp(instance: Instance, routes: Sequence[Sequence[int]]) -> bool:
    customers = [node for route in routes for node in route]
    if sorted(customers) != sorted(set(instance.coords) - {instance.depot}):
        return False
    return all(route_load(instance, route) <= (instance.capacity or 0) for route in routes)


def route_load(instance: Instance, route: Sequence[int]) -> int:
    return sum(instance.demands.get(node, 0) for node in route)


def distance(instance: Instance, a: int, b: int) -> int:
    ax, ay = instance.coords[a]
    bx, by = instance.coords[b]
    value = math.hypot(ax - bx, ay - by)
    if instance.edge_weight_type == "CEIL_2D":
        return int(math.ceil(value))
    if instance.edge_weight_type == "EUC_2D":
        return int(math.floor(value + 0.5))
    return int(value)


def raw_euclidean_distance(instance: Instance, a: int, b: int) -> float:
    ax, ay = instance.coords[a]
    bx, by = instance.coords[b]
    return math.hypot(ax - bx, ay - by)


def normalized_distance(instance: Instance, a: int, b: int) -> float:
    ax, ay = instance.coords[a]
    bx, by = instance.coords[b]
    xs = [x for x, _y in instance.coords.values()]
    ys = [y for _x, y in instance.coords.values()]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-12)
    return math.hypot(ax - bx, ay - by) / scale


def load_instance(problem: str) -> Instance:
    return load_tsp(DEFAULT_TSP) if problem == "tsp" else load_cvrp(DEFAULT_CVRP)


def load_instance_path(problem: str, path: Path) -> Instance:
    return load_tsp(path) if problem == "tsp" else load_cvrp(path)


def instance_paths(problem: str, size: str | None) -> list[Path]:
    if size is None:
        return [DEFAULT_TSP if problem == "tsp" else DEFAULT_CVRP]
    if size not in {"dev", "dev-medium3", "dev-medium", "small", "medium", "large", "full"}:
        raise ValueError(f"unknown size: {size}")

    directory = TSP_BENCH_DIR if problem == "tsp" else CVRP_BENCH_DIR
    suffix = ".tsp" if problem == "tsp" else ".vrp"
    dimensioned_paths: list[tuple[int, Path]] = []
    for path in sorted(directory.glob(f"*{suffix}")):
        dimension = read_dimension(path)
        if dimension is None:
            continue
        dimensioned_paths.append((dimension, path))
    if size == "dev":
        return [path for _dimension, path in sorted(dimensioned_paths, key=lambda item: (item[0], item[1].name))[:15]]
    if size == "dev-medium3":
        return [
            path
            for dimension, path in sorted(dimensioned_paths, key=lambda item: (item[0], item[1].name))
            if 1000 <= dimension < 10000
        ][:3]
    if size == "dev-medium":
        return [
            path
            for dimension, path in sorted(dimensioned_paths, key=lambda item: (item[0], item[1].name))
            if 1000 <= dimension < 10000
        ][:5]

    paths = []
    for dimension, path in sorted(dimensioned_paths, key=lambda item: item[1].name):
        if size == "full":
            paths.append(path)
        elif size == "small" and dimension < 1000:
            paths.append(path)
        elif size == "medium" and 1000 <= dimension < 10000:
            paths.append(path)
        elif size == "large" and dimension >= 10000:
            paths.append(path)
    return paths


def read_dimension(path: Path) -> int | None:
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("DIMENSION"):
            value = line.split(":", 1)[-1] if ":" in line else line.split()[-1]
            try:
                return int(value.strip())
            except ValueError:
                return None
        if line in {"NODE_COORD_SECTION", "DEMAND_SECTION"}:
            return None
    return None


def load_tsp(path: Path) -> Instance:
    coords: dict[int, tuple[float, float]] = {}
    name = path.stem
    edge_weight_type = "EUC_2D"
    in_coords = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("NAME"):
            name = line.split(":", 1)[-1].strip()
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":", 1)[-1].strip() if ":" in line else line.split()[-1].strip()
        elif line == "NODE_COORD_SECTION":
            in_coords = True
        elif line == "EOF":
            break
        elif in_coords and line:
            node, x, y = line.split()[:3]
            coords[int(node)] = (float(x), float(y))
    return Instance(name, "tsp", coords, {}, None, tsp_bks(name, path.stem), path, edge_weight_type)


def load_cvrp(path: Path) -> Instance:
    coords: dict[int, tuple[float, float]] = {}
    demands: dict[int, int] = {}
    capacity: int | None = None
    bks: float | None = None
    name = path.stem
    section = ""
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("NAME"):
            name = line.split(":", 1)[-1].strip()
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":", 1)[-1])
        elif "COMMENT" in line and "cost" in line.lower():
            digits = "".join(ch if ch.isdigit() or ch == "." else " " for ch in line).split()
            if digits:
                bks = float(digits[-1])
        elif line in {"NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"}:
            section = line
        elif line == "EOF":
            break
        elif section == "NODE_COORD_SECTION":
            node, x, y = line.split()[:3]
            coords[int(node)] = (float(x), float(y))
        elif section == "DEMAND_SECTION":
            node, demand = line.split()[:2]
            demands[int(node)] = int(demand)
    if bks is None:
        bks = cvrp_bks(name, path.stem)
    return Instance(name, "cvrp", coords, demands, capacity, bks, path)


def tsp_bks(name: str, stem: str) -> float | None:
    return survey_bks("tsp_survey_bench_cost_all", name, stem)


def cvrp_bks(name: str, stem: str) -> float | None:
    return survey_bks("cvrp_survey_bench_cost_all", name, stem)


def survey_bks(variable_name: str, name: str, stem: str) -> float | None:
    bks_path = ROOT / "0_data_survey/survey_bench_opt_tsp_same_file_name.py"
    spec = importlib.util.spec_from_file_location("_nrs_bks", bks_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = getattr(module, variable_name, {})
    value = data.get(stem) or data.get(name)
    return float(value) if value is not None else None


def run_one(
    strategy_id: str,
    policy_id: str,
    problem: str,
    seed: int,
    *,
    instance: Instance | None = None,
    iterations: int | None = None,
    min_new_edges: int = 24,
    refine_k: int = 64,
    refine: bool = True,
    neural_knn_k: int = 32,
    neural_backup_k: int = 32,
    neural_knn_mask: bool = True,
    STAR_samples: int = 64,
    STAR_memory: bool = True,
    STAR_memory_k: int = 32,
    STAR_memory_rho: float = 0.9,
    STAR_memory_tau_min: float | None = None,
    STAR_memory_tau_max: float = 1.0,
    STAR_memory_alpha: float = 1.0,
    STAR_start_mode: str = "random",
    STAR_start_probes: int = 32,
    STAR_start_cost_weight: float = 1.0,
    STAR_start_policy_weight: float = 1.0,
    STAR_start_memory_weight: float = 1.0,
    STAR_memory_update_mode: str = "auto",
    STAR_advantage_scale: float = 100.0,
    STAR_advantage_min: float = 0.0,
    STAR_trace: bool = False,
    STAR_profile: bool = False,
    STAR_trace_rows: list[dict[str, str]] | None = None,
    STAR_profile_rows: list[dict[str, str]] | None = None,
    STAR_progress_callback: Callable[[dict[str, str]], None] | None = None,
) -> dict[str, str]:
    if strategy_id.lower() == "star":
        strategy_id = "STAR"
    strategy = search_strategy(
        strategy_id,
        iterations=iterations,
        min_new_edges=min_new_edges,
        refine_k=refine_k,
        refine=refine,
        neural_knn_k=neural_knn_k,
        neural_backup_k=neural_backup_k,
        neural_knn_mask=neural_knn_mask,
        STAR_samples=STAR_samples,
        memory=STAR_memory,
        memory_k=STAR_memory_k,
        memory_rho=STAR_memory_rho,
        memory_tau_min=STAR_memory_tau_min,
        memory_tau_max=STAR_memory_tau_max,
        memory_alpha=STAR_memory_alpha,
        start_mode=STAR_start_mode,
        start_probes=STAR_start_probes,
        start_cost_weight=STAR_start_cost_weight,
        start_policy_weight=STAR_start_policy_weight,
        start_memory_weight=STAR_start_memory_weight,
        memory_update_mode=STAR_memory_update_mode,
        advantage_scale=STAR_advantage_scale,
        advantage_min=STAR_advantage_min,
        STAR_trace=STAR_trace,
        STAR_profile=STAR_profile,
        STAR_progress_callback=STAR_progress_callback,
    )
    instance = instance if instance is not None else load_instance(problem)
    policy = append_policy(policy_id, problem)
    if isinstance(strategy, UnsupportedStrategy):
        return row("unsupported", strategy_id, policy_id, problem, instance, reason=strategy.reason)
    if isinstance(policy, UnsupportedPolicy):
        return row("unsupported", strategy_id, policy_id, problem, instance, reason=policy.reason)
    if isinstance(policy, NativeCVRPNeuralPolicy) and not (
        (strategy_id == "greedy" and policy_id in {"bq", "lehd", "sil", "icam", "elg"})
        or (strategy_id in {"lehd_rrc", "sil_prc", "rrc", "prc"} and policy_id in {"bq", "lehd", "sil", "icam", "elg"})
        or (strategy_id == "STAR" and policy_id in {"bq", "lehd", "sil", "icam", "elg"})
    ):
        return row(
            "unsupported",
            strategy_id,
            policy_id,
            problem,
            instance,
            reason="CVRP neural policy is only enabled for greedy and generic repair shells with bq/lehd/sil/icam/elg",
        )
    prepare_policy_for_timing(policy, instance)
    start = time.perf_counter()
    torch.manual_seed(seed)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(seed)
    initial, final, valid = strategy.run(instance, policy, random.Random(seed))
    if STAR_trace_rows is not None and isinstance(strategy, STARStrategy):
        for trace_row in strategy.STAR_trace_rows:
            trace_row["problem"] = problem
            trace_row["strategy_id"] = strategy_id
        STAR_trace_rows.extend(strategy.STAR_trace_rows)
    if STAR_profile_rows is not None and isinstance(strategy, STARStrategy):
        for profile_row in strategy.STAR_profile_rows:
            profile_row.setdefault("strategy_id", strategy_id)
            profile_row.setdefault("run_policy_id", policy_id)
            profile_row.setdefault("run_problem", problem)
        STAR_profile_rows.extend(strategy.STAR_profile_rows)
    bks = instance.bks_cost
    gap = ((final - bks) / bks * 100) if bks else ""
    result = row(
        "ok" if valid else "invalid",
        strategy_id,
        policy_id,
        problem,
        instance,
        initial=initial,
        final=final,
        gap=gap,
        elapsed=time.perf_counter() - start,
    )
    return result


def prepare_policy_for_timing(policy: AppendPolicy | UnsupportedPolicy, instance: Instance) -> None:
    if isinstance(policy, NativeTSPNeuralPolicy):
        policy._ensure_instance(instance)
        if policy._coords_tensor is not None:
            native_forward_swap._load_neural_model(policy._spec, policy._coords_tensor.device)
    elif isinstance(policy, NativeCVRPNeuralPolicy):
        policy._ensure_instance(instance)
        device = policy._base_problems.device if policy._base_problems is not None else DEFAULT_TORCH_DEVICE
        native_forward_swap._load_neural_model(policy._spec, device)


def row(
    status: str,
    strategy_id: str,
    policy_id: str,
    problem: str,
    instance: Instance,
    *,
    initial: float | str = "",
    final: float | str = "",
    gap: float | str = "",
    elapsed: float | str = "",
    reason: str = "",
) -> dict[str, str]:
    del reason
    return {
        "status": status,
        "strategy_id": strategy_id,
        "policy_id": policy_id,
        "problem": problem,
        "instance": instance.name,
        "final_cost": f"{final:.6f}" if isinstance(final, float) else str(final),
        "gap": f"{gap:.6f}" if isinstance(gap, float) else str(gap),
        "time": f"{elapsed:.6f}" if isinstance(elapsed, float) else str(elapsed),
    }


def csv_items(value: str, all_values: Sequence[str]) -> list[str]:
    if value == "all":
        return list(all_values)
    return [item.strip() for item in value.split(",") if item.strip()]


def run_matrix(
    strategies: str,
    policies: str,
    problems: str,
    *,
    out_dir: Path,
    seed: int,
    size: str | None = None,
    stream: bool = False,
    iterations: int | None = None,
    min_new_edges: int = 24,
    refine_k: int = 64,
    refine: bool = True,
    neural_knn_k: int = 32,
    neural_backup_k: int = 32,
    neural_knn_mask: bool = True,
    STAR_samples: int = 64,
    STAR_memory: bool = True,
    STAR_memory_k: int = 32,
    STAR_memory_rho: float = 0.9,
    STAR_memory_tau_min: float | None = None,
    STAR_memory_tau_max: float = 1.0,
    STAR_memory_alpha: float = 1.0,
    STAR_start_mode: str = "random",
    STAR_start_probes: int = 32,
    STAR_start_cost_weight: float = 1.0,
    STAR_start_policy_weight: float = 1.0,
    STAR_start_memory_weight: float = 1.0,
    STAR_memory_update_mode: str = "auto",
    STAR_advantage_scale: float = 100.0,
    STAR_advantage_min: float = 0.0,
    STAR_trace: bool = False,
    STAR_profile: bool = False,
) -> list[dict[str, str]]:
    rows = []
    STAR_trace_rows: list[dict[str, str]] = []
    STAR_profile_rows: list[dict[str, str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "STAR_progress.csv"
    if stream:
        print_stream_header()
    with progress_path.open("w", encoding="utf-8", newline="") as progress_handle:
        progress_writer = csv.DictWriter(progress_handle, fieldnames=STAR_PROGRESS_FIELDS)
        progress_writer.writeheader()

        def write_STAR_progress(progress_row: dict[str, str]) -> None:
            progress_writer.writerow({field: progress_row.get(field, "") for field in STAR_PROGRESS_FIELDS})
            progress_handle.flush()

        for strategy_id in csv_items(strategies, search_strategy_ids()):
            for policy_id in csv_items(policies, append_policy_ids()):
                for problem in csv_items(problems, ["tsp", "cvrp"]):
                    for path in instance_paths(problem, size):
                        result = run_one(
                            strategy_id,
                            policy_id,
                            problem,
                            seed,
                            instance=load_instance_path(problem, path),
                            iterations=iterations,
                            min_new_edges=min_new_edges,
                            refine_k=refine_k,
                            refine=refine,
                            neural_knn_k=neural_knn_k,
                            neural_backup_k=neural_backup_k,
                            neural_knn_mask=neural_knn_mask,
                            STAR_samples=STAR_samples,
                            STAR_memory=STAR_memory,
                            STAR_memory_k=STAR_memory_k,
                            STAR_memory_rho=STAR_memory_rho,
                            STAR_memory_tau_min=STAR_memory_tau_min,
                            STAR_memory_tau_max=STAR_memory_tau_max,
                            STAR_memory_alpha=STAR_memory_alpha,
                            STAR_start_mode=STAR_start_mode,
                            STAR_start_probes=STAR_start_probes,
                            STAR_start_cost_weight=STAR_start_cost_weight,
                            STAR_start_policy_weight=STAR_start_policy_weight,
                            STAR_start_memory_weight=STAR_start_memory_weight,
                            STAR_memory_update_mode=STAR_memory_update_mode,
                            STAR_advantage_scale=STAR_advantage_scale,
                            STAR_advantage_min=STAR_advantage_min,
                            STAR_trace=STAR_trace,
                            STAR_profile=STAR_profile,
                            STAR_trace_rows=STAR_trace_rows,
                            STAR_profile_rows=STAR_profile_rows,
                            STAR_progress_callback=write_STAR_progress,
                        )
                        rows.append(result)
                        if stream:
                            print_stream_row(result)
    write_outputs(rows, out_dir, STAR_trace_rows, STAR_profile_rows)
    return rows


def write_outputs(
    rows: Sequence[dict[str, str]],
    out_dir: Path,
    STAR_trace_rows: Sequence[dict[str, str]] | None = None,
    STAR_profile_rows: Sequence[dict[str, str]] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "run_status.jsonl").open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    if STAR_trace_rows:
        trace_fields = STAR_trace_fields(STAR_trace_rows)
        with (out_dir / "STAR_trace.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=trace_fields)
            writer.writeheader()
            writer.writerows(STAR_trace_rows)
    if STAR_profile_rows:
        profile_fields = STAR_trace_fields(STAR_profile_rows)
        with (out_dir / "STAR_profile.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=profile_fields)
            writer.writeheader()
            writer.writerows(STAR_profile_rows)


def STAR_trace_fields(rows: Sequence[dict[str, str]]) -> list[str]:
    preferred = [
        "strategy_id",
        "policy_id",
        "problem",
        "instance",
        "iteration",
        "sample",
        "start_mode",
        "start_node",
        "successor",
        "start_score",
        "cost_score",
        "policy_score",
        "memory_score",
        "successor_prob",
        "best_alt_prob",
        "source_cost",
        "refined_cost",
        "advantage",
        "introduced_edges",
        "removed_edges",
        "changed_nodes",
        "memory_update_mode",
    ]
    extra = sorted({key for row_item in rows for key in row_item} - set(preferred))
    return preferred + extra


def print_table(rows: Iterable[dict[str, str]]) -> None:
    fields = ["status", "strategy_id", "policy_id", "problem", "instance", "final_cost", "gap", "time"]
    rows = [{field: row[field] for field in fields} for row in rows]
    widths = {field: max([len(field)] + [len(str(row[field])) for row in rows]) for field in fields}
    print("  ".join(field.ljust(widths[field]) for field in fields))
    print("  ".join("-" * widths[field] for field in fields))
    for item in rows:
        print("  ".join(str(item[field]).ljust(widths[field]) for field in fields))


STREAM_FIELDS = ["status", "strategy_id", "policy_id", "problem", "instance", "final_cost", "gap", "time"]


def print_stream_header() -> None:
    print("\t".join(STREAM_FIELDS), flush=True)


def print_stream_row(row: dict[str, str]) -> None:
    print("\t".join(row[field] for field in STREAM_FIELDS), flush=True)


def result_summary(rows: Sequence[dict[str, str]]) -> dict[str, str]:
    gaps = [float(row["gap"]) for row in rows if row.get("gap")]
    times = [float(row["time"]) for row in rows if row.get("time")]
    solved = sum(1 for row in rows if row["status"] == "ok")
    return {
        "avg_gap": f"{sum(gaps) / len(gaps):.6f}" if gaps else "",
        "avg_time": f"{sum(times) / len(times):.6f}" if times else "",
        "total_time": f"{sum(times):.6f}" if times else "",
        "total_solved": f"{solved}/{len(rows)}",
    }


def print_summary(rows: Sequence[dict[str, str]]) -> None:
    summary = result_summary(rows)
    print(
        "avg_gap  avg_time  total_time  solved",
        flush=True,
    )
    print(
        f"{summary['avg_gap'] or 'n/a'}  "
        f"{summary['avg_time'] or 'n/a'}  "
        f"{summary['total_time'] or 'n/a'}  "
        f"{summary['total_solved']}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run in-process search/policy experiments.")
    parser.add_argument("strategies", nargs="?", default="lehd_rrc")
    parser.add_argument("policies", nargs="?", default="nearest")
    parser.add_argument("problems", nargs="?", default="tsp")
    parser.add_argument("scope", nargs="?", default="smallest")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=None, help="accepted for CLI compatibility; unused by in-process runner")
    parser.add_argument(
        "--size",
        choices=["dev", "dev-medium3", "dev-medium", "small", "medium", "large", "full"],
        default=None,
        help=(
            "run benchmark instances in the selected size bucket; dev runs the first 15 smallest "
            "instances per requested problem; dev-medium3 runs the first 3 medium instances; "
            "dev-medium runs the first 5 medium instances; full runs every benchmark instance"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="search-loop budget override; defaults to STAR=100, LEHD RRC=1000, SIL PRC=1000",
    )
    parser.add_argument("--min-new-edges", type=int, default=24, help="STAR new-edge budget per iteration")
    parser.add_argument("--refine-k", type=int, default=64, help="STAR nearest-neighbor refinement candidate count")
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="disable STAR scoped refinement after perturbation for both TSP and CVRP",
    )
    parser.add_argument("--neural-knn-k", type=int, default=32, help="STAR TSP neural decode candidate-list size")
    parser.add_argument("--neural-backup-k", type=int, default=32, help="STAR TSP neural backup-list size used when kNN is exhausted")
    parser.add_argument(
        "--no-neural-knn-mask",
        action="store_true",
        help="do not mask LEHD decoder inputs to kNN candidates; kNN selection restriction still applies",
    )
    parser.add_argument(
        "--star-samples",
        "--STAR-samples",
        dest="STAR_samples",
        type=int,
        default=64,
        help="parallel perturbation samples per STAR iteration; currently implemented for STAR+lehd/sil+tsp",
    )
    parser.add_argument(
        "--star-memory",
        "--STAR-memory",
        dest="STAR_memory",
        action="store_true",
        default=True,
        help="Enable sparse Smooth-MMAS-style STAR edge memory and bias TSP neural/heuristic selection by it; enabled by default.",
    )
    parser.add_argument(
        "--no-star-memory",
        "--no-STAR-memory",
        dest="STAR_memory",
        action="store_false",
        help="Disable sparse Smooth-MMAS-style STAR edge memory.",
    )
    parser.add_argument(
        "--star-start-mode",
        "--STAR-start-mode",
        dest="STAR_start_mode",
        choices=["random", "cost", "policy-disagreement", "hybrid"],
        default="random",
        help="STAR perturbation start-node selection mode",
    )
    parser.add_argument("--star-start-probes", "--STAR-start-probes", dest="STAR_start_probes", type=int, default=32, help="candidate start nodes scored before perturbation")
    parser.add_argument("--star-start-cost-weight", "--STAR-start-cost-weight", dest="STAR_start_cost_weight", type=float, default=1.0, help="cost score weight for STAR hybrid start selection")
    parser.add_argument("--star-start-policy-weight", "--STAR-start-policy-weight", dest="STAR_start_policy_weight", type=float, default=1.0, help="policy-disagreement score weight for STAR hybrid start selection")
    parser.add_argument("--star-start-memory-weight", "--STAR-start-memory-weight", dest="STAR_start_memory_weight", type=float, default=1.0, help="memory score weight for STAR hybrid start selection")
    parser.add_argument("--star-memory-k", "--STAR-memory-k", dest="STAR_memory_k", type=int, default=32, help="number of nearest outgoing edges stored per node")
    parser.add_argument("--star-memory-rho", "--STAR-memory-rho", dest="STAR_memory_rho", type=float, default=0.9, help="memory interpolation rate")
    parser.add_argument("--star-memory-tau-min", "--STAR-memory-tau-min", dest="STAR_memory_tau_min", type=float, default=None, help="minimum memory target; default is 1/k")
    parser.add_argument("--star-memory-tau-max", "--STAR-memory-tau-max", dest="STAR_memory_tau_max", type=float, default=1.0, help="maximum memory target")
    parser.add_argument("--star-memory-alpha", "--STAR-memory-alpha", dest="STAR_memory_alpha", type=float, default=1.0, help="exponent applied to memory weights before selection")
    parser.add_argument(
        "--star-memory-update-mode",
        "--STAR-memory-update-mode",
        dest="STAR_memory_update_mode",
        choices=["auto", "source", "advantage-introduced", "source-advantage"],
        default="auto",
        help="STAR memory update rule; auto uses advantage-introduced for TSP and CVRP",
    )
    parser.add_argument("--star-advantage-scale", "--STAR-advantage-scale", dest="STAR_advantage_scale", type=float, default=100.0, help="scale applied to normalized improvement before advantage memory reinforcement")
    parser.add_argument("--star-advantage-min", "--STAR-advantage-min", dest="STAR_advantage_min", type=float, default=0.0, help="minimum normalized improvement needed for advantage memory reinforcement")
    parser.add_argument("--star-trace", "--STAR-trace", dest="STAR_trace", action="store_true", help="write STAR iteration/sample diagnostics to STAR_trace.csv")
    parser.add_argument("--star-profile", "--STAR-profile", dest="STAR_profile", action="store_true", help="write STAR per-iteration timing breakdown to STAR_profile.csv")
    return parser


def has_out_dir_arg(argv: Sequence[str]) -> bool:
    return any(arg == "--out-dir" or arg.startswith("--out-dir=") for arg in argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not has_out_dir_arg(raw_argv):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.out_dir = Path("results") / timestamp
    rows = run_matrix(
        args.strategies,
        args.policies,
        args.problems,
        out_dir=args.out_dir,
        seed=args.seed,
        size=args.size,
        stream=args.size is not None,
        iterations=args.iterations,
        min_new_edges=args.min_new_edges,
        refine_k=args.refine_k,
        refine=not args.no_refine,
        neural_knn_k=args.neural_knn_k,
        neural_backup_k=args.neural_backup_k,
        neural_knn_mask=not args.no_neural_knn_mask,
        STAR_samples=args.STAR_samples,
        STAR_memory=args.STAR_memory,
        STAR_memory_k=args.STAR_memory_k,
        STAR_memory_rho=args.STAR_memory_rho,
        STAR_memory_tau_min=args.STAR_memory_tau_min,
        STAR_memory_tau_max=args.STAR_memory_tau_max,
        STAR_memory_alpha=args.STAR_memory_alpha,
        STAR_start_mode=args.STAR_start_mode,
        STAR_start_probes=args.STAR_start_probes,
        STAR_start_cost_weight=args.STAR_start_cost_weight,
        STAR_start_policy_weight=args.STAR_start_policy_weight,
        STAR_start_memory_weight=args.STAR_start_memory_weight,
        STAR_memory_update_mode=args.STAR_memory_update_mode,
        STAR_advantage_scale=args.STAR_advantage_scale,
        STAR_advantage_min=args.STAR_advantage_min,
        STAR_trace=args.STAR_trace,
        STAR_profile=args.STAR_profile,
    )
    print(f"Wrote results to {args.out_dir}")
    if args.size is None:
        print_table(rows)
    print_summary(rows)
    return 1 if any(row["status"] == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
