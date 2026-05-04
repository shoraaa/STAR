"""Forward-pass replacement helpers for native LNS inference loops.

The native LEHD/SIL/DRHG testers call neural models inside repair loops.  These
helpers provide drop-in selected-node tensors for heuristic policies and can
load checkpoint-backed appending policies by setting ``NRS_FORWARD_OVERRIDE``.
"""

from __future__ import annotations

import importlib.util
import hashlib
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


_NN_ALIASES = {"nn", "nearest", "nearest_neighbor", "greedy"}
_SOFTDIST_ALIASES = {"softdist", "soft_dist", "soft-distance", "soft_distance"}
_BQ_KNNS = 250
_NEURAL_PREFIXES = ("neural:", "model:", "policy:")
_ANNOUNCED = False
_ROOT = Path(__file__).resolve().parents[1]
_MODEL_CACHE: Dict[Tuple[str, str], torch.nn.Module] = {}
_TOUR_CACHE: Dict[Tuple[str, str], torch.Tensor] = {}


@dataclass(frozen=True)
class NeuralPolicySpec:
    key: str
    family: str
    problem: str
    module_path: Path
    class_name: str
    checkpoint_path: Path
    model_params: Mapping[str, Any]
    checkpoint_key: str = "model_state_dict"
    import_roots: Tuple[Path, ...] = ()


_COMMON_PARAMS = {
    "mode": "test",
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** (1 / 2),
    "qkv_dim": 16,
    "head_num": 8,
    "ff_hidden_dim": 512,
}
_BQ_PARAMS = {
    "emb_size": 192,
    "dim_ff": 512,
    "activation_ff": "relu",
    "nb_layers_encoder": 9,
    "nb_heads": 12,
    "activation_attention": "softmax",
    "dropout": 0.0,
    "batchnorm": False,
}
_ICAM_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** (1 / 2),
    "encoder_layer_num": 12,
    "logit_clipping": 50,
    "ff_hidden_dim": 512,
    "eval_type": "greedy",
}
_ELG_TSP_PARAMS = {
    "ensemble": True,
    "distance_penalty": True,
    "positional": True,
    "ensemble_size": 1,
    "xi": -1,
    "local_size": [30],
    "euclidean": False,
    "embedding_dim": 128,
    "encoder_layer_num": 6,
    "head_num": 8,
    "qkv_dim": 16,
    "logit_clipping": 50,
    "ff_hidden_dim": 512,
    "local_att_hidden_dim": 32,
    "local_att_head_num": 4,
    "local_att_qkv_dim": 8,
}
_ELG_CVRP_PARAMS = {
    **_ELG_TSP_PARAMS,
    "local_size": [40],
    "demand": True,
}
_INVIT_COMMON_PARAMS = {
    "dim_input_nodes": 2,
    "dim_emb": 128,
    "dim_ff": 512,
    "num_state_encoder": 3,
    "nb_layers_state_encoder": 2,
    "nb_layers_action_encoder": 4,
    "nb_layers_decoder": 3,
    "nb_heads": 8,
    "batchnorm": False,
    "if_agg_whole_graph": False,
}
_DGL_APPEND_INFORMATION = [True, True, True, False, True, True, False, False, False, False, False]
_DGL_PARAMS = {
    "mode": "test",
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** (1 / 2),
    "decoder_layer_num": 3,
    "qkv_dim": 16,
    "head_num": 8,
    "ff_hidden_dim": 512,
    "append_information": _DGL_APPEND_INFORMATION,
}
_RELD_PARAMS = {
    "embedding_dim": 128,
    "encoder_layer_num": 6,
    "head_num": 8,
    "qkv_dim": 16,
    "forcing_first_step": False,
    "logit_clipping": 50,
    "ff_hidden_dim": 512,
    "eval_type": "greedy",
}

_NEURAL_POLICIES: Dict[str, NeuralPolicySpec] = {
    "bq_appending_tsp": NeuralPolicySpec(
        key="bq_appending_tsp",
        family="bq",
        problem="tsp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/1_BQ/model/model.py",
        class_name="BQModel",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/1_BQ/pretrained_models/tsp.best",
        model_params={**_BQ_PARAMS, "dim_input_nodes": 2, "problem": "tsp"},
        checkpoint_key="net",
        import_roots=(_ROOT / "NRS/Construction/single-stage/appending/1_BQ",),
    ),
    "bq_appending_cvrp": NeuralPolicySpec(
        key="bq_appending_cvrp",
        family="bq",
        problem="cvrp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/1_BQ/model/model.py",
        class_name="BQModel",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/1_BQ/pretrained_models/cvrp.best",
        model_params={**_BQ_PARAMS, "dim_input_nodes": 4, "problem": "cvrp"},
        checkpoint_key="net",
        import_roots=(_ROOT / "NRS/Construction/single-stage/appending/1_BQ",),
    ),
    "lehd_appending_tsp": NeuralPolicySpec(
        key="lehd_appending_tsp",
        family="lehd",
        problem="tsp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/2_LEHD/TSP/TSPModel.py",
        class_name="TSPModel",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/2_LEHD/TSP/result/20230509_153705_train/checkpoint-150.pt",
        model_params={**_COMMON_PARAMS, "decoder_layer_num": 6},
    ),
    "lehd_appending_cvrp": NeuralPolicySpec(
        key="lehd_appending_cvrp",
        family="lehd",
        problem="cvrp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/2_LEHD/CVRP/VRPModel.py",
        class_name="VRPModel",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/2_LEHD/CVRP/result/20230817_235537_train/checkpoint-40.pt",
        model_params={**_COMMON_PARAMS, "decoder_layer_num": 6},
    ),
    "sil_appending_tsp": NeuralPolicySpec(
        key="sil_appending_tsp",
        family="sil",
        problem="tsp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/3_SIL/TSP/Test_All/TSPModel.py",
        class_name="TSPModel",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/3_SIL/TSP/Test_All/result/checkpoint-tsp1k.pt",
        model_params={
            **_COMMON_PARAMS,
            "encoder_layer_num": 6,
            "logit_clipping": 10,
            "eval_type": "argmax",
            "use_k_nearest": True,
            "k_nearest_num": 1000,
        },
    ),
    "sil_appending_cvrp": NeuralPolicySpec(
        key="sil_appending_cvrp",
        family="sil",
        problem="cvrp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/3_SIL/CVRP/Test_All/VRPModel.py",
        class_name="VRPModel",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/3_SIL/CVRP/Test_All/result/checkpoint-cvrp1k.pt",
        model_params={
            **_COMMON_PARAMS,
            "decoder_layer_num": 6,
            "logit_clipping": 10,
            "eval_type": "argmax",
            "use_k_nearest": False,
            "k_nearest_num": 1000,
        },
    ),
    "drhg_appending_tsp": NeuralPolicySpec(
        key="drhg_appending_tsp",
        family="drhg",
        problem="tsp",
        module_path=_ROOT / "NRS/Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/3_DRHG/TSP/TSPModel_DRHG.py",
        class_name="TSPModel",
        checkpoint_path=_ROOT / "NRS/Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/3_DRHG/TSP/result/no_finetune/checkpoint-100.pt",
        model_params={
            **_COMMON_PARAMS,
            "encoder_layer_num": 6,
            "logit_clipping": 1,
            "eval_type": "argmax",
        },
    ),
    "drhg_appending_cvrp": NeuralPolicySpec(
        key="drhg_appending_cvrp",
        family="drhg",
        problem="cvrp",
        module_path=_ROOT / "NRS/Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/3_DRHG/CVRP/VRPModel.py",
        class_name="VRPModel",
        checkpoint_path=_ROOT / "NRS/Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/3_DRHG/CVRP/result/vrp_pretrained/checkpoint-100.pt",
        model_params={
            **_COMMON_PARAMS,
            "decoder_layer_num": 6,
            "logit_clipping": 10,
            "eval_type": "argmax",
        },
    ),
    "icam_appending_tsp": NeuralPolicySpec(
        key="icam_appending_tsp",
        family="icam",
        problem="tsp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/4_ICAM/ICAM_TSP/TSPModel_ICAM.py",
        class_name="TSPModel",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/4_ICAM/pretrained/icam_tsp.pt",
        model_params=_ICAM_PARAMS,
        import_roots=(
            _ROOT / "NRS/Construction/single-stage/appending/4_ICAM/ICAM_TSP",
            _ROOT / "NRS/Construction/single-stage/appending/4_ICAM",
        ),
    ),
    "icam_appending_cvrp": NeuralPolicySpec(
        key="icam_appending_cvrp",
        family="icam",
        problem="cvrp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/4_ICAM/ICAM_CVRP/CVRPModel_ICAM.py",
        class_name="CVRPModel",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/4_ICAM/pretrained/icam_cvrp.pt",
        model_params=_ICAM_PARAMS,
        import_roots=(
            _ROOT / "NRS/Construction/single-stage/appending/4_ICAM/ICAM_CVRP",
            _ROOT / "NRS/Construction/single-stage/appending/4_ICAM",
        ),
    ),
    "elg_appending_tsp": NeuralPolicySpec(
        key="elg_appending_tsp",
        family="elg",
        problem="tsp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/5_ELG/TSP/TSPModel.py",
        class_name="TSPModel",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/5_ELG/TSP/weights/ELG.pt",
        model_params=_ELG_TSP_PARAMS,
        import_roots=(_ROOT / "NRS/Construction/single-stage/appending/5_ELG/TSP",),
    ),
    "elg_appending_cvrp": NeuralPolicySpec(
        key="elg_appending_cvrp",
        family="elg",
        problem="cvrp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/5_ELG/CVRP/CVRPModel.py",
        class_name="CVRPModel",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/5_ELG/CVRP/weights/ELG.pt",
        model_params=_ELG_CVRP_PARAMS,
        import_roots=(_ROOT / "NRS/Construction/single-stage/appending/5_ELG/CVRP",),
    ),
    "invit_appending_tsp": NeuralPolicySpec(
        key="invit_appending_tsp",
        family="invit",
        problem="tsp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/6_INViT/TSP_net.py",
        class_name="TSP_net",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/6_INViT/ckpt/tsp/train/model/checkpoint_24-04-24--14-34-54-n100-gpu0.pkl",
        model_params=_INVIT_COMMON_PARAMS,
        checkpoint_key="model_baseline",
        import_roots=(_ROOT / "NRS/Construction/single-stage/appending/6_INViT",),
    ),
    "invit_appending_cvrp": NeuralPolicySpec(
        key="invit_appending_cvrp",
        family="invit",
        problem="cvrp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/6_INViT/VRP_net.py",
        class_name="VRP_net",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/6_INViT/ckpt/cvrp/train/model/checkpoint_24-04-18--14-14-20-n100-gpu0.pkl",
        model_params={**_INVIT_COMMON_PARAMS, "nb_layers_action_encoder": 2},
        checkpoint_key="model_baseline",
        import_roots=(_ROOT / "NRS/Construction/single-stage/appending/6_INViT",),
    ),
    "dgl_appending_tsp": NeuralPolicySpec(
        key="dgl_appending_tsp",
        family="dgl",
        problem="tsp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/8_DGL/TSP/TSPModel.py",
        class_name="TSPModel",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/8_DGL/TSP/pretrain/checkpoint-100.pt",
        model_params=_DGL_PARAMS,
        import_roots=(_ROOT / "NRS/Construction/single-stage/appending/8_DGL/TSP",),
    ),
    "dgl_appending_cvrp": NeuralPolicySpec(
        key="dgl_appending_cvrp",
        family="dgl",
        problem="cvrp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/8_DGL/CVRP/CVRPModel.py",
        class_name="CVRPModel",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/8_DGL/CVRP/pretrain/checkpoint-100.pt",
        model_params=_DGL_PARAMS,
        import_roots=(_ROOT / "NRS/Construction/single-stage/appending/8_DGL/CVRP",),
    ),
    "reld_appending_cvrp": NeuralPolicySpec(
        key="reld_appending_cvrp",
        family="reld",
        problem="cvrp",
        module_path=_ROOT / "NRS/Construction/single-stage/appending/9_ReLD/CVRP/CVRPModel.py",
        class_name="CVRPModel",
        checkpoint_path=_ROOT / "NRS/Construction/single-stage/appending/9_ReLD/CVRP/weights/ReLD/model_epoch_90.pt",
        model_params=_RELD_PARAMS,
        import_roots=(_ROOT / "NRS/Construction/single-stage/appending/9_ReLD/CVRP",),
    ),
}


def mode() -> Optional[str]:
    value = os.environ.get("NRS_FORWARD_OVERRIDE", "").strip().lower()
    return value or None


def is_enabled() -> bool:
    current = mode()
    return current in _NN_ALIASES or current in _SOFTDIST_ALIASES or _neural_name(current) is not None


def label() -> str:
    current = mode()
    if current in _SOFTDIST_ALIASES:
        return "softdist"
    if current in _NN_ALIASES:
        return "nearest_neighbor"
    neural = _neural_name(current)
    if neural:
        return f"neural:{neural}"
    return current or "model"


def tsp_forward_or_model(model: Any, state: Any, selected_node_list: torch.Tensor,
                         solution: torch.Tensor, current_step: int,
                         **kwargs: Any) -> Tuple[torch.Tensor, None, None, torch.Tensor]:
    current = mode()
    neural = _neural_policy(current, "tsp")
    if neural is not None:
        _announce_once(neural)
        return _call_tsp_neural(neural, state, selected_node_list, solution, current_step, **kwargs)

    if current not in _NN_ALIASES and current not in _SOFTDIST_ALIASES:
        return model(state, selected_node_list, solution, current_step, **kwargs)

    _announce_once()
    selected = _select_next_node(_coords_from_state(state), selected_node_list, current)
    return selected, None, None, selected


def cvrp_forward_or_model(model: Any, state: Any, selected_node_list: torch.Tensor,
                          solution: torch.Tensor, current_step: int,
                          **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current = mode()
    neural = _neural_policy(current, "cvrp")
    if neural is not None:
        _announce_once(neural)
        return _call_cvrp_neural(neural, state, selected_node_list, solution, current_step, **kwargs)

    if current not in _NN_ALIASES and current not in _SOFTDIST_ALIASES:
        return model(state, selected_node_list, solution, current_step, **kwargs)

    _announce_once()
    problems = _problems_from_state(state)
    selected = _select_next_node(problems[:, :, :2], selected_node_list, current, start_index=1)

    batch = selected.size(0)
    device = selected.device
    demand = _gather_feature(problems, selected, feature=2)
    remaining = problems[:, 0, 3].to(device=device)
    selected_flag = (demand > remaining + 1e-9).to(dtype=torch.int, device=device)
    loss_node = torch.zeros(batch, dtype=problems.dtype, device=device)
    return loss_node, selected, selected, selected_flag, selected_flag.clone()


def _coords_from_state(state: Any) -> torch.Tensor:
    if hasattr(state, "data"):
        return state.data
    if hasattr(state, "problems"):
        return state.problems[:, :, :2]
    raise AttributeError("state must expose data or problems for forward override")


def _problems_from_state(state: Any) -> torch.Tensor:
    if hasattr(state, "problems"):
        return state.problems
    if hasattr(state, "data"):
        return state.data
    raise AttributeError("state must expose problems or data for forward override")


def _select_next_node(coords: torch.Tensor, selected_node_list: torch.Tensor, policy: str,
                      start_index: int = 0) -> torch.Tensor:
    if coords.dim() != 3:
        raise ValueError(f"expected coords [batch, nodes, xy], got shape {tuple(coords.shape)}")

    batch, nodes, _ = coords.shape
    device = coords.device
    selected = selected_node_list.to(device=device, dtype=torch.long)
    if selected.numel() == 0 or selected.size(1) == 0:
        last = torch.full((batch,), start_index, dtype=torch.long, device=device)
    else:
        last = selected[:, -1].clamp(min=0, max=nodes - 1)

    visited = torch.zeros(batch, nodes, dtype=torch.bool, device=device)
    if selected.numel() > 0:
        visited.scatter_(1, selected.clamp(min=0, max=nodes - 1), True)
    if start_index > 0:
        visited[:, :start_index] = True

    has_candidate = (~visited).any(dim=1)
    if not bool(has_candidate.all()):
        fallback = torch.arange(nodes, device=device).expand(batch, nodes)
        fallback = fallback.masked_fill(visited, nodes)
        return fallback.min(dim=1).values.clamp(max=nodes - 1).to(dtype=torch.long, device=device)

    softdist_coords = _normalize_coords(coords) if policy in _SOFTDIST_ALIASES else coords
    current_xy = softdist_coords.gather(1, last[:, None, None].expand(batch, 1, softdist_coords.size(-1))).squeeze(1)
    distances = torch.linalg.vector_norm(softdist_coords - current_xy[:, None, :], dim=-1)
    distances = distances.masked_fill(visited, float("inf"))

    if policy in _SOFTDIST_ALIASES:
        temperature = float(os.environ.get("NRS_SOFTDIST_TEMPERATURE", "0.0051"))
        temperature = max(temperature, 1e-9)
        k_nearest = max(1, int(os.environ.get("NRS_SOFTDIST_KNN", "50")))
        finite = torch.isfinite(distances)
        if distances.size(1) > k_nearest:
            masked_distances = distances.masked_fill(~finite, float("inf"))
            topk = torch.topk(masked_distances, k=min(k_nearest, distances.size(1)), dim=1, largest=False).indices
            keep = torch.zeros_like(finite)
            keep.scatter_(1, topk, True)
            distances = distances.masked_fill(~keep, float("inf"))
        logits = -distances / temperature
        logits = logits.masked_fill(torch.isinf(distances), -1e9)
        probs = torch.softmax(logits, dim=1)
        sampled = torch.multinomial(probs, 1).squeeze(1)
        return sampled.to(dtype=torch.long, device=device)

    nearest = distances.argmin(dim=1)
    return nearest.to(dtype=torch.long, device=device)


def _normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    mins = coords.amin(dim=1, keepdim=True)
    spans = coords.amax(dim=1, keepdim=True) - mins
    scale = spans.amax(dim=2, keepdim=True).clamp_min(1e-12)
    return (coords - mins) / scale


def _gather_feature(problems: torch.Tensor, selected: torch.Tensor, feature: int) -> torch.Tensor:
    gather_index = selected.to(device=problems.device, dtype=torch.long)[:, None, None]
    gather_index = gather_index.expand(selected.size(0), 1, problems.size(-1))
    return problems.gather(1, gather_index).squeeze(1)[:, feature].to(device=selected.device)


def _neural_name(current: Optional[str]) -> Optional[str]:
    if not current:
        return None
    if current in {"bq", "lehd", "sil", "drhg", "icam", "elg", "invit", "dgl", "reld"}:
        return current
    for prefix in _NEURAL_PREFIXES:
        if current.startswith(prefix):
            return current[len(prefix):].strip()
    if current in _NEURAL_POLICIES:
        return current
    return None


def _neural_policy(current: Optional[str], problem: str) -> Optional[NeuralPolicySpec]:
    name = _neural_name(current)
    if not name:
        return None
    if name in {"bq", "lehd", "sil", "drhg", "icam", "elg", "invit", "dgl", "reld"}:
        name = f"{name}_appending_{problem}"
    elif name.endswith("_appending"):
        name = f"{name}_{problem}"
    spec = _NEURAL_POLICIES.get(name)
    if spec is None:
        raise ValueError(f"unknown neural forward override policy: {current!r}")
    if spec.problem != problem:
        raise ValueError(f"policy {spec.key} cannot be used for {problem}")
    return spec


def _call_tsp_neural(spec: NeuralPolicySpec, state: Any, selected_node_list: torch.Tensor,
                     solution: torch.Tensor, current_step: int, **kwargs: Any) -> Tuple[torch.Tensor, Any, Any, torch.Tensor]:
    original_device = _coords_from_state(state).device
    target_device = _forward_device(original_device)
    if spec.family == "invit":
        target_device = torch.device("cpu")
    neural_model = _load_neural_model(spec, target_device)
    if spec.family == "bq":
        selected = _call_bq_tsp(neural_model, _state_to_device(state, target_device), selected_node_list.to(device=target_device))
        selected = selected.to(device=original_device)
        return selected, None, None, selected
    if spec.family == "dgl":
        selected = _call_dgl_tsp(neural_model, _state_to_device(state, target_device), selected_node_list.to(device=target_device), solution, current_step)
        selected = selected.to(device=original_device)
        return selected, None, None, selected
    state_on_device = _state_to_device(state, target_device)
    selected = selected_node_list.to(device=target_device, dtype=torch.long)
    solution_on_device = solution.to(device=target_device) if hasattr(solution, "to") else solution
    with _cpu_default_tensor_type(target_device, original_device):
        if spec.family == "icam":
            selected_node = _call_icam_tsp(neural_model, state_on_device, selected)
            selected_node = selected_node.to(device=original_device)
            return selected_node, None, None, selected_node
        if spec.family == "elg":
            selected_node = _call_elg_tsp(neural_model, state_on_device, selected)
            selected_node = selected_node.to(device=original_device)
            return selected_node, None, None, selected_node
        if spec.family == "invit":
            selected_node = _call_invit_tsp(neural_model, state_on_device, selected)
            selected_node = selected_node.to(device=original_device)
            return selected_node, None, None, selected_node
        if spec.family == "sil":
            return _to_device(neural_model(
                state_on_device,
                selected,
                solution_on_device,
                current_step,
                decode_method=kwargs.get("decode_method", "greedy"),
                repair=kwargs.get("repair", False),
            ), original_device)
        if spec.family == "drhg":
            point_couples, endpoint_mask = _drhg_tsp_context(
                state_on_device,
                _tensor_to_device(kwargs.get("point_couples"), target_device),
                _tensor_to_device(kwargs.get("endpoint_mask"), target_device),
            )
            model_step = 1 if current_step <= 2 else current_step
            return _to_device(
                neural_model(
                    state_on_device,
                    selected,
                    solution_on_device,
                    model_step,
                    point_couples=point_couples,
                    endpoint_mask=endpoint_mask,
                ),
                original_device,
            )
        return _to_device(
            neural_model(state_on_device, selected, solution_on_device, current_step, repair=kwargs.get("repair", False)),
            original_device,
        )


def _call_cvrp_neural(spec: NeuralPolicySpec, state: Any, selected_node_list: torch.Tensor,
                      solution: torch.Tensor, current_step: int, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    problems = _problems_from_state(state)
    original_device = problems.device
    target_device = _forward_device(original_device)
    if spec.family == "invit":
        target_device = torch.device("cpu")
    neural_model = _load_neural_model(spec, target_device)
    state_on_device = _state_to_device(state, target_device)
    raw_capacity = kwargs.get("raw_data_capacity")
    if raw_capacity is None:
        raw_capacity = _raw_capacity_from_state(state_on_device)
    raw_capacity = raw_capacity.to(device=target_device) if hasattr(raw_capacity, "to") else torch.tensor([raw_capacity], device=target_device)
    if spec.family == "bq":
        result = _call_bq_cvrp(
            neural_model,
            state_on_device,
            selected_node_list.to(device=target_device),
            raw_capacity,
        )
        return _capacity_safe_cvrp_result(problems, _to_device(result, original_device))
    if spec.family == "dgl":
        result = _call_dgl_cvrp(neural_model, state_on_device, selected_node_list.to(device=target_device), solution, current_step)
        return _capacity_safe_cvrp_result(problems, _to_device(result, original_device))
    selected = selected_node_list.to(device=target_device, dtype=torch.long)
    solution_on_device = solution.to(device=target_device) if hasattr(solution, "to") else solution

    with _cpu_default_tensor_type(target_device, original_device):
        if spec.family == "icam":
            model_state = SimpleNamespace(problems=_unit_capacity_cvrp_problems(_problems_from_state(state_on_device), raw_capacity))
            result = _call_explicit_depot_cvrp(neural_model, model_state, selected, family="icam")
            return _capacity_safe_cvrp_result(problems, _to_device(result, original_device))
        if spec.family == "elg":
            model_state = SimpleNamespace(problems=_unit_capacity_cvrp_problems(_problems_from_state(state_on_device), raw_capacity))
            result = _call_explicit_depot_cvrp(neural_model, model_state, selected, family="elg")
            return _capacity_safe_cvrp_result(problems, _to_device(result, original_device))
        if spec.family == "invit":
            result = _call_invit_cvrp(neural_model, state_on_device, selected)
            return _capacity_safe_cvrp_result(problems, _to_device(result, original_device))
        if spec.family == "reld":
            result = _call_reld_cvrp(neural_model, state_on_device, selected)
            return _capacity_safe_cvrp_result(problems, _to_device(result, original_device))
        if spec.family == "sil":
            result = _to_device(neural_model(
                state_on_device,
                selected,
                solution_on_device,
                current_step,
                raw_data_capacity=raw_capacity,
                decode_method=kwargs.get("decode_method", "greedy"),
            ), original_device)
            return _capacity_safe_cvrp_result(problems, result)
        if spec.family == "drhg":
            point_couples, endpoint_mask = _drhg_cvrp_context(
                state_on_device,
                _tensor_to_device(kwargs.get("point_couples"), target_device),
                _tensor_to_device(kwargs.get("endpoint_mask"), target_device),
            )
            result = _to_device(neural_model(
                state_on_device,
                selected,
                solution_on_device,
                current_step,
                raw_data_capacity=raw_capacity,
                point_couples=point_couples,
                endpoint_mask=endpoint_mask,
            ), original_device)
            return _capacity_safe_cvrp_result(problems, result)
        result = _to_device(neural_model(
            state_on_device,
            selected,
            solution_on_device,
            current_step,
            raw_data_capacity=raw_capacity,
        ), original_device)
        return _capacity_safe_cvrp_result(problems, result)


def _call_bq_tsp(neural_model: torch.nn.Module, state: Any, selected_node_list: torch.Tensor) -> torch.Tensor:
    coords = _coords_from_state(state)
    selected = selected_node_list.to(device=coords.device, dtype=torch.long)
    batch, nodes, _ = coords.shape
    if selected.numel() == 0 or selected.size(1) == 0:
        selected = torch.zeros(batch, 1, dtype=torch.long, device=coords.device)
    visited = _visited_mask(nodes, selected, 0)
    raw_coords = getattr(state, "raw_coords", coords)
    raw_dist_matrix = getattr(state, "raw_dist_matrix", None)
    edge_weight_type = getattr(state, "edge_weight_type", "EUC_2D")
    local_inputs, local_to_global = _bq_local_tsp_inputs(
        coords,
        selected,
        visited,
        raw_coords,
        edge_weight_type,
        raw_dist_matrix,
    )
    scores = neural_model(local_inputs)
    local_pick = scores.argmax(dim=1)
    picked = local_to_global.gather(1, local_pick[:, None]).squeeze(1)
    return _valid_or_raise(coords, selected, picked.to(dtype=torch.long, device=coords.device), start_index=0, policy_key="bq_appending_tsp")


def _call_bq_cvrp(
    neural_model: torch.nn.Module,
    state: Any,
    selected_node_list: torch.Tensor,
    raw_capacity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    problems = _problems_from_state(state)
    coords = problems[:, :, :2]
    selected = selected_node_list.to(device=problems.device, dtype=torch.long)
    batch, nodes, _ = problems.shape
    if selected.numel() == 0 or selected.size(1) == 0:
        selected = torch.zeros(batch, 1, dtype=torch.long, device=problems.device)
    visited = _visited_mask(nodes, selected, 1)
    local_inputs, local_to_global, local_demands = _bq_local_cvrp_inputs(problems, selected, visited, raw_capacity)
    remaining = problems[:, 0, 3].to(device=problems.device, dtype=problems.dtype)
    scores = neural_model(
        local_inputs,
        demands=local_demands,
        remaining_capacities=remaining,
    )
    local_flat = scores.argmax(dim=1)
    via_depot = (local_flat % 2).to(dtype=torch.int, device=problems.device)
    local_pick = torch.div(local_flat, 2, rounding_mode="floor")
    picked = local_to_global.gather(1, local_pick[:, None]).squeeze(1).to(dtype=torch.long)
    picked = _valid_cvrp_or_raise(problems, selected, picked, policy_key="bq_appending_cvrp")
    loss_node = torch.zeros(batch, dtype=problems.dtype, device=problems.device)
    return loss_node, picked, picked, via_depot, via_depot.clone()


def _call_dgl_tsp(
    neural_model: torch.nn.Module,
    state: Any,
    selected_node_list: torch.Tensor,
    solution: torch.Tensor,
    current_step: int,
) -> torch.Tensor:
    coords = _coords_from_state(state)
    data = _dgl_tsp_data(coords, selected_node_list)
    dgl_state = SimpleNamespace(data=data)
    neural_model.pre_forward(_pairwise_distances(coords), coords.size(0))
    _teacher, probs, _ = neural_model(
        dgl_state,
        selected_node_list.to(device=coords.device, dtype=torch.long),
        solution.to(device=coords.device) if hasattr(solution, "to") else solution,
        min(coords.size(1), 50),
        current_step,
    )
    picked = probs.argmax(dim=1).to(dtype=torch.long, device=coords.device)
    return _valid_or_raise(coords[:, :, :2], selected_node_list.to(device=coords.device), picked, start_index=0, policy_key="dgl_appending_tsp")


def _call_dgl_cvrp(
    neural_model: torch.nn.Module,
    state: Any,
    selected_node_list: torch.Tensor,
    solution: torch.Tensor,
    current_step: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    problems = _problems_from_state(state)
    selected = selected_node_list.to(device=problems.device, dtype=torch.long)
    if selected.size(1) > 1 and bool((selected[:, 0] == 0).all()):
        selected = selected[:, 1:]
    if selected.size(1) == 0:
        raise ValueError("dgl_appending_cvrp cannot run without a neural-selected prefix; refusing nearest-neighbor bootstrap")
    data = _dgl_cvrp_data(problems, selected)
    remaining = problems[:, 0, 3].to(device=problems.device, dtype=problems.dtype)
    demand = problems[:, :, 2].to(device=problems.device, dtype=problems.dtype)
    ninf_mask = torch.zeros(problems.size(0), problems.size(1), dtype=problems.dtype, device=problems.device)
    ninf_mask[demand > remaining[:, None] + 1e-9] = float("-inf")
    ninf_mask[:, 0] = float("-inf")
    dgl_state = SimpleNamespace(
        data=data,
        capacity=remaining,
        ninf_mask=ninf_mask,
        distance_to_depot=torch.linalg.vector_norm(problems[:, :, :2] - problems[:, :1, :2], dim=-1),
    )
    neural_model.pre_forward(_pairwise_distances(problems[:, :, :2]), problems.size(0))
    dummy_flags = torch.zeros_like(solution, dtype=torch.long, device=problems.device) if hasattr(solution, "to") else solution
    _teacher, probs, _ = neural_model(
        dgl_state,
        selected,
        solution.to(device=problems.device) if hasattr(solution, "to") else solution,
        dummy_flags,
        min(problems.size(1), 10),
        current_step,
        min(problems.size(1), 10),
        selected_flag=dummy_flags,
    )
    flat = probs.argmax(dim=1)
    nodes = problems.size(1)
    via_depot = (flat >= nodes).to(dtype=torch.int, device=problems.device)
    picked = (flat % nodes).to(dtype=torch.long, device=problems.device)
    picked = _valid_or_raise(problems[:, :, :2], selected_node_list.to(device=problems.device), picked, start_index=1, policy_key="dgl_appending_cvrp")
    loss_node = torch.zeros(problems.size(0), dtype=problems.dtype, device=problems.device)
    return loss_node, picked, picked, via_depot, via_depot.clone()


def _call_icam_tsp(neural_model: torch.nn.Module, state: Any, selected_node_list: torch.Tensor) -> torch.Tensor:
    coords = _coords_from_state(state)
    selected = _selected_2d(selected_node_list, coords.device)
    dist = _pairwise_distances(coords)
    reset_state = SimpleNamespace(
        problems=coords,
        dist=dist,
        log_scale=math.log2(max(2, coords.size(1))),
    )
    neural_model.pre_forward(reset_state)
    _set_tsp_decoder_first_query(neural_model, selected)
    step_state = _tsp_step_state(coords, selected)
    cur_dist, _cur_theta, _relative_xy = _tsp_local_features(coords, selected)
    picked, _prob = neural_model(step_state, cur_dist)
    return _valid_or_raise(coords, selected, picked.squeeze(1), start_index=0, policy_key="icam_appending_tsp")


def _call_elg_tsp(neural_model: torch.nn.Module, state: Any, selected_node_list: torch.Tensor) -> torch.Tensor:
    coords = _coords_from_state(state)
    selected = _selected_2d(selected_node_list, coords.device)
    neural_model.pre_forward(SimpleNamespace(problems=coords))
    _set_tsp_decoder_first_query(neural_model, selected)
    step_state = _tsp_step_state(coords, selected)
    cur_dist, cur_theta, relative_xy = _tsp_local_features(coords, selected)
    picked, _prob = neural_model.one_step_rollout(
        step_state,
        cur_dist=cur_dist,
        cur_theta=cur_theta,
        xy=relative_xy,
        eval_type="greedy",
    )
    return _valid_or_raise(coords, selected, picked.squeeze(1), start_index=0, policy_key="elg_appending_tsp")


def _call_invit_tsp(neural_model: torch.nn.Module, state: Any, selected_node_list: torch.Tensor) -> torch.Tensor:
    coords = _coords_from_state(state)
    selected = _selected_2d(selected_node_list, coords.device)
    cache_key = ("invit_tsp", _tensor_cache_key(coords))
    tours = _TOUR_CACHE.get(cache_key)
    if tours is None:
        action_k = min(10, coords.size(1))
        state_k = [min(50, coords.size(1))] * int(getattr(neural_model, "num_state_encoder", 0))
        tours, _log_probs = neural_model(
            coords,
            action_k,
            state_k,
            choice_deterministic=True,
            if_use_local_mask=False,
            if_aug=False,
        )
        _TOUR_CACHE[cache_key] = tours.detach().to(device=coords.device)
    return _first_unvisited_from_tour(coords, selected, tours.to(device=coords.device), start_index=0)


def _call_icam_cvrp(
    neural_model: torch.nn.Module,
    state: Any,
    selected_node_list: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    problems = _problems_from_state(state)
    selected = _selected_2d(selected_node_list, problems.device)
    reset_state = _cvrp_reset_state(problems, include_log_scale=True)
    neural_model.pre_forward(reset_state)
    step_state = _cvrp_step_state(problems, selected)
    cur_dist, _cur_theta, _relative_xy, _norm_demand = _cvrp_local_features(problems, selected)
    picked, _prob = neural_model(step_state, cur_dist)
    picked = _valid_cvrp_or_raise(problems, selected, picked.squeeze(1), policy_key="icam_appending_cvrp")
    return _cvrp_policy_result(problems, picked)


def _call_elg_cvrp(
    neural_model: torch.nn.Module,
    state: Any,
    selected_node_list: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    problems = _problems_from_state(state)
    selected = _selected_2d(selected_node_list, problems.device)
    neural_model.pre_forward(_cvrp_reset_state(problems, include_log_scale=False))
    step_state = _cvrp_step_state(problems, selected)
    cur_dist, cur_theta, relative_xy, norm_demand = _cvrp_local_features(problems, selected)
    picked, _prob = neural_model.one_step_rollout(
        step_state,
        cur_dist,
        cur_theta,
        relative_xy,
        norm_demand=norm_demand,
        eval_type="greedy",
    )
    picked = _valid_cvrp_or_raise(problems, selected, picked.squeeze(1), policy_key="elg_appending_cvrp")
    return _cvrp_policy_result(problems, picked)


def _call_explicit_depot_cvrp(
    neural_model: torch.nn.Module,
    state: Any,
    selected_node_list: torch.Tensor,
    *,
    family: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    problems = _problems_from_state(state)
    selected = _explicit_depot_prefix(problems, _selected_2d(selected_node_list, problems.device))
    neural_model.pre_forward(_cvrp_reset_state(problems, include_log_scale=(family == "icam")))
    route_break = torch.zeros(problems.size(0), dtype=torch.int, device=problems.device)
    policy_key = f"{family}_appending_cvrp"

    for _ in range(problems.size(1) + 2):
        step_state = _cvrp_step_state(problems, selected)
        if family == "icam":
            cur_dist = _icam_cvrp_local_feature(problems, selected)
            picked, _prob = neural_model(step_state, cur_dist)
        elif family == "elg":
            cur_dist, cur_theta, relative_xy, norm_demand = _cvrp_local_features(problems, selected)
            picked, _prob = neural_model.one_step_rollout(
                step_state,
                cur_dist,
                cur_theta,
                relative_xy,
                norm_demand=norm_demand,
                eval_type="greedy",
            )
        else:
            raise ValueError(f"unsupported explicit-depot CVRP family: {family}")

        picked = _valid_explicit_cvrp_or_raise(problems, selected, picked.squeeze(1), policy_key=policy_key)
        depot_rows = picked == 0
        if not bool(depot_rows.any()):
            return _cvrp_policy_result_with_flag(problems, picked, route_break)

        route_break = route_break | depot_rows.to(dtype=torch.int)
        selected = torch.cat((selected, picked[:, None]), dim=1)
        problems = problems.clone()
        problems[depot_rows, :, 3] = 1.0

    raise ValueError(f"{policy_key} emitted only depot actions before selecting a customer")


def _call_invit_cvrp(
    neural_model: torch.nn.Module,
    state: Any,
    selected_node_list: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    problems = _problems_from_state(state)
    selected = _selected_2d(selected_node_list, problems.device)
    cache_key = ("invit_cvrp", _tensor_cache_key(problems[:, :, :3]))
    tours = _TOUR_CACHE.get(cache_key)
    if tours is None:
        demand = problems[:, 1:, 2].round().clamp_min(0).to(dtype=torch.long)
        capacity = _raw_capacity_from_state(state).round().clamp_min(1).to(dtype=torch.long)
        action_k = min(10, max(1, problems.size(1) - 1))
        state_k = [min(50, max(1, problems.size(1) - 1))] * int(getattr(neural_model, "num_state_encoder", 0))
        tours, _log_probs = neural_model(
            {
                "loc": problems[:, 1:, :2],
                "demand": demand,
                "depot": problems[:, 0, :2],
            },
            action_k,
            state_k,
            capacity,
            problem="cvrp",
            choice_deterministic=True,
            if_use_local_mask=False,
        )
        _TOUR_CACHE[cache_key] = tours.detach().to(device=problems.device)
    # INViT uses -1 for depot and 0-based customer ids. Convert to this shim's
    # global CVRP indexing: depot=0, customers=1..N.
    global_tour = torch.where(tours < 0, torch.zeros_like(tours), tours + 1).to(device=problems.device)
    picked = _first_cvrp_from_tour(problems, selected, global_tour)
    return _cvrp_policy_result(problems, picked)


def _call_reld_cvrp(
    neural_model: torch.nn.Module,
    state: Any,
    selected_node_list: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    problems = _problems_from_state(state)
    selected = _selected_2d(selected_node_list, problems.device)
    neural_model.pre_forward(_cvrp_reset_state(problems, include_log_scale=False))
    step_state = _cvrp_step_state(problems, selected)
    cur_dist, _cur_theta, _relative_xy, _norm_demand = _cvrp_local_features(problems, selected)
    picked, _prob = neural_model.one_step_rollout(step_state, cur_dist, eval_type="greedy")
    picked = _valid_cvrp_or_raise(problems, selected, picked.squeeze(1), policy_key="reld_appending_cvrp")
    return _cvrp_policy_result(problems, picked)


def _selected_2d(selected_node_list: torch.Tensor, device: torch.device) -> torch.Tensor:
    selected = selected_node_list.to(device=device, dtype=torch.long)
    if selected.dim() == 1:
        selected = selected[:, None]
    if selected.dim() > 2:
        selected = selected.reshape(selected.size(0), -1)
    return selected


def _tensor_cache_key(value: torch.Tensor) -> str:
    cpu_value = value.detach().to(device="cpu").contiguous()
    digest = hashlib.blake2b(cpu_value.numpy().tobytes(), digest_size=16)
    return f"{tuple(cpu_value.shape)}:{cpu_value.dtype}:{digest.hexdigest()}"


def _set_tsp_decoder_first_query(neural_model: torch.nn.Module, selected: torch.Tensor) -> None:
    if selected.numel() == 0 or selected.size(1) == 0:
        return
    decoder = getattr(neural_model, "decoder", None)
    set_q1 = getattr(decoder, "set_q1", None)
    encoded = getattr(neural_model, "encoded_nodes", None)
    if not callable(set_q1) or encoded is None:
        return
    first = selected[:, :1].clamp(min=0, max=encoded.size(1) - 1)
    encoded_first = encoded.gather(1, first[:, :, None].expand(first.size(0), 1, encoded.size(2)))
    set_q1(encoded_first)


def _tsp_step_state(coords: torch.Tensor, selected: torch.Tensor) -> SimpleNamespace:
    batch = coords.size(0)
    current_node = selected[:, -1:] if selected.numel() and selected.size(1) else None
    return SimpleNamespace(
        batch_size=batch,
        pomo_size=1,
        BATCH_IDX=torch.arange(batch, device=coords.device)[:, None],
        POMO_IDX=torch.zeros(batch, 1, dtype=torch.long, device=coords.device),
        current_node=current_node,
        ninf_mask=_visited_mask(coords.size(1), selected, 0)[:, None, :].to(dtype=coords.dtype).masked_fill(
            _visited_mask(coords.size(1), selected, 0)[:, None, :],
            float("-inf"),
        ),
    )


def _tsp_local_features(coords: torch.Tensor, selected: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, nodes, _ = coords.shape
    dist = _pairwise_distances(coords)
    if selected.numel() == 0 or selected.size(1) == 0:
        current = torch.zeros(batch, 1, dtype=torch.long, device=coords.device)
    else:
        current = selected[:, -1:].clamp(min=0, max=nodes - 1)
    cur_dist = torch.take_along_dim(
        dist[:, None, :, :].expand(batch, 1, nodes, nodes),
        current[:, :, None, None].expand(batch, 1, 1, nodes),
        dim=2,
    ).squeeze(2)
    expanded_xy = coords[:, None, :, :].expand(batch, 1, nodes, 2)
    current_xy = torch.take_along_dim(
        expanded_xy,
        current[:, :, None, None].expand(batch, 1, 1, 2),
        dim=2,
    )
    relative_xy = expanded_xy - current_xy
    cur_theta = torch.atan2(relative_xy[:, :, :, 1], relative_xy[:, :, :, 0])
    return cur_dist, cur_theta, relative_xy


def _cvrp_reset_state(problems: torch.Tensor, include_log_scale: bool) -> SimpleNamespace:
    reset_state = SimpleNamespace(
        depot_xy=problems[:, :1, :2],
        node_xy=problems[:, 1:, :2],
        node_demand=problems[:, 1:, 2],
        dist=_pairwise_distances(problems[:, :, :2]),
    )
    if include_log_scale:
        reset_state.log_scale = math.log2(max(2, problems.size(1) - 1))
    return reset_state


def _unit_capacity_cvrp_problems(problems: torch.Tensor, raw_capacity: torch.Tensor) -> torch.Tensor:
    capacity = _capacity_vector(raw_capacity, problems)
    normalized = problems.clone()
    normalized[:, :, 2] = normalized[:, :, 2] / capacity[:, None]
    normalized[:, :, 3] = normalized[:, :, 3] / capacity[:, None]
    return normalized


def _capacity_vector(raw_capacity: torch.Tensor, problems: torch.Tensor) -> torch.Tensor:
    capacity = raw_capacity.to(device=problems.device, dtype=problems.dtype)
    while capacity.dim() > 1:
        capacity = capacity[:, 0]
    capacity = capacity.reshape(-1).clamp_min(1e-12)
    if capacity.size(0) == 1 and problems.size(0) > 1:
        capacity = capacity.expand(problems.size(0))
    if capacity.size(0) != problems.size(0):
        raise ValueError(
            f"raw CVRP capacity batch mismatch: capacity rows {capacity.size(0)}, problem rows {problems.size(0)}"
        )
    return capacity


def _explicit_depot_prefix(problems: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    depot = torch.zeros(problems.size(0), 1, dtype=torch.long, device=problems.device)
    if selected.numel() == 0 or selected.size(1) == 0:
        return depot
    return torch.cat((depot, selected.to(device=problems.device, dtype=torch.long)), dim=1)


def _cvrp_step_state(problems: torch.Tensor, selected: torch.Tensor) -> SimpleNamespace:
    batch, nodes, _ = problems.shape
    current_node = selected[:, -1:] if selected.numel() and selected.size(1) else None
    load = _raw_capacity_from_state(SimpleNamespace(problems=problems))[:, None].to(dtype=problems.dtype, device=problems.device)
    visited = _cvrp_visited_customer_mask(nodes, selected)
    mask_bool = visited
    current = current_node if current_node is not None else torch.zeros(batch, 1, dtype=torch.long, device=problems.device)
    feasible_customer = (~mask_bool[:, 1:]).any(dim=1)
    mask_bool[:, 0] = (current.squeeze(1) == 0) & feasible_customer
    ninf_mask = torch.zeros(batch, 1, nodes, dtype=problems.dtype, device=problems.device)
    ninf_mask = ninf_mask.masked_fill(mask_bool[:, None, :], float("-inf"))
    return SimpleNamespace(
        selected_count=selected.size(1),
        load=load,
        current_node=current,
        ninf_mask=ninf_mask,
        finished=visited[:, 1:].all(dim=1)[:, None],
        batch_size=batch,
        pomo_size=1,
        BATCH_IDX=torch.arange(batch, device=problems.device)[:, None],
        POMO_IDX=torch.zeros(batch, 1, dtype=torch.long, device=problems.device),
    )


def _icam_cvrp_local_feature(problems: torch.Tensor, selected: torch.Tensor) -> torch.Tensor | None:
    if selected.numel() == 0 or selected.size(1) == 0:
        return None
    batch, nodes, _ = problems.shape
    dist = _pairwise_distances(problems[:, :, :2])
    current = selected[:, -1:].clamp(min=0, max=nodes - 1)
    return dist.gather(dim=1, index=current[:, :, None].expand(batch, 1, nodes))


def _cvrp_local_features(problems: torch.Tensor, selected: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, nodes, _ = problems.shape
    dist = _pairwise_distances(problems[:, :, :2])
    if selected.numel() == 0 or selected.size(1) == 0:
        current = torch.zeros(batch, 1, dtype=torch.long, device=problems.device)
    else:
        current = selected[:, -1:].clamp(min=0, max=nodes - 1)
    cur_dist = torch.take_along_dim(
        dist[:, None, :, :].expand(batch, 1, nodes, nodes),
        current[:, :, None, None].expand(batch, 1, 1, nodes),
        dim=2,
    ).squeeze(2)
    expanded_xy = problems[:, None, :, :2].expand(batch, 1, nodes, 2)
    current_xy = torch.take_along_dim(
        expanded_xy,
        current[:, :, None, None].expand(batch, 1, 1, 2),
        dim=2,
    )
    relative_xy = expanded_xy - current_xy
    cur_theta = torch.atan2(relative_xy[:, :, :, 1], relative_xy[:, :, :, 0])
    load = _raw_capacity_from_state(SimpleNamespace(problems=problems))[:, None, None].clamp_min(1e-12)
    norm_demand = problems[:, None, :, 2] / load
    return cur_dist, cur_theta, relative_xy, norm_demand


def _cvrp_policy_result(
    problems: torch.Tensor,
    picked: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    picked = picked.to(device=problems.device, dtype=torch.long)
    flag = _capacity_safe_cvrp_flag(
        problems,
        picked,
        torch.zeros(picked.size(0), dtype=torch.int, device=problems.device),
    )
    loss_node = torch.zeros(picked.size(0), dtype=problems.dtype, device=problems.device)
    return loss_node, picked, picked, flag, flag.clone()


def _cvrp_policy_result_with_flag(
    problems: torch.Tensor,
    picked: torch.Tensor,
    flag: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    picked = picked.to(device=problems.device, dtype=torch.long)
    flag = _capacity_safe_cvrp_flag(problems, picked, flag.to(device=problems.device, dtype=torch.int))
    loss_node = torch.zeros(picked.size(0), dtype=problems.dtype, device=problems.device)
    return loss_node, picked, picked, flag, flag.clone()


def _cvrp_visited_customer_mask(nodes: int, selected: torch.Tensor) -> torch.Tensor:
    visited = torch.zeros(selected.size(0), nodes, dtype=torch.bool, device=selected.device)
    if selected.numel() > 0:
        customer_selected = selected.clamp(min=0, max=nodes - 1)
        customer_selected = customer_selected.masked_fill(customer_selected == 0, 0)
        visited.scatter_(1, customer_selected, True)
        visited[:, 0] = False
    return visited


def _valid_cvrp_or_raise(
    problems: torch.Tensor,
    selected_node_list: torch.Tensor,
    picked: torch.Tensor,
    policy_key: str,
) -> torch.Tensor:
    selected = _selected_2d(selected_node_list, problems.device)
    picked = picked.to(device=problems.device, dtype=torch.long)
    out_of_range = (picked < 0) | (picked >= problems.size(1))
    if bool(out_of_range.any()):
        rows = out_of_range.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"{policy_key} returned out-of-range CVRP node for rows {rows}")
    visited = _cvrp_visited_customer_mask(problems.size(1), selected)
    has_feasible_customer = (~visited[:, 1:]).any(dim=1)
    bad = visited.gather(1, picked[:, None]).squeeze(1)
    bad = bad | ((picked == 0) & has_feasible_customer)
    if bool(bad.any()):
        rows = bad.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"{policy_key} returned visited/depot CVRP node while customers remain for rows {rows}")
    return picked


def _valid_explicit_cvrp_or_raise(
    problems: torch.Tensor,
    selected_node_list: torch.Tensor,
    picked: torch.Tensor,
    policy_key: str,
) -> torch.Tensor:
    selected = _selected_2d(selected_node_list, problems.device)
    picked = picked.to(device=problems.device, dtype=torch.long)
    out_of_range = (picked < 0) | (picked >= problems.size(1))
    if bool(out_of_range.any()):
        rows = out_of_range.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"{policy_key} returned out-of-range CVRP node for rows {rows}")
    visited = _cvrp_visited_customer_mask(problems.size(1), selected)
    bad = visited.gather(1, picked[:, None]).squeeze(1)
    if bool(bad.any()):
        rows = bad.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"{policy_key} returned visited CVRP customer for rows {rows}")
    return picked


def _first_cvrp_from_tour(problems: torch.Tensor, selected_node_list: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    selected = _selected_2d(selected_node_list, problems.device)
    tour = tour.to(device=problems.device, dtype=torch.long)
    out_of_range = (tour < 0) | (tour >= problems.size(1))
    if bool(out_of_range.any()):
        rows = sorted(set(out_of_range.nonzero(as_tuple=False)[:, 0].tolist()))
        raise ValueError(f"neural CVRP tour contained out-of-range node ids for rows {rows}")
    visited = _cvrp_visited_customer_mask(problems.size(1), selected)
    feasible = ~visited
    has_feasible_customer = feasible[:, 1:].any(dim=1)
    feasible[:, 0] = ~has_feasible_customer
    picks = []
    for row_index in range(tour.size(0)):
        row_pick = None
        for candidate in tour[row_index]:
            if bool(feasible[row_index, candidate]):
                row_pick = candidate
                break
        if row_pick is None:
            raise ValueError(f"neural CVRP tour did not contain a feasible next node for row {row_index}")
        picks.append(row_pick)
    return torch.stack(picks).to(dtype=torch.long, device=problems.device)


def _first_unvisited_from_tour(
    coords: torch.Tensor,
    selected_node_list: torch.Tensor,
    tour: torch.Tensor,
    start_index: int,
) -> torch.Tensor:
    selected = _selected_2d(selected_node_list, coords.device)
    visited = _visited_mask(coords.size(1), selected, start_index)
    tour = tour.to(device=coords.device, dtype=torch.long)
    out_of_range = (tour < 0) | (tour >= coords.size(1))
    if bool(out_of_range.any()):
        rows = sorted(set(out_of_range.nonzero(as_tuple=False)[:, 0].tolist()))
        raise ValueError(f"neural TSP tour contained out-of-range node ids for rows {rows}")
    picks = []
    for row_index in range(tour.size(0)):
        row_pick = None
        for candidate in tour[row_index]:
            if not bool(visited[row_index, candidate]):
                row_pick = candidate
                break
        if row_pick is None:
            raise ValueError(f"neural TSP tour did not contain an unvisited next node for row {row_index}")
        picks.append(row_pick)
    return torch.stack(picks).to(dtype=torch.long, device=coords.device)


def _valid_or_raise(
    coords: torch.Tensor,
    selected_node_list: torch.Tensor,
    picked: torch.Tensor,
    start_index: int,
    policy_key: str,
) -> torch.Tensor:
    nodes = coords.size(1)
    selected = selected_node_list.to(device=coords.device, dtype=torch.long)
    visited = _visited_mask(nodes, selected, start_index)
    picked = picked.to(device=coords.device, dtype=torch.long)
    out_of_range = (picked < 0) | (picked >= nodes)
    if bool(out_of_range.any()):
        rows = out_of_range.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"{policy_key} returned out-of-range TSP node for rows {rows}")
    bad = visited.gather(1, picked[:, None]).squeeze(1)
    if bool(bad.any()):
        rows = bad.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"{policy_key} returned visited TSP node for rows {rows}")
    return picked


def _visited_mask(nodes: int, selected: torch.Tensor, start_index: int) -> torch.Tensor:
    visited = torch.zeros(selected.size(0), nodes, dtype=torch.bool, device=selected.device)
    if selected.numel() > 0:
        visited.scatter_(1, selected.clamp(min=0, max=nodes - 1), True)
    if start_index > 0:
        visited[:, :start_index] = True
    return visited


def _bq_local_tsp_inputs(
    coords: torch.Tensor,
    selected: torch.Tensor,
    visited: torch.Tensor,
    raw_coords: torch.Tensor | None = None,
    edge_weight_type: str = "EUC_2D",
    raw_dist_matrix: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, nodes, dims = coords.shape
    last = selected[:, -1].clamp(min=0, max=nodes - 1)
    first = selected[:, 0].clamp(min=0, max=nodes - 1)
    local_to_global = _bq_tsp_original_subproblem_nodes(
        coords,
        raw_coords if raw_coords is not None else coords,
        last,
        visited,
        target=first,
        edge_weight_type=edge_weight_type,
        raw_dist_matrix=raw_dist_matrix,
    )
    local = coords.gather(1, local_to_global[:, :, None].expand(batch, local_to_global.size(1), dims))
    return local, local_to_global


def _bq_tsp_original_subproblem_nodes(
    coords: torch.Tensor,
    raw_coords: torch.Tensor,
    current: torch.Tensor,
    visited: torch.Tensor,
    *,
    target: torch.Tensor,
    edge_weight_type: str,
    raw_dist_matrix: torch.Tensor | None,
) -> torch.Tensor:
    batch, nodes, _dims = coords.shape
    blocked = visited.clone()
    blocked.scatter_(1, current[:, None], True)
    blocked.scatter_(1, target[:, None], True)
    remaining = _candidate_index_matrix(blocked)
    subproblem_nodes = torch.cat((current[:, None], remaining, target[:, None]), dim=1)
    if not (0 < _BQ_KNNS < subproblem_nodes.size(1)):
        return subproblem_nodes

    nonterminal = subproblem_nodes[:, :-1]
    if raw_dist_matrix is not None:
        row_distances = raw_dist_matrix.gather(1, current[:, None, None].expand(batch, 1, raw_dist_matrix.size(-1))).squeeze(1)
        distances = row_distances.gather(1, nonterminal)
    else:
        current_xy = raw_coords.gather(1, current[:, None, None].expand(batch, 1, raw_coords.size(-1))).squeeze(1)
        raw_order_coords = raw_coords.gather(1, nonterminal[:, :, None].expand(batch, nonterminal.size(1), raw_coords.size(-1)))
        distances = torch.linalg.vector_norm(raw_order_coords - current_xy[:, None, :], dim=-1)
        if edge_weight_type == "EUC_2D":
            distances = torch.floor(distances + 0.5)
        elif edge_weight_type == "CEIL_2D":
            distances = torch.ceil(distances)

    sorted_nodes_idx = torch.sort(distances, dim=-1).indices
    knn_positions = sorted_nodes_idx[:, : _BQ_KNNS - 1]
    knn_nodes = nonterminal.gather(1, knn_positions)
    return torch.cat((knn_nodes, target[:, None]), dim=1)


def _bq_local_cvrp_inputs(
    problems: torch.Tensor,
    selected: torch.Tensor,
    visited: torch.Tensor,
    raw_capacity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, nodes, _dims = problems.shape
    last = selected[:, -1].clamp(min=0, max=nodes - 1)
    depot = torch.zeros(batch, dtype=torch.long, device=problems.device)
    candidates = _bq_nearest_candidates(problems[:, :, :2], last, visited, target=depot)
    local_to_global = torch.cat((last[:, None], candidates, torch.zeros(batch, 1, dtype=torch.long, device=problems.device)), dim=1)
    coords = problems[:, :, :2]
    demand = problems[:, :, 2]
    local_coords = coords.gather(1, local_to_global[:, :, None].expand(batch, local_to_global.size(1), 2))
    local_demand = demand.gather(1, local_to_global)
    capacity = _capacity_vector(raw_capacity, problems)
    remaining = problems[:, 0, 3]
    local = torch.cat(
        (
            local_coords,
            (local_demand / capacity[:, None])[:, :, None],
            (remaining / capacity)[:, None, None].expand(batch, local_to_global.size(1), 1),
        ),
        dim=-1,
    )
    return local, local_to_global, local_demand


def _bq_nearest_candidates(
    coords: torch.Tensor,
    current: torch.Tensor,
    visited: torch.Tensor,
    *,
    target: torch.Tensor,
) -> torch.Tensor:
    batch, nodes, _dims = coords.shape
    current_xy = coords.gather(1, current[:, None, None].expand(batch, 1, coords.size(-1))).squeeze(1)
    distances = torch.linalg.vector_norm(coords - current_xy[:, None, :], dim=-1)
    blocked = visited.clone()
    blocked.scatter_(1, current[:, None], True)
    blocked.scatter_(1, target[:, None], True)
    available = (~blocked).sum(dim=1)
    if bool((available + 2 <= _BQ_KNNS).all()):
        return _candidate_index_matrix(blocked)
    distances = distances.masked_fill(blocked, float("inf"))
    width = max(1, min(_BQ_KNNS - 2, nodes - 2))
    candidates = torch.topk(distances, k=width, dim=1, largest=False).indices
    finite = torch.isfinite(distances.gather(1, candidates))
    if bool(finite.all()):
        return candidates
    rows = []
    for row_index in range(batch):
        valid = candidates[row_index][finite[row_index]]
        if valid.numel() == 0:
            valid = current[row_index : row_index + 1]
        if valid.numel() < width:
            valid = torch.cat((valid, valid[-1:].expand(width - valid.numel())))
        rows.append(valid)
    return torch.stack(rows, dim=0)


def _candidate_index_matrix(visited: torch.Tensor) -> torch.Tensor:
    batch, nodes = visited.shape
    candidates = []
    for row in visited:
        candidate = torch.arange(nodes, device=visited.device, dtype=torch.long)[~row]
        if candidate.numel() == 0:
            candidate = torch.zeros(1, dtype=torch.long, device=visited.device)
        candidates.append(candidate)
    width = max(candidate.numel() for candidate in candidates)
    padded = [
        torch.cat((candidate, candidate[-1:].expand(width - candidate.numel())))
        if candidate.numel() < width
        else candidate
        for candidate in candidates
    ]
    return torch.stack(padded, dim=0)


def _pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(coords[:, :, None, :] - coords[:, None, :, :], dim=-1)


def _dgl_tsp_data(coords: torch.Tensor, selected_node_list: torch.Tensor) -> torch.Tensor:
    selected = selected_node_list.to(device=coords.device, dtype=torch.long)
    batch, nodes, _ = coords.shape
    if selected.numel() == 0 or selected.size(1) == 0:
        selected = torch.zeros(batch, 1, dtype=torch.long, device=coords.device)
    last = selected[:, -1].clamp(min=0, max=nodes - 1)
    first = selected[:, 0].clamp(min=0, max=nodes - 1)
    current_xy = coords.gather(1, last[:, None, None].expand(batch, 1, 2)).squeeze(1)
    first_xy = coords.gather(1, first[:, None, None].expand(batch, 1, 2)).squeeze(1)
    deltas = coords - current_xy[:, None, :]
    dist_current = torch.linalg.vector_norm(deltas, dim=-1)
    avg_dist = _pairwise_distances(coords).mean(dim=-1)
    std_dist = _pairwise_distances(coords).std(dim=-1, unbiased=False)
    dest_delta = first_xy[:, None, :] - coords
    dest_norm = torch.linalg.vector_norm(dest_delta, dim=-1).clamp_min(1e-12)
    sin_dest = dest_delta[:, :, 1] / dest_norm
    cos_dest = dest_delta[:, :, 0] / dest_norm
    return torch.cat(
        (
            coords,
            dist_current[:, :, None],
            avg_dist[:, :, None],
            std_dist[:, :, None],
            sin_dest[:, :, None],
            cos_dest[:, :, None],
        ),
        dim=-1,
    )


def _dgl_cvrp_data(problems: torch.Tensor, selected_node_list: torch.Tensor) -> torch.Tensor:
    base = _dgl_tsp_data(problems[:, :, :2], selected_node_list)
    demand = problems[:, :, 2:3]
    return torch.cat((base[:, :, :2], demand, base[:, :, 2:]), dim=-1)


def _capacity_safe_cvrp_result(
    problems: torch.Tensor,
    result: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_node, selected_teacher, selected_student, flag_teacher, flag_student = result
    return (
        loss_node,
        selected_teacher,
        selected_student,
        _capacity_safe_cvrp_flag(problems, selected_teacher, flag_teacher),
        _capacity_safe_cvrp_flag(problems, selected_student, flag_student),
    )


def _capacity_safe_cvrp_flag(problems: torch.Tensor, selected: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    demand = _gather_feature(problems, selected, feature=2)
    remaining = problems[:, 0, 3].to(device=selected.device, dtype=demand.dtype)
    must_refill = demand > remaining + 1e-9
    flag_bool = flag.to(device=selected.device).bool() | must_refill
    return flag_bool.to(device=flag.device, dtype=flag.dtype)


def _load_neural_model(spec: NeuralPolicySpec, device: torch.device) -> torch.nn.Module:
    cache_key = (spec.key, str(device))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not spec.checkpoint_path.exists():
        raise FileNotFoundError(f"missing neural policy checkpoint: {spec.checkpoint_path}")
    if not spec.module_path.exists():
        raise FileNotFoundError(f"missing neural policy module: {spec.module_path}")

    module_name = f"_nrs_forward_swap_{spec.key}"
    module_spec = importlib.util.spec_from_file_location(module_name, spec.module_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"could not load module spec for {spec.module_path}")
    module = importlib.util.module_from_spec(module_spec)
    import_roots = spec.import_roots or (spec.module_path.parent,)
    with _temporary_import_roots(import_roots):
        module_spec.loader.exec_module(module)
    model_cls = getattr(module, spec.class_name)
    neural_model = model_cls(**dict(spec.model_params))
    if spec.family == "elg" and bool(spec.model_params.get("ensemble")) and hasattr(neural_model, "decoder"):
        add_local_policy = getattr(neural_model.decoder, "add_local_policy", None)
        if callable(add_local_policy):
            add_local_policy(device)
    try:
        checkpoint = torch.load(spec.checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(spec.checkpoint_path, map_location=device)
    if isinstance(checkpoint, Mapping):
        state_dict = checkpoint.get(spec.checkpoint_key, checkpoint)
    else:
        state_dict = checkpoint
    neural_model.load_state_dict(state_dict)
    neural_model.to(device)
    neural_model.eval()
    _MODEL_CACHE[cache_key] = neural_model
    return neural_model


@contextmanager
def _temporary_import_roots(roots: Sequence[Path]):
    previous_path = list(sys.path)
    shadowed_names = {
        "model",
        "models",
        "encoder",
        "decoder",
        "utils",
        "TSPModel",
        "CVRPModel",
        "TSPEnv",
        "CVRPEnv",
    }
    previous_modules = {name: sys.modules.get(name) for name in shadowed_names if name in sys.modules}
    for name in shadowed_names:
        sys.modules.pop(name, None)
    try:
        for root in reversed([str(path) for path in roots if path.exists()]):
            if root not in sys.path:
                sys.path.insert(0, root)
        yield
    finally:
        sys.path[:] = previous_path
        for name in shadowed_names:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


def _drhg_tsp_context(state: Any, point_couples: Optional[torch.Tensor],
                      endpoint_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = _coords_from_state(state)
    if point_couples is None:
        nodes = coords.size(1)
        node_ids = torch.arange(nodes, device=coords.device, dtype=torch.long)
        point_couples = torch.stack((node_ids, node_ids), dim=-1).unsqueeze(0).expand(coords.size(0), nodes, 2)
    if endpoint_mask is None:
        endpoint_mask = torch.zeros(coords.size(0), coords.size(1), dtype=coords.dtype, device=coords.device)
    return point_couples.to(device=coords.device, dtype=torch.long), endpoint_mask.to(device=coords.device, dtype=coords.dtype)


def _drhg_cvrp_context(state: Any, point_couples: Optional[torch.Tensor],
                       endpoint_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    problems = _problems_from_state(state)
    if point_couples is None:
        nodes = problems.size(1)
        node_ids = torch.arange(nodes, device=problems.device, dtype=torch.long)
        point_couples = torch.stack((node_ids, node_ids), dim=-1).unsqueeze(0).expand(problems.size(0), nodes, 2)
    if endpoint_mask is None:
        endpoint_mask = torch.zeros(problems.size(0), problems.size(1), dtype=problems.dtype, device=problems.device)
    return point_couples.to(device=problems.device, dtype=torch.long), endpoint_mask.to(device=problems.device, dtype=problems.dtype)


def _raw_capacity_from_state(state: Any) -> torch.Tensor:
    problems = _problems_from_state(state)
    if problems.size(-1) > 3:
        return problems[:, 0, 3]
    return torch.ones(problems.size(0), dtype=problems.dtype, device=problems.device)


def _forward_device(original_device: torch.device) -> torch.device:
    configured = os.environ.get("NRS_FORWARD_SWAP_DEVICE", "native").strip().lower()
    if configured in {"native", "input", "same"}:
        return original_device
    return torch.device(configured)


def _tensor_to_device(value: Any, device: torch.device) -> Any:
    if hasattr(value, "to"):
        return value.to(device=device)
    return value


def _state_to_device(state: Any, device: torch.device) -> Any:
    attrs = {}
    for name in ("data", "problems", "raw_coords", "raw_dist_matrix", "edge_weight_type", "first_node", "current_node"):
        if hasattr(state, name):
            attrs[name] = _tensor_to_device(getattr(state, name), device)
    if attrs:
        return SimpleNamespace(**attrs)
    return state


def _to_device(value: Any, device: torch.device) -> Any:
    if hasattr(value, "to"):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


@contextmanager
def _cpu_default_tensor_type(device: torch.device, restore_device: torch.device):
    if device.type != "cpu":
        yield
        return
    default_dtype = torch.get_default_dtype()
    restore_cuda_default = restore_device.type == "cuda"
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_dtype(default_dtype)
    try:
        yield
    finally:
        if restore_cuda_default and torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(default_dtype)


def _announce_once(spec: Optional[NeuralPolicySpec] = None) -> None:
    global _ANNOUNCED
    if _ANNOUNCED:
        return
    _ANNOUNCED = True
    if spec is None:
        print(f"[NRS_FORWARD_OVERRIDE] native forward replaced by {label()}", flush=True)
        return
    print(
        "[NRS_FORWARD_OVERRIDE] native forward replaced by "
        f"actual neural policy {spec.key} from {spec.checkpoint_path}",
        flush=True,
    )
