# j2020_algorithm.py
# Paper-inspired J2020 heuristic (multi-criteria, min-min style) for real servers.
# Adds exec_time_fn hook so the scheduler can pass a profiled predictor.

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable
import random

# VM must have: .name, .ip, .cpu_cores, .ram_gb
# Task must have: .id, .name, .index, .cpu_load

@dataclass
class J2020Weights:
    w_ect: float = 0.70     # weight for Estimated Completion Time (ready_time + exec)
    w_load: float = 0.25    # weight for predicted imbalance after assignment
    w_energy: float = 0.05  # weight for simple energy proxy

@dataclass
class J2020EnergyModel:
    base_watt: float = 1.0
    per_core_watt: float = 1.0

def _default_exec_time(task, vm) -> float:
    cores = max(1, int(getattr(vm, "cpu_cores", 1)))
    return float(getattr(task, "cpu_load")) / float(cores)

def _predicted_ect(vm_ready_time: Dict[str, float], vm_name: str, t_exec: float) -> float:
    return vm_ready_time[vm_name] + t_exec

def _energy_proxy(t_exec: float, vm) -> float:
    return t_exec * (1.0 + float(getattr(vm, "cpu_cores", 1)))

def _degree_of_imbalance_after(
    vm_loads: Dict[str, float],
    vm_name: str,
    add_time: float
) -> float:
    tmp = vm_loads.copy()
    tmp[vm_name] = tmp.get(vm_name, 0.0) + add_time
    loads = list(tmp.values())
    if not loads:
        return 0.0
    mx = max(loads)
    mn = min(loads)
    avg = sum(loads) / len(loads)
    if avg <= 0.0:
        return 0.0
    return (mx - mn) / avg

def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return values
    mn, mx = min(values), max(values)
    if mx - mn <= 1e-12:
        return [0.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]

def j2020_schedule(
    tasks: List[Any],
    vms: List[Any],
    weights: J2020Weights = J2020Weights(),
    energy: J2020EnergyModel = J2020EnergyModel(),
    tie_break: str = "min_ect",      # "min_ect" | "random"
    shuffle_tasks: bool = False,
    exec_time_fn: Callable[[Any, Any], float] | None = None
) -> Dict[int, str]:
    """
    Returns mapping: task_id -> vm_name
    Composite score per (task, vm):
      score = w_ect * norm(ECT) + w_load * norm(DOI_after) + w_energy * norm(E_proxy)
    with ECT = VM_ready_time + exec_time(task, vm).

    exec_time_fn: optional callable to predict exec time; if None uses _default_exec_time.
    """
    if not tasks or not vms:
        return {}

    remaining: List[Any] = list(tasks)
    if shuffle_tasks:
        random.shuffle(remaining)

    vm_names = [vm.name for vm in vms]
    vm_by_name = {vm.name: vm for vm in vms}

    vm_ready_time: Dict[str, float] = {vm.name: 0.0 for vm in vms}
    vm_loads: Dict[str, float] = {vm.name: 0.0 for vm in vms}

    et_fn = exec_time_fn if exec_time_fn is not None else _default_exec_time
    assignment: Dict[int, str] = {}

    while remaining:
        per_task_best: List[Tuple[Any, str, float, float]] = []  # (task, best_vm, best_score, best_ect)

        for task in remaining:
            ects: List[float] = []
            dois: List[float] = []
            eners: List[float] = []
            cand:  List[str]  = []

            for vm_name in vm_names:
                vm = vm_by_name[vm_name]
                t_exec = et_fn(task, vm)
                ect = _predicted_ect(vm_ready_time, vm_name, t_exec)
                doi = _degree_of_imbalance_after(vm_loads, vm_name, t_exec)
                eproxy = t_exec * (energy.base_watt + energy.per_core_watt * float(getattr(vm, "cpu_cores", 1)))

                ects.append(ect); dois.append(doi); eners.append(eproxy); cand.append(vm_name)

            n_ect = _minmax_norm(ects)
            n_doi = _minmax_norm(dois)
            n_enr = _minmax_norm(eners)

            scores = [
                weights.w_ect * n_ect[i] +
                weights.w_load * n_doi[i] +
                weights.w_energy * n_enr[i]
                for i in range(len(cand))
            ]

            min_score = min(scores)
            idxs = [i for i, s in enumerate(scores) if abs(s - min_score) <= 1e-12]
            if len(idxs) > 1:
                if tie_break == "min_ect":
                    best_i = min(idxs, key=lambda i: ects[i])
                else:
                    best_i = random.choice(idxs)
            else:
                best_i = idxs[0]

            per_task_best.append((task, cand[best_i], scores[best_i], ects[best_i]))

        chosen_task, chosen_vm_name, _, _ = min(per_task_best, key=lambda t: (t[2], t[3]))
        vm = vm_by_name[chosen_vm_name]
        t_exec = et_fn(chosen_task, vm)

        vm_ready_time[chosen_vm_name] += t_exec
        vm_loads[chosen_vm_name] += t_exec
        assignment[int(chosen_task.id)] = chosen_vm_name

        remaining = [t for t in remaining if int(t.id) != int(chosen_task.id)]

    return assignment
