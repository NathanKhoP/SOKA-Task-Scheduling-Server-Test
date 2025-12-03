# scheduler.py
import asyncio
import httpx
import time
from datetime import datetime
import csv
import json
import pandas as pd
import sys
import os
from dotenv import load_dotenv
from collections import namedtuple
from typing import Callable, Dict, List, Optional

from shc_algorithm import stochastic_hill_climb
from j2020_algorithm import j2020_schedule, J2020Weights, J2020EnergyModel

# --- Config ---

load_dotenv()

VM_SPECS = {
    'vm1': {'ip': os.getenv("VM1_IP"), 'cpu': 1, 'ram_gb': 1},
    'vm2': {'ip': os.getenv("VM2_IP"), 'cpu': 2, 'ram_gb': 2},
    'vm3': {'ip': os.getenv("VM3_IP"), 'cpu': 4, 'ram_gb': 4},
    'vm4': {'ip': os.getenv("VM4_IP"), 'cpu': 8, 'ram_gb': 4},
}

VM_PORT = int(os.getenv("VM_PORT", "5000"))

# Output files per pass
SHC_RESULTS_FILE = os.getenv("SHC_RESULTS_FILE", "shc_results.csv")
J2020_RESULTS_FILE = os.getenv("J2020_RESULTS_FILE", "j2020_results.csv")
RR_RESULTS_FILE = os.getenv("RR_RESULTS_FILE", "rr_results.csv")
FCFS_RESULTS_FILE = os.getenv("FCFS_RESULTS_FILE", "fcfs_results.csv")
SUMMARY_FILE = os.getenv("SUMMARY_FILE", "summary.txt")
COMPARISON_RESULTS_FILE = os.getenv("COMPARISON_RESULTS_FILE", "comparison_results.csv")

# SHC tunables
SHC_ITERATIONS = int(os.getenv("SHC_ITERATIONS", "1000"))

# J2020 dispatch capacity per VM (sequential by default)
PER_VM_CONCURRENCY = int(os.getenv("PER_VM_CONCURRENCY", "1"))
# J2020 profiling probe (task index 1..10, mirrors server’s scaling)
J2020_PROBE_INDEX = int(os.getenv("J2020_PROBE_INDEX", "5"))

DATASET_ITERATIONS = int(os.getenv("DATASET_ITERATIONS", "10"))
RUN_LOG_FILE = os.getenv("RUN_LOG_FILE", "scheduler_run.log")

DATASET_FILES = {
    "low_high": os.getenv("DATASET_LOW_HIGH", "dataset-low-high.txt"),
    "rand": os.getenv("DATASET_RAND", "dataset-rand.txt"),
    "rand_stratified": os.getenv("DATASET_RAND_STRATIFIED", "dataset-rand-stratified.txt"),
}

DATASET_DISPLAY_NAMES = {
    "low_high": "Low-High_Dataset",
    "rand": "Random_Simple_Dataset",
    "rand_stratified": "Stratified_Random_Dataset",
}

ALGORITHM_DISPLAY_NAMES = {
    "shc": "Stochastic_Hill_Climbing",
    "j2020": "J2020",
    "rr": "Round_Robin",
    "fcfs": "FCFS",
}

METRIC_FIELDS = [
    "num_tasks",
    "makespan",
    "throughput",
    "total_cpu_time",
    "total_wait_time",
    "avg_wait_time",
    "avg_start_time",
    "avg_exec_time",
    "avg_finish_time",
    "imbalance_degree",
    "resource_utilization",
    "effective_slots",
]

VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load'])

# --- Helpers ---

def get_task_load(index: int) -> float:
    # must mirror the server's time scaling (index^2 * 10000 loop in the reference harness)
    return float(index * index * 10000)

def load_tasks(dataset_path: str) -> List[Task]:
    if not os.path.exists(dataset_path):
        print(f"Error: File dataset '{dataset_path}' tidak ditemukan.", file=sys.stderr)
        sys.exit(1)

    tasks: List[Task] = []
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                index = int(line.strip())
                if not 1 <= index <= 10:
                    print(f"Peringatan: Task index {index} di baris {i+1} di luar rentang (1-10).")
                    continue
                cpu_load = get_task_load(index)
                task_name = f"task-{index}-{i}"
                tasks.append(Task(
                    id=i,
                    name=task_name,
                    index=index,
                    cpu_load=cpu_load,
                ))
            except ValueError:
                print(f"Peringatan: Mengabaikan baris {i+1} yang tidak valid: '{line.strip()}'")

    print(f"Berhasil memuat {len(tasks)} tugas dari {dataset_path}")
    return tasks


def _ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def serialize_results_for_log(results_list: List[dict]) -> List[dict]:
    serialized = []
    for item in results_list:
        serialized.append({
            "index": item.get("index"),
            "task_name": item.get("task_name"),
            "vm_assigned": item.get("vm_assigned"),
            "start_time": item.get("start_time").isoformat() if item.get("start_time") else None,
            "exec_time": item.get("exec_time"),
            "finish_time": item.get("finish_time").isoformat() if item.get("finish_time") else None,
            "wait_time": item.get("wait_time"),
        })
    return serialized


def append_log_entry(log_path: str, entry: dict) -> None:
    _ensure_parent_dir(log_path)
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(json.dumps(entry) + "\n")


def append_summary_entry(iteration_label: str, metrics: Dict[str, float], summary_path: str = SUMMARY_FILE) -> None:
    if not metrics:
        return

    _ensure_parent_dir(summary_path)
    lines = [
        f"iterasi {iteration_label}:",
        f"Semua eksekusi tugas selesai dalam {metrics['makespan']:.4f} detik.",
        "",
        "--- Hasil ---",
        f"Total Tugas Selesai       : {int(metrics['num_tasks'])}",
        f"Makespan (Waktu Total)    : {metrics['makespan']:.4f} detik",
        f"Throughput                : {metrics['throughput']:.4f} tugas/detik",
        f"Total CPU Time            : {metrics['total_cpu_time']:.4f} detik",
        f"Total Wait Time           : {metrics['total_wait_time']:.4f} detik",
        f"Average Start Time (rel)  : {metrics['avg_start_time']:.4f} detik",
        f"Average Execution Time    : {metrics['avg_exec_time']:.4f} detik",
        f"Average Finish Time (rel) : {metrics['avg_finish_time']:.4f} detik",
        f"Imbalance Degree          : {metrics['imbalance_degree']:.4f}",
        f"Resource Utilization (CPU): {metrics['resource_utilization'] * 100:.4f}%",
        "",
    ]

    with open(summary_path, 'a', encoding='utf-8') as summary_file:
        summary_file.write("\n".join(lines) + "\n")


def average_metric_records(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    result: Dict[str, float] = {}
    for field in METRIC_FIELDS:
        values = [rec[field] for rec in records if field in rec]
        if not values:
            continue
        result[field] = sum(values) / len(values)
    return result


def write_average_metrics_csv(out_file: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        print(f"Tidak ada data untuk ditulis ke {out_file}")
        return

    headers = ["algorithm", "dataset", "runs"] + METRIC_FIELDS
    _ensure_parent_dir(out_file)
    try:
        with open(out_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in headers})
        print(f"Ringkasan rata-rata disimpan ke {out_file}")
    except IOError as e:
        print(f"Error menulis ringkasan ke {out_file}: {e}", file=sys.stderr)


def build_comparison_row(algorithm_key: str, dataset_key: str, metrics: Dict[str, float]) -> Dict[str, object]:
    return {
        "algorithm": ALGORITHM_DISPLAY_NAMES.get(algorithm_key, algorithm_key.upper()),
        "dataset": DATASET_DISPLAY_NAMES.get(dataset_key, dataset_key),
        "total_tasks": metrics.get("num_tasks", 0.0),
        "makespan": metrics.get("makespan", 0.0),
        "throughput": metrics.get("throughput", 0.0),
        "total_cpu_time": metrics.get("total_cpu_time", 0.0),
        "total_wait_time": metrics.get("total_wait_time", 0.0),
        "avg_start_time": metrics.get("avg_start_time", 0.0),
        "avg_exec_time": metrics.get("avg_exec_time", 0.0),
        "avg_finish_time": metrics.get("avg_finish_time", 0.0),
        "avg_wait_time": metrics.get("avg_wait_time", 0.0),
        "imbalance_degree": metrics.get("imbalance_degree", 0.0),
        "resource_utilization": metrics.get("resource_utilization", 0.0),
    }


def write_comparison_results(rows: List[Dict[str, object]], out_file: str) -> None:
    if not rows:
        print("Tidak ada data untuk comparison_results.csv")
        return

    headers = [
        "algorithm",
        "dataset",
        "total_tasks",
        "makespan",
        "throughput",
        "total_cpu_time",
        "total_wait_time",
        "avg_start_time",
        "avg_exec_time",
        "avg_finish_time",
        "avg_wait_time",
        "imbalance_degree",
        "resource_utilization",
    ]

    _ensure_parent_dir(out_file)
    try:
        with open(out_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in headers})
        print(f"Comparison results disimpan ke {out_file}")
    except IOError as e:
        print(f"Error menulis comparison results ke {out_file}: {e}", file=sys.stderr)

# --- Profiling real per-VM speeds (for J2020) ---

async def _probe_vm(vm: VM, probe_index: int) -> float:
    """Run a tiny task on vm to measure baseline runtime (seconds)."""
    url = f"http://{vm.ip}:{VM_PORT}/task/{probe_index}"
    start = time.monotonic()
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=300.0)
        resp.raise_for_status()
    return time.monotonic() - start

async def profile_vms(vms: List[VM], probe_index: int) -> Dict[str, float]:
    """
    Returns dict vm_name -> t_profile_seconds for the probe task.
    If any probe fails, fill with median of successful probes; if none succeed, default 1.0s.
    """
    results: Dict[str, float] = {}
    tasks = []
    for vm in vms:
        async def run(vm=vm):
            try:
                t = await _probe_vm(vm, probe_index)
                results[vm.name] = t
            except Exception as e:
                print(f"Peringatan: probe pada {vm.name} gagal: {e}", file=sys.stderr)
        tasks.append(run())
    await asyncio.gather(*tasks, return_exceptions=True)

    if results:
        vals = sorted(results.values())
        median = vals[len(vals)//2]
    else:
        median = 1.0
    for vm in vms:
        if vm.name not in results:
            results[vm.name] = median

    print("\nProfil kecepatan VM (detik untuk task index "
          f"{probe_index}): " + ", ".join(f"{k}={v:.4f}s" for k,v in results.items()))
    return results

def make_profiled_exec_time_fn(t_profile: Dict[str, float], probe_index: int) -> Callable[[Task, VM], float]:
    """Scale measured baseline time linearly with (cpu_load / probe_load)."""
    base_load = get_task_load(probe_index)
    def _fn(task: Task, vm: VM) -> float:
        t_base = t_profile.get(vm.name, 1.0)
        return t_base * (float(task.cpu_load) / base_load)
    return _fn

# --- Async Execution primitives (shared) ---

async def execute_task_on_vm(task: Task, vm: VM, client: httpx.AsyncClient,
                             vm_semaphore: asyncio.Semaphore, results_list: list):
    url = f"http://{vm.ip}:{VM_PORT}/task/{task.index}"
    task_start_time = None
    task_finish_time = None
    task_exec_time = -1.0
    task_wait_time = -1.0

    wait_start_mono = time.monotonic()

    try:
        async with vm_semaphore:
            task_wait_time = time.monotonic() - wait_start_mono
            print(f"Mengeksekusi {task.name} (idx: {task.id}) di {vm.name} (IP: {vm.ip})...")

            task_start_mono = time.monotonic()
            task_start_time = datetime.now()

            response = await client.get(url, timeout=300.0)
            response.raise_for_status()

            task_finish_time = datetime.now()
            task_exec_time = time.monotonic() - task_start_mono
            print(f"Selesai {task.name} (idx: {task.id}) di {vm.name}. Waktu: {task_exec_time:.4f}s")

    except httpx.HTTPStatusError as e:
        print(f"Error HTTP pada {task.name} di {vm.name}: {e}", file=sys.stderr)
    except httpx.RequestError as e:
        print(f"Error Request pada {task.name} di {vm.name}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error tidak diketahui pada {task.name} di {vm.name}: {e}", file=sys.stderr)
    finally:
        if task_start_time is None:
            task_start_time = datetime.now()
        if task_finish_time is None:
            task_finish_time = datetime.now()

        results_list.append({
            "index": task.id,
            "task_name": task.name,
            "vm_assigned": vm.name,
            "start_time": task_start_time,
            "exec_time": task_exec_time,
            "finish_time": task_finish_time,
            "wait_time": task_wait_time
        })

async def vm_worker(vm: VM, queue: List[Task], client: httpx.AsyncClient,
                    vm_semaphore: asyncio.Semaphore, results_list: list):
    if not queue:
        return
    print(f"[{vm.name}] Queue length = {len(queue)}. Dispatching sequentially (concurrency={vm_semaphore._value}).")
    for task in queue:
        await execute_task_on_vm(task, vm, client, vm_semaphore, results_list)

# --- Results & Metrics ---


def compute_metrics(results_list: list, makespan: float, effective_slots: int):
    if not results_list:
        print("Error: Hasil kosong, tidak ada metrik untuk dihitung.", file=sys.stderr)
        return None

    try:
        df = pd.DataFrame(results_list)
    except pd.errors.EmptyDataError:
        print("Error: Hasil kosong, tidak ada metrik untuk dihitung.", file=sys.stderr)
        return None

    df['start_time'] = pd.to_datetime(df['start_time'])
    df['finish_time'] = pd.to_datetime(df['finish_time'])

    success_df = df[df['exec_time'] > 0].copy()
    if success_df.empty:
        print("Tidak ada tugas yang berhasil diselesaikan. Metrik tidak dapat dihitung.")
        return None

    num_tasks = len(success_df)

    total_cpu_time = success_df['exec_time'].sum()
    total_wait_time = success_df['wait_time'].sum()
    avg_wait_time = success_df['wait_time'].mean()

    avg_exec_time = success_df['exec_time'].mean()

    min_start = success_df['start_time'].min()
    success_df['rel_start_time'] = (success_df['start_time'] - min_start).dt.total_seconds()
    success_df['rel_finish_time'] = (success_df['finish_time'] - min_start).dt.total_seconds()

    avg_start_time = success_df['rel_start_time'].mean()
    avg_finish_time = success_df['rel_finish_time'].mean()

    makespan = float(makespan)
    throughput = num_tasks / makespan if makespan > 0 else 0.0

    vm_exec_times = success_df.groupby('vm_assigned')['exec_time'].sum()
    max_load = vm_exec_times.max()
    min_load = vm_exec_times.min()
    avg_load = vm_exec_times.mean()
    imbalance_degree = (max_load - min_load) / avg_load if avg_load > 0 else 0.0

    total_available_cpu_time = makespan * max(1, int(effective_slots))
    resource_utilization = (total_cpu_time / total_available_cpu_time) if total_available_cpu_time > 0 else 0.0

    return {
        "num_tasks": float(num_tasks),
        "makespan": makespan,
        "throughput": throughput,
        "total_cpu_time": float(total_cpu_time),
        "total_wait_time": float(total_wait_time),
        "avg_wait_time": float(avg_wait_time),
        "avg_start_time": float(avg_start_time),
        "avg_exec_time": float(avg_exec_time),
        "avg_finish_time": float(avg_finish_time),
        "imbalance_degree": float(imbalance_degree),
        "resource_utilization": float(resource_utilization),
        "effective_slots": float(effective_slots),
    }


def calculate_and_print_metrics(results_list: list, makespan: float, effective_slots: int):
    metrics = compute_metrics(results_list, makespan, effective_slots)
    if not metrics:
        return None

    print("\n--- Hasil ---")
    print(f"Total Tugas Selesai       : {int(metrics['num_tasks'])}")
    print(f"Makespan (Waktu Total)    : {metrics['makespan']:.4f} detik")
    print(f"Throughput                : {metrics['throughput']:.4f} tugas/detik")
    print(f"Total CPU Time            : {metrics['total_cpu_time']:.4f} detik")
    print(f"Total Wait Time           : {metrics['total_wait_time']:.4f} detik")
    print(f"Average Start Time (rel)  : {metrics['avg_start_time']:.4f} detik")
    print(f"Average Execution Time    : {metrics['avg_exec_time']:.4f} detik")
    print(f"Average Finish Time (rel) : {metrics['avg_finish_time']:.4f} detik")
    print(f"Imbalance Degree          : {metrics['imbalance_degree']:.4f}")
    print(f"Resource Utilization (CPU): {metrics['resource_utilization']:.4%}")
    return metrics

# --- Run modes (Pass A: SHC concurrent; Pass B: J2020 sequential per-VM) ---

async def run_shc_pass(tasks: List[Task], vms: List[VM]) -> Optional[Dict[str, object]]:
    print("\n=== PASS A: SHC (old-style parallel, semaphore = CPU cores) ===")
    assignment = stochastic_hill_climb(tasks, vms, SHC_ITERATIONS)

    print("\nPenugasan Tugas (SHC) contoh:")
    for i, (tid, vmn) in enumerate(assignment.items()):
        if i >= 10:
            print("  - ... etc.")
            break
        print(f"  - Tugas {tid} -> {vmn}")

    tasks_dict = {t.id: t for t in tasks}
    vms_dict = {vm.name: vm for vm in vms}

    results_list: List[dict] = []
    # Semaphore capacity = cpu_cores (old behavior)
    vm_semaphores = {vm.name: asyncio.Semaphore(vm.cpu_cores) for vm in vms}

    async with httpx.AsyncClient() as client:
        coros = []
        for task_id, vm_name in assignment.items():
            coros.append(execute_task_on_vm(tasks_dict[task_id], vms_dict[vm_name],
                                            client, vm_semaphores[vm_name], results_list))

        print(f"\nMemulai eksekusi {len(coros)} tugas secara paralel...")
        t0 = time.monotonic()
        await asyncio.gather(*coros)
        makespan = time.monotonic() - t0
        print(f"\nSemua eksekusi tugas selesai dalam {makespan:.4f} detik.")

    # effective capacity = sum of cpu_cores (old behavior)
    effective_slots = sum(vm.cpu_cores for vm in vms)

    metrics = calculate_and_print_metrics(results_list, makespan, effective_slots)
    return {
        "results": results_list,
        "metrics": metrics,
    }

async def run_j2020_pass(tasks: List[Task], vms: List[VM]) -> Optional[Dict[str, object]]:
    print("\n=== PASS B: J2020 (profiled predictor, per-VM sequential queues) ===")

    # Profile VMs, build predictor
    t_profile = await profile_vms(vms, J2020_PROBE_INDEX)
    exec_time_fn = make_profiled_exec_time_fn(t_profile, J2020_PROBE_INDEX)

    # Placement with J2020
    weights = J2020Weights(w_ect=0.70, w_load=0.25, w_energy=0.05)
    energy = J2020EnergyModel(base_watt=1.0, per_core_watt=1.0)

    assignment = j2020_schedule(
        tasks, vms, weights=weights, energy=energy,
        tie_break="min_ect", shuffle_tasks=False,
        exec_time_fn=exec_time_fn
    )

    print("\nPenugasan Tugas (J2020) contoh:")
    for i, (tid, vmn) in enumerate(assignment.items()):
        if i >= 10:
            print("  - ... etc.")
            break
        print(f"  - Tugas {tid} -> {vmn}")

    tasks_dict = {t.id: t for t in tasks}
    vms_dict = {vm.name: vm for vm in vms}

    # Build per-VM queues and sort LPT by predicted time
    per_vm_queues: Dict[str, List[Task]] = {vm.name: [] for vm in vms}
    for task_id, vm_name in assignment.items():
        per_vm_queues[vm_name].append(tasks_dict[task_id])

    for vm_name, q in per_vm_queues.items():
        vm = vms_dict[vm_name]
        q.sort(key=lambda t: exec_time_fn(t, vm), reverse=True)

    results_list: List[dict] = []
    vm_semaphores = {vm.name: asyncio.Semaphore(PER_VM_CONCURRENCY) for vm in vms}

    async with httpx.AsyncClient() as client:
        workers = [vm_worker(vm, per_vm_queues[vm.name], client,
                             vm_semaphores[vm.name], results_list)
                   for vm in vms if per_vm_queues[vm.name]]

        total_tasks = sum(len(q) for q in per_vm_queues.values())
        print(f"\nMemulai eksekusi {total_tasks} tugas dengan {len(workers)} worker VM (sequential per VM)...")
        t0 = time.monotonic()
        await asyncio.gather(*workers)
        makespan = time.monotonic() - t0
        print(f"\nSemua eksekusi tugas selesai dalam {makespan:.4f} detik.")

    # effective capacity = number of per-VM slots actually used in this pass
    effective_slots = sum(PER_VM_CONCURRENCY for _ in vms)

    metrics = calculate_and_print_metrics(results_list, makespan, effective_slots)
    return {
        "results": results_list,
        "metrics": metrics,
    }


async def run_round_robin_pass(tasks: List[Task], vms: List[VM]) -> Optional[Dict[str, object]]:
    if not vms:
        print("Tidak ada VM untuk menjalankan algoritma Round Robin.", file=sys.stderr)
        return None

    print("\n=== PASS C: Round Robin (penugasan siklik, paralel per core) ===")
    vm_cycle = [vm.name for vm in vms]
    assignment: Dict[int, str] = {}
    for idx, task in enumerate(tasks):
        assignment[task.id] = vm_cycle[idx % len(vm_cycle)]

    tasks_dict = {t.id: t for t in tasks}
    vms_dict = {vm.name: vm for vm in vms}

    results_list: List[dict] = []
    vm_semaphores = {vm.name: asyncio.Semaphore(vm.cpu_cores) for vm in vms}

    async with httpx.AsyncClient() as client:
        coros = [
            execute_task_on_vm(tasks_dict[tid], vms_dict[vm_name], client,
                               vm_semaphores[vm_name], results_list)
            for tid, vm_name in assignment.items()
        ]

        print(f"\nMemulai eksekusi {len(coros)} tugas dengan penjadwalan round robin...")
        t0 = time.monotonic()
        await asyncio.gather(*coros)
        makespan = time.monotonic() - t0
        print(f"\nSemua eksekusi Round Robin selesai dalam {makespan:.4f} detik.")

    effective_slots = sum(vm.cpu_cores for vm in vms)
    metrics = calculate_and_print_metrics(results_list, makespan, effective_slots)
    return {
        "results": results_list,
        "metrics": metrics,
    }


async def run_fcfs_pass(tasks: List[Task], vms: List[VM]) -> Optional[Dict[str, object]]:
    if not vms:
        print("Tidak ada VM untuk menjalankan algoritma FCFS.", file=sys.stderr)
        return None

    print("\n=== PASS D: FCFS (single queue, worker VM) ===")
    task_queue: asyncio.Queue[Optional[Task]] = asyncio.Queue()
    for task in tasks:
        task_queue.put_nowait(task)
    for _ in vms:
        task_queue.put_nowait(None)

    results_list: List[dict] = []
    vm_semaphores = {vm.name: asyncio.Semaphore(vm.cpu_cores) for vm in vms}

    async def worker(vm: VM, client: httpx.AsyncClient):
        while True:
            task = await task_queue.get()
            if task is None:
                task_queue.task_done()
                break
            await execute_task_on_vm(task, vm, client, vm_semaphores[vm.name], results_list)
            task_queue.task_done()

    async with httpx.AsyncClient() as client:
        worker_tasks = [asyncio.create_task(worker(vm, client)) for vm in vms]
        t0 = time.monotonic()
        await asyncio.gather(*worker_tasks)
        makespan = time.monotonic() - t0
        print(f"\nSemua eksekusi FCFS selesai dalam {makespan:.4f} detik.")

    effective_slots = sum(vm.cpu_cores for vm in vms)
    metrics = calculate_and_print_metrics(results_list, makespan, effective_slots)
    return {
        "results": results_list,
        "metrics": metrics,
    }


ALGORITHMS = [
    ("shc", run_shc_pass, SHC_RESULTS_FILE),
    ("j2020", run_j2020_pass, J2020_RESULTS_FILE),
    ("rr", run_round_robin_pass, RR_RESULTS_FILE),
    ("fcfs", run_fcfs_pass, FCFS_RESULTS_FILE),
]

# --- Main ---

async def main():
    vms = [VM(name, spec['ip'], spec['cpu'], spec['ram_gb']) for name, spec in VM_SPECS.items()]
    if not vms:
        print("Tidak ada VM yang terdefinisi pada konfigurasi.", file=sys.stderr)
        return

    # Reset log file
    _ensure_parent_dir(RUN_LOG_FILE)
    with open(RUN_LOG_FILE, 'w', encoding='utf-8') as _:
        pass
    _ensure_parent_dir(SUMMARY_FILE)
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as _:
        pass

    comparison_rows: List[Dict[str, object]] = []

    for algorithm_name, runner, result_file in ALGORITHMS:
        summary_rows: List[Dict[str, object]] = []
        for dataset_label, dataset_path in DATASET_FILES.items():
            metrics_runs: List[Dict[str, float]] = []
            for iteration in range(1, DATASET_ITERATIONS + 1):
                print(f"\n=== Menjalankan {algorithm_name.upper()} | Dataset: {dataset_label} | Iterasi {iteration}/{DATASET_ITERATIONS} ===")
                tasks = load_tasks(dataset_path)
                if not tasks:
                    print(f"Dataset {dataset_label} kosong, melewati iterasi ini.")
                    continue

                outcome = await runner(tasks, vms)
                if not outcome or not outcome.get('metrics'):
                    print(f"Tidak ada metrik untuk {algorithm_name} pada dataset {dataset_label} iterasi {iteration}.")
                    continue

                metrics = outcome['metrics']
                metrics_runs.append(metrics)

                append_log_entry(RUN_LOG_FILE, {
                    "timestamp": datetime.now().isoformat(),
                    "algorithm": algorithm_name,
                    "dataset": dataset_label,
                    "dataset_path": dataset_path,
                    "iteration": iteration,
                    "metrics": metrics,
                    "results": serialize_results_for_log(outcome['results']),
                })

                summary_iteration_label = f"{algorithm_name.upper()}-{dataset_label}-{iteration}"
                append_summary_entry(summary_iteration_label, metrics)

            if not metrics_runs:
                print(f"Tidak ada metrik terkumpul untuk {algorithm_name} dataset {dataset_label}. Melewati ringkasan.")
                continue

            avg_metrics = average_metric_records(metrics_runs)
            avg_metrics.update({
                "algorithm": algorithm_name,
                "dataset": dataset_label,
                "runs": len(metrics_runs),
            })
            summary_rows.append(avg_metrics)
            comparison_rows.append(build_comparison_row(algorithm_name, dataset_label, avg_metrics))

        write_average_metrics_csv(result_file, summary_rows)

    write_comparison_results(comparison_rows, COMPARISON_RESULTS_FILE)

if __name__ == "__main__":
    asyncio.run(main())
