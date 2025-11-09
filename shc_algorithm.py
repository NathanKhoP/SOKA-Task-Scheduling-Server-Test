import random
from collections import namedtuple

VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load', 'ram_mb'])

# --- Algoritma Stochastic Hill Climbing ---

def calculate_estimated_makespan(solution: dict, tasks_dict: dict, vms_dict: dict) -> float:
    """
    Fungsi Biaya (Cost Function).
    Memperkirakan makespan (waktu selesai maks) untuk solusi tertentu.
    Model sederhana: makespan = max(total_beban_cpu_vm / core_vm)
    """
    vm_loads = {vm.name: 0.0 for vm in vms_dict.values()}
    
    for task_id, vm_name in solution.items():
        task = tasks_dict[task_id]
        vm = vms_dict[vm_name]
        
        # Estimasi waktu eksekusi: beban / jumlah core
        # Ini model yang sangat sederhana, tapi umum digunakan
        estimated_time = task.cpu_load / vm.cpu_cores
        vm_loads[vm_name] += estimated_time
        
    # Makespan adalah VM yang paling lama selesai
    return max(vm_loads.values())

def get_random_neighbor(solution: dict, vm_names: list) -> dict:
    """
    Membuat solusi 'tetangga' dengan memindahkan satu tugas acak
    ke VM acak yang berbeda.
    """
    new_solution = solution.copy()
    
    # Pilih tugas acak untuk dipindah
    task_id_to_move = random.choice(list(new_solution.keys()))
    current_vm = new_solution[task_id_to_move]
    
    # Pilih VM baru (pastikan berbeda)
    possible_new_vms = [vm for vm in vm_names if vm != current_vm]
    if not possible_new_vms:
        return new_solution # Terjadi jika hanya ada 1 VM
        
    new_vm = random.choice(possible_new_vms)
    
    # Pindahkan tugas
    new_solution[task_id_to_move] = new_vm
    return new_solution

def stochastic_hill_climb(tasks: list[Task], vms: list[VM], iterations: int) -> dict:
    """Menjalankan algoritma SHC untuk menemukan solusi (penugasan) terbaik."""
    
    print(f"Memulai Stochastic Hill Climbing ({iterations} iterasi)...")
    
    vms_dict = {vm.name: vm for vm in vms}
    tasks_dict = {task.id: task for task in tasks}
    vm_names = list(vms_dict.keys())

    # 1. Buat Solusi Awal (Acak)
    current_solution = {}
    for task in tasks:
        current_solution[task.id] = random.choice(vm_names)
        
    current_cost = calculate_estimated_makespan(current_solution, tasks_dict, vms_dict)
    
    best_solution = current_solution
    best_cost = current_cost
    
    print(f"Estimasi Makespan Awal (Acak): {best_cost:.2f}")

    # 2. Iterasi SHC
    for i in range(iterations):
        # Buat tetangga (neighbor)
        neighbor_solution = get_random_neighbor(current_solution, vm_names)
        neighbor_cost = calculate_estimated_makespan(neighbor_solution, tasks_dict, vms_dict)
        
        # 3. Bandingkan
        # Ini adalah "Hill Climbing" sederhana, hanya menerima yang lebih baik
        if neighbor_cost < current_cost:
            current_solution = neighbor_solution
            current_cost = neighbor_cost
            
            # Perbarui solusi terbaik jika ditemukan
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution
                print(f"Iterasi {i}: Estimasi Makespan Baru: {best_cost:.2f}")

    print(f"SHC Selesai. Estimasi Makespan Terbaik: {best_cost:.2f}")
    return best_solution