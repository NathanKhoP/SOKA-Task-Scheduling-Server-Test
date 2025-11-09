from flask import Flask, jsonify
import time
import os
from numpy.linalg import det
from numpy.random import rand
from datetime import date, datetime
from concurrent.futures import ProcessPoolExecutor

app = Flask(__name__)

CPU_CORES = os.cpu_count() or 1


def cpu_heavy_task(iterations: int):
    for _ in range(iterations):
        det(rand(40, 40))
    return True

def simulate_task(cpu_load: int):
    start_time = time.time()

    try:
        # --- CPU LOAD (parallelized) ---
        per_core_load = max(1, cpu_load // CPU_CORES)
        with ProcessPoolExecutor(max_workers=CPU_CORES) as executor:
            executor.map(cpu_heavy_task, [per_core_load] * CPU_CORES)
    except Exception as e:
        exec_time = time.time() - start_time
        return exec_time, f"CPU Task Error: {str(e)}"

    exec_time = time.time() - start_time
    return exec_time, None

@app.route("/health", methods=["GET"])
def health_check():
    current_date = date.today().strftime("%d-%m-%Y")
    current_time = datetime.now().strftime("%H:%M:%S")
    return jsonify({
        "status": True,
        "message": "🌟 Server is healthy!",
        "date": f"{current_date} {current_time}"
    }), 200

@app.route("/task/<task_size>", methods=["GET"])
def task_simulator_router(task_size):
    try:
        index = int(task_size)
    except ValueError:
        return jsonify({
            "status": False,
            "message": "Index must be a number between 1 - 10"
        }), 400

    if index < 1 or index > 10:
        return jsonify({
            "status": False,
            "message": "Index must be between 1 - 10"
        }), 400
    
    cpu_load = (index * index * 10000)
    exec_time, error_msg = simulate_task(cpu_load)

    if error_msg:
        return jsonify({
            "status": False,
            "message": error_msg,
            "task": f"task-{index}",
            "requested_cpu_load": cpu_load,
            "execution_time": f"{exec_time:.4f}s"
        }), 500

    return jsonify({
        "status": True,
        "message": f"task-{index} run successfully",
        "task": f"task-{index}",
        "requested_cpu_load": cpu_load,
        "execution_time": f"{exec_time:.4f}s"
    }), 200

if __name__ == "__main__":
    print(f"Server menggunakan {CPU_CORES} core CPU")
    app.run(host="0.0.0.0", port=5000)