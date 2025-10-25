"""Monitor ligero de recursos para ejecución remota por SSH.

Muestra uso de CPU, RAM del sistema y memoria de GPU a intervalos regulares.
Cuando está disponible NVML se reporta también el uso porcentual de cada GPU.
"""

import argparse
import os
import socket
import sys
import time
from typing import List

import psutil
import torch

try:
    import pynvml

    _NVML_AVAILABLE = True
    pynvml.nvmlInit()
except Exception:  # pragma: no cover - NVML opcional
    _NVML_AVAILABLE = False


def _format_bytes(num_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def _gpu_stats() -> List[str]:
    if not torch.cuda.is_available():
        return ["GPU no disponible (torch.cuda.is_available() == False)"]

    messages: List[str] = []
    device_count = torch.cuda.device_count()

    if _NVML_AVAILABLE:
        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            messages.append(
                f"GPU {index} ({name}) | Utilización: {util.gpu}% | "
                f"Memoria: {_format_bytes(mem_info.used)} / {_format_bytes(mem_info.total)}"
            )
    else:
        for index in range(device_count):
            props = torch.cuda.get_device_properties(index)
            allocated = torch.cuda.memory_allocated(index)
            reserved = torch.cuda.memory_reserved(index)
            messages.append(
                f"GPU {index} ({props.name}) | Memoria usada: {_format_bytes(allocated)} | "
                f"Reservada: {_format_bytes(reserved)}"
            )
        messages.append(
            "Instala 'nvidia-ml-py' para métricas completas de utilización (pip install nvidia-ml-py)."
        )

    return messages


def monitor_loop(interval: float, one_shot: bool = False) -> None:
    hostname = socket.gethostname()
    process = psutil.Process(os.getpid())

    print(f"Iniciando monitor en {hostname}. Intervalo: {interval}s", flush=True)

    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        proc_mem = process.memory_info().rss

        header = (
            f"[{timestamp}] CPU: {cpu_percent:.1f}% | RAM: {mem.percent:.1f}% "
            f"({_format_bytes(mem.used)} / {_format_bytes(mem.total)}) | "
            f"RAM proceso: {_format_bytes(proc_mem)}"
        )
        print(header, flush=True)

        for line in _gpu_stats():
            print(f"  - {line}", flush=True)

        if one_shot:
            break
        time.sleep(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor ligero de recursos")
    parser.add_argument("--interval", type=float, default=5.0, help="Intervalo entre muestras en segundos")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Imprime una única lectura y termina.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        monitor_loop(arguments.interval, arguments.once)
    except KeyboardInterrupt:
        print("Monitor detenido por el usuario", file=sys.stderr)
        sys.exit(0)
