"""
Environment Check — PEFT Experiment (23K-0068)
Detects hardware/software environment and saves results to results/logs/environment.json.
"""

import json
import os
import platform
import shutil
import sys
from pathlib import Path


def get_environment_info() -> dict:
    """Collect comprehensive environment information."""
    info = {
        "os": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
    }

    # PyTorch
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
        info["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem
            info["gpu_memory_gb"] = round(mem / (1024 ** 3), 2)
            info["gpu_capability"] = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
        else:
            info["gpu_name"] = None
            info["gpu_memory_gb"] = None
            info["gpu_capability"] = None
    except ImportError:
        info["torch_version"] = "NOT INSTALLED"
        info["cuda_available"] = False

    # Library versions
    for lib_name in ["transformers", "peft", "accelerate", "bitsandbytes", "datasets",
                     "trl", "evaluate", "scipy", "sklearn", "numpy", "pandas"]:
        try:
            mod = __import__(lib_name)
            info[f"{lib_name}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{lib_name}_version"] = "NOT INSTALLED"

    # System resources
    disk = shutil.disk_usage("/")
    info["disk_total_gb"] = round(disk.total / (1024 ** 3), 2)
    info["disk_free_gb"] = round(disk.free / (1024 ** 3), 2)

    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
        info["ram_available_gb"] = round(mem.available / (1024 ** 3), 2)
    except ImportError:
        # Fallback: read from /proc/meminfo on Linux (Colab)
        meminfo_path = "/proc/meminfo"
        if os.path.exists(meminfo_path):
            with open(meminfo_path) as f:
                lines = f.readlines()
            memdata = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    memdata[parts[0].strip()] = parts[1].strip()
            total_kb = int(memdata.get("MemTotal", "0").replace("kB", "").strip())
            avail_kb = int(memdata.get("MemAvailable", "0").replace("kB", "").strip())
            info["ram_total_gb"] = round(total_kb / (1024 ** 2), 2)
            info["ram_available_gb"] = round(avail_kb / (1024 ** 2), 2)
        else:
            info["ram_total_gb"] = None
            info["ram_available_gb"] = None

    return info


def print_environment(info: dict) -> None:
    """Pretty-print environment information."""
    print("=" * 60)
    print("  PEFT Experiment — Environment Check (23K-0068)")
    print("=" * 60)

    sections = [
        ("System", ["os", "python_version"]),
        ("PyTorch & CUDA", ["torch_version", "cuda_available", "cuda_version",
                            "cudnn_version", "gpu_count", "gpu_name",
                            "gpu_memory_gb", "gpu_capability"]),
        ("Libraries", [k for k in info if k.endswith("_version") and k != "torch_version"]),
        ("Resources", ["ram_total_gb", "ram_available_gb", "disk_total_gb", "disk_free_gb"]),
    ]

    for section_name, keys in sections:
        print(f"\n--- {section_name} ---")
        for key in keys:
            val = info.get(key, "N/A")
            label = key.replace("_", " ").title()
            print(f"  {label}: {val}")

    print("\n" + "=" * 60)

    # Readiness check
    if info.get("cuda_available"):
        print("✅ GPU detected — ready for PEFT training.")
    else:
        print("⚠️  No GPU detected — training will be slow on CPU.")


def save_environment(info: dict, output_path: str = "results/logs/environment.json") -> str:
    """Save environment info to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"\n📄 Environment saved to {output_path}")
    return output_path


def main():
    info = get_environment_info()
    print_environment(info)
    save_environment(info)
    return info


if __name__ == "__main__":
    main()
