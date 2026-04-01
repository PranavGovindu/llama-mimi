from __future__ import annotations

import json
import os
import subprocess
import sys

import torch


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def main() -> int:
    payload = {
        "python": _run(["python", "-V"]),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_home": os.environ.get("CUDA_HOME", ""),
        "nvcc_version": _run(["bash", "-lc", "nvcc --version | tail -n 1"]),
        "gpu_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        payload["gpu_name"] = torch.cuda.get_device_name(0)
        payload["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
