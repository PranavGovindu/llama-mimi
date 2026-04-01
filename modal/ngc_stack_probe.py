from __future__ import annotations

import modal


APP_NAME = "tinyaya-mimi-tts-ngc-stack-probe"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.02-py3")
    .apt_install("build-essential", "git")
    .pip_install(
        "transformers==4.57.6",
        "accelerate==1.12.0",
        "huggingface_hub==0.36.2",
        "packaging",
        "psutil",
        "ninja",
        "wheel",
    )
)


@app.function(image=image, gpu="H200", timeout=60 * 10)
def probe_ngc_torch_stack():
    import json
    import os
    import subprocess

    import torch

    payload = {
        "python": subprocess.check_output(["python", "-V"], text=True).strip(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_home": os.environ.get("CUDA_HOME", ""),
        "nvcc_version": subprocess.check_output(
            ["bash", "-lc", "nvcc --version | tail -n 1"], text=True
        ).strip(),
        "gpu_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        payload["gpu_name"] = torch.cuda.get_device_name(0)
        payload["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload
