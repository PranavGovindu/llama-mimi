from __future__ import annotations

import json
import sys

import torch
from kernels import get_kernel
from transformers import AutoModelForCausalLM


def main() -> int:
    model_id = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Qwen/Qwen2.5-0.5B"
    )
    backend = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "kernels-community/flash-attn3"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FA3 runtime probe.")

    kernel = get_kernel(backend, version=1)
    flash_attn_func = getattr(kernel, "flash_attn_func", None)
    if flash_attn_func is None:
        raise RuntimeError(f"{backend} did not expose flash_attn_func")

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=backend,
    ).eval().to(device)

    input_ids = torch.randint(
        low=0,
        high=int(model.config.vocab_size),
        size=(1, 32),
        device=device,
    )
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    payload = {
        "model_id": model_id,
        "backend": backend,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "kernel_module": getattr(getattr(kernel, "__class__", None), "__name__", None),
        "resolved_attn_implementation": getattr(
            getattr(model, "config", None),
            "_attn_implementation",
            None,
        ),
        "logits_shape": list(logits.shape),
        "head_dim": int(model.config.hidden_size // model.config.num_attention_heads),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
