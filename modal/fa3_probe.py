from __future__ import annotations

import json
import os
import time

import modal


APP_NAME = "tinyaya-mimi-tts-fa3-probe"
HF_SECRET_NAMES = (
    "datasynthgen-secrets",
    "huggingface",
    "huggingface-secret",
    "huggingface-secret-nullhawk",
    "hf",
    "hf-token",
)

app = modal.App(APP_NAME)
HF_SECRETS = [modal.Secret.from_name(name) for name in HF_SECRET_NAMES]

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
        "kernels==0.12.3",
    )
)


@app.function(image=image, gpu="H200", timeout=60 * 20, secrets=HF_SECRETS)
def probe_flash_attention_3(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    backend: str = "kernels-community/flash-attn3",
):
    import torch
    from kernels import get_kernel
    from transformers import AutoModelForCausalLM

    kernel = get_kernel(backend, version=1)
    flash_attn_func = getattr(kernel, "flash_attn_func", None)
    if flash_attn_func is None:
        raise RuntimeError(f"{backend} did not expose flash_attn_func")

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
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
        "resolved_attn_implementation": getattr(
            getattr(model, "config", None),
            "_attn_implementation",
            None,
        ),
        "logits_shape": list(logits.shape),
        "head_dim": int(model.config.hidden_size // model.config.num_attention_heads),
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


@app.function(image=image, gpu="H200", timeout=60 * 30, secrets=HF_SECRETS)
def benchmark_attention_backends(
    model_id: str = "CohereLabs/tiny-aya-fire",
    seq_len: int = 2048,
    batch_size: int = 1,
    warmup_steps: int = 1,
    measure_steps: int = 3,
) -> dict:
    import gc

    import torch
    from kernels import get_kernel
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for backend benchmark.")

    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or None
    )

    device = torch.device("cuda")
    backends = [
        "sdpa",
        "kernels-community/flash-attn3",
    ]
    results: list[dict[str, object]] = []

    for backend in backends:
        if backend.startswith("kernels-community/"):
            kernel = get_kernel(backend, version=1)
            if getattr(kernel, "flash_attn_func", None) is None:
                raise RuntimeError(f"{backend} did not expose flash_attn_func")
        else:
            kernel = None

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            dtype=torch.bfloat16,
            attn_implementation=backend,
        ).train().to(device)
        model.config.use_cache = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

        vocab_size = int(model.config.vocab_size)
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )
        labels = input_ids.clone()

        for _ in range(max(0, warmup_steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        step_times: list[float] = []
        losses: list[float] = []
        for _ in range(max(1, measure_steps)):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - start)
            losses.append(float(loss.detach().cpu()))

        avg_step_s = sum(step_times) / len(step_times)
        tokens_per_step = batch_size * seq_len
        result = {
            "backend": backend,
            "kernel_loaded": bool(kernel is not None),
            "resolved_attn_implementation": getattr(
                getattr(model, "config", None),
                "_attn_implementation",
                None,
            ),
            "head_dim": int(model.config.hidden_size // model.config.num_attention_heads),
            "avg_step_s": avg_step_s,
            "tokens_per_step": tokens_per_step,
            "tokens_per_second": tokens_per_step / avg_step_s,
            "mean_loss": sum(losses) / len(losses),
        }
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        results.append(result)

        del optimizer
        del model
        gc.collect()
        torch.cuda.empty_cache()

    by_backend = {row["backend"]: row for row in results}
    fa2 = by_backend.get("sdpa")
    fa3 = by_backend.get("kernels-community/flash-attn3")
    speedup = None
    if fa2 and fa3:
        fa2_tps = float(fa2["tokens_per_second"])
        fa3_tps = float(fa3["tokens_per_second"])
        if fa2_tps > 0:
            speedup = fa3_tps / fa2_tps

    payload = {
        "model_id": model_id,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "results": results,
        "fa3_vs_sdpa_speedup": speedup,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload
