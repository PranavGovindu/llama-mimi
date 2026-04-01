from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "tinyaya-tts-lab-train-fa3"
DATA_VOL_NAME = os.environ.get("DATA_VOL_NAME", "tinyaya-mimi-tts-data")
DEFAULT_DATASET_REPO = "Pranavz/emilia-en-mimi-q8-s4096-dynamic-20260329a-public"
DEFAULT_DATASET_DIR = "/vol/data/datasets/emilia-en-mimi-q8-s4096-dynamic-20260329a-public"
DEFAULT_CONFIG_FILE = (
    "recipes/tinyaya/mimi/train/tinyaya_mimi_q8_s4096_emilia40k_en_clone_flat.toml"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DOCKERFILE_PATH = REPO_ROOT / "Dockerfile.train.base"
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile.train"
WANDB_SECRET_NAME = "wandb"
HF_SECRET_NAMES = (
    "datasynthgen-secrets",
    "huggingface",
    "huggingface-secret",
    "huggingface-secret-nullhawk",
    "hf",
    "hf-token",
)


def _build_base_image() -> tuple[modal.Image, dict[str, str]]:
    return (
        modal.Image.from_dockerfile(
            str(BASE_DOCKERFILE_PATH),
            context_dir=str(REPO_ROOT),
        ),
        {"source": "dockerfile", "ref": str(BASE_DOCKERFILE_PATH)},
    )


def _build_train_image() -> tuple[modal.Image, dict[str, str]]:
    registry_image = os.environ.get("MODAL_TRAIN_REGISTRY_IMAGE", "").strip()
    registry_secret_name = os.environ.get("MODAL_TRAIN_REGISTRY_SECRET", "").strip()
    if registry_image:
        secret = (
            modal.Secret.from_name(registry_secret_name)
            if registry_secret_name
            else None
        )
        return (
            modal.Image.from_registry(registry_image, secret=secret),
            {"source": "registry", "ref": registry_image},
        )

    build_args = {"FA3_MAX_JOBS": os.environ.get("FA3_MAX_JOBS", "4")}
    return (
        modal.Image.from_dockerfile(
            str(DOCKERFILE_PATH),
            context_dir=str(REPO_ROOT),
            build_args=build_args,
        ),
        {
            "source": "dockerfile",
            "ref": str(DOCKERFILE_PATH),
            "fa3_max_jobs": build_args["FA3_MAX_JOBS"],
        },
    )


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(DATA_VOL_NAME, create_if_missing=True)
HF_SECRETS = [modal.Secret.from_name(name) for name in HF_SECRET_NAMES]
BASE_IMAGE, BASE_IMAGE_INFO = _build_base_image()
TRAIN_IMAGE, TRAIN_IMAGE_INFO = _build_train_image()


def _runtime_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": "/vol/cache/huggingface",
        "HF_HUB_CACHE": "/vol/cache/huggingface/hub",
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "tinyaya-tts-lab"),
        "WANDB_DIR": "/vol/cache/wandb",
        "WANDB_ARTIFACT_DIR": "/vol/cache/wandb/artifacts",
        "TORCH_HOME": "/vol/cache/torch",
        "TMPDIR": "/vol/cache/tmp",
        "OMP_NUM_THREADS": "1",
    }
    if extra:
        env.update({k: v for k, v in extra.items() if v is not None})
    return env


def _run_json_command(cmd: list[str], extra_env: dict[str, str] | None = None) -> dict:
    output = subprocess.check_output(cmd, text=True, env=_runtime_env(extra_env))
    print(output, flush=True)
    return json.loads(output)


def _parse_extra_args(extra_args_json: str) -> list[str]:
    if not extra_args_json.strip():
        return []
    payload = json.loads(extra_args_json)
    if not isinstance(payload, list) or any(not isinstance(v, str) for v in payload):
        raise ValueError("extra_args_json must be a JSON array of strings")
    return payload


@app.function(
    image=BASE_IMAGE,
    gpu="H200",
    timeout=60 * 20,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def probe_ngc_stack() -> dict:
    payload = _run_json_command(
        ["/workspace/scripts/train_entrypoint.sh", "probe-ngc-stack"]
    )
    payload["image"] = BASE_IMAGE_INFO
    return payload


@app.function(
    image=TRAIN_IMAGE,
    gpu="H200",
    timeout=60 * 20,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def probe_fa3(
    model_id: str = "Qwen/Qwen2.5-0.5B",
) -> dict:
    payload = _run_json_command(
        ["/workspace/scripts/train_entrypoint.sh", "probe-fa3", model_id]
    )
    payload["image"] = TRAIN_IMAGE_INFO
    return payload


@app.function(
    image=BASE_IMAGE,
    timeout=60 * 60,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def sync_dataset(
    repo_id: str = DEFAULT_DATASET_REPO,
    local_dir: str = DEFAULT_DATASET_DIR,
    revision: str = "",
) -> dict:
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "/workspace/scripts/train_entrypoint.sh",
        "sync-dataset",
        "--repo-id",
        repo_id,
        "--local-dir",
        local_dir,
    ]
    if revision.strip():
        cmd.extend(["--revision", revision.strip()])
    payload = _run_json_command(cmd)
    volume.commit()
    payload["image"] = BASE_IMAGE_INFO
    return payload


@app.function(
    image=TRAIN_IMAGE,
    gpu="H200",
    timeout=60 * 60 * 24,
    volumes={"/vol": volume},
    secrets=[*HF_SECRETS, modal.Secret.from_name(WANDB_SECRET_NAME)],
)
def train(
    config_file: str = DEFAULT_CONFIG_FILE,
    dataset_path: str = DEFAULT_DATASET_DIR,
    steps: int = 0,
    experiment_id: str = "",
    attn_implementation: str = "kernels-community/flash-attn3",
    dump_folder: str = "/vol/outputs",
    checkpoint_folder: str = "",
    checkpoint_interval: int = 0,
    checkpoint_keep_latest_k: int = 0,
    extra_args_json: str = "",
    nproc_per_node: int = 1,
) -> dict:
    Path(dump_folder).mkdir(parents=True, exist_ok=True)
    env = _runtime_env({"NPROC_PER_NODE": str(max(1, int(nproc_per_node)))})
    cmd = [
        "/workspace/scripts/train_entrypoint.sh",
        "train",
        "--job.config_file",
        config_file,
        "--job.dump_folder",
        dump_folder,
        "--training.dataset_path",
        dataset_path,
    ]
    if steps > 0:
        cmd.extend(["--training.steps", str(steps)])
    if experiment_id.strip():
        cmd.extend(["--experiment.id", experiment_id.strip()])
    if attn_implementation.strip():
        cmd.extend(["--model.attn_implementation", attn_implementation.strip()])
    if checkpoint_interval > 0:
        cmd.extend(["--checkpoint.enable_checkpoint", "true"])
        cmd.extend(["--checkpoint.interval", str(checkpoint_interval)])
    if checkpoint_keep_latest_k > 0:
        cmd.extend(["--checkpoint.keep_latest_k", str(checkpoint_keep_latest_k)])
    if checkpoint_folder.strip():
        cmd.extend(["--checkpoint.folder", checkpoint_folder.strip()])
    cmd.extend(_parse_extra_args(extra_args_json))

    print(json.dumps({"cmd": cmd, "image": TRAIN_IMAGE_INFO}, indent=2), flush=True)
    subprocess.run(cmd, check=True, env=env)
    volume.commit()
    return {
        "status": "ok",
        "config_file": config_file,
        "dataset_path": dataset_path,
        "dump_folder": dump_folder,
        "steps": steps,
        "attn_implementation": attn_implementation,
        "image": TRAIN_IMAGE_INFO,
    }
