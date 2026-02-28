import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "tinyaya-mimi-tts"
DATA_VOL_NAME = "tinyaya-mimi-tts-data"
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO_ROOT = "/root/repo"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(DATA_VOL_NAME, create_if_missing=True)
HF_SECRETS = [
    modal.Secret.from_name("datasynthgen-secrets"),
    modal.Secret.from_name("huggingface"),
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("huggingface-secret-nullhawk"),
    modal.Secret.from_name("hf"),
    modal.Secret.from_name("hf-token"),
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "torch",
        "torchaudio",
        "torchdata",
        "datasets",
        "blobfile",
        "tiktoken",
        "tabulate",
        "tyro",
        "soundfile",
        "librosa",
        "transformers",
        "accelerate",
        "moshi",
        "seaborn",
        "wandb",
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=REMOTE_REPO_ROOT,
        ignore=[".venv", ".git", "__pycache__", "outputs", "assets"],
    )
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 12,
    volumes={"/vol": volume},
    secrets=[*HF_SECRETS, modal.Secret.from_name("wandb")],
)
def pretokenize_fleurs(
    split: str = "train",
    languages: str = "en hi es fr de ar sw ta bn zh",
    quantizers: int = 1,
    max_samples_per_language: int = 0,
):
    out_dir = f"/vol/data/fleurs_pretok_q{quantizers}"
    cmd = [
        "python",
        "scripts/pretokenize_fleurs.py",
        "--languages",
        *languages.split(),
        "--split",
        split,
        "--num-quantizers",
        str(quantizers),
        "--output-dir",
        out_dir,
        "--max-samples-per-language",
        str(max_samples_per_language),
    ]
    subprocess.run(cmd, check=True, cwd=REMOTE_REPO_ROOT)
    volume.commit()
    return {"output_dir": out_dir, "split": split}


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    volumes={"/vol": volume},
    secrets=[*HF_SECRETS, modal.Secret.from_name("wandb")],
)
def train(path: str = "q1", experiment_id: str = "", steps: int = 0):
    config_map = {
        "q1": "config/tinyaya_q1_fleurs.toml",
        "q8": "config/tinyaya_q8_fleurs.toml",
        "overfit1": "config/tinyaya_q1_fleurs_overfit_1sample.toml",
        "overfit_smoke": "config/tinyaya_q1_fleurs_overfit_1sample_smoke.toml",
        "overfit_strict": "config/tinyaya_q1_fleurs_overfit_1sample_strict.toml",
        "overfit_viz5": "config/tinyaya_q1_fleurs_overfit_1sample_viz5.toml",
    }
    if path not in config_map:
        raise ValueError(
            "path must be one of: q1, q8, overfit1, overfit_smoke, overfit_strict, overfit_viz5"
        )
    config_file = config_map[path]
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "-m",
        "torchtitan.train",
        "--job.config_file",
        config_file,
        "--job.dump_folder",
        "/vol/outputs",
    ]
    if experiment_id:
        cmd.extend(["--experiment.id", experiment_id])
    if steps > 0:
        cmd.extend(["--training.steps", str(steps)])
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_API_TOKEN")
        or os.environ.get("HF_API_TOKEN")
        or os.environ.get("TOKEN")
    )
    wandb_key = (
        os.environ.get("WANDB_API_KEY")
        or os.environ.get("WANDB_KEY")
        or os.environ.get("WANDB_TOKEN")
        or os.environ.get("WANDB")
    )

    env = {"WANDB_PROJECT": "tinyaya-mimi-tts"}
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if wandb_key:
        env["WANDB_API_KEY"] = wandb_key

    subprocess.run(cmd, check=True, cwd=REMOTE_REPO_ROOT, env={**os.environ, **env})
    volume.commit()
    return {"status": "ok", "config": config_file}


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 30,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def infer(checkpoint_ref: str, text: str, lang: str = "en", num_quantizers: int = 1):
    logs_dir = Path("/vol/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(logs_dir / "infer_output.wav")
    cmd = [
        "python",
        "inference_tts.py",
        "--model-id",
        checkpoint_ref,
        "--text",
        text,
        "--lang",
        lang,
        "--num-quantizers",
        str(num_quantizers),
        "--output-file",
        output_file,
    ]
    subprocess.run(cmd, check=True, cwd=REMOTE_REPO_ROOT)
    volume.commit()
    return {"output_file": output_file}


@app.local_entrypoint()
def main(
    mode: str = "train",
    path: str = "overfit1",
    split: str = "train",
    quantizers: int = 1,
    experiment_id: str = "",
    steps: int = 0,
):
    if mode == "pretokenize":
        print(
            pretokenize_fleurs.remote(
                split=split,
                quantizers=quantizers,
            )
        )
        return
    if mode == "train":
        print(train.remote(path=path, experiment_id=experiment_id, steps=steps))
        return
    raise ValueError("mode must be one of: pretokenize, train")
