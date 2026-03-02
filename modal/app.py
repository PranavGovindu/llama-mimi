import os
import re
import subprocess
import tomllib
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


def _safe_slug(value: str, max_len: int = 96) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    slug = slug.strip("._-")
    if not slug:
        return "run"
    return slug[:max_len]


def _load_run_name_defaults(config_file: str) -> dict[str, object]:
    cfg_path = Path(REMOTE_REPO_ROOT) / config_file
    if not cfg_path.exists():
        cfg_path = REPO_ROOT / config_file
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    model_name = str(raw.get("model", {}).get("name", "model")).split("/")[-1]
    dataset_name = str(raw.get("training", {}).get("dataset", "dataset"))
    seq_len = int(raw.get("training", {}).get("seq_len", 2048))
    pretrained = bool(raw.get("model", {}).get("pretrained", True))
    checkpoint_folder = str(raw.get("checkpoint", {}).get("folder", "checkpoint"))
    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "seq_len": seq_len,
        "pretrained": pretrained,
        "checkpoint_folder": checkpoint_folder,
    }


def _resolve_run_name(
    model_name: str,
    dataset_name: str,
    num_quantizers: int,
    seq_len: int,
    pretrained: bool,
    experiment_id: str,
) -> str:
    run_name = (
        f"{model_name}_{dataset_name}"
        f"-q{num_quantizers}"
        f"-s{seq_len}"
        f"{'-random' if not pretrained else ''}"
    )
    if experiment_id:
        run_name = f"{run_name}-{experiment_id}"
    return run_name


def _ensure_model_in_hf_collection(
    hf_repo_id: str,
    hf_repo_private: bool,
    hf_collection_slug: str,
    hf_token: str | None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "repo_id": hf_repo_id,
        "collection_slug": hf_collection_slug,
        "repo_ensured": False,
        "collection_item_added": False,
        "error": "",
    }
    if not hf_repo_id.strip():
        return result
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover
        result["error"] = f"huggingface_hub import failed: {exc}"
        return result

    api = HfApi(token=hf_token or None)
    try:
        api.create_repo(
            repo_id=hf_repo_id.strip(),
            repo_type="model",
            private=hf_repo_private,
            exist_ok=True,
        )
        result["repo_ensured"] = True
    except Exception as exc:
        result["error"] = f"create_repo failed: {exc}"
        return result

    if not hf_collection_slug.strip():
        return result

    add_fn = getattr(api, "add_collection_item", None)
    if add_fn is None:
        result["error"] = "huggingface_hub lacks add_collection_item API"
        return result

    try:
        try:
            add_fn(
                collection_slug=hf_collection_slug.strip(),
                item_id=hf_repo_id.strip(),
                item_type="model",
                exists_ok=True,
            )
        except TypeError:
            try:
                add_fn(
                    collection_slug=hf_collection_slug.strip(),
                    item_id=hf_repo_id.strip(),
                    item_type="model",
                )
            except TypeError:
                add_fn(hf_collection_slug.strip(), hf_repo_id.strip(), "model")
        result["collection_item_added"] = True
    except Exception as exc:
        result["error"] = f"add_collection_item failed: {exc}"
    return result


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
    timeout=60 * 60 * 2,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def pretokenize_single_wav(
    input_wav_path: str = "/vol/data/raw/download.wav",
    text: str = "",
    lang: str = "en",
    sample_id: str = "download_001",
    quantizers: int = 8,
    max_seconds: float = 20.0,
    output_dir: str = "/vol/data/custom_download_q8",
):
    cmd = [
        "python",
        "scripts/pretokenize_single_wav.py",
        "--input-wav",
        input_wav_path,
        "--output-dir",
        output_dir,
        "--split",
        "train",
        "--lang",
        lang,
        "--sample-id",
        sample_id,
        "--num-quantizers",
        str(quantizers),
        "--max-seconds",
        str(max_seconds),
    ]
    if text.strip():
        cmd.extend(["--text", text.strip()])
    subprocess.run(cmd, check=True, cwd=REMOTE_REPO_ROOT)
    volume.commit()
    return {
        "status": "ok",
        "output_dir": output_dir,
        "input_wav_path": input_wav_path,
        "quantizers": quantizers,
        "max_seconds": max_seconds,
        "sample_id": sample_id,
    }


@app.function(
    image=image,
    gpu="H200",
    timeout=60 * 60 * 24,
    volumes={"/vol": volume},
    secrets=[*HF_SECRETS, modal.Secret.from_name("wandb")],
)
def train(
    path: str = "q1",
    experiment_id: str = "",
    steps: int = 0,
    num_quantizers: int = 0,
    seed: int = -1,
    deterministic: bool = False,
    overfit_num_samples: int = 0,
    dataset_path: str = "",
    checkpoint_interval: int = 0,
    checkpoint_keep_latest_k: int = 0,
    checkpoint_folder: str = "",
    hf_repo_id: str = "",
    hf_repo_private: bool = False,
    hf_collection_slug: str = "",
    wandb_group: str = "",
    wandb_tags: str = "",
):
    config_map = {
        "q1": "config/tinyaya_q1_fleurs.toml",
        "q8": "config/tinyaya_q8_fleurs.toml",
        "overfit1": "config/tinyaya_q1_fleurs_overfit_1sample.toml",
        "overfit_smoke": "config/tinyaya_q1_fleurs_overfit_1sample_smoke.toml",
        "overfit_strict": "config/tinyaya_q1_fleurs_overfit_1sample_strict.toml",
        "overfit_viz5": "config/tinyaya_q1_fleurs_overfit_1sample_viz5.toml",
        "overfit_download_q8": "config/tinyaya_q8_download_overfit_1sample.toml",
    }
    if path not in config_map:
        raise ValueError(
            "path must be one of: q1, q8, overfit1, overfit_smoke, overfit_strict, overfit_viz5, overfit_download_q8"
        )
    config_file = config_map[path]
    run_defaults = _load_run_name_defaults(config_file)
    if int(num_quantizers) > 0:
        resolved_q = int(num_quantizers)
    else:
        cfg_path = Path(REMOTE_REPO_ROOT) / config_file
        if not cfg_path.exists():
            cfg_path = REPO_ROOT / config_file
        raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        resolved_q = int(raw.get("model", {}).get("num_quantizers", 1))

    run_name = _resolve_run_name(
        model_name=str(run_defaults["model_name"]),
        dataset_name=str(run_defaults["dataset_name"]),
        num_quantizers=resolved_q,
        seq_len=int(run_defaults["seq_len"]),
        pretrained=bool(run_defaults["pretrained"]),
        experiment_id=experiment_id.strip(),
    )

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
    if num_quantizers > 0:
        cmd.extend(["--model.num_quantizers", str(num_quantizers)])
    if seed >= 0:
        cmd.extend(["--training.seed", str(seed)])
    if deterministic:
        cmd.extend(["--training.deterministic", "true"])
    if overfit_num_samples > 0:
        cmd.extend(["--training.overfit_num_samples", str(overfit_num_samples)])
    if dataset_path.strip():
        cmd.extend(["--training.dataset_path", dataset_path.strip()])
    if checkpoint_interval > 0:
        cmd.extend(["--checkpoint.enable_checkpoint", "true"])
        cmd.extend(["--checkpoint.interval", str(checkpoint_interval)])
    if checkpoint_keep_latest_k > 0:
        cmd.extend(["--checkpoint.keep_latest_k", str(checkpoint_keep_latest_k)])
    if checkpoint_folder.strip():
        cmd.extend(["--checkpoint.folder", checkpoint_folder.strip()])
    elif checkpoint_interval > 0:
        auto_ckpt = f"checkpoint_{_safe_slug(experiment_id or run_name)}"
        cmd.extend(["--checkpoint.folder", auto_ckpt])
        checkpoint_folder = auto_ckpt

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
    if wandb_group.strip():
        env["WANDB_RUN_GROUP"] = wandb_group.strip()
    if wandb_tags.strip():
        env["WANDB_TAGS"] = wandb_tags.strip()

    hf_collection_sync: dict[str, object] = {}
    if hf_repo_id.strip():
        hf_collection_sync = _ensure_model_in_hf_collection(
            hf_repo_id=hf_repo_id.strip(),
            hf_repo_private=hf_repo_private,
            hf_collection_slug=hf_collection_slug.strip(),
            hf_token=hf_token,
        )

    uploader_proc = None
    if hf_repo_id.strip():
        resolved_checkpoint_folder = checkpoint_folder.strip() or str(
            run_defaults["checkpoint_folder"]
        )
        ckpt_dir = Path("/vol/outputs") / run_name / resolved_checkpoint_folder
        upload_cmd = [
            "python",
            "scripts/exp/upload_checkpoints_hf.py",
            "--checkpoint-dir",
            str(ckpt_dir),
            "--repo-id",
            hf_repo_id.strip(),
            "--poll-seconds",
            "30",
            "--idle-exit-seconds",
            "900",
        ]
        if hf_repo_private:
            upload_cmd.append("--private")
        uploader_proc = subprocess.Popen(
            upload_cmd,
            cwd=REMOTE_REPO_ROOT,
            env={**os.environ, **env},
        )

    try:
        subprocess.run(cmd, check=True, cwd=REMOTE_REPO_ROOT, env={**os.environ, **env})
    finally:
        if uploader_proc is not None:
            try:
                uploader_proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                uploader_proc.terminate()
                try:
                    uploader_proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    uploader_proc.kill()

    volume.commit()
    return {
        "status": "ok",
        "config": config_file,
        "run_name": run_name,
        "num_quantizers": resolved_q,
        "checkpoint_folder": checkpoint_folder,
        "hf_repo_id": hf_repo_id,
        "hf_collection_slug": hf_collection_slug,
        "hf_collection_sync": hf_collection_sync,
    }


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


@app.function(
    image=image,
    timeout=60 * 10,
    secrets=HF_SECRETS,
)
def create_hf_collection(
    title: str,
    namespace: str = "rumik-ai",
    description: str = "",
    private: bool = False,
):
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_API_TOKEN")
        or os.environ.get("HF_API_TOKEN")
        or os.environ.get("TOKEN")
    )
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token or None)
    create_fn = getattr(api, "create_collection", None)
    if create_fn is None:
        raise RuntimeError("huggingface_hub does not expose create_collection")

    try:
        coll = create_fn(
            title=title.strip(),
            namespace=namespace.strip(),
            description=description.strip(),
            private=private,
        )
    except TypeError:
        coll = create_fn(title.strip(), namespace.strip(), description.strip(), private)

    slug = ""
    if isinstance(coll, dict):
        slug = str(coll.get("slug") or coll.get("id") or "").strip()
    else:
        slug = str(getattr(coll, "slug", "") or getattr(coll, "id", "")).strip()
    if not slug:
        raise RuntimeError(f"Unable to resolve collection slug from response: {coll!r}")

    print(f"HF_COLLECTION_SLUG={slug}", flush=True)
    return {"slug": slug}


@app.local_entrypoint()
def main(
    mode: str = "train",
    path: str = "overfit1",
    split: str = "train",
    quantizers: int = 1,
    experiment_id: str = "",
    steps: int = 0,
    input_wav_path: str = "/vol/data/raw/download.wav",
    text: str = "",
    lang: str = "en",
    sample_id: str = "download_001",
    max_seconds: float = 20.0,
):
    if mode == "pretokenize":
        print(
            pretokenize_fleurs.remote(
                split=split,
                quantizers=quantizers,
            )
        )
        return
    if mode == "pretokenize_single":
        print(
            pretokenize_single_wav.remote(
                input_wav_path=input_wav_path,
                text=text,
                lang=lang,
                sample_id=sample_id,
                quantizers=quantizers,
                max_seconds=max_seconds,
            )
        )
        return
    if mode == "train":
        print(train.remote(path=path, experiment_id=experiment_id, steps=steps))
        return
    raise ValueError("mode must be one of: pretokenize, pretokenize_single, train")
