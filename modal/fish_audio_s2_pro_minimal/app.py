import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal
import requests


APP_NAME = "fish-audio-s2-pro-minimal"
RUNTIME_VOL_NAME = "fishaudio-s2pro-runtime"
DATA_VOL_NAME = os.environ.get("DATA_VOL_NAME", "tinyaya-mimi-tts-data")
UPSTREAM_REPO = "https://github.com/sgl-project-dev/sglang-omni.git"
UPSTREAM_DIR = Path("/models/sglang-omni")
UPSTREAM_VENV = UPSTREAM_DIR / ".venv"
UPSTREAM_PYTHON = UPSTREAM_VENV / "bin" / "python"
UPSTREAM_HF_CLI = UPSTREAM_VENV / "bin" / "huggingface-cli"
MODEL_DIR = Path("/models/s2-pro")
CONFIG_PATH = UPSTREAM_DIR / "examples" / "configs" / "s2pro_tts.yaml"
REF_WAV_PATH = Path("/vol/data/raw/SP_SP010_1.wav")
OUTPUT_DIR = Path("/vol/outputs/fish_audio_s2_pro_minimal/sp_sp010_hindi_demo")

REF_TEXT = (
    "Oh god, I'm just so happy. Oh, and it's all your fault. Oh honestly, probably still your house. "
    "But still I mean running the dishes through the dishwasher, putting them up. Yeah yeah, alright, "
    "alright yeah, okay. I guess I do have a lot of explaining to do don't I? Huh, but feeding off of "
    "other people just feels weird. Like, like I'm cheating you know. You're sorry? Oh baby I'm sorry too. "
    "I'm sorry this whole thing happened."
)

DEVANAGARI_TEXT = (
    "आज शाम मौसम बहुत सुहावना है, हल्की ठंडी हवा चल रही है और दूर कहीं से चाय की खुशबू आ रही है। "
    "मैं बस थोड़ा टहलने निकला हूँ और सोच रहा हूँ कि ज़िंदगी में कभी-कभी धीरे चलना भी ज़रूरी होता है।"
)

LATIN_TEXT = (
    "Aaj shaam mausam bahut suhaavna hai, halki thandi hawa chal rahi hai aur door kahin se chai ki khushboo "
    "aa rahi hai. Main bas thoda tahalne nikla hoon aur soch raha hoon ki zindagi mein kabhi-kabhi dheere "
    "chalna bhi zaroori hota hai."
)

app = modal.App(APP_NAME)
runtime_volume = modal.Volume.from_name(RUNTIME_VOL_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOL_NAME, create_if_missing=True)
secrets = [
    modal.Secret.from_name("datasynthgen-secrets"),
    modal.Secret.from_name("huggingface"),
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("huggingface-secret-nullhawk"),
    modal.Secret.from_name("hf"),
    modal.Secret.from_name("hf-token"),
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "libcurl4")
    .pip_install("uv", "huggingface_hub", "requests")
)


def _hf_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
    ):
        token = os.environ.get(key)
        if token:
            env["HF_TOKEN"] = token
            env["HUGGINGFACE_HUB_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token
            env["HUGGINGFACE_TOKEN"] = token
            break
    return env


def _run_bash(command: str, *, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        ["/bin/bash", "-lc", command],
        check=True,
        env=env,
    )


def _progress(progress_path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    print(message, flush=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _ensure_runtime() -> None:
    env = _hf_env()
    if not UPSTREAM_PYTHON.exists():
        print("Runtime venv missing; cloning sglang-omni and installing [s2pro]...", flush=True)
        if UPSTREAM_DIR.exists():
            shutil.rmtree(UPSTREAM_DIR)
        _run_bash(
            f"git clone --depth 1 {UPSTREAM_REPO} {UPSTREAM_DIR}",
            env=env,
        )
        _run_bash(
            f"cd {UPSTREAM_DIR} && uv venv .venv -p 3.12 && source .venv/bin/activate && uv pip install -v '.[s2pro]'",
            env=env,
        )
        runtime_volume.commit()
        print("Runtime venv ready.", flush=True)
    else:
        print("Reusing cached sglang-omni runtime.", flush=True)
    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        print("Model weights missing; downloading fishaudio/s2-pro...", flush=True)
        _run_bash(
            f"{UPSTREAM_HF_CLI} download fishaudio/s2-pro --local-dir {MODEL_DIR}",
            env=env,
        )
        runtime_volume.commit()
        print("Model download complete.", flush=True)
    else:
        print("Reusing cached fishaudio/s2-pro weights.", flush=True)


def _wait_for_health(base_url: str, server_proc: subprocess.Popen[str], server_log_path: Path) -> None:
    for _ in range(180):
        if server_proc.poll() is not None:
            logs = server_log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError("S2 Pro server exited before healthy.\n" + logs[-20000:])
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.ok:
                payload = resp.json()
                if payload.get("status") == "healthy":
                    return
        except Exception:
            pass
        time.sleep(5)
    logs = server_log_path.read_text(encoding="utf-8", errors="replace")
    raise TimeoutError("Timed out waiting for S2 Pro health.\n" + logs[-20000:])


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={"/models": runtime_volume, "/vol": data_volume},
    secrets=secrets,
)
def generate_demo():
    if not REF_WAV_PATH.exists():
        raise FileNotFoundError(f"Missing reference WAV at {REF_WAV_PATH}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server_log_path = OUTPUT_DIR / "server.log"
    progress_path = OUTPUT_DIR / "progress.log"
    _progress(progress_path, f"Starting generate_demo with ref wav: {REF_WAV_PATH}")

    _ensure_runtime()
    _progress(progress_path, f"Output dir prepared at: {OUTPUT_DIR}")

    env = _hf_env()
    env["PYTHONUNBUFFERED"] = "1"

    server_cmd = [
        str(UPSTREAM_PYTHON),
        "-m",
        "sglang_omni.cli.cli",
        "serve",
        "--model-path",
        str(MODEL_DIR),
        "--config",
        str(CONFIG_PATH),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]

    with server_log_path.open("w", encoding="utf-8") as server_log:
        _progress(progress_path, "Launching S2 Pro server...")
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(UPSTREAM_DIR),
            text=True,
        )
        try:
            base_url = "http://127.0.0.1:8000"
            _wait_for_health(base_url, server_proc, server_log_path)
            _progress(progress_path, "S2 Pro server is healthy.")

            outputs: dict[str, str] = {}
            for name, text in (
                ("devanagari_hindi", DEVANAGARI_TEXT),
                ("latin_hindi", LATIN_TEXT),
            ):
                _progress(progress_path, f"Generating: {name}")
                payload = {
                    "input": text,
                    "max_new_tokens": 512,
                    "references": [
                        {
                            "audio_path": str(REF_WAV_PATH),
                            "text": REF_TEXT,
                        }
                    ],
                }
                response = requests.post(
                    f"{base_url}/v1/audio/speech",
                    json=payload,
                    timeout=1800,
                )
                response.raise_for_status()
                out_path = OUTPUT_DIR / f"{name}.wav"
                out_path.write_bytes(response.content)
                outputs[name] = str(out_path)
                _progress(progress_path, f"Saved {name} to {out_path}")

            metadata = {
                "status": "ok",
                "ref_wav": str(REF_WAV_PATH),
                "outputs": outputs,
                "server_log_path": str(server_log_path),
                "config": str(CONFIG_PATH),
                "model_dir": str(MODEL_DIR),
            }
            (OUTPUT_DIR / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            data_volume.commit()
            _progress(progress_path, "All outputs committed to Modal volume.")
            return metadata
        finally:
            if server_proc.poll() is None:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server_proc.kill()
                    server_proc.wait(timeout=30)
