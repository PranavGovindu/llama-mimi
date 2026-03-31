import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

import modal


APP_NAME = "fish-audio-s2-pro"
DATA_VOL_NAME = "tinyaya-mimi-tts-data"
UPSTREAM_REPO = "https://github.com/sgl-project-dev/sglang-omni.git"
UPSTREAM_REPO_DIR = Path("/root/sglang-omni")
UPSTREAM_VENV_DIR = UPSTREAM_REPO_DIR / ".venv"
UPSTREAM_PYTHON = UPSTREAM_VENV_DIR / "bin" / "python"
UPSTREAM_HF_CLI = UPSTREAM_VENV_DIR / "bin" / "huggingface-cli"
S2PRO_CONFIG_PATH = UPSTREAM_REPO_DIR / "examples" / "configs" / "s2pro_tts.yaml"

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
    modal.Image.from_registry("frankleeeee/sglang-omni:dev")
    .apt_install("git")
    .run_commands(
        f"rm -rf {UPSTREAM_REPO_DIR} && git clone --depth 1 {UPSTREAM_REPO} {UPSTREAM_REPO_DIR}",
        f"cd {UPSTREAM_REPO_DIR} && uv venv .venv -p 3.12 && . .venv/bin/activate && uv pip install -v '.[s2pro]'",
    )
)


def _resolve_hf_token() -> str | None:
    for key in (
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
    ):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _ensure_upstream_repo() -> Path:
    if not S2PRO_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing S2 Pro config: {S2PRO_CONFIG_PATH}")
    if not UPSTREAM_PYTHON.exists():
        raise FileNotFoundError(f"Missing upstream venv python: {UPSTREAM_PYTHON}")
    return UPSTREAM_REPO_DIR


def _download_model(checkpoint_repo_id: str, checkpoint_dir: Path, hf_token: str | None) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    subprocess.run(
        [
            str(UPSTREAM_HF_CLI),
            "download",
            checkpoint_repo_id,
            "--local-dir",
            str(checkpoint_dir),
        ],
        env=env,
        check=True,
    )


def _post_json_bytes(url: str, payload: dict[str, object], timeout_s: float) -> bytes:
    req = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def _get_health(url: str, timeout_s: float) -> str:
    with urllib_request.urlopen(url, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def reference_tts(
    input_wav_path: str = "/vol/data/raw/reference.wav",
    prompt_text: str = "",
    devanagari_text: str = "",
    latin_text: str = "",
    checkpoint_repo_id: str = "fishaudio/s2-pro",
    checkpoint_dir: str = "/vol/checkpoints/fishaudio_s2_pro",
    output_dir: str = "/vol/outputs/fish_audio_s2_pro/reference_demo",
    host: str = "127.0.0.1",
    port: int = 8000,
):
    input_path = Path(input_wav_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Reference WAV not found: {input_wav_path}")
    if not prompt_text.strip():
        raise ValueError("prompt_text must be provided for FishAudio S2 Pro voice cloning.")
    if not devanagari_text.strip():
        raise ValueError("devanagari_text must be non-empty.")
    if not latin_text.strip():
        raise ValueError("latin_text must be non-empty.")

    repo_dir = _ensure_upstream_repo()
    hf_token = _resolve_hf_token()
    ckpt_dir = Path(checkpoint_dir)
    _download_model(checkpoint_repo_id, ckpt_dir, hf_token)
    volume.commit()

    out_dir = Path(output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = out_dir / "server.log"

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{repo_dir}:{env.get('PYTHONPATH', '')}".rstrip(":")
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    server_cmd = [
        str(UPSTREAM_PYTHON),
        "-m",
        "sglang_omni.cli.cli",
        "serve",
        "--model-path",
        checkpoint_repo_id,
        "--config",
        str(S2PRO_CONFIG_PATH),
        "--host",
        host,
        "--port",
        str(port),
    ]

    with server_log_path.open("w", encoding="utf-8") as server_log:
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(repo_dir),
        )
        try:
            health_url = f"http://{host}:{port}/health"
            for _ in range(120):
                if server_proc.poll() is not None:
                    server_log.flush()
                    logs = server_log_path.read_text(encoding="utf-8", errors="replace")
                    raise RuntimeError(
                        "S2 Pro server exited before becoming healthy.\n"
                        + logs[-16000:]
                    )
                try:
                    body = _get_health(health_url, timeout_s=5.0)
                    if "ok" in body.lower() or "healthy" in body.lower() or "true" in body.lower():
                        break
                except Exception:
                    pass
                time.sleep(5)
            else:
                server_log.flush()
                logs = server_log_path.read_text(encoding="utf-8", errors="replace")
                raise TimeoutError(
                    "Timed out waiting for S2 Pro server health.\n"
                    + logs[-16000:]
                )

            outputs: dict[str, str] = {}
            request_logs: list[dict[str, object]] = []
            for name, text in (
                ("devanagari_hindi", devanagari_text.strip()),
                ("latin_hindi", latin_text.strip()),
            ):
                payload = {
                    "input": text,
                    "max_new_tokens": 512,
                    "references": [
                        {
                            "audio_path": str(input_path),
                            "text": prompt_text.strip(),
                        }
                    ],
                }
                try:
                    audio_bytes = _post_json_bytes(
                        f"http://{host}:{port}/v1/audio/speech",
                        payload,
                        timeout_s=1800.0,
                    )
                except urllib_error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"S2 Pro request failed for {name}: {exc}\n{body}") from exc
                wav_path = out_dir / f"{name}.wav"
                wav_path.write_bytes(audio_bytes)
                outputs[name] = str(wav_path)
                request_logs.append(
                    {
                        "name": name,
                        "text": text,
                        "wav_path": str(wav_path),
                        "bytes": len(audio_bytes),
                    }
                )

            metadata = {
                "status": "ok",
                "model_repo_id": checkpoint_repo_id,
                "checkpoint_dir": str(ckpt_dir),
                "config_path": str(S2PRO_CONFIG_PATH),
                "input_wav_path": str(input_path),
                "prompt_text": prompt_text.strip(),
                "outputs": outputs,
                "requests": request_logs,
                "server_log_path": str(server_log_path),
            }
            (out_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            volume.commit()
            return metadata
        finally:
            if server_proc.poll() is None:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server_proc.kill()
                    server_proc.wait(timeout=30)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={"/vol": volume},
    secrets=HF_SECRETS,
)
def sp_sp010_hindi_demo():
    return reference_tts.local(
        input_wav_path="/vol/data/raw/SP_SP010_1.wav",
        prompt_text=(
            "Oh god, I'm just so happy. Oh, and it's all your fault. Oh honestly, "
            "probably still your house. But still I mean running the dishes through "
            "the dishwasher, putting them up. Yeah yeah, alright, alright yeah, okay. "
            "I guess I do have a lot of explaining to do don't I? Huh, but feeding off "
            "of other people just feels weird. Like, like I'm cheating you know. "
            "You're sorry? Oh baby I'm sorry too. I'm sorry this whole thing happened."
        ),
        devanagari_text=(
            "आज शाम मौसम बहुत सुहावना है, हल्की ठंडी हवा चल रही है और दूर कहीं से "
            "चाय की खुशबू आ रही है। मैं बस थोड़ा टहलने निकला हूँ और सोच रहा हूँ कि "
            "ज़िंदगी में कभी-कभी धीरे चलना भी ज़रूरी होता है।"
        ),
        latin_text=(
            "Aaj shaam mausam bahut suhaavna hai, halki thandi hawa chal rahi hai aur "
            "door kahin se chai ki khushboo aa rahi hai. Main bas thoda tahalne nikla "
            "hoon aur soch raha hoon ki zindagi mein kabhi-kabhi dheere chalna bhi "
            "zaroori hota hai."
        ),
        output_dir="/vol/outputs/fish_audio_s2_pro/sp_sp010_hindi_demo",
    )
