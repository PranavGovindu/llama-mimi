#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen codec wrapper for single-wav pretokenization."
    )
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--sample-id", default="download_001")
    parser.add_argument("--text", default="")
    parser.add_argument("--num-quantizers", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--audio-codec-source", default="hf_pretrained")
    parser.add_argument("--audio-codec-model-id", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--audio-codec-ckpt-path", default="")
    parser.add_argument("--audio-codec-trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = str(repo_root)
    if os.environ.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}:{os.environ['PYTHONPATH']}"
    env = {**os.environ, "PYTHONPATH": pythonpath}
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "pretokenize_single_wav.py"),
        "--input-wav",
        args.input_wav,
        "--output-dir",
        args.output_dir,
        "--split",
        args.split,
        "--lang",
        args.lang,
        "--sample-id",
        args.sample_id,
        "--num-quantizers",
        str(args.num_quantizers),
        "--max-seconds",
        str(args.max_seconds),
        "--audio-codec-backend",
        "qwen_codec",
        "--audio-codec-source",
        args.audio_codec_source,
        "--audio-codec-model-id",
        args.audio_codec_model_id,
    ]
    if args.text.strip():
        cmd.extend(["--text", args.text.strip()])
    if args.audio_codec_ckpt_path.strip():
        cmd.extend(["--audio-codec-ckpt-path", args.audio_codec_ckpt_path.strip()])
    if args.audio_codec_trust_remote_code:
        cmd.append("--audio-codec-trust-remote-code")
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)


if __name__ == "__main__":
    main()
