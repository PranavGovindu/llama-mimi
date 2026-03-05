#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S1-DAC TinyAya inference wrapper.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--num-quantizers", type=int, default=9)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--restrict-audio-vocab", action="store_true")
    parser.add_argument("--output-file", default="output_tts_s1.wav")
    parser.add_argument(
        "--audio-codec-source",
        choices=["official_fish", "hf_pretrained"],
        default="official_fish",
    )
    parser.add_argument(
        "--audio-codec-model-id",
        default="jordand/fish-s1-dac-min",
    )
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
        str(repo_root / "inference_tts.py"),
        "--model-id",
        args.model_id,
        "--text",
        args.text,
        "--lang",
        args.lang,
        "--num-quantizers",
        str(args.num_quantizers),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--output-file",
        args.output_file,
        "--audio-codec-backend",
        "s1_dac",
        "--audio-codec-source",
        args.audio_codec_source,
        "--audio-codec-model-id",
        args.audio_codec_model_id,
    ]
    if args.greedy:
        cmd.append("--greedy")
    if args.restrict_audio_vocab:
        cmd.append("--restrict-audio-vocab")
    if args.audio_codec_ckpt_path.strip():
        cmd.extend(["--audio-codec-ckpt-path", args.audio_codec_ckpt_path.strip()])
    if args.audio_codec_trust_remote_code:
        cmd.append("--audio-codec-trust-remote-code")
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)


if __name__ == "__main__":
    main()
