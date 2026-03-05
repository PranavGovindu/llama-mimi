#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S1-track wrapper for FLEURS pretokenization."
    )
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-quantizers", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--max-samples-per-language", type=int, default=0)
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
    repo_root = Path(__file__).resolve().parents[2]
    pythonpath = str(repo_root)
    if os.environ.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}:{os.environ['PYTHONPATH']}"
    env = {**os.environ, "PYTHONPATH": pythonpath}
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "pretokenize_fleurs.py"),
        "--languages",
        *args.languages,
        "--split",
        args.split,
        "--num-quantizers",
        str(args.num_quantizers),
        "--output-dir",
        args.output_dir,
        "--shard-size",
        str(args.shard_size),
        "--max-samples-per-language",
        str(args.max_samples_per_language),
        "--audio-codec-backend",
        "s1_dac",
        "--audio-codec-source",
        args.audio_codec_source,
        "--audio-codec-model-id",
        args.audio_codec_model_id,
    ]
    if args.audio_codec_ckpt_path.strip():
        cmd.extend(["--audio-codec-ckpt-path", args.audio_codec_ckpt_path.strip()])
    if args.audio_codec_trust_remote_code:
        cmd.append("--audio-codec-trust-remote-code")
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)


if __name__ == "__main__":
    main()
