#!/usr/bin/env python3
import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S1-track wrapper for single-wav pretokenization."
    )
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--sample-id", default="download_001")
    parser.add_argument("--text", default="")
    parser.add_argument("--num-quantizers", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=20.0)
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
    cmd = [
        sys.executable,
        "scripts/pretokenize_single_wav.py",
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
        "s1_dac",
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
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

