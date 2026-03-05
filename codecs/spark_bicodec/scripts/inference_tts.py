#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spark BiCodec TinyAya inference wrapper.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--num-quantizers", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--restrict-audio-vocab", action="store_true")
    parser.add_argument("--output-file", default="output_tts_spark.wav")
    parser.add_argument("--spark-global-tokens", default="")
    parser.add_argument("--spark-global-tokens-file", default="")
    parser.add_argument("--spark-prompt-audio", default="")
    parser.add_argument("--audio-codec-source", default="hf_pretrained")
    parser.add_argument(
        "--audio-codec-model-id",
        default="/root/spark-tts/pretrained_models/Spark-TTS-0.5B",
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
        "spark_bicodec",
        "--audio-codec-source",
        args.audio_codec_source,
        "--audio-codec-model-id",
        args.audio_codec_model_id,
    ]
    if args.greedy:
        cmd.append("--greedy")
    if args.restrict_audio_vocab:
        cmd.append("--restrict-audio-vocab")
    if args.spark_global_tokens.strip():
        cmd.extend(["--spark-global-tokens", args.spark_global_tokens.strip()])
    if args.spark_global_tokens_file.strip():
        cmd.extend(["--spark-global-tokens-file", args.spark_global_tokens_file.strip()])
    if args.spark_prompt_audio.strip():
        cmd.extend(["--spark-prompt-audio", args.spark_prompt_audio.strip()])
    if args.audio_codec_ckpt_path.strip():
        cmd.extend(["--audio-codec-ckpt-path", args.audio_codec_ckpt_path.strip()])
    if args.audio_codec_trust_remote_code:
        cmd.append("--audio-codec-trust-remote-code")
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)


if __name__ == "__main__":
    main()
