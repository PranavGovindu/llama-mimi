import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Dataset
from transformers import pipeline

from torchtitan.config_manager import JobConfig
from torchtitan.tools.audio_codec import load_audio_codec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize a single WAV file into audio-code parquet."
    )
    parser.add_argument("--input-wav", required=True, help="Input WAV path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output directory (e.g. /vol/data/custom_download_q8).",
    )
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--lang", default="en", help="Language code (e.g. en).")
    parser.add_argument("--sample-id", default="custom_0001", help="Sample ID.")
    parser.add_argument("--text", default="", help="Transcription text for TTS pairing.")
    parser.add_argument(
        "--num-quantizers",
        type=int,
        default=8,
        help="Number of codec quantizers/codebooks to export.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=20.0,
        help="Truncate audio to this many seconds if longer.",
    )
    parser.add_argument(
        "--asr-model-id",
        default="openai/whisper-small.en",
        help="ASR model used only when --text is empty.",
    )
    parser.add_argument(
        "--audio-codec-backend",
        choices=["mimi", "s1_dac", "spark_bicodec"],
        default="mimi",
        help="Audio codec backend used for tokenization.",
    )
    parser.add_argument(
        "--audio-codec-source",
        choices=["official_fish", "hf_pretrained"],
        default="official_fish",
        help="Codec source hint used by the adapter.",
    )
    parser.add_argument(
        "--audio-codec-model-id",
        default="",
        help="Codec HF model id/path override.",
    )
    parser.add_argument(
        "--audio-codec-ckpt-path",
        default="",
        help="Optional local checkpoint path for official codec assets.",
    )
    parser.add_argument(
        "--audio-codec-trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for codec model loading.",
    )
    return parser.parse_args()


def _load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), int(sr)


def _resample_if_needed(
    audio: np.ndarray, sr: int, target_sr: int
) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return audio, sr
    import librosa

    resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return resampled.astype(np.float32), int(target_sr)


def _encode_audio_codes(
    audio_array: np.ndarray,
    input_sr: int,
    feature_extractor,
    audio_tokenizer,
    num_quantizers: int,
    codec_backend: str = "mimi",
) -> tuple[list[list[int]], list[int], list[int], np.ndarray]:
    audio_array, used_sr = _resample_if_needed(
        audio_array, input_sr, feature_extractor.sampling_rate
    )
    inputs = feature_extractor(
        raw_audio=audio_array,
        sampling_rate=used_sr,
        return_tensors="pt",
    ).to(audio_tokenizer.device)
    padding_mask = inputs.get("padding_mask")
    if padding_mask is None:
        padding_mask = inputs.get("attention_mask")
    if padding_mask is None:
        padding_mask = torch.ones_like(inputs["input_values"], dtype=torch.long)
    with torch.no_grad():
        outputs = audio_tokenizer.encode(
            inputs["input_values"],
            padding_mask,
            num_quantizers=num_quantizers,
        )
    audio_codes = outputs.audio_codes
    if audio_codes.ndim == 2:
        audio_codes = audio_codes.unsqueeze(0)
    if audio_codes.ndim != 3:
        raise RuntimeError(f"Unsupported codec output shape: {tuple(audio_codes.shape)}")
    codes_tq = audio_codes[0].transpose(0, 1).detach().cpu().tolist()

    semantic_tokens: list[int] = []
    global_tokens: list[int] = []
    if codec_backend.strip().lower() == "spark_bicodec":
        semantic_tokens = audio_codes[0, 0].detach().cpu().to(torch.int64).tolist()
        raw_global = getattr(outputs, "global_codes", None)
        if raw_global is not None:
            if not torch.is_tensor(raw_global):
                raw_global = torch.as_tensor(raw_global)
            if raw_global.ndim == 1:
                raw_global = raw_global.unsqueeze(0)
            if raw_global.ndim == 3:
                if raw_global.shape[1] == 1:
                    raw_global = raw_global[:, 0, :]
                elif raw_global.shape[2] == 1:
                    raw_global = raw_global[:, :, 0]
                else:
                    raw_global = raw_global.reshape(raw_global.shape[0], -1)
            if raw_global.ndim == 2 and raw_global.shape[0] >= 1:
                global_tokens = raw_global[0].detach().cpu().to(torch.int64).tolist()

    return codes_tq, semantic_tokens, global_tokens, audio_array


def _auto_transcribe(audio_array: np.ndarray, sr: int, asr_model_id: str) -> str:
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline(
        task="automatic-speech-recognition",
        model=asr_model_id,
        device=device,
    )
    result = asr({"array": audio_array, "sampling_rate": sr})
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_wav)
    if not input_path.exists():
        raise FileNotFoundError(f"Input WAV not found: {input_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    job_cfg = JobConfig()
    job_cfg.model.num_quantizers = int(args.num_quantizers)
    job_cfg.audio_codec.backend = args.audio_codec_backend
    job_cfg.audio_codec.source = args.audio_codec_source
    if args.audio_codec_model_id.strip():
        job_cfg.audio_codec.model_id = args.audio_codec_model_id.strip()
    if args.audio_codec_ckpt_path.strip():
        job_cfg.audio_codec.codec_ckpt_path = args.audio_codec_ckpt_path.strip()
    job_cfg.audio_codec.trust_remote_code = bool(args.audio_codec_trust_remote_code)
    audio_tokenizer, feature_extractor, codec_info = load_audio_codec(job_cfg, device)

    original_audio, original_sr = _load_audio_mono(input_path)
    original_duration = float(len(original_audio) / max(original_sr, 1))
    max_samples = int(args.max_seconds * original_sr) if args.max_seconds > 0 else 0
    truncated = bool(max_samples > 0 and len(original_audio) > max_samples)
    if truncated:
        original_audio = original_audio[:max_samples]

    codes_tq, spark_semantic_tokens, spark_global_tokens, model_audio = _encode_audio_codes(
        original_audio,
        original_sr,
        feature_extractor,
        audio_tokenizer,
        args.num_quantizers,
        codec_backend=args.audio_codec_backend,
    )
    used_duration = float(len(model_audio) / feature_extractor.sampling_rate)

    text = args.text.strip()
    if not text:
        try:
            text = _auto_transcribe(
                model_audio, feature_extractor.sampling_rate, args.asr_model_id
            )
        except Exception as e:
            print(f"ASR transcription failed, using fallback text. reason={e}")
            text = input_path.stem.replace("_", " ").strip() or "custom sample"

    record = {
        "id": str(args.sample_id),
        "lang": args.lang,
        "text": text,
        "audio_codes": codes_tq,
        "mimi_codes": codes_tq,
        "sample_rate": int(feature_extractor.sampling_rate),
        "duration_sec": float(used_duration),
        "split": args.split,
    }
    if args.audio_codec_backend.strip().lower() == "spark_bicodec":
        record["spark_semantic_tokens"] = [int(x) for x in spark_semantic_tokens]
        record["spark_global_tokens"] = [int(x) for x in spark_global_tokens]

    out_dir = Path(args.output_dir) / args.split / f"lang={args.lang}"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "part-00000.parquet"
    Dataset.from_list([record]).to_parquet(str(parquet_path))

    meta = {
        "input_wav": str(input_path),
        "output_parquet": str(parquet_path),
        "lang": args.lang,
        "sample_id": str(args.sample_id),
        "text": text,
        "num_quantizers": int(args.num_quantizers),
        "codec_backend": codec_info.backend,
        "codec_source": codec_info.source,
        "codec_model_ref": codec_info.model_ref,
        "codec_codebook_size": int(codec_info.codebook_size),
        "codec_max_codebooks": int(codec_info.max_codebooks),
        "original_sample_rate": int(original_sr),
        "model_sample_rate": int(feature_extractor.sampling_rate),
        "original_duration_sec": original_duration,
        "used_duration_sec": used_duration,
        "truncated": truncated,
        "max_seconds": float(args.max_seconds),
        "frames": int(len(codes_tq)),
        "spark_semantic_tokens": int(len(spark_semantic_tokens)),
        "spark_global_tokens": int(len(spark_global_tokens)),
    }
    meta_path = Path(args.output_dir) / "metadata_single_sample.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
