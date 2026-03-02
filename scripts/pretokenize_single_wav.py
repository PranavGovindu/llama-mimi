import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Dataset
from transformers import AutoFeatureExtractor, MimiModel, pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize a single WAV file into Mimi codes parquet."
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
        help="Number of Mimi quantizers to export.",
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
    mimi_model: MimiModel,
    num_quantizers: int,
) -> tuple[list[list[int]], np.ndarray]:
    audio_array, used_sr = _resample_if_needed(
        audio_array, input_sr, feature_extractor.sampling_rate
    )
    inputs = feature_extractor(
        raw_audio=audio_array,
        sampling_rate=used_sr,
        return_tensors="pt",
    ).to(mimi_model.device)
    with torch.no_grad():
        outputs = mimi_model.encode(
            inputs["input_values"],
            inputs["padding_mask"],
            num_quantizers=num_quantizers,
        )
    codes_tq = outputs.audio_codes[0].transpose(0, 1).cpu().tolist()
    return codes_tq, audio_array


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
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    mimi_model = MimiModel.from_pretrained("kyutai/mimi").to(device)
    mimi_model.eval()

    original_audio, original_sr = _load_audio_mono(input_path)
    original_duration = float(len(original_audio) / max(original_sr, 1))
    max_samples = int(args.max_seconds * original_sr) if args.max_seconds > 0 else 0
    truncated = bool(max_samples > 0 and len(original_audio) > max_samples)
    if truncated:
        original_audio = original_audio[:max_samples]

    codes_tq, model_audio = _encode_audio_codes(
        original_audio,
        original_sr,
        feature_extractor,
        mimi_model,
        args.num_quantizers,
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
        "mimi_codes": codes_tq,
        "sample_rate": int(feature_extractor.sampling_rate),
        "duration_sec": float(used_duration),
        "split": args.split,
    }

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
        "original_sample_rate": int(original_sr),
        "model_sample_rate": int(feature_extractor.sampling_rate),
        "original_duration_sec": original_duration,
        "used_duration_sec": used_duration,
        "truncated": truncated,
        "max_seconds": float(args.max_seconds),
        "frames": int(len(codes_tq)),
    }
    meta_path = Path(args.output_dir) / "metadata_single_sample.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
