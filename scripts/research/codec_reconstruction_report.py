#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

from torchtitan.config_manager import JobConfig
from torchtitan.tools.audio_codec import load_audio_codec
from torchtitan.tools.research_eval import (
    compute_mel_pair_metrics,
    save_mel_image,
)
from torchtitan.tools.text_norm import normalize_text_for_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode/decode a single WAV with a codec and write a reconstruction report."
    )
    parser.add_argument("--input-wav", required=True, help="Input WAV path.")
    parser.add_argument("--output-dir", required=True, help="Output report directory.")
    parser.add_argument("--text", default="", help="Reference text. If empty, ASR the original.")
    parser.add_argument("--lang", default="en", help="Language hint for eval normalization.")
    parser.add_argument("--sample-id", default="sample_0001", help="Sample identifier.")
    parser.add_argument(
        "--num-quantizers",
        type=int,
        default=8,
        help="Number of codec quantizers/codebooks to use.",
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
        help="ASR model used for transcript comparison.",
    )
    parser.add_argument(
        "--audio-codec-backend",
        choices=["mimi", "s1_dac", "spark_bicodec", "dualcodec", "qwen_codec"],
        default="mimi",
        help="Audio codec backend used for reconstruction.",
    )
    parser.add_argument(
        "--audio-codec-source",
        choices=["official_fish", "hf_pretrained"],
        default="hf_pretrained",
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


def _edit_distance(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def _compute_wer_cer(prediction: str, reference: str, lang_hint: str = "") -> tuple[float, float]:
    pred_norm = normalize_text_for_eval(prediction, lang_hint=lang_hint)
    ref_norm = normalize_text_for_eval(reference, lang_hint=lang_hint)
    pred_words = pred_norm.split()
    ref_words = ref_norm.split()
    wer = float(_edit_distance(pred_words, ref_words)) / float(max(len(ref_words), 1))
    cer = float(_edit_distance(list(pred_norm), list(ref_norm))) / float(max(len(ref_norm), 1))
    return wer, cer


def _compute_snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    est = np.asarray(estimate, dtype=np.float32).reshape(-1)
    t = min(ref.shape[0], est.shape[0])
    if t <= 0:
        return float("nan")
    ref = ref[:t]
    est = est[:t]
    noise = est - ref
    signal_power = float(np.mean(np.square(ref))) + 1e-12
    noise_power = float(np.mean(np.square(noise))) + 1e-12
    return 10.0 * math.log10(signal_power / noise_power)


def _extract_waveform(decoded: Any) -> np.ndarray:
    audio_values = getattr(decoded, "audio_values", None)
    if torch.is_tensor(audio_values):
        decoded = audio_values
    if isinstance(decoded, (tuple, list)):
        for item in decoded:
            if torch.is_tensor(item):
                decoded = item
                break
    if not torch.is_tensor(decoded):
        raise TypeError(f"Unsupported decoded audio type: {type(decoded)!r}")
    waveform = decoded.detach().float().cpu()
    while waveform.ndim > 1:
        waveform = waveform[0]
    waveform_np = np.asarray(waveform.numpy(), dtype=np.float32).reshape(-1)
    if not np.isfinite(waveform_np).all():
        waveform_np = np.nan_to_num(waveform_np, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(waveform_np, -1.0, 1.0).astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)
    sf.write(
        str(path),
        audio_np,
        int(sample_rate),
        format="WAV",
        subtype="PCM_16",
    )


def _codebook_stats(codes_bqt: torch.Tensor, codebook_size: int) -> dict[str, Any]:
    codes_qt = codes_bqt.detach().cpu().to(torch.int64)[0]
    q_count = int(codes_qt.shape[0]) if codes_qt.ndim == 2 else 0
    t_count = int(codes_qt.shape[1]) if codes_qt.ndim == 2 else 0
    coverage_per_q: list[float] = []
    unique_per_q: list[int] = []
    for q_idx in range(q_count):
        unique_q = int(torch.unique(codes_qt[q_idx]).numel())
        unique_per_q.append(unique_q)
        coverage_per_q.append(float(unique_q) / float(max(codebook_size, 1)))
    unique_total = int(torch.unique(codes_qt).numel()) if t_count > 0 else 0
    return {
        "quantizers": q_count,
        "frames": t_count,
        "unique_total": unique_total,
        "coverage_total": float(unique_total) / float(max(codebook_size, 1)),
        "unique_per_q": unique_per_q,
        "coverage_per_q": coverage_per_q,
        "codes_qt": codes_qt.numpy().astype(np.int32),
    }


def _maybe_build_asr(asr_model_id: str):
    device = 0 if torch.cuda.is_available() else -1
    try:
        return pipeline(
            task="automatic-speech-recognition",
            model=asr_model_id,
            device=device,
        )
    except Exception:
        return None


def _transcribe(asr_pipe, audio_np: np.ndarray, sample_rate: int) -> str:
    if asr_pipe is None:
        return ""
    try:
        result = asr_pipe({"array": audio_np, "sampling_rate": sample_rate})
    except Exception:
        return ""
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()


def _save_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Codec Reconstruction Report: {report.get('sample_id', '-')}",
        "",
        "## Codec",
        f"- backend: {report.get('codec_backend', '-')}",
        f"- source: {report.get('codec_source', '-')}",
        f"- model_ref: {report.get('codec_model_ref', '-')}",
        f"- quantizers: {report.get('num_quantizers', '-')}",
        "",
        "## Audio",
        f"- original_sample_rate: {report.get('original_sample_rate', '-')}",
        f"- model_sample_rate: {report.get('model_sample_rate', '-')}",
        f"- original_duration_sec: {report.get('original_duration_sec', '-')}",
        f"- reconstructed_duration_sec: {report.get('reconstructed_duration_sec', '-')}",
        f"- duration_ratio: {report.get('duration_ratio', '-')}",
        "",
        "## Metrics",
        f"- mel_l1: {report.get('mel_l1', '-')}",
        f"- mel_l2: {report.get('mel_l2', '-')}",
        f"- mel_cosine: {report.get('mel_cosine', '-')}",
        f"- snr_db: {report.get('snr_db', '-')}",
        f"- wer_vs_reference: {report.get('wer_vs_reference', '-')}",
        f"- cer_vs_reference: {report.get('cer_vs_reference', '-')}",
        "",
        "## ASR",
        f"- reference_text: {report.get('reference_text', '-')}",
        f"- original_asr_text: {report.get('original_asr_text', '-')}",
        f"- reconstructed_asr_text: {report.get('reconstructed_asr_text', '-')}",
        "",
        "## Codebooks",
        f"- frames: {report.get('frames', '-')}",
        f"- coverage_total: {report.get('coverage_total', '-')}",
        f"- coverage_per_q: {report.get('coverage_per_q', [])}",
        "",
        "## Files",
        "- original_audio.wav",
        "- reconstructed_audio.wav",
        "- original_mel.png",
        "- reconstructed_mel.png",
        "- mel_diff.png",
        "- codes_qt.csv",
        "- report.json",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_wav).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input WAV not found: {input_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    model_audio = feature_extractor(
        raw_audio=original_audio,
        sampling_rate=original_sr,
        return_tensors="pt",
    ).to(audio_tokenizer.device)
    padding_mask = model_audio.get("padding_mask")
    if padding_mask is None:
        padding_mask = model_audio.get("attention_mask")
    if padding_mask is None:
        padding_mask = torch.ones_like(model_audio["input_values"], dtype=torch.long)

    with torch.no_grad():
        encoder_outputs = audio_tokenizer.encode(
            model_audio["input_values"],
            padding_mask,
            num_quantizers=int(args.num_quantizers),
        )
    codes_bqt = encoder_outputs.audio_codes
    reconstructed = audio_tokenizer.decode(codes_bqt.to(audio_tokenizer.device))
    reconstructed_audio = _extract_waveform(reconstructed)
    reconstructed_duration = float(len(reconstructed_audio) / feature_extractor.sampling_rate)

    if original_sr != feature_extractor.sampling_rate:
        import librosa

        original_audio_for_metrics = librosa.resample(
            original_audio,
            orig_sr=original_sr,
            target_sr=feature_extractor.sampling_rate,
        ).astype(np.float32)
    else:
        original_audio_for_metrics = original_audio

    mel_metrics, original_mel, reconstructed_mel, mel_diff = compute_mel_pair_metrics(
        original_audio_for_metrics,
        reconstructed_audio,
        sample_rate=feature_extractor.sampling_rate,
    )
    save_mel_image(output_dir / "original_mel.png", original_mel)
    save_mel_image(output_dir / "reconstructed_mel.png", reconstructed_mel)
    save_mel_image(output_dir / "mel_diff.png", np.abs(mel_diff))

    _write_wav(
        output_dir / "original_audio.wav",
        original_audio_for_metrics,
        feature_extractor.sampling_rate,
    )
    _write_wav(
        output_dir / "reconstructed_audio.wav",
        reconstructed_audio,
        feature_extractor.sampling_rate,
    )

    codebook = _codebook_stats(codes_bqt, int(codec_info.codebook_size))
    np.savetxt(output_dir / "codes_qt.csv", codebook["codes_qt"], delimiter=",", fmt="%d")

    asr = _maybe_build_asr(args.asr_model_id)
    reference_text = args.text.strip()
    original_asr = _transcribe(asr, original_audio_for_metrics, feature_extractor.sampling_rate)
    reconstructed_asr = _transcribe(
        asr, reconstructed_audio, feature_extractor.sampling_rate
    )
    if not reference_text:
        reference_text = original_asr

    wer_vs_reference = None
    cer_vs_reference = None
    if reference_text and reconstructed_asr:
        wer_vs_reference, cer_vs_reference = _compute_wer_cer(
            reconstructed_asr,
            reference_text,
            lang_hint=args.lang,
        )

    report = {
        "sample_id": args.sample_id,
        "input_wav": str(input_path),
        "codec_backend": codec_info.backend,
        "codec_source": codec_info.source,
        "codec_model_ref": codec_info.model_ref,
        "num_quantizers": int(args.num_quantizers),
        "original_sample_rate": int(original_sr),
        "model_sample_rate": int(feature_extractor.sampling_rate),
        "original_duration_sec": original_duration,
        "reconstructed_duration_sec": reconstructed_duration,
        "duration_ratio": (
            reconstructed_duration / original_duration if original_duration > 0 else None
        ),
        "truncated": truncated,
        "max_seconds": float(args.max_seconds),
        "frames": int(codebook["frames"]),
        "coverage_total": float(codebook["coverage_total"]),
        "coverage_per_q": [float(x) for x in codebook["coverage_per_q"]],
        "unique_per_q": [int(x) for x in codebook["unique_per_q"]],
        "reference_text": reference_text,
        "original_asr_text": original_asr,
        "reconstructed_asr_text": reconstructed_asr,
        "wer_vs_reference": wer_vs_reference,
        "cer_vs_reference": cer_vs_reference,
        "snr_db": _compute_snr_db(original_audio_for_metrics, reconstructed_audio),
        **mel_metrics,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _save_markdown(output_dir / "report.md", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
