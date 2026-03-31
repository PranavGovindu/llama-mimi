from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from PIL import Image


def safe_mean(values: list[float]) -> float | None:
    filtered = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    values = [
        float(row[key])
        for row in rows
        if key in row and row[key] is not None and not math.isnan(float(row[key]))
    ]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_full_eval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_language: dict[str, dict[str, list[float] | int]] = {}
    status_counts: dict[str, int] = {}
    generated_count = 0
    malformed_count = 0

    for row in rows:
        language = str(row.get("language", "")).strip() or "unknown"
        bucket = per_language.setdefault(
            language,
            {
                "samples": 0,
                "generated_count": 0,
                "wer": [],
                "cer": [],
                "mel_l1": [],
                "dnsmos_ovr": [],
                "speaker_similarity": [],
            },
        )
        bucket["samples"] = int(bucket["samples"]) + 1

        status = str(row.get("generation_status", "")).strip() or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

        generated_frames = int(row.get("generated_frames", 0) or 0)
        if generated_frames > 0:
            generated_count += 1
            bucket["generated_count"] = int(bucket["generated_count"]) + 1
        else:
            malformed_count += 1

        for bucket_key, row_key in (
            ("wer", "wer"),
            ("cer", "cer"),
            ("mel_l1", "mel_l1"),
            ("dnsmos_ovr", "dnsmos_ovr"),
            ("speaker_similarity", "speaker_similarity"),
        ):
            if row_key in row and row[row_key] is not None:
                try:
                    bucket[bucket_key].append(float(row[row_key]))
                except (TypeError, ValueError):
                    continue

    per_language_summary: dict[str, dict[str, Any]] = {}
    for lang, payload in sorted(per_language.items()):
        per_language_summary[lang] = {
            "samples": int(payload["samples"]),
            "generated_count": int(payload["generated_count"]),
            "wer_mean": safe_mean(payload["wer"]),
            "cer_mean": safe_mean(payload["cer"]),
            "mel_l1_mean": safe_mean(payload["mel_l1"]),
            "dnsmos_ovr_mean": safe_mean(payload["dnsmos_ovr"]),
            "speaker_similarity_mean": safe_mean(payload["speaker_similarity"]),
        }

    return {
        "sample_count": len(rows),
        "generated_count": generated_count,
        "malformed_decode_count": malformed_count,
        "malformed_decode_rate": (
            float(malformed_count) / float(max(len(rows), 1)) if rows else None
        ),
        "wer_mean": safe_mean(
            [
                float(row["wer"])
                for row in rows
                if "wer" in row and row["wer"] is not None
            ]
        ),
        "cer_mean": safe_mean(
            [
                float(row["cer"])
                for row in rows
                if "cer" in row and row["cer"] is not None
            ]
        ),
        "utmos_mean": safe_mean(
            [
                float(row["utmos"])
                for row in rows
                if "utmos" in row and row["utmos"] is not None
            ]
        ),
        "dnsmos_p808_mean": safe_mean(
            [
                float(row["dnsmos_p808"])
                for row in rows
                if "dnsmos_p808" in row and row["dnsmos_p808"] is not None
            ]
        ),
        "dnsmos_ovr_mean": safe_mean(
            [
                float(row["dnsmos_ovr"])
                for row in rows
                if "dnsmos_ovr" in row and row["dnsmos_ovr"] is not None
            ]
        ),
        "speaker_similarity_mean": safe_mean(
            [
                float(row["speaker_similarity"])
                for row in rows
                if "speaker_similarity" in row and row["speaker_similarity"] is not None
            ]
        ),
        "salmon_mean": safe_mean(
            [
                float(row["salmon"])
                for row in rows
                if "salmon" in row and row["salmon"] is not None
            ]
        ),
        "mel_l1_mean": safe_mean(
            [
                float(row["mel_l1"])
                for row in rows
                if "mel_l1" in row and row["mel_l1"] is not None
            ]
        ),
        "mel_l2_mean": safe_mean(
            [
                float(row["mel_l2"])
                for row in rows
                if "mel_l2" in row and row["mel_l2"] is not None
            ]
        ),
        "mel_cosine_mean": safe_mean(
            [
                float(row["mel_cosine"])
                for row in rows
                if "mel_cosine" in row and row["mel_cosine"] is not None
            ]
        ),
        "target_frames_mean": safe_mean(
            [
                float(row["target_frames"])
                for row in rows
                if "target_frames" in row and row["target_frames"] is not None
            ]
        ),
        "generated_frames_mean": safe_mean(
            [
                float(row["generated_frames"])
                for row in rows
                if "generated_frames" in row and row["generated_frames"] is not None
            ]
        ),
        "frame_ratio_mean": safe_mean(
            [
                float(row["frame_ratio"])
                for row in rows
                if "frame_ratio" in row and row["frame_ratio"] is not None
            ]
        ),
        "target_coverage_total_mean": safe_mean(
            [
                float(row["target_coverage_total"])
                for row in rows
                if "target_coverage_total" in row
                and row["target_coverage_total"] is not None
            ]
        ),
        "generated_coverage_total_mean": safe_mean(
            [
                float(row["generated_coverage_total"])
                for row in rows
                if "generated_coverage_total" in row
                and row["generated_coverage_total"] is not None
            ]
        ),
        "coverage_q_min_mean": safe_mean(
            [
                float(row["coverage_q_min"])
                for row in rows
                if "coverage_q_min" in row and row["coverage_q_min"] is not None
            ]
        ),
        "coverage_q_abs_diff_max_mean": safe_mean(
            [
                float(row["coverage_q_abs_diff_max"])
                for row in rows
                if "coverage_q_abs_diff_max" in row
                and row["coverage_q_abs_diff_max"] is not None
            ]
        ),
        "per_language": per_language_summary,
        "generation_status_counts": status_counts,
    }


def compute_log_mel(audio_np: np.ndarray, sample_rate: int, n_mels: int = 80) -> np.ndarray:
    audio = np.asarray(audio_np, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    if audio.size == 0:
        return np.zeros((n_mels, 1), dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=n_mels,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def mel_to_rgb(mel_db: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel_db, dtype=np.float32)
    if mel.size == 0:
        mel = np.zeros((80, 1), dtype=np.float32)
    mel = np.nan_to_num(mel, nan=-80.0, neginf=-80.0, posinf=0.0)
    mel = np.clip((mel + 80.0) / 80.0, 0.0, 1.0)
    img = (mel * 255.0).astype(np.uint8)
    return np.repeat(img[:, :, None], 3, axis=2)


def save_mel_image(path: Path, mel_db: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mel_to_rgb(mel_db)).save(path)


def compute_mel_pair_metrics(
    reference_audio: np.ndarray,
    predicted_audio: np.ndarray,
    sample_rate: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    ref_mel = compute_log_mel(reference_audio, sample_rate=sample_rate)
    pred_mel = compute_log_mel(predicted_audio, sample_rate=sample_rate)
    t = min(ref_mel.shape[1], pred_mel.shape[1])
    if t <= 0:
        zero = np.zeros((ref_mel.shape[0], 1), dtype=np.float32)
        return (
            {"mel_l1": 0.0, "mel_l2": 0.0, "mel_cosine": 1.0},
            ref_mel,
            pred_mel,
            zero,
        )
    ref_aligned = ref_mel[:, :t]
    pred_aligned = pred_mel[:, :t]
    diff = pred_aligned - ref_aligned
    ref_flat = ref_aligned.reshape(-1)
    pred_flat = pred_aligned.reshape(-1)
    denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(pred_flat))
    cosine = 1.0
    if denom > 0:
        cosine = float(np.dot(ref_flat, pred_flat) / denom)
    metrics = {
        "mel_l1": float(np.mean(np.abs(diff))),
        "mel_l2": float(np.sqrt(np.mean(np.square(diff)))),
        "mel_cosine": cosine,
    }
    return metrics, ref_aligned, pred_aligned, diff


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
