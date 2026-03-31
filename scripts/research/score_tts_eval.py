#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from torchtitan.tools.research_eval import (
    compute_mel_pair_metrics,
    save_mel_image,
    summarize_full_eval_rows,
    write_tsv,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _resolve_run_dir(repo_root: Path, experiment_id: str) -> Path | None:
    runs_root = repo_root / "experiments" / "runs"
    for codec_dir in sorted(runs_root.iterdir() if runs_root.exists() else []):
        candidate = codec_dir / experiment_id
        if candidate.exists():
            return candidate
    return None


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio_np, sample_rate = sf.read(path, always_2d=False)
    audio_np = np.asarray(audio_np, dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)
    return audio_np, int(sample_rate)


def _maybe_build_dnsmos():
    try:
        from torchmetrics.functional.audio.dnsmos import (
            deep_noise_suppression_mean_opinion_score,
        )

        return deep_noise_suppression_mean_opinion_score
    except Exception:
        return None


def _maybe_build_utmos(enable_utmos: bool):
    if not enable_utmos:
        return None
    try:
        predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
        )
        if hasattr(predictor, "to"):
            predictor = predictor.to(DEVICE)
        predictor.eval()
        return predictor
    except Exception:
        return None


def _maybe_build_speaker_scorer(enable_speaker_similarity: bool):
    if not enable_speaker_similarity:
        return None
    try:
        from speechbrain.inference.speaker import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(DEVICE)},
        )
        return classifier
    except Exception:
        return None


def _score_dnsmos(audio_np: np.ndarray, sample_rate: int, scorer) -> dict[str, float] | None:
    if scorer is None:
        return None
    try:
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0).to(DEVICE)
        scores = scorer(waveform, sample_rate, personalized=False)
        values = scores[0].detach().cpu().tolist()
        return {
            "dnsmos_p808": float(values[0]),
            "dnsmos_sig": float(values[1]),
            "dnsmos_bak": float(values[2]),
            "dnsmos_ovr": float(values[3]),
        }
    except Exception:
        return None


def _score_utmos(audio_np: np.ndarray, sample_rate: int, scorer) -> float | None:
    if scorer is None:
        return None
    try:
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0).to(DEVICE)
        score = scorer(waveform, sample_rate)
        return float(score.detach().cpu().item())
    except Exception:
        return None


def _score_speaker_similarity(
    reference_audio: np.ndarray,
    generated_audio: np.ndarray,
    sample_rate: int,
    scorer,
) -> float | None:
    if scorer is None:
        return None
    try:
        target_sr = 16000
        if sample_rate != target_sr:
            reference_audio = librosa.resample(
                reference_audio, orig_sr=sample_rate, target_sr=target_sr
            )
            generated_audio = librosa.resample(
                generated_audio, orig_sr=sample_rate, target_sr=target_sr
            )

        ref_tensor = torch.from_numpy(reference_audio).float().unsqueeze(0).to(DEVICE)
        gen_tensor = torch.from_numpy(generated_audio).float().unsqueeze(0).to(DEVICE)
        ref_embedding = scorer.encode_batch(ref_tensor).reshape(1, -1)
        gen_embedding = scorer.encode_batch(gen_tensor).reshape(1, -1)
        similarity = F.cosine_similarity(ref_embedding, gen_embedding, dim=1)
        return float(similarity.detach().cpu().item())
    except Exception:
        return None


def _iter_step_dirs(run_dir: Path, step_filter: int | None) -> list[Path]:
    root = run_dir / "full_eval"
    if not root.exists():
        return []
    step_dirs = sorted(path for path in root.glob("step_*") if path.is_dir())
    if step_filter is None:
        return step_dirs
    needle = f"step_{step_filter:06d}"
    return [path for path in step_dirs if path.name == needle]


def _build_codebook_report(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    q_count = 0
    for row in sample_rows:
        target_q = row.get("target_coverage_q") or []
        generated_q = row.get("generated_coverage_q") or []
        q_count = max(q_count, len(target_q), len(generated_q))

    per_quantizer: list[dict[str, Any]] = []
    for q_idx in range(q_count):
        target_values: list[float] = []
        generated_values: list[float] = []
        abs_diffs: list[float] = []
        for row in sample_rows:
            target_q = row.get("target_coverage_q") or []
            generated_q = row.get("generated_coverage_q") or []
            if q_idx < len(target_q):
                target_values.append(float(target_q[q_idx]))
            if q_idx < len(generated_q):
                generated_values.append(float(generated_q[q_idx]))
            if q_idx < len(target_q) and q_idx < len(generated_q):
                abs_diffs.append(abs(float(generated_q[q_idx]) - float(target_q[q_idx])))
        per_quantizer.append(
            {
                "quantizer": q_idx,
                "target_coverage_mean": float(np.mean(target_values)) if target_values else None,
                "generated_coverage_mean": float(np.mean(generated_values)) if generated_values else None,
                "coverage_abs_diff_mean": float(np.mean(abs_diffs)) if abs_diffs else None,
                "coverage_abs_diff_max": float(np.max(abs_diffs)) if abs_diffs else None,
            }
        )

    return {
        "quantizers": q_count,
        "per_quantizer": per_quantizer,
    }


def _summarize_step(
    step_dir: Path,
    dnsmos_scorer,
    utmos_scorer,
    speaker_scorer,
    progress_every: int,
) -> dict[str, Any]:
    sample_rows: list[dict[str, Any]] = []
    sample_tsv_rows: list[dict[str, Any]] = []
    metric_availability = {
        "dnsmos": dnsmos_scorer is not None,
        "utmos": utmos_scorer is not None,
        "speaker_similarity": speaker_scorer is not None,
        "salmon": False,
    }

    for sample_dir in sorted(path for path in step_dir.glob("sample_*") if path.is_dir()):
        sample_row = _load_json(sample_dir / "sample_metrics.json")
        if not sample_row:
            continue

        target_path = sample_dir / "target_audio.wav"
        generated_path = sample_dir / "generated_audio.wav"
        offline_metrics_path = sample_dir / "offline_metrics.json"
        offline_metrics: dict[str, Any] = _load_json(offline_metrics_path)

        need_offline = not offline_metrics
        if need_offline and target_path.exists() and generated_path.exists():
            target_audio, target_sr = _load_audio(target_path)
            generated_audio, generated_sr = _load_audio(generated_path)
            if generated_sr != target_sr:
                generated_audio = librosa.resample(
                    generated_audio, orig_sr=generated_sr, target_sr=target_sr
                )
                generated_sr = target_sr

            mel_metrics, target_mel, generated_mel, diff_mel = compute_mel_pair_metrics(
                target_audio,
                generated_audio,
                sample_rate=target_sr,
            )
            offline_metrics.update(mel_metrics)
            save_mel_image(sample_dir / "target_mel.png", target_mel)
            save_mel_image(sample_dir / "generated_mel.png", generated_mel)
            save_mel_image(sample_dir / "mel_diff.png", np.abs(diff_mel))

            dnsmos_scores = _score_dnsmos(generated_audio, generated_sr, dnsmos_scorer)
            if dnsmos_scores:
                offline_metrics.update(dnsmos_scores)
            utmos_score = _score_utmos(generated_audio, generated_sr, utmos_scorer)
            if utmos_score is not None:
                offline_metrics["utmos"] = utmos_score
            speaker_similarity = _score_speaker_similarity(
                target_audio,
                generated_audio,
                target_sr,
                speaker_scorer,
            )
            if speaker_similarity is not None:
                offline_metrics["speaker_similarity"] = speaker_similarity
            offline_metrics_path.write_text(
                json.dumps(offline_metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        external_metrics = _load_json(sample_dir / "external_metrics.json")
        if external_metrics:
            offline_metrics.update(external_metrics)
            if external_metrics.get("speaker_similarity") is not None:
                metric_availability["speaker_similarity"] = True
            if external_metrics.get("salmon") is not None:
                metric_availability["salmon"] = True

        merged = {**sample_row, **offline_metrics}
        sample_rows.append(merged)
        sample_tsv_rows.append(
            {
                "sample_idx": merged.get("sample_idx", ""),
                "language": merged.get("language", ""),
                "generation_status": merged.get("generation_status", ""),
                "wer": merged.get("wer", ""),
                "cer": merged.get("cer", ""),
                "mel_l1": merged.get("mel_l1", ""),
                "mel_l2": merged.get("mel_l2", ""),
                "mel_cosine": merged.get("mel_cosine", ""),
                "utmos": merged.get("utmos", ""),
                "dnsmos_p808": merged.get("dnsmos_p808", ""),
                "dnsmos_ovr": merged.get("dnsmos_ovr", ""),
                "speaker_similarity": merged.get("speaker_similarity", ""),
                "salmon": merged.get("salmon", ""),
                "target_frames": merged.get("target_frames", ""),
                "generated_frames": merged.get("generated_frames", ""),
                "frame_ratio": merged.get("frame_ratio", ""),
                "coverage_q_min": merged.get("coverage_q_min", ""),
                "coverage_q_abs_diff_max": merged.get("coverage_q_abs_diff_max", ""),
            }
        )
        if progress_every > 0 and len(sample_rows) % progress_every == 0:
            print(
                f"[score_tts_eval] {step_dir.name}: processed "
                f"{len(sample_rows)} samples",
                flush=True,
            )

    summary = summarize_full_eval_rows(sample_rows)
    summary["metric_availability"] = metric_availability
    existing_summary = _load_json(step_dir / "summary.json")
    if existing_summary:
        summary = {**existing_summary, **summary}
    if (step_dir / "salmon.json").exists():
        salmon_payload = _load_json(step_dir / "salmon.json")
        salmon_value = salmon_payload.get("salmon_mean")
        if salmon_value is not None:
            summary["salmon_mean"] = salmon_value
            summary["metric_availability"]["salmon"] = True

    codebook_report = _build_codebook_report(sample_rows)
    (step_dir / "codebook_report.json").write_text(
        json.dumps(codebook_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    (step_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_tsv(
        step_dir / "sample_metrics.tsv",
        sample_tsv_rows,
        [
            "sample_idx",
            "language",
            "generation_status",
            "wer",
            "cer",
            "mel_l1",
            "mel_l2",
            "mel_cosine",
            "utmos",
            "dnsmos_p808",
            "dnsmos_ovr",
            "speaker_similarity",
            "salmon",
            "target_frames",
            "generated_frames",
            "frame_ratio",
            "coverage_q_min",
            "coverage_q_abs_diff_max",
        ],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score full-pack TTS eval artifacts.")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument(
        "--enable-utmos",
        action="store_true",
        help="Attempt to score UTMOS via torch.hub if the model is available.",
    )
    parser.add_argument(
        "--enable-speaker-sim",
        action="store_true",
        help="Attempt to score speaker similarity via SpeechBrain ECAPA.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Emit progress logs every N samples while scoring.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if run_dir is None and args.experiment_id:
        run_dir = _resolve_run_dir(repo_root, args.experiment_id)
    if run_dir is None:
        raise SystemExit("Provide --run-dir or --experiment-id.")
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    dnsmos_scorer = _maybe_build_dnsmos()
    utmos_scorer = _maybe_build_utmos(args.enable_utmos)
    speaker_scorer = _maybe_build_speaker_scorer(args.enable_speaker_sim)

    step_summaries = []
    for step_dir in _iter_step_dirs(run_dir, args.step):
        step_summaries.append(
            _summarize_step(
                step_dir,
                dnsmos_scorer,
                utmos_scorer,
                speaker_scorer,
                args.progress_every,
            )
        )

    output = {
        "run_dir": str(run_dir),
        "steps_scored": len(step_summaries),
        "step_summaries": step_summaries,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
