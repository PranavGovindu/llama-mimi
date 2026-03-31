import math

import numpy as np

from torchtitan.tools.research_eval import (
    compute_mel_pair_metrics,
    summarize_full_eval_rows,
)


def test_summarize_full_eval_rows_aggregates_metrics():
    rows = [
        {
            "language": "en",
            "generation_status": "ok",
            "generated_frames": 100,
            "wer": 0.1,
            "cer": 0.05,
            "mel_l1": 1.0,
            "dnsmos_ovr": 3.5,
            "speaker_similarity": 0.8,
            "target_frames": 90,
            "frame_ratio": 100.0 / 90.0,
            "target_coverage_total": 0.2,
            "generated_coverage_total": 0.19,
            "coverage_q_min": 0.01,
            "coverage_q_abs_diff_max": 0.02,
        },
        {
            "language": "en",
            "generation_status": "no_audio_span",
            "generated_frames": 0,
            "wer": 0.2,
            "cer": 0.1,
            "mel_l1": 2.0,
            "dnsmos_ovr": 3.0,
            "speaker_similarity": 0.6,
            "target_frames": 95,
            "frame_ratio": 0.0,
            "target_coverage_total": 0.22,
            "generated_coverage_total": 0.0,
            "coverage_q_min": 0.0,
            "coverage_q_abs_diff_max": 0.2,
        },
    ]

    summary = summarize_full_eval_rows(rows)
    assert summary["sample_count"] == 2
    assert summary["generated_count"] == 1
    assert math.isclose(summary["malformed_decode_rate"], 0.5)
    assert math.isclose(summary["wer_mean"], 0.15)
    assert math.isclose(summary["mel_l1_mean"], 1.5)
    assert math.isclose(summary["frame_ratio_mean"], (100.0 / 90.0) / 2.0)
    assert summary["generation_status_counts"]["ok"] == 1
    assert summary["generation_status_counts"]["no_audio_span"] == 1
    assert math.isclose(summary["per_language"]["en"]["speaker_similarity_mean"], 0.7)


def test_compute_mel_pair_metrics_returns_zero_for_identical_audio():
    sample_rate = 24000
    time_axis = np.linspace(0.0, 0.5, int(sample_rate * 0.5), endpoint=False)
    audio = 0.1 * np.sin(2.0 * np.pi * 440.0 * time_axis).astype(np.float32)

    metrics, ref_mel, pred_mel, diff = compute_mel_pair_metrics(
        audio,
        audio.copy(),
        sample_rate=sample_rate,
    )

    assert ref_mel.shape == pred_mel.shape == diff.shape
    assert metrics["mel_l1"] == 0.0
    assert metrics["mel_l2"] == 0.0
    assert math.isclose(metrics["mel_cosine"], 1.0, rel_tol=1e-6, abs_tol=1e-6)
