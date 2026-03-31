import importlib.util
import json
from pathlib import Path


def _load_render_results_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "research"
        / "render_results_tsv.py"
    )
    spec = importlib.util.spec_from_file_location("render_results_tsv", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_rows_reads_full_eval_summaries(tmp_path: Path):
    module = _load_render_results_module()
    runs_root = tmp_path / "experiments" / "runs"
    run_dir = runs_root / "mimi" / "exp-test"
    (run_dir / "full_eval" / "step_0002000").mkdir(parents=True)
    (run_dir / "run_snapshot").mkdir(parents=True)

    (run_dir / "manifest_start.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-test",
                "campaign_id": "camp-1",
                "phase": "ablation",
                "stage": "screen5k",
                "variant": "lr_1e-4",
                "track": "english",
                "axis": "lr",
                "family": "mimi_q8",
                "baseline_experiment_id": "exp-baseline",
                "config": "codecs/mimi/configs/test.toml",
            }
        )
    )
    (run_dir / "run_snapshot" / "resolved_config.json").write_text(
        json.dumps(
            {
                "model": {"num_quantizers": 8},
                "training": {
                    "seq_len": 4096,
                    "max_audio_seconds": 20,
                    "global_batch_size": 12,
                    "local_batch_size": 3,
                    "dataset": "fleurs_pretok",
                    "dataset_path": "/tmp/data",
                },
                "optimizer": {"lr": 5e-5},
                "lr_scheduler": {"warmup_steps": 40},
                "validation": {"dataset": "tts_pretok", "split": "validation"},
            }
        )
    )
    (run_dir / "run_snapshot" / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "fingerprint_sha256": "dataset-hash-123",
                "sample_id_sha256_by_split": {"validation": "eval-hash-456"},
            }
        )
    )
    (run_dir / "full_eval" / "step_0002000" / "summary.json").write_text(
        json.dumps(
            {
                "step": 2000,
                "eval_pack": "validation",
                "sample_count": 16,
                "generated_count": 15,
                "wer_mean": 0.12,
                "cer_mean": 0.06,
                "dnsmos_ovr_mean": 3.2,
                "mel_l1_mean": 1.1,
                "malformed_decode_rate": 0.0625,
                "frame_ratio_mean": 1.03,
                "target_coverage_total_mean": 0.2,
                "generated_coverage_total_mean": 0.19,
                "coverage_q_min_mean": 0.01,
                "coverage_q_abs_diff_max_mean": 0.03,
            }
        )
    )

    per_codec = module._collect_rows(runs_root)
    assert "mimi" in per_codec
    assert len(per_codec["mimi"]) == 1
    row = per_codec["mimi"][0]
    assert row["experiment_id"] == "exp-test"
    assert row["stage"] == "screen5k"
    assert row["variant"] == "lr_1e-4"
    assert row["seq_len"] == 4096
    assert row["quantizers"] == 8
    assert row["dataset_manifest_hash"] == "dataset-hash-123"
    assert row["eval_manifest_hash"] == "eval-hash-456"
    assert row["validation_split"] == "validation"
    assert row["wer_mean"] == 0.12
    assert row["frame_ratio_mean"] == 1.03
    assert row["rank"] == 1
    assert row["target_audio_tokens_per_update"] == 24000
    assert row["delta_wer"] == ""


def test_assign_anchor_deltas_uses_stage_anchor_mean():
    module = _load_render_results_module()
    rows = [
        {
            "campaign_id": "camp-1",
            "stage": "screen5k",
            "step": 5000,
            "eval_pack": "validation",
            "variant": "anchor",
            "wer_mean": 0.20,
            "cer_mean": 0.10,
            "utmos_mean": 3.0,
            "dnsmos_ovr_mean": 3.5,
            "mel_l1_mean": 1.4,
        },
        {
            "campaign_id": "camp-1",
            "stage": "screen5k",
            "step": 5000,
            "eval_pack": "validation",
            "variant": "anchor",
            "wer_mean": 0.10,
            "cer_mean": 0.08,
            "utmos_mean": 3.2,
            "dnsmos_ovr_mean": 3.7,
            "mel_l1_mean": 1.2,
        },
        {
            "campaign_id": "camp-1",
            "stage": "screen5k",
            "step": 5000,
            "eval_pack": "validation",
            "variant": "lr_1e-4",
            "wer_mean": 0.12,
            "cer_mean": 0.07,
            "utmos_mean": 3.4,
            "dnsmos_ovr_mean": 3.9,
            "mel_l1_mean": 1.0,
        },
    ]

    module._assign_anchor_deltas(rows)

    challenger = rows[-1]
    assert round(challenger["delta_wer"], 6) == round(0.12 - 0.15, 6)
    assert round(challenger["delta_cer"], 6) == round(0.07 - 0.09, 6)
    assert round(challenger["delta_utmos"], 6) == round(3.4 - 3.1, 6)
    assert round(challenger["delta_dnsmos_ovr"], 6) == round(3.9 - 3.6, 6)
    assert round(challenger["delta_mel_l1"], 6) == round(1.0 - 1.3, 6)
